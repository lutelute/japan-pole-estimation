#!/usr/bin/env python3
"""
OSMデータ（高精度）+ CV検出結果を統合してファイナル出力する。

処理:
  1. 全OSMファイルから電柱を抽出・補間
  2. CV検出結果とマージ（重複除去、OSM優先）
  3. 都道府県別GeoJSON出力
  4. 統計サマリ

出力:
  output/japan_poles_FINAL.geojson
  output/by_pref/{pref_name}.geojson
"""

import json
import math
from pathlib import Path

from utils import (
    PREFECTURES,
    haversine,
    interpolate_segment,
    load_geojson,
    build_spatial_grid,
    find_near_grid,
    save_geojson_atomic,
    Progress,
)

OSM_DIR      = Path("data/osm")
CV_FILE      = Path("output/japan_poles_raw.geojson")
OUT_FILE     = Path("output/japan_poles_FINAL.geojson")
BY_PREF_DIR  = Path("output/by_pref")
MERGE_RADIUS = 20    # CV検出とOSMが同一電柱とみなす距離(m)
INTERP_THRESH = 80   # この距離(m)超の区間を補間
SPAN_M        = 40   # 補間間隔(m)
DEDUP_RADIUS  = 10   # OSM内重複除去半径(m)


def extract_osm_poles(osm_data, pref_name):
    """OSM JSON から電柱フィーチャーリストを生成する"""
    elements = osm_data.get("elements", [])
    nodes = {e["id"]: e for e in elements if e["type"] == "node"}

    # power=minor_line (6.6kV配電線) のみ対象。power=line (送電鉄塔線) は除外。
    # minor_line のノード間隔≈34m は電柱間隔に対応するため、ノードを直接電柱位置として採用。
    minor_line_ways = [
        e for e in elements
        if e["type"] == "way" and e.get("tags", {}).get("power") == "minor_line"
    ]

    features = []

    for way in minor_line_ways:
        tags     = way.get("tags", {})
        operator = tags.get("operator", "")
        geom     = way.get("geometry", [])
        if len(geom) < 2:
            continue

        for i, g in enumerate(geom):
            features.append({
                "lat"     : g["lat"],
                "lon"     : g["lon"],
                "source"  : "osm_way_node",
                "operator": operator,
                "pref"    : pref_name,
                "way_id"  : str(way["id"]),
            })

            if i < len(geom) - 1:
                p1 = (g["lat"], g["lon"])
                p2 = (geom[i + 1]["lat"], geom[i + 1]["lon"])
                if haversine(p1, p2) > INTERP_THRESH:
                    for lat, lon in interpolate_segment(p1, p2, SPAN_M):
                        features.append({
                            "lat"     : lat,
                            "lon"     : lon,
                            "source"  : "osm_interpolated",
                            "operator": operator,
                            "pref"    : pref_name,
                            "way_id"  : str(way["id"]),
                        })

    # 明示的 power=pole ノード（最も信頼性が高い）
    for nid, n in nodes.items():
        if n.get("tags", {}).get("power") == "pole":
            features.append({
                "lat"     : n["lat"],
                "lon"     : n["lon"],
                "source"  : "osm_pole_tag",
                "operator": n.get("tags", {}).get("operator", ""),
                "pref"    : pref_name,
                "way_id"  : "",
            })

    return features


def point_in_pref_bbox(lat, lon, bbox):
    """(lat, lon) が都道府県 bbox 内か判定"""
    w, s, e, n = bbox
    return s <= lat <= n and w <= lon <= e


def assign_pref(lat, lon):
    """緯度経度から都道府県を推定（bbox 判定）"""
    for name, bbox in PREFECTURES.items():
        if point_in_pref_bbox(lat, lon, bbox):
            return name
    return "unknown"


def dedup_features(features, radius_m):
    """フィーチャーリストの近接点を統合"""
    r_deg = radius_m / 111000
    seen = {}
    result = []

    for feat in features:
        lat, lon = feat["lat"], feat["lon"]
        gx = int(lat / r_deg)
        gy = int(lon / r_deg)
        duplicate = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for q in seen.get((gx + dx, gy + dy), []):
                    if haversine((lat, lon), (q["lat"], q["lon"])) < radius_m:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            seen.setdefault((gx, gy), []).append(feat)
            result.append(feat)
    return result


def to_geojson_feature(pole, confidence=1.0, note=""):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [pole["lon"], pole["lat"]],
        },
        "properties": {
            "source"    : pole.get("source", "osm"),
            "operator"  : pole.get("operator", ""),
            "voltage_kv": "6.6",
            "pref"      : pole.get("pref", ""),
            "way_id"    : pole.get("way_id", ""),
            "confidence": confidence,
            "note"      : note,
        },
    }


def main():
    BY_PREF_DIR.mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    # ── OSMデータ読み込み ──────────────────────────
    print("OSMデータを読み込み中...")
    all_osm_poles = []
    pref_stats = {}

    prefs = list(PREFECTURES.keys())
    for name in prefs:
        osm_path = OSM_DIR / f"{name}.json"
        if not osm_path.exists():
            print(f"  {name:<12} OSMファイルなし（スキップ）")
            continue
        with open(osm_path, encoding="utf-8") as f:
            osm = json.load(f)
        poles = extract_osm_poles(osm, name)
        poles = dedup_features(poles, DEDUP_RADIUS)
        pref_stats[name] = {"osm": len(poles), "cv": 0, "total": 0}
        all_osm_poles.extend(poles)
        print(f"  {name:<12} OSM: {len(poles):>6,} 本")

    print(f"\nOSM合計: {len(all_osm_poles):,} 本")

    # ── CV検出結果の読み込みとマージ ────────────────
    cv_feats = load_geojson(CV_FILE) if CV_FILE.exists() else []
    print(f"CV検出: {len(cv_feats):,} 本")

    cv_only_poles = []
    if cv_feats:
        osm_grid, osm_r_deg = build_spatial_grid(
            [(p["lat"], p["lon"]) for p in all_osm_poles],
            MERGE_RADIUS,
        )
        for f in cv_feats:
            lon, lat = f["geometry"]["coordinates"]
            if not find_near_grid(lat, lon, osm_grid, osm_r_deg, MERGE_RADIUS):
                pref = assign_pref(lat, lon)
                score = f["properties"].get("score", 0.5)
                cv_only_poles.append({
                    "lat"     : lat,
                    "lon"     : lon,
                    "source"  : "cv_only",
                    "operator": "",
                    "pref"    : pref,
                    "way_id"  : "",
                    "score"   : score,
                })
                if pref in pref_stats:
                    pref_stats[pref]["cv"] += 1

        print(f"CV追加分（OSM重複除外）: {len(cv_only_poles):,} 本")

    # ── 全体GeoJSON出力 ────────────────────────────
    final_features = []

    for p in all_osm_poles:
        final_features.append(to_geojson_feature(p, confidence=1.0))

    for p in cv_only_poles:
        feat = to_geojson_feature(p, confidence=p.get("score", 0.5), note="CV検出・要確認")
        final_features.append(feat)

    save_geojson_atomic(final_features, OUT_FILE)
    print(f"\n全国GeoJSON: {OUT_FILE}  ({len(final_features):,} 本)")

    # ── 都道府県別 GeoJSON ────────────────────────
    print("\n都道府県別ファイルを出力中...")
    pref_buckets = {}
    for feat in final_features:
        pref = feat["properties"].get("pref", "unknown")
        pref_buckets.setdefault(pref, []).append(feat)

    for pref, feats in pref_buckets.items():
        out_path = BY_PREF_DIR / f"{pref}.geojson"
        save_geojson_atomic(feats, out_path)
        if pref in pref_stats:
            pref_stats[pref]["total"] = len(feats)

    # ── 統計サマリ ────────────────────────────────
    print("\n=== 都道府県別電柱数 ===")
    print(f"{'都道府県':<12} {'OSM':>8} {'CV追加':>8} {'合計':>8}")
    print("-" * 42)
    grand_total = 0
    for name in prefs:
        stat = pref_stats.get(name, {"osm": 0, "cv": 0, "total": 0})
        total = stat["total"] or stat["osm"] + stat["cv"]
        print(f"{name:<12} {stat['osm']:>8,} {stat['cv']:>8,} {total:>8,}")
        grand_total += total

    unknown = len(pref_buckets.get("unknown", []))
    if unknown > 0:
        print(f"{'unknown':<12} {'':>8} {'':>8} {unknown:>8,}")
        grand_total += unknown

    print("-" * 42)
    print(f"{'合計':<12} {'':>8} {'':>8} {grand_total:>8,}")
    print(f"\n出力:")
    print(f"  全国: {OUT_FILE}")
    print(f"  都道府県別: {BY_PREF_DIR}/")


if __name__ == "__main__":
    main()
