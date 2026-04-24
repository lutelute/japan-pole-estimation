#!/usr/bin/env python3
"""
福井県データを使ってYOLOv8学習データを作成する。

処理:
  1. 福井県タイルに対して影検出（blob抽出）
  2. OSM電柱と照合 → 正例ラベル生成
  3. YOLO format (.txt) で保存
  4. dataset.yaml 作成

出力: yolo_dataset/images/{train,val}/  yolo_dataset/labels/{train,val}/
"""

import cv2
import json
import math
import random
import shutil
import sys
from pathlib import Path

from utils import (
    haversine, tile_to_latlon, latlon_to_tile,
    load_geojson, Progress,
)

import argparse as _ap
_args, _ = _ap.ArgumentParser(add_help=False).parse_known_args()

def _resolve(env_key, default):
    import os
    return Path(os.environ.get(env_key, default))

FUKUI_OSM_FILE  = _resolve("FUKUI_OSM",   "../pole_estimation/output/fukui_6600v.json")
FUKUI_POLE_FILE = _resolve("FUKUI_POLES", "../pole_estimation/output/fukui_poles_6600v.geojson")
TILE_DIR        = _resolve("FUKUI_TILES", "../pole_estimation/tiles")
OUT_DIR         = Path("yolo_dataset")
ZOOM            = 18
TILE_SIZE       = 256
MATCH_RADIUS_M  = 15   # OSM電柱と照合する半径(m)
VAL_RATIO       = 0.15  # 検証用データの割合

# 影検出パラメータ（04b_shadow_detect_cpu.py と同じ設定）
DARK_THRESH = 60
MIN_AREA    = 3
MAX_AREA    = 80


def detect_blobs(img_path):
    """1タイルから暗い blob 候補を返す (pixel cx, cy, w, h)"""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, dark = cv2.threshold(gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    blobs = []
    for i in range(1, num_labels):
        area   = stats[i, cv2.CC_STAT_AREA]
        width  = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        aspect = max(width, height) / (min(width, height) + 1e-6)
        if aspect < 1.5:
            continue
        blobs.append((cx, cy, width, height))
    return blobs


def load_osm_poles(osm_json_path, geojson_path):
    """OSM電柱の (lat, lon) リストを返す"""
    poles = []
    # GeoJSON 優先
    if geojson_path and Path(geojson_path).exists():
        feats = load_geojson(geojson_path)
        for f in feats:
            lon, lat = f["geometry"]["coordinates"]
            poles.append((lat, lon))
    elif osm_json_path and Path(osm_json_path).exists():
        with open(osm_json_path, encoding="utf-8") as f:
            osm = json.load(f)
        for e in osm.get("elements", []):
            if e["type"] == "node" and e.get("tags", {}).get("power") == "pole":
                poles.append((e["lat"], e["lon"]))
    return poles


def poles_near_tile(osm_poles, tx, ty, zoom, size, margin_m=20):
    """タイル内またはマージン以内の OSM 電柱を返す"""
    # タイルの bbox を緯度経度で計算
    n = 2 ** zoom
    lon_w = tx / n * 360 - 180
    lon_e = (tx + 1) / n * 360 - 180
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))

    dlat = margin_m / 111000
    dlon = margin_m / (111000 * math.cos(math.radians((lat_n + lat_s) / 2)))

    result = []
    for lat, lon in osm_poles:
        if (lat_s - dlat <= lat <= lat_n + dlat and
                lon_w - dlon <= lon <= lon_e + dlon):
            result.append((lat, lon))
    return result


def latlon_to_pixel(lat, lon, tx, ty, zoom, size=256):
    """緯度経度 → タイル内ピクセル座標"""
    n = 2 ** zoom
    px = ((lon + 180) / 360 * n - tx) * size
    merc = math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
    py = (1 - (merc / math.pi + 1) / 2) * n
    py = (py - ty) * size
    return px, py


def make_yolo_label(cx, cy, bw, bh, size=256):
    """YOLO format: class cx_norm cy_norm w_norm h_norm"""
    cx_n = cx / size
    cy_n = cy / size
    w_n  = max(bw, 8) / size    # 最小8px幅を保証
    h_n  = max(bh, 8) / size
    # クリップ
    cx_n = max(0.0, min(1.0, cx_n))
    cy_n = max(0.0, min(1.0, cy_n))
    w_n  = min(w_n, 1.0)
    h_n  = min(h_n, 1.0)
    return f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"


def main():
    if not TILE_DIR.exists():
        print(f"タイルディレクトリが見つかりません: {TILE_DIR}")
        print("福井県実装の tiles/ が必要です。")
        sys.exit(1)

    osm_poles = load_osm_poles(FUKUI_OSM_FILE, FUKUI_POLE_FILE)
    print(f"OSM電柱: {len(osm_poles):,} 本")

    if not osm_poles:
        print("OSM電柱データが見つかりません。")
        sys.exit(1)

    tile_paths = sorted(TILE_DIR.glob("*.jpg"))
    print(f"タイル: {len(tile_paths):,} 枚")

    # 出力ディレクトリを初期化
    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    labeled_tiles = []
    total_labels  = 0
    BOX_PX = 12  # 電柱の bounding box サイズ（pixel）
    prog = Progress(len(tile_paths), prefix="ラベル生成", interval=100)

    for tp in tile_paths:
        parts = tp.stem.split("_")
        tx, ty = int(parts[1]), int(parts[2])

        nearby_poles = poles_near_tile(osm_poles, tx, ty, ZOOM, TILE_SIZE)
        if not nearby_poles:
            prog.update()
            continue

        # OSM電柱座標をそのままYOLOラベルに変換（blob照合不要）
        labels = []
        for lat, lon in nearby_poles:
            px, py = latlon_to_pixel(lat, lon, tx, ty, ZOOM, TILE_SIZE)
            # タイル範囲内のみ採用
            if 0 <= px <= TILE_SIZE and 0 <= py <= TILE_SIZE:
                labels.append(make_yolo_label(px, py, BOX_PX, BOX_PX))

        if not labels:
            prog.update()
            continue

        labeled_tiles.append((tp, labels))
        total_labels += len(labels)
        prog.update()

    print(f"\nラベル付きタイル: {len(labeled_tiles):,}  総ラベル: {total_labels:,}")

    if not labeled_tiles:
        print("ラベルが生成されませんでした。タイルや OSM データを確認してください。")
        sys.exit(1)

    # train/val 分割
    random.shuffle(labeled_tiles)
    n_val   = max(1, int(len(labeled_tiles) * VAL_RATIO))
    val_set = labeled_tiles[:n_val]
    trn_set = labeled_tiles[n_val:]

    def write_split(items, split):
        for tp, labels in items:
            dst_img = OUT_DIR / "images" / split / tp.name
            dst_lbl = OUT_DIR / "labels" / split / (tp.stem + ".txt")
            shutil.copy2(tp, dst_img)
            dst_lbl.write_text("\n".join(labels) + "\n", encoding="utf-8")

    write_split(trn_set, "train")
    write_split(val_set, "val")

    print(f"  train: {len(trn_set):,}  val: {len(val_set):,}")

    # dataset.yaml
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_content = f"""path: {OUT_DIR.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: pole
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n出力: {OUT_DIR.resolve()}/")
    print(f"dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
