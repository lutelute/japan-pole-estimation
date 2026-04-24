#!/usr/bin/env python3
"""
学習済みYOLOv8モデルで全タイルをバッチ推論する。

処理:
  1. YOLOv8で全タイル推論（GPU batch_size=64）
  2. 配電線バッファフィルタ適用（100m以内のみ）
  3. NMS（15m半径で重複除去）
  4. GeoJSON出力

前提: pip install ultralytics
GPU必須（GPUサーバーで実行）。
出力: output/japan_poles_raw.geojson
"""

import json
import math
import sys
from pathlib import Path

from utils import (
    PREFECTURES,
    haversine,
    pixel_to_latlon,
    extract_way_points,
    build_spatial_grid,
    find_near_grid,
    dedup_points,
    save_geojson_atomic,
    Progress,
)

MODEL_PATH   = Path("models/pole_yolov8.pt")
TILE_DIR     = Path("tiles")
OSM_DIR      = Path("data/osm")
OUT_FILE     = Path("output/japan_poles_raw.geojson")
ZOOM         = 18
CONF_THRESH  = 0.35
BATCH_SIZE   = 64
BUFFER_M     = 100   # 配電線からのバッファ
NMS_RADIUS_M = 15    # NMS半径

# 2台GPUで分散処理する場合の担当プレフィックス（run_distributed.shが設定）
# 例: RANK=0 WORLD_SIZE=2 → 偶数インデックスの県を担当
import os as _os
_RANK       = int(_os.environ.get("RANK", 0))
_WORLD_SIZE = int(_os.environ.get("WORLD_SIZE", 1))


def load_all_line_points():
    """全OSMデータから配電線ノード座標リストを返す"""
    points = []
    for name in PREFECTURES:
        osm_path = OSM_DIR / f"{name}.json"
        if not osm_path.exists():
            continue
        with open(osm_path, encoding="utf-8") as f:
            osm = json.load(f)
        points.extend(extract_way_points(osm))
    return points


def tile_in_buffer(tx, ty, zoom, grid, r_deg, buffer_m):
    """タイルの中心が配電線バッファ内かどうか判定"""
    n = 2 ** zoom
    lon = (tx + 0.5) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n))))
    return find_near_grid(lat, lon, grid, r_deg, buffer_m)


def run_inference(tile_paths, model, device):
    """YOLOv8バッチ推論 → list of {lat, lon, score}"""
    detections = []
    batch_size = BATCH_SIZE
    total = len(tile_paths)
    prog = Progress(total, prefix="推論", interval=500)

    for i in range(0, total, batch_size):
        batch = tile_paths[i:i + batch_size]
        batch_strs = [str(tp) for tp in batch]

        results = model.predict(
            source=batch_strs,
            conf=CONF_THRESH,
            device=device,
            imgsz=256,
            batch=len(batch),
            verbose=False,
        )

        for tp, result in zip(batch, results):
            parts = tp.stem.split("_")
            tx, ty = int(parts[1]), int(parts[2])

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                xywhn = box.xywhn[0].tolist()
                cx_px = xywhn[0] * 256
                cy_px = xywhn[1] * 256
                lat, lon = pixel_to_latlon(cx_px, cy_px, tx, ty, ZOOM)
                detections.append({
                    "lat"  : lat,
                    "lon"  : lon,
                    "score": round(float(box.conf), 4),
                })

        prog.update(len(batch))

    return detections


def main():
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("ultralytics がインストールされていません。")
        print("  pip install ultralytics")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"モデルが見つかりません: {MODEL_PATH}")
        print("先に 04_train_yolo.py を実行してください。")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("警告: GPUが検出されません。GPUサーバーで実行してください。")
        resp = input("CPUで続行しますか？ [y/N]: ").strip().lower()
        if resp != "y":
            sys.exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"推論デバイス: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 配電線バッファグリッドの構築
    print("\n配電線ノードを読み込み中...")
    line_points = load_all_line_points()
    print(f"配電線ノード: {len(line_points):,} 点")

    if not line_points:
        print("OSMデータが見つかりません。01_get_osm_japan.py を実行してください。")
        sys.exit(1)

    grid, r_deg = build_spatial_grid(line_points, BUFFER_M)

    # バッファ内タイルに絞る（RANK分割対応）
    all_tiles = sorted(TILE_DIR.glob("*.jpg"))
    if _WORLD_SIZE > 1:
        all_tiles = [t for i, t in enumerate(all_tiles) if i % _WORLD_SIZE == _RANK]
        print(f"全タイル（rank {_RANK}/{_WORLD_SIZE}）: {len(all_tiles):,} 枚")
    else:
        print(f"全タイル: {len(all_tiles):,} 枚")

    filtered_tiles = [
        tp for tp in all_tiles
        if tile_in_buffer(
            int(tp.stem.split("_")[1]),
            int(tp.stem.split("_")[2]),
            ZOOM, grid, r_deg, BUFFER_M
        )
    ]
    print(f"バッファフィルタ後: {len(filtered_tiles):,} 枚")

    if not filtered_tiles:
        print("対象タイルがありません。")
        sys.exit(1)

    # モデルロード
    print(f"\nモデルロード: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 推論
    print(f"バッチサイズ: {BATCH_SIZE}")
    detections = run_inference(filtered_tiles, model, device)
    print(f"\n検出数（NMS前）: {len(detections):,}")

    # NMS（距離ベース）
    raw_points = [(d["lat"], d["lon"]) for d in detections]
    deduped = dedup_points(raw_points, radius_m=NMS_RADIUS_M)
    print(f"NMS後: {len(deduped):,}")

    # 検出結果のスコアを紐付け
    det_grid, det_r_deg = build_spatial_grid(
        [(d["lat"], d["lon"]) for d in detections], NMS_RADIUS_M
    )

    features = []
    for lat, lon in deduped:
        # 最近傍スコアを取得
        score = 0.5
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                gx = int(lat / det_r_deg)
                gy = int(lon / det_r_deg)
                for d in [dd for dd in detections
                           if abs(dd["lat"] / det_r_deg - gx) < 2 and
                              abs(dd["lon"] / det_r_deg - gy) < 2]:
                    if haversine((lat, lon), (d["lat"], d["lon"])) < NMS_RADIUS_M:
                        score = max(score, d["score"])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "source" : "yolov8",
                "score"  : round(score, 4),
                "voltage": "6600",
            },
        })

    # RANK分割時は rank別ファイルに出力
    if _WORLD_SIZE > 1:
        out = OUT_FILE.parent / f"japan_poles_rank{_RANK}.geojson"
    else:
        out = OUT_FILE
    save_geojson_atomic(features, out)
    print(f"\n出力: {out}")
    print(f"電柱数: {len(features):,} 本")


if __name__ == "__main__":
    main()
