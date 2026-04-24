#!/usr/bin/env python3
"""共通ユーティリティ関数"""

import json
import math
import os
import tempfile
import time
from pathlib import Path


# ──────────────────────────────────────────────
# 地理計算
# ──────────────────────────────────────────────

def haversine(p1, p2):
    """(lat1, lon1), (lat2, lon2) 間の距離(m)"""
    R = 6371000
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def latlon_to_tile(lat, lon, zoom):
    """WGS84 → タイル(x, y)"""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_to_latlon(tx, ty, px, py, zoom, size=256):
    """タイル内ピクセル座標 → WGS84"""
    n = 2 ** zoom
    lon = (tx + px / size) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + py / size) / n))))
    return lat, lon


def pixel_to_latlon(px, py, tile_x, tile_y, zoom, img_size=256):
    """タイル内ピクセル座標 → WGS84"""
    return tile_to_latlon(tile_x, tile_y, px, py, zoom, img_size)


def buffer_tiles(lat, lon, buffer_m, zoom):
    """中心点の buffer_m 半径内にあるタイルの set を返す"""
    dlat = buffer_m / 111000
    dlon = buffer_m / (111000 * math.cos(math.radians(lat)))
    tiles = set()
    for la in [lat - dlat, lat, lat + dlat]:
        for lo in [lon - dlon, lon, lon + dlon]:
            tiles.add(latlon_to_tile(la, lo, zoom))
    return tiles


# ──────────────────────────────────────────────
# GeoJSON
# ──────────────────────────────────────────────

def load_geojson(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    return gj.get("features", [])


def save_geojson_atomic(features, path):
    """アトミック書き込みで GeoJSON を保存する"""
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ──────────────────────────────────────────────
# グリッドベース近傍探索
# ──────────────────────────────────────────────

def build_spatial_grid(points, radius_m):
    """
    points: list of (lat, lon)
    return: (grid_dict, r_deg)
    """
    r_deg = radius_m / 111000
    grid = {}
    for lat, lon in points:
        gx = int(lat / r_deg)
        gy = int(lon / r_deg)
        grid.setdefault((gx, gy), []).append((lat, lon))
    return grid, r_deg


def find_near_grid(lat, lon, grid, r_deg, radius_m):
    """グリッドを使って radius_m 以内の点があるか判定"""
    gx = int(lat / r_deg)
    gy = int(lon / r_deg)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for q_lat, q_lon in grid.get((gx + dx, gy + dy), []):
                if haversine((lat, lon), (q_lat, q_lon)) < radius_m:
                    return True
    return False


def dedup_points(points, radius_m=10):
    """近接する点を統合する（list of (lat, lon)）"""
    result = []
    r_deg = radius_m / 111000
    seen = {}
    for lat, lon in points:
        gx = int(lat / r_deg)
        gy = int(lon / r_deg)
        duplicate = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for q_lat, q_lon in seen.get((gx + dx, gy + dy), []):
                    if haversine((lat, lon), (q_lat, q_lon)) < radius_m:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            seen.setdefault((gx, gy), []).append((lat, lon))
            result.append((lat, lon))
    return result


# ──────────────────────────────────────────────
# 進捗表示
# ──────────────────────────────────────────────

class Progress:
    def __init__(self, total, prefix="", interval=50):
        self.total = total
        self.prefix = prefix
        self.interval = interval
        self.count = 0
        self.start = time.time()

    def update(self, n=1):
        self.count += n
        if self.count % self.interval == 0 or self.count == self.total:
            elapsed = time.time() - self.start
            rate = self.count / elapsed if elapsed > 0 else 0
            eta = (self.total - self.count) / rate if rate > 0 else 0
            pct = self.count / self.total * 100
            print(
                f"\r{self.prefix} [{self.count}/{self.total}] {pct:.1f}%  "
                f"{rate:.1f}/s  ETA {eta:.0f}s  ",
                end="",
                flush=True,
            )
            if self.count == self.total:
                print()


# ──────────────────────────────────────────────
# OSMデータ操作
# ──────────────────────────────────────────────

def extract_way_points(osm_data):
    """OSM JSON から配電線 way の全ノード座標を返す"""
    elements = osm_data.get("elements", [])
    ways = [e for e in elements if e["type"] == "way"]
    points = []
    for way in ways:
        for g in way.get("geometry", []):
            points.append((g["lat"], g["lon"]))
    return points


def interpolate_segment(p1, p2, span_m=40):
    """p1→p2 間を span_m 間隔で補間（端点は含まない）"""
    dist = haversine(p1, p2)
    pts = []
    n = max(1, int(dist // span_m))
    for i in range(1, n):
        t = i / n
        lat = p1[0] + t * (p2[0] - p1[0])
        lon = p1[1] + t * (p2[1] - p1[1])
        pts.append((lat, lon))
    return pts


# ──────────────────────────────────────────────
# 47都道府県 bbox
# ──────────────────────────────────────────────

PREFECTURES = {
    "hokkaido":   (139.35, 41.35, 148.90, 45.55),
    "aomori":     (139.87, 40.19, 141.68, 41.56),
    "iwate":      (140.65, 38.73, 141.77, 40.45),
    "miyagi":     (140.27, 37.77, 141.68, 38.98),
    "akita":      (139.70, 39.00, 141.47, 40.52),
    "yamagata":   (139.53, 37.73, 140.87, 38.98),
    "fukushima":  (139.20, 36.77, 141.05, 37.97),
    "ibaraki":    (139.69, 35.73, 140.85, 36.80),
    "tochigi":    (139.32, 36.20, 140.28, 37.15),
    "gunma":      (138.40, 36.17, 139.68, 37.00),
    "saitama":    (138.72, 35.74, 139.90, 36.27),
    "chiba":      (139.73, 35.03, 140.90, 35.91),
    "tokyo":      (138.94, 35.51, 139.92, 35.90),
    "kanagawa":   (139.02, 35.13, 139.78, 35.67),
    "niigata":    (137.62, 36.77, 139.60, 38.59),
    "toyama":     (136.77, 36.24, 137.68, 36.98),
    "ishikawa":   (136.20, 36.10, 137.35, 37.55),
    "fukui":      (135.42, 35.42, 136.90, 36.33),
    "yamanashi":  (138.33, 35.24, 138.98, 35.90),
    "nagano":     (137.33, 35.19, 138.58, 37.00),
    "shizuoka":   (137.47, 34.60, 139.14, 35.67),
    "aichi":      (136.67, 34.57, 137.83, 35.43),
    "gifu":       (136.28, 35.15, 137.73, 36.45),
    "mie":        (135.85, 33.73, 136.88, 35.02),
    "shiga":      (135.77, 34.80, 136.53, 35.61),
    "kyoto":      (135.06, 34.76, 136.00, 35.77),
    "osaka":      (135.31, 34.30, 135.72, 34.90),
    "hyogo":      (134.27, 34.15, 135.48, 35.65),
    "nara":       (135.57, 33.85, 136.25, 34.69),
    "wakayama":   (135.07, 33.43, 136.15, 34.35),
    "tottori":    (133.28, 35.07, 134.41, 35.57),
    "shimane":    (131.67, 34.30, 133.40, 35.42),
    "okayama":    (133.22, 34.48, 134.47, 35.22),
    "hiroshima":  (132.00, 34.05, 133.40, 35.17),
    "yamaguchi":  (130.67, 33.73, 132.37, 34.73),
    "tokushima":  (133.68, 33.57, 134.83, 34.35),
    "kagawa":     (133.48, 33.97, 134.50, 34.42),
    "ehime":      (132.00, 33.10, 133.68, 34.07),
    "kochi":      (132.57, 32.70, 134.33, 33.88),
    "fukuoka":    (130.03, 33.10, 131.12, 34.23),
    "saga":       (129.72, 32.90, 130.45, 33.62),
    "nagasaki":   (128.55, 32.57, 130.20, 34.72),
    "kumamoto":   (130.05, 32.07, 131.35, 33.20),
    "oita":       (130.77, 32.72, 131.90, 33.75),
    "miyazaki":   (130.65, 31.35, 131.88, 32.72),
    "kagoshima":  (129.30, 30.97, 131.27, 32.22),
    "okinawa":    (122.93, 24.05, 131.33, 27.08),
}
