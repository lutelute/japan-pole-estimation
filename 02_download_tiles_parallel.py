#!/usr/bin/env python3
"""
全OSMデータから配電線ルートを読み込み、
配電線から100mバッファ内のzoom 18タイルを並列ダウンロードする。
既DLタイルはスキップ（再実行可能）。
出力: tiles/{z}_{x}_{y}.jpg
"""

import json
import sys
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

from utils import PREFECTURES, buffer_tiles, extract_way_points, Progress

GSI_URL    = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
ZOOM       = 18
BUFFER_M   = 100
TILE_DIR   = Path("tiles")
OSM_DIR    = Path("data/osm")
NUM_WORKERS = 32
RATE_SLEEP  = 0.01   # DLスレッドごとの待機


def collect_tiles_from_osm():
    """全OSMファイルから配電線沿いのタイルセットを収集する"""
    tile_set = set()
    pref_names = list(PREFECTURES.keys())
    loaded = 0

    print("OSMデータを読み込んでタイルセットを生成中...")
    for name in pref_names:
        osm_path = OSM_DIR / f"{name}.json"
        if not osm_path.exists():
            continue
        with open(osm_path, encoding="utf-8") as f:
            osm = json.load(f)
        points = extract_way_points(osm)
        for lat, lon in points:
            tile_set.update(buffer_tiles(lat, lon, BUFFER_M, ZOOM))
        loaded += 1
        print(f"  {name:<12} {len(points):>6,} 点  累計タイル {len(tile_set):,}", flush=True)

    print(f"\n{loaded} 県分のOSMを処理  対象タイル合計: {len(tile_set):,} 枚")
    return tile_set


def download_one(args):
    """1タイルをDLして (ok, x, y) を返す"""
    x, y = args
    path = TILE_DIR / f"{ZOOM}_{x}_{y}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return True, x, y

    url = GSI_URL.format(z=ZOOM, x=x, y=y)
    for attempt in range(3):
        try:
            tmp_path = path.with_suffix(".tmp")
            urlretrieve(url, tmp_path)
            tmp_path.rename(path)
            time.sleep(RATE_SLEEP)
            return True, x, y
        except URLError as e:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
            else:
                return False, x, y
        except Exception:
            return False, x, y
    return False, x, y


def estimate_size_gb(n_tiles):
    avg_bytes = 15 * 1024  # 約15KB/タイル（航空写真JPEG）
    return n_tiles * avg_bytes / 1e9


def main():
    TILE_DIR.mkdir(exist_ok=True)

    tile_set = collect_tiles_from_osm()
    if not tile_set:
        print("OSMデータが見つかりません。01_get_osm_japan.py を先に実行してください。")
        sys.exit(1)

    # 既DL済みをスキップ
    pending = [(x, y) for x, y in tile_set
               if not (TILE_DIR / f"{ZOOM}_{x}_{y}.jpg").exists()]
    already = len(tile_set) - len(pending)

    print(f"\n対象タイル : {len(tile_set):,} 枚")
    print(f"既DL済み   : {already:,} 枚")
    print(f"DL予定     : {len(pending):,} 枚  (推定 {estimate_size_gb(len(pending)):.1f} GB)")

    if not pending:
        print("全タイルDL済みです。")
        return

    # 所要時間の目安 (8スレッド × 0.05s/tile)
    eta_min = len(pending) * RATE_SLEEP / NUM_WORKERS / 60
    print(f"スレッド数 : {NUM_WORKERS}  推定完了時間: {eta_min:.0f} 分")
    print()

    ok_count  = 0
    err_count = 0
    prog = Progress(len(pending), prefix="DL", interval=200)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(download_one, args): args for args in pending}
        for fut in as_completed(futs):
            ok, x, y = fut.result()
            if ok:
                ok_count += 1
            else:
                err_count += 1
            prog.update()

    print(f"\n=== DL完了 ===")
    print(f"  成功: {ok_count:,}  失敗: {err_count:,}")
    print(f"  保存先: {TILE_DIR.resolve()}/")

    if err_count > 0:
        print(f"  ※ {err_count} 枚が失敗しました。再実行すると再試行されます。")
        sys.exit(1)


if __name__ == "__main__":
    main()
