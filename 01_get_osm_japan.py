#!/usr/bin/env python3
"""
47都道府県の6.6kV配電線データをOverpass APIで取得する。
既取得分はスキップ（再実行可能）。
出力: data/osm/{pref_name}.json
"""

import json
import os
import sys
import time
import tempfile
from pathlib import Path

import requests

from utils import PREFECTURES

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
OUT_DIR      = Path("data/osm")
TIMEOUT_SEC  = 180
RETRY_MAX    = 3
RETRY_WAIT   = 30    # 秒（サーバーエラー時）
REQUEST_WAIT = 2.0   # リクエスト間の待機（DoS対策）

HEADERS = {
    "User-Agent": "PoleEstimation/1.0 (research; contact: pws-lab)",
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded",
}

QUERY_TEMPLATE = """[out:json][timeout:{timeout}];
(
  way["power"="minor_line"]({s},{w},{n},{e});
  way["power"="line"]["voltage"~"6600|6.6kV"]({s},{w},{n},{e});
  node["power"="pole"]({s},{w},{n},{e});
);
out geom;
"""


def fetch_pref(name, bbox):
    w, s, e, n = bbox
    query = QUERY_TEMPLATE.format(timeout=TIMEOUT_SEC, s=s, w=w, n=n, e=e)

    for attempt in range(1, RETRY_MAX + 1):
        url = OVERPASS_URLS[(attempt - 1) % len(OVERPASS_URLS)]
        try:
            resp = requests.post(
                url,
                data={"data": query},
                headers=HEADERS,
                timeout=TIMEOUT_SEC + 10,
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (429, 406):
                wait = RETRY_WAIT * attempt
                print(f"    {resp.status_code} ({url.split('/')[2]}), {wait}s 待機...", flush=True)
                time.sleep(wait)
            elif resp.status_code == 504:
                wait = RETRY_WAIT * attempt
                print(f"    timeout(504), {wait}s 待機...", flush=True)
                time.sleep(wait)
            else:
                print(f"    HTTP {resp.status_code}, リトライ {attempt}/{RETRY_MAX}", flush=True)
                time.sleep(RETRY_WAIT)
        except requests.exceptions.Timeout:
            print(f"    タイムアウト, リトライ {attempt}/{RETRY_MAX}", flush=True)
            time.sleep(RETRY_WAIT)
        except requests.exceptions.RequestException as e:
            print(f"    ネットワークエラー: {e}, リトライ {attempt}/{RETRY_MAX}", flush=True)
            time.sleep(RETRY_WAIT)

    return None


def save_atomic(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prefs = list(PREFECTURES.items())
    total = len(prefs)
    done  = 0
    skipped = 0
    failed  = 0

    print(f"47都道府県のOSMデータ取得を開始します")
    print(f"出力先: {OUT_DIR.resolve()}")
    print()

    for i, (name, bbox) in enumerate(prefs, 1):
        out_path = OUT_DIR / f"{name}.json"

        if out_path.exists():
            size = out_path.stat().st_size
            if size > 100:
                print(f"[{i:2d}/{total}] {name:<12} スキップ（既取得 {size:,} bytes）")
                skipped += 1
                continue
            else:
                print(f"[{i:2d}/{total}] {name:<12} 不完全なファイルを再取得します")

        print(f"[{i:2d}/{total}] {name:<12} 取得中...", end=" ", flush=True)

        data = fetch_pref(name, bbox)
        if data is None:
            print(f"失敗（スキップ）")
            failed += 1
            time.sleep(REQUEST_WAIT)
            continue

        elements = data.get("elements", [])
        ways  = sum(1 for e in elements if e["type"] == "way")
        nodes = sum(1 for e in elements if e["type"] == "node")
        print(f"way={ways:,}  pole={nodes:,}")

        save_atomic(data, out_path)
        done += 1
        time.sleep(REQUEST_WAIT)

    print()
    print(f"=== 完了 ===")
    print(f"  取得: {done} 県  スキップ: {skipped} 県  失敗: {failed} 県")
    print(f"  出力: {OUT_DIR.resolve()}/")

    if failed > 0:
        print(f"\n  ※ {failed} 県が失敗しました。再実行してください。")
        sys.exit(1)


if __name__ == "__main__":
    main()
