#!/usr/bin/env python3
"""OSMデータ取得の進捗をリアルタイム表示する。"""

import json, time, os, sys
from pathlib import Path
from datetime import datetime, timedelta

OSM_DIR   = Path("data/osm")
TOTAL     = 47
INTERVAL  = 5   # 秒

# ANSI
CLR  = "\033[2J\033[H"
BOLD = "\033[1m"
GRN  = "\033[32m"
YEL  = "\033[33m"
CYN  = "\033[36m"
RST  = "\033[0m"
DIM  = "\033[2m"

PREF_JA = {
    "hokkaido":"北海道","aomori":"青森","iwate":"岩手","miyagi":"宮城",
    "akita":"秋田","yamagata":"山形","fukushima":"福島","ibaraki":"茨城",
    "tochigi":"栃木","gunma":"群馬","saitama":"埼玉","chiba":"千葉",
    "tokyo":"東京","kanagawa":"神奈川","niigata":"新潟","toyama":"富山",
    "ishikawa":"石川","fukui":"福井","yamanashi":"山梨","nagano":"長野",
    "shizuoka":"静岡","aichi":"愛知","mie":"三重","shiga":"滋賀",
    "kyoto":"京都","osaka":"大阪","hyogo":"兵庫","nara":"奈良",
    "wakayama":"和歌山","tottori":"鳥取","shimane":"島根","okayama":"岡山",
    "hiroshima":"広島","yamaguchi":"山口","tokushima":"徳島","kagawa":"香川",
    "ehime":"愛媛","kochi":"高知","fukuoka":"福岡","saga":"佐賀",
    "nagasaki":"長崎","kumamoto":"熊本","oita":"大分","miyazaki":"宮崎",
    "kagoshima":"鹿児島","okinawa":"沖縄","gifu":"岐阜",
}

def bar(done, total, width=30):
    filled = int(width * done / total) if total else 0
    b = "█" * filled + "░" * (width - filled)
    pct = done / total * 100 if total else 0
    return f"[{b}] {done}/{total} ({pct:.0f}%)"

def fmt_size(b):
    return f"{b/1024:.0f}KB" if b < 1024*1024 else f"{b/1024/1024:.1f}MB"

def read_stats(p):
    try:
        with open(p) as f:
            d = json.load(f)
        els   = d.get("elements", [])
        ways  = sum(1 for e in els if e["type"] == "way")
        nodes = sum(1 for e in els if e["type"] == "node")
        return ways, nodes
    except Exception:
        return 0, 0

start_time = None
prev_done  = 0

while True:
    files     = sorted(OSM_DIR.glob("*.json")) if OSM_DIR.exists() else []
    done      = len(files)
    remaining = TOTAL - done
    now       = datetime.now()

    if done > 0 and start_time is None:
        start_time = now
    if done > prev_done and start_time:
        elapsed = (now - start_time).total_seconds()
        rate    = done / elapsed if elapsed > 0 else 0
        eta_sec = remaining / rate if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_sec)))
        prev_done = done
    else:
        elapsed = (now - start_time).total_seconds() if start_time else 0
        rate    = done / elapsed if elapsed > 0 else 0
        eta_sec = remaining / rate if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_sec))) if rate > 0 else "--:--:--"

    # 画面クリア & 描画
    sys.stdout.write(CLR)
    sys.stdout.write(f"{BOLD}{CYN}■ 日本全国 OSMデータ取得進捗{RST}  "
                     f"{DIM}{now.strftime('%H:%M:%S')}{RST}\n")
    sys.stdout.write("─" * 50 + "\n")
    sys.stdout.write(f" {GRN}{bar(done, TOTAL)}{RST}\n")
    sys.stdout.write(f" 取得済み: {BOLD}{done}{RST}/{TOTAL} 県  "
                     f"残り: {remaining} 県  "
                     f"ETA: {YEL}{eta_str}{RST}\n")

    # 総電柱・配電線数
    total_ways = total_nodes = 0
    recent = []
    for fp in files:
        w, n = read_stats(fp)
        total_ways  += w
        total_nodes += n
        mtime = fp.stat().st_mtime
        recent.append((mtime, fp.stem, w, n, fp.stat().st_size))

    sys.stdout.write(f" 配電線 way: {BOLD}{total_ways:,}{RST}  "
                     f"電柱 node: {BOLD}{total_nodes:,}{RST}\n")
    sys.stdout.write("─" * 50 + "\n")

    # 直近5件
    recent.sort(reverse=True)
    sys.stdout.write(f"{BOLD} 直近の取得{RST}\n")
    for mtime, name, w, n, sz in recent[:8]:
        ja   = PREF_JA.get(name, name)
        t    = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        sys.stdout.write(f"  {GRN}✓{RST} {ja:<5} {DIM}way={w:>4,} pole={n:>5,} {fmt_size(sz):>7}{RST}  {DIM}{t}{RST}\n")

    if done >= TOTAL:
        sys.stdout.write(f"\n{GRN}{BOLD}✓ 全47県 取得完了！{RST}\n")
        sys.stdout.write("  次のステップ: python3 02_download_tiles_parallel.py\n")
        break

    sys.stdout.write(f"\n{DIM}  {INTERVAL}秒ごとに更新 (Ctrl+C で終了){RST}\n")
    sys.stdout.flush()
    time.sleep(INTERVAL)
