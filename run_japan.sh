#!/usr/bin/env bash
# 日本全国6.6kV電柱位置推定パイプライン
# ワンショット実行スクリプト

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

step_done() {
    local marker="$1"
    local file="$2"
    if [ -f "$file" ]; then
        return 0
    fi
    return 1
}

echo "========================================"
echo " 日本全国6.6kV電柱位置推定パイプライン"
echo "========================================"
echo ""

# ── ステップ 1: OSMデータ取得 ────────────────────
info "ステップ 1/6: 47都道府県のOSMデータ取得"

DONE_COUNT=$(ls data/osm/*.json 2>/dev/null | wc -l || echo 0)
if [ "$DONE_COUNT" -ge 47 ]; then
    ok "OSMデータ取得済み (${DONE_COUNT} 県)"
else
    info "未取得: $((47 - DONE_COUNT)) 県分を取得します"
    python3 01_get_osm_japan.py
    ok "ステップ 1 完了"
fi
echo ""

# ── ステップ 2: タイルDL ─────────────────────────
info "ステップ 2/6: 国土地理院タイルDL（配電線バッファ100m）"
warn "推定タイル数: 数百万枚 / 推定容量: 数十GB"
warn "ネットワーク速度により数時間〜数日かかります"
echo ""

TILE_COUNT=$(ls tiles/*.jpg 2>/dev/null | wc -l || echo 0)
if [ "$TILE_COUNT" -ge 1000 ]; then
    ok "タイル取得中 or 済み (${TILE_COUNT} 枚)"
    read -r -p "タイルDLを実行しますか？ [y/N]: " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        python3 02_download_tiles_parallel.py
    fi
else
    python3 02_download_tiles_parallel.py
fi
ok "ステップ 2 完了"
echo ""

# ── ステップ 3: YOLOデータセット作成 ─────────────
info "ステップ 3/6: YOLOv8学習データセット作成（福井県データ使用）"

if [ -f "yolo_dataset/dataset.yaml" ]; then
    ok "dataset.yaml 既存 → スキップ"
else
    if [ ! -d "../pole_estimation/tiles" ]; then
        error "福井県タイルが見つかりません: ../pole_estimation/tiles/"
        error "先に福井県実装を実行してください。"
        exit 1
    fi
    python3 03_prepare_yolo_dataset.py
    ok "ステップ 3 完了"
fi
echo ""

# ── ステップ 4: YOLO学習（GPU必須）────────────────
info "ステップ 4/6: YOLOv8 ファインチューニング"
echo ""
warn "★★ GPU必須ステップ ★★"
warn "このステップはGPUサーバーで実行してください："
warn "  - pws-gpu   : RTX 4090 (24GB VRAM)  ssh root@100.126.144.110"
warn "  - pws-gpu3060: RTX 3060 (12GB VRAM)  ssh ubuntu@10.0.70.81"
echo ""
warn "GPUサーバーでの実行手順:"
warn "  1. scp -r $(pwd) <server>:~/japan_poles/"
warn "  2. ssh <server>"
warn "  3. cd ~/japan_poles && pip install ultralytics"
warn "  4. python3 04_train_yolo.py"
warn "  5. scp <server>:~/japan_poles/models/pole_yolov8.pt ./models/"
echo ""

if [ -f "models/pole_yolov8.pt" ]; then
    ok "モデル既存 (models/pole_yolov8.pt) → スキップ"
else
    read -r -p "ローカルで強制実行しますか？（GPUなしは遅い）[y/N]: " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        python3 04_train_yolo.py
    else
        warn "ステップ 4 をスキップ。GPUサーバーで学習後にモデルを配置してください。"
        warn "  models/pole_yolov8.pt"
        exit 0
    fi
fi
ok "ステップ 4 完了"
echo ""

# ── ステップ 5: GPU推論（GPU必須）────────────────
info "ステップ 5/6: 全タイルYOLO推論"
echo ""
warn "★★ GPU必須ステップ ★★"
warn "GPUサーバー (pws-gpu / pws-gpu3060) で実行してください。"
echo ""

if [ -f "output/japan_poles_raw.geojson" ]; then
    ok "推論結果既存 → スキップ"
else
    if [ ! -f "models/pole_yolov8.pt" ]; then
        error "モデルが見つかりません: models/pole_yolov8.pt"
        error "ステップ 4 を完了してください。"
        exit 1
    fi
    read -r -p "推論を実行しますか？ [y/N]: " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        python3 05_gpu_inference.py
    else
        warn "ステップ 5 をスキップ。GPUサーバーで実行後に output/japan_poles_raw.geojson を配置してください。"
        exit 0
    fi
fi
ok "ステップ 5 完了"
echo ""

# ── ステップ 6: マージ・エクスポート ──────────────
info "ステップ 6/6: OSM + CV統合・都道府県別出力"
python3 06_merge_export.py
ok "ステップ 6 完了"
echo ""

echo "========================================"
ok "パイプライン完了！"
echo "========================================"
echo ""
echo "出力ファイル:"
echo "  全国: output/japan_poles_FINAL.geojson"
echo "  都道府県別: output/by_pref/*.geojson"
echo ""
echo "QGISでの利用:"
echo "  レイヤー → レイヤーを追加 → ベクタレイヤー → GeoJSONを選択"
