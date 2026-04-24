#!/usr/bin/env bash
# 2台のGPUサーバーで並列推論を実行するスクリプト
#
# GPU0: pws-gpu    (RTX 4090)  root@100.126.144.110  ← 速いので多めに担当
# GPU1: pws-gpu3060(RTX 3060)  ubuntu@10.0.70.81
#
# 使い方:
#   bash run_distributed.sh             # 全ステップ実行
#   bash run_distributed.sh --infer-only # 推論のみ（タイルDL済みの場合）

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU0_HOST="root@100.126.144.110"
GPU1_HOST="ubuntu@10.0.70.81"
REMOTE_DIR="/home/pole_estimation"   # GPU0 は root なので /root 配下
GPU0_DIR="/root/pole_estimation"
GPU1_DIR="/home/ubuntu/pole_estimation"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
step()  { echo -e "\n${YELLOW}━━━ $* ━━━${NC}"; }

# ─────────────────────────────────────────────────────────────
step "1. コード + OSMデータを GPU0 (RTX 4090) へ転送"
# ─────────────────────────────────────────────────────────────
info "GPU0 (RTX 4090) へ転送中..."
ssh $GPU0_HOST "mkdir -p $GPU0_DIR/data/osm $GPU0_DIR/output $GPU0_DIR/models $GPU0_DIR/yolo_dataset"
rsync -az --progress \
    utils.py \
    03_prepare_yolo_dataset.py \
    04_train_yolo.py \
    05_gpu_inference.py \
    06_merge_export.py \
    data/osm/ \
    $GPU0_HOST:$GPU0_DIR/ || { warn "GPU0 転送失敗"; exit 1; }

# 学習用に福井データも転送
FUKUI_DIR="../pole_estimation"
if [ -d "$FUKUI_DIR" ]; then
    rsync -az $FUKUI_DIR/output/ $GPU0_HOST:$GPU0_DIR/fukui_output/ 2>/dev/null || true
    rsync -az --progress $FUKUI_DIR/tiles/ $GPU0_HOST:$GPU0_DIR/fukui_tiles/ &
    FUKUI_PID=$!
fi
ok "GPU0 コード転送完了"

# ─────────────────────────────────────────────────────────────
step "1b. YOLOv8 学習データ作成 + モデル訓練 (GPU0 RTX 4090)"
# ─────────────────────────────────────────────────────────────
info "GPU0 で学習データ作成中..."
wait $FUKUI_PID 2>/dev/null || true
ssh $GPU0_HOST "cd $GPU0_DIR && \
    pip install ultralytics --quiet 2>/dev/null && \
    FUKUI_TILES=fukui_tiles \
    FUKUI_OSM=fukui_output/fukui_6600v.json \
    FUKUI_POLES=fukui_output/fukui_poles_6600v.geojson \
    python3 03_prepare_yolo_dataset.py && \
    python3 04_train_yolo.py" || { warn "GPU0 学習失敗"; exit 1; }
ok "モデル訓練完了"

info "学習済みモデルを取得..."
mkdir -p models
rsync -az $GPU0_HOST:$GPU0_DIR/models/pole_yolov8.pt models/ || { warn "モデル取得失敗"; exit 1; }
ok "models/pole_yolov8.pt 取得完了"

info "GPU1 (RTX 3060) へ転送中..."
ssh $GPU1_HOST "mkdir -p $GPU1_DIR/data/osm $GPU1_DIR/output $GPU1_DIR/models"
rsync -az --progress \
    utils.py 05_gpu_inference.py 06_merge_export.py \
    data/osm/ \
    models/ \
    $GPU1_HOST:$GPU1_DIR/ || { warn "GPU1 転送失敗"; exit 1; }
ok "GPU1 転送完了"

# ─────────────────────────────────────────────────────────────
step "2. タイルを分散転送（GPU0: 偶数インデックス / GPU1: 奇数インデックス）"
# ─────────────────────────────────────────────────────────────
TILE_COUNT=$(ls tiles/*.jpg 2>/dev/null | wc -l || echo 0)
info "総タイル数: $TILE_COUNT 枚 → 各サーバーに約 $((TILE_COUNT / 2)) 枚"

# GPU0 へ偶数タイルを転送
info "GPU0 へタイル転送中（並列rsync）..."
ssh $GPU0_HOST "mkdir -p $GPU0_DIR/tiles"
rsync -az --progress tiles/ $GPU0_HOST:$GPU0_DIR/tiles/ &
GPU0_RSYNC_PID=$!

# GPU1 へ奇数タイルを転送（全タイルを転送してRANKで分割）
info "GPU1 へタイル転送中（並列rsync）..."
ssh $GPU1_HOST "mkdir -p $GPU1_DIR/tiles"
rsync -az --progress tiles/ $GPU1_HOST:$GPU1_DIR/tiles/ &
GPU1_RSYNC_PID=$!

wait $GPU0_RSYNC_PID && ok "GPU0 タイル転送完了"
wait $GPU1_RSYNC_PID && ok "GPU1 タイル転送完了"

# ─────────────────────────────────────────────────────────────
step "3. 両サーバーで並列推論開始"
# ─────────────────────────────────────────────────────────────
SETUP_CMD="pip install ultralytics --quiet 2>/dev/null || true"
INFER_CMD_GPU0="cd $GPU0_DIR && RANK=0 WORLD_SIZE=2 python3 05_gpu_inference.py"
INFER_CMD_GPU1="cd $GPU1_DIR && RANK=1 WORLD_SIZE=2 python3 05_gpu_inference.py"

info "GPU0 (RTX 4090) で推論開始..."
ssh $GPU0_HOST "$SETUP_CMD && $INFER_CMD_GPU0" 2>&1 | sed 's/^/[GPU0] /' &
GPU0_PID=$!

info "GPU1 (RTX 3060) で推論開始..."
ssh $GPU1_HOST "$SETUP_CMD && $INFER_CMD_GPU1" 2>&1 | sed 's/^/[GPU1] /' &
GPU1_PID=$!

# 両方完了を待つ
info "両GPUの推論完了を待機中..."
GPU0_OK=0; GPU1_OK=0
wait $GPU0_PID && GPU0_OK=1 || warn "GPU0 推論失敗"
wait $GPU1_PID && GPU1_OK=1 || warn "GPU1 推論失敗"

[ $GPU0_OK -eq 1 ] && ok "GPU0 推論完了"
[ $GPU1_OK -eq 1 ] && ok "GPU1 推論完了"
[ $GPU0_OK -eq 0 ] || [ $GPU1_OK -eq 0 ] && { warn "一部失敗。出力を確認してください"; }

# ─────────────────────────────────────────────────────────────
step "4. 推論結果を回収"
# ─────────────────────────────────────────────────────────────
mkdir -p output
[ $GPU0_OK -eq 1 ] && rsync -az $GPU0_HOST:$GPU0_DIR/output/japan_poles_rank0.geojson output/ && ok "rank0 回収完了"
[ $GPU1_OK -eq 1 ] && rsync -az $GPU1_HOST:$GPU1_DIR/output/japan_poles_rank1.geojson output/ && ok "rank1 回収完了"

# ─────────────────────────────────────────────────────────────
step "5. 結果をマージして最終出力"
# ─────────────────────────────────────────────────────────────
python3 - << 'PYEOF'
import json, math
from pathlib import Path
from utils import haversine, dedup_points, save_geojson_atomic

parts = ["output/japan_poles_rank0.geojson", "output/japan_poles_rank1.geojson"]
all_feats = []
for p in parts:
    if Path(p).exists():
        with open(p) as f:
            all_feats.extend(json.load(f)["features"])
        print(f"  {p}: {len(all_feats):,} 件")

# 全体NMS（2台の検出結果の重複除去）
pts = [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in all_feats]
scores = {(round(f["geometry"]["coordinates"][1],6), round(f["geometry"]["coordinates"][0],6)):
          f["properties"].get("score", 0.5) for f in all_feats}
deduped = dedup_points(pts, radius_m=15)

features = [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [lon, lat]},
    "properties": {"source": "yolov8_distributed", "voltage": "6600",
                   "score": scores.get((round(lat,6), round(lon,6)), 0.5)},
} for lat, lon in deduped]

save_geojson_atomic(features, "output/japan_poles_raw.geojson")
print(f"\n合計（NMS後）: {len(features):,} 本")
print("→ output/japan_poles_raw.geojson")
PYEOF

# ─────────────────────────────────────────────────────────────
step "6. OSMデータとマージして最終出力"
# ─────────────────────────────────────────────────────────────
python3 06_merge_export.py

echo ""
ok "全処理完了。output/japan_poles_FINAL.geojson を確認してください。"
