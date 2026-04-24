"""
Grounding DINO で「utility pole」を自動検出するテスト。
pws-gpu (RTX 4090) で実行。
"""
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import json, os

DEVICE = "cuda"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # まず tiny で速度確認
TILE_DIR = Path("/root/pole_estimation/tiles")
OUT_JSON  = Path("/root/pole_estimation/gdino_test_results.json")

TEXT_PROMPT = "utility pole . electric pole . power pole ."
BOX_THRESH  = 0.25
TEXT_THRESH = 0.25

# タイルを20枚サンプリング（OSMの電柱ノードがある付近を優先）
# 福井県: タイル名に 230222, 230223 付近
sample_tiles = sorted([
    t for t in TILE_DIR.glob("18_2302*.jpg")
])[:20]

if not sample_tiles:
    sample_tiles = sorted(TILE_DIR.glob("*.jpg"))[:20]

print(f"テスト対象: {len(sample_tiles)} 枚")
print(f"モデル: {MODEL_ID}")
print(f"デバイス: {DEVICE}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("モデルロード完了")

results = []
for i, tile_path in enumerate(sample_tiles):
    img = Image.open(tile_path).convert("RGB")
    inputs = processor(
        images=img,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    w, h = img.size
    results_proc = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESH,
        text_threshold=TEXT_THRESH,
        target_sizes=[(h, w)]
    )[0]

    boxes  = results_proc["boxes"].cpu().tolist()
    scores = results_proc["scores"].cpu().tolist()
    labels = results_proc["labels"]

    tile_result = {
        "tile": tile_path.name,
        "detections": len(boxes),
        "boxes": [[round(v, 2) for v in b] for b in boxes],
        "scores": [round(s, 4) for s in scores],
        "labels": labels,
    }
    results.append(tile_result)
    print(f"[{i+1:2d}/{len(sample_tiles)}] {tile_path.name}: {len(boxes)} 検出 scores={[round(s,2) for s in scores]}")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

total = sum(r["detections"] for r in results)
print(f"\n合計検出数: {total} / {len(sample_tiles)} タイル")
print(f"出力: {OUT_JSON}")
