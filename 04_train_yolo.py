#!/usr/bin/env python3
"""
YOLOv8x を電柱検出用にファインチューニングする。

前提: pip install ultralytics
GPU 必須（GPUサーバーで実行すること）。
出力: models/pole_yolov8.pt
"""

import sys
from pathlib import Path

DATASET_YAML = Path("yolo_dataset/dataset.yaml")
OUT_DIR      = Path("models")
EPOCHS       = 50
IMGSZ        = 256
BATCH        = 16
BASE_MODEL   = "yolov8x.pt"   # COCO事前学習済み


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics がインストールされていません。")
        print("  pip install ultralytics")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("警告: GPUが検出されません。CPUで学習すると非常に時間がかかります。")
        print("GPUサーバー（pws-gpu: RTX 4090, pws-gpu3060: RTX 3060）で実行してください。")
        resp = input("続行しますか？ [y/N]: ").strip().lower()
        if resp != "y":
            sys.exit(0)

    if not DATASET_YAML.exists():
        print(f"dataset.yaml が見つかりません: {DATASET_YAML}")
        print("先に 03_prepare_yolo_dataset.py を実行してください。")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"学習デバイス: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nbase model: {BASE_MODEL}")
    print(f"epochs: {EPOCHS}  imgsz: {IMGSZ}  batch: {BATCH}")
    print(f"dataset: {DATASET_YAML}\n")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(DATASET_YAML.resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        project=str(OUT_DIR / "runs"),
        name="pole_yolov8",
        patience=15,           # early stopping
        save=True,
        save_period=10,
        val=True,
        workers=0,       # pin_memory エラー回避
        lr0=1e-3,
        lrf=0.01,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        degrees=10.0,
    )

    # 最良モデルをコピー
    best_pt = OUT_DIR / "runs" / "pole_yolov8" / "weights" / "best.pt"
    final_pt = OUT_DIR / "pole_yolov8.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, final_pt)
        print(f"\n学習済みモデル保存: {final_pt}")
    else:
        print(f"best.pt が見つかりません: {best_pt}")
        sys.exit(1)

    # 評価サマリ
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    if metrics:
        print("\n=== 評価結果 ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
