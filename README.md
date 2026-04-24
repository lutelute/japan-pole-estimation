# 日本全国 6.6kV 電柱位置推定パイプライン

再生可能エネルギーの系統連系申請（6.6kV 配電線接続）に必要な全国電柱位置データを、OpenStreetMap と国土地理院航空写真および YOLOv8 を組み合わせて自動生成するパイプラインです。

---

## 背景と目的

6.6kV 配電線への連系申請では、接続する電柱の「柱番号」を電力会社に提示する必要があります。しかし柱番号は電力会社の非公開情報であり、申請者が自力で取得することは困難です。本パイプラインでは航空写真から電柱を自動検出し、GeoJSON 形式で位置データを生成します。QGIS 上で現地確認・絞り込みを行った上で、電力会社に柱番号の照合を依頼するフローを想定しています。

---

## パイプライン概要

```
OSMデータ取得 (01)
    │ 6.6kV 配電線ルート
    ▼
航空写真タイルDL (02)
    │ 国土地理院 zoom18 (~60cm/px)
    │ 配電線バッファ 100m 内のみ
    │ 454,579 枚 / 7.0 GB
    ▼
YOLOデータセット作成 (03)
    │ OSMの電柱ノードをラベルとして使用
    │ 福井県: 1,240 タイル / 4,934 ラベル
    ▼
YOLOv8x 学習 (04)
    │ COCO 事前学習済みモデルからファインチューニング
    │ RTX 4090 / 50 epochs / ~8時間
    ▼
全国 GPU 推論 (05)
    │ RTX 4090 + RTX 3060 並列推論
    │ RANK=0 / RANK=1 で奇偶分割
    ▼
OSM + CV 統合・出力 (06)
    OSM確定電柱 + YOLOv8検出電柱 → 都道府県別 GeoJSON
```

---

## 技術スタック

| 要素 | 内容 |
|------|------|
| 地図データ | OpenStreetMap（Overpass API） |
| 航空写真 | 国土地理院シームレス写真（zoom 18, 256×256 px, JPEG） |
| 物体検出 | YOLOv8x（ultralytics） |
| 学習データ | OSM の `power=pole` ノードを教師ラベルとして自動生成 |
| 推論環境 | RTX 4090 (24 GB) + RTX 3060 (12 GB) 並列 |
| 出力形式 | GeoJSON（EPSG:4326 / WGS84） |
| 言語 | Python 3.11 |

---

## ファイル構成

```
japan_poles/
├── 01_get_osm_japan.py        # Overpass API で 47 県分の配電線・電柱データ取得
├── 02_download_tiles_parallel.py  # 国土地理院タイル並列ダウンロード（32スレッド）
├── 03_prepare_yolo_dataset.py # OSMラベルから YOLO 学習データセット生成
├── 04_train_yolo.py           # YOLOv8x ファインチューニング（GPU必須）
├── 05_gpu_inference.py        # 全タイル推論（RANK分割対応）
├── 06_merge_export.py         # OSM + CV 統合、都道府県別 GeoJSON 出力
├── utils.py                   # 共通ユーティリティ（地理計算・グリッド近傍探索）
├── run_japan.sh               # ワンショット実行スクリプト
├── run_distributed.sh         # 2台 GPU 並列推論スクリプト
├── watch_progress.py          # 推論進捗モニタリング
├── data/
│   └── osm/                   # 47 県分の OSM JSON（取得済み）
├── tiles/                     # 国土地理院タイル（454,579 枚, 7.0 GB）※gitignore
├── yolo_dataset/              # YOLO 学習データセット※gitignore
│   ├── dataset.yaml
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── models/                    # 学習済みモデル※gitignore
│   └── pole_yolov8.pt
└── output/                    # 推論結果 GeoJSON※gitignore
```

---

## 実行手順

### ステップ 1: OSM データ取得（〜20 分）

```bash
pip install requests
python3 01_get_osm_japan.py
# 出力: data/osm/{pref_name}.json × 47 県
```

Overpass API（3 サーバーで負荷分散）から `power=minor_line` および `power=pole` を取得します。取得済み県はスキップするため再実行可能です。

### ステップ 2: タイルダウンロード（〜2 時間）

```bash
python3 02_download_tiles_parallel.py
# 出力: tiles/18_{x}_{y}.jpg
```

各 OSM ノードから半径 100m のバッファ内にある zoom 18 タイルを 32 スレッドで並列ダウンロードします。取得済みタイルはスキップします。

| 統計 | 値 |
|------|-----|
| 対象タイル数 | 454,579 枚 |
| 合計サイズ | 7.0 GB |
| 解像度 | zoom 18 ≈ 60 cm/px |

### ステップ 3: YOLO データセット作成

```bash
python3 03_prepare_yolo_dataset.py
# 出力: yolo_dataset/
```

OSM の `power=pole` ノード座標をタイル内ピクセル座標に変換し、YOLO 形式（`{cls} {cx} {cy} {w} {h}`）のラベルファイルを生成します。

| 統計（福井県） | 値 |
|----------------|-----|
| 学習タイル数 | 1,240 枚 |
| ラベル数 | 4,934 個 |
| 検証タイル数 | 310 枚 |

### ステップ 4: YOLOv8x 学習（GPU 必須・〜8 時間）

```bash
# GPUサーバー（pws-gpu: RTX 4090）で実行
python3 04_train_yolo.py
# 出力: models/pole_yolov8.pt
```

| 学習設定 | 値 |
|----------|-----|
| ベースモデル | yolov8x（COCO 事前学習済み） |
| エポック数 | 50 |
| 画像サイズ | 256×256 px |
| バッチサイズ | 16 |
| 学習率 | 1e-3 → 1e-5（cosine decay） |
| データ拡張 | flip(ud/lr), rotate±10°, HSV jitter |
| Early stopping | patience=15 |
| GPU | RTX 4090 (24 GB VRAM) |

**学習結果（Epoch 50/50）:**

| 指標 | 値 |
|------|-----|
| Precision | 0.066 |
| Recall | 0.066 |
| mAP@50 | 0.021 |
| mAP@50-95 | 0.005 |

> 精度が低い主な原因: 学習データが福井県のみ（地域偏り）、OSM の電柱ラベルが不完全（見落とし・位置誤差あり）。改善策として多地点サンプリングによる学習データ拡充を計画中。

### ステップ 5: 全国 GPU 推論（GPU 必須・〜数時間）

```bash
# 2台並列実行（run_distributed.sh を参照）
# GPU0 (RTX 4090)
RANK=0 WORLD_SIZE=2 python3 05_gpu_inference.py

# GPU1 (RTX 3060) — 別サーバーで同時実行
RANK=1 WORLD_SIZE=2 python3 05_gpu_inference.py

# または 1台で全処理
python3 05_gpu_inference.py
```

- バッチサイズ 64、confidence threshold 0.35
- 配電線バッファ 100m フィルタで不要な検出を除去
- 半径 15m の距離ベース NMS で重複除去
- 出力: `output/japan_poles_rank{0,1}.geojson`

### ステップ 6: マージ・GeoJSON 出力

```bash
python3 06_merge_export.py
# 出力: output/japan_poles_FINAL.geojson
#       output/by_pref/{pref}.geojson × 47 県
```

OSM 確定電柱（`power=pole`）と YOLOv8 検出電柱を統合し、半径 15m NMS で重複除去した上で都道府県別に GeoJSON を出力します。

---

## GPU サーバー構成

| サーバー | 接続先 | GPU | 役割 |
|---------|--------|-----|------|
| pws-gpu | `root@100.126.144.110` (Tailscale) | RTX 4090 24 GB | 学習 + RANK=0 推論 |
| pws-gpu3060 | `ubuntu@100.117.16.18` (Tailscale) | RTX 3060 12 GB | RANK=1 推論 |

---

## 2 台 GPU 並列推論の実行方法

```bash
# ローカルから一括実行（タイル転送 → 学習 → 推論 → マージ）
bash run_distributed.sh
```

このスクリプトは以下を自動化します:
1. コード・OSMデータを GPU0/GPU1 へ rsync
2. GPU0 で学習データ作成 + YOLOv8 学習
3. 全タイルを GPU0/GPU1 へ転送（RANK 分割）
4. 両 GPU で並列推論
5. 推論結果を回収してマージ

---

## 依存ライブラリ

```bash
pip install ultralytics requests
```

推論・学習には CUDA 対応 GPU が必要です。CPU での実行も可能ですが実用的な速度は出ません。

---

## 出力の QGIS での利用

1. レイヤー → レイヤーを追加 → ベクタレイヤー
2. `output/by_pref/{県名}.geojson` を選択
3. 国土地理院背景図（XYZ タイル）と重ねて目視確認

各電柱の属性:
- `source`: `osm`（OSM確定）または `yolov8`（CV検出）
- `score`: YOLOv8 の信頼スコア（0〜1）
- `voltage`: `6600`（6.6 kV）

---

## 改善計画

- **多地点学習データ拡充**: 福井県のみの学習データを都市部（東京・大阪）、農村（秋田・岩手）、山間（長野）、沿岸（沖縄）等から追加サンプリングしてファインチューン
- **精度評価**: Precision/Recall の地域別分析
- **北陸電力への研究協力依頼**: 柱番号データによる ground truth 整備

---

## 現在の進捗（2026-04-24 時点）

| ステップ | 状態 |
|---------|------|
| OSMデータ取得（47県） | 完了 |
| タイルDL（454,579枚） | 完了 |
| YOLOv8学習（50 epoch） | 完了（RTX 4090） |
| 全国推論 | **実行中**（タイル転送中） |
| GeoJSON出力 | 未実施 |
