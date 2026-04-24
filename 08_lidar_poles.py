#!/usr/bin/env python3
"""
LiDAR点群データから電柱を自動検出してGeoJSONを生成する。

## データソース（日本国内の無料点群データ）

1. G空間情報センター（geospatial.jp）の国交省・自治体LASデータ
   - 大阪市建設局: 街路・道路周辺の三次元点群 (LAS形式, JGD2011)
   - 天竜川地形計測など河川管理LASデータ
   - API: https://www.geospatial.jp/ckan/api/3/action/resource_search?query=format:LAS
   - ダウンロード例: https://www.geospatial.jp/ckan/dataset/{id}/resource/{rid}/download/{file}.las

2. 国土地理院 点群データ（有料 13,900円/2次メッシュ、LAZ形式）
   - 密度: 4点/m² 以上、高さ精度25cm
   - 問合: 日本地図センター 03-3485-5416
   - 刊行範囲: 北海道・東北・関東・北陸一部

3. PLATEAU（国土交通省都市局）
   - 形式: CityGML / 3DTiles（点群LAZは非提供）
   - 点群については別途国交省整備局LASデータを利用

## 電柱の特徴（LiDAR検出根拠）
   - 高さ: 8〜12m（地上基部から頂部まで）
   - 直径: 15〜30cm（断面が小さい）
   - 形状: ほぼ完全な垂直円柱
   - 密度: LiDAR点密度4点/m²では1本あたり数点〜数十点の帰還

## アルゴリズム概要
   1. LAZ/LASファイル読み込み（laspy）
   2. 地面点除去（最小高フィルタ＋グリッド最低値）
   3. XYグリッドでVoxel集約（5cm解像度）
   4. 各XYセルの高さ方向プロファイルで「細長い垂直柱」を判定
   5. 中心座標をJGD2011からWGS84へ変換
   6. 電柱候補をGeoJSON出力

## GPU活用
   - pws-gpu (RTX 4090) での大規模処理を想定
   - cupy / open3d-gpu でVoxel演算を高速化（オプション）
   - CPU版（numpy/scipy）でも動作する

## 必要ライブラリ
   pip install laspy lazrs-python numpy scipy pyproj tqdm requests
   pip install open3d   # オプション（可視化・GPU処理）
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

# ──────────────────────────────────────────────────────────
# 設定定数
# ──────────────────────────────────────────────────────────

# 電柱形状パラメータ
POLE_HEIGHT_MIN_M   = 6.0    # 検出する電柱の最低高さ [m]
POLE_HEIGHT_MAX_M   = 14.0   # 検出する電柱の最高高さ [m]
POLE_DIAM_MAX_M     = 0.60   # 電柱断面の最大直径 [m]（余裕を持たせる）
GROUND_PERCENTILE   = 10     # グラウンド推定パーセンタイル

# Voxelグリッド解像度
VOXEL_XY_M          = 0.30   # XY方向の集約セルサイズ [m]
VOXEL_Z_M           = 0.20   # Z方向のスライス厚 [m]

# 垂直性判定
MIN_FILLED_RATIO    = 0.50   # 高さ方向でこの割合以上スライスに点があれば「垂直」
MIN_HEIGHT_SLICES   = 20     # 最低スライス数（= POLE_HEIGHT_MIN_M / VOXEL_Z_M）

# NMS（重複除去）
NMS_RADIUS_M        = 3.0    # この半径内で近接する候補を1本に統合

# 出力
OUT_DIR             = Path("output")
OUT_GEOJSON         = OUT_DIR / "lidar_poles.geojson"

# G空間情報センターのサンプルLASデータセット（無料・オープン）
SAMPLE_DATASETS = [
    {
        "name": "大阪市野中南周辺（国交省/大阪市建設局, 令和3年度）",
        "package_id": "8eff3e5e-f9c0-455b-a08e-1d542d83183c",
        "resources": [
            # 実URLは package_show API で取得して埋める
        ],
        "crs": "EPSG:6677",   # JGD2011 / 日本平面直角座標系 VII
    },
    {
        "name": "天竜川地形計測（国交省浜松河川国道事務所）",
        "package_id": "a782e7ed-9c6d-43aa-9e32-209ac0d7aaa9",
        "resources": [],
        "crs": "EPSG:6676",   # JGD2011 / 日本平面直角座標系 VI
    },
]

GEOSPATIAL_API = "https://www.geospatial.jp/ckan/api/3/action"


# ──────────────────────────────────────────────────────────
# データ取得
# ──────────────────────────────────────────────────────────

def fetch_resource_urls(package_id: str) -> list[dict]:
    """G空間情報センターAPIからLAS/LAZリソースのURL一覧を取得する。"""
    url = f"{GEOSPATIAL_API}/package_show?id={package_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    result = resp.json()["result"]
    resources = []
    for r in result.get("resources", []):
        fmt = r.get("format", "").upper()
        if fmt in ("LAS", "LAZ"):
            resources.append({
                "name": r.get("name", ""),
                "url": r.get("url", ""),
                "format": fmt,
                "size": r.get("size", 0),
            })
    return resources


def download_las(url: str, dest_path: Path, show_progress: bool = True) -> Path:
    """LAS/LAZファイルをダウンロードする（再開不可・上書き）。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"        -> {dest_path}")

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        disable=not show_progress,
        desc=dest_path.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    return dest_path


# ──────────────────────────────────────────────────────────
# 点群処理（numpy/scipy版）
# ──────────────────────────────────────────────────────────

def load_las_points(path: Path) -> tuple[np.ndarray, dict]:
    """
    LAZ/LASを読み込んでXYZ配列(N,3)と付属情報を返す。

    Returns
    -------
    xyz : ndarray (N, 3)  [m単位、元座標系のまま]
    info : dict  {crs_wkt, scale, offset, ...}
    """
    try:
        import laspy
    except ImportError:
        sys.exit("laspy が見つかりません。'pip install laspy lazrs-python' を実行してください。")

    with laspy.open(str(path)) as las_file:
        las = las_file.read()

    scale  = np.array([las.header.x_scale, las.header.y_scale, las.header.z_scale])
    offset = np.array([las.header.x_offset, las.header.y_offset, las.header.z_offset])

    x = np.array(las.X, dtype=np.float64) * scale[0] + offset[0]
    y = np.array(las.Y, dtype=np.float64) * scale[1] + offset[1]
    z = np.array(las.Z, dtype=np.float64) * scale[2] + offset[2]

    xyz = np.column_stack([x, y, z])

    info = {
        "n_points":     len(x),
        "x_range":      (float(x.min()), float(x.max())),
        "y_range":      (float(y.min()), float(y.max())),
        "z_range":      (float(z.min()), float(z.max())),
        "scale":        scale.tolist(),
        "offset":       offset.tolist(),
    }
    print(f"  読込完了: {info['n_points']:,} 点  "
          f"X=[{info['x_range'][0]:.1f}, {info['x_range'][1]:.1f}]  "
          f"Z=[{info['z_range'][0]:.1f}, {info['z_range'][1]:.1f}]")
    return xyz, info


def estimate_ground(xyz: np.ndarray, grid_m: float = 2.0) -> np.ndarray:
    """
    グリッド別最低標高でグラウンドサーフェスを推定し、各点の「地面からの高さ」を返す。

    Parameters
    ----------
    xyz     : (N,3) 点群
    grid_m  : グラウンド推定セルサイズ [m]

    Returns
    -------
    height_above_ground : (N,) 各点の地面高 [m]
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # グリッドインデックス
    xi = ((x - x.min()) / grid_m).astype(np.int32)
    yi = ((y - y.min()) / grid_m).astype(np.int32)
    cell_id = xi * (yi.max() + 1) + yi

    # セルごとの最低標高（= 地面近似）
    cell_min_z = np.full(cell_id.max() + 1, np.inf)
    np.minimum.at(cell_min_z, cell_id, z)

    # 周辺セルの最低標高を median 平滑化（外れ値除去）
    # ここでは簡易版として cell_min_z をそのまま使用
    ground_z = cell_min_z[cell_id]

    return z - ground_z


def detect_poles_voxel(
    xyz: np.ndarray,
    hag: np.ndarray,
    voxel_xy: float = VOXEL_XY_M,
    voxel_z:  float = VOXEL_Z_M,
    height_min: float = POLE_HEIGHT_MIN_M,
    height_max: float = POLE_HEIGHT_MAX_M,
    diam_max:   float = POLE_DIAM_MAX_M,
    min_filled: float = MIN_FILLED_RATIO,
) -> list[dict]:
    """
    XYグリッド + 高さプロファイル解析で電柱候補を検出する。

    アルゴリズム:
      1. 地上 height_min 〜 height_max の点のみ対象
      2. XYをvoxel_xyセルに集約
      3. 各XYセルで Z方向プロファイルを作成
      4. 連続した高さスライスに点が存在する割合 (filled_ratio) を計算
      5. filled_ratio >= min_filled かつ 断面直径 <= diam_max なら電柱候補

    Returns
    -------
    list of dict: {"x_center", "y_center", "height", "filled_ratio", "n_points"}
    """
    # --- 高さ範囲フィルタ ---
    mask = (hag >= height_min) & (hag <= height_max)
    xyz_f = xyz[mask]
    hag_f = hag[mask]

    if len(xyz_f) == 0:
        print("  [警告] フィルタ後の点が0件。高さ推定パラメータを確認してください。")
        return []

    x, y, z = xyz_f[:, 0], xyz_f[:, 1], xyz_f[:, 2]

    # --- XYグリッド ---
    xi = np.floor((x - x.min()) / voxel_xy).astype(np.int32)
    yi = np.floor((y - y.min()) / voxel_xy).astype(np.int32)

    # セルIDごとに点を集約
    cell_id = xi * (yi.max() + 1) + yi
    unique_cells = np.unique(cell_id)

    # セルの実XY座標（中心）
    x_origin = x.min()
    y_origin = y.min()

    n_z_slices_expected = int((height_max - height_min) / voxel_z)

    candidates = []
    for cid in unique_cells:
        pts_mask = cell_id == cid
        pts_z   = hag_f[pts_mask]     # この列の地面からの高さ
        n_pts   = pts_mask.sum()

        if n_pts < 3:
            continue

        # 高さプロファイル: 各スライスに点があるか？
        z_bins = np.floor((pts_z - height_min) / voxel_z).astype(np.int32)
        z_bins = np.clip(z_bins, 0, n_z_slices_expected - 1)
        filled_slices = len(np.unique(z_bins))
        filled_ratio  = filled_slices / max(n_z_slices_expected, 1)

        if filled_ratio < min_filled:
            continue

        # 断面直径チェック（XY広がりが大きすぎる→木や建物）
        pts_x = xyz_f[pts_mask, 0]
        pts_y = xyz_f[pts_mask, 1]
        dx = pts_x.max() - pts_x.min()
        dy = pts_y.max() - pts_y.min()
        if dx > diam_max or dy > diam_max:
            continue

        # 電柱の高さ（= 点群の最大 hag）
        pole_height = float(pts_z.max())

        candidates.append({
            "x_center":    float(pts_x.mean()),
            "y_center":    float(pts_y.mean()),
            "height":      pole_height,
            "filled_ratio": float(filled_ratio),
            "n_points":    int(n_pts),
        })

    print(f"  Voxel候補: {len(candidates)} 本（フィルタ前 {len(xyz_f):,} 点から）")
    return candidates


def nms_poles(candidates: list[dict], radius_m: float = NMS_RADIUS_M) -> list[dict]:
    """
    近接する電柱候補を1本に統合する（Non-Maximum Suppression）。
    filled_ratio が最も高いものを残す。
    """
    if not candidates:
        return []

    # filled_ratio降順でソート
    cands = sorted(candidates, key=lambda c: c["filled_ratio"], reverse=True)
    kept = []
    suppressed = [False] * len(cands)

    for i, c in enumerate(cands):
        if suppressed[i]:
            continue
        kept.append(c)
        for j in range(i + 1, len(cands)):
            if suppressed[j]:
                continue
            dx = c["x_center"] - cands[j]["x_center"]
            dy = c["y_center"] - cands[j]["y_center"]
            if (dx ** 2 + dy ** 2) < radius_m ** 2:
                suppressed[j] = True

    print(f"  NMS後: {len(kept)} 本（{len(cands)} 候補 → {len(kept)} 本）")
    return kept


# ──────────────────────────────────────────────────────────
# 座標変換
# ──────────────────────────────────────────────────────────

def proj_to_wgs84(candidates: list[dict], src_crs: str) -> list[dict]:
    """
    平面直角座標系 (JGD2011) の XY を WGS84 緯度経度に変換する。

    Parameters
    ----------
    src_crs : str  e.g. "EPSG:6677" (日本平面直角座標系VII)
    """
    try:
        from pyproj import Transformer
    except ImportError:
        print("[警告] pyproj が見つかりません。座標変換をスキップ（XY値をそのまま使用）。")
        print("       pip install pyproj で インストールしてください。")
        for c in candidates:
            c["lon"], c["lat"] = c["x_center"], c["y_center"]
        return candidates

    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    for c in candidates:
        lon, lat = transformer.transform(c["x_center"], c["y_center"])
        c["lon"] = float(lon)
        c["lat"] = float(lat)

    return candidates


def las_epsg_from_vlr(path: Path) -> str | None:
    """
    LASファイルのVLR（Variable Length Record）からEPSGコードを読み取る。
    失敗したら None を返す。
    """
    try:
        import laspy
        with laspy.open(str(path)) as lf:
            las = lf.read()
        for vlr in las.header.vlrs:
            if vlr.record_id == 2112:   # WKT
                wkt = vlr.record_data.decode(errors="replace")
                import re
                m = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', wkt)
                if m:
                    return f"EPSG:{m.group(1)}"
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────
# GeoJSON出力
# ──────────────────────────────────────────────────────────

def to_geojson(poles: list[dict], source_file: str = "") -> dict:
    """電柱候補リストを GeoJSON FeatureCollection に変換する。"""
    features = []
    for i, p in enumerate(poles):
        lat = p.get("lat", p.get("y_center"))
        lon = p.get("lon", p.get("x_center"))
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "id":           i,
                "height_m":     round(p["height"], 2),
                "filled_ratio": round(p["filled_ratio"], 3),
                "n_points":     p["n_points"],
                "source":       source_file,
                "method":       "lidar_voxel",
            },
        }
        features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def save_geojson(geojson: dict, path: Path) -> None:
    """GeoJSONをアトミックに書き込む。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.geojson")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    tmp.rename(path)
    print(f"  保存: {path}  ({len(geojson['features'])} 件)")


# ──────────────────────────────────────────────────────────
# パイプライン
# ──────────────────────────────────────────────────────────

def process_las_file(
    las_path: Path,
    src_crs:  str,
    voxel_xy: float = VOXEL_XY_M,
    voxel_z:  float = VOXEL_Z_M,
) -> list[dict]:
    """
    1つのLAS/LAZファイルを処理して電柱候補リスト（WGS84）を返す。
    """
    print(f"\n=== 処理中: {las_path.name} ===")

    # 1. 読み込み
    xyz, info = load_las_points(las_path)

    # 2. EPSGをVLRから自動取得（フォールバック: 引数）
    epsg = las_epsg_from_vlr(las_path) or src_crs
    print(f"  CRS: {epsg}")

    # 3. 地面高推定
    print("  地面高推定中...")
    hag = estimate_ground(xyz)

    # 4. Voxelベース検出
    print("  電柱候補検出中（Voxelグリッド）...")
    candidates = detect_poles_voxel(xyz, hag, voxel_xy=voxel_xy, voxel_z=voxel_z)

    # 5. NMS
    candidates = nms_poles(candidates)

    # 6. WGS84変換
    candidates = proj_to_wgs84(candidates, epsg)

    return candidates


def run_pipeline(
    las_files: list[tuple[Path, str]],
    out_geojson: Path = OUT_GEOJSON,
    voxel_xy: float = VOXEL_XY_M,
    voxel_z:  float = VOXEL_Z_M,
) -> None:
    """
    複数LASファイルを処理して結果を統合GeoJSONに書き出す。

    Parameters
    ----------
    las_files : list of (Path, src_crs_str)
    """
    all_poles = []
    for las_path, src_crs in las_files:
        poles = process_las_file(las_path, src_crs, voxel_xy, voxel_z)
        for p in poles:
            p["source_file"] = las_path.name
        all_poles.extend(poles)

    print(f"\n合計 {len(all_poles)} 本の電柱候補を検出")
    geojson = to_geojson(all_poles)
    save_geojson(geojson, out_geojson)


# ──────────────────────────────────────────────────────────
# サンプルデータ自動ダウンロード
# ──────────────────────────────────────────────────────────

def download_sample_dataset(dataset_info: dict, dest_dir: Path) -> list[tuple[Path, str]]:
    """
    G空間情報センターからサンプルLASデータをダウンロードして返す。

    Returns
    -------
    list of (Path, src_crs)
    """
    pkg_id  = dataset_info["package_id"]
    src_crs = dataset_info["crs"]
    name    = dataset_info["name"]
    print(f"\n--- データ取得: {name} ---")

    resources = fetch_resource_urls(pkg_id)
    if not resources:
        print(f"  [スキップ] LAS/LAZリソースが見つかりません (package_id={pkg_id})")
        return []

    print(f"  LAS/LAZファイル: {len(resources)} 個")
    downloaded = []
    for r in resources[:3]:    # デモ用に先頭3ファイルのみ
        fname   = Path(r["name"]).name or f"{pkg_id}_{len(downloaded)}.las"
        dest    = dest_dir / fname
        if dest.exists():
            print(f"  [スキップ] 既存ファイル: {dest}")
        else:
            download_las(r["url"], dest)
        downloaded.append((dest, src_crs))

    return downloaded


# ──────────────────────────────────────────────────────────
# open3d版（GPU対応、オプション）
# ──────────────────────────────────────────────────────────

def detect_poles_open3d(xyz: np.ndarray, hag: np.ndarray) -> list[dict]:
    """
    open3dのVoxelDownSamplingとクラスタリングを使った検出（オプション）。
    より高精度だが open3d のインストールが必要。

    Usage: pip install open3d
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[警告] open3d が見つかりません。voxelベースにフォールバックします。")
        return detect_poles_voxel(xyz, hag)

    # 高さフィルタ
    mask = (hag >= POLE_HEIGHT_MIN_M) & (hag <= POLE_HEIGHT_MAX_M)
    xyz_f = xyz[mask]
    if len(xyz_f) == 0:
        return []

    # open3d PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_f)

    # Voxelダウンサンプリング
    pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_XY_M)

    # DBSCAN クラスタリング
    labels = np.array(pcd_down.cluster_dbscan(
        eps=POLE_DIAM_MAX_M,
        min_points=5,
        print_progress=False,
    ))

    pts = np.asarray(pcd_down.points)
    candidates = []
    for label in np.unique(labels):
        if label < 0:
            continue
        cluster_pts = pts[labels == label]
        x_c = cluster_pts[:, 0]
        y_c = cluster_pts[:, 1]
        z_c = cluster_pts[:, 2]

        # 断面チェック
        dx = x_c.max() - x_c.min()
        dy = y_c.max() - y_c.min()
        if dx > POLE_DIAM_MAX_M or dy > POLE_DIAM_MAX_M:
            continue

        # 高さチェック（hag近似: z最大-z最小 ≈ 高さ）
        dz = z_c.max() - z_c.min()
        if dz < POLE_HEIGHT_MIN_M or dz > POLE_HEIGHT_MAX_M:
            continue

        candidates.append({
            "x_center":    float(x_c.mean()),
            "y_center":    float(y_c.mean()),
            "height":      float(dz),
            "filled_ratio": 1.0,
            "n_points":    len(cluster_pts),
        })

    return candidates


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LiDAR点群から電柱を検出してGeoJSONを生成する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # サンプルデータ自動ダウンロード + 処理
  python 08_lidar_poles.py --sample

  # 既存LASファイルを処理（CRS指定）
  python 08_lidar_poles.py --input data.las --crs EPSG:6677

  # ディレクトリ内の全LAS/LAZを一括処理
  python 08_lidar_poles.py --input-dir /path/to/las/ --crs EPSG:6675

  # open3dを使う（より高精度）
  python 08_lidar_poles.py --input data.las --crs EPSG:6677 --method open3d

  # Voxelサイズを調整（高密度点群向け）
  python 08_lidar_poles.py --input data.las --crs EPSG:6677 --voxel-xy 0.20

データソース確認:
  https://www.geospatial.jp/ckan/api/3/action/resource_search?query=format:LAS
  https://www.geospatial.jp/ckan/api/3/action/resource_search?query=format:LAZ

国土地理院点群（有料）:
  https://www.gsi.go.jp/gazochosa/tengun.html
  日本地図センター 03-3485-5416 / 2次メッシュ 13,900円
        """,
    )
    p.add_argument("--sample", action="store_true",
                   help="G空間情報センターからサンプルLASデータをダウンロードして処理")
    p.add_argument("--input", type=Path, default=None,
                   help="処理するLAS/LAZファイル（単体）")
    p.add_argument("--input-dir", type=Path, default=None,
                   help="LAS/LAZファイルが含まれるディレクトリ（一括処理）")
    p.add_argument("--crs", type=str, default="EPSG:6677",
                   help="入力座標系 (デフォルト: EPSG:6677 = 日本平面直角VII)")
    p.add_argument("--out", type=Path, default=OUT_GEOJSON,
                   help=f"出力GeoJSONパス (デフォルト: {OUT_GEOJSON})")
    p.add_argument("--method", choices=["voxel", "open3d"], default="voxel",
                   help="検出アルゴリズム (デフォルト: voxel)")
    p.add_argument("--voxel-xy", type=float, default=VOXEL_XY_M,
                   help=f"XYグリッドサイズ [m] (デフォルト: {VOXEL_XY_M})")
    p.add_argument("--voxel-z", type=float, default=VOXEL_Z_M,
                   help=f"Z方向スライス幅 [m] (デフォルト: {VOXEL_Z_M})")
    p.add_argument("--list-datasets", action="store_true",
                   help="G空間情報センターで利用可能な点群データセット一覧を表示")
    return p.parse_args()


def list_available_datasets():
    """G空間情報センターで利用可能なLAS/LAZデータセットを検索・表示する。"""
    print("=== G空間情報センター LAS/LAZ データセット検索 ===\n")
    url = f"{GEOSPATIAL_API}/resource_search?query=format:LAS&rows=20"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    total = data["result"]["count"]
    print(f"LAS形式: 合計 {total} リソース\n")

    seen_pkgs = set()
    for r in data["result"]["results"]:
        pkg_id = r.get("package_id", "")
        if pkg_id in seen_pkgs:
            continue
        seen_pkgs.add(pkg_id)
        print(f"  名前: {r.get('name', '(不明)')}")
        print(f"  URL:  {r.get('url', '')}")
        print(f"  pkg:  {pkg_id}")
        print()

    # LAZ も表示
    url2 = f"{GEOSPATIAL_API}/resource_search?query=format:LAZ&rows=10"
    resp2 = requests.get(url2, timeout=30)
    resp2.raise_for_status()
    data2 = resp2.json()
    total2 = data2["result"]["count"]
    print(f"LAZ形式: 合計 {total2} リソース")
    for r in data2["result"]["results"][:5]:
        print(f"  名前: {r.get('name', '(不明)')}")
        print(f"  URL:  {r.get('url', '')}")
        print()


def main():
    args = parse_args()

    if args.list_datasets:
        list_available_datasets()
        return

    las_files: list[tuple[Path, str]] = []

    # --- サンプルデータ自動取得 ---
    if args.sample:
        data_dir = Path("data/lidar_sample")
        data_dir.mkdir(parents=True, exist_ok=True)
        for ds in SAMPLE_DATASETS:
            pairs = download_sample_dataset(ds, data_dir)
            las_files.extend(pairs)

    # --- 単体ファイル指定 ---
    if args.input:
        if not args.input.exists():
            sys.exit(f"ファイルが見つかりません: {args.input}")
        las_files.append((args.input, args.crs))

    # --- ディレクトリ一括 ---
    if args.input_dir:
        if not args.input_dir.is_dir():
            sys.exit(f"ディレクトリが見つかりません: {args.input_dir}")
        for ext in ("*.las", "*.LAS", "*.laz", "*.LAZ"):
            for f in sorted(args.input_dir.glob(ext)):
                las_files.append((f, args.crs))

    if not las_files:
        print("処理対象ファイルがありません。")
        print("  --sample  でサンプルデータを自動ダウンロード")
        print("  --input   で既存LASファイルを指定")
        print("  --list-datasets  で利用可能データセットを一覧表示")
        print()
        print("データソース:")
        print("  国土地理院点群（有料）: https://www.gsi.go.jp/gazochosa/tengun.html")
        print("  G空間情報センター(LAS): https://www.geospatial.jp/ckan/api/3/action/resource_search?query=format:LAS")
        return

    # --- 処理実行 ---
    if args.method == "open3d":
        # open3dモードは process_las_file 内で自動選択させる
        # detect_poles_open3d に差し替えるため、個別処理
        all_poles = []
        for las_path, src_crs in las_files:
            print(f"\n=== 処理中 (open3d): {las_path.name} ===")
            xyz, info = load_las_points(las_path)
            epsg = las_epsg_from_vlr(las_path) or src_crs
            hag  = estimate_ground(xyz)
            cands = detect_poles_open3d(xyz, hag)
            cands = nms_poles(cands)
            cands = proj_to_wgs84(cands, epsg)
            for c in cands:
                c["source_file"] = las_path.name
            all_poles.extend(cands)
        print(f"\n合計 {len(all_poles)} 本の電柱候補を検出 (open3d)")
        geojson = to_geojson(all_poles)
        save_geojson(geojson, args.out)
    else:
        run_pipeline(las_files, args.out, args.voxel_xy, args.voxel_z)


if __name__ == "__main__":
    main()
