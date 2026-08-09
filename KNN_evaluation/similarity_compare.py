"""双集合 embedding 相似度热力图对比 — 核心计算模块.

CLI 与 WebUI 共用单一事实源：采样 → 双集合批量提取 → 余弦相似度矩阵 →
并排热力图渲染。与 `metrics.py` 抽公共层的模式一致（CLI/WebUI 复用）。

采样模式：
- 数据库模式：复用 `sampling_map.ensure_sampling_map` 全库 point_id 候选池随机抽取；
- 图片模式：在指定 image_id 的 128×128 网格内随机抽不重复 (row, col) 像素，
  point_id 用与 `importer.build_points` 完全一致的 uuid5 公式换算。

导出功能（Task 8，方案 D9）：可附加导出双集合相似度矩阵（npy）与采样信息
（json，含 params + pixels），由 CLI --export-dir / WebUI 导出目录触发。
"""
import io
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np

from KNN_evaluation.sampling_map import ensure_sampling_map
from KNN_evaluation.visualization import plot_similarity_heatmap_pair

MAX_N = 600
IMAGE_GRID = 128
IMAGE_CELLS = IMAGE_GRID * IMAGE_GRID  # 16384


def _validate_n(n: int) -> None:
    """校验采样数 N（1..600，Deep Design D6 双保险之一）."""
    if not (1 <= int(n) <= MAX_N):
        raise ValueError(f"n 必须在 1..{MAX_N} 之间，实际: {n}")


def _point_id(image_id: str, row: int, col: int) -> str:
    """与 importer.build_points 完全一致的 point_id 换算公式（importer.py:106,129）."""
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, image_id)
    return str(uuid.uuid5(ns, f"{row}_{col}"))


def sample_random_points(
    manager,
    n: int,
    seed: int,
    image_id: str | None = None,
) -> list[dict]:
    """随机采样 N 个点（数据库全库模式 / 单张图片模式）.

    Args:
        manager: google 集合的 QdrantManager（作为采样侧；两集合 point_id 集合一致）。
        n: 目标采样数（1..600）。
        seed: 随机种子（random.Random(seed)，保证可复现）。
        image_id: None → 数据库模式（采样地图全库候选池）；
                  非 None → 图片模式（该 image_id 的 128×128 网格内随机不重复像素）。

    Returns:
        点字典列表；数据库模式每项 {point_id}；图片模式每项
        {point_id, image_id, pixel_row, pixel_col}。UTM 坐标不在采样阶段携带，
        提取阶段统一以 retrieve 返回的 payload 为准（Deep Design D2）。

    Raises:
        ValueError: n 越界；collection 不存在/为空；image_id 在 google 集合不存在。
        RuntimeError: 采样地图构建失败（by_label 全空但集合非空）。
    """
    _validate_n(n)
    rng = random.Random(seed)

    if image_id is None:
        # 数据库模式：复用采样地图候选池（按 label 合并即全库 point_id 池）
        if not manager.collection_exists():
            raise ValueError(
                f"Collection '{manager.collection_name}' 不存在，请先执行 import"
            )
        info = manager.collection_info()
        if info.get("total_points", 0) == 0:
            raise ValueError("Collection 为空，请先导入数据")
        sampling_map = ensure_sampling_map(manager)
        by_label = sampling_map.get("by_label") or {}
        candidates = [pid for ids in by_label.values() for pid in ids]
        if not candidates and info.get("total_points", 0) > 0:
            raise RuntimeError(
                "采样地图为空且构建失败：无法从 Qdrant 读取 point_id→label 地图，"
                f"请检查 collection '{manager.collection_name}' 与本地 qdrant_sampling_map.json"
            )
        n_actual = min(n, len(candidates))
        picked = rng.sample(candidates, n_actual)
        return [{"point_id": pid} for pid in picked]

    # 图片模式：128×128 网格内随机抽不重复像素
    from KNN_evaluation.data_loader import PixelDataLoader  # 函数内 import 防循环
    if PixelDataLoader.check_image_count(image_id, manager) <= 0:
        raise ValueError(
            f"image_id '{image_id}' 在 collection '{manager.collection_name}' 中不存在"
        )
    n_actual = min(n, IMAGE_CELLS)
    cells = rng.sample(
        [(r, c) for r in range(IMAGE_GRID) for c in range(IMAGE_GRID)],
        n_actual,
    )
    return [
        {
            "point_id": _point_id(image_id, r, c),
            "image_id": image_id,
            "pixel_row": r,
            "pixel_col": c,
        }
        for r, c in cells
    ]


def extract_embeddings(points, google_manager, xian_manager):
    """按同一 ids 列表从两个集合批量 retrieve，单侧缺失剔除并保持行序对齐.

    Args:
        points: sample_random_points 的返回（每项含 point_id）。
        google_manager: google 集合 QdrantManager。
        xian_manager: xian 集合 QdrantManager。

    Returns:
        (mat_g, mat_x, dropped, kept_records)：mat_g/mat_x 为 (N', 64) float64 矩阵，
        行序 = points 原始顺序的保留子序列（两侧逐行像素级对应）；
        dropped 为单侧缺失剔除数；
        kept_records 为保留像素信息列表，与 kept_ids 行序严格一致，每项
        {point_id, image_id, pixel_row, pixel_col, utm_easting, utm_northing,
        utm_zone}，字段从 google 侧 retrieve 返回的 payload 提取（缺失为 None）。

    Raises:
        RuntimeError: kept == 0（两侧无任何对齐点）。
        ValueError: 向量维度不是 64（与 searcher.py:113 防御一致）。
        ConnectionError / qdrant 连接异常：向上传播，由 CLI/WebUI 各自捕获。
    """
    ids = [p["point_id"] for p in points]
    g_recs = google_manager.client.retrieve(
        collection_name=google_manager.collection_name,
        ids=ids, with_payload=True, with_vectors=True,
    )
    x_recs = xian_manager.client.retrieve(
        collection_name=xian_manager.collection_name,
        ids=ids, with_payload=True, with_vectors=True,
    )
    g_by_id = {str(r.id): np.array(r.vector, dtype=np.float64) for r in g_recs}
    g_payload_by_id = {str(r.id): (r.payload or {}) for r in g_recs}
    x_by_id = {str(r.id): np.array(r.vector, dtype=np.float64) for r in x_recs}
    kept_ids = [pid for pid in ids if pid in g_by_id and pid in x_by_id]
    if not kept_ids:
        raise RuntimeError("两侧集合无任何对齐点：请检查双集合 image 集是否一致")
    mat_g = np.stack([g_by_id[pid] for pid in kept_ids])
    mat_x = np.stack([x_by_id[pid] for pid in kept_ids])
    if mat_g.shape[1] != 64:
        raise ValueError(f"google 集合向量维度应为 64，实际: {mat_g.shape[1]}")
    if mat_x.shape[1] != 64:
        raise ValueError(f"xian 集合向量维度应为 64，实际: {mat_x.shape[1]}")
    kept_records = [
        {
            "point_id": pid,
            "image_id": g_payload_by_id[pid].get("image_id"),
            "pixel_row": g_payload_by_id[pid].get("pixel_row"),
            "pixel_col": g_payload_by_id[pid].get("pixel_col"),
            "utm_easting": g_payload_by_id[pid].get("utm_easting"),
            "utm_northing": g_payload_by_id[pid].get("utm_northing"),
            "utm_zone": g_payload_by_id[pid].get("utm_zone"),
        }
        for pid in kept_ids
    ]
    return mat_g, mat_x, len(ids) - len(kept_ids), kept_records


def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    """N×N 余弦相似度矩阵（numpy 向量化，对角恰为 1.0）.

    与 Qdrant COSINE 度量一致（cos_sim = dot / (|a| |b|)）。
    零向量防御：范数为 0 时置 1 避免 NaN，并将对角强制为 1.0
    （Deep Design D4）。
    """
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    v = vecs / norm
    m = v @ v.T
    np.fill_diagonal(m, 1.0)
    return m


def export_similarity_outputs(
    sim_g: np.ndarray,
    sim_x: np.ndarray,
    meta: dict,
    pixels: list[dict],
    export_dir,
    collection_names: tuple[str, str],
    prefix: str = "",
    export_npy: bool = True,
) -> list[str]:
    """导出双集合相似度矩阵（npy）与采样信息（json）到指定目录（Task 8，方案 D9）.

    导出文件：
    - `{prefix}{collection_names[0]}_similarity.npy`：google 侧 N'×N' 余弦相似度矩阵；
    - `{prefix}{collection_names[1]}_similarity.npy`：xian 侧 N'×N' 余弦相似度矩阵；
    - `{prefix}similarity_sampling.json`：`{"params": meta, "pixels": pixels}`
      （`ensure_ascii=False, indent=2`），pixels 为保留像素信息列表，
      与 kept_ids 行序严格一致。

    Args:
        sim_g: google 侧相似度矩阵 (N', N')。
        sim_x: xian 侧相似度矩阵 (N', N')。
        meta: 采样/对比元数据（写入 json 的 params 块）。
        pixels: 保留像素信息列表（kept_records）。
        export_dir: 导出目录（不存在时自动 `mkdir(parents=True, exist_ok=True)`）。
        collection_names: (google 集合名, xian 集合名)，用于 npy 文件名。
        prefix: 可选文件名前缀（如 "full_col_" / "single_img_"）；为空时保持原名，
            CLI 默认不传。
        export_npy: 是否自动导出 npy 矩阵（默认 True）；False 时只写 sampling json，
            不写 npy（WebUI 手动导出场景，由用户点击「输出 npy 文件」落盘）。

    Returns:
        导出文件路径列表（export_npy 时 npy × 2 + json × 1 共 3 个；
        False 时仅 json × 1）。
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / f"{prefix}similarity_sampling.json"
    written: list[str] = []
    if export_npy:
        g_path = export_dir / f"{prefix}{collection_names[0]}_similarity.npy"
        x_path = export_dir / f"{prefix}{collection_names[1]}_similarity.npy"
        np.save(g_path, sim_g)
        np.save(x_path, sim_x)
        written.extend([str(g_path), str(x_path)])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"params": meta, "pixels": pixels}, f, ensure_ascii=False, indent=2)
    written.append(str(json_path))
    return written


def compare_similarity_heatmaps(
    google_manager,
    xian_manager,
    n: int = 200,
    seed: int = 42,
    image_id: str | None = None,
    output="similarity_heatmap.png",
    collection_names: tuple[str, str] = ("google_aef_embedding", "xian_aef_embedding"),
    export_dir="outputs",
    prefix: str = "",
    export_npy: bool = True,
) -> dict:
    """编排：采样 → 双集合提取 → 双矩阵 → 并排热力图渲染 → 元数据（D8 契约）.

    Args:
        google_manager: google 集合 QdrantManager（采样侧）。
        xian_manager: xian 集合 QdrantManager。
        n: 目标采样数（1..600，默认 200）。
        seed: 随机种子（默认 42）。
        image_id: None → 数据库模式；非 None → 图片模式。
        output: PNG 输出路径（CLI 落盘）；WebUI 传 io.BytesIO 时图已渲染进 buffer，
                output_path 返回空字符串。
        collection_names: (google 名, xian 名)，用于热力图标题。
        export_dir: 默认 "outputs"（Task 9：默认导出到 outputs/）；显式传 None 或
                空串 "" 时禁用导出。非禁用时导出采样信息到该目录（Task 8），
                export_npy=True 时同时导出双集合相似度矩阵，返回元数据新增
                `exported_files`。
        prefix: 可选导出文件名前缀（透传 export_similarity_outputs）；为空时原名。
        export_npy: 是否自动导出 npy 矩阵（默认 True，CLI 行为不变）；WebUI 传 False
            改由「输出 npy 文件」按钮手动导出（带时间戳、不覆盖旧文件）。

    Returns:
        {"sampled", "kept", "dropped", "matrix_shape": [N', N'],
         "elapsed_sec", "output_path", "sim_g", "sim_x"}；export_dir 非禁用时另有
         "exported_files"。

    Raises:
        ValueError / RuntimeError / ConnectionError：透传下层函数异常，
        由 CLI/WebUI 各自捕获转错误信息。
    """
    _validate_n(n)
    start = time.perf_counter()
    points = sample_random_points(google_manager, n, seed, image_id=image_id)
    sampled = len(points)
    mat_g, mat_x, dropped, kept_records = extract_embeddings(points, google_manager, xian_manager)
    sim_g = cosine_similarity_matrix(mat_g)
    sim_x = cosine_similarity_matrix(mat_x)
    # output 为 str 时归一化为 Path 再传给渲染函数（str 无 write_bytes，无法直接落盘）
    plot_similarity_heatmap_pair(
        sim_g,
        sim_x,
        Path(output) if isinstance(output, str) else output,
        collection_names=collection_names,
    )
    elapsed_sec = round(time.perf_counter() - start, 3)
    result = {
        "sampled": sampled,
        "kept": mat_g.shape[0],
        "dropped": dropped,
        "matrix_shape": [mat_g.shape[0], mat_g.shape[0]],
        "elapsed_sec": elapsed_sec,
        "output_path": "" if isinstance(output, io.BytesIO) else output,
        "sim_g": sim_g,
        "sim_x": sim_x,
    }
    if export_dir:
        meta = {
            "n": n,
            "seed": seed,
            "image_id": image_id,
            "collections": list(collection_names),
            "sampled": sampled,
            "kept": mat_g.shape[0],
            "dropped": dropped,
            "elapsed_sec": elapsed_sec,
        }
        result["exported_files"] = export_similarity_outputs(
            sim_g, sim_x, meta, kept_records, export_dir, collection_names,
            prefix=prefix, export_npy=export_npy,
        )
    return result
