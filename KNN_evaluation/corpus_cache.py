"""全库向量磁盘缓存：避免评估重复下载全量向量（Section 11）.

背景：`evaluate_knn` 每次评估都调用 `_scroll_full_vectors` 下载全库向量
（10.2M 点 ≈ 2.6GB，约 205 批 scroll，3-4 分钟）。同一 collection 重复评估
（不同采样/参数）时每次都重下。本模块在磁盘上缓存全量向量，首次下载后写入，
后续评估直接加载（NVMe 读 2.6GB npz 约 5-10s，对比每次 3-4 分钟下载）。

与 `sampling_map.py` / `manifest.py` 同模式：**可重建缓存，不是唯一真相**；
文件缺失/损坏返回空结构，由 `ensure_corpus_cache` 自动重建；原子写
（tmp + os.replace）。与采样地图（ID 地图）互补：地图管采样定位，缓存管评估全量。

缓存文件：`qdrant_corpus_cache/{sha256(collection)[:16]}.npz`，内容：
`vectors (N,64) float32` + `labels (N,) int64` + `point_ids (N,) str` +
元数据 `collection`（str）+ `total_points`（int，指纹对账用）。
"""
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from KNN_evaluation.qdrant_client import QdrantManager

# 缓存目录（与 qdrant_sampling_map.json / qdrant_import_manifest.json 同级，项目根目录）
CORPUS_CACHE_DIR = Path("qdrant_corpus_cache")

_SCROLL_BATCH = 50000
_VECTOR_DIM = 64


def _empty_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """缺失/损坏/空缓存对应的空数组（触发重建或空 Collection 语义）."""
    return (
        np.empty((0, _VECTOR_DIM), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=str),
    )


def _cache_path(collection: str, dirpath: str | os.PathLike | None = None) -> Path:
    """collection 名称 → 缓存文件路径（sha256 前 16 位防路径穿越/特殊字符）."""
    base = Path(dirpath) if dirpath is not None else CORPUS_CACHE_DIR
    digest = hashlib.sha256(str(collection).encode("utf-8")).hexdigest()[:16]
    return base / f"{digest}.npz"


def load_corpus_cache(
    collection: str,
    dirpath: str | os.PathLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取缓存；文件缺失/损坏/维度异常返回空数组（不报错）.

    Returns:
        vectors: (N, 64) float32；labels: (N,) int64；point_ids: (N,) str。
        任何失败（缺失/损坏/截断/维度不齐）都返回空数组，由 ensure 触发重建。
    """
    path = _cache_path(collection, dirpath)
    try:
        with np.load(path, allow_pickle=False) as d:
            vectors = np.asarray(d["vectors"], dtype=np.float32)
            labels = np.asarray(d["labels"], dtype=np.int64)
            point_ids = np.asarray(d["point_ids"], dtype=str)
        # 防御：损坏文件可能绕过 np.load 检查但维度/对齐异常 → 返回空触发重建
        if vectors.ndim != 2 or vectors.shape[1] != _VECTOR_DIM:
            return _empty_arrays()
        if vectors.shape[0] != labels.shape[0] or vectors.shape[0] != point_ids.shape[0]:
            return _empty_arrays()
        return vectors, labels, point_ids
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return _empty_arrays()


def save_corpus_cache(
    collection: str,
    vectors: np.ndarray,
    labels: np.ndarray,
    point_ids: np.ndarray,
    dirpath: str | os.PathLike | None = None,
) -> None:
    """原子写：tmp + os.replace（npz 内嵌 collection 名称 + total_points 元数据）."""
    path = _cache_path(collection, dirpath)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                collection=np.asarray(str(collection)),
                total_points=np.asarray(int(vectors.shape[0])),
                vectors=np.asarray(vectors, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.int64),
                point_ids=np.asarray(point_ids),
            )
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def build_corpus_cache(
    manager: QdrantManager,
    dirpath: str | os.PathLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 Qdrant 全量 scroll 向量、标签、point_id，构建并保存缓存.

    分页 scroll（每批 50000）遍历整个 Collection，转 float32 向量
    （适配 KnnEngine 显存驻留），保存到磁盘缓存后返回。

    Returns:
        vectors: (N, 64) float32；labels: (N,) int64；point_ids: (N,) str。
        空 collection 时返回空数组（并保存空缓存）。
    """
    all_vectors: list[np.ndarray] = []
    all_labels: list[int] = []
    all_point_ids: list[str] = []
    offset = None
    while True:
        result = manager.client.scroll(
            collection_name=manager.collection_name,
            scroll_filter=None,
            limit=_SCROLL_BATCH,
            offset=offset,
            with_payload=["label"],
            with_vectors=True,
        )
        # 防御 mock 返回空值或意外类型（单元测试中 MagicMock 可能返回空）
        if not result or not isinstance(result, (tuple, list)) or len(result) < 2:
            break
        records, offset = result[:2]
        if not records:
            break
        for rec in records:
            vec = np.array(rec.vector, dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] != _VECTOR_DIM:
                raise ValueError(
                    f"向量维度异常: 期望 {_VECTOR_DIM} 维, 实际 {getattr(vec, 'shape', '?')} "
                    f"(point_id={rec.id})"
                )
            all_vectors.append(vec)
            p = rec.payload or {}
            all_labels.append(int(p.get("label", -1)))
            all_point_ids.append(str(rec.id))
        if offset is None:
            break
    vectors = (
        np.array(all_vectors, dtype=np.float32).reshape(-1, _VECTOR_DIM)
        if all_vectors
        else np.empty((0, _VECTOR_DIM), dtype=np.float32)
    )
    labels = np.array(all_labels, dtype=np.int64)
    point_ids = np.asarray(all_point_ids, dtype=str)
    save_corpus_cache(manager.collection_name, vectors, labels, point_ids, dirpath)
    return vectors, labels, point_ids


def _cache_valid(
    collection: str,
    total_points: int,
    dirpath: str | os.PathLike | None,
    cached: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    """校验缓存文件元数据 + 数组完整性（指纹对账）.

    文件存在且 npz 内嵌 collection 名称 == 当前 collection、total_points == 当前
    点数，且数组维度/行数对齐 → 有效。缺失/损坏/元数据不符 → False（触发重建）。
    """
    path = _cache_path(collection, dirpath)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            meta_col = str(d["collection"].item())
            meta_n = int(d["total_points"])
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return False
    if meta_col != collection or meta_n != total_points:
        return False
    v = cached[0]
    if v.ndim != 2 or v.shape[1] != _VECTOR_DIM:
        return False
    if v.shape[0] != meta_n or cached[1].shape[0] != meta_n or cached[2].shape[0] != meta_n:
        return False
    return True


def ensure_corpus_cache(
    manager: QdrantManager,
    dirpath: str | os.PathLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """自动对账：缓存指纹与当前 collection 一致则直接加载，否则重新下载构建.

    比较缓存文件的 collection 名称 + total_points 与当前 `manager.collection_info()`；
    不一致 / 文件缺失 / 损坏 → 自动 `build_corpus_cache` 重建并保存；一致 → 直接
    `np.load` 返回缓存，跳过全量下载（避免重复下载 2.6GB）。

    Returns:
        vectors: (N, 64) float32；labels: (N,) int64；point_ids: (N,) str。
    """
    collection = manager.collection_name
    cached = load_corpus_cache(collection, dirpath)
    info = manager.collection_info()
    current_total = int(info.get("total_points", 0) or 0)
    if _cache_valid(collection, current_total, dirpath, cached):
        return cached
    return build_corpus_cache(manager, dirpath)
