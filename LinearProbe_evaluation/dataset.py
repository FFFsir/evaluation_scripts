"""数据集加载与分层采样：从 Qdrant 读取像素 embedding + DW 硬标签.

两个数据通路：

- ``stratified_train_val_split``（默认）：复用 KNN 的采样地图
  （``qdrant_sampling_map.json``，point_id→label 地图），本地随机选 ID
  后按 ID 精确 ``retrieve`` 向量，只下载样本量（MB 级），避免全量下载
  2.6GB 向量。适合类别不平衡的 DW 数据（trees 621 万 vs snow_and_ice 240）。
- ``load_full_dataset``：scroll 全量读取（预分配 numpy 数组），用于
  evaluate 全量推理或不受采样式训练。
"""
import random
from dataclasses import dataclass

import numpy as np

from LinearProbe_evaluation.config import (
    SCROLL_BATCH_SIZE, VECTOR_SIZE, DEFAULT_SEED,
    DEFAULT_TRAIN_PER_CLASS, DEFAULT_VAL_PER_CLASS, DEFAULT_VAL_RATIO,
)
from LinearProbe_evaluation.label_mapping import LABEL_NAMES


@dataclass
class PixelDataset:
    """像素数据集：64 维 embedding + 硬分类标签（0-8）.

    Attributes:
        X: (N, 64) float32 embedding 矩阵（1 行 = 1 个像素）.
        y: (N,) int64 标签（0-8，对应 DW 硬分类 label）.
        point_ids: (N,) str 像素 point_id（可能为空数组）.
    """

    X: np.ndarray
    y: np.ndarray
    point_ids: np.ndarray

    @property
    def size(self) -> int:
        return int(self.X.shape[0])

    @property
    def class_counts(self) -> dict[int, int]:
        """每类样本数（{label_id: count}）."""
        return {lid: int((self.y == lid).sum()) for lid in sorted(LABEL_NAMES)}


def empty_dataset() -> PixelDataset:
    return PixelDataset(
        X=np.empty((0, VECTOR_SIZE), dtype=np.float32),
        y=np.empty((0,), dtype=np.int64),
        point_ids=np.empty((0,), dtype=object),
    )


def _retrieve_batch(
    manager,
    ids: list[str],
    batch_size: int = 5000,
    progress_callback=None,
    cancel_event=None,
) -> PixelDataset:
    """按 ID 分批 retrieve 向量与 label，组装为 PixelDataset.

    Args:
        manager: 具备 ``client.retrieve`` 的 QdrantManager（薄封装或 KNN 版均可）.
        ids: 待读取的 point_id 列表（有序）.
        batch_size: 单次 retrieve 的 ID 数.
        progress_callback: ``cb(done, total)`` 进度回调（可空）.
        cancel_event: threading.Event，置位时抛出 ``CancelledError``（可空）.

    Raises:
        CancelledError: 取消事件被置位.
    """
    total = len(ids)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kept_ids: list[str] = []
    done = 0
    for start in range(0, total, batch_size):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("数据集读取已取消")
        chunk = ids[start:start + batch_size]
        records = manager.client.retrieve(
            collection_name=manager.collection_name,
            ids=chunk,
            with_payload=["label"],
            with_vectors=True,
        )
        for rec in records:
            label = (rec.payload or {}).get("label")
            if label is None or rec.vector is None:
                continue
            xs.append(np.asarray(rec.vector, dtype=np.float32))
            ys.append(int(label))
            kept_ids.append(str(rec.id))
        done += len(chunk)
        if progress_callback is not None:
            progress_callback(done, total)
    if not xs:
        return empty_dataset()
    return PixelDataset(
        X=np.stack(xs),
        y=np.asarray(ys, dtype=np.int64),
        point_ids=np.asarray(kept_ids, dtype=str),
    )


def stratified_train_val_split(
    manager,
    *,
    train_per_class: int = DEFAULT_TRAIN_PER_CLASS,
    val_per_class: int = DEFAULT_VAL_PER_CLASS,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    progress_callback=None,
    cancel_event=None,
) -> tuple[PixelDataset, PixelDataset]:
    """按标签分层划分训练/验证集，只从 Qdrant 下载样本量.

    流程（每类独立）：
    1. 读采样地图 ``by_label``（指纹一致走缓存，不一致自动重建）.
    2. 组内用 seed 打乱 ID 列表.
    3. 先按 ``val_ratio`` 切出验证集（再受 ``val_per_class`` 上限约束），
       剩余部分取前 ``train_per_class`` 个作为训练集 —— 稀有类即使少于
       训练上限也能保证验证集非空（如 snow_and_ice 240：val 48 / train 192）.
    4. 按 ID 批量 retrieve 向量（只下载样本量，MB 级）.

    Args:
        manager: QdrantManager（薄封装或 KNN 版均可，需有 ``collection_info`` 与 ``client``）.
        train_per_class: 每类最多进入训练集的样本数；<=0 表示不限制（该类全取）.
        val_per_class: 每类最多进入验证集的样本数；<=0 表示不限制.
        val_ratio: 先按该比例从每类切出验证集（默认 0.2）.
        seed: 随机种子，可复现.
        progress_callback: ``cb(label_id, n_train, n_val)`` 每类划分完成回调
            （label_id + 该类进入训练/验证集的样本数，可空）.
        cancel_event: threading.Event，置位时抛出 ``CancelledError``.

    Returns:
        (train_ds, val_ds) PixelDataset 对。Collection 为空时两者均为空数据集.
    """
    from KNN_evaluation.sampling_map import ensure_sampling_map  # 延迟导入：仅需要时依赖

    if not manager.health_check():
        raise ConnectionError(f"Qdrant 不可达: {manager.url}")
    if not manager.collection_exists():
        raise ValueError(f"Collection '{manager.collection_name}' 不存在，请先执行 KNN import")
    info = manager.collection_info()
    if info.get("total_points", 0) == 0:
        return empty_dataset(), empty_dataset()

    by_label = (ensure_sampling_map(manager).get("by_label") or {})
    rng = random.Random(seed)

    train_ids: list[str] = []
    val_ids: list[str] = []
    for label_id in sorted(LABEL_NAMES):
        ids = list(by_label.get(label_id, []))
        rng.shuffle(ids)
        n_total = len(ids)
        # 先按比例切出验证集；稀有类即使少于训练上限也保证验证集非空
        # （val_ratio>0 且该类有点时至少 1 个样本进验证集）.
        n_val = round(n_total * val_ratio)
        if val_per_class is not None and val_per_class > 0:
            n_val = min(n_val, val_per_class)
        n_val = min(n_val, n_total)
        if val_ratio > 0 and n_total > 0 and n_val < 1:
            n_val = 1
        v_ids = ids[:n_val]
        rest = ids[n_val:]
        if train_per_class is not None and train_per_class > 0:
            rest = rest[:train_per_class]
        train_ids.extend(rest)
        val_ids.extend(v_ids)
        if progress_callback is not None:
            progress_callback(label_id, len(rest), len(v_ids))

    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("数据集划分已取消")

    train_ds = _retrieve_batch(
        manager, train_ids, progress_callback=None, cancel_event=cancel_event,
    )
    val_ds = _retrieve_batch(
        manager, val_ids, progress_callback=None, cancel_event=cancel_event,
    )
    return train_ds, val_ds


def sample_dataset(
    manager,
    *,
    samples_per_class: int,
    seed: int = DEFAULT_SEED,
    cancel_event=None,
) -> PixelDataset:
    """按标签分层独立采样（评估用）：每类最多 ``samples_per_class`` 个样本.

    与 ``stratified_train_val_split`` 的区别：不做 train/val 划分，每类从
    采样地图随机抽取至多 N 个 ID 后批量 retrieve。``samples_per_class <= 0``
    表示不限制（该类全取）。

    Raises:
        ConnectionError: Qdrant 不可达.
        ValueError: Collection 不存在.
        CancelledError: 取消事件被置位.
    """
    from KNN_evaluation.sampling_map import ensure_sampling_map  # 延迟导入

    if not manager.health_check():
        raise ConnectionError(f"Qdrant 不可达: {manager.url}")
    if not manager.collection_exists():
        raise ValueError(f"Collection '{manager.collection_name}' 不存在，请先执行 KNN import")
    info = manager.collection_info()
    if info.get("total_points", 0) == 0:
        return empty_dataset()

    by_label = (ensure_sampling_map(manager).get("by_label") or {})
    rng = random.Random(seed)
    ids: list[str] = []
    for label_id in sorted(LABEL_NAMES):
        class_ids = list(by_label.get(label_id, []))
        n = min(len(class_ids), samples_per_class) if samples_per_class > 0 else len(class_ids)
        ids.extend(rng.sample(class_ids, n))
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("数据集采样已取消")
    return _retrieve_batch(manager, ids, cancel_event=cancel_event)


def load_full_dataset(
    manager,
    *,
    max_points: int | None = None,
    batch_size: int = SCROLL_BATCH_SIZE,
    progress_callback=None,
    cancel_event=None,
) -> PixelDataset:
    """scroll 全量读取像素向量 + label（预分配 numpy，内存友好）.

    Args:
        manager: QdrantManager 实例.
        max_points: 最多读取点数（None = 全量）.
        batch_size: scroll 每批点数.
        progress_callback: ``cb(done, total)`` 进度回调（可空）.
        cancel_event: threading.Event，置位时抛出 ``CancelledError``.

    Returns:
        全量 PixelDataset。Collection 为空时返回空数据集.
    """
    if not manager.health_check():
        raise ConnectionError(f"Qdrant 不可达: {manager.url}")
    if not manager.collection_exists():
        raise ValueError(f"Collection '{manager.collection_name}' 不存在，请先执行 KNN import")

    info = manager.collection_info()
    total = int(info.get("total_points", 0) or 0)
    if total == 0:
        return empty_dataset()
    if max_points is not None and max_points > 0:
        total = min(total, max_points)

    X = np.empty((total, VECTOR_SIZE), dtype=np.float32)
    y = np.empty((total,), dtype=np.int64)
    # 注意：np.empty(dtype=str) 在 numpy 2.x 下是 <U1（单字符），会截断 uuid，
    # 必须用 object 预分配再在收尾时 astype(str) 推断正确宽度。
    point_ids = np.empty((total,), dtype=object)
    filled = 0

    offset = None
    while filled < total:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("数据集读取已取消")
        batch_limit = min(batch_size, total - filled)
        result = manager.client.scroll(
            collection_name=manager.collection_name,
            limit=batch_limit,
            offset=offset,
            with_vectors=True,
            with_payload=["label"],
        )
        records, offset = result[0], result[1]
        if not records:
            break
        for rec in records:
            label = (rec.payload or {}).get("label")
            if label is None or rec.vector is None:
                continue
            if filled >= total:
                break
            X[filled] = np.asarray(rec.vector, dtype=np.float32)
            y[filled] = int(label)
            point_ids[filled] = str(rec.id)
            filled += 1
        if progress_callback is not None:
            progress_callback(filled, total)
        if offset is None:
            break

    X = X[:filled]
    y = y[:filled]
    point_ids = point_ids[:filled].astype(str)
    return PixelDataset(X=X, y=y, point_ids=point_ids)


class CancelledError(Exception):
    """训练/读取被用户主动取消."""
