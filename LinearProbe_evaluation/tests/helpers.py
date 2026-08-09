"""Fixtures for LinearProbe_evaluation tests（内存 Fake Qdrant，无需真实服务）."""
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from LinearProbe_evaluation.dataset import PixelDataset
from LinearProbe_evaluation.config import VECTOR_SIZE, NUM_CLASSES
from LinearProbe_evaluation.label_mapping import LABEL_NAMES


def make_points(n_per_class: int = 20, seed: int = 42) -> list[dict]:
    """生成确定性测试点集：每类 n_per_class 个高斯 64 维向量.

    Returns:
        list[dict]，每项 {"id", "vector"(list[float]), "label"(int)}.
    """
    rng = np.random.default_rng(seed)
    points = []
    for label_id in range(NUM_CLASSES):
        for _ in range(n_per_class):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": rng.normal(0, 1, VECTOR_SIZE).tolist(),
                "label": label_id,
            })
    return points


class _FakeClient:
    """内存版 qdrant-client：retrieve / scroll / count 最小实现."""

    def __init__(self, points: list[dict]):
        self.points = points
        self._by_id = {p["id"]: p for p in points}

    def _record(self, p: dict) -> SimpleNamespace:
        return SimpleNamespace(
            id=p["id"],
            payload={"label": p["label"]},
            vector=p["vector"],
        )

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=True):
        return [self._record(self._by_id[i]) for i in ids if i in self._by_id]

    def scroll(self, collection_name, limit, offset=None, with_vectors=True, with_payload=None):
        start = 0
        if offset is not None:
            ids = [p["id"] for p in self.points]
            start = (ids.index(offset) + 1) if offset in ids else len(self.points)
        chunk = self.points[start:start + limit]
        records = [self._record(p) for p in chunk]
        next_offset = None
        if chunk and start + limit < len(self.points):
            next_offset = chunk[-1]["id"]
        return records, next_offset

    def count(self, collection_name, exact=True, count_filter=None):
        n = len(self.points)
        if count_filter is not None and getattr(count_filter, "must", None):
            for cond in count_filter.must:
                if getattr(cond, "key", None) == "label":
                    value = cond.match.value
                    n = sum(1 for p in self.points if p["label"] == value)
        return SimpleNamespace(count=n)


class FakeQdrantManager:
    """内存版 QdrantManager（与 LinearProbe_evaluation.qdrant_client.QdrantManager 同接口）."""

    def __init__(
        self,
        points: list[dict] | None = None,
        url: str = "http://fake:6333",
        collection_name: str = "pixel_embeddings",
        healthy: bool = True,
        exists: bool = True,
    ):
        self.url = url
        self.collection_name = collection_name
        self.points = points if points is not None else []
        self.client = _FakeClient(self.points)
        self._healthy = healthy
        self._exists = exists

    def health_check(self) -> bool:
        return self._healthy

    def collection_exists(self) -> bool:
        return self._exists

    def collection_info(self) -> dict:
        return {
            "total_points": len(self.points),
            "vectors_count": len(self.points),
            "segments_count": 1,
            "status": "green",
        }


@pytest.fixture
def fake_manager():
    """默认测试 Qdrant：每类 20 个点（共 180 点），高斯向量."""
    return FakeQdrantManager(points=make_points(n_per_class=20, seed=42))


def synthetic_pixel_dataset(
    n_per_class: int = 20,
    seed: int = 42,
    noise: float = 1.0,
) -> PixelDataset:
    """合成 PixelDataset：每类 n_per_class 个样本，向量带类别信号.

    为让训练 loss 能下降，每类向量中心不同：中心 = one-hot(label) * 5 + 高斯噪声.
    """
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label_id in range(NUM_CLASSES):
        center = np.zeros(VECTOR_SIZE, dtype=np.float32)
        center[label_id % VECTOR_SIZE] = 5.0
        for _ in range(n_per_class):
            xs.append(center + rng.normal(0, noise, VECTOR_SIZE).astype(np.float32))
            ys.append(label_id)
    return PixelDataset(
        X=np.stack(xs),
        y=np.asarray(ys, dtype=np.int64),
        point_ids=np.asarray([f"p{i}" for i in range(len(ys))], dtype=str),
    )
