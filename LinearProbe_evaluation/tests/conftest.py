"""Fixtures for LinearProbe_evaluation tests（内存 Fake Qdrant，无需真实服务）."""
import pytest

from LinearProbe_evaluation.tests.helpers import FakeQdrantManager, make_points


@pytest.fixture
def fake_manager():
    """默认测试 Qdrant：每类 20 个点（共 180 点），高斯向量."""
    return FakeQdrantManager(points=make_points(n_per_class=20, seed=42))
