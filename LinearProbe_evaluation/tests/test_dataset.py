"""Tests for dataset loading & stratified sampling."""
import threading

import numpy as np
import pytest

from LinearProbe_evaluation import dataset as ds_mod
from LinearProbe_evaluation.dataset import (
    PixelDataset, empty_dataset, stratified_train_val_split, sample_dataset,
    load_full_dataset, CancelledError,
)
from LinearProbe_evaluation.config import VECTOR_SIZE, NUM_CLASSES
from LinearProbe_evaluation.tests.helpers import FakeQdrantManager, make_points


def _sampling_map_by_label(points: list[dict]) -> dict:
    """由测试点集构建 by_label 地图（与 KNN sampling_map 结构一致）."""
    by_label: dict[int, list[str]] = {}
    for p in points:
        by_label.setdefault(p["label"], []).append(p["id"])
    return by_label


def _patch_map(monkeypatch, points: list[dict]):
    """把 ensure_sampling_map 替换为内存地图（绕过真实 480MB JSON 文件）."""
    by_label = _sampling_map_by_label(points)
    monkeypatch.setattr(
        "KNN_evaluation.sampling_map.ensure_sampling_map",
        lambda mgr: {"collection": mgr.collection_name, "by_label": by_label},
    )


class TestStratifiedTrainValSplit:
    def test_balanced_caps(self, monkeypatch):
        """每类 20 点、train_per_class=10 / val_per_class=5：train 每类 10、val 每类 5."""
        points = make_points(n_per_class=20, seed=1)
        _patch_map(monkeypatch, points)
        mgr = FakeQdrantManager(points=points)
        train_ds, val_ds = stratified_train_val_split(
            mgr, train_per_class=10, val_per_class=5, val_ratio=0.5, seed=42,
        )
        assert train_ds.size == NUM_CLASSES * 10
        assert val_ds.size == NUM_CLASSES * 5
        for lid in range(NUM_CLASSES):
            assert int((train_ds.y == lid).sum()) == 10
            assert int((val_ds.y == lid).sum()) == 5

    def test_rare_class_keeps_val(self, monkeypatch):
        """稀有类（2 个样本）即使少于训练上限也有验证集，且 train/val 不重叠."""
        points = make_points(n_per_class=10, seed=2)
        # 只保留 2 个 snow_and_ice(label 8) 样本模拟稀有类
        points = [p for p in points if p["label"] != 8] + \
                 [p for p in points if p["label"] == 8][:2]
        _patch_map(monkeypatch, points)
        mgr = FakeQdrantManager(points=points)
        train_ds, val_ds = stratified_train_val_split(
            mgr, train_per_class=10, val_per_class=5, val_ratio=0.2, seed=42,
        )
        # 稀有类 label 8：val_ratio 0.2 → val 至少 1 个（不足 2 时）
        assert int((val_ds.y == 8).sum()) >= 1
        assert int((train_ds.y == 8).sum()) >= 1
        # 不重叠：train 与 val 的 point_id 无交集
        overlap = set(train_ds.point_ids) & set(val_ds.point_ids)
        assert len(overlap) == 0

    def test_reproducible_with_seed(self, monkeypatch):
        """相同 seed 产生相同划分（同 ID 集合）。"""
        points = make_points(n_per_class=15, seed=3)
        _patch_map(monkeypatch, points)
        mgr = FakeQdrantManager(points=points)
        t1, v1 = stratified_train_val_split(mgr, seed=7)
        t2, v2 = stratified_train_val_split(mgr, seed=7)
        assert set(t1.point_ids) == set(t2.point_ids)
        assert set(v1.point_ids) == set(v2.point_ids)

    def test_unlimited_per_class(self, monkeypatch):
        """train_per_class=0 → 该类剩余全部进入训练集（不截断）。"""
        points = make_points(n_per_class=6, seed=4)
        _patch_map(monkeypatch, points)
        mgr = FakeQdrantManager(points=points)
        train_ds, val_ds = stratified_train_val_split(
            mgr, train_per_class=0, val_per_class=0, val_ratio=0.5, seed=42,
        )
        assert train_ds.size == NUM_CLASSES * 3
        assert val_ds.size == NUM_CLASSES * 3

    def test_cancel_event(self, monkeypatch):
        """取消事件置位 → 抛出 CancelledError."""
        points = make_points(n_per_class=10, seed=5)
        _patch_map(monkeypatch, points)
        mgr = FakeQdrantManager(points=points)
        ev = threading.Event()
        ev.set()
        with pytest.raises(CancelledError):
            stratified_train_val_split(mgr, cancel_event=ev)

    def test_qdrant_unreachable(self, monkeypatch):
        _patch_map(monkeypatch, make_points(n_per_class=5))
        mgr = FakeQdrantManager(points=[], healthy=False)
        with pytest.raises(ConnectionError):
            stratified_train_val_split(mgr)

    def test_empty_collection_returns_empty(self, monkeypatch):
        _patch_map(monkeypatch, [])
        mgr = FakeQdrantManager(points=[], exists=True)
        train_ds, val_ds = stratified_train_val_split(mgr)
        assert train_ds.size == 0 and val_ds.size == 0


class TestSampleDataset:
    def test_per_class_cap(self, monkeypatch):
        """每类最多 samples_per_class 个；超过则截断."""
        points = make_points(n_per_class=20, seed=10)
        _patch_map(monkeypatch, points)
        mgr = FakeQdrantManager(points=points)
        ds = sample_dataset(mgr, samples_per_class=5, seed=42)
        assert ds.size == NUM_CLASSES * 5
        for lid in range(NUM_CLASSES):
            assert int((ds.y == lid).sum()) == 5

    def test_zero_means_unlimited(self, monkeypatch):
        """samples_per_class<=0 → 该类全取."""
        points = make_points(n_per_class=6, seed=11)
        _patch_map(monkeypatch, points)
        mgr = FakeQdrantManager(points=points)
        ds = sample_dataset(mgr, samples_per_class=0, seed=42)
        assert ds.size == len(points)

    def test_reproducible(self, monkeypatch):
        points = make_points(n_per_class=10, seed=12)
        _patch_map(monkeypatch, points)
        mgr = FakeQdrantManager(points=points)
        ids1 = set(sample_dataset(mgr, samples_per_class=4, seed=7).point_ids)
        ids2 = set(sample_dataset(mgr, samples_per_class=4, seed=7).point_ids)
        assert ids1 == ids2


class TestLoadFullDataset:
    def test_full_read(self):
        points = make_points(n_per_class=4, seed=6)
        mgr = FakeQdrantManager(points=points)
        ds = load_full_dataset(mgr)
        assert ds.size == len(points)
        assert ds.X.shape == (len(points), VECTOR_SIZE)
        assert ds.X.dtype == np.float32
        assert ds.y.dtype == np.int64
        # 标签与构造一致
        for p in points:
            idx = np.where(ds.point_ids == p["id"])[0][0]
            assert ds.y[idx] == p["label"]
            assert np.allclose(ds.X[idx], p["vector"])

    def test_max_points(self):
        points = make_points(n_per_class=10, seed=7)
        mgr = FakeQdrantManager(points=points)
        ds = load_full_dataset(mgr, max_points=25)
        assert ds.size == 25

    def test_cancel_event(self):
        points = make_points(n_per_class=10, seed=8)
        mgr = FakeQdrantManager(points=points)
        ev = threading.Event()
        ev.set()
        with pytest.raises(CancelledError):
            load_full_dataset(mgr, cancel_event=ev)

    def test_progress_callback(self):
        points = make_points(n_per_class=20, seed=9)
        mgr = FakeQdrantManager(points=points)
        calls: list[tuple[int, int]] = []
        ds = load_full_dataset(mgr, batch_size=30, progress_callback=lambda d, t: calls.append((d, t)))
        assert calls, "应至少回调一次"
        assert calls[-1][0] == len(points)


class TestPixelDataset:
    def test_empty_dataset(self):
        ds = empty_dataset()
        assert ds.size == 0
        assert ds.X.shape == (0, VECTOR_SIZE)
        assert ds.class_counts == {lid: 0 for lid in range(NUM_CLASSES)}

    def test_class_counts(self):
        ds = PixelDataset(
            X=np.zeros((4, VECTOR_SIZE), dtype=np.float32),
            y=np.array([0, 0, 1, 1], dtype=np.int64),
            point_ids=np.array(["a", "b", "c", "d"]),
        )
        assert ds.class_counts == {0: 2, 1: 2, **{l: 0 for l in range(2, NUM_CLASSES)}}
