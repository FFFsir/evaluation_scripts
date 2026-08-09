"""Tests for KNN_evaluation.corpus_cache（Section 11 全库向量磁盘缓存）.

缓存构建 / 自动对账 / 原子写：首次下载全量后写盘（mock scroll 计数 1）、二次调用
读缓存（mock scroll 计数 0）、collection 指纹变化（total_points / 名称）触发重建、
缺失/损坏可重建、npz 读写正确（向量 float32 / 标签 int64 / point_id str roundtrip）。
"""
import os
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from KNN_evaluation.corpus_cache import (
    _cache_path,
    load_corpus_cache,
    save_corpus_cache,
    build_corpus_cache,
    ensure_corpus_cache,
)
from KNN_evaluation.qdrant_client import QdrantManager


def _mock_scroll_record(idx: int, label: int = 0):
    """构造带 label payload + 64 维向量的 mock scroll record."""
    rec = MagicMock()
    rec.id = f"pt-{idx}"
    rec.vector = [float(idx)] * 64
    rec.payload = {"label": label}
    return rec


def _manager(total_points: int, collection: str = "test_collection"):
    """构造带 collection_info 指纹的 mock manager."""
    manager = MagicMock(spec=QdrantManager)
    manager.collection_name = collection
    manager.collection_info.return_value = {"total_points": total_points}
    return manager


def _sample_data(n: int):
    """生成 n 行 (N,64) float32 向量 + int64 标签 + str point_id."""
    vectors = np.arange(n * 64, dtype=np.float32).reshape(n, 64)
    labels = np.arange(n, dtype=np.int64)
    point_ids = np.asarray([f"pt-{i}" for i in range(n)])
    return vectors, labels, point_ids


class TestLoadSaveCorpusCache:
    def test_save_load_roundtrip(self, tmp_path: Path):
        """npz 读写正确：向量/labels/point_ids 与 dtype 完全还原."""
        v, l, ids = _sample_data(3)
        save_corpus_cache("test_collection", v, l, ids, tmp_path)
        v2, l2, ids2 = load_corpus_cache("test_collection", tmp_path)
        assert np.array_equal(v, v2)
        assert np.array_equal(l, l2)
        assert list(ids) == list(ids2)
        assert v2.dtype == np.float32
        assert l2.dtype == np.int64

    def test_metadata_fingerprint_stored(self, tmp_path: Path):
        """缓存文件内嵌 collection 名称 + total_points 元数据（指纹对账用）."""
        v, l, ids = _sample_data(3)
        save_corpus_cache("test_collection", v, l, ids, tmp_path)
        path = _cache_path("test_collection", tmp_path)
        with np.load(path, allow_pickle=False) as d:
            assert d["collection"].item() == "test_collection"
            assert int(d["total_points"]) == 3

    def test_missing_file_returns_empty(self, tmp_path: Path):
        """文件缺失 → load 返回空数组（不报错）."""
        v, l, ids = load_corpus_cache("nope_collection", tmp_path)
        assert v.shape == (0, 64)
        assert v.dtype == np.float32
        assert l.shape == (0,)
        assert l.dtype == np.int64
        assert ids.shape == (0,)

    def test_corrupt_file_returns_empty(self, tmp_path: Path):
        """文件损坏 → load 返回空数组（不报错，由 ensure 重建）."""
        path = _cache_path("test_collection", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"this is not a valid npz !!!")
        v, l, ids = load_corpus_cache("test_collection", tmp_path)
        assert v.shape == (0, 64)
        assert ids.shape == (0,)

    def test_truncated_npz_returns_empty(self, tmp_path: Path):
        """文件被截断（半写）→ load 返回空数组（重建路径兜底）."""
        v, l, ids = _sample_data(2)
        save_corpus_cache("test_collection", v, l, ids, tmp_path)
        path = _cache_path("test_collection", tmp_path)
        data = path.read_bytes()[: len(path.read_bytes()) // 2]  # 截断一半
        path.write_bytes(data)
        v2, l2, ids2 = load_corpus_cache("test_collection", tmp_path)
        assert v2.shape == (0, 64)

    def test_atomic_no_tmp_leftover(self, tmp_path: Path):
        """原子写：无 .tmp 残留."""
        v, l, ids = _sample_data(3)
        save_corpus_cache("test_collection", v, l, ids, tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_parent_dir_auto_created(self, tmp_path: Path):
        """父目录自动创建."""
        sub = tmp_path / "nested" / "dir"
        v, l, ids = _sample_data(2)
        save_corpus_cache("test_collection", v, l, ids, sub)
        assert _cache_path("test_collection", sub).exists()

    def test_filename_safe_hash(self, tmp_path: Path):
        """collection 名称 hash 为文件名，防路径穿越/特殊字符."""
        evil = "../../etc/passwd & special chars"
        path = _cache_path(evil, tmp_path)
        path = os.path.abspath(path)
        assert path.startswith(os.path.abspath(tmp_path) + os.sep), "文件名不得逃逸缓存目录"
        assert ".." not in path.replace(os.sep + "..", "")  # 无目录穿越段
        assert path.endswith(".npz")


class TestBuildCorpusCache:
    def test_scroll_with_vectors_stores_arrays(self, tmp_path: Path):
        """scroll 应 with_vectors=True、with_payload=['label']、limit=50000、无 filter."""
        manager = _manager(total_points=3)
        manager.client.scroll.side_effect = [
            ([_mock_scroll_record(i, label=i) for i in range(3)], None),
        ]
        v, l, ids = build_corpus_cache(manager, tmp_path)
        assert v.shape == (3, 64)
        assert v.dtype == np.float32
        assert list(l) == [0, 1, 2]
        assert list(ids) == ["pt-0", "pt-1", "pt-2"]
        # 已写盘
        assert _cache_path("test_collection", tmp_path).exists()
        kwargs = manager.client.scroll.call_args.kwargs
        assert kwargs.get("with_vectors") is True, "必须下载向量"
        assert kwargs.get("with_payload") == ["label"]
        assert kwargs.get("scroll_filter") is None, "无 filter（一次全量）"
        assert kwargs.get("limit") == 50000

    def test_pagination_accumulates_all_pages(self, tmp_path: Path):
        """多页 scroll（offset 续传）累积全部记录."""
        manager = _manager(total_points=4)
        manager.client.scroll.side_effect = [
            ([_mock_scroll_record(0), _mock_scroll_record(1)], "off-1"),
            ([_mock_scroll_record(2), _mock_scroll_record(3)], None),
        ]
        v, l, ids = build_corpus_cache(manager, tmp_path)
        assert v.shape == (4, 64)
        assert list(ids) == ["pt-0", "pt-1", "pt-2", "pt-3"]
        assert len(manager.client.scroll.call_args_list) == 2

    def test_empty_collection_returns_empty_arrays(self, tmp_path: Path):
        """空 collection → 返回空数组."""
        manager = _manager(total_points=0)
        manager.client.scroll.return_value = ([], None)
        v, l, ids = build_corpus_cache(manager, tmp_path)
        assert v.shape == (0, 64)
        assert v.dtype == np.float32
        assert ids.shape == (0,)

    def test_wrong_dimension_raises(self, tmp_path: Path):
        """非 64 维向量应抛 ValueError（防静默 reshape 拼接）."""
        rec = MagicMock()
        rec.vector = [1.0] * 128  # 维度错误（128 维，reshape(-1,64) 会静默拼成 2 行）
        rec.payload = {"label": 0}
        rec.id = "pt-bad"
        manager = _manager(total_points=1)
        manager.client.scroll.side_effect = [([rec], None)]
        with pytest.raises(ValueError, match="期望 64 维"):
            build_corpus_cache(manager, tmp_path)


class TestEnsureCorpusCache:
    def test_consistent_cache_returned_without_build(self, tmp_path: Path, monkeypatch):
        """指纹一致（名称 + total_points）→ 直接返回缓存，不触发 build/写盘."""
        import KNN_evaluation.corpus_cache as CC
        v, l, ids = _sample_data(3)
        save_corpus_cache("test_collection", v, l, ids, tmp_path)
        manager = _manager(total_points=3)
        built: list = []
        monkeypatch.setattr(
            CC, "build_corpus_cache",
            lambda mgr, dirpath=None: built.append(1) or (v, l, ids),
        )
        out = ensure_corpus_cache(manager, tmp_path)
        assert np.array_equal(out[0], v)
        assert np.array_equal(out[1], l)
        assert list(out[2]) == list(ids)
        assert built == []

    def test_missing_file_builds(self, tmp_path: Path):
        """文件缺失 → 自动下载构建并保存（mock scroll 计数 1）."""
        manager = _manager(total_points=3)
        manager.client.scroll.side_effect = [
            ([_mock_scroll_record(i, label=i) for i in range(3)], None),
        ]
        out = ensure_corpus_cache(manager, tmp_path)
        assert out[0].shape == (3, 64)
        assert manager.client.scroll.called
        assert _cache_path("test_collection", tmp_path).exists()

    def test_total_points_changed_triggers_rebuild(self, tmp_path: Path):
        """collection 指纹变化（total_points 不同）→ 重建并保存新指纹."""
        stale_v, stale_l, stale_ids = _sample_data(3)
        save_corpus_cache("test_collection", stale_v, stale_l, stale_ids, tmp_path)
        manager = _manager(total_points=5)
        manager.client.scroll.side_effect = [
            ([_mock_scroll_record(i) for i in range(5)], None),
        ]
        out = ensure_corpus_cache(manager, tmp_path)
        assert out[0].shape == (5, 64)  # 新数据，非陈旧缓存 3 行
        assert manager.client.scroll.called
        with np.load(_cache_path("test_collection", tmp_path), allow_pickle=False) as d:
            assert int(d["total_points"]) == 5

    def test_collection_name_mismatch_triggers_rebuild(self, tmp_path: Path):
        """collection 名称变化 → 重建并保存."""
        stale_v, stale_l, stale_ids = _sample_data(3)
        save_corpus_cache("old_collection", stale_v, stale_l, stale_ids, tmp_path)
        manager = _manager(total_points=3, collection="test_collection")
        manager.client.scroll.side_effect = [
            ([_mock_scroll_record(i) for i in range(3)], None),
        ]
        out = ensure_corpus_cache(manager, tmp_path)
        assert out[0].shape == (3, 64)
        assert manager.client.scroll.called
        assert _cache_path("test_collection", tmp_path).exists()  # 新集合名文件

    def test_corrupt_file_triggers_rebuild(self, tmp_path: Path):
        """文件损坏 → 自动重建."""
        path = _cache_path("test_collection", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"corrupt npz data !!")
        manager = _manager(total_points=2)
        manager.client.scroll.side_effect = [
            ([_mock_scroll_record(i) for i in range(2)], None),
        ]
        out = ensure_corpus_cache(manager, tmp_path)
        assert out[0].shape == (2, 64)
        assert manager.client.scroll.called
