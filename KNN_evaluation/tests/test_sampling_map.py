"""Tests for KNN_evaluation.sampling_map（Section 10 采样地图 manifest）.

地图构建 / 自动对账 / 原子写：验证 scroll 只取 point_id+label（with_vectors=False）、
分页累积、指纹变化（total_points / collection 名称）触发重建、缺失/损坏可重建。
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from KNN_evaluation.sampling_map import (
    load_sampling_map,
    save_sampling_map,
    build_sampling_map,
    ensure_sampling_map,
)
from KNN_evaluation.qdrant_client import QdrantManager


def _empty_map() -> dict:
    return {"collection": "", "total_points": 0, "updated_at": "", "by_label": {}}


def _mock_scroll_record(label: int, idx: int):
    """构造只有 label payload（无向量访问）的 mock scroll record."""
    rec = MagicMock()
    rec.id = f"pt-{label}-{idx}"
    rec.payload = {"label": label}
    rec.vector = None  # with_vectors=False 时不应访问
    return rec


class TestLoadSaveSamplingMap:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        data = load_sampling_map(tmp_path / "nope.json")
        assert data == _empty_map()

    def test_corrupt_file_returns_empty(self, tmp_path: Path):
        p = tmp_path / "m.json"
        p.write_text("{ not json !!", encoding="utf-8")
        assert load_sampling_map(p) == _empty_map()

    def test_roundtrip(self, tmp_path: Path):
        p = tmp_path / "m.json"
        data = {
            "collection": "c",
            "total_points": 3,
            "updated_at": "2026-08-03T00:00:00",
            "by_label": {0: ["pt-a", "pt-b"], 1: ["pt-c"]},
        }
        save_sampling_map(data, p)
        assert load_sampling_map(p) == data

    def test_int_keys_roundtrip_from_json_strings(self, tmp_path: Path):
        """JSON 存盘后 key 变字符串，load 应还原为 int."""
        p = tmp_path / "m.json"
        save_sampling_map(_empty_map(), p)
        p.write_text(json.dumps({
            "collection": "c", "total_points": 1, "updated_at": "x",
            "by_label": {"0": ["a"]},
        }), encoding="utf-8")
        assert load_sampling_map(p)["by_label"] == {0: ["a"]}

    def test_atomic_no_tmp_leftover(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_sampling_map(_empty_map(), p)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_parent_dir_auto_created(self, tmp_path: Path):
        p = tmp_path / "nested" / "sub" / "m.json"
        save_sampling_map(_empty_map(), p)
        assert p.exists()
        assert load_sampling_map(p) == _empty_map()


class TestBuildSamplingMap:
    def test_scroll_without_vectors_stores_only_id_and_label(self):
        """scroll 应 with_vectors=False、with_payload=['label']、无 filter，按 label 分组只存 point_id."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        records = []
        for label in (0, 1):
            for i in range(3):
                records.append(_mock_scroll_record(label, i))
        manager.client.scroll.return_value = (records, None)

        result = build_sampling_map(manager)

        assert result["collection"] == "test_collection"
        assert result["total_points"] == 6
        assert result["updated_at"]  # 非空时间戳
        assert result["by_label"] == {
            0: ["pt-0-0", "pt-0-1", "pt-0-2"],
            1: ["pt-1-0", "pt-1-1", "pt-1-2"],
        }
        calls = manager.client.scroll.call_args_list
        assert len(calls) == 1
        kwargs = calls[0].kwargs
        assert kwargs.get("with_vectors") is False, "不下载向量"
        assert kwargs.get("with_payload") == ["label"]
        assert kwargs.get("scroll_filter") is None, "无 label filter（一次全量）"

    def test_pagination_accumulates_all_pages(self):
        """多页 scroll（offset 续传）累积全部 point_id."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        page1 = [_mock_scroll_record(0, 0), _mock_scroll_record(0, 1)]
        page2 = [_mock_scroll_record(1, 0)]
        manager.client.scroll.side_effect = [
            (page1, "off-1"),
            (page2, None),
        ]
        result = build_sampling_map(manager)
        assert result["total_points"] == 3
        assert result["by_label"] == {0: ["pt-0-0", "pt-0-1"], 1: ["pt-1-0"]}
        assert len(manager.client.scroll.call_args_list) == 2

    def test_limit_is_50000(self):
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.scroll.return_value = ([], None)
        build_sampling_map(manager)
        assert manager.client.scroll.call_args.kwargs["limit"] == 50000

    def test_empty_scroll_returns_zero_map(self):
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.scroll.return_value = ([], None)
        result = build_sampling_map(manager)
        assert result["total_points"] == 0
        assert result["by_label"] == {}


class TestEnsureSamplingMap:
    def _manager(self, total_points: int, collection: str = "test_collection"):
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = collection
        manager.collection_info.return_value = {"total_points": total_points}
        return manager

    def test_consistent_cache_returned_without_build(self, monkeypatch, tmp_path: Path):
        """指纹一致（名称 + total_points）→ 直接返回缓存，不触发 build/写盘."""
        manager = self._manager(total_points=5)
        cached = {"collection": "test_collection", "total_points": 5,
                  "updated_at": "x", "by_label": {0: ["a"]}}
        p = tmp_path / "m.json"
        save_sampling_map(cached, p)
        built: list = []
        monkeypatch.setattr(
            "KNN_evaluation.sampling_map.build_sampling_map",
            lambda mgr: built.append(1) or {},
        )
        result = ensure_sampling_map(manager, path=p)
        assert result == cached
        assert built == []

    def test_total_points_changed_triggers_rebuild(self, tmp_path: Path):
        """collection 指纹变化（total_points 不同）→ 自动重建并保存."""
        manager = self._manager(total_points=10)
        stale = {"collection": "test_collection", "total_points": 5,
                 "updated_at": "x", "by_label": {0: ["a"]}}
        p = tmp_path / "m.json"
        save_sampling_map(stale, p)
        records = [_mock_scroll_record(0, i) for i in range(10)]
        manager.client.scroll.return_value = (records, None)

        result = ensure_sampling_map(manager, path=p)
        assert result["total_points"] == 10
        assert result["by_label"] == {0: [f"pt-0-{i}" for i in range(10)]}
        assert load_sampling_map(p) == result  # 已保存
        assert manager.client.scroll.called

    def test_collection_name_mismatch_triggers_rebuild(self, tmp_path: Path):
        """collection 名称变化 → 自动重建并保存."""
        manager = self._manager(total_points=5)
        stale = {"collection": "old_collection", "total_points": 5,
                 "updated_at": "x", "by_label": {0: ["a"]}}
        p = tmp_path / "m.json"
        save_sampling_map(stale, p)
        records = [_mock_scroll_record(0, i) for i in range(5)]
        manager.client.scroll.return_value = (records, None)

        result = ensure_sampling_map(manager, path=p)
        assert result["collection"] == "test_collection"
        assert manager.client.scroll.called

    def test_missing_file_rebuilds(self, tmp_path: Path):
        """文件缺失 → load 返回空结构，指纹不匹配 → 自动重建."""
        manager = self._manager(total_points=3)
        records = [_mock_scroll_record(0, i) for i in range(3)]
        manager.client.scroll.return_value = (records, None)
        result = ensure_sampling_map(manager, path=tmp_path / "nope.json")
        assert result["total_points"] == 3

    def test_corrupt_file_triggers_rebuild(self, tmp_path: Path):
        """文件损坏 → load 返回空结构，指纹不匹配 → 自动重建."""
        manager = self._manager(total_points=2)
        p = tmp_path / "m.json"
        p.write_text("{ corrupt !!", encoding="utf-8")
        records = [_mock_scroll_record(0, i) for i in range(2)]
        manager.client.scroll.return_value = (records, None)

        result = ensure_sampling_map(manager, path=p)
        assert result["total_points"] == 2


class TestSamplingMapPathIsolation:
    """Task: 采样地图按 collection 隔离路径 + ensure_sampling_map 缺省派生（D4）."""

    def _manager(self, total_points: int, collection: str):
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = collection
        manager.collection_info.return_value = {"total_points": total_points}
        manager.client.scroll.return_value = (
            [_mock_scroll_record(0, i) for i in range(total_points)], None,
        )
        return manager

    def test_sampling_map_path_naming(self):
        from KNN_evaluation.sampling_map import sampling_map_path
        assert sampling_map_path("google_aef_embedding") == "qdrant_sampling_map_google_aef_embedding.json"

    def test_sampling_map_path_sanitizes(self):
        from KNN_evaluation.sampling_map import sampling_map_path
        assert sampling_map_path("a/b\\c") == "qdrant_sampling_map_a_b_c.json"

    def test_ensure_sampling_map_derives_path_from_collection(self, tmp_path, monkeypatch):
        """path 缺省时按 manager.collection_name 派生路径，两个 collection 互不覆盖."""
        monkeypatch.chdir(tmp_path)
        m1 = self._manager(total_points=3, collection="google_aef_embedding")
        m2 = self._manager(total_points=2, collection="xian_aef_embedding")
        ensure_sampling_map(m1)
        ensure_sampling_map(m2)
        p1 = tmp_path / "qdrant_sampling_map_google_aef_embedding.json"
        p2 = tmp_path / "qdrant_sampling_map_xian_aef_embedding.json"
        assert p1.exists() and p2.exists()
        assert load_sampling_map(p1)["collection"] == "google_aef_embedding"
        assert load_sampling_map(p2)["collection"] == "xian_aef_embedding"
