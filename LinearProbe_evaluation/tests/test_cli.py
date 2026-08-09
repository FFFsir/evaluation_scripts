"""Tests for LinearProbe_evaluation CLI collection 选择.

覆盖 `--collection` 参数解析与传入 QdrantManager（未指定时用默认值）、
`stats` 输出展示实际 collection 名称。
"""
from types import SimpleNamespace

import pytest

from LinearProbe_evaluation import cli
from LinearProbe_evaluation.config import DEFAULT_COLLECTION


class TestParserCollectionArg:
    """Task 2.1: 各子命令 --collection 参数解析与默认值."""

    def test_default_collection_for_all_subcommands(self):
        parser = cli._build_parser()
        cases = {
            "train": ["train"],
            "evaluate": ["evaluate", "--checkpoint", "x.pt"],
            "stats": ["stats"],
        }
        for cmd, argv in cases.items():
            args = parser.parse_args(argv)
            assert args.collection == DEFAULT_COLLECTION, cmd
            assert args.collection == "xian_aef_embedding"

    def test_override_collection(self):
        parser = cli._build_parser()
        args = parser.parse_args(["stats", "--collection", "google_aef_embedding"])
        assert args.collection == "google_aef_embedding"


class TestCmdCollectionInjection:
    """Task 2.2/2.3: cmd_* 以 args.collection 构造 QdrantManager 并展示."""

    def test_cmd_stats_passes_collection_to_manager(self, monkeypatch, capsys):
        captured: dict = {}

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=10):
                captured["collection_name"] = collection_name
                self.collection_name = collection_name

            def health_check(self):
                return True

            def collection_exists(self):
                return True

            def collection_info(self):
                return {"total_points": 10, "status": "green"}

            @property
            def client(self):
                class _C:
                    def count(self, collection_name, exact=True, count_filter=None):
                        return SimpleNamespace(count=10)
                return _C()

        monkeypatch.setattr(cli, "QdrantManager", FakeManager)
        args = SimpleNamespace(qdrant_url="http://localhost:1",
                               collection="xian_aef_embedding", json=False)
        assert cli.cmd_stats(args) == 0
        assert captured["collection_name"] == "xian_aef_embedding"
        out = capsys.readouterr().out
        assert "xian_aef_embedding" in out

    def test_cmd_stats_default_collection(self, monkeypatch):
        captured: dict = {}

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=10):
                captured["collection_name"] = collection_name
                self.collection_name = collection_name

            def health_check(self):
                return True

            def collection_exists(self):
                return True

            def collection_info(self):
                return {"total_points": 10, "status": "green"}

            @property
            def client(self):
                class _C:
                    def count(self, collection_name, exact=True, count_filter=None):
                        return SimpleNamespace(count=10)
                return _C()

        monkeypatch.setattr(cli, "QdrantManager", FakeManager)
        args = SimpleNamespace(qdrant_url="http://localhost:1",
                               collection=DEFAULT_COLLECTION, json=True)
        assert cli.cmd_stats(args) == 0
        assert captured["collection_name"] == DEFAULT_COLLECTION

    def test_cmd_train_error_uses_actual_collection(self, monkeypatch, capsys):
        """train 遇到不存在的 collection 时错误提示含实际 collection 名."""

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=10):
                self.collection_name = collection_name

            def health_check(self):
                return True

            def collection_exists(self):
                return False

        monkeypatch.setattr(cli, "QdrantManager", FakeManager)
        args = SimpleNamespace(qdrant_url="http://localhost:1",
                               collection="my_custom_collection")
        assert cli.cmd_train(args) == 1
        err = capsys.readouterr().err
        assert "my_custom_collection" in err
