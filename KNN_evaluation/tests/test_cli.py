"""Tests for CLI helpers: volume pre-check & migrate fail-fast.

覆盖 Task 3.3 修复的核心安全逻辑：migrate 删除 Collection 前必须确认
Qdrant 容器挂载持久化 volume（先挂卷再迁移），无卷时 fail-fast 中止。
"""
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from KNN_evaluation import cli


class TestContainerHasVolume:
    """_container_has_volume 判定容器是否挂载 qdrant_data 卷."""

    def test_returns_true_when_volume_mounted(self):
        mounts = json.dumps([
            {"Type": "volume", "Name": "qdrant_data",
             "Destination": "/qdrant/storage"},
        ])
        proc = MagicMock(returncode=0, stdout=mounts)
        with patch.object(subprocess, "run", return_value=proc) as mock_run:
            assert cli._container_has_volume("qdrant") is True
        mock_run.assert_called_once()
        # 校验 inspect 命令确实检查了 qdrant_data 卷挂载点
        args = mock_run.call_args.args[0]
        assert args[0] == "docker"
        assert args[1] == "inspect"

    def test_returns_false_when_no_volume(self):
        # 容器存在但未挂载 qdrant_data 卷（变更前 docker run 无 -v）
        proc = MagicMock(returncode=0, stdout=json.dumps([
            {"Type": "bind", "Source": "/tmp/data", "Destination": "/qdrant/storage"},
        ]))
        with patch.object(subprocess, "run", return_value=proc):
            assert cli._container_has_volume("qdrant") is False

    def test_returns_false_on_empty_mounts(self):
        proc = MagicMock(returncode=0, stdout=json.dumps([]))
        with patch.object(subprocess, "run", return_value=proc):
            assert cli._container_has_volume("qdrant") is False

    def test_returns_false_when_inspect_fails(self):
        proc = MagicMock(returncode=1, stdout="", stderr="No such object")
        with patch.object(subprocess, "run", return_value=proc):
            assert cli._container_has_volume("qdrant") is False

    def test_returns_false_on_subprocess_exception(self):
        with patch.object(subprocess, "run",
                         side_effect=subprocess.SubprocessError("docker missing")):
            assert cli._container_has_volume("qdrant") is False

    def test_returns_false_on_invalid_json(self):
        proc = MagicMock(returncode=0, stdout="not-json{{{")
        with patch.object(subprocess, "run", return_value=proc):
            assert cli._container_has_volume("qdrant") is False


class TestCmdMigrateFailFast:
    """cmd_migrate 在无 volume 保障时中止，绝不执行删除."""

    def _args(self):
        return SimpleNamespace(
            qdrant_url="http://localhost:6333",
            collection="google_aef_embedding",
            storage="disk", dir="data_demo", no_resume=False,
        )

    def test_aborts_without_deleting_when_volume_missing(self):
        manager = MagicMock()
        manager.health_check.return_value = True  # Qdrant 已可达
        with patch("KNN_evaluation.cli.QdrantManager", return_value=manager), \
             patch("KNN_evaluation.cli._container_has_volume", return_value=False):
            code = cli.cmd_migrate(self._args())
        assert code == 1
        # 无 volume 时绝不执行删除/重建
        manager.client.delete_collection.assert_not_called()
        manager.create_collection.assert_not_called()

    def test_returns_1_when_qdrant_unreachable_after_start(self):
        manager = MagicMock()
        manager.health_check.return_value = False
        with patch("KNN_evaluation.cli.QdrantManager", return_value=manager), \
             patch("KNN_evaluation.cli._start_qdrant", return_value=True), \
             patch("KNN_evaluation.cli._wait_for_qdrant", return_value=False):
            code = cli.cmd_migrate(self._args())
        assert code == 1
        manager.client.delete_collection.assert_not_called()


def _eval_args(**overrides):
    """构造 evaluate 子命令的 SimpleNamespace 参数（默认 device=cpu 走 CPU 分块路径）."""
    base = dict(
        device="cpu", gpu_batch_q=None, max_gpu_mem=16, max_eval_ram=6.0,
        qdrant_url="http://localhost:6333", collection="test_collection",
        samples_per_class=10, seed=42,
        k_f1=5, k_values="5,10", ann=False, output=None, plot=False,
        plot_dir="./eval_plots",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _eval_queries():
    return [{"point_id": "q0", "label": 0, "label_name": "water ˮ",
             "vector": np.ones(64, dtype=np.float32)}]


def _eval_f1(acc_by_k=None):
    from KNN_evaluation.label_mapping import LABEL_NAMES
    from KNN_evaluation.metrics import _per_class_metrics_from_confusion
    conf = np.zeros((9, 9), dtype=np.int64)
    conf[0][0] = 1
    return {
        "overall_accuracy": 1.0,
        "per_class_metrics": _per_class_metrics_from_confusion(conf, LABEL_NAMES),
        "confusion_matrix": conf,
        "k": 5,
        "num_queries": 1,
        "elapsed_sec": 0.1,
        "accuracy_by_k": acc_by_k if acc_by_k is not None else {5: 1.0},
    }


def _eval_f2():
    return {
        "k_values": [5, 10],
        "global_purity": [1.0, 0.5],
        "global_recall": [0.01, 0.02],
        "per_class_purity": {}, "per_class_recall": {},
        "num_queries": 1, "elapsed_sec": 0.1,
    }


class TestEvaluateArgParse:
    """Task 5: evaluate 子命令参数迁移 --batch → --device/--gpu-batch-q/--max-gpu-mem/--max-eval-ram."""

    def test_evaluate_parser_drops_batch_adds_device_args(self):
        with patch.object(sys, "argv", ["knn-eval", "evaluate"]):
            parser = cli._build_parser()
            args = parser.parse_args(["evaluate"])
        assert args.device == "auto"
        assert args.gpu_batch_q is None
        assert args.max_gpu_mem == 16
        assert args.max_eval_ram == 6.0
        assert not hasattr(args, "batch")
        # 非法 device 值应被 argparse choices 拒绝
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate", "--device", "banana"])

    def test_k_f1_range_validated_up_to_1000(self):
        """Section 9.1: --k-f1 支持到 1000（与 top-1001 预算一致）；越界被拒绝."""
        with patch.object(sys, "argv", ["knn-eval", "evaluate"]):
            parser = cli._build_parser()
            assert parser.parse_args(["evaluate", "--k-f1", "1000"]).k_f1 == 1000
            assert parser.parse_args(["evaluate", "--k-f1", "1"]).k_f1 == 1
        # 1001 与 0 应被 argparse 拒绝
        for bad in ("1001", "0", "-1"):
            with pytest.raises(SystemExit):
                parser.parse_args(["evaluate", "--k-f1", bad])


class TestEvaluateDeviceResolution:
    """Task 5: cmd_evaluate 设备解析 — auto 回退 / cuda 不可用抛错 / 显式 cpu."""

    def _manager(self, total_points=1000):
        manager = MagicMock()
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True
        manager.collection_info.return_value = {"total_points": total_points}
        manager.collection_name = "test_collection"
        return manager

    def _patch_core(self, manager, f1, f2):
        """cmd_evaluate 内部局部 import，须在源模块打补丁（Section 8: 单次 evaluate_knn）."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("KNN_evaluation.cli.QdrantManager",
                                  return_value=manager))
        stack.enter_context(patch("KNN_evaluation.metrics.sample_queries_by_label",
                                  return_value=_eval_queries()))
        stack.enter_context(patch("KNN_evaluation.metrics.evaluate_knn",
                                  return_value={"f1": f1, "f2": f2}))
        return stack

    def test_auto_device_falls_back_to_cpu_when_cuda_unavailable(self, capsys):
        manager = self._manager()
        args = _eval_args(device="auto")
        # resolve_device("auto") 模拟 CUDA 不可用回退为 cpu（CLI 层仅解析并打印）
        with self._patch_core(manager, _eval_f1(), _eval_f2()), \
             patch("KNN_evaluation.gpu_knn.resolve_device",
                   return_value="cpu") as mock_resolve:
            code = cli.cmd_evaluate(args)
        assert code == 0
        mock_resolve.assert_called_once_with("auto")
        out = capsys.readouterr().out
        assert "设备: cpu" in out
        assert "(请求: auto)" in out

    def test_cuda_unavailable_returns_1_and_prints_error(self):
        manager = self._manager()
        args = _eval_args(device="cuda")
        # 采样在设备解析之前执行（cmd_evaluate 顺序），本测试聚焦设备解析，
        # 因此 mock 采样返回固定查询，避免真实采样地图路径的副作用。
        with patch("KNN_evaluation.cli.QdrantManager", return_value=manager), \
             patch("KNN_evaluation.metrics.sample_queries_by_label",
                   return_value=_eval_queries()), \
             patch("KNN_evaluation.gpu_knn.resolve_device",
                   side_effect=RuntimeError("device='cuda' 但当前环境 CUDA 不可用")):
            code = cli.cmd_evaluate(args)
        assert code == 1


class TestEvaluateGpuPathInfoAndF1Table:
    """Task 5: cmd_evaluate 打印 GPU 路径信息 + F1 多 K 表 + JSON config 新字段."""

    def _manager(self):
        manager = MagicMock()
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True
        manager.collection_info.return_value = {"total_points": 1000}
        manager.collection_name = "test_collection"
        return manager

    def _patch_core(self, manager, f1, f2, **extra):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("KNN_evaluation.cli.QdrantManager",
                                  return_value=manager))
        stack.enter_context(patch("KNN_evaluation.metrics.sample_queries_by_label",
                                  return_value=_eval_queries()))
        stack.enter_context(patch("KNN_evaluation.metrics.evaluate_knn",
                                  return_value={"f1": f1, "f2": f2}))
        for name, val in extra.items():
            stack.enter_context(patch(name, val))
        return stack

    def test_prints_gpu_path_info_and_f1_multi_k_table(self, capsys):
        manager = self._manager()
        args = _eval_args()
        with self._patch_core(manager, _eval_f1({5: 1.0, 10: 0.5}), _eval_f2()):
            code = cli.cmd_evaluate(args)
        assert code == 0
        out = capsys.readouterr().out
        # GPU 路径信息
        assert "设备: cpu" in out
        assert "corpus: 1,000" in out
        assert "分块大小:" in out
        assert "预算:" in out
        # F1 多 K 表
        assert "Overall Accuracy" in out
        assert "Accuracy by K:" in out
        assert "K=5" in out and "K=10" in out
        # F2 表
        assert "Purity" in out

    def test_json_export_config_has_device_fields(self, tmp_path):
        manager = self._manager()
        out_path = tmp_path / "eval.json"
        args = _eval_args(output=str(out_path))
        with self._patch_core(manager, _eval_f1(), _eval_f2()):
            code = cli.cmd_evaluate(args)
        assert code == 0
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["config"]["device"] == "cpu"
        assert data["config"]["gpu_batch_q"] is None
        assert data["config"]["max_gpu_mem"] == 16
        assert data["config"]["max_eval_ram"] == 6.0
        # JSON 序列化把 int key 转字符串
        assert data["f1"]["accuracy_by_k"] == {"5": 1.0}
        assert data["f1"]["overall_accuracy"] == 1.0

    def test_calls_metrics_without_use_batch_and_with_device(self):
        manager = self._manager()
        args = _eval_args()
        with patch("KNN_evaluation.cli.QdrantManager", return_value=manager), \
             patch("KNN_evaluation.metrics.sample_queries_by_label",
                   return_value=_eval_queries()), \
             patch("KNN_evaluation.metrics.evaluate_knn") as me:
            me.return_value = {"f1": _eval_f1({5: 1.0, 10: 0.5}), "f2": _eval_f2()}
            code = cli.cmd_evaluate(args)
        assert code == 0
        # Section 8: 单次 evaluate_knn 调用（不再顺序调 compute_knn_accuracy + compute_purity_recall_curve）
        assert me.call_count == 1
        assert "use_batch" not in me.call_args.kwargs
        assert me.call_args.kwargs["device"] == "cpu"
        assert me.call_args.kwargs["gpu_batch_q"] is None
        assert me.call_args.kwargs["max_gpu_mem"] == 16
        assert me.call_args.kwargs["max_eval_ram"] == 6.0
        # F1 多 K 透传（accuracy_by_k 输出所需）
        assert me.call_args.kwargs["k_f1"] == 5
        assert me.call_args.kwargs["k_values"] == [5, 10]

    def test_auto_fallback_passes_resolved_device_to_metrics(self):
        """auto 回退为 cpu 时，metrics 收到 resolved_device='cpu'（而非原始 'auto'）.

        修复 [Important]：原始 'auto' 会让 _device_budget 的 device=='cpu' 判断为 False，
        误用 max_gpu_mem 推导 block_q，导致 CLI 打印预算(6.0GB) 与实际执行(16GB) 不一致。
        """
        manager = self._manager()
        args = _eval_args(device="auto")
        with patch("KNN_evaluation.cli.QdrantManager", return_value=manager), \
             patch("KNN_evaluation.metrics.sample_queries_by_label",
                   return_value=_eval_queries()), \
             patch("KNN_evaluation.gpu_knn.resolve_device", return_value="cpu"), \
             patch("KNN_evaluation.metrics.evaluate_knn") as me:
            me.return_value = {"f1": _eval_f1({5: 1.0, 10: 0.5}), "f2": _eval_f2()}
            code = cli.cmd_evaluate(args)
        assert code == 0
        # 关键断言：metrics 收到的是已解析的 'cpu'，不是原始 'auto'
        assert me.call_args.kwargs["device"] == "cpu"
        assert me.call_args.kwargs["device"] != "auto"


class TestCollectionArg:
    """Task 2.1/2.2: --collection 参数解析与传入 manager（D2）."""

    def test_default_collection_for_all_subcommands(self):
        from KNN_evaluation.config import DEFAULT_COLLECTION
        parser = cli._build_parser()
        cases = {
            "import": ["import", "data_demo"],
            "search": ["search", "--random"],
            "stats": ["stats"],
            "evaluate": ["evaluate"],
            "migrate": ["migrate"],
        }
        for cmd, argv in cases.items():
            args = parser.parse_args(argv)
            assert args.collection == DEFAULT_COLLECTION, cmd

    def test_override_collection(self):
        parser = cli._build_parser()
        args = parser.parse_args(["stats", "--collection", "xian_aef_embedding"])
        assert args.collection == "xian_aef_embedding"


class TestCmdCollectionInjection:
    """cmd_* 以 args.collection 构造 QdrantManager."""

    def test_cmd_import_passes_collection_to_manager(self, monkeypatch):
        captured: dict = {}

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=5):
                captured["collection_name"] = collection_name

            def collection_exists(self):
                return True

            def collection_info(self):
                return {"total_points": 0}

        class FakeImporter:
            def __init__(self, manager, batch_size=None):
                pass

            def import_directory(self, data_dir, no_resume=False, reindex=False,
                                 progress_callback=None):
                return {"total_pixels": 0, "total_images": 0, "imported_images": 0,
                        "skipped_images": 0, "label_counts": {},
                        "elapsed_sec": 0.0, "rate_pps": 0.0}

        monkeypatch.setattr(cli, "QdrantManager", FakeManager)
        monkeypatch.setattr(cli, "PixelImporter", FakeImporter)
        args = SimpleNamespace(
            qdrant_url="http://localhost:1", collection="xian_aef_embedding",
            batch_size=10000, directory="data_demo", no_resume=False, reindex=False,
        )
        assert cli.cmd_import(args) == 0
        assert captured["collection_name"] == "xian_aef_embedding"

    def test_cmd_evaluate_passes_collection_to_manager(self, monkeypatch):
        manager = MagicMock()
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True
        manager.collection_info.return_value = {"total_points": 1000}
        manager.collection_name = "test_collection"
        args = _eval_args(collection="xian_aef_embedding")
        with patch("KNN_evaluation.cli.QdrantManager", return_value=manager) as mq, \
             patch("KNN_evaluation.metrics.sample_queries_by_label",
                   return_value=_eval_queries()), \
             patch("KNN_evaluation.metrics.evaluate_knn",
                   return_value={"f1": _eval_f1(), "f2": _eval_f2()}):
            code = cli.cmd_evaluate(args)
        assert code == 0
        assert mq.call_args.kwargs["collection_name"] == "xian_aef_embedding"

    def test_cmd_evaluate_missing_collection_returns_1_with_name(self, capsys, monkeypatch):
        mgr = MagicMock()
        mgr.health_check.return_value = True
        mgr.collection_exists.return_value = False
        mgr.collection_name = "xian_aef_embedding"
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        args = _eval_args(collection="xian_aef_embedding")
        assert cli.cmd_evaluate(args) == 1
        assert "xian_aef_embedding" in capsys.readouterr().err


class TestStatsCollection:
    """Task 2.3: stats 输出展示实际 collection 名称 + 不存在的 collection 报错."""

    def test_stats_missing_collection_returns_1_with_name(self, capsys, monkeypatch):
        mgr = MagicMock()
        mgr.collection_exists.return_value = False
        mgr.collection_name = "xian_aef_embedding"
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        args = SimpleNamespace(
            qdrant_url="http://localhost:1", collection="xian_aef_embedding", json=False,
        )
        code = cli.cmd_stats(args)
        assert code == 1
        assert "xian_aef_embedding" in capsys.readouterr().err

    def test_stats_output_shows_collection_name(self, capsys, monkeypatch):
        mgr = MagicMock()
        mgr.collection_exists.return_value = True
        mgr.collection_name = "xian_aef_embedding"
        mgr.collection_info.return_value = {
            "total_points": 100, "vectors_count": 100,
            "segments_count": 1, "status": "green",
        }
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        args = SimpleNamespace(
            qdrant_url="http://localhost:1", collection="xian_aef_embedding", json=False,
        )
        assert cli.cmd_stats(args) == 0
        assert "xian_aef_embedding" in capsys.readouterr().out


class TestSimilarityHeatmap:
    """Task 3: similarity-heatmap 子命令 — 参数、两类模式、错误码、输出文件."""

    def _manager(self, exists=True, total_points=100):
        mgr = MagicMock()
        mgr.health_check.return_value = True
        mgr.collection_exists.return_value = exists
        mgr.collection_info.return_value = {"total_points": total_points}
        mgr.collection_name = "c"
        return mgr

    def _args(self, **overrides):
        base = dict(
            qdrant_url="http://localhost:6333",
            google_collection="google_aef_embedding",
            xian_collection="xian_aef_embedding",
            n=200, seed=42, image_id=None, output="similarity_heatmap.png",
            export_dir=None,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_parser_defaults(self):
        parser = cli._build_parser()
        args = parser.parse_args(["similarity-heatmap"])
        assert args.n == 200
        assert args.seed == 42
        assert args.image_id is None
        assert args.output == "similarity_heatmap.png"
        assert args.google_collection == "google_aef_embedding"
        assert args.xian_collection == "xian_aef_embedding"
        assert args.qdrant_url == "http://localhost:6333"

    def test_n_out_of_range_rejected(self):
        parser = cli._build_parser()
        for bad in ("0", "601", "-1"):
            with pytest.raises(SystemExit):
                parser.parse_args(["similarity-heatmap", "--n", bad])
        assert parser.parse_args(["similarity-heatmap", "--n", "600"]).n == 600

    def test_image_id_selects_image_mode(self):
        parser = cli._build_parser()
        args = parser.parse_args(["similarity-heatmap", "--image-id", "E121.4_N25.1"])
        assert args.image_id == "E121.4_N25.1"

    def test_qdrant_unreachable_returns_1(self, capsys, monkeypatch):
        mgr = self._manager()
        mgr.health_check.return_value = False
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        assert cli.cmd_similarity_heatmap(self._args()) == 1
        assert "Qdrant 不可达" in capsys.readouterr().err

    def test_missing_collection_returns_1(self, capsys, monkeypatch):
        mgr = self._manager(exists=False)
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        assert cli.cmd_similarity_heatmap(self._args()) == 1
        assert "不存在" in capsys.readouterr().err

    def test_builds_two_managers(self, monkeypatch):
        captured: list = []

        def fake_factory(url=None, collection_name=None, timeout=5):
            captured.append(collection_name)
            mgr = self._manager()
            mgr.collection_name = collection_name
            return mgr

        monkeypatch.setattr(cli, "QdrantManager", fake_factory)
        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps",
            lambda gm, xm, **kw: {
                "sampled": 1, "kept": 1, "dropped": 0,
                "matrix_shape": [1, 1], "elapsed_sec": 0.1, "output_path": kw["output"],
            },
        )
        assert cli.cmd_similarity_heatmap(self._args()) == 0
        assert captured == ["google_aef_embedding", "xian_aef_embedding"]

    def test_output_file_generated_and_printed(self, tmp_path, capsys, monkeypatch):
        out = tmp_path / "heatmap.png"
        args = self._args(output=str(out))
        managers = iter([self._manager(), self._manager()])

        def fake_factory(url=None, collection_name=None, timeout=5):
            m = next(managers)
            m.collection_name = collection_name
            return m

        monkeypatch.setattr(cli, "QdrantManager", fake_factory)

        def fake_compare(gm, xm, **kw):
            out.write_bytes(b"PNGDATA")
            return {
                "sampled": 200, "kept": 199, "dropped": 1,
                "matrix_shape": [199, 199], "elapsed_sec": 1.0,
                "output_path": str(out),
            }

        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps",
            fake_compare,
        )
        assert cli.cmd_similarity_heatmap(args) == 0
        assert out.exists()
        captured_out = capsys.readouterr().out
        assert "相似度热力图对比完成" in captured_out
        assert "剔除 1" in captured_out
        assert str(out) in captured_out

    def test_parser_export_dir_default_outputs(self):
        """Task 9: --export-dir 默认 outputs；显式传空串可禁用."""
        parser = cli._build_parser()
        args = parser.parse_args(["similarity-heatmap"])
        assert args.export_dir == "outputs"
        args = parser.parse_args(["similarity-heatmap", "--export-dir", "out/sim"])
        assert args.export_dir == "out/sim"
        args = parser.parse_args(["similarity-heatmap", "--export-dir", ""])
        assert args.export_dir == ""

    def test_empty_export_dir_passed_through_to_compare(self, monkeypatch):
        """Task 9: --export-dir '' → 空串透传给编排函数（compare 层归一化为禁用）."""
        managers = iter([self._manager(), self._manager()])

        def fake_factory(url=None, collection_name=None, timeout=5):
            m = next(managers)
            m.collection_name = collection_name
            return m

        monkeypatch.setattr(cli, "QdrantManager", fake_factory)
        received: dict = {}

        def fake_compare(gm, xm, **kw):
            received["export_dir"] = kw.get("export_dir")
            return {
                "sampled": 1, "kept": 1, "dropped": 0,
                "matrix_shape": [1, 1], "elapsed_sec": 0.1, "output_path": "",
            }

        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps",
            fake_compare,
        )
        assert cli.cmd_similarity_heatmap(self._args(export_dir="")) == 0
        assert received["export_dir"] == ""

    def test_blank_export_dir_stripped_and_disabled(self, tmp_path, monkeypatch):
        """Minor 3: --export-dir '  ' 纯空白 → strip 为空串禁用导出，不创建带空格目录、不崩溃."""
        out = tmp_path / "heatmap.png"
        args = self._args(output=str(out), export_dir="   ")
        managers = iter([self._manager(), self._manager()])

        def fake_factory(url=None, collection_name=None, timeout=5):
            m = next(managers)
            m.collection_name = collection_name
            return m

        monkeypatch.setattr(cli, "QdrantManager", fake_factory)
        received: dict = {}

        def fake_compare(gm, xm, **kw):
            received["export_dir"] = kw.get("export_dir")
            out.write_bytes(b"PNGDATA")
            return {
                "sampled": 200, "kept": 200, "dropped": 0,
                "matrix_shape": [200, 200], "elapsed_sec": 0.5,
                "output_path": str(out),
            }

        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps",
            fake_compare,
        )
        assert cli.cmd_similarity_heatmap(args) == 0
        # strip 后为空 → 视为禁用导出（compare 层对空串归一化，不创建空格目录）
        assert received["export_dir"] == ""

    def test_export_oserror_returns_1(self, capsys, monkeypatch):
        """导出目录不可写：export 抛 OSError 时返回 1 且 stderr 含错误，不抛裸异常."""
        managers = iter([self._manager(), self._manager()])

        def fake_factory(url=None, collection_name=None, timeout=5):
            m = next(managers)
            m.collection_name = collection_name
            return m

        monkeypatch.setattr(cli, "QdrantManager", fake_factory)

        def boom(gm, xm, **kw):
            raise PermissionError("导出目录不可写")

        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps", boom,
        )
        assert cli.cmd_similarity_heatmap(self._args(export_dir="readonly")) == 1
        assert "导出目录不可写" in capsys.readouterr().err

    def test_export_dir_passed_and_files_printed(self, tmp_path, capsys, monkeypatch):
        out = tmp_path / "heatmap.png"
        export_dir = tmp_path / "exports"
        args = self._args(output=str(out), export_dir=str(export_dir))
        managers = iter([self._manager(), self._manager()])

        def fake_factory(url=None, collection_name=None, timeout=5):
            m = next(managers)
            m.collection_name = collection_name
            return m

        monkeypatch.setattr(cli, "QdrantManager", fake_factory)
        exported = ["g.npy", "x.npy", "sim.json"]
        received: dict = {}

        def fake_compare(gm, xm, **kw):
            received["export_dir"] = kw.get("export_dir")
            out.write_bytes(b"PNGDATA")
            return {
                "sampled": 200, "kept": 200, "dropped": 0,
                "matrix_shape": [200, 200], "elapsed_sec": 0.5,
                "output_path": str(out),
                "exported_files": exported,
            }

        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps",
            fake_compare,
        )
        assert cli.cmd_similarity_heatmap(args) == 0
        # export_dir 传给编排函数
        assert received["export_dir"] == str(export_dir)
        captured_out = capsys.readouterr().out
        for f in exported:
            assert f in captured_out
