"""Tests for webui training-monitoring page (FakeUI 驱动，不启动真实 NiceGUI).

webui.py 只在 `__main__` 分支启动 NiceGUI，import 不触发 ui.run；但 @ui.page
装饰器会注册页面。测试用 FakeUI 替换 `webui.ui` 驱动 `webui.index()` 构建页面，
捕获「开始训练 / 中止训练 / 导出 JSON」按钮回调并驱动 do_train 全流程。
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from LinearProbe_evaluation import webui
from LinearProbe_evaluation.trainer import TrainResult, TrainConfig, CancelledError
from LinearProbe_evaluation.tests.helpers import FakeQdrantManager, make_points, synthetic_pixel_dataset


class _FakeUI:
    """假 nicegui ui 模块：记录控件与回调，其余元素 MagicMock 兜底."""

    def __init__(self):
        self.labels: list[MagicMock] = []
        self.numbers: dict[str, MagicMock] = {}
        self.selects: dict[str, MagicMock] = {}
        self.inputs: dict[str, MagicMock] = {}
        self.buttons: dict[str, MagicMock] = {}
        self.progress_bars: list = []
        self.tables: list = []
        self.notify_calls: list = []
        self.download_calls: list = []
        self.images: list = []
        self._sections: list = []
        self.button_count = 0
        self.do_train = None
        self.cancel_callback = None
        self.export_callback = None

    def _cm(self, mock, title=None):
        mock.classes.return_value = mock
        mock.props.return_value = mock
        mock.style.return_value = mock
        mock.__enter__.side_effect = (
            lambda: (self._sections.append(title) if title else None, mock)[1]
        )
        mock.__exit__.side_effect = (
            lambda *a: (self._sections.pop() if title else None) or False
        )
        return mock

    def _make_element(self, value=None):
        m = MagicMock()
        m.value = value
        m.classes = Mock(side_effect=lambda *c, **kw: m)
        m.props = Mock(side_effect=lambda *a, **kw: m)
        m.style = Mock(side_effect=lambda *a, **kw: m)
        m.set_visibility = Mock()
        m.set_value = Mock(side_effect=lambda v: setattr(m, "value", v))
        return m

    def expansion(self, title=None, value=False, **kw):
        return self._cm(MagicMock(), title=title)

    def row(self, **kw):
        return self._cm(MagicMock())

    def column(self, **kw):
        return self._cm(MagicMock())

    def header(self, **kw):
        return self._cm(MagicMock())

    def label(self, text="", **kw):
        m = self._make_element(text)
        m.set_text = Mock(side_effect=lambda t: setattr(m, "text", t))
        m._classes = ""
        m.classes = Mock(side_effect=lambda *c: (setattr(m, "_classes", " ".join(c)), m)[1])
        self.labels.append(m)
        return m

    def number(self, label=None, value=None, **kw):
        m = self._make_element(value)
        m.min = kw.get("min")
        m.max = kw.get("max")
        m.step = kw.get("step")
        self.numbers[label] = m
        return m

    def select(self, options=None, value=None, label=None, **kw):
        m = self._make_element(value)
        m.options = options
        m._on_handlers = {}

        def _on(event, handler):
            m._on_handlers[event] = handler
            return m

        m.on = Mock(side_effect=_on)
        self.selects[label] = m
        return m

    def input(self, label=None, value=None, **kw):
        m = self._make_element(value)
        self.inputs[label] = m
        return m

    def timer(self, *a, **kw):
        m = MagicMock()
        return m

    def button(self, label=None, on_click=None, **kw):
        m = self._make_element(None)
        m.label = label
        m.on_click = on_click
        self.button_count += 1
        if label == "开始训练":
            self.do_train = on_click
        if label == "中止训练":
            self.cancel_callback = on_click
        if label == "导出 JSON":
            self.export_callback = on_click
        self.buttons[label] = m
        return m

    def table(self, columns=None, rows=None, row_key=None, **kw):
        m = self._make_element(None)
        m.columns = columns
        m.rows = rows
        self.tables.append((columns, rows))
        return m

    def linear_progress(self, value=0, **kw):
        m = self._make_element(value)
        section = self._sections[-1] if self._sections else None
        self.progress_bars.append((section, m))
        return m

    def image(self, **kw):
        m = self._make_element(None)
        m.set_source = Mock(side_effect=lambda s: setattr(m, "source", s))
        self.images.append(m)
        return m

    def markdown(self, *a, **kw):
        return MagicMock()

    def notify(self, message=None, type=None, **kw):
        self.notify_calls.append((message, type))

    def download(self, content, filename=None, media_type=None, **kw):
        self.download_calls.append((content, filename))

    def page_title(self, *a, **kw):
        return MagicMock()

    def __getattr__(self, name):
        def factory(*a, **kw):
            return MagicMock()
        return factory

    def progress_bar(self, section):
        for s, m in self.progress_bars:
            if s == section:
                return m
        raise AssertionError(f"未找到 section={section!r} 的进度条")


def _build_harness(monkeypatch) -> _FakeUI:
    fake = _FakeUI()
    monkeypatch.setattr(webui, "ui", fake)

    async def _noop_sleep(seconds):
        return None

    monkeypatch.setattr(webui, "asyncio_sleep", _noop_sleep)
    webui.index()
    return fake


@pytest.fixture(autouse=True)
def _reset_module_state():
    webui.state = {}
    webui._train_cancel_events = {}
    webui._set_train_cancel_btn_visible = None
    webui._current_collection = webui.DEFAULT_COLLECTION
    webui._known_collections = list(webui.PRESET_COLLECTIONS)
    yield
    webui.state = {}
    webui._train_cancel_events = {}
    webui._set_train_cancel_btn_visible = None
    webui._current_collection = webui.DEFAULT_COLLECTION
    webui._known_collections = list(webui.PRESET_COLLECTIONS)


class _TrainPanelFixture:
    """do_train 的公共 mock 数据与 helper."""

    @staticmethod
    def manager(total_points=1000):
        mgr = FakeQdrantManager(points=make_points(n_per_class=10))
        return mgr

    @staticmethod
    def result():
        """构造已完成训练的 TrainResult（含报告与 checkpoint 元数据）."""
        hist = [
            {"epoch": 1, "train_loss": 2.0, "train_acc": 0.4,
             "val_loss": 1.9, "val_acc": 0.42, "val_macro_f1": 0.35},
            {"epoch": 2, "train_loss": 1.5, "train_acc": 0.6,
             "val_loss": 1.4, "val_acc": 0.65, "val_macro_f1": 0.6},
        ]
        return TrainResult(
            history=hist,
            best_epoch=2,
            best_val_accuracy=0.65,
            best_val_macro_f1=0.6,
            best_checkpoint=Path("out/checkpoints/mlp_label_best.pt"),
            final_checkpoint=Path("out/checkpoints/mlp_label_final.pt"),
            val_report={
                "accuracy": 0.65,
                "macro_f1": 0.6,
                "weighted_f1": 0.64,
                "per_class": {
                    i: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 10}
                    for i in range(9)
                },
            },
            train_report={},
            elapsed_seconds=1.2,
            device="cuda",
        )

    def patch_train_pipeline(self, monkeypatch, result=None):
        """把采样/训练/出图替换为 mock，返回 (train_mock, split_mock).

        train_mock 的 side_effect 会按 result.history 触发 progress_callback
        （epoch_start / batch / epoch_end），以覆盖 WebUI 的实时曲线更新路径.
        """
        result = result or self.result()
        ds = synthetic_pixel_dataset(n_per_class=6, seed=1)
        split_mock = Mock(return_value=(ds, ds))

        def fake_train(train_ds, val_ds, config, progress_callback=None, cancel_event=None):
            total = len(result.history)
            if progress_callback is not None:
                for ep in result.history:
                    progress_callback({"event": "epoch_start", "epoch": ep["epoch"],
                                       "total_epochs": total})
                    progress_callback({"event": "batch", "epoch": ep["epoch"],
                                       "total_epochs": total, "batch": 1,
                                       "total_batches": 3, "loss": ep["train_loss"]})
                    progress_callback({"event": "epoch_end", **ep, "total_epochs": total})
            return result

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        monkeypatch.setattr(webui, "asyncio_to_thread", fake_to_thread)
        monkeypatch.setattr(webui, "stratified_train_val_split", split_mock)
        train_mock = Mock(side_effect=fake_train)
        monkeypatch.setattr(webui, "train_mlp", train_mock)
        monkeypatch.setattr(
            webui, "training_curves_base64",
            lambda hist: "data:image/png;base64,AAAA",
        )
        cm = np.zeros((9, 9), dtype=np.int64)
        monkeypatch.setattr(
            webui, "_compute_confusion_uri",
            lambda ckpt, ds, device: (cm, "data:image/png;base64,BBBB"),
        )
        return train_mock, split_mock


class TestPageStructure:
    def test_page_builds_controls(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        assert fake.do_train is not None, "应有「开始训练」按钮回调"
        assert fake.cancel_callback is not None, "应有「中止训练」按钮回调"
        assert fake.export_callback is not None, "应有「导出 JSON」按钮回调"
        # 参数控件
        assert "Epochs" in fake.numbers
        assert "每类训练样本数" in fake.numbers
        assert "每类验证样本数" in fake.numbers
        assert fake.selects["优化器"].value == "adam"
        assert fake.selects["模型结构"].value == "mlp"
        # 设备固定 CUDA（无设备选择器），GPU 信息 label 存在
        assert any("CUDA" in (getattr(m, "text", "") or "") for m in fake.labels)
        # 训练区进度条
        fake.progress_bar("MLP_label 训练（64 维 embedding → 9 类硬标签）")
        # 中止按钮初始隐藏
        fake.buttons["中止训练"].set_visibility.assert_called_with(False)
        # 导出按钮初始隐藏
        fake.buttons["导出 JSON"].set_visibility.assert_called_with(False)

    def test_manager_created_and_status(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        assert webui.state["manager"] is not None
        assert webui.state["manager"].health_check() is True


class TestDoTrain:
    def test_concurrent_train_guard(self, monkeypatch):
        """训练进行中再次点击开始训练 → warning 通知且不启动新任务."""
        fake = _build_harness(monkeypatch)
        webui.state["training"] = True
        asyncio.run(fake.do_train())
        assert fake.notify_calls and "进行中" in str(fake.notify_calls[-1][0])
        assert fake.notify_calls[-1][1] == "warning"

    def test_qdrant_unreachable_notifies(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        mgr = FakeQdrantManager(points=[], healthy=False)
        webui.state["manager"] = mgr
        asyncio.run(fake.do_train())
        assert fake.notify_calls and "不可达" in str(fake.notify_calls[-1][0])

    def test_successful_train_shows_results(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        fixture = _TrainPanelFixture()
        train_mock, split_mock = fixture.patch_train_pipeline(monkeypatch)
        webui.state["manager"] = fixture.manager()
        asyncio.run(fake.do_train())

        # 采样与训练均被调用，参数正确（默认值来自 config.DEFAULT_*）
        assert split_mock.call_count == 1
        args = split_mock.call_args
        assert args.kwargs["train_per_class"] == webui.DEFAULT_TRAIN_PER_CLASS   # 默认每类训练样本数
        assert args.kwargs["val_per_class"] == webui.DEFAULT_VAL_PER_CLASS       # 默认每类验证样本数
        train_cfg: TrainConfig = train_mock.call_args.args[2]
        assert isinstance(train_cfg, TrainConfig)
        assert train_cfg.epochs == webui.DEFAULT_EPOCHS
        # 结果区可见、导出按钮可见、有 per-class 表
        assert fake.buttons["导出 JSON"].set_visibility.called
        assert any(call.args == (True,) for call in fake.buttons["导出 JSON"].set_visibility.call_args_list)
        assert fake.tables, "应渲染 per-class 指标表"
        # 表格列：全部靠左，类别列可换行加宽，数值列合理宽度
        cols = fake.tables[-1][0]
        assert all(c.get("align") == "left" for c in cols), "所有列应统一靠左"
        label_col = next(c for c in cols if c["name"] == "label")
        assert "min-width: 200px" in label_col["style"]
        assert "word-break: break-word" in label_col["style"]
        for c in cols[1:]:
            assert "min-width: 110px" in c["style"], f"数值列 {c['name']} 应有合理宽度"
        # 曲线已更新
        assert any(img.source == "data:image/png;base64,AAAA" for img in fake.images)
        # 无 negative 通知
        assert not any(t == "negative" for _, t in fake.notify_calls)

    def test_custom_hyperparams_passed(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        fake.numbers["Epochs"].value = 3
        fake.numbers["每类训练样本数"].value = 100
        fake.numbers["每类验证样本数"].value = 50
        fake.selects["优化器"].value = "sgd"
        fake.selects["模型结构"].value = "linear"
        fixture = _TrainPanelFixture()
        train_mock, split_mock = fixture.patch_train_pipeline(monkeypatch)
        webui.state["manager"] = fixture.manager()
        asyncio.run(fake.do_train())
        split_args = split_mock.call_args.kwargs
        assert split_args["train_per_class"] == 100
        assert split_args["val_per_class"] == 50
        cfg = train_mock.call_args.args[2]
        assert cfg.epochs == 3 and cfg.optimizer == "sgd"
        assert cfg.model_variant == "linear"

    def test_cancel_notifies_and_resets(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        fixture = _TrainPanelFixture()
        fixture.patch_train_pipeline(monkeypatch)

        def raise_cancelled(train_ds, val_ds, config, progress_callback=None, cancel_event=None):
            raise CancelledError("训练已取消")

        monkeypatch.setattr(webui, "train_mlp", raise_cancelled)
        webui.state["manager"] = fixture.manager()
        asyncio.run(fake.do_train())
        assert any("取消" in str(m) for m, _ in fake.notify_calls)
        # 中止按钮最终隐藏、进度条隐藏
        cancel_btn = fake.buttons["中止训练"]
        assert cancel_btn.set_visibility.call_args_list[-1].args == (False,)
        prog_bar = fake.progress_bar("MLP_label 训练（64 维 embedding → 9 类硬标签）")
        assert prog_bar.set_visibility.call_args_list[-1].args == (False,)
        # 取消事件已清理（下次训练可重跑）
        assert not any(e.is_set() for e in webui._train_cancel_events.values())

    def test_new_train_recreates_cancel_event(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        fixture = _TrainPanelFixture()
        train_mock, _ = fixture.patch_train_pipeline(monkeypatch)
        webui.state["manager"] = fixture.manager()
        asyncio.run(fake.do_train())
        first_event = train_mock.call_args.kwargs.get("cancel_event")
        asyncio.run(fake.do_train())
        second_event = train_mock.call_args.kwargs.get("cancel_event")
        assert first_event is not None and second_event is not None
        assert first_event is not second_event, "每次训练应创建新的取消事件"

    def test_train_failure_notifies(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        fixture = _TrainPanelFixture()
        fixture.patch_train_pipeline(monkeypatch)

        def raise_error(*a, **kw):
            raise RuntimeError("OOM")

        monkeypatch.setattr(webui, "train_mlp", raise_error)
        webui.state["manager"] = fixture.manager()
        asyncio.run(fake.do_train())
        assert fake.notify_calls and "OOM" in str(fake.notify_calls[-1][0])
        assert fake.notify_calls[-1][1] == "negative"


class TestExportJson:
    """训练结果导出：JSON 与图像图表均落盘到 outputs/evaluation/linearprobe."""

    def _run_train(self, monkeypatch, tmp_path):
        fake = _build_harness(monkeypatch)
        fixture = _TrainPanelFixture()
        fixture.patch_train_pipeline(monkeypatch)
        webui.state["manager"] = fixture.manager()
        monkeypatch.setattr(webui, "_PROJECT_ROOT", tmp_path)
        asyncio.run(fake.do_train())
        return fake, tmp_path / "outputs" / "evaluation" / "linearprobe"

    def test_export_json_writes_to_lp_dir(self, monkeypatch, tmp_path):
        fake, target = self._run_train(monkeypatch, tmp_path)
        fake.export_callback()
        import json as _json
        jsons = list(target.glob("xian_mlp_result_*.json"))
        assert len(jsons) == 1, "应落盘 JSON 且文件名遵循 集合_模型_内容_时间戳"
        # 文件名含来源 collection 缩写与模型结构缩写（xian / mlp）
        assert "xian_aef_embedding" not in jsons[0].name, "应使用缩写而非全名"
        data = _json.loads(jsons[0].read_text(encoding="utf-8"))
        assert data["model"] == "MLP_label"
        assert data["best_val_accuracy"] == 0.65
        assert len(data["history"]) == 2
        # architecture 含结构信息（variant/hidden_dims/num_parameters）
        arch = data["architecture"]
        assert arch["class"] == "MLPLabel"
        assert arch["variant"] == "mlp"
        assert arch["hidden_dims"] == [256, 256, 256]
        assert arch["num_parameters"] == 150537
        # 不再走浏览器下载
        assert not fake.download_calls
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("已导出" in m for m in msgs)

    def test_export_linear_variant_architecture(self, monkeypatch, tmp_path):
        """选择 linear 结构后导出 JSON 记录单层结构信息，文件名含 linear."""
        fake, target = self._run_train(monkeypatch, tmp_path)
        fake.selects["模型结构"].value = "linear"
        # 训练后手动重设导出数据中的 variant（模拟 linear 结构训练完成）
        webui.state["lp_export"]["variant"] = "linear"
        webui._export_lp_results("json")
        import json as _json
        jsons = list(target.glob("xian_linear_result_*.json"))
        assert len(jsons) == 1, "文件名应含 linear 模型结构"
        data = _json.loads(jsons[-1].read_text(encoding="utf-8"))
        assert data["architecture"]["variant"] == "linear"
        assert data["architecture"]["hidden_dims"] == []
        assert data["architecture"]["num_parameters"] == 585

    def test_export_without_train_is_noop(self, monkeypatch, tmp_path):
        fake = _build_harness(monkeypatch)
        monkeypatch.setattr(webui, "_PROJECT_ROOT", tmp_path)
        fake.export_callback()
        target = tmp_path / "outputs" / "evaluation" / "linearprobe"
        assert not target.exists(), "未训练不应生成任何文件"
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("请先完成训练" in m for m in msgs)

    def test_export_images_writes_pngs(self, monkeypatch, tmp_path):
        fake, target = self._run_train(monkeypatch, tmp_path)
        webui._export_lp_results("images")
        pngs = list(target.glob("*.png"))
        names = {p.name for p in pngs}
        assert any("_curves_" in n for n in names)
        assert any("_cm_" in n for n in names)
        # 文件名统一格式：集合缩写_模型结构_内容缩写_时间戳（xian_mlp_...）
        for n in names:
            assert n.startswith("xian_mlp_"), f"文件名应遵循 xian_mlp_ 前缀: {n}"
            assert "xian_aef_embedding" not in n, "应使用缩写而非全名"
        # 均为 PNG 格式
        for p in pngs:
            assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("已导出" in m for m in msgs)

    def test_export_uses_google_short_after_switch(self, monkeypatch, tmp_path):
        """切换到 google collection 后导出文件名使用缩写 google."""
        fake, target = self._run_train(monkeypatch, tmp_path)
        webui._current_collection = "google_aef_embedding"
        webui._export_lp_results("json")
        jsons = list(target.glob("google_mlp_result_*.json"))
        assert len(jsons) == 1
        assert "google_aef_embedding" not in jsons[-1].name

    def test_export_buttons_visible_after_train(self, monkeypatch, tmp_path):
        fake, _ = self._run_train(monkeypatch, tmp_path)
        assert fake.buttons["导出 JSON"].set_visibility.call_args_list[-1] == ((True,),)
        assert fake.buttons["导出图片图表"].set_visibility.call_args_list[-1] == ((True,),)

    def test_export_buttons_hidden_before_train(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        assert fake.buttons["导出 JSON"].set_visibility.call_args_list[-1] == ((False,),)
        assert fake.buttons["导出图片图表"].set_visibility.call_args_list[-1] == ((False,),)


class TestCollectionSelector:
    """Collection 选择：选择器构建、切换后 manager 用新 collection、自定义、localStorage."""

    def test_page_builds_collection_selector(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        sel = fake.selects["Qdrant Collection"]
        assert sel.value == webui.DEFAULT_COLLECTION == "xian_aef_embedding"
        assert sorted(sel.options) == sorted(["google_aef_embedding", "xian_aef_embedding"])
        # 自定义名称输入框与使用按钮存在
        assert "自定义 collection 名称" in fake.inputs
        assert "使用自定义 Collection" in fake.buttons
        # 「切换 collection」按钮存在
        assert "切换 collection" in fake.buttons

    def test_switch_collection_button_applies_selection(self, monkeypatch):
        """修改下拉后点击「切换 collection」才生效，manager 重建."""
        fake = _build_harness(monkeypatch)
        persisted: list[str] = []
        monkeypatch.setattr(webui, "_persist_collection_choice", lambda c: persisted.append(c))

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=10):
                self.collection_name = collection_name
                self.url = url

        monkeypatch.setattr(webui, "QdrantManager", FakeManager)
        webui.state = {"manager": MagicMock()}
        # 修改下拉为 google，但未点击切换 → 当前 collection 不变
        fake.selects["Qdrant Collection"].value = "google_aef_embedding"
        assert webui._current_collection == "xian_aef_embedding"
        assert webui.state["manager"].collection_name != "google_aef_embedding"
        # 点击「切换 collection」→ 生效
        asyncio.run(fake.buttons["切换 collection"].on_click())
        assert webui._current_collection == "google_aef_embedding"
        assert webui.state["manager"].collection_name == "google_aef_embedding"
        assert persisted == ["google_aef_embedding"]

    def test_switch_collection_same_value_noop(self, monkeypatch):
        """下拉值等于当前 collection 时点击切换 → 提示当前已是，不重建 manager."""
        fake = _build_harness(monkeypatch)
        webui.state = {"manager": MagicMock()}
        manager = webui.state["manager"]
        fake.selects["Qdrant Collection"].value = webui._current_collection  # 默认 xian
        asyncio.run(fake.buttons["切换 collection"].on_click())
        assert webui.state["manager"] is manager, "同 collection 不应重建 manager"
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("当前已是" in m for m in msgs)

    def test_collection_short_name(self):
        assert webui.collection_short_name("google_aef_embedding") == "google"
        assert webui.collection_short_name("xian_aef_embedding") == "xian"
        assert webui.collection_short_name("my_embedding") == "my_embedding"

    def test_apply_collection_rebuilds_manager(self, monkeypatch):
        """切换 collection：更新模块级状态、重建 manager、持久化."""
        persisted: list[str] = []
        monkeypatch.setattr(webui, "_persist_collection_choice", lambda c: persisted.append(c))

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=10):
                self.collection_name = collection_name
                self.url = url

        monkeypatch.setattr(webui, "QdrantManager", FakeManager)
        webui.state = {"manager": MagicMock()}
        asyncio.run(webui._apply_collection("google_aef_embedding"))
        assert webui._current_collection == "google_aef_embedding"
        assert webui.state["manager"].collection_name == "google_aef_embedding"
        assert persisted == ["google_aef_embedding"]

    def test_apply_same_collection_is_noop(self, monkeypatch):
        """同 collection 切换不应重建 manager."""
        monkeypatch.setattr(webui, "_current_collection", "xian_aef_embedding")
        webui.state = {"manager": MagicMock()}
        manager = webui.state["manager"]
        asyncio.run(webui._apply_collection("xian_aef_embedding"))
        assert webui.state["manager"] is manager

    def test_add_custom_collection_switches(self, monkeypatch):
        """自定义名称：加入已知列表并切换，manager 用新 collection."""
        monkeypatch.setattr(webui, "_known_collections", list(webui.PRESET_COLLECTIONS))
        monkeypatch.setattr(webui, "QdrantManager",
                            lambda url=None, collection_name=None, timeout=10: SimpleNamespace(
                                collection_name=collection_name, url=url))
        webui.state = {"manager": MagicMock()}
        asyncio.run(webui._add_custom_collection("my_embedding"))
        assert webui._current_collection == "my_embedding"
        assert webui._known_collections == ["google_aef_embedding", "xian_aef_embedding", "my_embedding"]
        assert webui.state["manager"].collection_name == "my_embedding"

    def test_add_custom_collection_rejects_invalid(self, monkeypatch):
        """空名称 / 含路径分隔符的自定义 collection 拒绝，不切换."""
        fake = _build_harness(monkeypatch)
        asyncio.run(webui._add_custom_collection("   "))
        assert webui._current_collection == webui.DEFAULT_COLLECTION
        asyncio.run(webui._add_custom_collection("a/b"))
        assert webui._current_collection == webui.DEFAULT_COLLECTION
        assert fake.notify_calls and fake.notify_calls[-1][1] == "negative"

    def test_resolve_stored_collection_valid(self, monkeypatch):
        """localStorage 记录有效则恢复，失效/空回退默认."""
        monkeypatch.setattr(webui, "_known_collections", ["google_aef_embedding", "xian_aef_embedding"])
        assert webui._resolve_stored_collection("xian_aef_embedding") == "xian_aef_embedding"
        assert webui._resolve_stored_collection("deleted_col") == webui.DEFAULT_COLLECTION
        assert webui._resolve_stored_collection("") == webui.DEFAULT_COLLECTION
        assert webui._resolve_stored_collection(None) == webui.DEFAULT_COLLECTION

    def test_restore_stored_collection_applies(self, monkeypatch):
        """页面加载恢复上次选择：与当前不同则切换."""
        monkeypatch.setattr(webui, "ui", _FakeUI())
        monkeypatch.setattr(webui, "asyncio_to_thread", lambda fn, *a, **kw: fn(*a, **kw))
        webui._current_collection = "xian_aef_embedding"
        monkeypatch.setattr(webui, "_resolve_stored_collection", lambda stored: "google_aef_embedding")
        applied: list[str] = []

        async def _fake_apply(new_col):
            applied.append(new_col)

        monkeypatch.setattr(webui, "_apply_collection", _fake_apply)
        asyncio.run(webui._restore_stored_collection())
        assert applied == ["google_aef_embedding"]

    def test_restore_stored_collection_same_ensures_manager(self, monkeypatch):
        """恢复值等于当前选择：确保 manager 已就绪且使用当前 collection."""
        monkeypatch.setattr(webui, "_current_collection", "xian_aef_embedding")
        monkeypatch.setattr(webui, "_resolve_stored_collection", lambda stored: "xian_aef_embedding")
        monkeypatch.setattr(webui, "ui", _FakeUI())
        webui.state = {"manager": None}

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=10):
                self.collection_name = collection_name
                self.url = url

        monkeypatch.setattr(webui, "QdrantManager", FakeManager)
        asyncio.run(webui._restore_stored_collection())
        assert webui.state["manager"].collection_name == "xian_aef_embedding"
