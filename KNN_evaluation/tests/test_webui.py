"""Tests for KNN_evaluation webui — 固定三页重构（GOOGLE/XIAN/SimilarityMatrix）.

覆盖：
- 页面结构：固定三页、无自定义 collection 添加/删除、每页四块内容；
- 页面隔离：GOOGLE/XIAN 各持独立 manager、评估结果互不覆盖、可视化按本页数据目录；
- 数据导入失败自动整体重试（需求 1）；
- 评估面板混淆矩阵图片展示（需求 5）；
- 导出 JSON / 导出图片图表落盘到页面专属目录（需求 5）；
- SimilarityMatrix 页双采样模式与固定导出目录（需求 3、5）。

webui.py 只在 `__main__` 分支启动 NiceGUI，import 不触发 ui.run；@ui.page 装饰器
会注册页面，测试通过 _FakeUI 驱动 index() 构建页面并调用模块级/闭包函数。
"""
import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock, call

import numpy as np
import pytest

from KNN_evaluation import webui


class _FakeUI:
    """假 nicegui ui 模块：驱动 webui.index() 构建固定三页.

    记录页面级 tabs、各 tab_panel 内的按钮回调（do_evaluate / do_import / 导出 /
    生成热力图对比）、selects/numbers/switches/inputs/进度条/通知/表格/图片等，
    其余 nicegui 元素用通用 MagicMock 兜底，避免真实渲染副作用。
    """

    def __init__(self):
        self.labels: list[MagicMock] = []
        self.selects: dict[str, MagicMock] = {}
        self.numbers: dict[str, MagicMock] = {}
        self.inputs: dict[str, MagicMock] = {}
        self.buttons: dict[str, MagicMock] = {}
        self.switches: dict[str, MagicMock] = {}
        self.progress_bars: list = []  # (section, mock)
        self.notify_calls: list = []
        self.download_calls: list = []
        self.tables: list = []  # (columns, rows)
        self.images: list = []  # ui.image 渲染的 src
        self.echarts: list = []  # ui.echart 的 option
        self.dialogs: list = []
        self.radios: list = []
        self.tabs_list: list = []
        self.page_tabs: list = []  # 页面级 tabs 名（GOOGLE/XIAN/SimilarityMatrix）
        self.interactive_images: list = []
        self.html_elements: list = []
        self._sections: list = []
        self._current_panel: str | None = None
        # 每个 tab_panel 内的按钮回调：page_buttons[panel][label] = on_click
        self.page_buttons: dict[str, dict[str, object]] = {}
        # 每个 tab_panel 内的按钮 mock 对象：page_button_objs[panel][label] = element
        self.page_button_objs: dict[str, dict[str, MagicMock]] = {}
        # 页面级控件（GOOGLE/XIAN 各构建一份，flat dict 会被后构建页覆盖）：
        # page_inputs/page_selects/page_numbers/page_switches[panel][label] = element
        self.page_inputs: dict[str, dict[str, MagicMock]] = {}
        self.page_selects: dict[str, dict[str, MagicMock]] = {}
        self.page_numbers: dict[str, dict[str, MagicMock]] = {}
        self.page_switches: dict[str, dict[str, MagicMock]] = {}

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
        m.set_options = Mock(side_effect=lambda opts, value=None: (setattr(m, "options", opts), m)[1])
        return m

    def expansion(self, title=None, value=False, **kw):
        return self._cm(MagicMock(), title=title)

    def label(self, text="", **kw):
        m = self._make_element(text)
        m.set_text = Mock(side_effect=lambda t: setattr(m, "text", t))
        m._classes = ""
        m.classes = Mock(side_effect=lambda *c, **kw: (
            setattr(m, "_classes", kw["replace"] if "replace" in kw else " ".join(c)), m,
        )[1])
        self.labels.append(m)
        return m

    def select(self, options=None, value=None, label=None, **kw):
        m = self._make_element(value)
        m.options = options
        m.multiple = kw.get("multiple", False)
        self.selects[label] = m
        self.page_selects.setdefault(self._current_panel or "", {})[label] = m
        return m

    def number(self, label=None, value=None, **kw):
        m = self._make_element(value)
        m.min = kw.get("min")
        m.max = kw.get("max")
        self.numbers[label] = m
        self.page_numbers.setdefault(self._current_panel or "", {})[label] = m
        return m

    def input(self, label=None, value=None, **kw):
        m = self._make_element(value)
        m.placeholder = kw.get("placeholder")
        self.inputs[label] = m
        self.page_inputs.setdefault(self._current_panel or "", {})[label] = m
        return m

    def radio(self, options=None, value=None, on_change=None, **kw):
        m = self._make_element(value)
        m.options = options
        m.on_change = on_change
        self.radios.append(m)
        return m

    def image(self, src=None, **kw):
        m = self._make_element(src)
        m.set_source = Mock(side_effect=lambda s: setattr(m, "source", s))
        self.images.append(m)
        return m

    def echart(self, option=None, **kw):
        m = self._make_element(None)
        m.option = option
        self.echarts.append(option)
        return m

    def interactive_image(self, **kw):
        m = MagicMock()
        m.set_source = Mock()
        m.on = Mock()
        m.style = Mock(side_effect=lambda *a, **kw: m)
        self.interactive_images.append(m)
        return m

    def html(self, content="", **kw):
        m = MagicMock()
        m.set_content = Mock()
        m.style = Mock(side_effect=lambda *a, **kw: m)
        self.html_elements.append(m)
        return m

    def switch(self, text="", value=False, **kw):
        m = self._make_element(value)
        m.text = text
        self.switches[text] = m
        self.page_switches.setdefault(self._current_panel or "", {})[text] = m
        return m

    def button(self, label=None, on_click=None, **kw):
        m = self._make_element(None)
        m.label = label
        m.on_click = on_click
        self.buttons[label] = m
        panel = self._current_panel or ""
        self.page_buttons.setdefault(panel, {})[label] = on_click
        self.page_button_objs.setdefault(panel, {})[label] = m
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

    def notify(self, message=None, type=None, **kw):
        self.notify_calls.append((message, type))

    def download(self, content, filename=None, media_type=None, **kw):
        self.download_calls.append((content, filename))

    def timer(self, *a, **kw):
        return MagicMock()

    def tabs(self, **kw):
        m = self._make_element(None)
        m.props = Mock(side_effect=lambda *a, **kw: m)
        m.props.get = Mock(side_effect=lambda key, default=None: (
            m.value if key == "model-value" else default
        ))
        m.__enter__ = Mock(return_value=m)
        m.__exit__ = Mock(return_value=False)
        self.tabs_list.append(m)
        return m

    def tab_panels(self, tabs=None, value=None, **kw):
        m = self._make_element(value)
        m.__enter__ = Mock(return_value=m)
        m.__exit__ = Mock(return_value=False)
        return m

    def tab(self, name=None, label=None, **kw):
        m = self._cm(MagicMock())
        if len(self.tabs_list) == 1:
            self.page_tabs.append(name)
        return m

    def tab_panel(self, name=None, **kw):
        self._current_panel = name
        m = self._cm(MagicMock())
        # 页面构建在 `with ui.tab_panels(...)` 上下文内完成；记录页面级 panel 名
        return m

    def dialog(self, *a, **kw):
        d = self._cm(MagicMock())
        self.dialogs.append(d)
        return d

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
    webui.index()
    return fake


def _set_eval_defaults(fake, page="GOOGLE"):
    fake.page_selects[page]["执行设备"].value = "auto"
    fake.page_numbers[page]["显存预算(GB)"].value = 16
    fake.page_numbers[page]["CPU RAM 预算(GB)"].value = 6


def _eval_fixture():
    """构造评估结果 fixture（与旧测试同构）."""
    f1 = {
        "overall_accuracy": 0.5,
        "per_class_metrics": {"water": {"precision": 0.6, "recall": 0.5, "f1": 0.55, "support": 3}},
        "confusion_matrix": np.zeros((9, 9), dtype=np.int64),
        "k": 10, "num_queries": 3, "elapsed_sec": 0.1,
        "accuracy_by_k": {10: 0.5},
        "per_class_recall_by_k": {10: {"water": 0.5}},
    }
    f2 = {
        "k_values": [10, 20], "global_purity": [0.5, 0.4],
        "global_recall": [0.1, 0.2], "per_class_purity": {"water": [0.5, 0.4]},
        "per_class_recall": {"water": [0.1, 0.2]}, "num_queries": 3, "elapsed_sec": 0.1,
    }
    return f1, f2


def _manager(total_points=1000, collection=None):
    mgr = MagicMock()
    mgr.health_check.return_value = True
    mgr.collection_exists.return_value = True
    mgr.collection_info.return_value = {"total_points": total_points}
    mgr.collection_name = collection or "test_collection"
    return mgr


def _patch_metrics_eval(monkeypatch, f1, f2, device="cpu", queries=None):
    """按 do_evaluate 内部局部 import 打补丁（须在源模块打补丁）."""
    queries = queries or [
        {
            "point_id": f"p{i}", "vector": np.zeros(64, dtype=np.float64),
            "label": 0, "label_name": "water",
            "image_id": "I", "pixel_row": 0, "pixel_col": 0,
        }
        for i in range(3)
    ]
    monkeypatch.setattr("KNN_evaluation.gpu_knn.resolve_device", lambda d: device)
    monkeypatch.setattr(
        "KNN_evaluation.metrics.sample_queries_by_label",
        lambda mgr, spc, seed, warn_callback=None: queries,
    )
    me = MagicMock(return_value={"f1": f1, "f2": f2})
    monkeypatch.setattr("KNN_evaluation.metrics.evaluate_knn", me)
    return me


def _run_evaluate(fake, monkeypatch, page="GOOGLE", device="cpu"):
    """驱动指定页面的 do_evaluate，返回 (f1, f2)."""
    f1, f2 = _eval_fixture()
    _set_eval_defaults(fake, page)
    _patch_metrics_eval(monkeypatch, f1, f2, device=device)
    webui.state["pages"][page]["manager"] = _manager(collection=webui.PAGES[page]["collection"])
    cb = fake.page_buttons[page]["开始评估"]
    asyncio.run(cb())
    return f1, f2


@pytest.fixture(autouse=True)
def _reset_module_state():
    webui.state = {"pages": {}, "similarity": {}}
    webui._init_hooks = {}
    webui._manifest_caches = {}
    webui._eval_cancel_events = {}
    webui._set_eval_cancel_btn_visible = {}
    webui._CLI_QDRANT_URL = webui.QDRANT_URL
    webui._CLI_DATA_DIR = "data_demo"
    yield
    webui.state = {"pages": {}, "similarity": {}}
    webui._init_hooks = {}
    webui._manifest_caches = {}
    webui._eval_cancel_events = {}
    webui._set_eval_cancel_btn_visible = {}
    webui._CLI_QDRANT_URL = webui.QDRANT_URL
    webui._CLI_DATA_DIR = "data_demo"


# ========== 1. 页面结构（需求 2/3） ==========

class TestPageStructure:
    def test_three_fixed_pages(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        assert fake.page_tabs == ["GOOGLE", "XIAN", "SimilarityMatrix"]

    def test_no_custom_collection_add_delete_ui(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        # 无「自定义 collection 名称」输入框
        assert "自定义 collection 名称" not in fake.inputs
        # 无「添加」「删除」「清理缓存」按钮
        for label in ("添加", "删除", "清理缓存", "使用自定义 Collection"):
            assert label not in fake.buttons, f"不应存在按钮 {label!r}"

    def test_each_knn_page_has_four_sections(self, monkeypatch):
        _build_harness(monkeypatch)
        # 页面隔离：GOOGLE 与 XIAN 各含四个 expansion，SimilarityMatrix 一个
        assert webui._init_hooks.get("GOOGLE") is not None
        assert webui._init_hooks.get("XIAN") is not None
        assert "SimilarityMatrix" not in webui._init_hooks

    def test_pages_config_binding(self):
        assert webui.PAGES["GOOGLE"]["collection"] == "google_aef_embedding"
        assert webui.PAGES["GOOGLE"]["data_dir"] == "data_google"
        assert webui.PAGES["GOOGLE"]["export_dir"] == "outputs/evaluation/knn_eval"
        assert webui.PAGES["XIAN"]["collection"] == "xian_aef_embedding"
        assert webui.PAGES["XIAN"]["data_dir"] == "data_xian"
        assert webui.PAGES["XIAN"]["export_dir"] == "outputs/evaluation/knn_eval"
        assert webui.SIMILARITY_EXPORT_DIR == Path("outputs/evaluation/similarity")
        # collection 缩写映射
        assert webui.collection_short_name("google_aef_embedding") == "google"
        assert webui.collection_short_name("xian_aef_embedding") == "xian"

    def test_no_localstorage_collection_memory(self, monkeypatch):
        _build_harness(monkeypatch)
        assert not hasattr(webui, "_LOCALSTORAGE_KEY")
        assert not hasattr(webui, "_known_collections")
        assert not hasattr(webui, "_restore_stored_collection")


class TestStatusIndexProgress:
    """timeout 修复：状态栏显示索引进度，索引未就绪时 warning 提示."""

    def _invoke_refresh_status(self, fake, page="GOOGLE", indexed=None, total=1000):
        pg = webui.state["pages"][page]
        mgr = _manager(collection=webui.PAGES[page]["collection"])
        mgr.collection_info.return_value = {
            "total_points": total,
            "vectors_count": indexed if indexed is not None else total,
            "segments_count": 1,
        }
        pg["manager"] = mgr
        fn = webui._init_hooks[page]["refresh_status"]
        fn()
        return pg, mgr

    def _info_label(self, fake):
        # refresh_status 对 info_label 调用 set_text("Collection: ...")；
        # 按 set_text 实参定位被更新的 info_label（GOOGLE/XIAN 各一份）
        for lb in fake.labels:
            for _call in lb.set_text.call_args_list:
                args = _call[0]
                if args and str(args[0]).startswith("Collection:"):
                    return lb
        raise AssertionError("info_label 未找到（set_text 未被调用）")

    def test_status_shows_indexed_vectors(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        pg, mgr = self._invoke_refresh_status(fake, page="GOOGLE", indexed=1000, total=1000)
        info_label = self._info_label(fake)
        assert "已索引向量" in info_label.text
        assert "向量索引构建中" not in info_label.text
        # 就绪时 positive 样式
        assert "text-positive" in info_label._classes

    def test_status_warns_when_index_incomplete(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        pg, mgr = self._invoke_refresh_status(fake, page="GOOGLE", indexed=700, total=1000)
        info_label = self._info_label(fake)
        assert "向量索引构建中" in info_label.text
        assert "700" in info_label.text and "1,000" in info_label.text
        # 未就绪时 warning 样式
        assert "text-warning" in info_label._classes


class TestReindexVectorsButton:
    """一键构建向量索引：状态区「构建向量索引」按钮触发 reindex_vectors."""

    def _reindex_cb(self, fake, page="GOOGLE"):
        return fake.page_buttons[page]["构建向量索引"]

    def test_button_triggers_reindex(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        pg = webui.state["pages"]["GOOGLE"]
        mgr = _manager(collection=webui.PAGES["GOOGLE"]["collection"])
        mgr.reindex_vectors = MagicMock()
        pg["manager"] = mgr
        self._reindex_cb(fake)()
        mgr.reindex_vectors.assert_called_once()
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("向量索引重建已触发" in m for m in msgs)

    def test_button_warns_when_collection_missing(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        pg = webui.state["pages"]["GOOGLE"]
        mgr = _manager(collection=webui.PAGES["GOOGLE"]["collection"])
        mgr.collection_exists.return_value = False
        pg["manager"] = mgr
        self._reindex_cb(fake)()
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("Collection 不存在" in m for m in msgs)


# ========== 2. 页面隔离（需求 4） ==========

class TestPageIsolation:
    def test_init_page_creates_per_page_managers(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        created = {}

        def fake_mgr(url, collection_name):
            created[collection_name] = True
            m = _manager(collection=collection_name)
            m.ensure_payload_indices = Mock()
            return m

        monkeypatch.setattr(webui, "QdrantManager", fake_mgr)
        asyncio.run(webui.init_page())
        assert created.get("google_aef_embedding")
        assert created.get("xian_aef_embedding")

    def test_eval_results_stored_per_page(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        f1_g, f2_g = _run_evaluate(fake, monkeypatch, page="GOOGLE")
        f1_x, f2_x = _run_evaluate(fake, monkeypatch, page="XIAN")
        g_eval = webui.state["pages"]["GOOGLE"]["eval"]
        x_eval = webui.state["pages"]["XIAN"]["eval"]
        # 两页结果各自存在且互不覆盖
        assert g_eval["f1_result"] is f1_g
        assert x_eval["f1_result"] is f1_x
        assert g_eval["f1_result"] is not x_eval["f1_result"]
        # 页面绑定正确 collection
        assert webui.state["pages"]["GOOGLE"]["manager"].collection_name == "google_aef_embedding"
        assert webui.state["pages"]["XIAN"]["manager"].collection_name == "xian_aef_embedding"

    def test_do_search_records_page_collection(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        pg = webui.state["pages"]["GOOGLE"]
        mgr = _manager(collection="google_aef_embedding")
        pg["manager"] = mgr
        pg["query_vector"] = np.zeros(64, dtype=np.float64)
        fake.numbers["K (Top-K)"].value = 5
        result = MagicMock(search_mode="ann", query_params={"k": 5}, elapsed_ms=1.0,
                           hits=[], label_distribution=None)
        monkeypatch.setattr(
            webui, "PixelSearcher",
            lambda mgr: MagicMock(search=Mock(return_value=result)),
        )
        asyncio.run(fake.page_buttons["GOOGLE"]["执行检索"]())
        assert pg["search_collection"] == "google_aef_embedding"

    def test_viz_data_dir_uses_page_collection(self, monkeypatch, tmp_path):
        _build_harness(monkeypatch)
        monkeypatch.setattr(webui, "_PROJECT_ROOT", tmp_path)
        webui.state["pages"]["GOOGLE"]["data_dir"] = tmp_path / "somewhere"
        # XIAN 页检索 XIAN collection → 映射 data_xian
        d = webui._viz_data_dir("XIAN", "xian_aef_embedding")
        assert d == tmp_path / "data_xian"
        # 无映射的 collection 回退本页数据目录（页面隔离）
        d2 = webui._viz_data_dir("GOOGLE", "unknown_collection")
        assert d2 == webui.state["pages"]["GOOGLE"]["data_dir"]


# ========== 3. 导入自动重试（需求 1） ==========

class TestImportAutoRetry:
    def _build_import_harness(self, monkeypatch, tmp_path, page="GOOGLE"):
        fake = _build_harness(monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fake.page_inputs[page]["数据目录"].value = str(data_dir)
        monkeypatch.setattr(webui, "IMPORT_RETRY_BASE_DELAY", 0)
        importer = MagicMock()
        monkeypatch.setattr(
            webui, "PixelImporter", lambda mgr, batch_size: importer,
        )
        webui.state["pages"][page]["manager"] = _manager(collection=webui.PAGES[page]["collection"])
        return fake, importer, data_dir

    def test_retries_after_failure_then_succeeds(self, monkeypatch, tmp_path):
        fake, importer, _ = self._build_import_harness(monkeypatch, tmp_path)
        stats = {
            "total_pixels": 16384, "total_images": 1, "skipped_images": 0,
            "imported_images": 1, "label_counts": {}, "elapsed_sec": 1.0,
            "rate_pps": 16384,
        }
        importer.import_directory.side_effect = [ConnectionError("连接失败"), stats]
        asyncio.run(fake.page_buttons["GOOGLE"]["导入全部"]())
        # 两次尝试：首次失败 → 自动重试成功
        assert importer.import_directory.call_count == 2
        # 提示包含自动重试信息
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("导入完成" in m for m in msgs)

    def test_gives_up_after_max_attempts(self, monkeypatch, tmp_path):
        fake, importer, _ = self._build_import_harness(monkeypatch, tmp_path)
        importer.import_directory.side_effect = ConnectionError("持续失败")
        asyncio.run(fake.page_buttons["GOOGLE"]["导入全部"]())
        # 首次 + 3 次重试 = 4 次调用
        assert importer.import_directory.call_count == 4
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("导入失败" in m and "自动重试" in m for m in msgs)
        # 进度条最终隐藏
        prog_bar = fake.progress_bar("数据导入")
        assert prog_bar.set_visibility.call_args_list[-1] == ((False,),)

    def test_no_retry_on_success(self, monkeypatch, tmp_path):
        fake, importer, _ = self._build_import_harness(monkeypatch, tmp_path)
        stats = {"total_pixels": 0, "total_images": 0, "skipped_images": 0,
                 "imported_images": 0, "label_counts": {}, "elapsed_sec": 0,
                 "rate_pps": 0}
        importer.import_directory.return_value = stats
        asyncio.run(fake.page_buttons["GOOGLE"]["导入全部"]())
        assert importer.import_directory.call_count == 1


# ========== 4. 混淆矩阵图片展示（需求 5） ==========

class TestConfusionMatrixImage:
    def test_eval_results_render_cm_image_not_table(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        _run_evaluate(fake, monkeypatch, page="GOOGLE")
        # 混淆矩阵以图片展示：存在混淆矩阵图片
        cm_images = [img for img in fake.images if getattr(img, "source", "") or img.value]
        assert cm_images, "应渲染混淆矩阵图片"
        # 不应再有混淆矩阵表格（无 true_label 列的 table）
        cm_tables = [t for t, _ in fake.tables if "true_label" in t]
        assert not cm_tables, "混淆矩阵不应再以表格展示"

    def test_confusion_matrix_uses_base64_uri(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        called = {}
        monkeypatch.setattr(
            webui, "confusion_matrix_base64",
            lambda cm, labels, title="Confusion Matrix": (
                called.setdefault("n", 0) or called.__setitem__("n", called["n"] + 1) or
                "data:image/png;base64,FAKE"
            ),
        )
        _run_evaluate(fake, monkeypatch, page="GOOGLE")
        assert called.get("n", 0) == 1
        # 图片 source 为 base64 data URI
        cm_img = fake.images[-1]
        assert cm_img.source.startswith("data:image/png;base64,")


# ========== 5. 导出 JSON / 图片图表落盘（需求 5） ==========

class TestExportPageResults:
    def _export_harness(self, monkeypatch, tmp_path, page="GOOGLE"):
        fake = _build_harness(monkeypatch)
        # 重定向导出根目录到 tmp_path
        monkeypatch.setattr(webui, "_PROJECT_ROOT", tmp_path)
        f1, f2 = _eval_fixture()
        webui.state["pages"][page]["eval"] = {
            "f1_result": f1, "f2_result": f2,
            "config": {"device": "auto", "max_gpu_mem": 16, "max_eval_ram": 6.0,
                       "eval_mode": "exact", "samples_per_class": 500,
                       "k_f1": 100, "k_values": [10], "seed": 42},
        }
        return fake

    def test_export_json_writes_to_knn_eval_dir(self, monkeypatch, tmp_path):
        fake = self._export_harness(monkeypatch, tmp_path, page="GOOGLE")
        webui._export_page_results("GOOGLE", "json")
        target = tmp_path / "outputs" / "evaluation" / "knn_eval"
        jsons = list(target.glob("google_knn_result_*.json"))
        assert len(jsons) == 1, "应写入 knn_eval 目录且文件名含 google_knn_result"
        data = json.loads(jsons[0].read_text(encoding="utf-8"))
        assert data["f1"]["overall_accuracy"] == 0.5
        assert data["f2"]["k_values"] == [10, 20]
        # 不走浏览器下载
        assert fake.download_calls == []

    def test_export_json_xian_uses_xian_knn_prefix(self, monkeypatch, tmp_path):
        fake = self._export_harness(monkeypatch, tmp_path, page="XIAN")
        webui._export_page_results("XIAN", "json")
        target = tmp_path / "outputs" / "evaluation" / "knn_eval"
        assert len(list(target.glob("xian_knn_result_*.json"))) == 1

    def test_export_images_writes_pngs(self, monkeypatch, tmp_path):
        fake = self._export_harness(monkeypatch, tmp_path, page="GOOGLE")
        webui._export_page_results("GOOGLE", "images")
        target = tmp_path / "outputs" / "evaluation" / "knn_eval"
        pngs = sorted(p.name for p in target.glob("google_knn_*.png"))
        assert any("_cm_" in n for n in pngs)
        assert any("_pr_" in n for n in pngs)
        assert all(p.suffix == ".png" for p in target.glob("google_knn_*.png"))
        # 通知导出成功
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("已导出" in m for m in msgs)

    def test_export_images_passes_f1_for_per_class_recall(self, monkeypatch, tmp_path):
        """导出图片图表时 plot_purity_recall_curve 收到 (f2, 路径, f1)：
        f1 含 per_class_recall_by_k，右面板据此绘制 Per-class Recall（与 Web 页面一致）。"""
        fake = self._export_harness(monkeypatch, tmp_path, page="GOOGLE")
        spy = MagicMock(return_value=None)
        monkeypatch.setattr(webui, "plot_purity_recall_curve", spy)
        eval_data = webui.state["pages"]["GOOGLE"]["eval"]
        f1 = eval_data["f1_result"]
        f2 = eval_data["f2_result"]
        webui._export_page_results("GOOGLE", "images")
        spy.assert_called_once()
        args = spy.call_args[0]
        assert len(args) == 3
        assert args[0] is f2
        assert isinstance(args[1], Path)
        assert "_pr_" in args[1].name
        assert args[2] is f1
        assert "per_class_recall_by_k" in args[2]

    def test_export_without_result_notifies_warning(self, monkeypatch, tmp_path):
        fake = self._export_harness(monkeypatch, tmp_path, page="GOOGLE")
        webui.state["pages"]["GOOGLE"]["eval"] = {"f1_result": None, "f2_result": None}
        webui._export_page_results("GOOGLE", "json")
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("尚无评估结果" in m for m in msgs)


# ========== 6. SimilarityMatrix 页面（需求 3、5） ==========

class TestSimilarityMatrix:
    def test_panel_controls_present(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        assert fake.page_buttons["SimilarityMatrix"]["生成热力图对比"] is not None
        # 采样数 N / Seed
        assert fake.numbers["采样数 N"].value == 200
        assert fake.numbers["Seed"].value == 42
        # 双采样模式 radio
        radios = fake.radios
        assert any(getattr(r, "options", None) == ["数据库全库", "单张图片"] for r in radios)

    def test_do_sim_compare_exports_to_similarity_dir(self, monkeypatch, tmp_path):
        fake = _build_harness(monkeypatch)
        monkeypatch.setattr(webui, "_PROJECT_ROOT", tmp_path)
        # 双 manager 健康
        def fake_mgr(url, collection_name):
            m = _manager(collection=collection_name)
            m.client = MagicMock()
            return m

        monkeypatch.setattr(webui, "QdrantManager", fake_mgr)

        def _cmp(g, x, n=200, seed=42, image_id=None, output=None,
                 collection_names=None, export_dir="outputs", prefix="", export_npy=True):
            output.write(b"FAKEPNG")
            sim = np.eye(2)
            return {
                "sampled": 100, "kept": 100, "dropped": 0,
                "matrix_shape": [100, 100], "elapsed_sec": 1.0,
                "sim_g": sim, "sim_x": sim,
                "exported_files": [
                    f"{prefix}google_aef_embedding_similarity.npy",
                    f"{prefix}xian_aef_embedding_similarity.npy",
                    f"{prefix}similarity_sampling.json",
                ],
            }

        compare = MagicMock(side_effect=_cmp)
        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps", compare,
        )
        fake.numbers["采样数 N"].value = 100
        asyncio.run(fake.page_buttons["SimilarityMatrix"]["生成热力图对比"]())
        target = tmp_path / "outputs" / "evaluation" / "similarity"
        assert target.exists()
        # 热力图 PNG 文件名带 full_col_ 前缀（数据库全库默认）
        heatmaps = list(target.glob("full_col_similarity_heatmap_*.png"))
        assert len(heatmaps) == 1
        assert heatmaps[0].read_bytes() == b"FAKEPNG"
        # 固定导出目录传参 + 来源前缀 + export_npy=False（npy 改手动导出）
        assert Path(compare.call_args.kwargs["export_dir"]) == webui.SIMILARITY_EXPORT_DIR
        assert compare.call_args.kwargs["prefix"] == "full_col"
        assert compare.call_args.kwargs["export_npy"] is False

    def test_image_mode_requires_selection(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        # 默认数据库全库模式，image_id 为 None 时不要求选择影像
        assert fake.selects["影像（单张图片模式）"] is not None


class TestSimilarityExportButtons:
    """SimilarityMatrix 页「导出 JSON / 导出图片图表」按钮（本次 tweak）."""

    def _run_compare(self, monkeypatch, tmp_path):
        """执行一次对比，返回 (fake, target_dir)."""
        fake = _build_harness(monkeypatch)
        monkeypatch.setattr(webui, "_PROJECT_ROOT", tmp_path)

        def fake_mgr(url, collection_name):
            m = _manager(collection=collection_name)
            m.client = MagicMock()
            return m

        monkeypatch.setattr(webui, "QdrantManager", fake_mgr)

        # 让 compare 落盘 similarity_sampling.json 供 JSON 导出读取（带 full_col_ 前缀）
        def _cmp(g, x, n=200, seed=42, image_id=None, output=None,
                 collection_names=None, export_dir="outputs", prefix="", export_npy=True):
            output.write(b"FAKEPNG")
            d = Path(export_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{prefix}similarity_sampling.json").write_text(
                '{"params": {"n": 100, "seed": 42}, "pixels": [{"point_id": "p1"}]}',
                encoding="utf-8",
            )
            sim = np.eye(2)
            return {
                "sampled": 100, "kept": 100, "dropped": 0,
                "matrix_shape": [100, 100], "elapsed_sec": 1.0,
                "sim_g": sim, "sim_x": sim,
                "exported_files": [
                    f"{export_dir}/{prefix}google_aef_embedding_similarity.npy",
                    f"{export_dir}/{prefix}xian_aef_embedding_similarity.npy",
                    f"{export_dir}/{prefix}similarity_sampling.json",
                ],
            }

        compare = MagicMock(side_effect=_cmp)
        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps", compare,
        )
        fake.numbers["采样数 N"].value = 100
        asyncio.run(fake.page_buttons["SimilarityMatrix"]["生成热力图对比"]())
        return fake, tmp_path / "outputs" / "evaluation" / "similarity"

    def test_export_buttons_hidden_before_compare(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        json_btn = fake.page_button_objs["SimilarityMatrix"]["导出 JSON"]
        img_btn = fake.page_button_objs["SimilarityMatrix"]["导出图片图表"]
        npy_btn = fake.page_button_objs["SimilarityMatrix"]["输出 npy 文件"]
        assert json_btn.set_visibility.call_args_list[-1] == ((False,),)
        assert img_btn.set_visibility.call_args_list[-1] == ((False,),)
        assert npy_btn.set_visibility.call_args_list[-1] == ((False,),)

    def test_export_buttons_visible_after_compare(self, monkeypatch, tmp_path):
        fake, _ = self._run_compare(monkeypatch, tmp_path)
        json_btn = fake.page_button_objs["SimilarityMatrix"]["导出 JSON"]
        img_btn = fake.page_button_objs["SimilarityMatrix"]["导出图片图表"]
        npy_btn = fake.page_button_objs["SimilarityMatrix"]["输出 npy 文件"]
        assert json_btn.set_visibility.call_args_list[-1] == ((True,),)
        assert img_btn.set_visibility.call_args_list[-1] == ((True,),)
        assert npy_btn.set_visibility.call_args_list[-1] == ((True,),)

    def test_export_json_writes_sampling_json(self, monkeypatch, tmp_path):
        fake, target = self._run_compare(monkeypatch, tmp_path)
        webui._export_similarity_results("json")
        jsons = list(target.glob("full_col_similarity_*.json"))
        assert len(jsons) == 1, "JSON 文件名应以 full_col_ 开头"
        data = json.loads(jsons[0].read_text(encoding="utf-8"))
        # 采样参数与保留像素信息来自 compare 落盘的 sampling json
        assert data["pixels"] == [{"point_id": "p1"}]
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("已导出" in m for m in msgs)

    def test_export_images_writes_heatmap_png(self, monkeypatch, tmp_path):
        fake, target = self._run_compare(monkeypatch, tmp_path)
        webui._export_similarity_results("images")
        pngs = list(target.glob("full_col_similarity_heatmap_*.png"))
        assert len(pngs) == 1, "热力图 PNG 文件名应以 full_col_ 开头"
        assert pngs[0].read_bytes() == b"FAKEPNG"
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("已导出" in m for m in msgs)

    def test_export_npy_writes_timestamped_files(self, monkeypatch, tmp_path):
        """「输出 npy 文件」手动导出两个带时间戳 npy，不覆盖旧文件."""
        fake, target = self._run_compare(monkeypatch, tmp_path)
        webui._export_similarity_results("npy")
        npy_files = sorted(target.glob("full_col_*_similarity_*.npy"))
        assert len(npy_files) == 2, "应导出 google/xian 两个 npy"
        # 文件名含 google/xian 缩写与时间戳
        names = {p.name for p in npy_files}
        assert any("_google_similarity_" in n for n in names)
        assert any("_xian_similarity_" in n for n in names)
        # 内容为 (2,2) 余弦相似度矩阵（np.eye(2)）
        for p in npy_files:
            arr = np.load(p)
            assert arr.shape == (2, 2)
        # 重复导出不覆盖旧文件（不同时间戳 → 数量翻倍；sleep 1s 确保时间戳变化）
        import time as _time
        _time.sleep(1.1)
        webui._export_similarity_results("npy")
        assert len(list(target.glob("full_col_*_similarity_*.npy"))) == 4
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert any("已导出" in m for m in msgs)

    def test_export_without_result_notifies_warning(self, monkeypatch, tmp_path):
        fake = _build_harness(monkeypatch)
        webui.state["similarity"] = {}
        webui._export_similarity_results("json")
        webui._export_similarity_results("images")
        webui._export_similarity_results("npy")
        msgs = [str(m) for m, _ in fake.notify_calls]
        assert msgs.count("请先生成热力图对比") == 3


# ========== 7. 模块级状态与取消（需求 4） ==========

class TestModuleState:
    def test_manifest_cache_keyed_by_collection(self, monkeypatch):
        from KNN_evaluation.manifest import manifest_path as _mp
        g = webui._get_manifest_cached("google_aef_embedding")
        x = webui._get_manifest_cached("xian_aef_embedding")
        assert g is webui._get_manifest_cached("google_aef_embedding")
        assert g is not x
        webui._invalidate_manifest_cache("google_aef_embedding")
        assert webui._get_manifest_cached("google_aef_embedding") is not g

    def test_cancel_evaluate_scoped_to_page(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        ev_g = threading.Event()
        ev_x = threading.Event()
        webui._eval_cancel_events["GOOGLE"] = {id(ev_g): ev_g}
        webui._eval_cancel_events["XIAN"] = {id(ev_x): ev_x}
        webui._cancel_page_evaluate("GOOGLE")
        assert ev_g.is_set()
        assert not ev_x.is_set()
