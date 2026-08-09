"""Qdrant Linear Probe 评估系统 — NiceGUI Web 训练监测界面.

提供 MLP_label（64 维像素 embedding → 9 类 DW 硬标签的线性分类器）
训练过程的实时监测：Qdrant 数据状态、训练参数配置、分层采样进度、
训练进度条、逐 epoch 更新的 loss/accuracy 曲线、最终指标与混淆矩阵、
结果 JSON 导出。视觉风格与 KNN_evaluation/webui.py 保持一致。

启动方式:
    cd D:\\Project\\光机所项目\\evaluation_scripts
    uv run python LinearProbe_evaluation/webui.py --port 8004
"""
import asyncio
import json as _json
import sys
import threading
from asyncio import sleep as asyncio_sleep
from asyncio import to_thread as asyncio_to_thread
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from nicegui import ui, core

# 确保项目根目录在 path 中（同 KNN_evaluation/webui.py 的路径处理：
# 直接脚本运行时把脚本目录移出 sys.path，避免本地 qdrant_client 模块
# 与 pip 包 qdrant_client 冲突）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _SCRIPT_DIR]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------- 包导入 ----------
from LinearProbe_evaluation.config import (
    QDRANT_URL, DEFAULT_COLLECTION, PRESET_COLLECTIONS, NUM_CLASSES,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_OPTIMIZER,
    DEFAULT_TRAIN_PER_CLASS, DEFAULT_VAL_PER_CLASS, DEFAULT_VAL_RATIO, DEFAULT_SEED,
)
from LinearProbe_evaluation.label_mapping import LABEL_NAMES
from LinearProbe_evaluation.qdrant_client import QdrantManager
from LinearProbe_evaluation.dataset import stratified_train_val_split, CancelledError as DatasetCancelled
from LinearProbe_evaluation.trainer import (
    TrainConfig, train_mlp, load_mlp_label, predict_labels,
    CancelledError as TrainCancelled,
)
from LinearProbe_evaluation.metrics import classification_report, confusion_matrix
from LinearProbe_evaluation.visualization import (
    training_curves_base64, confusion_matrix_base64,
    plot_training_curves, plot_confusion_matrix,
)
from LinearProbe_evaluation.model import VARIANT_HIDDEN_DIMS, VARIANT_MLP, build_model

# ---------- 默认配置 ----------
DEFAULT_PORT = 8004

# 训练结果导出目录（需求：JSON 与图像图表均落盘到该目录，目录不存在时自动创建）
LP_EXPORT_DIR = Path("outputs/evaluation/linearprobe")

# 模型结构预置信息（从 model 模块派生，避免硬编码漂移）
_ARCH_INFO: dict[str, dict] = {
    v: {
        "hidden_dims": list(dims),
        "num_parameters": sum(p.numel() for p in build_model(v).parameters()),
    }
    for v, dims in VARIANT_HIDDEN_DIMS.items()
}

# ---------- 模块级状态 / 钩子（同 KNN webui Task 2.1 模式） ----------
_CLI_QDRANT_URL: str = QDRANT_URL

# 当前 collection 状态：与 _CLI_QDRANT_URL 同模式存模块级。
# state 字典在 index() 每次请求重建，collection 选择必须存模块级避免切换丢失。
_current_collection: str = DEFAULT_COLLECTION
# 会话内已知 collection 列表（预置 + 自定义），用于选择器渲染与 localStorage 校验
_known_collections: list[str] = list(PRESET_COLLECTIONS)
_LOCALSTORAGE_KEY = "comet.linearprobe.current_collection"

# 导出文件名中 collection 缩写映射（预置用 google/xian，自定义用原名）
_COLLECTION_SHORT = {
    "google_aef_embedding": "google",
    "xian_aef_embedding": "xian",
}


def collection_short_name(collection: str) -> str:
    """导出文件名使用的 collection 缩写：预置 google/xian，自定义原名."""
    return _COLLECTION_SHORT.get(collection, collection)

state: dict = {}

# 训练取消令牌注册表：do_train 每次训练注册 threading.Event，
# 「中止训练」按钮经 _cancel_current_train 设置事件（协作式取消）。
_train_cancel_events: dict[int, "threading.Event"] = {}
# 闭包注入：页面构建时记录中止按钮控件，供模块级 helper 切换可见性。
_set_train_cancel_btn_visible: object | None = None


def _cancel_current_train() -> None:
    """设置当前训练任务的取消事件（按钮 on_click 处理器）."""
    for ev in _train_cancel_events.values():
        ev.set()
    if _set_train_cancel_btn_visible is not None:
        _set_train_cancel_btn_visible(False)


def _resolve_stored_collection(stored: str | None) -> str:
    """localStorage 记录值有效则恢复，否则回退默认（spec：记录失效回退）."""
    if stored and stored in _known_collections:
        return stored
    return DEFAULT_COLLECTION


def _persist_collection_choice(collection: str) -> None:
    """将当前选择写回 localStorage（fire-and-forget，不 await）."""
    try:
        if core.loop is None:
            return  # 无运行中的 app 事件循环（测试/脚本环境）→ 无浏览器可执行 JS
        ui.run_javascript(
            f"localStorage.setItem({_json.dumps(_LOCALSTORAGE_KEY)}, {_json.dumps(collection)})",
        )
        # fire-and-forget：返回值（AwaitableResponse）直接丢弃，不检测协程/不派发任务
    except Exception:
        # 页面/循环不可用或 JS 不可执行时静默失败（记忆功能尽力而为）
        pass


async def _apply_collection(new_col: str) -> None:
    """切换当前 collection：更新模块级状态、重建 manager、持久化.

    关键点：`state["manager"]` 必须替换为新的 manager 实例，否则旧 collection
    句柄残留导致数据串扰；`_current_collection` 存模块级，避免 index() 重建 state 丢失。
    """
    global _current_collection
    if new_col == _current_collection:
        return
    _current_collection = new_col
    state["manager"] = QdrantManager(url=_CLI_QDRANT_URL, collection_name=_current_collection)
    _persist_collection_choice(_current_collection)


async def _add_custom_collection(name: str) -> None:
    """自定义 collection：校验名称并加入已知列表，随后切换.

    校验：非空、不含路径分隔符（/ 或 \\）。加入后切换到新 collection.
    """
    name = (name or "").strip()
    if not name:
        ui.notify("collection 名称不能为空", type="warning")
        return
    if "/" in name or "\\" in name:
        ui.notify("collection 名称不能包含路径分隔符 / 或 \\", type="negative")
        return
    if name not in _known_collections:
        _known_collections.append(name)
    await _apply_collection(name)


async def _restore_stored_collection() -> None:
    """页面加载：读取 localStorage 记录并恢复（记录失效回退默认）."""
    try:
        stored = await ui.run_javascript(
            f"localStorage.getItem({_json.dumps(_LOCALSTORAGE_KEY)}) || ''",
        )
    except Exception:
        stored = ""
    target = _resolve_stored_collection(stored)
    if target != _current_collection:
        await _apply_collection(target)
    else:
        # 记录与当前一致：仍须确保 manager 已就绪（快速路径可能尚未执行）
        mgr = state.get("manager")
        if mgr is None or mgr.collection_name != _current_collection:
            state["manager"] = QdrantManager(
                url=_CLI_QDRANT_URL, collection_name=_current_collection,
            )


# ---------- 页面构建 ----------

@ui.page("/")
def index():
    """主页面：Qdrant Linear Probe 评估系统（MLP_label 训练监测）."""
    global state
    state = {
        "manager": None,
        "history": [],       # 训练过程中每个 epoch 的指标（驱动实时曲线）
        "training": False,   # 并发训练保护：同一时间只允许一个训练任务
        "lp_export": None,   # 最近一次训练导出数据（result/variant/confusion_cm），供导出按钮使用
    }

    ui.page_title("Qdrant Linear Probe 评估系统 — MLP_label")

    # ===== HEADER =====
    with ui.header(elevated=True).classes("bg-primary text-white"):
        ui.label("Qdrant Linear Probe 评估系统 — MLP_label 训练监测").classes("text-h4")

    # ===== Qdrant 连接 & Collection 状态 =====
    with ui.expansion("Qdrant 连接 & Collection 状态", value=True).classes("w-full mt-4"):
        status_row = ui.row().classes("items-center gap-4")
        status_label = ui.label("⏳ 正在连接...").classes("text-sm")
        info_row = ui.row().classes("items-center gap-4 mt-2")
        info_label = ui.label("").classes("text-sm text-grey")

        def refresh_status():
            if state["manager"] is None:
                return
            try:
                if not state["manager"].health_check():
                    status_label.set_text("❌ Qdrant 不可达")
                    info_label.set_text(f"url={state['manager'].url}  docker: qdrant")
                    return
                if not state["manager"].collection_exists():
                    status_label.set_text("✅ Qdrant 可达 | ⚠️ Collection 不存在")
                    info_label.set_text(
                        f"collection={_current_collection} 尚未创建，请先执行 KNN import"
                    )
                    return
                info = state["manager"].collection_info()
                status_label.set_text("✅ Qdrant 可达 | Collection 就绪")
                info_label.set_text(
                    f"collection={_current_collection} | total_points={info['total_points']:,} "
                    f"| 向量维度={info.get('vectors_count', '?')} | status={info['status']}"
                )
            except Exception as e:
                status_label.set_text("❌ 状态查询失败")
                info_label.set_text(str(e))

        def _ensure_manager():
            if state["manager"] is None or not state["manager"].health_check():
                state["manager"] = QdrantManager(url=_CLI_QDRANT_URL, collection_name=_current_collection)
            return state["manager"]

        ui.button("刷新状态", on_click=refresh_status).props("flat dense size=sm")
        _ensure_manager()
        refresh_status()

        # ---- Collection 选择器：预置 + 自定义；修改下拉后须点击「切换 collection」生效 ----
        with ui.row().classes("items-center gap-4 mt-2"):
            collection_select = ui.select(
                {c: c for c in _known_collections},
                value=_current_collection,
                label="Qdrant Collection",
            ).classes("w-64")

            async def _switch_collection():
                """点击「切换 collection」后才应用下拉所选 collection.

                下拉仅更新选中值，不即时切换；切换后 manager 重建，后续训练
                使用所选 collection（do_train 经 state['manager'] 取数）。
                """
                new_col = collection_select.value
                if new_col and new_col != _current_collection:
                    await _apply_collection(new_col)
                    refresh_status()
                    ui.notify(f"已切换到 Collection {new_col}", type="positive")
                else:
                    ui.notify(f"当前已是 Collection {_current_collection}", type="info")

            ui.button(
                "切换 collection",
                on_click=_switch_collection,
            ).props("flat dense size=sm")

            custom_col_input = ui.input(
                label="自定义 collection 名称", placeholder="如 my_embedding",
            ).classes("w-56")
            ui.button(
                "使用自定义 Collection",
                on_click=lambda: asyncio.create_task(
                    _add_custom_collection(custom_col_input.value)
                ),
            ).props("flat dense size=sm")

        # localStorage 记忆恢复：快速路径稍后执行，避免与 collection 切换竞争
        ui.timer(0.2, callback=lambda: asyncio.create_task(_restore_stored_collection()), once=True)

    # ===== MLP_label 训练 =====
    with ui.expansion("MLP_label 训练（64 维 embedding → 9 类硬标签）", value=True).classes("w-full mt-4"):
        train_state = {
            "result": None,
            "val_ds": None,
        }

        # 参数说明区
        with ui.expansion("参数说明", value=False).classes("w-full mt-1"):
            ui.markdown("""
**MLP_label 网络结构（web 端「模型结构」下拉可选两种）：**
- 输入：**64 维**像素 embedding（与 Qdrant collection 的向量一致，1 行 = 1 个像素）。
- 输出：**9 维** logits，对应 9 个 DW 土地覆盖标签的**硬分类独热码**（water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice）。
- **`MLP`（默认，LeNet-5 规模，约 15 万参数）**：`Linear(64,256) → ReLU → Dropout(0.2) → Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,9)`，可拟合 embedding 中的非线性，追求更高精度。
- **`Linear Probe`（标准线性探测，585 参数）**：单一 `Linear(64,9)`，无隐藏层、无激活，学术惯例（Alain & Bengio 2016 等），用于检测 embedding 的**线性可分性**、纵向对比不同 embedding 版本。
- 损失：CrossEntropyLoss（真实标签为 0-8 硬分类，预测取 logits 的 argmax）。

**采样参数：**
- **每类训练样本数 (`train_per_class`)**：从 Qdrant 中每类标签分别随机采样的训练样本数。DW 数据严重不平衡（trees ≈ 621 万 vs snow_and_ice ≈ 240），分层采样保证稀有类不被淹没。0 表示该类全取。
- **每类验证样本数 (`val_per_class`)**：每类最多进入验证集的样本数。先按 `val_ratio` 从每类切出验证集，再受此上限约束 —— 稀有类也能保证验证集非空。
- **`val_ratio`**：每类中先按该比例切出验证集（默认 0.2）。
- **随机种子 (`seed`)**：采样与权重初始化种子，相同参数完全可复现。

**训练参数：**
- **`epochs`**：完整遍历训练集的轮数。
- **`batch_size`**：每批样本数（默认 2048，GPU 上建议 2048-8192）。
- **`lr`**：学习率（Adam 默认 1e-3）。
- **优化器**：Adam（默认）或 SGD（momentum=0.9）。
- **训练设备**：固定 **CUDA**（本模块已禁用 CPU fallback）——CUDA 不可用时直接报错，不会静默回退到 CPU。
""").classes("text-xs text-grey")

        # 参数配置区（默认值统一来自 config.DEFAULT_*，改 config 即生效）
        with ui.row().classes("items-center gap-4 mt-2"):
            epochs_input = ui.number(label="Epochs", value=DEFAULT_EPOCHS, min=1, max=1000).classes("w-24")
            batch_input = ui.number(label="Batch Size", value=DEFAULT_BATCH_SIZE, min=16, max=65536).classes("w-28")
            lr_input = ui.number(label="Learning Rate", value=DEFAULT_LR, min=1e-6, max=1.0,
                                 step=0.0001, format="%.5f").classes("w-32")
            optimizer_select = ui.select(["adam", "sgd"], value=DEFAULT_OPTIMIZER, label="优化器").classes("w-28")
            structure_select = ui.select(
                {
                    "mlp": "MLP（LeNet-5 规模，15 万参数）",
                    "linear": "Linear Probe（单层，585 参数）",
                },
                value=VARIANT_MLP,
                label="模型结构",
            ).classes("w-64")
        with ui.row().classes("items-center gap-4 mt-2"):
            train_pc_input = ui.number(label="每类训练样本数", value=DEFAULT_TRAIN_PER_CLASS, min=0, max=1000000).classes("w-36")
            val_pc_input = ui.number(label="每类验证样本数", value=DEFAULT_VAL_PER_CLASS, min=0, max=100000).classes("w-36")
            val_ratio_input = ui.number(label="验证比例", value=DEFAULT_VAL_RATIO, min=0.0, max=1.0, step=0.05).classes("w-24")
            seed_input = ui.number(label="Seed", value=DEFAULT_SEED, min=0, max=2**31).classes("w-24")

        # 训练设备固定为 CUDA（本模块已禁用 CPU fallback，见 trainer.resolve_device）
        with ui.row().classes("items-center gap-4 mt-2"):
            gpu_info_label = ui.label("").classes("text-sm text-grey")
            try:
                gpu_info_label.set_text(f"训练设备: CUDA — {torch.cuda.get_device_name(0)}")
            except Exception:
                gpu_info_label.set_text("训练设备: CUDA（未检测到 GPU，训练将报错）")

        # 进度区
        with ui.row().classes("items-center gap-4 mt-2"):
            train_progress_bar = ui.linear_progress(value=0).classes("w-96")
            train_progress_bar.set_visibility(False)
        train_progress_label = ui.label("").classes("text-sm text-grey mt-2")
        train_progress_label.set_visibility(False)

        # 实时曲线区
        curves_label = ui.label("训练曲线").classes("text-subtitle1 mt-4")
        curves_image = ui.image().classes("w-full max-w-4xl")
        curves_image.set_visibility(False)

        # 结果区
        result_container = ui.column().classes("w-full mt-4")
        result_container.set_visibility(False)

        export_btn = ui.button("导出 JSON", on_click=lambda: _export_lp_results("json"))
        export_btn.set_visibility(False)
        export_img_btn = ui.button("导出图片图表", on_click=lambda: _export_lp_results("images"))
        export_img_btn.set_visibility(False)

        # ---------- 训练流程 ----------
        async def do_train():
            if state.get("training"):
                ui.notify("已有训练任务进行中，请等待其完成或中止", type="warning")
                return
            if state["manager"] is None or not state["manager"].health_check():
                ui.notify("Qdrant 不可达", type="negative")
                return
            if not state["manager"].collection_exists():
                ui.notify(f"Collection '{_current_collection}' 不存在，请先执行 KNN import", type="negative")
                return
            info = state["manager"].collection_info()
            if info.get("total_points", 0) == 0:
                ui.notify("Collection 为空", type="negative")
                return

            manager = state["manager"]
            config = TrainConfig(
                epochs=int(epochs_input.value),
                batch_size=int(batch_input.value),
                lr=float(lr_input.value),
                optimizer=optimizer_select.value,
                model_variant=structure_select.value,
                seed=int(seed_input.value),
            )
            train_pc = int(train_pc_input.value)
            val_pc = int(val_pc_input.value)
            val_ratio = float(val_ratio_input.value)

            # 取消令牌：本次训练注册 threading.Event，中止按钮经
            # _cancel_current_train 设置；asyncio.to_thread 无法真正杀死线程，
            # 用采样/训练循环的协作式取消检查。
            cancel_event = threading.Event()
            _train_cancel_events[id(cancel_event)] = cancel_event
            _set_train_cancel_btn_visible(True)
            state["training"] = True

            train_progress_label.set_visibility(True)
            train_progress_bar.set_visibility(True)
            train_progress_bar.value = 0
            result_container.set_visibility(False)
            result_container.clear()
            export_btn.set_visibility(False)
            state["history"] = []

            try:
                # ---- 阶段 1：分层采样（CPU 密集：读采样地图 + 组装数据，与训练设备无关） ----
                train_progress_label.set_text(
                    "阶段 1/2 分层采样中...（CPU 密集，正在加载采样地图与数据）"
                )
                await asyncio_sleep(0.05)

                def sample_progress(label_id, n_train, n_val):
                    name = LABEL_NAMES.get(label_id, "?")
                    train_progress_label.set_text(
                        f"阶段 1/2 采样中... 类别 {label_id} {name}: train {n_train} / val {n_val}"
                    )
                    train_progress_bar.value = (label_id + 1) / NUM_CLASSES

                try:
                    train_ds, val_ds = await asyncio_to_thread(
                        stratified_train_val_split, manager,
                        train_per_class=train_pc,
                        val_per_class=val_pc,
                        val_ratio=val_ratio,
                        seed=config.seed,
                        progress_callback=sample_progress,
                        cancel_event=cancel_event,
                    )
                except DatasetCancelled:
                    ui.notify("采样已取消", type="warning")
                    return
                except Exception as e:
                    ui.notify(f"采样失败: {e}", type="negative")
                    return
                if cancel_event.is_set():
                    ui.notify("训练已取消", type="warning")
                    return
                if train_ds.size == 0:
                    ui.notify("训练集为空，请检查采样参数", type="negative")
                    return

                train_progress_label.set_text(
                    f"阶段 1/2 采样完成: 训练集 {train_ds.size:,} | 验证集 {val_ds.size:,}，"
                    f"进入阶段 2/2 CUDA 训练（GPU: {torch.cuda.get_device_name(0)}）..."
                )
                train_progress_bar.value = 0
                await asyncio_sleep(0.05)

                # ---- 阶段 2：训练（逐 epoch 更新进度与实时曲线） ----
                total_epochs = config.epochs

                def train_progress(event: dict):
                    if event["event"] == "epoch_start":
                        train_progress_label.set_text(
                            f"Epoch {event['epoch']}/{event['total_epochs']} 训练中..."
                        )
                    elif event["event"] == "batch":
                        frac = (
                            (event["epoch"] - 1) + event["batch"] / max(event["total_batches"], 1)
                        ) / max(total_epochs, 1)
                        train_progress_bar.value = min(frac, 1.0)
                        train_progress_label.set_text(
                            f"Epoch {event['epoch']}/{event['total_epochs']} | "
                            f"batch {event['batch']}/{event['total_batches']} | loss={event['loss']:.4f}"
                        )
                    elif event["event"] == "epoch_end":
                        train_progress_bar.value = event["epoch"] / max(total_epochs, 1)
                        state["history"].append(event)
                        # 空验证集时 val_* 为 None，用 0.0 占位避免格式化崩溃
                        val_acc = event["val_acc"] if event["val_acc"] is not None else 0.0
                        val_mf1 = event["val_macro_f1"] if event["val_macro_f1"] is not None else 0.0
                        train_progress_label.set_text(
                            f"Epoch {event['epoch']}/{event['total_epochs']} | "
                            f"train_loss={event['train_loss']:.4f} train_acc={event['train_acc']:.4f} | "
                            f"val_acc={val_acc:.4f} val_macro_f1={val_mf1:.4f}"
                        )
                        # 实时曲线：重绘到当前 epoch（PNG → base64 → ui.image）
                        curves_image.set_source(training_curves_base64(state["history"]))
                        curves_image.set_visibility(True)

                try:
                    result = await asyncio_to_thread(
                        train_mlp, train_ds, val_ds, config,
                        progress_callback=train_progress,
                        cancel_event=cancel_event,
                    )
                except TrainCancelled:
                    ui.notify("训练已取消", type="warning")
                    return
                except Exception as e:
                    ui.notify(f"训练失败: {e}", type="negative")
                    return

                train_state["result"] = result
                train_state["val_ds"] = val_ds

                # ---- 阶段 3：展示结果 ----
                train_progress_label.set_text(
                    f"训练完成（{result.elapsed_seconds:.1f}s, GPU: {result.device_name}），正在生成评估报告..."
                )
                await asyncio_sleep(0.05)
                await _show_train_results(train_ds, val_ds, result)
                # 保存导出数据（result + 模型结构 + 混淆矩阵），供导出按钮使用
                state["lp_export"] = {
                    "result": result,
                    "variant": structure_select.value,
                    "confusion_cm": train_state.get("confusion_cm"),
                }
                export_btn.set_visibility(True)
                export_img_btn.set_visibility(True)
            finally:
                # 无论成功/失败/取消，复位进度 UI、隐藏中止按钮并清理取消令牌
                state["training"] = False
                train_progress_label.set_visibility(False)
                train_progress_bar.set_visibility(False)
                _set_train_cancel_btn_visible(False)
                _train_cancel_events.pop(id(cancel_event), None)

        async def _show_train_results(train_ds, val_ds, result):
            """展示最终指标、per-class 表与混淆矩阵."""
            result_container.clear()
            result_container.set_visibility(True)

            with result_container:
                ui.label("训练结果").classes("text-h6")
                ui.label(
                    f"Best epoch: {result.best_epoch} | val_acc={result.best_val_accuracy:.4f} | "
                    f"val_macro_f1={result.best_val_macro_f1:.4f} | "
                    f"GPU: {result.device_name} | GPU kernel {result.gpu_kernel_seconds:.3f}s | "
                    f"总耗时 {result.elapsed_seconds:.1f}s"
                ).classes("text-sm")

                if result.val_report:
                    rep = result.val_report
                    with ui.row().classes("items-center gap-4 mt-1"):
                        ui.label(f"Accuracy: {rep['accuracy']:.4f}").classes("text-md")
                        ui.label(f"Macro-F1: {rep['macro_f1']:.4f}").classes("text-md")
                        ui.label(f"Weighted-F1: {rep['weighted_f1']:.4f}").classes("text-md")

                    rows = [
                        {
                            "label": f"{lid} {LABEL_NAMES.get(int(lid), '?')}",
                            "precision": f"{m['precision']:.4f}",
                            "recall": f"{m['recall']:.4f}",
                            "f1": f"{m['f1']:.4f}",
                            "support": m["support"],
                        }
                        for lid, m in rep["per_class"].items()
                    ]
                    ui.table(
                        columns=[
                            {"name": "label", "label": "类别", "field": "label", "align": "left",
                             "style": "white-space: normal; word-break: break-word; min-width: 200px;",
                             "headerStyle": "white-space: normal; word-break: break-word;"},
                            {"name": "precision", "label": "Precision", "field": "precision", "align": "left",
                             "style": "min-width: 110px;", "headerStyle": "min-width: 110px;"},
                            {"name": "recall", "label": "Recall", "field": "recall", "align": "left",
                             "style": "min-width: 110px;", "headerStyle": "min-width: 110px;"},
                            {"name": "f1", "label": "F1", "field": "f1", "align": "left",
                             "style": "min-width: 110px;", "headerStyle": "min-width: 110px;"},
                            {"name": "support", "label": "Support", "field": "support", "align": "left",
                             "style": "min-width: 110px;", "headerStyle": "min-width: 110px;"},
                        ],
                        rows=rows,
                        row_key="label",
                    ).classes("w-full mt-2")

                # 混淆矩阵：用 best（或 final）checkpoint 在验证集上推理
                ckpt_path = result.best_checkpoint or result.final_checkpoint
                if ckpt_path is not None and val_ds.size > 0:
                    ui.label("混淆矩阵（验证集）").classes("text-subtitle1 mt-3")
                    cm_img = ui.image().classes("w-full max-w-4xl")
                    cm_img.set_visibility(False)
                    try:
                        cm, cm_uri = await asyncio_to_thread(
                            _compute_confusion_uri, ckpt_path, val_ds, result.device,
                        )
                        train_state["confusion_cm"] = cm  # 供导出图像图表使用
                        cm_img.set_source(cm_uri)
                        cm_img.set_visibility(True)
                    except Exception as e:
                        train_state["confusion_cm"] = None
                        ui.notify(f"混淆矩阵生成失败: {e}", type="warning")

        with ui.row().classes("gap-2 mt-4"):
            ui.button("开始训练", on_click=do_train).props("flat")

            # 中止训练：仅训练运行时可见/可用（do_train 在开始/结束时切换可见性）
            train_cancel_btn = ui.button("中止训练", on_click=_cancel_current_train).props("flat")
            train_cancel_btn.set_visibility(False)

            def _set_cancel_btn_visible(visible: bool):
                train_cancel_btn.set_visibility(visible)
            global _set_train_cancel_btn_visible
            _set_train_cancel_btn_visible = _set_cancel_btn_visible

    # ===== 底部说明 =====
    with ui.expansion("使用说明", value=False).classes("w-full mt-4"):
        ui.markdown("""
**MLP_label（两种结构可选，web 端「模型结构」下拉切换）：**

- 输入：Qdrant 所选 collection 中每个像素的 **64 维 embedding**；
- 输出：**9 维** logits，对应 9 个 DW 标签的硬分类独热码；
- **MLP**（默认）：`Linear(64,256) → ReLU → Dropout(0.2) → Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,9)`（约 15 万参数）；
- **Linear Probe**：单一 `Linear(64,9)`（585 参数，标准线性探测，测 embedding 线性可分性）；
- 损失均为 CrossEntropyLoss（真实标签 0-8）。

**训练数据来源：** 复用 KNN 的采样地图（`qdrant_sampling_map.json`）分层随机采样，
只下载样本量（MB 级），避免全量下载 2.6GB 向量。类别严重不平衡
（trees ≈ 621 万 vs snow_and_ice ≈ 240），分层采样保证稀有类参与训练。

**后续扩展：** 若引入像素 DW 数据集的 prob（软分类概率）信息，将新增
`MLPProb`（同为 64→9，输出概率分布，软标签训练），本界面结构可复用。

启动命令:
```
uv run python LinearProbe_evaluation/webui.py --port 8004
```
""").classes("text-xs text-grey")


def _compute_confusion_uri(ckpt_path, val_ds, device) -> tuple[np.ndarray, str]:
    """用 checkpoint 在验证集上推理，返回 (混淆矩阵, base64 图 URI)."""
    model, _ = load_mlp_label(ckpt_path, device)
    pred = predict_labels(model, val_ds.X, device)
    cm = confusion_matrix(val_ds.y, pred, NUM_CLASSES)
    uri = confusion_matrix_base64(cm, [LABEL_NAMES[i] for i in range(NUM_CLASSES)])
    return cm, uri


def _export_lp_results(kind: str) -> None:
    """导出最近一次训练结果到页面专属目录.

    固定导出到 outputs/evaluation/linearprobe（自动创建）；JSON 与图像图表共用
    同一时间戳前缀（同一次导出成组），图像为 PNG 格式。

    Args:
        kind: "json" 导出训练结果 JSON（参数 + history + 报告）；
              "images" 导出训练曲线 PNG 与混淆矩阵 PNG。
    """
    lp_export = state.get("lp_export") or {}
    result = lp_export.get("result")
    if result is None:
        ui.notify("请先完成训练", type="warning")
        return
    export_dir = Path(_PROJECT_ROOT) / LP_EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文件名统一格式：集合缩写_模型结构_内容缩写_时间戳
    collection = collection_short_name(_current_collection)
    variant = lp_export.get("variant") or VARIANT_MLP
    written: list[str] = []

    if kind == "json":
        def _conv(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return str(obj)

        arch = _ARCH_INFO[variant]
        export_data = {
            "model": "MLP_label",
            "architecture": {
                "class": "MLPLabel",
                "variant": variant,
                "in_features": 64,
                "num_classes": NUM_CLASSES,
                **arch,
            },
            "label_names": LABEL_NAMES,
            "history": result.history,
            "best_epoch": result.best_epoch,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_macro_f1": result.best_val_macro_f1,
            "val_report": result.val_report,
            "train_report": result.train_report,
            "elapsed_seconds": result.elapsed_seconds,
            "device": result.device,
            "device_name": result.device_name,
            "gpu_kernel_seconds": result.gpu_kernel_seconds,
        }
        json_path = export_dir / f"{collection}_{variant}_result_{ts}.json"
        json_path.write_text(
            _json.dumps(export_data, indent=2, ensure_ascii=False, default=_conv),
            encoding="utf-8",
        )
        written.append(json_path.name)
    else:
        # 图像图表：训练曲线 PNG + 混淆矩阵 PNG
        curves_path = export_dir / f"{collection}_{variant}_curves_{ts}.png"
        curves_path.write_bytes(plot_training_curves(state.get("history") or []))
        written.append(curves_path.name)
        cm = lp_export.get("confusion_cm")
        if cm is not None:
            cm_path = export_dir / f"{collection}_{variant}_cm_{ts}.png"
            cm_path.write_bytes(
                plot_confusion_matrix(
                    np.asarray(cm),
                    [LABEL_NAMES[i] for i in range(NUM_CLASSES)],
                    title="MLP_label 混淆矩阵（验证集）",
                )
            )
            written.append(cm_path.name)

    ui.notify(
        f"已导出 {len(written)} 个文件 → {export_dir}: {', '.join(written)}",
        type="positive",
    )


# ---------- asyncio 别名（便于测试 monkeypatch） ----------


def run(port: int = DEFAULT_PORT) -> None:
    """启动 NiceGUI 服务（供 __main__ 与测试使用）."""
    ui.run(
        title="Qdrant Linear Probe 评估系统 — MLP_label",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    import argparse

    parser = argparse.ArgumentParser(description="Qdrant Linear Probe WebUI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"监听端口 (默认: {DEFAULT_PORT})")
    parser.add_argument("--qdrant-url", default=QDRANT_URL,
                        help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")
    args = parser.parse_args()
    _CLI_QDRANT_URL = args.qdrant_url
    run(port=args.port)
