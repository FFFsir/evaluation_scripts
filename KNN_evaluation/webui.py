"""Qdrant KNN 像素评估系统 — NiceGUI Web 可视化界面.

固定三个页面（不再允许用户自定义 collection 页面）：

- **GOOGLE**：绑定 Qdrant Collection `google_aef_embedding`（数据目录 data_google），
  含 Qdrant 连接 & Collection 状态、数据导入、向量检索、评估面板四块内容；
- **XIAN**：绑定 `xian_aef_embedding`（data_xian），同样四块内容；
- **SimilarityMatrix**：相似度热力图对比独立页面（固定对比 google × xian 预置对）。

页面隔离：GOOGLE / XIAN 两页各自持有独立的 manager、评估结果、查询向量与导入状态，
切页不丢失、评估结果互不覆盖。数据导入失败自动整体重试（最多 3 次、指数退避），
利用按影像断点续传从失败处继续。评估面板混淆矩阵以图片展示；评估结果 JSON 与
图片图表均导出到页面专属目录（outputs/evaluation/google_aef、xian_aef、
similarity），图片为 PNG。

启动方式:
    cd D:\\Project\\光机所项目\\evaluation_scripts
    uv run python KNN_evaluation/webui.py --port 8003
"""
import asyncio
import base64
import inspect
import io
import json
import random
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from nicegui import ui

# 确保项目根目录在 path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 直接脚本运行（python KNN_evaluation/webui.py）时，解释器会把脚本所在目录
# KNN_evaluation/ 置于 sys.path 首位，导致内部 `from qdrant_client import ...`
# 解析到本地模块 KNN_evaluation/qdrant_client.py（循环导入）而非 pip 的 qdrant_client。
# 把脚本目录移出 sys.path，仅保留项目根目录，消除命名冲突。
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _SCRIPT_DIR]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------- KNN 包导入 ----------
from KNN_evaluation.config import (
    QDRANT_URL, PRESET_COLLECTIONS, COLLECTION_DATA_DIRS,
    BATCH_SIZE, EF_SEARCH_DEFAULT, UTM_RESOLUTION_M,
)
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.data_loader import PixelDataLoader
from KNN_evaluation.coordinate_utils import compute_utm_grid_from_name
from KNN_evaluation.manifest import manifest_path, load_manifest
from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.searcher import PixelSearcher, SearchResult, HitRecord
from KNN_evaluation.label_mapping import LABEL_NAMES, LABEL_IDS
from KNN_evaluation.ui_pagination import (
    PAGE_SIZE, paginate_slice, total_pages, page_controls,
)
from KNN_evaluation.visualization import (
    confusion_matrix_base64, plot_confusion_matrix, plot_purity_recall_curve,
)

# ---------- 默认配置 ----------
DEFAULT_PORT = 8003
# 导入失败自动重试的基础延迟（秒）；第 i 次重试等待 base * 2**i（2s/4s/8s）。
# 测试可置 0 跳过真实等待。
IMPORT_RETRY_BASE_DELAY = 2

# ---------- 页面定义（需求 2：固定 GOOGLE / XIAN / SimilarityMatrix 三页） ----------
# 每个 KNN 页绑定固定 collection 与数据目录（COLLECTION_DATA_DIRS 映射）；
# GOOGLE/XIAN 导出统一到 outputs/evaluation/knn_eval（命名统一，见 _export_page_results）。
PAGES = {
    "GOOGLE": {
        "collection": PRESET_COLLECTIONS[0],            # google_aef_embedding
        "data_dir": COLLECTION_DATA_DIRS[PRESET_COLLECTIONS[0]],  # data_google
        "export_dir": "outputs/evaluation/knn_eval",
    },
    "XIAN": {
        "collection": PRESET_COLLECTIONS[1],            # xian_aef_embedding
        "data_dir": COLLECTION_DATA_DIRS[PRESET_COLLECTIONS[1]],  # data_xian
        "export_dir": "outputs/evaluation/knn_eval",
    },
}
PAGE_ORDER = ["GOOGLE", "XIAN", "SimilarityMatrix"]
# SimilarityMatrix 页固定导出目录（需求 5）
SIMILARITY_EXPORT_DIR = Path("outputs/evaluation/similarity")

# 导出文件名中 collection 缩写映射（预置 google/xian，自定义原名）
_COLLECTION_SHORT = {
    "google_aef_embedding": "google",
    "xian_aef_embedding": "xian",
}


def collection_short_name(collection: str) -> str:
    """导出文件名使用的 collection 缩写：预置 google/xian，自定义原名."""
    return _COLLECTION_SHORT.get(collection, collection)

# ---------- 模块级状态 / 缓存 / 钩子 ----------
# CLI 覆盖参数需有模块级默认（__main__ 启动时再覆盖）
_CLI_QDRANT_URL: str = QDRANT_URL
_CLI_DATA_DIR: str = "data_demo"

# 页面状态：state["pages"][page_key] 为各 KNN 页独立状态（需求 4 页面隔离），
# SimilarityMatrix 状态存 state["similarity"]。state 在 index() 每次请求重建，
# 页面状态不依赖模块级单例，保证切页不串扰。
state: dict = {}
# 页面钩子：_init_hooks[page_key][hook_name] = 闭包（index() 构建页面时注入）
_init_hooks: dict[str, dict] = {}
# 进程级 manifest 缓存：按 collection 隔离（两页 collection 不同，互不串扰）
_manifest_caches: dict[str, dict] = {}

# 评估取消令牌注册表：_eval_cancel_events[page_key] = {id: Event}，
# 两页评估任务相互独立（需求 4：评估结果与任务不互相干扰）。
_eval_cancel_events: dict[str, dict[int, "threading.Event"]] = {}
# 闭包注入：页面构建时记录中止按钮控件，供模块级 helper 切换可见性。
_set_eval_cancel_btn_visible: dict[str, object] = {}


def _new_page_state(data_dir: Path) -> dict:
    """构建单个 KNN 页面的初始状态字典（GOOGLE / XIAN 共用）."""
    return {
        "manager": None,
        "data_dir": data_dir,
        "file_pairs": [],
        "preview_page": 0,
        "se_paths_map": {},
        "last_search_result": None,
        "query_vector": None,
        "query_image_id": None,
        "query_row": 0,
        "query_col": 0,
        "query_utm_easting": None,
        "query_utm_northing": None,
        "search_collection": None,
        # 评估结果：GOOGLE/XIAN 各自保存，互不覆盖（需求 4）
        "eval": {"f1_result": None, "f2_result": None, "config": None},
    }


def _cancel_page_evaluate(page_key: str) -> None:
    """设置指定页面评估任务的取消事件（按钮 on_click 处理器）."""
    for ev in _eval_cancel_events.get(page_key, {}).values():
        ev.set()
    fn = _set_eval_cancel_btn_visible.get(page_key)
    if fn is not None:
        fn(False)


async def _call_hook(page_key: str, name: str):
    """调用 _init_hooks[page_key] 中的闭包钩子，兼容同步/异步函数.

    闭包内 refresh_status / _refresh_image_list / _render_preview 捕获 UI 控件，
    无法提到模块级；模块级路径经此间接调用。
    """
    fn = _init_hooks.get(page_key, {}).get(name)
    if fn is None:
        return None
    result = fn()
    if inspect.isawaitable(result):
        return await result
    return result


# ---------- manifest 缓存（按 collection 隔离） ----------

def _get_manifest_cached(collection: str) -> dict:
    if collection not in _manifest_caches:
        _manifest_caches[collection] = load_manifest(manifest_path(collection))
    return _manifest_caches[collection]


def _invalidate_manifest_cache(collection: str) -> None:
    _manifest_caches.pop(collection, None)


def _manifest_pixels(collection: str, image_id: str) -> int:
    """从进程级 manifest 缓存读取已导入像素数（无则 0）."""
    data = _get_manifest_cached(collection)
    return int((data.get("images") or {}).get(image_id, 0))


def _imported_image_ids(collection: str) -> set[str]:
    """读 manifest 缓存的 image_id 集合（无 manager / 空清单时返回空集）."""
    return set((_get_manifest_cached(collection).get("images") or {}).keys())


def _viz_data_dir(page_key: str, search_collection: str | None = None) -> Path:
    """按检索 collection 解析可视化数据目录（页面隔离）.

    页面隔离：image_id 归一化后两 collection 完全重叠，可视化必须按检索所属
    collection（COLLECTION_DATA_DIRS 映射）解析数据目录，否则背景图串集
    （GOOGLE 检索取到 XIAN 的 SE 文件）。搜索 collection 无映射时回退本页数据目录。
    """
    col = search_collection or PAGES[page_key]["collection"]
    mapped = COLLECTION_DATA_DIRS.get(col)
    if mapped:
        return Path(_PROJECT_ROOT) / mapped
    return state["pages"][page_key]["data_dir"]


# ---------- 页面初始化 ----------

async def init_page() -> None:
    """快速路径（事件循环内，毫秒级）：为 GOOGLE/XIAN 两页创建 manager + 置 ✅.

    慢速路径（load_manifest/scan/reconcile）经 asyncio.create_task 后台执行，
    不 await —— 避免全库 scroll / scan 阻塞 Web 启动。
    """
    for pk in ("GOOGLE", "XIAN"):
        mgr = QdrantManager(url=_CLI_QDRANT_URL, collection_name=PAGES[pk]["collection"])
        state["pages"][pk]["manager"] = mgr
        # Task 13：页面加载即对当前 collection 自动补齐 payload 索引（幂等），
        # 防 UTM 过滤因缺索引触发全量扫描超时。失败不阻塞页面（try/except 降级）。
        try:
            await asyncio.to_thread(mgr.ensure_payload_indices)
        except Exception:
            pass
        await _call_hook(pk, "refresh_status")  # 健康检查 + 置 ✅
    await asyncio.sleep(0)  # 让出事件循环，状态 flush 到浏览器
    for pk in ("GOOGLE", "XIAN"):
        asyncio.create_task(_background_init(pk))  # 慢速路径：不 await


async def _background_init(page_key: str) -> None:
    """后台慢速初始化：load_manifest → scan_directory → facet 对账 → 渲染.

    整体 try/except 吞异常：后台路径失败不阻塞页面。
    """
    try:
        collection = PAGES[page_key]["collection"]
        await asyncio.to_thread(_load_manifest_cached, collection)
        data_dir = state["pages"][page_key]["data_dir"]
        if data_dir.exists():
            await asyncio.to_thread(_scan_directory_only, page_key)
        await asyncio.to_thread(_reconcile_background, page_key)
        await _call_hook(page_key, "refresh_image_list")
        await _call_hook(page_key, "render_preview")
    except Exception:
        pass


def _load_manifest_cached(collection: str) -> None:
    _get_manifest_cached(collection)


def _scan_directory_only(page_key: str) -> None:
    pairs = PixelDataLoader.scan_directory(state["pages"][page_key]["data_dir"])
    state["pages"][page_key]["file_pairs"] = pairs
    state["pages"][page_key]["se_paths_map"] = {p.image_id: p.se_path for p in pairs}
    state["pages"][page_key]["preview_page"] = 0


def _reconcile_background(page_key: str) -> None:
    mgr = state["pages"][page_key]["manager"]
    collection = PAGES[page_key]["collection"]
    if mgr is not None and mgr.collection_exists():
        try:
            _manifest_caches[collection] = mgr.reconcile_manifest()
        except Exception:
            pass


# 命中标记颜色（Matplotlib tab10 前 10 色）
_HIT_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


# ---------- 灰度渲染 ----------

def render_grayscale_with_markers(
    data: np.ndarray,
    channel_idx: int,
    zoom: int = 4,
    markers: list[dict] | None = None,
) -> str:
    """渲染带标记的通道灰度图。

    Args:
        data: shape (64, 128, 128) float64 数组.
        channel_idx: 通道索引 0-63.
        zoom: 放大倍率.
        markers: 标记列表，每项含 row/col/index/label/score.

    Returns:
        base64 PNG data URI.
    """
    channel = data[channel_idx]
    gray = ((channel + 1.0) / 2.0 * 255.0)
    gray = gray.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    if zoom > 1:
        img = img.resize((128 * zoom, 128 * zoom), Image.NEAREST)
    img_rgb = img.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    if markers is None:
        markers = []

    for m in markers:
        row, col = m["row"], m["col"]
        cx = col * zoom + zoom // 2
        cy = row * zoom + zoom // 2
        r = max(3, zoom)

        if m.get("is_query"):
            # 红色十字准线 (查询像素)
            draw.line([(cx - r, cy), (cx + r, cy)], fill=(255, 0, 0), width=max(1, zoom // 2))
            draw.line([(cx, cy - r), (cx, cy + r)], fill=(255, 0, 0), width=max(1, zoom // 2))
            draw.rectangle(
                [cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1],
                outline=(255, 0, 0), width=1,
            )
        else:
            # 彩色圆点 + 编号 (命中)
            color = _HIT_COLORS[m.get("index", 0) % len(_HIT_COLORS)]
            draw.ellipse(
                [cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1],
                outline=color, width=max(2, zoom // 2),
            )
            # 命中编号文字
            idx_str = str(m.get("index", 0) + 1)
            draw.text((cx - len(idx_str) * 3, cy - 6), idx_str, fill=(0, 0, 0))

    buf = io.BytesIO()
    img_rgb.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _build_label_dist_html(dist: dict[str, int]) -> str:
    """构建标签分布 HTML 片段."""
    if not dist:
        return '<span style="color: #999;">无数据</span>'
    parts = []
    for name, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        parts.append(
            f'<span style="display:inline-block;margin:2px 4px;padding:2px 6px;'
            f'background:#e8e8e8;border-radius:4px;">{name}: {cnt}</span>'
        )
    return "".join(parts)


def _image_utm_bounds(
    img_id: str,
    ref_easting: float | None = None,
    ref_northing: float | None = None,
    ref_row: int | None = None,
    ref_col: int | None = None,
) -> dict | None:
    """估算指定影像 128×128 像素覆盖的 UTM 范围（东/北向 min/max）.

    优先从 image_id 坐标段（如 "E121.4794_N25.1378"）解析经纬度，用与导入
    一致的坐标模型（compute_utm_grid_from_name）推算整张影像的 UTM 网格，
    取网格 min/max 即覆盖范围；解析失败时回退用参考像素的 UTM + 行列反推
    （像素分辨率取 UTM_RESOLUTION_M）。两者都不可用时返回 None。
    """
    try:
        lon, lat = PixelDataLoader.parse_location_coord(img_id)
        easting, northing, _ = compute_utm_grid_from_name(lon, lat)
        return {
            "min_e": float(easting.min()),
            "max_e": float(easting.max()),
            "min_n": float(northing.min()),
            "max_n": float(northing.max()),
        }
    except Exception:
        pass
    if None not in (ref_easting, ref_northing, ref_row, ref_col) and ref_row >= 0 and ref_col >= 0:
        s = UTM_RESOLUTION_M
        return {
            "min_e": ref_easting - ref_col * s,
            "max_e": ref_easting + (127 - ref_col) * s,
            "min_n": ref_northing - (127 - ref_row) * s,
            "max_n": ref_northing + ref_row * s,
        }
    return None


# ---------- Docker 辅助 ----------

def _qdrant_is_running() -> bool:
    """检查本地 Qdrant Docker 是否运行."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        return "qdrant" in result.stdout
    except Exception:
        return False


def _start_qdrant() -> bool:
    """幂等启动 Qdrant Docker 容器（挂 volume 持久化）.

    与 cli._start_qdrant（Task 3.3）保持一致的幂等三态，避免两处逻辑漂移：
    1. `docker ps -a` 全量列出（含停止容器）：
       - 运行中 → 复用；
       - 存在但停止 → `docker start`；
       - 不存在 → `docker run -v qdrant_data:/qdrant/storage` 创建并启动.
    2. 任何异常（Docker 未安装/守护进程未启动等）捕获并返回 False，
       不影响既有 WebUI（仅状态区提示）.

    Returns:
        True 表示容器已处于运行状态，False 表示无法确保（调用方应重试健康检查）.
    """
    name = "qdrant"
    try:
        ps = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        names = {line.strip() for line in ps.stdout.splitlines() if line.strip()}
        if name in names:
            # 容器已存在：检查是否运行中，停止则启动
            running = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10,
            )
            if name in {
                line.strip() for line in running.stdout.splitlines() if line.strip()
            }:
                return True
            start = subprocess.run(
                ["docker", "start", name],
                capture_output=True, text=True, timeout=30,
            )
            return start.returncode == 0
        run = subprocess.run(
            [
                "docker", "run", "-d", "--name", name,
                "-p", "6333:6333", "-p", "6334:6334",
                "-v", "qdrant_data:/qdrant/storage",
                "qdrant/qdrant:latest",
            ],
            capture_output=True, text=True, timeout=60,
        )
        return run.returncode == 0
    except Exception:
        return False


# ---------- 评估结果导出（需求 5：JSON 与图片图表均落盘到页面目录） ----------

def _export_page_results(page_key: str, kind: str) -> None:
    """导出评估结果到统一目录.

    GOOGLE/XIAN 页统一导出到 outputs/evaluation/knn_eval，文件名遵循
    `{集合缩写}_knn_{内容缩写}_{时间戳}.xxx`（如 google_knn_result_...json、
    xian_knn_cm_...png、xian_knn_pr_...png）。JSON 与图片共用同一时间戳前缀。

    Args:
        page_key: "GOOGLE" | "XIAN"。
        kind: "json" 导出评估结果 JSON；"images" 导出混淆矩阵与 Purity & Per-class Recall 曲线 PNG。
    """
    pg = state["pages"][page_key]
    eval_data = pg.get("eval") or {}
    f1 = eval_data.get("f1_result")
    f2 = eval_data.get("f2_result")
    if f1 is None or f2 is None:
        ui.notify("尚无评估结果，请先开始评估", type="warning")
        return
    export_dir = Path(_PROJECT_ROOT) / PAGES[page_key]["export_dir"]
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = collection_short_name(PAGES[page_key]["collection"])
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

        export_data = {
            "config": eval_data.get("config"),
            "f1": {k: v for k, v in f1.items()} if f1 else None,
            "f2": {
                "k_values": f2["k_values"],
                "global_purity": f2["global_purity"],
                "global_recall": f2["global_recall"],
                "per_class_purity": f2["per_class_purity"],
                "per_class_recall": f2["per_class_recall"],
            } if f2 else None,
        }
        json_path = export_dir / f"{short}_knn_result_{ts}.json"
        json_path.write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False, default=_conv),
            encoding="utf-8",
        )
        written.append(json_path.name)
    else:
        # 图片图表：混淆矩阵 PNG + Purity & Per-class Recall 曲线 PNG
        label_names = list(LABEL_NAMES.values())
        cm_path = export_dir / f"{short}_knn_cm_{ts}.png"
        plot_confusion_matrix(
            np.asarray(f1["confusion_matrix"]), label_names,
            cm_path, title="Confusion Matrix",
        )
        written.append(cm_path.name)
        pr_path = export_dir / f"{short}_knn_pr_{ts}.png"
        plot_purity_recall_curve(f2, pr_path, f1)
        written.append(pr_path.name)

    ui.notify(
        f"已导出 {len(written)} 个文件 → {export_dir}: {', '.join(written)}",
        type="positive",
    )


def _export_similarity_results(kind: str) -> None:
    """导出 SimilarityMatrix 页最近一次对比结果到页面专属目录.

    固定导出到 outputs/evaluation/similarity（需求 5 目录约定）；文件名以
    `full_col_`（数据库全库）或 `single_img_`（单张图片）开头区分采样数据来源；
    JSON/PNG/npy 共用同一时间戳前缀（同一次导出成组），图片为 PNG 格式。

    Args:
        kind: "json" 导出采样参数与保留像素信息 JSON；"images" 导出并排热力图 PNG；
              "npy" 导出两个相似度矩阵 npy（带时间戳，不覆盖旧文件）。
    """
    sim = state.get("similarity") or {}
    result = sim.get("result")
    png_bytes = sim.get("png_bytes")
    if not result or not png_bytes:
        ui.notify("请先生成热力图对比", type="warning")
        return
    export_dir = Path(_PROJECT_ROOT) / SIMILARITY_EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 采样数据来源前缀：数据库全库 → full_col，单张图片 → single_img
    mode_prefix = sim.get("mode") or "full_col"
    written: list[str] = []

    if kind == "json":
        # 复用 compare 落盘的采样信息（params + pixels）作为单一事实源
        sampling: dict = {}
        for p in result.get("exported_files") or []:
            p = Path(p)
            if p.name.endswith("similarity_sampling.json") and p.exists():
                try:
                    sampling = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    sampling = {}
                break
        export_data = {
            "params": {
                "n": result.get("sampled"),
                "kept": result.get("kept"),
                "dropped": result.get("dropped"),
                "matrix_shape": result.get("matrix_shape"),
                "elapsed_sec": result.get("elapsed_sec"),
                "collections": list(collection_pair_default()),
            },
            "sampling": sampling.get("params", sampling.get("sampling")),
            "pixels": sampling.get("pixels"),
        }
        json_path = export_dir / f"{mode_prefix}_similarity_{ts}.json"
        json_path.write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(json_path.name)
    elif kind == "npy":
        # 相似度矩阵 npy × 2（google/xian），带时间戳、不覆盖旧文件
        sim_g = sim.get("sim_g")
        sim_x = sim.get("sim_x")
        if sim_g is None or sim_x is None:
            ui.notify("对比结果缺少相似度矩阵，请重新生成热力图对比", type="warning")
            return
        g_short = collection_short_name(collection_pair_default()[0])  # google
        x_short = collection_short_name(collection_pair_default()[1])  # xian
        g_path = export_dir / f"{mode_prefix}_{g_short}_similarity_{ts}.npy"
        x_path = export_dir / f"{mode_prefix}_{x_short}_similarity_{ts}.npy"
        np.save(g_path, sim_g)
        np.save(x_path, sim_x)
        written.extend([g_path.name, x_path.name])
    else:
        # 图片图表：并排热力图 PNG（对比时已渲染进 png_bytes）
        png_path = export_dir / f"{mode_prefix}_similarity_heatmap_{ts}.png"
        png_path.write_bytes(png_bytes)
        written.append(png_path.name)

    ui.notify(
        f"已导出 {len(written)} 个文件 → {export_dir}: {', '.join(written)}",
        type="positive",
    )


def collection_pair_default() -> tuple[str, str]:
    """SimilarityMatrix 页固定对比的预置集合对（google × xian）."""
    return (PRESET_COLLECTIONS[0], PRESET_COLLECTIONS[1])


# ---------- 页面构建 ----------

@ui.page("/")
def index():
    """主页面：Qdrant KNN 像素评估系统（固定 GOOGLE / XIAN / SimilarityMatrix 三页）."""
    global state
    state = {
        "pages": {
            pk: _new_page_state(Path(_PROJECT_ROOT / PAGES[pk]["data_dir"]))
            for pk in ("GOOGLE", "XIAN")
        },
        "similarity": {},
    }

    ui.page_title("Qdrant KNN 像素评估系统")

    # ===== HEADER =====
    with ui.header(elevated=True).classes("bg-primary text-white"):
        ui.label("Qdrant KNN 像素评估系统").classes("text-h4")

    # ===== 固定三页面（需求 2/3）：页面内容在各 tab_panel 内完整构建 =====
    tabs = ui.tabs()
    with tabs:
        for name in PAGE_ORDER:
            ui.tab(name, label=name)
    with ui.tab_panels(tabs, value=PAGE_ORDER[0]):
        with ui.tab_panel("GOOGLE"):
            _build_knns_page("GOOGLE")
        with ui.tab_panel("XIAN"):
            _build_knns_page("XIAN")
        with ui.tab_panel("SimilarityMatrix"):
            _build_similarity_page()

    # 快速路径（模块级）：创建两页 manager + 置 ✅ + create_task 后台慢速路径
    ui.timer(0.1, callback=init_page, once=True)


def _build_knns_page(page_key: str) -> None:
    """构建一个 KNN 页面的完整内容（GOOGLE / XIAN 共用，需求 4 页面隔离）.

    四块内容：Qdrant 连接 & Collection 状态、数据导入、向量检索、评估面板。
    所有闭包引用 `state["pages"][page_key]`（模块级重查），页面重建后仍绑定新 state；
    两页各自持有独立 manager / 评估结果 / 查询向量 / 导入进度，互不覆盖。
    """
    collection = PAGES[page_key]["collection"]
    dir_default = str(_PROJECT_ROOT / PAGES[page_key]["data_dir"])

    # ===== Qdrant 连接 & Collection 状态 =====
    with ui.expansion("Qdrant 连接 & Collection 状态", value=True).classes("w-full mt-4"):
        status_row = ui.row().classes("items-center gap-4")
        status_label = ui.label("⏳ 正在连接...").classes("text-sm")
        info_row = ui.row().classes("items-center gap-4 mt-2")
        info_label = ui.label("").classes("text-sm text-grey")
        create_col_btn = ui.button(
            "创建 Collection", on_click=lambda: _create_collection(),
        ).props("flat dense size=sm")
        create_col_btn.set_visibility(False)
        reindex_btn = ui.button(
            "构建向量索引", on_click=lambda: _reindex_vectors(),
        ).props("flat dense size=sm")
        reindex_btn.set_visibility(False)

        def _create_collection():
            mgr = state["pages"][page_key]["manager"]
            if mgr and not mgr.collection_exists():
                mgr.create_collection()
                mgr.create_payload_indices()
                ui.notify("Collection 创建完成", type="positive")
                refresh_status()

        def _reindex_vectors():
            """一键触发全量 HNSW 向量索引重建（后台异步执行）."""
            mgr = state["pages"][page_key]["manager"]
            if mgr is None:
                ui.notify("QdrantManager 未初始化，请刷新页面", type="negative")
                return
            if not mgr.collection_exists():
                ui.notify("Collection 不存在，请先创建或导入", type="warning")
                return
            try:
                mgr.reindex_vectors()
                ui.notify(
                    "向量索引重建已触发，Qdrant 后台构建中；"
                    "可点击「刷新状态」查看索引进度",
                    type="positive",
                )
                refresh_status()
            except Exception as e:
                ui.notify(f"触发索引重建失败: {e}", type="negative")

        def refresh_status():
            mgr = state["pages"][page_key]["manager"]
            if mgr is None:
                return
            try:
                if not mgr.health_check():
                    raise ConnectionError("health check failed")
                status_label.set_text("✅ Qdrant 连接正常")
                status_label.classes("text-sm text-positive")
                if mgr.collection_exists():
                    try:
                        info = mgr.collection_info()
                        indexed = info.get("vectors_count", 0) or 0
                        total = info.get("total_points", 0) or 0
                        if total > 0 and indexed < total:
                            info_label.classes(replace="text-sm text-warning")
                            indexing_info = (
                                f"  |  ⚠️ 向量索引构建中（已索引 {indexed:,} / "
                                f"{total:,}）"
                            )
                        else:
                            info_label.classes(replace="text-sm text-positive")
                            indexing_info = f"  |  已索引向量: {indexed:,} / {total:,}"
                        info_label.set_text(
                            f"Collection: {collection}  |  "
                            f"总点数: {info.get('total_points', 0):,}  |  "
                            f"分段数: {info.get('segments_count', 0)}"
                            f"{indexing_info}"
                        )
                    except Exception as ci_err:
                        info_label.set_text(
                            f"Collection '{collection}' 存在但获取信息失败"
                        )
                        ui.notify(f"获取 Collection 信息失败: {ci_err}", type="warning")
                    create_col_btn.set_visibility(False)
                    reindex_btn.set_visibility(True)
                else:
                    info_label.set_text(f"Collection '{collection}' 不存在")
                    create_col_btn.set_visibility(True)
                    reindex_btn.set_visibility(False)
                    create_col_btn.set_visibility(True)
            except Exception as exc:
                status_label.set_text("❌ Qdrant 不可达")
                status_label.classes("text-sm text-negative")
                info_label.set_text(f"错误详情: {exc}")
                create_col_btn.set_visibility(False)

        ui.button("刷新状态", on_click=refresh_status).props("flat dense size=sm")
        refresh_status()

    # ===== 数据导入（需求 1：失败自动整体重试，按影像断点续传） =====
    with ui.expansion("数据导入", value=False).classes("w-full mt-2"):
        async def do_import():
            mgr = state["pages"][page_key]["manager"]
            if mgr is None:
                ui.notify("QdrantManager 未初始化，请刷新页面", type="negative")
                return
            if not mgr.health_check():
                ui.notify("Qdrant 不可达，请先启动 Qdrant", type="negative")
                return
            if not mgr.collection_exists():
                mgr.create_collection()
                mgr.create_payload_indices()
                ui.notify("已自动创建 Collection", type="info")
            # 从输入框当前值解析目录并同步 state（与「浏览」按钮同源）
            raw_dir = (dir_input.value or "").strip()
            if not raw_dir:
                ui.notify("请填写数据目录", type="negative")
                return
            data_dir = Path(raw_dir)
            state["pages"][page_key]["data_dir"] = data_dir
            if not data_dir.exists():
                ui.notify(f"目录不存在: {dir_input.value}", type="negative")
                return
            importer = PixelImporter(mgr, batch_size=BATCH_SIZE)
            ui.notify("导入已开始，请稍候...", type="info")

            import_progress_bar.set_visibility(True)
            import_progress_label.set_visibility(True)
            import_progress_bar.value = 0
            import_progress_label.set_text("准备中...")

            def cb(imported, total):
                import_progress_bar.value = imported / max(total, 1)
                import_progress_label.set_text(f"已导入 {imported:,} / {total:,}")

            # 需求 1：导入失败自动从断点继续导入 —— 整体自动重试（最多 3 次重试、
            # 指数退避 2s/4s/8s）。每次重试沿用 import_directory 的按影像断点续传
            # （check_image_count 跳过已完整导入影像），从失败处继续。
            max_attempts = 4  # 首次 + 3 次自动重试
            stats = None
            last_err: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    stats = await asyncio.to_thread(
                        importer.import_directory,
                        data_dir, False, True, cb,
                    )
                    break
                except Exception as e:  # noqa: BLE001 — 报告用户并决定是否重试
                    last_err = e
                    if attempt >= max_attempts - 1:
                        break
                    wait = IMPORT_RETRY_BASE_DELAY * (2 ** attempt)
                    import_progress_label.set_text(
                        f"导入失败（第 {attempt + 1}/{max_attempts} 次尝试），"
                        f"{wait}s 后自动从断点继续重试... 原因: {e}"
                    )
                    await asyncio.sleep(wait)

            if stats is None:
                ui.notify(
                    f"导入失败（已自动重试 {max_attempts - 1} 次）: {last_err}",
                    type="negative",
                )
                import_progress_bar.set_visibility(False)
                import_progress_label.set_visibility(False)
                return

            # 完成后进度到满并隐藏进度条
            import_progress_bar.value = 1.0
            import_progress_bar.set_visibility(False)
            import_progress_label.set_visibility(False)

            if stats["imported_images"] == 0:
                ui.notify("无新导入（数据已存在或目录为空）", type="info")
            else:
                _show_import_stats(stats)
                ui.notify(
                    f"导入完成: {stats['imported_images']} 张影像, "
                    f"{stats['total_pixels']:,} 像素",
                    type="positive",
                )
            refresh_status()
            # 导入已完成，manifest 已更新：失效进程级缓存，刷新下拉立即读到新影像
            _invalidate_manifest_cache(collection)
            await browse_directory()

        async def browse_directory():
            raw_dir = (dir_input.value or "").strip()
            if not raw_dir:
                ui.notify("请填写数据目录", type="negative")
                return
            data_dir = Path(raw_dir)
            state["pages"][page_key]["data_dir"] = data_dir
            if not data_dir.exists():
                ui.notify(f"目录不存在: {dir_input.value}", type="negative")
                state["pages"][page_key]["file_pairs"] = []
                _render_preview()
                return
            try:
                pairs = PixelDataLoader.scan_directory(data_dir)
            except Exception as e:
                ui.notify(f"扫描失败: {e}", type="negative")
                return
            state["pages"][page_key]["file_pairs"] = pairs
            state["pages"][page_key]["se_paths_map"] = {
                p.image_id: p.se_path for p in pairs
            }
            state["pages"][page_key]["preview_page"] = 0

            # 同时更新指定像素的影像选择器选项
            await _refresh_image_list()

            _render_preview()

        def _set_page(delta: int):
            total = len(state["pages"][page_key]["file_pairs"])
            if total == 0:
                return
            last_page = total_pages(total, PAGE_SIZE) - 1
            state["pages"][page_key]["preview_page"] = max(
                0, min(state["pages"][page_key]["preview_page"] + delta, last_page)
            )
            _render_preview()

        def _render_preview():
            file_column.clear()
            pairs = state["pages"][page_key]["file_pairs"]
            total = len(pairs)
            if not pairs:
                with file_column:
                    ui.label("未找到匹配的 SE/DW 文件对").classes("text-grey text-sm")
                prev_btn.set_enabled(False)
                next_btn.set_enabled(False)
                page_label.set_text("")
                return
            page = state["pages"][page_key]["preview_page"]
            can_prev, can_next = page_controls(page, total, PAGE_SIZE)
            prev_btn.set_enabled(can_prev)
            next_btn.set_enabled(can_next)
            page_label.set_text(
                f"第 {page + 1}/{total_pages(total, PAGE_SIZE)} 页 · 共 {total} 条"
            )
            for pair in paginate_slice(pairs, page, PAGE_SIZE):
                count = _manifest_pixels(collection, pair.image_id)  # 读 manifest 缓存
                status = (
                    "✅ 已导入" if count >= 16384
                    else f"⏳ {count}/16384" if count > 0
                    else "📦 待导入"
                )
                with ui.row().classes("w-full items-center border-b py-1"):
                    ui.label(pair.image_id).classes("font-mono text-sm")
                    ui.label(status).classes("text-xs text-grey")

        def _show_import_stats(stats: dict):
            with ui.dialog() as d, ui.card().classes("w-full max-w-lg p-4"):
                d.open()
                ui.label("导入统计").classes("text-h5")
                ui.label(
                    f"总像素: {stats['total_pixels']:,}  |  "
                    f"总影像: {stats['total_images']}  |  "
                    f"新导入: {stats['imported_images']}  |  "
                    f"跳过: {stats['skipped_images']}"
                ).classes("text-sm")
                ui.label(
                    f"耗时: {stats['elapsed_sec']:.1f}s  |  "
                    f"速率: {stats['rate_pps']:.0f} 像素/秒"
                ).classes("text-sm text-grey")
                if stats.get("label_counts"):
                    ui.label("标签分布:").classes("text-sm mt-2")
                    for name, cnt in sorted(
                        stats["label_counts"].items(), key=lambda x: -x[1],
                    ):
                        ui.label(f"  {name}: {cnt:,}").classes("text-xs text-grey")
                ui.button("关闭", on_click=lambda: d.close()).props("flat mt-4")

        # 导入全部按钮 + 进度条：位于数据目录栏上方，无需滚动越过预览列表即可点击
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("导入全部", on_click=do_import).props("flat")
            import_progress_bar = ui.linear_progress(value=0).classes("w-96")
            import_progress_bar.set_visibility(False)
            import_progress_label = ui.label("").classes("text-sm text-grey")
            import_progress_label.set_visibility(False)

        # 数据目录栏（页面隔离：默认值为本页绑定目录，页面独立维护）
        with ui.row().classes("w-full items-center gap-4"):
            dir_input = ui.input(
                label="数据目录", value=dir_default,
            ).classes("w-96")
            ui.button("浏览", on_click=browse_directory).props("flat")

        # 预览列表
        file_column = ui.column().classes("w-full")

        # 分页控件
        with ui.row().classes("items-center gap-2 mt-2"):
            prev_btn = ui.button(
                "上一页", on_click=lambda: _set_page(-1),
            ).props("flat dense size=sm")
            page_label = ui.label("").classes("text-sm text-grey")
            next_btn = ui.button(
                "下一页", on_click=lambda: _set_page(1),
            ).props("flat dense size=sm")

    # ===== 向量检索（页面隔离：查询向量/检索结果绑定本页 state） =====
    with ui.expansion("向量检索", value=True).classes("w-full mt-2"):
        query_vector_lbl = ui.label("未选择查询向量").classes("text-sm text-grey")

        async def _refresh_image_list():
            """用 collection 中已有数据（manifest 缓存）或文件扫描结果填充影像选择器."""
            image_options = {}
            mgr = state["pages"][page_key]["manager"]
            if mgr and mgr.collection_exists():
                try:
                    imported = await asyncio.to_thread(
                        mgr.get_imported_image_ids,
                    )
                    for img_id in sorted(imported):
                        image_options[img_id] = img_id
                except Exception:
                    pass
            # 同时加入文件浏览器中扫描到的 image_id
            for pair in state["pages"][page_key].get("file_pairs", []):
                if pair.image_id not in image_options:
                    image_options[pair.image_id] = pair.image_id + " (未导入)"
            spec_image_select.set_options(
                image_options or {"": "无可用影像"}, value=None,
            )

        with ui.row().classes("items-center gap-4 mt-2"):
            query_mode = ui.radio(
                ["随机选取", "指定像素"], value="随机选取",
            ).props("inline")

            # 随机选取
            async def do_random_query():
                pg = state["pages"][page_key]
                mgr = pg["manager"]
                if mgr is None or not mgr.collection_exists():
                    ui.notify("Collection 不可用", type="negative")
                    return
                try:
                    from qdrant_client import models
                    scroll_filter = None
                    if random_label_select.value:
                        scroll_filter = models.Filter(must=[
                            models.FieldCondition(
                                key="label",
                                match=models.MatchAny(
                                    any=[int(v) for v in random_label_select.value],
                                ),
                            ),
                        ])
                    count_result = await asyncio.to_thread(
                        mgr.client.count,
                        collection_name=mgr.collection_name,
                        count_filter=scroll_filter,
                        exact=True,
                    )
                    total = count_result.count
                    if total == 0:
                        ui.notify("所选标签下无数据，请换标签或先导入", type="negative")
                        return
                    offset = random.randrange(total)
                    scroll_result = await asyncio.to_thread(
                        mgr.client.scroll,
                        collection_name=mgr.collection_name,
                        scroll_filter=scroll_filter,
                        limit=1,
                        offset=offset,
                        with_vectors=True,
                    )
                    if not scroll_result[0]:
                        ui.notify("Collection 为空，请先导入数据", type="negative")
                        return
                    rec = scroll_result[0][0]
                    pg["query_vector"] = np.array(rec.vector, dtype=np.float64)
                    p = rec.payload or {}
                    pg["query_image_id"] = p.get("image_id", "")
                    pg["query_row"] = p.get("pixel_row", -1)
                    pg["query_col"] = p.get("pixel_col", -1)
                    pg["query_utm_easting"] = p.get("utm_easting")
                    pg["query_utm_northing"] = p.get("utm_northing")
                    label_desc = p.get("label_name", "?")
                    if random_label_select.value:
                        label_desc = f"{label_desc} (按标签随机)"
                    query_vector_lbl.set_text(
                        f"✅ 查询向量: {rec.id}  (label={label_desc})"
                    )
                    ui.notify("随机查询向量已选取", type="positive")
                    _apply_utm_for_query()
                except Exception as e:
                    ui.notify(f"获取查询向量失败: {e}", type="negative")

            with ui.row().classes("items-center gap-2"):
                random_label_select = ui.select(
                    label="随机选取标签（可多选）",
                    options={None: "全部标签"} | LABEL_NAMES,
                    multiple=True,
                    value=[],
                ).classes("w-48")
                ui.button("从 Collection 随机获取", on_click=do_random_query).props("flat dense size=sm")

            # 指定像素
            with ui.row().classes("items-center gap-2"):
                spec_image_select = ui.select(
                    label="影像（下拉选择）", options={}, value=None,
                ).classes("w-64")
                spec_image_input = ui.input(
                    label="或手动输入影像名",
                    placeholder="如 E121.4794_N25.1378",
                ).classes("w-64")
                spec_row = ui.number(label="行", value=0, min=0, max=127).classes("w-20")
                spec_col = ui.number(label="列", value=0, min=0, max=127).classes("w-20")

                async def do_spec_query():
                    pg = state["pages"][page_key]
                    mgr = pg["manager"]
                    if mgr is None or not mgr.collection_exists():
                        ui.notify("Collection 不可用", type="negative")
                        return
                    img_id = (spec_image_input.value or "").strip() or spec_image_select.value
                    if not img_id:
                        ui.notify("请先选择或输入影像名称", type="negative")
                        return
                    r, c = int(spec_row.value), int(spec_col.value)
                    try:
                        from qdrant_client import models
                        scroll_result = await asyncio.to_thread(
                            mgr.client.scroll,
                            collection_name=mgr.collection_name,
                            scroll_filter=models.Filter(must=[
                                models.FieldCondition(
                                    key="image_id",
                                    match=models.MatchValue(value=img_id),
                                ),
                                models.FieldCondition(
                                    key="pixel_row",
                                    match=models.MatchValue(value=r),
                                ),
                                models.FieldCondition(
                                    key="pixel_col",
                                    match=models.MatchValue(value=c),
                                ),
                            ]),
                            limit=1, with_vectors=True,
                        )
                        if not scroll_result[0]:
                            ui.notify(f"未找到像素 {img_id}_{r}_{c}", type="negative")
                            return
                        rec = scroll_result[0][0]
                        p = rec.payload or {}
                        pg["query_vector"] = np.array(rec.vector, dtype=np.float64)
                        pg["query_image_id"] = img_id
                        pg["query_row"] = r
                        pg["query_col"] = c
                        pg["query_utm_easting"] = p.get("utm_easting")
                        pg["query_utm_northing"] = p.get("utm_northing")
                        query_vector_lbl.set_text(
                            f"✅ 查询像素: {img_id}_{r}_{c}"
                        )
                        ui.notify("指定像素向量已获取", type="positive")
                        _apply_utm_for_query()
                    except Exception as e:
                        ui.notify(f"获取查询向量失败: {e}", type="negative")

                ui.button("获取向量", on_click=do_spec_query).props("flat dense size=sm")

        # 检索参数
        with ui.row().classes("items-center gap-4 mt-4"):
            k_input = ui.number(label="K (Top-K)", value=10, min=1).classes("w-20")

            label_filter_select = ui.select(
                label="标签过滤",
                options={None: "无过滤"} | LABEL_NAMES,
                multiple=True,
                value=[],
            ).classes("w-48")

            with ui.column().classes("gap-1"):
                ui.label("UTM 范围过滤").classes("text-xs text-grey")
                with ui.row().classes("items-center gap-1"):
                    utm_min_e = ui.number(label="东 min", value=0).classes("w-24")
                    utm_max_e = ui.number(label="东 max", value=9999999).classes("w-24")
                with ui.row().classes("items-center gap-1"):
                    utm_min_n = ui.number(label="北 min", value=0).classes("w-24")
                    utm_max_n = ui.number(label="北 max", value=9999999).classes("w-24")
                ui.label("选定查询向量后自动设为该影像覆盖范围").classes("text-xs text-grey")

            use_utm = ui.switch("启用 UTM 过滤", value=False)
            use_exact = ui.switch("精确搜索", value=True)
            with ui.column().classes("gap-1"):
                ef_input = ui.number(
                    label="ef_search", value=EF_SEARCH_DEFAULT, min=1, max=512,
                ).classes("w-20").props('hint="ANN 候选集大小"')
                ui.label("ef_search 越大越准但越慢；精确搜索时忽略").classes("text-xs text-grey")

        def _apply_utm_for_query():
            """选定查询向量后：把 UTM 过滤参数自动设为覆盖查询影像 128×128 的范围."""
            pg = state["pages"][page_key]
            img_id = pg.get("query_image_id")
            if not img_id:
                return
            bounds = _image_utm_bounds(
                img_id,
                ref_easting=pg.get("query_utm_easting"),
                ref_northing=pg.get("query_utm_northing"),
                ref_row=pg.get("query_row"),
                ref_col=pg.get("query_col"),
            )
            if bounds is None:
                return
            utm_min_e.value = bounds["min_e"]
            utm_max_e.value = bounds["max_e"]
            utm_min_n.value = bounds["min_n"]
            utm_max_n.value = bounds["max_n"]
            ui.notify(
                f"UTM 过滤参数已自动设为影像 {img_id} 的覆盖范围（启用开关即可过滤）",
                type="info",
            )

        async def do_search():
            pg = state["pages"][page_key]
            mgr = pg["manager"]
            if pg["query_vector"] is None:
                ui.notify("请先选择查询向量", type="negative")
                return
            if mgr is None or not mgr.collection_exists():
                ui.notify("Collection 不可用", type="negative")
                return

            # 记录本次检索所属 collection，供可视化按数据目录解析影像文件
            # （页面隔离：各页记录自己的 collection，结果不串集）。
            pg["search_collection"] = mgr.collection_name

            searcher = PixelSearcher(mgr)
            label_filter = None
            if label_filter_select.value:
                label_filter = [int(v) for v in label_filter_select.value]

            utm_range = None
            if use_utm.value:
                utm_range = {
                    "min_e": utm_min_e.value,
                    "max_e": utm_max_e.value,
                    "min_n": utm_min_n.value,
                    "max_n": utm_max_n.value,
                }

            try:
                result = await asyncio.to_thread(
                    searcher.search,
                    query_vector=pg["query_vector"],
                    k=int(k_input.value),
                    label_filter=label_filter,
                    utm_range=utm_range,
                    exact=use_exact.value,
                    ef_search=int(ef_input.value),
                )
            except Exception as e:
                ui.notify(f"检索失败: {e}", type="negative")
                return

            pg["last_search_result"] = result
            _show_search_result(result, page_key)

        ui.button("执行检索", on_click=do_search).props("flat mt-2")

    # ===== 评估面板（页面隔离：评估结果存本页 eval；混淆矩阵图片展示） =====
    with ui.expansion("评估面板", value=False).classes("w-full mt-2"):
        ui.label("Embedding 质量评估").classes("text-h6")

        # 参数说明区
        with ui.expansion("参数说明", value=False).classes("w-full mt-1"):
            ui.markdown("""
**参数功能原理说明：**

- **每类采样数 (`samples_per_class`)**：从 Qdrant Collection 中每个地表类别（water, trees, grass 等 9 类）分别随机采样的像素数量。采样值越大，统计结果越接近真实分布，但计算耗时也线性增长。默认 500 表示共约 4500 个查询像素（9 类 × 500）。
  - *为什么需要采样？* 全量评估（遍历所有像素）在大规模数据集上不可行，分层随机采样以可控成本估计整体质量。

- **K-F1 (`k_f1`)**：KNN 分类器的邻居数。评估模块使用 Leave-One-Out（留一法）策略——检索 K+1 个最近邻后剔除查询像素自身，取前 K 个有效邻居按多数投票预测标签。K 值越小越依赖局部一致性，K 值越大越平滑但可能混入噪声。
  - *学术依据*：Cover & Hart (1967) "Nearest Neighbor Pattern Classification"。

- **K 值序列 (`k_values`)**：Purity（邻居纯度）和 Recall@K 曲线在不同 K 值下的采样点。Purity 衡量 Top-K 邻居中与查询像素同标签的比例；Recall@K 衡量前 K 个邻居召回全量同类像素的能力。**优化策略**：对每个查询像素仅执行一次检索（K = max(k_values) + 1），从结果中递增取不同 K 值计算，避免重复检索。
  - *Purity(K)* = (1/N) × Σᵢ (Kᵢ_same / K) — Precision@K 在无查询意图场景下的等价形式
  - *Recall@K* = (1/N) × Σᵢ (Kᵢ_same / N_same_class(i)) — 标准 IR 指标，全量分母

- **随机种子 (`seed`)**：控制采样和随机过程的种子值。相同 seed + 相同参数 → 完全可复现的结果。用于纵向对比不同模型或不同 embedding 版本。
""").classes("text-xs text-grey")

        # 参数配置区
        with ui.row().classes("items-center gap-4 mt-2"):
            spc_input = ui.number(label="每类采样数", value=500, min=10, max=10000).classes("w-32")
            kf1_input = ui.number(label="K-F1", value=100, min=1, max=1000).classes("w-20")
            kvalues_input = ui.input(
                label="K 值序列", value="10,20,50,100,300,1000",
            ).classes("w-64")
            seed_input = ui.number(label="Seed", value=42).classes("w-20")

        # 评估模式（精确 / ANN）
        with ui.row().classes("items-center gap-4 mt-2"):
            eval_mode_switch = ui.switch("精确模式 (Exact)", value=True)
            ui.label(
                "精确：全库暴力 Top-K，结果最准；关闭后使用 ANN 近似检索，速度更快。"
            ).classes("text-xs text-grey")

        # device 配置区（替代批量 checkbox）
        with ui.row().classes("items-center gap-4 mt-2"):
            device_select = ui.select(
                ["auto", "cuda", "cpu"], value="auto", label="执行设备",
            ).classes("w-32")
            max_gpu_mem_input = ui.number(label="显存预算(GB)", value=16, min=1, max=64).classes("w-32")
            max_eval_ram_input = ui.number(label="CPU RAM 预算(GB)", value=6, min=1, max=64).classes("w-32")
            eval_ram_label = ui.label("").classes("text-xs text-grey")

        # 进度区: 包含进度条和文本各一个
        with ui.row().classes("items-center gap-4 mt-2"):
            eval_progress_bar = ui.linear_progress(value=0).classes("w-96")
            eval_progress_bar.set_visibility(False)
        eval_progress_label = ui.label("").classes("text-sm text-grey mt-2")
        eval_progress_label.set_visibility(False)

        # 结果区
        eval_result_container = ui.column().classes("w-full mt-4")
        eval_result_container.set_visibility(False)

        eval_export_json_btn = ui.button(
            "导出 JSON", on_click=lambda: _export_page_results(page_key, "json"),
        )
        eval_export_json_btn.set_visibility(False)
        eval_export_img_btn = ui.button(
            "导出图片图表", on_click=lambda: _export_page_results(page_key, "images"),
        )
        eval_export_img_btn.set_visibility(False)

        async def do_evaluate():
            pg = state["pages"][page_key]
            mgr = pg["manager"]
            if mgr is None or not mgr.health_check():
                ui.notify("Qdrant 不可达", type="negative")
                return
            if not mgr.collection_exists():
                ui.notify(f"Collection '{mgr.collection_name}' 不存在", type="negative")
                return
            info = mgr.collection_info()
            if info.get("total_points", 0) == 0:
                ui.notify("Collection 为空", type="negative")
                return

            spc = int(spc_input.value)
            k_f1 = int(kf1_input.value)
            k_values = [int(x.strip()) for x in kvalues_input.value.split(",") if x.strip()]
            seed_val = int(seed_input.value)

            from KNN_evaluation.metrics import (
                sample_queries_by_label,
                evaluate_knn,
                EvaluationCancelled,
            )
            from KNN_evaluation.gpu_knn import resolve_device

            device = device_select.value
            try:
                resolved = resolve_device(device)
            except RuntimeError as e:
                ui.notify(str(e), type="negative")
                eval_progress_label.set_visibility(False)
                eval_progress_bar.set_visibility(False)
                return
            n_points = info.get("total_points", 0)
            corpus_gb = n_points * 64 * 4 / 1e9
            budget = float(max_eval_ram_input.value) if resolved == "cpu" else float(max_gpu_mem_input.value)
            eval_ram_label.set_text(
                f"设备: {resolved} | corpus≈{corpus_gb:.2f}GB | 预算: {budget}GB"
            )

            # 取消令牌：本次评估注册 threading.Event，「中止评估」按钮经
            # _cancel_page_evaluate 设置；协作式取消 + engine.close() try/finally 释放资源。
            cancel_event = threading.Event()
            _eval_cancel_events.setdefault(page_key, {})[id(cancel_event)] = cancel_event
            _set_eval_cancel_btn_visible[page_key](True)
            try:
                eval_progress_label.set_visibility(True)
                eval_progress_bar.set_visibility(True)
                eval_progress_bar.value = 0
                eval_progress_label.set_text("正在采样查询像素...")
                await asyncio.sleep(0.05)

                try:
                    queries = await asyncio.to_thread(
                        sample_queries_by_label, mgr, spc, seed_val,
                        warn_callback=lambda msg: eval_progress_label.set_text(msg),
                    )
                except Exception as e:
                    ui.notify(f"采样失败: {e}", type="negative")
                    return
                if cancel_event.is_set():
                    ui.notify("评估已中止", type="warning")
                    return

                num_q = sum(1 for q in queries if "point_id" in q)
                eval_mode = "精确模式" if eval_mode_switch.value else "ANN 模式"
                eval_progress_label.set_text(
                    f"采样完成, 共 {num_q} 个查询像素 | 正在联合评估 F1/F2 ({eval_mode})..."
                )

                eval_progress_label.set_text(
                    f"设备: {resolved} | 正在驻留 corpus 到 {'GPU 显存' if resolved == 'cuda' else '内存'}..."
                )
                await asyncio.sleep(0.05)

                def make_cb(phase):
                    def cb(current, total):
                        eval_progress_label.set_text(f"{phase} ({current}/{total})")
                        eval_progress_bar.value = current / max(total, 1)
                    return cb

                try:
                    combined = await asyncio.to_thread(
                        evaluate_knn, mgr, queries, k_f1, k_values,
                        bool(eval_mode_switch.value),
                        resolved, None, float(max_gpu_mem_input.value), float(max_eval_ram_input.value),
                        make_cb(f"联合评估 ({eval_mode})"),
                        cancel_event=cancel_event,
                    )
                except EvaluationCancelled:
                    ui.notify("评估已中止", type="warning")
                    return
                except Exception as e:
                    ui.notify(f"评估失败: {e}", type="negative")
                    return
                f1 = combined["f1"]
                f2 = combined["f2"]

                # 需求 4：评估结果存入本页 eval 状态（GOOGLE/XIAN 互不覆盖）
                pg["eval"] = {
                    "f1_result": f1,
                    "f2_result": f2,
                    "config": {
                        "device": device_select.value,
                        "max_gpu_mem": float(max_gpu_mem_input.value),
                        "max_eval_ram": float(max_eval_ram_input.value),
                        "eval_mode": "exact" if eval_mode_switch.value else "ann",
                        "samples_per_class": spc,
                        "k_f1": k_f1,
                        "k_values": k_values,
                        "seed": seed_val,
                    },
                }

                _show_eval_results(f1, f2)
                eval_export_json_btn.set_visibility(True)
                eval_export_img_btn.set_visibility(True)
            finally:
                # 无论成功/失败/中止，复位进度 UI、隐藏中止按钮并清理取消令牌
                eval_progress_label.set_visibility(False)
                eval_progress_bar.set_visibility(False)
                _set_eval_cancel_btn_visible[page_key](False)
                _eval_cancel_events.get(page_key, {}).pop(id(cancel_event), None)

        def _show_eval_results(f1, f2):
            eval_result_container.clear()
            eval_result_container.set_visibility(True)

            # F1 结果
            with eval_result_container:
                # === 指标说明（可折叠，默认展开） ===
                with ui.expansion("指标说明", value=True).classes("w-full mt-1"):
                    ui.markdown("""
**评估指标解读：**

- **评估模式（精确 / ANN）**：**精确模式**对全库暴力检索 Top-K（GPU 分块矩阵乘），结果精确但耗时随库规模增长；**ANN 模式**走 Qdrant HNSW 近似检索，速度快但结果可能引入近似误差。大规模数据快速预览趋势时可用 ANN，正式结论建议用精确模式。

- **Overall Accuracy（整体准确率）**：所有采样像素中，KNN 分类器预测正确的比例。范围 0~1，**越高越好**。注意：9 类分类随机基线约 11%，因此显著高于 0.11 即说明 embedding 有区分能力。

- **Per-class Metrics（各类别指标）**：
  - **Precision（精确率）**：预测为该类的像素中，真正属于该类的比例。**越高越好**，低 Precision 说明模型容易把其他类误判为此类（过度预测）。
  - **Recall（召回率）**：该类的真实像素中，被正确识别出来的比例。**越高越好**，低 Recall 说明该类像素大量被漏判为其他类（预测不足）。
  - **F1（F1 分数）**：Precision 和 Recall 的调和平均数，综合衡量分类质量。范围 0~1，**越高越好**。当 Precision 和 Recall 都高时 F1 才高，任何一个低都会拉低 F1。
  - **Support（支持数）**：该类参与评估的采样像素数量。若某类 support=0，说明 Collection 中该类无像素，无法计算有效指标。

- **Confusion Matrix（混淆矩阵）**：N×N 热力图，行 = 真实标签（True Label），列 = 预测标签（Predicted Label）。**对角线数值越大越好**（正确分类），非对角线数值越小越好（误分类）。可用于定位最易混淆的类别对——例如"grass 行 crops 列"数值大说明草地常被误判为庄稼。

- **Purity@K（邻居纯度）**：Top-K 最近邻中与查询像素同标签的比例，对所有查询像素取平均。范围 0~1，**越高越好**。Purity 高说明 embedding 空间中同类像素倾向于聚集在一起。理想情况下，**Purity(K) 应随 K 增大而单调递减**——K 越小邻居越可能和查询点是同类，K 越大越可能混入异类。

- **Per-class Recall（不同 K 下各类别召回率）**：对每个 K 值，取 KNN 分类器（多数投票）在该 K 下的各类别 Recall（tp/(tp+fn)）。相比标准 Recall@K（分母为全量同类像素数，数值通常很小、缺少观测价值），本图表以分类器的 per-class Recall 为纵轴，能更直观地观察各类别召回率随 K 的变化趋势。
""").classes("text-xs text-grey")

                ui.label(f"Overall Accuracy: {f1['overall_accuracy']:.4f}").classes("text-h6 mt-2")

                label_names = list(LABEL_NAMES.values())

                f1_rows = []
                for ln in label_names:
                    m = f1["per_class_metrics"].get(ln, {})
                    f1_rows.append({
                        "label": ln,
                        "precision": f"{m.get('precision', 0):.4f}",
                        "recall": f"{m.get('recall', 0):.4f}",
                        "f1": f"{m.get('f1', 0):.4f}",
                        "support": str(m.get("support", 0)),
                    })

                ui.label("Per-class Metrics").classes("text-subtitle2 mt-2")
                ui.table(
                    columns=[
                        {"name": "label", "label": "Label", "field": "label", "align": "left",
                         "style": "white-space: normal; word-break: break-word; min-width: 200px;",
                         "headerStyle": "white-space: normal; word-break: break-word;"},
                        {"name": "precision", "label": "Precision", "field": "precision", "align": "left"},
                        {"name": "recall", "label": "Recall", "field": "recall", "align": "left"},
                        {"name": "f1", "label": "F1", "field": "f1", "align": "left"},
                        {"name": "support", "label": "Support", "field": "support", "align": "left"},
                    ],
                    rows=f1_rows, row_key="label",
                ).classes("w-full")

                # 混淆矩阵（需求 5）：改为图片展示（同 LinearProbe_evaluation 模式）
                ui.label("Confusion Matrix").classes("text-subtitle2 mt-4")
                cm_img = ui.image()
                try:
                    cm_uri = confusion_matrix_base64(
                        np.asarray(f1["confusion_matrix"]),
                        label_names, title="Confusion Matrix",
                    )
                    cm_img.set_source(cm_uri)
                except Exception as e:
                    ui.notify(f"混淆矩阵生成失败: {e}", type="warning")

                # F2 图表: Purity@K 与 Per-class Recall（不同 K）两个独立图表
                ui.label("Purity & Recall").classes("text-subtitle2 mt-4")

                purity_series = [
                    {"name": ln, "type": "line", "data": vals}
                    for ln, vals in f2["per_class_purity"].items()
                ]
                purity_series.append({
                    "name": "Global Purity", "type": "line",
                    "data": f2["global_purity"],
                    "lineStyle": {"width": 3, "color": "#000"},
                })

                # Recall 图表：优先用 KNN 分类器 per-class Recall（不同 K，来自
                # f1.per_class_recall_by_k，区分度更高）；ANN 逐条路径无多 K 混淆
                # 矩阵时回退 f2 的 per-class Recall@K。
                prbk = f1.get("per_class_recall_by_k") or {}
                if prbk:
                    prbk_keys = sorted(prbk)
                    recall_k_labels = [str(k) for k in prbk_keys]
                    first_label_metrics = prbk[prbk_keys[0]]
                    recall_series = [
                        {"name": ln, "type": "line",
                         "data": [prbk[k][ln] for k in prbk_keys]}
                        for ln in first_label_metrics
                    ]
                    recall_chart_title = "Per-class Recall (不同 K 下各类别召回率)"
                else:
                    # ANN 逐条路径无多 K 混淆矩阵：回退标准 per-class Recall@K
                    recall_k_labels = [str(k) for k in f2["k_values"]]
                    recall_series = [
                        {"name": ln, "type": "line", "data": vals}
                        for ln, vals in f2["per_class_recall"].items()
                    ]
                    recall_chart_title = "Per-class Recall@K (各类别召回率 · ANN 模式)"

                purity_x_axis = {
                    "type": "category",
                    "data": [str(k) for k in f2["k_values"]],
                    "name": "K (邻居数量 / Number of Neighbors)",
                    "nameLocation": "center",
                    "nameGap": 30,
                }
                recall_x_axis = {
                    "type": "category",
                    "data": recall_k_labels,
                    "name": "K (邻居数量 / Number of Neighbors)",
                    "nameLocation": "center",
                    "nameGap": 30,
                }
                base_grid = {"left": "8%", "right": "8%", "bottom": "18%", "containLabel": True}

                # Purity 图表 (独立)
                ui.echart({
                    "title": {"text": "Purity@K (邻居纯度)", "left": "center"},
                    "tooltip": {"trigger": "axis"},
                    "legend": {"type": "scroll", "bottom": 0},
                    "grid": base_grid,
                    "xAxis": purity_x_axis,
                    "yAxis": {
                        "type": "value",
                        "name": "Purity (纯度)",
                        "min": 0,
                        "max": 1,
                        "axisLabel": {"formatter": "{value}"},
                        "nameTextStyle": {"fontSize": 12},
                    },
                    "series": purity_series,
                })

                # Per-class Recall 图表 (独立)
                ui.echart({
                    "title": {"text": recall_chart_title, "left": "center"},
                    "tooltip": {"trigger": "axis"},
                    "legend": {"type": "scroll", "bottom": 0},
                    "grid": base_grid,
                    "xAxis": recall_x_axis,
                    "yAxis": {
                        "type": "value",
                        "name": "Recall (召回率)",
                        "min": 0,
                        "max": 1,
                        "axisLabel": {"formatter": "{value}"},
                        "nameTextStyle": {"fontSize": 12},
                    },
                    "series": recall_series,
                })

        with ui.row().classes("gap-2 mt-4"):
            ui.button("开始评估", on_click=do_evaluate).props("flat")

            # 中止评估：仅评估运行时可见/可用（do_evaluate 在开始/结束时切换可见性）
            eval_cancel_btn = ui.button("中止评估", on_click=lambda: _cancel_page_evaluate(page_key)).props("flat")
            eval_cancel_btn.set_visibility(False)
            # 闭包注入：模块级 helper 借以切换中止按钮可见性（测试可直接断言）
            def _set_cancel_btn_visible(visible: bool):
                eval_cancel_btn.set_visibility(visible)
            _set_eval_cancel_btn_visible[page_key] = _set_cancel_btn_visible

    # ===== 检索结果对话框（页面隔离：结果与可视化绑定本页 collection/数据目录） =====
    def _show_search_result(result: SearchResult, page_key: str = page_key):
        with ui.dialog() as d, ui.card().classes("w-full max-w-6xl p-4"):
            d.open()
            ui.label("检索结果").classes("text-h5")

            col = state["pages"][page_key]["manager"].collection_name

            async def do_visualize():
                d.close()
                _show_visualization(result, page_key)

            with ui.row().classes("gap-2 mt-1"):
                ui.button("可视化探索", on_click=do_visualize).props("flat")
                ui.button("关闭", on_click=lambda: d.close()).props("flat")

            # 元数据
            ui.label(
                f"模式: {result.search_mode}  |  K: {result.query_params.get('k', '?')}  |  "
                f"耗时: {result.elapsed_ms:.1f}ms  |  "
                f"命中: {len(result.hits)} 条"
            ).classes("text-sm text-grey")

            # 标签分布
            if result.label_distribution:
                with ui.expansion("标签分布", value=False):
                    dist_rows = [
                        {"name": name, "count": str(cnt)}
                        for name, cnt in sorted(
                            result.label_distribution.items(), key=lambda x: -x[1],
                        )
                    ]
                    ui.table(
                        columns=[
                            {"name": "name", "label": "标签", "field": "name"},
                            {"name": "count", "label": "数量", "field": "count"},
                        ],
                        rows=dist_rows, row_key="name",
                    ).classes("w-64")

            # 命中表格
            with ui.expansion(f"检索命中 ({len(result.hits)} 结果)", value=True).classes("w-full"):
                hit_rows = []
                for i, h in enumerate(result.hits):
                    hit_rows.append({
                        "index": i + 1,
                        "id": h.id,
                        "score": f"{h.score:.4f}",
                        "label": h.label_name,
                        "image_id": h.image_id,
                        "pos": f"({h.pixel_row},{h.pixel_col})",
                        "easting": f"{h.utm_easting:.0f}" if not np.isnan(h.utm_easting) else "N/A",
                        "northing": f"{h.utm_northing:.0f}" if not np.isnan(h.utm_northing) else "N/A",
                    })

                ui.table(
                    columns=[
                        {"name": "index", "label": "#", "field": "index", "align": "right"},
                        {"name": "id", "label": "ID", "field": "id"},
                        {"name": "score", "label": "Score", "field": "score", "align": "right"},
                        {"name": "label", "label": "标签", "field": "label"},
                        {"name": "image_id", "label": "影像", "field": "image_id"},
                        {"name": "pos", "label": "位置", "field": "pos"},
                        {"name": "easting", "label": "Easting", "field": "easting"},
                        {"name": "northing", "label": "Northing", "field": "northing"},
                    ],
                    rows=hit_rows, row_key="index",
                ).classes("w-full")

            with ui.row().classes("gap-2 mt-4"):
                ui.button("关闭", on_click=lambda: d.close()).props("flat")
                ui.button("可视化探索", on_click=do_visualize).props("flat")

    # ===== 可视化探索器对话框（页面隔离：按本页 collection 解析数据目录） =====
    def _show_visualization(result: SearchResult, page_key: str = page_key):
        pg = state["pages"][page_key]
        if not result.hits:
            ui.notify("无命中结果，无法可视化", type="negative")
            return

        # 按 image_id 分组
        hits_by_image: dict[str, list] = {}
        for i, h in enumerate(result.hits):
            img_id = h.image_id
            if img_id not in hits_by_image:
                hits_by_image[img_id] = []
            hits_by_image[img_id].append((i, h))

        # 查询像素所在影像
        query_img = pg.get("query_image_id", "")
        if query_img and query_img not in hits_by_image:
            hits_by_image[query_img] = []  # 查询影像也可能无命中

        img_ids = list(hits_by_image.keys())
        if not img_ids:
            ui.notify("无法确定可视化影像", type="negative")
            return

        # 页面隔离：按本页检索 collection 解析数据目录，构建局部 SE 映射。
        # image_id 归一化后两 collection 完全重叠，不能再用全局 se_paths_map
        # （它只随「浏览」更新，会取到另一页浏览的目录 → 背景图串集）。
        col = pg.get("search_collection") or PAGES[page_key]["collection"]
        data_dir = _viz_data_dir(page_key, col)
        pairs = PixelDataLoader.scan_directory(data_dir)
        viz_se_map = {p.image_id: p.se_path for p in pairs}

        viz_state = {
            "channel_idx": 0, "zoom": 4,
            "active_img_idx": 0,
        }

        with ui.dialog() as d, ui.card().classes("w-full max-w-6xl p-4"):
            d.open()
            ui.label("像素可视化探索").classes("text-h5")

            # 影像选择 tabs
            with ui.tabs() as tabs:
                for img_id in img_ids:
                    ui.tab(img_id, label=img_id)
            with ui.tab_panels(tabs, value=img_ids[0]) as panels:
                for img_id in img_ids:
                    with ui.tab_panel(img_id):
                        pass  # 实际内容在下方统一渲染

            # 切换图片时刷新
            def _on_tab_change():
                _refresh_viz()

            tabs.on("update:model-value", lambda e: _on_tab_change())

            # 通道 + 缩放
            channel_names = [f"A{i:02d}" for i in range(64)]
            with ui.row().classes("items-center gap-4 mt-2"):
                channel_select = ui.select(
                    label="通道", options=channel_names, value="A00",
                ).classes("w-32")
                zoom_select = ui.select(
                    label="缩放",
                    options={1: "1x", 2: "2x", 4: "4x"},
                    value=4,
                ).props("dense")

            # 可视化主区域
            with ui.row().classes("gap-4 items-start w-full mt-2"):
                image_display = ui.interactive_image().style(
                    "width:512px; height:512px; cursor:crosshair;"
                )
                hit_detail_col = ui.column().classes("gap-1").style(
                    "min-width:300px; max-height:520px; overflow-y:auto;"
                )
                hit_detail_html = ui.html("").style("max-height:460px; overflow-y:auto;")

            def _build_hit_detail_html(img_id: str):
                """构建命中详情 HTML."""
                hits = hits_by_image.get(img_id, [])
                if not hits and img_id != query_img:
                    return "<span style='color:#999;'>无命中</span>"

                lines = ['<table style="font-size:12px; border-collapse:collapse; width:100%;">']
                lines.append(
                    '<tr style="background:#f0f0f0;">'
                    '<th style="padding:2px 6px;border-bottom:1px solid #ccc;">#</th>'
                    '<th style="padding:2px 6px;border-bottom:1px solid #ccc;">ID</th>'
                    '<th style="padding:2px 6px;border-bottom:1px solid #ccc;">Score</th>'
                    '<th style="padding:2px 6px;border-bottom:1px solid #ccc;">Label</th>'
                    '<th style="padding:2px 6px;border-bottom:1px solid #ccc;">Row</th>'
                    '<th style="padding:2px 6px;border-bottom:1px solid #ccc;">Col</th>'
                    '</tr>'
                )
                for idx, h in hits:
                    color_rgb = _HIT_COLORS[idx % len(_HIT_COLORS)]
                    color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
                    lines.append(
                        f'<tr>'
                        f'<td style="padding:2px 6px;color:{color_hex};font-weight:bold;">{idx + 1}</td>'
                        f'<td style="padding:2px 6px;font-family:monospace;font-size:10px;">{h.id[:20]}…</td>'
                        f'<td style="padding:2px 6px;text-align:right;">{h.score:.4f}</td>'
                        f'<td style="padding:2px 6px;">{h.label_name}</td>'
                        f'<td style="padding:2px 6px;text-align:right;">{h.pixel_row}</td>'
                        f'<td style="padding:2px 6px;text-align:right;">{h.pixel_col}</td>'
                        f'</tr>'
                    )
                lines.append('</table>')

                # 图例
                if hits:
                    lines.append('<div style="margin-top:8px; font-size:11px;">')
                    for idx, h in hits:
                        color_rgb = _HIT_COLORS[idx % len(_HIT_COLORS)]
                        color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
                        lines.append(
                            f'<span style="display:inline-block;margin:2px 4px;">'
                            f'<span style="display:inline-block;width:10px;height:10px;'
                            f'background:{color_hex};border-radius:50%;vertical-align:middle;"></span>'
                            f' #{idx + 1} {h.label_name}'
                            f'</span>'
                        )
                    lines.append('</div>')

                return "".join(lines)

            def _refresh_viz():
                # tabs.value 才是当前选中的 tab 值
                active_img_id = tabs.value or img_ids[0]
                ch_idx = channel_names.index(channel_select.value)
                z = zoom_select.value

                # 加载 SE 数据（按本页 collection 扫描的局部映射，避免全局串集）
                se_path = viz_se_map.get(active_img_id)
                if se_path is None:
                    image_display.set_source("")
                    hit_detail_html.set_content(
                        "<span style='color:#f00;'>找不到影像文件，请确认数据目录正确</span>"
                    )
                    return

                try:
                    se_data = PixelDataLoader.load_se(se_path)
                except Exception:
                    hit_detail_html.set_content(
                        "<span style='color:#f00;'>加载影像数据失败</span>"
                    )
                    return

                # 构建标记
                markers = []
                # 查询像素标记（如果属于当前图片）
                if active_img_id == query_img:
                    markers.append({
                        "row": pg["query_row"], "col": pg["query_col"],
                        "is_query": True, "index": -1, "label": "", "score": 0,
                    })
                # 命中标记
                for idx, h in hits_by_image.get(active_img_id, []):
                    markers.append({
                        "row": h.pixel_row, "col": h.pixel_col,
                        "is_query": False, "index": idx,
                        "label": h.label_name, "score": h.score,
                    })

                uri = render_grayscale_with_markers(se_data, ch_idx, z, markers)
                image_display.set_source(uri)
                hit_detail_html.set_content(_build_hit_detail_html(active_img_id))

            # 图像点击
            def on_mouse(e):
                active_img_id = tabs.props.get("model-value") or img_ids[0]
                args = e.args
                if args and "image_x" in args and "image_y" in args:
                    ix = round(args["image_x"])
                    iy = round(args["image_y"])
                    z = zoom_select.value
                    col = ix // z
                    row = iy // z
                    if 0 <= row < 128 and 0 <= col < 128:
                        pg["query_row"] = row
                        pg["query_col"] = col
                        pg["query_image_id"] = active_img_id
                        # SE 文件无 payload UTM：清空参考值，UTM 范围改由影像名推算
                        pg["query_utm_easting"] = None
                        pg["query_utm_northing"] = None
                        # 获取向量（按本页 collection 的局部映射，避免背景图串集）
                        se_path = viz_se_map.get(active_img_id)
                        if se_path:
                            try:
                                se_data = PixelDataLoader.load_se(se_path)
                                pg["query_vector"] = se_data[:, row, col].astype(np.float64)
                                query_vector_lbl.set_text(
                                    f"✅ 查询像素: {active_img_id}_{row}_{col}"
                                )
                            except Exception:
                                pass
                        _refresh_viz()
                        _apply_utm_for_query()
                        ui.notify(f"已选择像素 ({row}, {col}) 作为新查询点", type="info")

            image_display.on("mouse", on_mouse)

            # 控件变更刷新
            channel_select.on("update:model-value", lambda _: _refresh_viz())
            zoom_select.on("update:model-value", lambda _: _refresh_viz())

            _refresh_viz()

            ui.button("关闭", on_click=lambda: d.close()).props("flat mt-4")

    # ===== 页面钩子注册（模块级 init_page/_background_init 间接调用） =====
    _init_hooks[page_key] = {
        "refresh_status": refresh_status,
        "refresh_image_list": _refresh_image_list,
        "render_preview": _render_preview,
    }


def _build_similarity_page() -> None:
    """构建 SimilarityMatrix 页面：相似度热力图对比（独立页面，需求 3）.

    保留数据库全库 / 单张图片双采样模式（决策 D3）；生成结果自动导出到
    outputs/evaluation/similarity（需求 5）：热力图 PNG + 双集合相似度矩阵
    npy + 采样信息 json。
    """
    collection_pair = collection_pair_default()

    with ui.expansion("相似度热力图对比", value=True).classes("w-full mt-4"):
        with ui.row().classes("items-center gap-4 mt-2"):
            sim_n_input = ui.number(label="采样数 N", value=200, min=1, max=600).classes("w-24")
            sim_seed_input = ui.number(label="Seed", value=42).classes("w-24")

        sim_mode = ui.radio(
            ["数据库全库", "单张图片"], value="数据库全库",
            on_change=lambda: asyncio.create_task(_apply_sim_mode()),
        ).props("inline")
        sim_image_select = ui.select(
            label="影像（单张图片模式）", options={}, value=None,
        ).classes("w-64")
        sim_image_select.set_visibility(False)

        async def _fill_sim_image_ids():
            """单张图片模式：从 manifest 缓存读已导入 image_id 填充下拉."""
            ids = _imported_image_ids(collection_pair[0])
            if not ids:
                return
            sim_image_select.set_options(
                {iid: iid for iid in sorted(ids)}, value=None,
            )

        async def _apply_sim_mode():
            if sim_mode.value == "单张图片":
                await _fill_sim_image_ids()
                sim_image_select.set_visibility(True)
            else:
                sim_image_select.set_visibility(False)

        sim_status_label = ui.label("").classes("text-sm text-grey mt-2")
        sim_status_label.set_visibility(False)
        sim_result_container = ui.column().classes("w-full mt-4")
        sim_result_container.set_visibility(False)

        async def do_sim_compare():
            from KNN_evaluation.similarity_compare import compare_similarity_heatmaps

            n = int(sim_n_input.value)
            seed = int(sim_seed_input.value)
            image_id = None
            # 采样数据来源前缀：数据库全库 → full_col，单张图片 → single_img
            mode_prefix = "full_col"
            if sim_mode.value == "单张图片":
                mode_prefix = "single_img"
                image_id = sim_image_select.value
                if not image_id:
                    ui.notify("请先选择影像", type="negative")
                    return
            # 固定对比预置对（D7）：与任何页面选择无关
            g_manager = QdrantManager(url=_CLI_QDRANT_URL, collection_name=collection_pair[0])
            x_manager = QdrantManager(url=_CLI_QDRANT_URL, collection_name=collection_pair[1])
            if not g_manager.health_check():
                ui.notify("Qdrant 不可达", type="negative")
                return
            export_dir = str(SIMILARITY_EXPORT_DIR)  # 需求 5：固定导出目录
            buf = io.BytesIO()
            sim_status_label.set_visibility(True)
            sim_status_label.set_text("正在采样并提取双集合 embedding...")
            try:
                result = await asyncio.to_thread(
                    compare_similarity_heatmaps,
                    g_manager, x_manager,
                    n=n, seed=seed, image_id=image_id,
                    output=buf,
                    collection_names=collection_pair,
                    export_dir=export_dir,
                    prefix=mode_prefix,  # sampling json 文件名带来源前缀
                    export_npy=False,    # npy 由「输出 npy 文件」按钮手动导出（带时间戳）
                )
            except (ValueError, RuntimeError, ConnectionError, OSError) as e:
                ui.notify(f"对比失败: {e}", type="negative")
                sim_status_label.set_text(f"失败: {e}")
                return
            buf.seek(0)
            png_bytes = buf.read()
            data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
            # 热力图 PNG 落盘到 outputs/evaluation/similarity（需求 5），文件名带来源前缀
            export_dir_path = Path(_PROJECT_ROOT) / export_dir
            export_dir_path.mkdir(parents=True, exist_ok=True)
            heatmap_path = export_dir_path / (
                f"{mode_prefix}_similarity_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            heatmap_path.write_bytes(png_bytes)
            exported_files = list(result.get("exported_files") or [])
            exported_files.append(str(heatmap_path))
            # 保存最近一次对比结果与热力图 PNG，供导出按钮使用（与 GOOGLE/XIAN
            # 页 eval 状态同模式；state 在页面生命周期内保留）。
            state["similarity"] = {
                "result": result,
                "png_bytes": png_bytes,
                "mode": mode_prefix,
                "sim_g": result.get("sim_g"),
                "sim_x": result.get("sim_x"),
            }
            sim_status_label.set_text(
                f"采样 {result['sampled']} | 保留 {result['kept']} "
                f"(剔除 {result['dropped']}) | 矩阵 {result['matrix_shape'][0]}×"
                f"{result['matrix_shape'][1]} | 耗时 {result['elapsed_sec']:.2f}s"
                f" | 已导出: " + ", ".join(Path(p).name for p in exported_files)
            )
            sim_result_container.clear()
            sim_result_container.set_visibility(True)
            with sim_result_container:
                ui.image(data_uri)
            # 对比成功：显示导出按钮（未对比时隐藏）
            sim_export_json_btn.set_visibility(True)
            sim_export_img_btn.set_visibility(True)
            sim_export_npy_btn.set_visibility(True)

        with ui.row().classes("items-center gap-4 mt-2"):
            ui.button("生成热力图对比", on_click=do_sim_compare).props("flat")
            sim_export_json_btn = ui.button(
                "导出 JSON", on_click=lambda: _export_similarity_results("json"),
            ).props("flat")
            sim_export_json_btn.set_visibility(False)
            sim_export_img_btn = ui.button(
                "导出图片图表", on_click=lambda: _export_similarity_results("images"),
            ).props("flat")
            sim_export_img_btn.set_visibility(False)
            sim_export_npy_btn = ui.button(
                "输出 npy 文件", on_click=lambda: _export_similarity_results("npy"),
            ).props("flat")
            sim_export_npy_btn.set_visibility(False)


# ---------- 启动入口 ----------
if __name__ in {"__main__", "__mp_main__"}:
    import argparse

    parser = argparse.ArgumentParser(
        description="Qdrant KNN 像素评估 Web 可视化界面",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"监听端口 (默认: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help=f"Qdrant 服务地址 (默认: {QDRANT_URL})",
    )
    args, unknown = parser.parse_known_args()

    # 用 CLI 参数更新默认值（供 init_page 使用）
    _CLI_QDRANT_URL = args.qdrant_url

    ui.run(port=args.port, reload=False)
