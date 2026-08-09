"""Satellite Embedding Web 可视化界面.

基于 NiceGUI 构建，提供文件浏览器、嵌入数据摘要预览和单通道灰度渲染功能。
支持点击灰度图像素查看该位置的 64 维向量值和地理坐标。

启动方式:
    uv run python src/embedding_viewer.py
    浏览器访问 http://127.0.0.1:8002
"""
import base64
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from nicegui import ui, app

try:
    from satellite_embedding_loader import load_embedding, get_summary
except ModuleNotFoundError:
    from src.satellite_embedding_loader import load_embedding, get_summary

# ---------- 默认配置 ----------
_DEFAULT_DATA_DIR = r"D:\Project\光机所项目\download_scripts\output\SE"
DEFAULT_PORT = 8002


# ---------- 辅助函数 ----------

def _parse_lon_lat_from_filename(filename: str) -> tuple[float, float] | None:
    """从文件名中提取中心点经纬度. 例: all_mean_E121.4025_N25.1947_2024.npy → (121.4025, 25.1947)"""
    m = re.search(r'E(\d+\.?\d*)_N(\d+\.?\d*)', filename)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _compute_pixel_coords(lon_center: float, lat_center: float,
                          scale: int = 10, target_pixels: int = 128) -> np.ndarray:
    """计算 128×128 像素网格中每个像素中心点的 (lon, lat) 坐标。

    与 download_scripts/SatelliteEmbedding/core.py 中 `_create_square_roi` 的投影
    规则一致: 以 (lon_center, lat_center) 为中心在 UTM 投影中构造正方形 ROI，
    再反向变换回 WGS84 得到每个像素的经纬度。

    返回 shape (128, 128, 2) 的数组，[:,:,0] 为经度，[:,:,1] 为纬度。
    """
    import math as _math
    from rasterio.warp import transform as _rasterio_transform

    half = (target_pixels * scale) / 2.0

    # UTM zone
    zone = int((lon_center + 180) / 6) + 1
    zone = max(1, min(60, zone))
    if lat_center >= 0:
        utm_crs = f"EPSG:326{zone:02d}"
    else:
        utm_crs = f"EPSG:327{zone:02d}"

    # WGS84 → UTM (lon_center, lat_center)
    lons_wgs, lats_wgs = _rasterio_transform(
        "EPSG:4326", utm_crs,
        [lon_center], [lat_center],
    )
    cx, cy = lons_wgs[0], lats_wgs[0]

    # NW 角点 (floored/ceiled to scale)
    nw_x = _math.floor((cx - half) / scale) * scale
    nw_y = _math.ceil((cy + half) / scale) * scale

    # 生成 128x128 像素中心点 (UTM)
    xs_utm = np.linspace(nw_x + scale / 2, nw_x + (target_pixels - 0.5) * scale, target_pixels)
    ys_utm = np.linspace(nw_y - scale / 2, nw_y - (target_pixels - 0.5) * scale, target_pixels)

    xx_utm, yy_utm = np.meshgrid(xs_utm, ys_utm)

    # UTM → WGS84
    lons, lats = _rasterio_transform(
        utm_crs, "EPSG:4326",
        xx_utm.ravel(), yy_utm.ravel(),
    )
    coords = np.stack([
        np.array(lons).reshape(target_pixels, target_pixels),
        np.array(lats).reshape(target_pixels, target_pixels),
    ], axis=-1)
    return coords


# ---------- 灰度渲染 ----------

def render_channel_grayscale(data, channel_idx, zoom=1):
    """将单个通道渲染为 PIL 灰度图.

    采用 [-1, 1] → [0, 255] 线性映射，缩放使用最近邻插值以保持像素边界清晰。
    """
    channel = data[channel_idx]
    gray = ((channel + 1.0) / 2.0 * 255.0)
    gray = gray.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    if zoom > 1:
        img = img.resize((128 * zoom, 128 * zoom), Image.NEAREST)
    return img


def render_grayscale_with_marker(data, channel_idx, sel_row, sel_col, zoom=1):
    """渲染带红点标记的灰度图。在选中像素位置画红色十字标记。

    Args:
        data: shape (64, 128, 128) float64 数组.
        channel_idx: 0-based 通道索引.
        sel_row: 选中的行 (0-127), -1 表示不标.
        sel_col: 选中的列 (0-127), -1 表示不标.
        zoom: 放大倍率.

    Returns:
        base64 PNG data URI string.
    """
    channel = data[channel_idx]
    gray = ((channel + 1.0) / 2.0 * 255.0)
    gray = gray.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    if zoom > 1:
        img = img.resize((128 * zoom, 128 * zoom), Image.NEAREST)

    # 画红点标记
    if 0 <= sel_row < 128 and 0 <= sel_col < 128:
        img_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)
        cx = sel_col * zoom + zoom // 2
        cy = sel_row * zoom + zoom // 2
        r = max(3, zoom)
        # 红色十字
        draw.line([(cx - r, cy), (cx + r, cy)], fill=(255, 0, 0), width=max(1, zoom // 2))
        draw.line([(cx, cy - r), (cx, cy + r)], fill=(255, 0, 0), width=max(1, zoom // 2))
        # 红色外框
        draw.rectangle(
            [cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1],
            outline=(255, 0, 0), width=1,
        )
        img = img_rgb

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ---------- 页面构建 ----------

@ui.page("/")
def index():
    """主页面：文件浏览器 + 数据预览."""
    current_data = {
        "data": None, "summary": None, "raw_shape": None, "raw_dtype": None,
        "pixel_coords": None, "lon_center": None, "lat_center": None,
    }
    # 当前选中的像素
    sel_state = {"row": 0, "col": 0}

    ui.page_title("Satellite Embedding 数据浏览器")

    with ui.header(elevated=True).classes("bg-primary text-white"):
        ui.label("Satellite Embedding 数据浏览器").classes("text-h4")

    with ui.row().classes("w-full items-center gap-4"):
        dir_input = ui.input(
            label="数据目录",
            value=_DEFAULT_DATA_DIR,
        ).classes("w-96")

        async def browse_directory():
            """扫描目录并刷新文件列表."""
            directory = Path(dir_input.value)
            if not directory.exists() or not directory.is_dir():
                ui.notify(f"目录不存在或无效: {dir_input.value}", type="negative")
                file_list.clear()
                return

            npy_files = sorted(directory.glob("*.npy"))
            npz_files = sorted(directory.glob("*.npz"))

            file_list.clear()
            if not npy_files and not npz_files:
                with file_list:
                    ui.label("未找到 .npy 或 .npz 文件").classes("text-grey")
                return

            processed = set()
            for fpath in npz_files:
                stem = fpath.stem
                if stem in processed:
                    continue
                processed.add(stem)
                stat = fpath.stat()
                size_kb = stat.st_size / 1024
                with file_list:
                    with ui.row().classes("w-full items-center border-b py-2"):
                        ui.label(f"{fpath.name}").classes("font-mono text-sm")
                        ui.label(f"{size_kb:.1f} KB").classes("text-grey text-xs")
                        ui.button(
                            "查看数据",
                            on_click=lambda _, fp=fpath: show_preview(fp),
                        ).props("flat dense size=sm")

            for fpath in npy_files:
                stem = fpath.stem
                if stem in processed:
                    continue
                processed.add(stem)
                stat = fpath.stat()
                size_kb = stat.st_size / 1024
                with file_list:
                    with ui.row().classes("w-full items-center border-b py-2"):
                        ui.label(f"{fpath.name}").classes("font-mono text-sm")
                        ui.label(f"{size_kb:.1f} KB").classes("text-grey text-xs")
                        ui.button(
                            "查看数据",
                            on_click=lambda _, fp=fpath: show_preview(fp),
                        ).props("flat dense size=sm")

        ui.button("浏览", on_click=browse_directory).props("flat")

    file_list = ui.column().classes("w-full")

    # ---------- 预览弹窗 ----------
    async def show_preview(filepath):
        """加载文件并在弹窗中展示原始结构、向量数据和灰度图."""
        try:
            data = load_embedding(filepath)
            summary = get_summary(data)
            current_data["data"] = data
            current_data["summary"] = summary
            sel_state["row"] = 0
            sel_state["col"] = 0

            # 计算像素坐标（从文件名解析中心经纬度）
            lonlat = _parse_lon_lat_from_filename(filepath.stem)
            if lonlat:
                current_data["lon_center"], current_data["lat_center"] = lonlat
                current_data["pixel_coords"] = _compute_pixel_coords(
                    lonlat[0], lonlat[1], scale=10, target_pixels=128,
                )
            else:
                current_data["pixel_coords"] = None

            # 读取原始文件信息
            raw_arr = np.load(filepath)
            if filepath.suffix.lower() == ".npz":
                keys = list(raw_arr.keys())
                raw_arr = raw_arr.get("embedding", raw_arr)
                raw_info = f"格式: .npz (keys: {keys})"
            else:
                raw_info = "格式: .npy"
            current_data["raw_shape"] = raw_arr.shape
            current_data["raw_dtype"] = str(raw_arr.dtype)
        except Exception as e:
            ui.notify(f"加载失败: {e}", type="negative")
            return

        with ui.dialog() as dialog, ui.card().classes("w-full max-w-6xl p-4"):
            dialog.open()

            ui.label(f"文件: {filepath.name}").classes("text-h5")

            # ---------- 原始数据结构 ----------
            with ui.expansion("原始数据结构", value=False).classes("w-full mt-2"):
                with ui.column().classes("gap-1 text-sm"):
                    ui.label(f"原始 shape: {current_data['raw_shape']}")
                    ui.label(f"原始 dtype: {current_data['raw_dtype']}")
                    ui.label(f"{raw_info}")
                    ui.label(f"文件大小: {filepath.stat().st_size / 1024:.1f} KB")

            # ---------- 转换后摘要 ----------
            with ui.expansion("转换后摘要", value=False).classes("w-full mt-2"):
                ui.label(
                    f"Shape: {summary['shape']}  |  Dtype: {summary['dtype']}  |  "
                    f"Global Min: {summary['global_min']:.4f}  Max: {summary['global_max']:.4f}  "
                    f"Mean: {summary['global_mean']:.4f}"
                ).classes("text-sm")
                if current_data["lon_center"]:
                    coords_np = current_data["pixel_coords"]
                    ui.label(
                        f"中心经纬度: ({current_data['lon_center']:.4f}, {current_data['lat_center']:.4f})  |  "
                        f"NW 角: ({coords_np[0, 0, 0]:.6f}, {coords_np[0, 0, 1]:.6f})  |  "
                        f"SE 角: ({coords_np[127, 127, 0]:.6f}, {coords_np[127, 127, 1]:.6f})"
                    ).classes("text-sm text-grey")

            # ---------- 通道 + 缩放控件 ----------
            channel_names = summary["channel_names"]

            with ui.row().classes("items-center gap-4 mt-4"):
                channel_select = ui.select(
                    label="通道",
                    options=channel_names,
                    value=channel_names[0],
                ).classes("w-32")

                zoom_options = {1: "1x", 2: "2x", 4: "4x"}
                zoom_select = ui.select(
                    label="缩放",
                    options=zoom_options,
                    value=4,
                ).props("dense")

            # ---------- 主体：灰度图 + 像素向量 ----------
            with ui.row().classes("gap-4 items-start w-full"):
                # 左栏：灰度图（可点击）
                with ui.column().classes("items-center"):
                    image_display = ui.interactive_image().style(
                        "width:512px; height:512px; cursor:crosshair;"
                    )

                    # 灰度条
                    with ui.row().classes("items-center mt-1"):
                        ui.label("-1").classes("text-xs")
                        ui.html(
                            '<div style="width:256px;height:16px;'
                            'background:linear-gradient(to right,black,white);'
                            'border:1px solid #ccc;"></div>'
                        )
                        ui.label("+1").classes("text-xs")

                    # 选中像素坐标
                    pixel_info_label = ui.label(
                        "点击图像选择像素，或使用下方行/列输入 →"
                    ).classes("text-xs text-grey mt-2")

                # 右栏：64 维向量表
                with ui.column().classes("gap-1").style(
                    "min-width:300px; max-height:520px; overflow-y:auto;"
                ):
                    ui.label("选中像素的 64 维向量").classes("text-sm font-bold")
                    vector_table = ui.html(
                        '<div style="font-size:11px; color:#999;">点击图像选择像素...</div>'
                    ).style("max-height:460px; overflow-y:auto;")

            # ---------- 行/列输入 + 按钮 ----------
            with ui.row().classes("items-center gap-2 mt-2"):
                row_input = ui.number(label="行 (0-127)", value=0, min=0, max=127).classes("w-24")
                col_input = ui.number(label="列 (0-127)", value=0, min=0, max=127).classes("w-24")

                def _build_vector_html(r, c):
                    """构建选中像素的 64 维向量 HTML 表格."""
                    vec = current_data["data"][:, r, c]  # shape (64,)
                    lines = ['<table style="font-size:11px; border-collapse:collapse; width:100%;">']
                    lines.append(
                        '<tr><th style="padding:1px 6px;border-bottom:1px solid #ccc;">通道</th>'
                        '<th style="padding:1px 6px;border-bottom:1px solid #ccc;">值</th></tr>'
                    )
                    for i in range(64):
                        name = f"A{i:02d}"
                        val = float(vec[i])
                        color = ""
                        if abs(val) > 0.3:
                            opacity = min(abs(val) / 1.5, 0.6)
                            if val > 0:
                                color = f'background-color:rgba(220,50,50,{opacity:.2f});'
                            else:
                                color = f'background-color:rgba(50,50,220,{opacity:.2f});'
                        lines.append(
                            f'<tr style="{color}">'
                            f'<td style="padding:1px 6px;">{name}</td>'
                            f'<td style="padding:1px 6px;text-align:right;font-family:monospace;">{val:.6f}</td>'
                            f'</tr>'
                        )
                    lines.append('</table>')
                    return ''.join(lines)

                def update_pixel_display(r, c):
                    """更新灰度图标记、向量表和坐标信息."""
                    sel_state["row"] = r
                    sel_state["col"] = c
                    row_input.value = r
                    col_input.value = c

                    # 更新灰度图（带红点标记）
                    ch_idx = channel_names.index(channel_select.value)
                    z = zoom_select.value
                    uri = render_grayscale_with_marker(current_data["data"], ch_idx, r, c, z)
                    image_display.set_source(uri)

                    # 更新向量表
                    vector_table.set_content(_build_vector_html(r, c))

                    # 更新坐标信息
                    if current_data["pixel_coords"] is not None:
                        lon = float(current_data["pixel_coords"][r, c, 0])
                        lat = float(current_data["pixel_coords"][r, c, 1])
                        pixel_info_label.set_text(
                            f"像素 ({r}, {c})  —  坐标: lon={lon:.8f}, lat={lat:.8f}"
                        )
                    else:
                        pixel_info_label.set_text(f"像素 ({r}, {c})  —  无坐标信息")

                def on_row_col_change():
                    r = max(0, min(127, int(row_input.value)))
                    c = max(0, min(127, int(col_input.value)))
                    update_pixel_display(r, c)

                row_input.on("update:model-value", on_row_col_change)
                col_input.on("update:model-value", on_row_col_change)

                ui.button("查看", on_click=on_row_col_change).props("flat dense size=sm")

            # ---------- 图像点击处理 ----------
            def on_mouse(e):
                """处理鼠标点击事件 — 从 interactive_image 的 mouse 事件获取像素坐标."""
                # e.args has image_x, image_y in original image coordinates
                args = e.args
                if args and "image_x" in args and "image_y" in args:
                    ix = round(args["image_x"])
                    iy = round(args["image_y"])
                    z = zoom_select.value
                    col = ix // z
                    row = iy // z
                    if 0 <= row < 128 and 0 <= col < 128:
                        update_pixel_display(row, col)

            image_display.on("mouse", on_mouse)

            # 初始渲染（默认选中像素 (0,0)）
            def full_refresh_image():
                ch_idx = channel_names.index(channel_select.value)
                z = zoom_select.value
                uri = render_grayscale_with_marker(
                    current_data["data"], ch_idx, sel_state["row"], sel_state["col"], z,
                )
                image_display.set_source(uri)
                vector_table.set_content(_build_vector_html(sel_state["row"], sel_state["col"]))

            channel_select.on("update:model-value", lambda _: full_refresh_image())
            zoom_select.on("update:model-value", lambda _: full_refresh_image())
            full_refresh_image()

            # ---------- 通道统计表 ----------
            with ui.expansion("各通道统计", value=False).classes("w-full mt-4"):
                with ui.element("div").classes("max-h-64 overflow-y-auto w-full"):
                    columns = [
                        {"name": "name", "label": "通道", "field": "name", "align": "left"},
                        {"name": "min", "label": "Min", "field": "min", "align": "right"},
                        {"name": "max", "label": "Max", "field": "max", "align": "right"},
                        {"name": "mean", "label": "Mean", "field": "mean", "align": "right"},
                    ]
                    rows = [
                        {
                            "name": ch["name"],
                            "min": f"{ch['min']:.6f}",
                            "max": f"{ch['max']:.6f}",
                            "mean": f"{ch['mean']:.6f}",
                        }
                        for ch in summary["channels"]
                    ]
                    ui.table(columns=columns, rows=rows, row_key="name").classes("w-full")

            ui.button("关闭", on_click=lambda: dialog.close()).props("flat mt-4")

    # 页面加载后自动扫描默认目录
    ui.timer(0.1, callback=browse_directory, once=True)


# ---------- 启动入口 ----------
if __name__ in {"__main__", "__mp_main__"}:
    import argparse

    parser = argparse.ArgumentParser(
        description="Satellite Embedding Web 可视化界面"
    )
    parser.add_argument(
        "--dir",
        default=_DEFAULT_DATA_DIR,
        help=f"默认数据目录 (默认: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"监听端口 (默认: {DEFAULT_PORT})",
    )
    args, unknown = parser.parse_known_args()

    ui.run(port=args.port, reload=False)
