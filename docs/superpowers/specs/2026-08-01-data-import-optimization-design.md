---
comet_change: data-import-optimization
role: technical-design
canonical_spec: openspec
archived-with: 2026-08-02-data-import-optimization
status: final
---

# Data Import Optimization — 深度技术设计

## 1. 目标与范围

针对 Qdrant KNN 像素评估系统数据导入模块的四项优化，详见 `openspec/changes/data-import-optimization/proposal.md`。本 Design Doc 细化实现方案、边界条件与测试策略，为 build 阶段提供可执行依据。

**验收目标**：
1. WebUI 待导入数据预览分页（每页 20 条）且可翻页查看全部条目。
2. "导入全部"按钮位于数据目录栏上方，无需滚动越过预览列表即可点击。
3. 导入期间显示按像素推进的线性进度条 + 进度文本，完成后到满。
4. 导入速度显著提升（主要来自向量化构造加速）。
5. `--reindex` 触发全量 HNSW 向量索引重建（既有行为确认并文档化）。
6. UTM 坐标仅从文件名坐标段 `E{lon}_N{lat}` + 128×128 像素位置推算，不加载 GeoTIFF。

## 2. 现状与约束

- `webui.py` 数据导入区（`browse_directory` / `do_import`）：预览一次性渲染全部 `file_pairs`，"导入全部"按钮在预览下方，无逐像素进度。
- `importer.py`：`build_points()` 双层循环逐像素构造 `PointStruct`（每像素 `uuid5` + `.tolist()`）；`_batch_upsert()` 以 `wait=True` 分 10,000 条/批 upsert；`import_directory()` 已含 `reindex` 参数，通过 `update_collection(optimizer_config=OptimizersConfigDiff(indexing_threshold=0))` 触发 HNSW 全量重建。
- 断点续传按 `image_id` count 判定（`PixelDataLoader.check_image_count`），依赖 `wait=True` 的写入可见性。
- 约束：不新增第三方依赖、不改 Collection schema、不改 `searcher.py` / `label_mapping.py` 语义、保持断点续传/ID 生成/统计输出行为不变。

## 3. 设计方案

### 3.1 分页辅助纯函数（可独立测试）

新增模块 `KNN_evaluation/ui_pagination.py`（或 `webui.py` 内独立函数区），暴露：

```python
PAGE_SIZE = 20

def paginate_slice(items: list, page: int, page_size: int = PAGE_SIZE) -> list:
    """返回 items[page*page_size : (page+1)*page_size]，越界安全返回空列表."""

def total_pages(total: int, page_size: int = PAGE_SIZE) -> int:
    """总页数，total==0 时返回 1."""

def page_controls(page: int, total: int, page_size: int = PAGE_SIZE) -> tuple[bool, bool]:
    """返回 (can_prev, can_next)."""
```

- `page` 从 0 开始；`page_controls` 由 `page > 0` 与 `(page+1)*page_size < total` 推导。
- **测试**：页切片、越界、总页数、边界禁用态，纯函数断言，不依赖 NiceGUI。

### 3.2 `build_points()` 向量化重构

保持签名 `build_points(se_data, dw_data, easting, northing, utm_zone, image_id) -> list[models.PointStruct]` 与返回语义（16,384 个点、点序 `row-major`、payload 字段一致）。

实现要点：
1. `vectors = se_data.reshape(64, -1).T` → shape `(16384, 64)`；逐像素 `float(e)` 转 Python float（Qdrant 序列化需要原生 float，`.tolist()` 提供）。
2. `labels = dw_data.reshape(-1)`；`label_names` 用预计算映射 `[LABEL_NAMES.get(int(l), "unknown") for l in labels]`（避免每像素 dict lookup 的重复开销，实际仍逐值）。
3. `rows, cols = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")` 展平得行列。
4. `point_ids = [str(uuid.uuid5(ns, f"{r}_{c}")) for r, c in zip(rows_flat, cols_flat)]`。
5. 固定字段 `utm_zone`（`zone = utm_zone if utm_zone is not None else -1`）、`image_id` 作为单值复用到每个 payload；easting/northing 用 `map(float, easting.reshape(-1))` 展平。

**关键**：保持与现状**完全一致**的点序（row-major: row 外层、col 内层）与 payload 键序。由现有 `test_importer.py` 断言回归（`points[0]` 字段、UUID 格式、NaN UTM、批次切分 call_count）。

### 3.3 `import_directory()` 进度回调

```python
ProgressCallback = Callable[[int, int], None] | None

def import_directory(
    self,
    data_dir: Path,
    no_resume: bool = False,
    reindex: bool = False,
    progress_callback: ProgressCallback = None,
) -> dict:
```

- `total` = 待导入像素总数（非跳过影像的 `128*128` 累加，预先遍历 `pairs` 计算；若全部跳过则 total=0）。
- 每批 upsert 成功后 `progress_callback(imported_so_far, total)`；`imported_so_far` 为已成功 upsert 的像素累计。
- 默认 `progress_callback=None` 时行为与现状完全一致（tqdm 影像级进度保留）。
- 断点续传跳过影像不计入 imported；`total` 只在真实导入前累加。

**注意**：`import_image_pair()` 内部每影像 upsert 2 批（16,384 / 10,000），回调在 `_batch_upsert` 每批后由 `import_directory` 统一驱动，避免改动 `import_image_pair` 签名（向后兼容）。

### 3.4 WebUI 数据导入区改造

- `browse_directory()`：扫描后 `state["preview_page"] = 0`；渲染时用 `paginate_slice(file_pairs, page)` 切片到 `file_column`；翻页按钮闭包 `set_page(delta)` 更新页码并重渲染。首/末页禁用态由 `page_controls` 推导。
- "导入全部"按钮：移到数据目录输入框所在 `ui.row`（或紧邻其上方），移除预览下方的原按钮。
- `do_import()`：创建 `ui.linear_progress` + 进度文本；`asyncio.to_thread(importer.import_directory, ..., progress_callback=cb)`，`cb` 更新进度条 `value` 与文本 `f"已导入 {imported:,} / {total:,}"`；完成后进度到满、显示统计对话框。

### 3.5 CLI `import` 子命令

- `--reindex` 已实现（HNSW 重建）：仅补充 `--help` 文档说明语义为"导入完成后重建全量 HNSW 向量索引"。
- 新增像素级进度：`import_directory` 支持 `progress_callback` 后，CLI 用 `tqdm` 像素级更新（复用现有 `_tqdm_context` 模式）替代/增强影像级进度。

### 3.6 UTM 坐标从文件名推算（不加载 TIF）

**背景**：用户要求导入时只加载 NPY/NPZ 文件，UTM 坐标从文件名坐标段 + 像素位置推算，不再加载 GeoTIFF。坐标模型以 DW 下载脚本 `download_scripts/DynamicWorld/core.py::_create_square_roi` 为准。

**坐标模型（权威）**：
- 文件名坐标段 `E{lon}_N{lat}` 是影像**中心点**（128×128 中心像素 64,64）。
- 分辨率 `scale = 10m`（config 常量 `UTM_RESOLUTION_M`，可配置）。
- UTM CRS：`zone = int((lon+180)/6)+1`，北半球 `EPSG:326xx`、南半球 `EPSG:327xx`（`_get_utm_epsg` 规则）。
- 网格 NW 角对齐 scale 整数倍：
  - `half = grid_size×scale/2 = 640m`
  - 中心点 (lon,lat) → UTM `(cx, cy)`
  - `nw_x = floor((cx − 640)/10)×10`，`nw_y = ceil((cy + 640)/10)×10`
- 逐像素：`easting[r,c] = nw_x + c×10 + 5`；`northing[r,c] = nw_y − r×10 − 5`（row 0 北、col 0 西）。

**实现**：
- `coordinate_utils.py` 新增 `compute_utm_grid_from_name(lon, lat, scale=UTM_RESOLUTION_M, grid_size=128) -> (easting(128,128), northing(128,128), utm_zone)`，用 pyproj `Transformer` 做 WGS84→UTM 投影。
- `data_loader.py` 新增 `parse_location_coord(location_key) -> (lon, lat)`，解析 `E{lon}_N{lat}`。
- `config.py` 新增 `UTM_RESOLUTION_M = 10`。
- `importer.py` `import_image_pair` 主路径：从 `pair.image_id` 解析坐标 → `compute_utm_grid_from_name`；`compute_utm_grid(tif_path)` 保留为仅文件名推算失败时回退。
- `ImagePair.tif_path` 保留字段但不再必需（scan_directory 仍收集，导入不再强依赖）。

**一致性保证**：由于 NW 角对齐 10m 网格且分辨率固定，从文件名推算的网格与 GeoTIFF transform 推导结果在像素级一致（demo 数据已验证：`E121.4025_N25.1947` → UTM 中心 ≈ 339035, 2787465，与 TIF transform `|10,0,338390| |0,−10,2788110|` 一致）。

## 4. 边界条件与错误处理

| 场景 | 行为 |
|---|---|
| `pairs` 为空 | `import_directory` 返回全零统计（现状保留） |
| 全部影像已导入（断点续传跳过） | `total=0`，回调以 `(0,0)` 或直接完成；WebUI 显示"无新导入" |
| 单影像部分存在（count>0 且 <16384） | 覆盖重传整影像（现状保留，`wait=True` 保证可见） |
| Qdrant 不可达 / Collection 不存在 | 抛 `ConnectionError` / `RuntimeError`（现状保留） |
| 进度回调在 `asyncio.to_thread` 中更新 UI | 由 NiceGUI 事件循环调度，与现有 `do_evaluate` 一致 |
| 翻页越界（末页后点下一页） | `paginate_slice` 返回空列表或夹紧到末页，按钮禁用态阻止 |
| 文件名坐标段缺失/无法解析 | `parse_location_coord` 抛 ValueError；`import_image_pair` 回退 `compute_utm_grid(tif_path)`（NaN 网格 + 警告） |
| 纬度超出 UTM 范围（>84° 或 <-80°） | `_get_utm_epsg` 回退 EPSG:4326（坐标精度降低但功能不中断） |

## 5. 测试策略

1. **`test_importer.py`（更新）**：向量化 `build_points()` 与现状输出一致（点数、字段、UUID 确定性、NaN UTM、批次切分）。
2. **`test_progress_callback.py`（新增）**：mock manager 下 `import_directory(progress_callback=cb)` 回调按批推进、`imported_so_far` 最终等于实际导入数、全部跳过时 total=0。
3. **`test_ui_pagination.py`（新增）**：`paginate_slice` / `total_pages` / `page_controls` 纯函数边界。
4. **`test_importer_reindex.py`（新增）**：`import_directory(reindex=True)` 断言 `update_collection(indexing_threshold=0)` 被调用。
5. **`test_coordinate_utils.py`（新增）**：`compute_utm_grid_from_name` 对已知坐标段的网格与 GeoTIFF transform 结果差分一致；UTM 带号推导（南北半球）；`parse_location_coord` 解析。
6. **集成**：`data_demo` 跑通 CLI `import --reindex`（含无 TIF 场景 UTM 推算）；WebUI 分页/按钮/进度验证；断点续传回归。

## 6. 验收对照

| 验收目标 | 对应实现 |
|---|---|
| 分页预览每页 20 条 | `3.1` + `3.4` |
| 导入全部按钮上移 | `3.4` |
| 按像素进度条 | `3.3` + `3.4` |
| 导入提速 | `3.2` 向量化构造（吞吐对比在 build 阶段量化） |
| `--reindex` HNSW 重建 | `3.5`（既有行为确认） |
| UTM 从文件名推算 | `3.6`（`compute_utm_grid_from_name` + `parse_location_coord`） |
