---
comet_change: similarity-heatmap-compare
role: technical-design
canonical_spec: openspec
archived-with: 2026-08-08-similarity-heatmap-compare
status: final
---

# 双集合 Embedding 相似度热力图对比 — 深度技术设计

## Context

承接 open 阶段 design.md（`openspec/changes/similarity-heatmap-compare/design.md`）的高层决策，本文件细化实现。动机与需求见 proposal 与 delta spec。

关键现状（已核实）：
- 两个 Qdrant Collection `google_aef_embedding` 与 `xian_aef_embedding` 的 image 集完全一致（624/624 重叠，每影像 128×128 像素）；point_id 由 `uuid5(uuid5(uuid.NAMESPACE_DNS, image_id), "row_col")` 确定性生成（`importer.py:106,129`）——同一地理位置在两集合中 point_id 相同，这是"按位置匹配"的实现基础。
- `sampling_map.py` 的 `ensure_sampling_map(manager)` 通过分页 scroll（`with_payload=["label"]`、`with_vectors=False`，每批 50000）构建 `by_label: {label_id: [point_id...]}` 全量地图，按 collection 隔离文件 `qdrant_sampling_map_<collection>.json`；指纹（collection + total_points）不一致/缺失/损坏时自动重建。合并全部 `by_label` 即得全库 point_id 候选池。
- `QdrantManager` 支持 `collection_name` 注入（`qdrant_client.py:13-22`）；`client.retrieve(ids=..., with_payload=True, with_vectors=True)` 可按 ID 批量精确取点。
- `visualization.py` 已配置 matplotlib Agg 后端与中文字体（SimHei/Microsoft YaHei），现有混淆矩阵/曲线函数。
- WebUI（`webui.py`）有模块级 `_current_collection`、`_known_collections`（预置 + 自定义）、`_imported_image_ids()`（已导入影像集合）；评估面板用 `ui.run.io_bound` 异步执行（`webui.py:1197` 附近 `do_evaluate`）。
- CLI 各子命令用 `QdrantManager(url=args.qdrant_url, collection_name=args.collection)` 构造（`cli.py`），`--collection`/`--qdrant-url` 参数模式统一。

## Goals / Non-Goals

**Goals:**
- 单一核心模块 `KNN_evaluation/similarity_compare.py` 承载采样、提取、矩阵、编排，CLI 与 WebUI 复用，行为一致。
- 数据库模式：从目标集合全库随机抽 N 个 point_id（复用采样地图候选池）；图片模式：指定 image_id 在 128×128 网格内随机抽 N 个不重复像素。
- 双集合按同一 ids 批量 `retrieve`，单侧缺失剔除保持对齐；各自计算 N×N 余弦相似度矩阵；并排（1×2）热力图统一色阶输出 PNG。
- N 默认 200、上限 600；seed 复现；WebUI 固定对比预置对 `google_aef_embedding` × `xian_aef_embedding`。

**Non-Goals:**
- 不做标签分布分析、不做跨集合矩阵、不做交互式联动。
- 不修改 F1/F2 评估、导入、检索、迁移流程与既有 spec 行为。
- 不引入新第三方依赖；不修改 manifest / 采样地图 / corpus 缓存文件格式。
- 不提供 WebUI 内自定义双集合选择（用户确认固定预置对）。

## Decisions

### D1: `similarity_compare.py` 三个函数 + 一个编排函数（单一事实源）

模块导出（纯逻辑函数直接可测，Qdrant 交互通过注入的 manager）：
- `sample_random_points(manager, n, seed, image_id=None) -> list[dict]`
- `extract_embeddings(points, google_manager, xian_manager) -> tuple[np.ndarray, np.ndarray, dict]`
- `plot_similarity_heatmap_pair` 放 `visualization.py`（D5），不在此模块
- `compare_similarity_heatmaps(g_manager, x_manager, n, seed, image_id=None, output, ...) -> dict` 编排：采样 → 提取 → 矩阵 → 渲染 → 返回 `{sampled, kept, dropped, elapsed_sec, output_path, matrix_shape}`

**理由**：CLI/WebUI 双入口行为一致；现有 `metrics.py` 抽公共层的模式（CLI/WebUI 共用 `evaluate_knn`）。

### D2: 采样实现细节

**数据库模式**：
```python
sampling_map = ensure_sampling_map(manager)      # 自动对账/重建
candidates = [pid for ids in sampling_map["by_label"].values() for pid in ids]
n_actual = min(n, len(candidates))
picked = rng.sample(candidates, n_actual)
```
- `manager` 为**google 集合**的 manager（预置对之一，作为采样侧；点 ID 集合与 xian 一致）。
- 不足 n 时按实际数采样并返回 `sampled=n_actual` 说明（不静默降级成功——返回元数据中记录，CLI 打印提示）。
- 空候选池（`by_label` 全空但集合非空 → 地图构建失败）抛 RuntimeError；集合 total_points=0 抛 ValueError。

**图片模式**：
```python
rng = random.Random(seed)
cells = rng.sample([(r, c) for r in range(128) for c in range(128)], min(n, 16384))
point_id = str(uuid.uuid5(uuid.uuid5(uuid.NAMESPACE_DNS, image_id), f"{r}_{c}"))
```
- point_id 换算公式与 `importer.build_points` 完全一致（`uuid5(NAMESPACE_DNS, image_id)` 命名空间 + `uuid5(ns, "row_col")`）。
- 校验 image_id 存在于 google 集合（用 `PixelDataLoader.check_image_count(image_id, manager) > 0` 或采样地图成员检查）；不存在抛 ValueError。
- 从点 dict 取 UTM 坐标：图片模式直接由 `compute_utm_grid_from_name(lon, lat)` 推算或复用 retrieve 返回的 payload `utm_easting/northing/zone`。**决策**：提取阶段统一以 `retrieve` 返回的 payload 为准（数据库与图片模式同一数据源，避免坐标推算逻辑重复与精度分歧）；采样点 dict 只携带 point_id/image_id/row/col。

**理由**：两模式共用同一 UTM 数据源（payload），消除坐标推算双实现漂移；图片模式无需在采样阶段重算坐标网格。

### D3: `extract_embeddings` 批量提取与剔除

```python
ids = [p["point_id"] for p in points]
g_recs = g_manager.client.retrieve(collection_name=g_manager.collection_name, ids=ids, with_payload=True, with_vectors=True)
x_recs = x_manager.client.retrieve(collection_name=x_manager.collection_name, ids=ids, with_payload=True, with_vectors=True)
g_by_id = {str(r.id): np.array(r.vector, dtype=np.float64) for r in g_recs}
x_by_id = {str(r.id): np.array(r.vector, dtype=np.float64) for r in x_recs}
kept_ids = [pid for pid in ids if pid in g_by_id and pid in x_by_id]
mat_g = np.stack([g_by_id[pid] for pid in kept_ids])   # (N', 64)
mat_x = np.stack([x_by_id[pid] for pid in kept_ids])
dropped = len(ids) - len(kept_ids)
```
- 行序 = `ids` 原始顺序的保留子序列 → 两侧矩阵行对齐。
- `dropped>0` 时返回 `dropped` 统计（CLI 打印、WebUI 展示）；`kept==0` 抛 RuntimeError。
- 向量维度防御：非 64 维时抛 ValueError（与 `searcher.py:113` 一致）。
- Qdrant 不可达/连接异常向上传播，由 CLI/WebUI 各自捕获转错误信息。

**理由**：`retrieve` 只下载 N 个向量（≤600×64×8B≈0.3MB）；行序确定性保证两矩阵像素级对应，热力图可比。

### D4: 余弦相似度矩阵（numpy 向量化）

```python
def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0                      # 零向量防御（避免 NaN）
    v = vecs / norm
    return v @ v.T                              # (N', N')，对角 1.0
```
- 与 Qdrant COSINE 度量一致（`distance=cos_sim=dot/|a||b|`）。
- N≤600 → 矩阵 ≤600×600×8B≈2.9MB，无内存/性能问题，无需 GPU/分块。
- 对称性天然成立（`v@v.T` 对称）；对角恰为 1.0（向量已归一化）。

### D5: `visualization.py` 并排热力图

```python
def plot_similarity_heatmap_pair(mat_g, mat_x, save_path, collection_names=("google_aef_embedding", "xian_aef_embedding")):
    vmin = min(float(mat_g.min()), float(mat_x.min()))
    vmax = max(float(mat_g.max()), float(mat_x.max()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    for ax, mat, name in ((ax1, mat_g, collection_names[0]), (ax2, mat_x, collection_names[1])):
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name} 相似度热力图")
        ax.set_xlabel("样本索引"); ax.set_ylabel("样本索引")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"N={mat_g.shape[0]}×{mat_g.shape[0]} 余弦相似度矩阵对比（统一色阶）")
    ...
```
- 统一 vmin/vmax（取两矩阵全局极值）保证并排视觉可比——spec 明确要求。
- 中文字体复用模块顶部 `plt.rcParams` 配置；`fig.savefig(save_path, dpi=150, bbox_inches="tight")`。

### D6: CLI `similarity-heatmap` 子命令

参数（与现有子命令风格一致）：
```
--n            int, default 200（校验 1..600，超限 argparse error）
--seed         int, default 42
--image-id     str, optional（缺省数据库模式）
--output       str, default "similarity_heatmap.png"
--google-collection str, default DEFAULT_COLLECTION
--xian-collection  str, default "xian_aef_embedding"
--qdrant-url   str, default QDRANT_URL
```
- `cmd_similarity_heatmap`：构造两个 manager（google/xian collection）→ 健康检查（不可达返回 1）→ 集合存在性检查 → 调用 `compare_similarity_heatmaps` → 打印输出路径与 `sampled/kept/dropped` → 返回 0；异常（ValueError/RuntimeError/ConnectionError/qdrant 异常）打印明确错误返回 1。
- 图片模式校验 image_id（D2）。N 校验放在 parser 层（`type` 校验函数）与模块函数层双保险。

### D7: WebUI 面板（固定预置对）

- 新增「相似度热力图对比」expansion（评估面板下方）：N 输入（默认 200，范围校验）、seed 输入、模式单选（`ui.radio` 数据库/单张图片）、图片下拉（单张图片模式时可见，选项来自 `_imported_image_ids()`）、执行按钮、进度文本、结果图片容器。
- 目标集合固定 `PRESET_COLLECTIONS` 的预置对（`google_aef_embedding` × `xian_aef_embedding`），**与 `_current_collection` 选择器无关**（用户确认）。
- 执行：`ui.run.io_bound(compare_similarity_heatmaps, ...)`（与评估面板一致的异步模式，不阻塞事件循环）；完成后读 PNG 用 `ui.image(...)` 内嵌展示；失败 `ui.notify(type="negative")` + 面板内错误文本。
- PNG 生成到临时文件（`tempfile.NamedTemporaryFile` 或内存 `io.BytesIO`→base64→`ui.image`）；**决策**：用 `io.BytesIO` + base64 直接渲染，不落盘（WebUI 场景无需持久文件；CLI 场景才落盘）。`visualization.py` 函数签名支持 `save_path: str | Path | io.BytesIO`（matplotlib `savefig` 原生支持类文件对象）。

### D8: 编排函数返回契约

`compare_similarity_heatmaps` 返回 dict：
```python
{
  "sampled": int,       # 请求的 n（或实际候选数）
  "kept": int,          # 剔除后行数 N'
  "dropped": int,       # 单侧缺失剔除数
  "matrix_shape": [N', N'],
  "elapsed_sec": float,
  "output_path": str,   # CLI 落盘路径；WebUI 传 BytesIO 时为空（图已渲染）
}
```
- CLI 打印全部字段；WebUI 展示 `sampled/kept/dropped` 与图。

### D9: 可选导出（npy 矩阵 + JSON 采样信息）

**需求来源**：用户新增——将两个相似度矩阵导出为对应 collection 命名的 npy 文件，将采样参数与采样像素信息总结为 JSON。

**决策**：
- `compare_similarity_heatmaps(..., export_dir="outputs")` 新增可选参数 `export_dir`，**默认 `"outputs"`**（相对项目根，自动 `mkdir`）；显式传 `None`/空串时禁用导出（只生成热力图 PNG）——修复后为**默认导出**，与既有版本行为变化需回归。
- `extract_embeddings` 返回值从 3 元扩展为 4 元 `(mat_g, mat_x, dropped, kept_records)`：`kept_records` 为保留像素的信息列表（每项 `{point_id, image_id, pixel_row, pixel_col, utm_easting, utm_northing, utm_zone}`，从 google 侧 `retrieve` payload 提取——同位置两集合 UTM 一致，且与 `kept_ids` 行序严格对齐）。
- 新增 `export_similarity_outputs(sim_g, sim_x, meta, pixels, export_dir, collection_names)`：
  - `np.save(export_dir / f"{collection_names[0]}_similarity.npy", sim_g)`
  - `np.save(export_dir / f"{collection_names[1]}_similarity.npy", sim_x)`
  - `json.dump({"params": {...}, "pixels": pixels}, export_dir / "similarity_sampling.json")`（`ensure_ascii=False, indent=2`）
- JSON `params` 含 `{n, seed, image_id, collections: [g, x], sampled, kept, dropped, elapsed_sec}`；`pixels` 数组与矩阵行序一致，单侧剔除后仅含保留的 N' 个像素。
- **CLI**：`similarity-heatmap` 的 `--export-dir` 默认 `"outputs"`（可传 `--export-dir ""` 禁用）；导出时打印导出文件路径。
- **WebUI**：面板「导出目录」输入框默认值 `"outputs"`（留空禁用），完成后展示导出的文件路径列表。
- **理由**：npy 与 Qdrant collection 名对应（`<collection>_similarity.npy`），用户可直接 `np.load` 复用；JSON 记录可复现参数与像素明细，便于下游分析；默认导出省去每次配置，符合"评估后自动留存数据"的诉求。
- **备选**：保持可选导出（默认不导）——用户明确要求改为默认导出，不采纳。

## Risks / Trade-offs

- [数据库模式候选池依赖采样地图；地图缺失/损坏时 `ensure_sampling_map` 重建需数分钟（全量 scroll label）] → 复用既有自动重建语义；CLI 打印/WebUI notify 提示"采样地图重建中"；重建期间不阻塞其他功能（读操作）。
- [两集合未来 image 集分叉 → 采样点单侧缺失] → D3 剔除 + `dropped` 统计；`dropped==sampled`（全缺）抛错；spec 已覆盖"单侧缺失点剔除"。
- [N=600 时 600×600 格子过密难以目视] → imshow 可渲染，PNG dpi 150；上限锁定 600（spec），用户可后续放宽。
- [图片模式 `image_id` 在两集合均存在但 google 集合该影像像素数 < N] → 实际按 min(N, 16384) 采样（网格内不重复），`sampled` 记录实际值。
- [WebUI `io.BytesIO` 渲染依赖 matplotlib savefig 支持类文件对象] → matplotlib 原生支持；测试覆盖 BytesIO 分支。
- [两个 manager 并发调用 Qdrant（一次 retrieve 两侧）] → 串行两次 retrieve（N 小、毫秒级），避免并发复杂度；失败语义清晰。
- [导出目录不存在/不可写] → `export_similarity_outputs` 先 `mkdir(parents=True, exist_ok=True)` 再写；写失败向上抛，由 CLI/WebUI 转错误信息（CLI/WebUI except 元组含 OSError）。
- [`extract_embeddings` 返回值扩展为 4 元破坏既有调用方] → 仅 `compare_similarity_heatmaps` 内部消费；`similarity_compare.py` 内既有测试同步更新解包。
- [WebUI 导入用旧 `state["data_dir"]` 导致导入错误数据目录（实测 xian collection 存了 google 数据）] → D10 修复：`do_import` 从输入框取值同步 state；分页切换按 collection 映射默认目录。
- [`data_google` / `data_xian` 目录不存在（用户环境）] → 分页联动后输入框显示默认路径但目录缺失时，导入/浏览给出明确错误提示（不静默失败）；目录由用户准备（只修代码不碰数据）。

### D10: WebUI 分页默认数据目录联动 + 导入目录同步（Bug 修复）

**需求来源**：实测 bug——`xian_aef_embedding` collection 存了 google 数据（`data/` 的 all_mean 向量，identical），热力图与 GOOGLE 完全一致。根因：`webui.py` 的 `do_import()` 用 `state["data_dir"]`（只在点「浏览」时由 `browse_directory` 更新），用户改输入框没点浏览 → 导入旧目录。用户要求修复 + 分页默认目录 `data_google` / `data_xian` + 默认导出 `outputs/`。

**决策**：
- `do_import` 导入前从 `dir_input.value` 解析目录并同步 `state["data_dir"]`（与「浏览」同源），目录不存在时明确报错，杜绝"改输入框不生效"。
- 新增 collection → 默认数据目录映射（放 `config.py` 或 `webui.py` 模块级）：
  ```python
  COLLECTION_DATA_DIRS = {
      "google_aef_embedding": "data_google",
      "xian_aef_embedding": "data_xian",
  }
  ```
  分页切换 `_apply_collection` 时若新 collection 在映射中，更新 `dir_input.value`（经闭包 hook `sync_data_dir`）与 `state["data_dir"]`；自定义 collection 保持 `_CLI_DATA_DIR`。
- **理由**：`image_id` 从文件名坐标段提取（`data_loader.py:22`），与文件名前缀无关——`data/SE/all_mean_*.npz` 与 `data_xian/SE/xian_aef_*.npy` 同坐标产生相同 point_id；因此**同坐标段在两目录间会被断点续传误判为"已存在"而跳过**（`check_image_count` 按 image_id 精确计数），这也是导入串数据的放大器。分页默认目录 + 导入读输入框从源头避免。
- **备选**：不做联动只修 do_import——用户仍可能手动填错目录；映射表给出明确默认，降低误操作。

### D11: 坐标段数值匹配配对（导入修复）+ D11a: image_id 全链路归一化

**D11 需求来源**：实测 `data_xian` 仅导入 499 张——SE 文件名坐标段为 4 位小数（如 `E121.4033_N25.1370`）而 DW 为 3 位小数（`E121.4033_N25.137`），`scan_directory` 按字符串精确配对失败 125 对。

**D11 决策**：
- `data_loader.py` 的 `scan_directory`：SE/DW/TIF 分别扫描后，以 `parse_location_coord(raw_key)` 得到的 `(lon, lat)` 浮点元组作为**配对 key**（数值比较），替代原字符串 key。
- TIF 配对同步数值化（一致性）；孤儿文件（数值无对应）保持跳过语义。

**D11a 需求来源（用户后续选择）**：D11 后 `image_id` 仍取 SE 原始字符串，两集合（google 混合精度如 `E121.403_N25.1601` vs xian 全 4 位如 `E121.4030_N25.1601`）的 image_id 字符串不一致 → point_id 不一致 → 双集合对比缺失 125 张。用户选择**全链路归一化（需重导）**：image_id 也用数值归一化串，保证两集合 point_id 一致，代价是需清空重导两个 collection。

**D11a 决策**：
- `data_loader.py` 新增 `normalize_location_key(raw_key) -> str`：解析 `(lon, lat)` 后 round 到 4 位小数、格式化为去尾随零的字符串（如 `E121.4033_N25.1370` → `E121.4033_N25.137`；`E121.4030_N25.1601` → `E121.403_N25.1601`）。
- `scan_directory` 的 `ImagePair.image_id` 改用 `normalize_location_key(raw_key)`（替代 SE 原始串）。
- 影响：`point_id = uuid5(uuid5(DNS, image_id), "row_col")` 随 image_id 变化 → **已导入数据 point_id 不再匹配新格式** → 需清空重导 `google_aef_embedding` / `xian_aef_embedding`（manifest / 采样地图 / corpus 缓存按可重建语义自动重建）。
- **验证**：真实数据 624/624 归一化无冲突；两集合归一化后交集 624（修复前 499）。
- **理由**：字符串精度差异是数据侧命名噪声，数值归一化从源头统一 key；两集合 point_id 对齐是双集合对比（本 change 核心功能）正确性的前提。
- **风险**：归一化 round 4 位是否可能使两个不同坐标折叠为同一 key → 实测 624 唯一无冲突；`compute_utm_grid_from_name` / `parse_location_coord` 解析数值不受格式影响（数值不变）。

## Migration Plan

- 纯新增能力，无数据/文件迁移；无回滚路径（新子命令/新面板不影响既有命令）。
- 部署即代码合并；`similarity_compare.py` 新增，`visualization.py`/`cli.py`/`webui.py` 增量修改。
- **D11a 重导要求**：升级后需清空 `google_aef_embedding` 与 `xian_aef_embedding` 两 collection，分别用 `data_google` / `data_xian`（WebUI 分页默认目录）重新导入；本地 manifest / 采样地图 / corpus 缓存自动重建。

### D12: 可视化探索按检索 collection 定位影像文件（回归修复）

**需求来源**：用户报告「可视化探索的各通道图像有问题，GOOGLE 提取的向量可视化背景图与 XIAN 一样」。根因（实证）：Task 11 image_id 归一化后，两 collection 的 image_id 完全重叠（624/624），`state["se_paths_map"]`（全局单例，只在「浏览」时更新）的 key 冲突——GOOGLE 检索后可视化会取到最后浏览的 XIAN 目录的 SE 文件。

**决策**：
- `do_search` 记录检索 collection：`state["search_collection"] = state["manager"].collection_name`。
- `_show_visualization` 启动时按 `search_collection` 解析数据目录：`COLLECTION_DATA_DIRS.get(col)`（如 `data_google`/`data_xian`）回退 `_CLI_DATA_DIR`；`scan_directory` 构建局部 `viz_se_map`（image_id → SE path）。
- `_refresh_viz` / `on_mouse` 用 `viz_se_map` 查找（替代 `state["se_paths_map"]`），不再受全局浏览目录污染。
- **理由**：归一化使 image_id 双集合一致（point_id 对齐的前提），但可视化需要按 collection 区分数据源；按检索 collection 定位是唯一正确来源。`scan_directory` 每次可视化调用（≤624 文件 glob，毫秒级）可接受，不引入缓存复杂度。
- **风险**：自定义 collection 无映射 → 回退 `_CLI_DATA_DIR`（与分页默认目录行为一致，D10）；可视化性能（glob 624 文件）→ 毫秒级，可接受。
- **备选**：按 collection 隔离维护多个 se_paths_map —— 状态复杂；或检索时携带 SE 文件路径 —— 检索结果只含 payload 无文件路径。按检索 collection 扫描最直接。

## Open Questions

无——采样模式、N 上限、度量、形态、WebUI 集合语义均已与用户确认（open 探索 + 本轮 brainstorming）。

### D13: payload 索引自动补齐（防检索超时加固）

**需求来源**：用户报告「GOOGLE 检索后切 XIAN 页检索 timed out（K1000 + UTM 过滤）；重启后先 XIAN 检索直接报错」。根因（实证）：`xian_aef_embedding` collection 的 payload schema 缺 `utm_easting`/`utm_northing` 索引（历史创建/重建时未建，代码 `create_payload_indices` 只在创建时调用、不会对已存在 collection 补建）。UTM 过滤在无索引字段上全量扫描 1023 万点 → 5.03s > `QDRANT_TIMEOUT=5s` → `timed out`。google collection 有索引 → 139ms 正常。补建索引后 xian 检索恢复 274ms。

**决策**：
- `QdrantManager` 新增幂等方法 `ensure_payload_indices()`：读 `get_collection().payload_schema`，对缺失的 5 个字段（label/label_name/utm_easting/utm_northing/image_id）逐个 `create_payload_index`，已有则跳过。**不改动** `create_payload_indices`（保持创建语义与既有测试：5 次调用断言）。
- WebUI `_apply_collection`（分页切换）与页面加载路径对当前 collection 幂等调用 `ensure_payload_indices()`——廉价（一次 `get_collection`，仅缺失才建），历史 collection 缺索引自动修复。
- CLI 不加固：CLI 检索命令每次独立，且 `import_directory` 已调用 `migrate_image_id_index`；UTM 检索主要在 WebUI。
- **理由**：索引是 collection 层持久状态，创建后不随代码自动补；幂等补齐是低成本防御，防历史/手工重建的 collection 因缺索引全量扫描超时。
- **风险**：`create_payload_index` 对 1023 万点建索引耗时（实测首次超 5s 默认 timeout）→ `ensure_payload_indices` 用长 timeout 的 client 调用（或复用 `QDRANT_TIMEOUT` 但捕获超时并提示）；幂等检查先读 schema，仅缺失才建。
