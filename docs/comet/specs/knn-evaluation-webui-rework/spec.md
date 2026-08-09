# 完整目标规格：KNN_evaluation WebUI 重构

本规格描述归档后 KNN_evaluation WebUI 的完整行为。

## 1. 页面结构（固定三页）

`KNN_evaluation/webui.py` 的 `index()` 渲染固定三个 tab 页面，页面集合不可由用户修改：

- **GOOGLE**：绑定 Qdrant Collection `google_aef_embedding`，数据目录 `data_google`。
- **XIAN**：绑定 Qdrant Collection `xian_aef_embedding`，数据目录 `data_xian`。
- **SimilarityMatrix**：相似度热力图对比页面（见 §5）。

### 1.1 移除自定义 collection 功能

- 不再有「自定义 collection 名称」输入框与「添加」按钮。
- 不再有「删除」按钮与删除确认对话框。
- `_known_collections` 固定为 `PRESET_COLLECTIONS`（`["google_aef_embedding", "xian_aef_embedding"]`），
  不再动态增删。
- 移除 `_add_custom_collection`、`_delete_custom_collection`、`_confirm_delete_collection`
  及 localStorage collection 记忆/恢复机制（页面即 collection，无需记忆）。

### 1.2 页面与内容

每个页面在其 `ui.tab_panel` 内独立构建全部内容（不是「空 tab + 下方统一渲染」）：

- **GOOGLE 页**：Qdrant 连接 & Collection 状态、数据导入、向量检索、评估面板，全部
  以 `google_aef_embedding` 为当前 collection、`data_google` 为数据目录。
- **XIAN 页**：同样四块内容，以 `xian_aef_embedding` 为当前 collection、
  `data_xian` 为数据目录。
- 页面切换只隐藏/显示对应 panel，不重建 panel 内控件；各页状态在页面生命周期内保留。

## 2. 页面隔离（需求 4）

- 每个页面持有独立的模块级状态字典（key 如 `state["pages"]["google"]`、
  `state["pages"]["xian"]`），包含各自的 manager、评估结果、查询向量、导入进度、
  文件对列表等。
- **评估结果互不覆盖**：GOOGLE 页完成评估后结果存入 GOOGLE 页状态；XIAN 页完成
  评估后结果存入 XIAN 页状态。切页后回到原页，原页评估结果仍显示，不被另一页覆盖。
- 各页的「导出 JSON / 导出图片图表」只导出本页状态中的结果。
- 各页数据目录互不干扰（GOOGLE 页浏览/导入只作用于 `data_google`，XIAN 页只作用于
  `data_xian`），与现有 `COLLECTION_DATA_DIRS` 映射一致。
- 检索可视化沿用现有按检索 collection 解析数据目录的机制，保证 GOOGLE/XIAN 结果
  不串集。

## 3. 数据导入自动续导（需求 1）

- WebUI 导入入口（GOOGLE/XIAN 页各自的「导入全部」）在 `import_directory` 抛出
  异常（连接失败、服务端错误等持久性失败）时，**自动整体重试**整个导入流程：
  - 最多 3 次整体重试（即最多 1 + 3 次尝试），失败间指数退避（如 2s、4s、8s）；
  - 每次重试沿用现有按影像 `check_image_count` 断点续传：已完整导入的影像跳过，
    部分导入的影像按 count 判定后覆盖重传；
  - 重试期间进度条与提示继续更新（提示「第 i/n 次尝试…」）。
- 全部尝试耗尽后，向用户提示最终失败原因，导入中止并复位进度 UI。
- 单张影像持久性失败不单独跳过；仍走整体中止 → 自动整体重试路径（决策 D1）。
- 现有 `PixelImporter.import_directory` 与 `_retry_call` 的瞬时错误重试保持不变。

## 4. 评估面板与导出（需求 5）

### 4.1 混淆矩阵图片展示

- GOOGLE/XIAN 页评估面板的混淆矩阵由表格改为 **matplotlib PNG 图片**展示
  （与 LinearProbe_evaluation 的 `confusion_matrix_base64` 模式一致）。
- 实现方式：在 `KNN_evaluation/visualization.py` 新增混淆矩阵 base64 辅助
  （复用现有 `plot_confusion_matrix` 渲染 → PNG bytes → data URI），评估完成后在
  面板内 `ui.image` 展示。
- 移除混淆矩阵表格渲染（`ui.table` 相关列/行构造），保留 per-class metrics 表格。

### 4.2 导出 JSON 与导出图片图表

- 评估面板提供两个导出按钮：**导出 JSON** 与 **导出图片图表**（需求 5）。
- **导出 JSON**：把评估结果（config + f1 + f2）写入对应页面目录：
  - GOOGLE 页 → `outputs/evaluation/google_aef`；
  - XIAN 页 → `outputs/evaluation/xian_aef`；
  - 文件名与图片成组（同一时间戳前缀），如 `evaluation_<timestamp>.json`。
  - 目录不存在时自动创建；不再走浏览器下载。
- **导出图片图表**：把评估图表 PNG 写入同一页面目录：
  - 混淆矩阵 PNG（`confusion_matrix.png` 或带时间戳）；
  - Purity & Recall 曲线 PNG（复用/适配 `plot_purity_recall_curve` 渲染 f2 数据）；
  - 均为 PNG 格式，文件名与 JSON 成组（同一时间戳前缀）。
- 导出成功后向用户提示文件路径列表；失败时提示错误。

## 5. SimilarityMatrix 页面（需求 3）

- 独立 tab 页面，包含原「相似度热力图对比」面板全部能力：
  - 采样数 N（1..600）、Seed；
  - 采样模式 radio：数据库全库 / 单张图片（含影像下拉，来自 manifest 缓存）；
  - 「生成热力图对比」按钮：固定对比 `google_aef_embedding` × `xian_aef_embedding`
    预置对（复用 `similarity_compare.compare_similarity_heatmaps`），页面内展示
    并排热力图 PNG。
- 移除可编辑「导出目录」输入框：生成结果自动导出到 `outputs/evaluation/similarity`：
  - 热力图 PNG（`similarity_heatmap.png` 或带时间戳）；
  - 相似度矩阵 npy × 2 与 `similarity_sampling.json`（沿用现有
    `export_similarity_outputs`，export_dir 固定为 `outputs/evaluation/similarity`）。
- 页面状态（N、Seed、模式、结果图）在页面生命周期内保留。

## 6. 不变项

- CLI（`cli.py`）的 import / search / stats / evaluate / similarity-heatmap 行为不变。
- `LinearProbe_evaluation` 不变（其「导出 JSON」保持浏览器下载现状）。
- Qdrant 数据结构、payload schema、`build_points` 点构造逻辑不变。
- `importer.import_directory` 的调用签名（`no_resume`/`reindex`/`progress_callback`）
  不变；WebUI 传入的参数与现状一致（no_resume=False, reindex=True, progress 回调）。
- 相似度热力图对比的计算语义与集合对不变。
