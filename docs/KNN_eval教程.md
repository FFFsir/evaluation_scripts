# KNN_eval 教程

> 面向 `KNN_evaluation` 包的项目结构分析、工作流程与 WebUI 使用说明。
> 核验时间：2026-08-09。本文档**全部以当前代码为准**，与旧 README 描述的差异点见
> [附录 10.2「现有文档与代码差异说明」](#102-现有文档与代码差异说明)。
> ⚠️ **CLI 已弃用**：本模块仅通过 WebUI（端口 8003）提供使用入口，`cli.py` 保留仅作代码参考。

## 目录

1. [项目概述](#1-项目概述)
2. [环境与依赖](#2-环境与依赖)
3. [WebUI 使用方法](#3-webui-使用方法)
4. [项目结构分析](#4-项目结构分析)
5. [数据模型](#5-数据模型)
6. [工作流程](#6-工作流程)
7. [输出物说明](#7-输出物说明)
8. [测试与诊断](#8-测试与诊断)
9. [常见问题（FAQ）](#9-常见问题faq)
10. [附录](#10-附录)

---

## 1. 项目概述

`KNN_evaluation` 是一套**像素级 embedding 向量检索与质量评估系统**：

- **数据来源**：卫星影像的 SE（Satellite Embedding）与 DW（Dynamic World 标签）文件，
  每张影像 128×128 像素，每个像素一个 **64 维** embedding 向量与一个 **0–8 地物类别**标签。
- **存储后端**：Qdrant 向量数据库（Docker 部署，REST `:6333` / gRPC `:6334`）。
  每张影像导入后产生 16,384 个点（Point），全量数据可达千万级。
- **检索**：支持精确（exact，暴力全量）与近似（ANN，HNSW）两种 KNN 检索，可叠加
  标签过滤与 UTM 地理范围过滤。
- **评估**：衡量 embedding 质量的两类指标——**F1**（KNN 分类准确率，含混淆矩阵）与
  **F2**（邻居纯度 Purity@K 与召回率 Recall 曲线）。
- **双集合对比**：google 与 xian 两套 embedding 集合因 point_id 确定性一致而天然按
  位置对齐，可对比二者的余弦相似度热力图。
- **交互入口**：NiceGUI Web 界面（`webui.py`，端口 8003）。

系统拓扑一览：

```
数据目录 (SE/ + DW/)
      │  扫描配对 + 加载
      ▼
PixelImporter ──批量 upsert──► Qdrant Collection（每像素一个 Point）
      │                            │
      │  manifest / sampling_map / corpus_cache（三类可重建缓存）
      ▼                            ▼
PixelSearcher ──► 交互检索（exact / ANN，标签 / UTM 过滤）
      │
      ▼
metrics（F1 准确率 + F2 Purity/Recall@K，GPU/CPU 分块）
      │
      ├──► visualization（混淆矩阵 / 曲线 / 热力图）
      └──► similarity_compare（google × xian 相似度矩阵对比）
```

---

## 2. 环境与依赖

- **Python ≥ 3.12.12**（`.python-version` 锁定），依赖管理使用 [uv](https://docs.astral.sh/uv/)。
- **Docker**：运行 Qdrant 服务容器。

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

- 关键依赖（`pyproject.toml`）：`qdrant-client>=1.12`、`torch>=2.2`（来自 CUDA 12.6
  wheel 源）、`nicegui>=3.15`（**WebUI 框架，非 Gradio**）、`numpy`、`matplotlib`、
  `rasterio`、`pyproj`、`pillow`、`tqdm`。
- 安装：项目根目录执行 `uv sync`。

---

## 3. WebUI 使用方法

### 3.1 启动

```bash
cd .\evaluation_scripts
uv run python KNN_evaluation/webui.py --port 8003
```

浏览器访问 **http://127.0.0.1:8003**。

- 可用参数仅两个：`--port`（默认 8003）、`--qdrant-url`（默认 `http://localhost:6333`）。
- Qdrant 需已启动（`docker ps | grep qdrant`）；若未启动，页面连接状态显示 ❌，
  需手动启动 Docker 容器。

### 3.2 界面总览

页面标题「Qdrant KNN 像素评估系统」，顶部固定 **3 个标签页**：

| 标签页 | 内容 |
|--------|------|
| **GOOGLE** | 绑定集合 `google_aef_embedding`，数据目录 `data_google` |
| **XIAN** | 绑定集合 `xian_aef_embedding`，数据目录 `data_xian` |
| **SimilarityMatrix** | 双集合相似度热力图对比（固定 google × xian） |

GOOGLE 与 XIAN 页面结构完全相同、状态完全隔离（各自的连接、数据、查询、评估结果互不
干扰），自上而下为 4 个面板 + 2 个对话框。

### 3.3 GOOGLE / XIAN 页操作

#### ① Qdrant 连接 & Collection 状态（默认展开）

- 打开页面自动连接并显示 ✅/❌ 状态、Collection 名称/总点数/分段数/已建索引向量数。
- Collection 不存在时出现**「创建 Collection」**按钮（点击即创建集合 + payload 索引）。
- **「构建向量索引」**按钮：后台重建全量 HNSW 索引（数据量大时耗时较长）。
- **「刷新状态」**：重新健康检查与统计。

#### ② 数据导入（默认收起）

1. 确认「数据目录」输入框路径（GOOGLE 页默认 `data_google`、XIAN 页默认 `data_xian`，
   相对项目根；可直接修改）。
2. 点击**「浏览」**扫描 SE/DW 文件对 → 下方预览列表逐行显示状态：
   - 📦 **待导入** / ⏳ **n/16384**（部分导入，可续传）/ ✅ **已导入**（满 16,384 像素）。
   - 每页 20 行，用**「上一页」「下一页」**翻页。
3. 点击顶部**「导入全部」**（按钮在数据目录行上方，避免翻过预览列表才能点击）：
   - 后台线程导入 + 线性进度条 + 状态文字；失败自动重试（最多 4 次尝试，退避 2s/4s/8s）。
   - 完成后弹出**导入统计**对话框（总像素数、新增/跳过影像、标签分布、耗时、速率）。
   - 自动刷新影像下拉列表与预览状态。

#### ③ 向量检索（默认展开）

1. 选择查询来源：
   - **随机选取**：可选标签多选后点**「从 Collection 随机获取」**，从全库随机抽一个像素
     作为查询向量。
   - **指定像素**：选影像（下拉）或手输影像名 + 行/列（0–127），点**「获取向量」**。
2. 设置检索参数：
   - **K (Top-K)**：默认 10。
   - **标签过滤**：多选类别；不选则不过滤。
   - **启用 UTM 过滤** + 东/北 min/max：选定查询向量后自动填入该影像的 UTM 覆盖范围。
   - **精确搜索**：默认开启（暴力全量，精确）；关闭则走 HNSW 近似（快），并可用
     **ef_search**（默认 64）调节精度/速度。
3. 点**「执行检索」**→ 弹出**检索结果**对话框：
   - 元数据（ANN/Exact 模式、耗时 ms、命中数）；**标签分布**（可展开）；**命中表格**
     （ID/Score/标签/影像/位置/Easting/Northing）。

#### ④ 评估面板（默认收起）

1. 设置参数：

   | 参数 | 默认值 | 说明 |
   |------|--------|------|
   | 每类采样数 | 500 | 每类采样查询像素数（9 类，范围 10–10000） |
   | K-F1 | 100 | KNN 分类器 K（范围 1–1000） |
   | K 值序列 | 10,20,50,100,300,1000 | Purity/Recall 曲线采样点（逗号分隔） |
   | Seed | 42 | 随机种子，保证可复现 |
   | 执行设备 | auto | `auto`（CUDA 可用则 GPU）/ `cuda` / `cpu` |
   | 显存预算(GB) | 16 | GPU 分块显存预算 |
   | CPU RAM 预算(GB) | 6 | CPU 回退内存预算 |

   （「参数说明」子面板内含各参数原理讲解。）

2. 点**「开始评估」**：后台执行「采样 → 合并 F1+F2 评估」；进度条显示采样与评估进度；
   可点**「中止评估」**（协作式中止，当前块完成后停止）。

3. 结果展示：**Overall Accuracy**、**Per-class Metrics 表格**、**Confusion Matrix**
   **图片**、**Purity@K 与 Per-class Recall 两张曲线图**。

4. 导出（结果出现后按钮可见）：文件名与目录见[第 7 节「输出物说明」](#7-输出物说明)。

#### ⑤ 可视化探索（检索结果对话框中点「可视化探索」）

- 每个命中影像 + 查询影像一个图片标签页；可选择**通道**（A00–A63）与**缩放**（1x/2x/4x）。
- 128×128 灰度底图：**红色十字准线**=查询像素，**编号彩色圆点**=命中像素，面板
  列出各命中点详情。
- **点击图中任意像素**可直接将其设为新查询点，重新渲染并自动更新 UTM 范围。

### 3.4 SimilarityMatrix 页

1. 设置**采样数 N**（1–600，默认 200）与 **Seed**（默认 42）。
2. 选择模式：
   - **数据库全库**（默认）：从全库采样地图候选池抽 N 个点。
   - **单张图片**：显示影像下拉框，选某张影像后在其 128×128 网格内采样。
3. 点**「生成热力图对比」**：后台执行双集合采样/提取/矩阵计算，面板内嵌展示
   google × xian 并排（1×2 统一色阶）余弦相似度热力图；热力图 PNG 自动保存到
   `outputs/evaluation/similarity/`（文件名见[第 7 节「输出物说明」](#7-输出物说明)）。
4. 对比成功后出现三个导出按钮（共用同一时间戳成组导出到 `outputs/evaluation/similarity/`）：
   - **「导出 JSON」** → `{full_col|single_img}_similarity_{时间戳}.json`
     （params + 按行序对齐的 pixels 采样信息）。
   - **「导出图片图表」** → 重新保存热力图 PNG。
   - **「输出 npy 文件」** → `{full_col|single_img}_google_similarity_{时间戳}.npy` 与
     `{full_col|single_img}_xian_similarity_{时间戳}.npy`（两个 N′×N′ 相似度矩阵，
     带时间戳不覆盖旧文件）。

> WebUI 固定对比 `google_aef_embedding × xian_aef_embedding`。

---

## 4. 项目结构分析

### 4.1 目录总览

`KNN_evaluation/` 是仓库根（`evaluation_scripts/`）下的独立包：

```
KNN_evaluation/
├── __init__.py              # 包标记（仅 docstring，无导出）
├── README.md                # 使用说明（部分内容已过时，见附录 10.2）
├── vector-db-knn-evaluation-requirements.md  # 需求方案设计文档
├── config.py                # 全局配置常量
├── webui.py                 # NiceGUI Web 界面（≈2000 行）
├── data_loader.py           # SE/DW/TIF 扫描、加载与配对
├── importer.py              # 批量导入 + 断点续传 + 重试
├── qdrant_client.py         # QdrantManager：Collection 生命周期与索引管理
├── manifest.py              # 导入清单缓存（JSON，原子写）
├── sampling_map.py          # 采样地图缓存（JSON，原子写）
├── corpus_cache.py          # 全量语料向量磁盘缓存（npz）
├── searcher.py              # 向量检索（exact / ANN + 标签/UTM 过滤）
├── gpu_knn.py               # GPU/CPU 分块精确 KNN 引擎（torch）
├── metrics.py               # F1 / F2 指标评估
├── similarity_compare.py    # 双集合相似度热力图对比（WebUI 共用）
├── coordinate_utils.py      # UTM 地理坐标推算（文件名或 GeoTIFF）
├── label_mapping.py         # 地物类别 0–8 ↔ 名称映射
├── visualization.py         # matplotlib 图表（混淆矩阵/曲线/热力图）
├── ui_pagination.py         # WebUI 预览列表翻页纯函数
├── cli.py                   # 命令行入口（已弃用，仅 WebUI 入口）
└── tests/                   # 22 个测试文件（355 个测试，见第 8 节）
```

### 4.2 模块职责一览表

| 模块 | 定位 | 核心内容 |
|------|------|----------|
| `config.py` | 全局配置 | Qdrant URL、Collection 名、向量维度、HNSW 参数、批量大小等常量 |
| `webui.py` | Web 界面 | NiceGUI 三页界面：GOOGLE / XIAN / SimilarityMatrix |
| `data_loader.py` | 数据层 | `PixelDataLoader`：SE/DW 加载、坐标提取与归一化、目录扫描配对 |
| `importer.py` | 数据层 | `PixelImporter`：构建 Point 并批量 upsert，断点续传、重试、重建索引 |
| `qdrant_client.py` | 存储层 | `QdrantManager`：建集/删集/建索引/迁移索引/manifest 对账/重索引 |
| `manifest.py` | 缓存层 | 导入清单 `qdrant_import_manifest_<collection>.json` |
| `sampling_map.py` | 缓存层 | 采样地图 `qdrant_sampling_map_<collection>.json` |
| `corpus_cache.py` | 缓存层 | 全量语料向量 npz 缓存 `qdrant_corpus_cache/` |
| `searcher.py` | 检索层 | `PixelSearcher`：向量检索 + 标签/UTM 过滤，exact/ANN |
| `gpu_knn.py` | 检索层 | `KnnEngine`：CUDA/torch-CPU 分块精确 KNN（`q @ corpus.T` + topk） |
| `metrics.py` | 评估层 | 分层采样、F1 准确率、F2 Purity/Recall@K、合并通道 `evaluate_knn` |
| `similarity_compare.py` | 评估层 | 双集合采样 → 提取 → 余弦矩阵 → 导出 npy/json + 热力图 |
| `coordinate_utils.py` | 工具 | UTM 网格推算（文件名坐标 / GeoTIFF 仿射变换） |
| `label_mapping.py` | 工具 | `LABEL_NAMES` / `LABEL_IDS`（0–8 ↔ 双语名称） |
| `visualization.py` | 可视化 | 混淆矩阵、Purity/Recall 曲线、相似度热力图（matplotlib Agg） |
| `ui_pagination.py` | 工具 | 导入预览翻页纯函数（`PAGE_SIZE = 20`） |
| `cli.py` | 已弃用 | 命令行入口（仅作代码参考，不提供使用说明） |

### 4.3 核心模块分组详解

**数据层（读取磁盘文件）**
- `data_loader.py`：`PixelDataLoader.scan_directory` 扫描 `SE/`、`DW/`、TIF 文件，
  从文件名提取坐标段 `E{lon}_N{lat}`，按数值坐标配对成 `ImagePair`；
  `load_se` 读取 64 通道 128×128 embedding（复用 `src/satellite_embedding_loader.py`），
  `load_dw` 读取 128×128 uint8 标签矩阵。
- `importer.py`：`PixelImporter.import_directory` 是整个导入流水线——健康检查 →
  迁移 image_id 索引 → 逐影像导入（每影像 16,384 点，按 `BATCH_SIZE=10000` 分批
  upsert，含指数退避重试与断点续传）→ 可选重建 HNSW 索引（WebUI「构建向量索引」按钮）。

**存储层（Qdrant 管理）**
- `qdrant_client.py`：`QdrantManager` 封装 Collection 创建（`on_disk` disk/ram）、
  payload 索引自愈（`ensure_payload_indices`）、image_id 索引 text→keyword 迁移
  （计数提速约 140×）、manifest 对账、HNSW 强制重建（`reindex_vectors`）。

**三类可重建缓存（同一套「可重建、非真相源、指纹自愈」模式）**

| 缓存 | 文件 | 用途 | 关键函数 |
|------|------|------|----------|
| 导入清单 | `qdrant_import_manifest_<collection>.json` | 逐影像已导入像素数（断点续传、WebUI 预览状态） | `manifest.py:update_manifest` |
| 采样地图 | `qdrant_sampling_map_<collection>.json` | `label → [point_id, ...]`，本地选点避免全量 2.6GB 下载 | `sampling_map.py:build_sampling_map` |
| 语料缓存 | `qdrant_corpus_cache/{sha256[collection][:16]}.npz` | 全量向量 `(N,64)`+标签+point_id，评估只读一次 | `corpus_cache.py:ensure_corpus_cache` |

三者均为**原子写入**（tmp + `os.replace`），并以 `collection 名 + total_points` 指纹
自动重建，安全可删。

**检索层**
- `searcher.py`：`PixelSearcher.search` 面向交互检索——过滤条件 AND 组合
  （`label` 用 `MatchAny`，UTM 用 `Range`），`exact=True` 走 Qdrant 精确搜索，
  `exact=False` 走 HNSW ANN（`ef_search` 可调）。
- `gpu_knn.py`：`KnnEngine` 面向**评估批处理**——全量语料上载设备后逐查询块
  `q_block @ corpus.T` + `torch.topk`，流式处理不物化 (Q,N) 大矩阵；
  `estimate_block_q` 按显存预算推导查询分块大小。

### 4.4 全局配置常量（`config.py`）

| 常量 | 值 | 说明 |
|------|-----|------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST 地址 |
| `DEFAULT_COLLECTION` | `google_aef_embedding` | 默认 Collection |
| `PRESET_COLLECTIONS` | `["google_aef_embedding", "xian_aef_embedding"]` | WebUI 固定两页的集合 |
| `COLLECTION_DATA_DIRS` | `{"google_aef_embedding": "data_google", "xian_aef_embedding": "data_xian"}` | 预置页数据目录（相对项目根） |
| `BATCH_SIZE` | `10000` | 导入 upsert 批次大小 |
| `VECTOR_SIZE` | `64` | 向量维度 |
| `HNSW_M` / `HNSW_EF_CONSTRUCT` | `16` / `100` | HNSW 建图参数 |
| `EF_SEARCH_DEFAULT` | `64` | ANN 检索默认 ef_search |
| `QDRANT_TIMEOUT` | `60` | Qdrant 客户端超时（秒） |
| `UTM_RESOLUTION_M` | `10` | UTM 坐标推算分辨率（米/像素） |

WebUI 默认端口 `8003` 定义在 `webui.py:DEFAULT_PORT`，不在此文件。

---

## 5. 数据模型

**每像素一个 Point**，导入时由 `importer.py:build_points` 构造：

- **id**：确定性 `uuid5(uuid5(NAMESPACE_DNS, image_id), f"{row}_{col}")`——同一
  image_id + 像素坐标在任何集合上生成相同 point_id，这是断点续传与双集合对齐的基石。
- **vector**：64 维 float32（SE 数据重排为 16,384×64）。
- **payload 字段**：

| 字段 | 类型 | 含义 |
|------|------|------|
| `label` | int | 地物类别 0–8 |
| `label_name` | str | 类别名称（如 `water 水`） |
| `utm_easting` / `utm_northing` | float | UTM 东/北坐标（米） |
| `utm_zone` | int | UTM 分带（南半球为负，未知为 -1） |
| `image_id` | str | 归一化后的影像标识（如 `E121.4033_N25.137`） |
| `pixel_row` / `pixel_col` | int | 影像内像素坐标 0–127 |

**image_id 归一化**（`data_loader.py:normalize_location_key`）：坐标段 round 4 位小数并
去尾随零（`E121.4033_N25.1370` → `E121.4033_N25.137`），保证 google（混合精度）与
xian（全 4 位）两集合的 image_id / point_id 一致。

**label 映射**（`label_mapping.py`）：`0 water 水`、`1 trees 树`、`2 grass 草`、
`3 flooded_vegetation 被淹植被`、`4 crops 农作物`、`5 shrub_and_scrub 灌木与矮树丛`、
`6 built 建筑物`、`7 bare 空地`、`8 snow_and_ice 冰雪`。

**UTM 坐标**（`coordinate_utils.py`）：默认以文件名坐标作为影像中心，NW 角对齐 10m
整数倍网格推算 128×128 像素坐标（pyproj）；GeoTIFF 仿射变换为回退路径。这些坐标支撑
检索的 UTM 范围过滤与热力图导出的像素地理信息。

---

## 6. 工作流程

### 6.1 总体数据流

```
① 数据导入        ② 采样/缓存             ③ 检索                ④ 评估
数据目录 ──► Qdrant ──► 采样地图 + 语料缓存 ──► PixelSearcher/KnnEngine ──► metrics
 (SE/DW)     Collection   (JSON/npz 重建式)     (exact/ANN, GPU/CPU)       (F1 + F2)
                                                                              │
                                                       ⑤ 相似度对比           ▼
                                              google × xian 热力图     ⑥ 可视化 / JSON 报告
```

### 6.2 阶段一：数据导入

**输入**：数据根目录，内含 `SE/` 与 `DW/` 子目录（`.npy`/`.npz` + 可选 `.tif`）。

**流程**（`PixelImporter.import_directory`，`importer.py:221`）：

1. 健康检查 Qdrant；必要时创建 Collection 与 payload 索引（label / label_name /
   utm_easting / utm_northing / image_id）。
2. 迁移 image_id 索引为 keyword（幂等，text→keyword 提速）。
3. `scan_directory` 扫描并配对 SE/DW 文件对。
4. 逐影像导入：断点续传检查（`check_image_count` 精确 count 该 image_id 已导入像素数）
   → 加载 SE+DW → 推算 UTM 网格 → 构建 16,384 个 PointStruct → 每 10,000 点一批
   upsert（`_retry_call` 指数退避：1s/2s/4s，最多 3 次重试，仅重试瞬时错误）。
5. 每影像完成后原子更新导入清单 manifest（`manifest.py:update_manifest`）。
6. 可选重建索引：`indexing_threshold=0` 强制重建全量 HNSW 索引（WebUI「构建向量索引」按钮）。

**幂等性**：同一数据目录重复导入会自动跳过已完整导入的影像（部分导入的影像
`⏳ n/16384` 续传）；WebUI「导入全部」可续传。

### 6.3 阶段二：语料采样与缓存

评估前避免反复全量下载 GB 级向量，采用两层机制：

- **采样地图**（`sampling_map.py`）：`ensure_sampling_map` 按指纹自动构建/重建
  `label → [point_id,...]` 映射，仅需拉取 id 列表。采样时本地随机选点后再
  `client.retrieve(ids, with_vectors=True)` 取回少量向量。
- **语料缓存**（`corpus_cache.py`）：`ensure_corpus_cache` 将全量向量/标签/point_id
  落盘 npz，评估的 GPU/CPU 通道共用。缓存失效（指纹不匹配）自动重建。

### 6.4 阶段三：KNN 检索

两种检索实现：

| 通道 | 实现 | 适用 |
|------|------|------|
| Qdrant 检索（`searcher.py:PixelSearcher`） | `query_points`，`exact` 或 HNSW ANN（`ef_search`），可叠加标签/UTM 过滤 | 交互式检索（WebUI 检索面板） |
| GPU/CPU 精确 KNN（`gpu_knn.py:KnnEngine`） | 全量语料 `q @ corpus.T` + `torch.topk`，查询分块流式计算 | 批量评估（WebUI 评估面板） |

`KnnEngine` 关键点：

- `resolve_device("auto")`：CUDA 可用则 GPU，否则 torch CPU 分块。
- `estimate_block_q(max_gpu_mem)`：按显存预算推导查询块大小，避免 OOM
  （约 `0.8 × (预算 − 语料占用GB) / (N × 4字节)`，限幅到 [1, N]）。
- `close()`：释放设备显存（`torch.cuda.empty_cache()`），防止重复评估累积 OOM。

**Top-K 语义**：返回相似度最高的 K 个点；评估流程取 K+1 个后按 point_id 剔除查询点
自身（Leave-One-Out）。

### 6.5 阶段四：指标评估

**采样**（`metrics.py:sample_queries_by_label`）：每类（共 9 类）以固定 seed 随机采样
「每类采样数」个点作为查询（默认 500，共约 4,500 查询）。

**F1 —— KNN 分类准确率**（`compute_knn_accuracy`）：

1. 对每个查询取 top-(K+1) 邻居，剔除自身；
2. 前 K 个有效邻居多数投票预测标签；
3. **平票裁决**（`_resolve_tie`）：依次用 K-1、K-3、K-5、K-7、K-9 重新计票，仍平则取
   最近邻；
4. 产出 9×9 混淆矩阵 → Overall Accuracy、每类 Precision/Recall/F1/Support、
   `accuracy_by_k`（单次 top-(max_k+1) 检索递增取 K，零额外检索）。

**F2 —— 邻居纯度与召回率曲线**（`compute_purity_recall_curve`，默认 K 序列
`10,20,50,100,300,1000`）：

- **Purity@K** = Top-K 邻居中与查询同标签的比例 → 随 K 增大**单调递减**；
- **Recall@K** = 前 K 邻居中召回的同类像素数 / **全局**同类像素总数（Qdrant `count`
  精确统计）→ 随 K 增大**单调递增**（标准 IR Recall@K，K 较小时天然很小）。

> Purity 与 Recall 方向相反是**正常现象**（Purity↓, Recall↑），恰恰说明指标在工作。

**合并通道**（`evaluate_knn`，`metrics.py:619`）：F1 与 F2 共用**一次**全量语料加载与
**一次** `topk(K=max(k_values)+1)` 检索，同时聚合两类指标——避免旧版双轮扫描/OOM 问题。
逐块检查 `cancel_event` 支持协作式中止（抛 `EvaluationCancelled`）。

### 6.6 阶段五：双集合相似度对比

`similarity_compare.py:compare_similarity_heatmaps`（WebUI 与 CLI 共用同一事实源）：

1. **采样**：数据库全库模式从采样地图候选池抽 N 个 point_id；单图模式在 128×128
   网格中抽 (row,col) 并按 uuid5 公式派生 point_id（`_point_id`）。
2. **提取**：同一批 point_id 分别在两个集合 `retrieve` 取 embedding，剔除任一侧缺失
   的点（行对齐）。
3. **矩阵**：各自计算 N′×N′ 余弦相似度矩阵（零向量安全、对角线恒 1.0）。
4. **输出**：并排（1×2）统一色阶热力图 PNG；WebUI 导出见[第 7 节「输出物说明」](#7-输出物说明)。

### 6.7 阶段六：可视化

`visualization.py`（matplotlib Agg 后端，中文使用 SimHei/微软雅黑）：

- `plot_confusion_matrix`：9×9 混淆矩阵热力图（无路径时返回 PNG bytes，供 WebUI 内嵌）。
- `plot_purity_recall_curve`：1×2 面板（Purity@K + Per-class Recall vs K，log-x）。
- `plot_similarity_heatmap_pair`：双集合并排热力图。
- WebUI 另有 `render_grayscale_with_markers`（`webui.py`）：128×128 灰度底图 + 查询点
  红色十字准线 + 命中点编号彩色圆点。

---

## 7. 输出物说明

### 7.1 WebUI 导出文件汇总

| 内容 | 目录 | 文件名模式 |
|------|------|------------|
| 评估 JSON | `outputs/evaluation/knn_eval/` | `{google\|xian}_knn_result_{ts}.json` |
| 混淆矩阵 / 曲线 PNG | `outputs/evaluation/knn_eval/` | `{google\|xian}_knn_cm_{ts}.png`、`{google\|xian}_knn_pr_{ts}.png` |
| 相似度热力图 PNG（自动保存） | `outputs/evaluation/similarity/` | `{full_col\|single_img}_similarity_heatmap_{ts}.png` |
| 相似度采样 JSON | `outputs/evaluation/similarity/` | `{full_col\|single_img}_similarity_{ts}.json` |
| 相似度矩阵 npy ×2 | `outputs/evaluation/similarity/` | `{mode}_google_similarity_{ts}.npy`、`{mode}_xian_similarity_{ts}.npy` |

（`ts` = `%Y%m%d_%H%M%S`。）

### 7.2 评估 JSON 报告结构

`{google|xian}_knn_result_{ts}.json` 的结构（WebUI「导出 JSON」产物）：

```json
{
  "config": { "samples_per_class", "k_f1", "k_values", "seed", "device", "gpu_batch_q", "max_gpu_mem", "max_eval_ram" },
  "f1": { "overall_accuracy", "per_class_metrics", "confusion_matrix", "k", "num_queries", "elapsed_sec", "accuracy_by_k" },
  "f2": { "k_values", "global_purity", "global_recall", "per_class_purity", "per_class_recall", "num_queries", "elapsed_sec" },
  "total_elapsed_sec": 0.0
}
```

---

## 8. 测试与诊断

`KNN_evaluation/tests/` 共 **22 个测试文件、355 个测试**，覆盖各模块；`conftest.py` 提供
session 级 `qdrant_manager` fixture（自动起 Docker 容器 `qdrant-knn-eval` 并灌入
确定性 9 类测试数据）。

```bash
# 全量测试（含需要真实 Qdrant 的 integration 用例）
uv run pytest KNN_evaluation/tests/ -v

# 跳过需要 Qdrant 服务的用例
uv run pytest KNN_evaluation/tests/ -v -m "not integration"
```

测试分布（按文件，`def test_` 计数）：test_cli 35 / test_config 4 /
test_coordinate_utils 12 / test_corpus_cache 17 / test_data_loader 21 / test_gpu_knn 10 /
test_import_retry 13 / test_importer 7 / test_importer_reindex 2 / test_manifest 18 /
test_manifest_in_import 3 / test_metrics 65 / test_migrate 5 / test_progress_callback 5 /
test_qdrant_client 27 / test_sampling_map 18 / test_searcher 6 / test_similarity_compare 28 /
test_start_qdrant 7 / test_ui_pagination 15 / test_visualization 3 / test_webui 34。

---

## 9. 常见问题（FAQ）

**Q：页面显示 Qdrant 不可达（❌）？**
A：确认 Docker 容器已启动：`docker ps | grep qdrant`；未启动则执行
`docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest`，
然后点「刷新状态」。

**Q：点击「导入全部」报错？**
A：检查数据目录下 `SE/` 与 `DW/` 子目录是否存在相匹配的 `.npy` 文件对（文件名需含
`E{lon}_N{lat}` 坐标段）。

**Q：「影像」下拉没有选项？**
A：先在「数据导入」面板点击「浏览」扫描数据目录，或完成一次导入后自动刷新。

**Q：检索结果为空？**
A：多半是过滤条件过严（如标签过滤选中了该集合不存在的类别、UTM 范围过窄）。
去掉标签过滤 / 关闭 UTM 过滤后重试。

**Q：评估很慢或显存不足？**
A：降低「每类采样数」；设备选 `cpu` 并调大「CPU RAM 预算(GB)」；或调小「显存预算(GB)」
让分块更小。GPU 评估结束后引擎会释放显存（`torch.cuda.empty_cache()`）。

**Q：双集合对比时 dropped 点较多？**
A：对比依赖两集合 point_id 一致。确认两集合数据均已按当前 image_id 归一化规则
（round 4 位去尾随零）**重新导入**（见[第 5 节「数据模型」](#5-数据模型)）；升级前后旧数据未重导会
导致大量单侧缺失。

**Q：如何彻底重跑数据？**
A：WebUI 的「创建 Collection」仅用于不存在时创建，没有删除按钮——如需删除重建 Collection，
请通过 Qdrant API 或相关运维手段完成（`migrate` 已随 CLI 弃用）。

---

## 10. 附录

### 10.1 关键函数索引

（行号以 2026-08-09 代码为准，供代码阅读定位。）

| 功能 | 函数 | 位置 |
|------|------|------|
| CLI 解析 / 分发（已弃用） | `_build_parser` / `main` | `cli.py:733` / `cli.py:850` |
| CLI 子命令（已弃用） | `cmd_import` / `cmd_search` / `cmd_stats` / `cmd_evaluate` / `cmd_similarity_heatmap` / `cmd_migrate` | `cli.py:100` / `168` / `276` / `363` / `297` / `668` |
| SE/DW 扫描配对 | `PixelDataLoader.scan_directory` | `data_loader.py:132` |
| image_id 归一化 | `PixelDataLoader.normalize_location_key` | `data_loader.py:61` |
| 构建 Point / 目录导入 | `PixelImporter.build_points` / `import_directory` | `importer.py:79` / `221` |
| 重试包装 | `_retry_call` | `importer.py:40` |
| Collection 生命周期 | `QdrantManager.create_collection` / `create_payload_indices` / `migrate_image_id_index` / `reindex_vectors` | `qdrant_client.py:59` / `93` / `165` / `275` |
| 清单 / 采样地图 / 语料缓存 | `update_manifest` / `build_sampling_map` / `ensure_corpus_cache` | `manifest.py:75` / `sampling_map.py:89` / `corpus_cache.py:192` |
| 检索 | `PixelSearcher.search` | `searcher.py:88` |
| GPU/CPU KNN 引擎 | `KnnEngine.knn_chunk` / `estimate_block_q` / `resolve_device` | `gpu_knn.py:75` / `93` / `11` |
| 采样 / F1 / F2 / 合并评估 | `sample_queries_by_label` / `compute_knn_accuracy` / `compute_purity_recall_curve` / `evaluate_knn` | `metrics.py:66` / `172` / `451` / `619` |
| 平票裁决 | `_resolve_tie` | `metrics.py:45` |
| 相似度对比编排 | `compare_similarity_heatmaps` / `sample_random_points` / `extract_embeddings` / `export_similarity_outputs` | `similarity_compare.py:236` / `43` / `113` / `185` |
| UTM 推算 | `compute_utm_grid_from_name` / `compute_utm_grid` | `coordinate_utils.py:91` / `48` |
| 图表 | `plot_confusion_matrix` / `plot_purity_recall_curve` / `plot_similarity_heatmap_pair` | `visualization.py:17` / `85` / `196` |
| WebUI 页面入口 | `index` / `init_page` / 评估 `do_evaluate` / 可视化 `do_visualize` / 热力图 `do_sim_compare` | `webui.py:610` / `208` / `1283` / `1584` / `1914` |
| WebUI 导出 | `_export_page_results` / `_export_similarity_results` | `webui.py:453` / `523` |

### 10.2 现有文档与代码差异说明

本教程正文一律以当前代码为准。以下为旧文档（`README.md` / 根 `README.md`）与代码
**不一致**之处，引用旧文档时请注意：

| # | 旧文档说法 | 当前代码实际 |
|---|------------|--------------|
| 1 | WebUI 启动命令含 `--dir data_demo` | `--dir` **未实现**，`parse_known_args` 静默忽略；数据目录由页面固定（GOOGLE→`data_google`、XIAN→`data_xian`）或输入框手动修改。正确启动命令见 [3.1](#31-启动) |
| 2 | 评估参数表「K-F1 默认 10」 | 实际默认 **100**（WebUI 为 100，范围 1–1000） |
| 3 | 根 README「Collection 选择器」：可添加自定义分页、切换记忆 localStorage、每分页「刷新」「清理缓存」按钮 | 当前代码**已移除**这些功能：仅固定 GOOGLE / XIAN / SimilarityMatrix 三个标签页，无自定义分页、无 localStorage、无刷新/清理按钮 |
| 4 | 相似度面板有「导出目录」输入框（默认 `outputs`，留空不导出） | 输入框**已移除**，导出目录固定 `outputs/evaluation/similarity`，导出由对比成功后的三个按钮触发 |
| 5 | 未明确 UI 框架 | WebUI 为 **NiceGUI 3.15**（非 Gradio；无 `demo.launch`、无 share/queue） |
| 6 | WebUI 自动启动 Qdrant（`_start_qdrant`） | `_start_qdrant()` 存在但**仅测试调用**，页面不自动启动容器；Qdrant 未启动时页面仅显示 ❌ |
| 7 | CLI 是使用方式之一（`knn-eval` 命令） | **CLI 已弃用**，仅 WebUI 入口；`cli.py` 保留作代码参考 |
