# 向量数据库驱动的像素级 KNN 评估系统 — 需求方案

## 背景

本项目需要将卫星遥感影像的逐像素数据分析加载到向量数据库中，以便进行高效的 K 近邻（KNN）检索和评估。数据来源于两个卫星数据集：

- **Satellite Embedding V1 (SE V1)**：每张 `(128, 128)` 的影像提供 `(64, 128, 128)` 的张量，即 128×128 = 16,384 个像素，每个像素对应一个 64 维的 embedding 向量。
- **Dynamic World V1 (DW V1)**：每张 `(128, 128)` 的影像提供 `(128, 128)` 的矩阵，每个像素对应一个 0-8 的九选一土地覆盖标签。

随着影像数量的累积，像素总量将达到 **千万至亿级**。因此需要引入向量数据库来承载这些数据，并支持以下核心操作：

- 基于 64 维向量的近似最近邻（ANN）检索；
- 基于标签（土地覆盖类型）和地理坐标（UTM 投影）的标量过滤；
- 对 ANN 检索结果进行 KNN 评估（Recall@K 计算），以量化近似搜索相对精确搜索的准确度损失。

## 数据模型

每个像素作为向量数据库的一条记录（Point），包含以下字段：

| 字段名 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `id` | string / uuid | 自动生成或基于坐标哈希 | 像素唯一标识 |
| `vector` | float[64] | SE V1 `.npy` / `.npz` | 64 维 embedding |
| `label` | int (0-8) | DW V1 `.npy` / `.npz` | 土地覆盖类型数值编码 |
| `label_name` | string | 查表映射 | 对应名称：`water`, `trees`, `grass`, `flooded_vegetation`, `crops`, `shrub_and_scrub`, `built`, `bare`, `snow_and_ice` |
| `utm_easting` | float | 根据像素位置和影像地理参考计算 | UTM 东向坐标（米） |
| `utm_northing` | float | 同上 | UTM 北向坐标（米） |
| `utm_zone` | int | 同上 | UTM 投影带号 |
| `image_id` | string | SE/DW 文件名 | 该像素所属的 `(128, 128)` 影像标识 |
| `pixel_row` | int | 像素在影像中的位置 | 行号 (0-127) |
| `pixel_col` | int | 像素在影像中的位置 | 列号 (0-127) |

### 标签映射表

| 值 | 名称 | 说明 |
|---|---|---|
| 0 | `water` | 水体 |
| 1 | `trees` | 树木 |
| 2 | `grass` | 草地 |
| 3 | `flooded_vegetation` | 被淹植被 |
| 4 | `crops` | 农作物 |
| 5 | `shrub_and_scrub` | 灌木和矮灌木 |
| 6 | `built` | 建成区 |
| 7 | `bare` | 裸地 |
| 8 | `snow_and_ice` | 积雪和冰 |

## 向量数据库选型

### 选定方案：**Qdrant**

基于对 Qdrant 和 Milvus 的调研，Qdrant 更适合本项目的理由如下：

1. **部署极简**：单二进制，`docker run qdrant/qdrant` 一条命令即可启动。相比之下 Milvus 需要 etcd + MinIO + Pulsar/Kafka 等多组件协调。
2. **原生支持精确搜索**：API 的 `search_params.SearchParams(exact=True)` 一行代码即可进行暴力 KNN，直接作为 ground truth 用于 Recall@K 评估。Milvus 必须单独建立 FLAT 索引。
3. **Payload 索引完善**：支持 `integer`、`float`、`keyword` 等类型的标量索引，完全覆盖 label 过滤和 UTM 坐标范围过滤的需求。
4. **面向千万~亿级数据的单机友好**：64 维是低维向量，HNSW 索引在千万级规模下的索引内存开销约为向量本身 1.5-2 倍，单节点 64-128GB 内存可承载亿级数据。
5. **Python SDK 成熟**：`qdrant-client` 库的 API 设计清晰，与 `numpy` 无缝对接，适合分析型工作流。

### 技术储备

若后续需要 GPU 加速或百亿级扩展，可在同等数据模型下迁移至 Milvus。数据模型（向量 + 标量 payload）在两者之间通用。

## 功能需求

### F1：数据导入管线

编写脚本，从 SE V1 和 DW V1 的 `.npy` / `.npz` 文件对中读取数据，逐像素解析并批量导入 Qdrant。

- 支持指定输入目录，自动匹配 SE 和 DW 的对应文件；
- 根据影像的地理参考元数据计算每个像素的 UTM 坐标；
- 批量 upsert（建议 10,000 条/批），显示导入进度条；
- 支持断点续传（根据 `image_id` 去重，跳过已导入的影像）；
- 输出导入统计：总像素数、每类标签的像素数、耗时。

### F2：向量 + 标量过滤联合检索

实现灵活的检索接口，支持：

- **纯向量检索**：给定一个 query vector（64 维），返回 Top-K 最近邻像素；
- **标签过滤 + 向量检索**：限定 label 为某一类或多类（如"只在水体像素中检索"）；
- **地理范围过滤 + 向量检索**：限定 UTM 坐标在某个矩形范围内；
- **组合过滤**：标签 + 地理范围的 AND 条件联合过滤后再做向量检索；
- 支持精确搜索（`exact=True`）和近似搜索（可配置 HNSW 参数）两种模式的切换。

### F3：KNN 评估模块（Recall@K）

基于 Qdrant 的 exact search 和 ANN search，实现 Recall@K 评估：

- 从数据集中随机抽样或按类别分层抽样生成 query 向量集合；
- 对每个 query，分别用 exact search 和 ANN search 获取 Top-K 结果；
- 计算 `Recall@K = |ANN结果 ∩ Exact结果| / K`；
- 评估不同索引参数（`m`、`ef_construct`、`ef_search`）对 Recall 和 QPS 的影响；
- 输出评估报告（表格 + 图表）。

### F4：批量导出与快照

- 支持按过滤条件（label、地理范围）导出像素数据为 `.npy` 格式；
- 支持导出所有像素数据的统计摘要（每类数量、坐标范围、向量范数分布等）。

## 非功能需求

### N1：数据规模
- 目标支持：1,000 万 ~ 1 亿条像素记录。
- 单批次查询响应时间 < 100ms（ANN 模式，Top-10）。
- 导入吞吐量 ≥ 50,000 像素/秒。

### N2：可靠性
- 导入过程异常中断后可恢复，不产生重复数据。
- 数据导入和检索不应因内存不足而崩溃。

### N3：可观测性
- 检索 API 输出包含耗时、命中标签分布等指标。
- 评估模块输出完整的可复现参数记录。

## 项目结构建议

项目已有 `KNN_evaluation/` 目录，建议在现有项目结构中扩展如下模块：

```
KNN_evaluation/
├── __init__.py
├── qdrant_client.py          # Qdrant 连接管理与 collection 创建
├── data_loader.py            # SE/DW .npy 文件读取与像素解析
├── importer.py               # 批量导入管线（含断点续传）
├── searcher.py               # 联合检索接口（向量 + 标量过滤）
├── recall_evaluator.py       # Recall@K 评估逻辑
├── label_mapping.py          # 土地覆盖标签映射表
├── coordinate_utils.py       # UTM 坐标计算工具
├── config.py                 # 配置管理
├── cli.py                    # 命令行入口
└── tests/
    ├── test_data_loader.py
    ├── test_importer.py
    ├── test_searcher.py
    └── test_recall_evaluator.py
```

## 技术依赖

```toml
# pyproject.toml 中需新增的依赖
dependencies = [
    "qdrant-client>=1.12",       # Qdrant Python SDK
    "numpy>=1.26",               # 向量数据处理
    "tqdm>=4.66",                # 进度条
    "matplotlib>=3.8",           # 评估图表
    "pyproj>=3.6",               # UTM 坐标投影计算
    "rasterio>=1.3",             # 读取地理参考元数据（如需从 GeoTIFF 获取坐标）
]
```

## 待决策事项

以下问题建议在设计阶段明确：

1. **ID 生成策略**：像素 ID 使用 `{image_id}_{row}_{col}` 字符串拼接，还是使用 UUID/ULID，还是用 UTM 坐标的数值 hash？
2. **Qdrant Collection 分区策略**：是创建单一 collection 存放所有像素，还是按 `utm_zone` 或 `label` 创建多个 collection？
3. **坐标来源**：地理参考元数据是从 GeoTIFF 文件读取，还是从独立的元数据文件中读取？SE V1 和 DW V1 的数据格式请具体说明。
4. **Query 向量来源**：KNN 评估的 query 向量是从数据集中抽样，还是使用外部独立的 query 集？
5. **评估规模**：计划评估多少个 query 向量？每个 query 的 K 值范围？
