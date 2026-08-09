# vector-scalar-search Specification

## Purpose
提供基于 Qdrant 向量数据库的灵活检索接口，支持 64 维向量与标量字段（土地覆盖标签、UTM 地理坐标）的联合查询，可切换精确搜索和近似搜索模式。
## Requirements
### Requirement: 纯向量检索
系统 SHALL 支持给定一个 64 维 query vector，返回 Qdrant collection 中 Top-K 个 cosine 距离最近邻像素的记录及 payload。

#### Scenario: Top-10 向量检索
- **WHEN** 传入一个有效的 64 维 float64 query vector 和 `k=10`
- **THEN** 返回 10 条最近邻记录，每条包含 `id`、`score`、`label`、`label_name`、`utm_easting`、`utm_northing`、`image_id`、`pixel_row`、`pixel_col`

#### Scenario: 向量维度不匹配
- **WHEN** 传入的 query vector 维度不等于 64
- **THEN** 返回明确的错误信息，说明期望维度和实际维度

### Requirement: 标签过滤检索
系统 SHALL 支持在向量检索前按土地覆盖标签进行标量过滤，可指定单个标签值或多个标签值的列表。

#### Scenario: 单个标签过滤
- **WHEN** 传入 query vector、`k=10` 和 `label_filter=[0]`（water）
- **THEN** 仅返回 `label=0` 的像素记录中的 Top-10 最近邻

#### Scenario: 多个标签过滤
- **WHEN** 传入 query vector、`k=10` 和 `label_filter=[0, 1]`（water 或 trees）
- **THEN** 仅返回 `label=0` 或 `label=1` 的像素记录中的 Top-10 最近邻

#### Scenario: 标签过滤无匹配
- **WHEN** 过滤条件 `label_filter=[8]` 但数据集中无 `snow_and_ice` 标签像素
- **THEN** 返回空结果集，并说明未找到匹配记录

### Requirement: 地理范围过滤检索
系统 SHALL 支持在向量检索前按 UTM 坐标矩形范围进行标量过滤。

#### Scenario: UTM 矩形范围过滤
- **WHEN** 传入 query vector、`k=10` 和 `utm_range={min_easting, max_easting, min_northing, max_northing}`
- **THEN** 仅返回 UTM 坐标落在矩形范围内的像素记录中的 Top-10 最近邻

#### Scenario: UTM 范围外无结果
- **WHEN** `utm_range` 指定的矩形区域与任何像素的 UTM 坐标均不重叠
- **THEN** 返回空结果集

### Requirement: 组合过滤检索
系统 SHALL 支持标签过滤和地理范围过滤的 AND 组合条件，在向量检索前先应用所有标量过滤。

#### Scenario: 标签 + UTM 组合过滤
- **WHEN** 传入 query vector、`k=10`、`label_filter=[0]` 和 `utm_range={...}`
- **THEN** 仅返回同时满足标签为 water 且 UTM 坐标在指定范围内的像素的 Top-10 最近邻

### Requirement: 精确搜索与近似搜索切换
系统 SHALL 支持通过参数切换精确搜索（暴力 KNN，`exact=True`）和近似搜索（HNSW，可配置 `ef_search` 参数）。

#### Scenario: 精确搜索模式
- **WHEN** 设置 `search_mode="exact"` 进行 Top-10 检索
- **THEN** 调用 Qdrant `SearchParams(exact=True)`，返回精确的 Top-10 结果

#### Scenario: 近似搜索模式
- **WHEN** 设置 `search_mode="ann"` 和 `ef_search=128` 进行 Top-10 检索
- **THEN** 调用 Qdrant HNSW 近似搜索，`ef` 参数为 128

### Requirement: 检索结果指标
系统 SHALL 在每次检索响应中附带耗时（毫秒）和命中结果的标签分布统计。

#### Scenario: 检索耗时和标签分布
- **WHEN** 完成一次 Top-10 检索
- **THEN** 结果中包含 `elapsed_ms`（检索耗时）和 `label_distribution`（命中结果中各标签值的计数）

### Requirement: 无向量不匹配
系统 SHALL 确保所有查询在标记删除（tombstone）模式下不返回已删除或不存在向量的记录。

#### Scenario: 无效向量过滤
- **WHEN** Qdrant collection 中存在被标记为已删除的点
- **THEN** 检索结果中不包含这些已删除的记录

