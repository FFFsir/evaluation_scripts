# pixel-data-import Specification

## Purpose
将 Satellite Embedding V1 和 Dynamic World V1 数据集的像素级数据批量导入 Qdrant 向量数据库，支持断点续传和进度跟踪。
## Requirements
### Requirement: 结构化数据文件读取
系统 SHALL 从 SE V1 的 `.npy`/`.npz` 文件中读取结构化 dtype（shape `(128,128)`，每元素含 64 个 float64 字段）并转换为 `(64, 128, 128)` float64 数组；从 DW V1 的 `.npy` 文件中读取 `dtype[('label', 'u1')]` 数据并转换为 `(128, 128)` uint8 标签矩阵。

#### Scenario: 加载 SE V1 npy 文件
- **WHEN** 调用数据加载器加载 `all_mean_E121.4025_N25.1947_2024.npy`
- **THEN** 返回 `(64, 128, 128)` 的 `float64` numpy 数组

#### Scenario: 加载 SE V1 npz 文件
- **WHEN** 调用数据加载器加载 `all_mean_E121.4025_N25.1947_2024.npz`
- **THEN** 自动识别 `embedding` key 并返回 `(64, 128, 128)` 的 `float64` numpy 数组

#### Scenario: 加载 DW V1 npy 文件
- **WHEN** 调用数据加载器加载 `label_mode_E121.4025_N25.1947*.npy`
- **THEN** 返回 `(128, 128)` 的 `uint8` 标签矩阵，值范围为 0-8

### Requirement: SE/DW 文件自动配对
系统 SHALL 根据文件名中的经纬度坐标段（`E{lon}_N{lat}`）自动匹配 SE 和 DW 对应文件对，并提取对应 GeoTIFF 文件计算地理参考。

#### Scenario: 扫描目录自动配对
- **WHEN** 指定包含 7 对 SE+DW 文件的 `data_demo` 目录
- **THEN** 识别出 7 对匹配的文件对，每对包含 SE 的 `.npy`/`.npz`、DW 的 `.npy` 及可选 GeoTIFF

#### Scenario: 无法配对的孤立文件
- **WHEN** SE 文件存在但无对应坐标段的 DW 文件（或反之）
- **THEN** 跳过该文件并输出警告信息

### Requirement: UTM 坐标计算
系统 SHALL 支持从 SE 文件名中的坐标段（`E{lon}_N{lat}`）和 128×128 像素位置推算 UTM 坐标网格（主路径，无需加载 GeoTIFF）；也 SHALL 保留从配套 GeoTIFF 读取 `transform` 和 `crs` 元数据计算 UTM 的能力（文件名推算不可行时的回退）。GeoTIFF 路径在文件名推算可行时不再必需。

#### Scenario: 从文件名坐标推算 UTM 网格
- **WHEN** 文件名坐标段为 `E121.4025_N25.1947`，`scale=10m`，网格 128×128
- **THEN** 生成 (128,128) 的 easting/northing 网格与 utm_zone，中心像素（row=64, col=64）落在文件名经纬度对应的 UTM 坐标附近，网格 NW 角经 scale 对齐后向东南展开

#### Scenario: UTM 带号推导
- **WHEN** 纬度 ≥ 0（北半球）
- **THEN** utm_zone = `int((lon + 180) / 6) + 1`（EPSG:326xx），与 DW 下载脚本 `_get_utm_epsg` 一致

#### Scenario: 分辨率常量
- **WHEN** 未指定分辨率
- **THEN** 使用默认 `UTM_RESOLUTION_M = 10`（米/像素，config.py 常量，可配置）

#### Scenario: 与 GeoTIFF 结果一致
- **WHEN** 同一影像同时存在文件名坐标段与 GeoTIFF 文件
- **THEN** 从文件名推算的 UTM 网格与 GeoTIFF Affine transform 推导的网格在像素级一致（数值相等或可接受容差内）

#### Scenario: 从 GeoTIFF 计算像素 UTM 坐标
- **WHEN** 图像 `(128, 128)` 且 GeoTIFF transform 为 `(10.0, 0.0, 338390.0, 0.0, -10.0, 2788110.0)`，CRS 为 `EPSG:32651`
- **THEN** 像素 `(row=0, col=0)` 的 `utm_easting=338395.0`, `utm_northing=2788105.0`, `utm_zone=51`

#### Scenario: 缺少 GeoTIFF 文件时使用占位坐标
- **WHEN** 某对数据的 GeoTIFF 文件不存在且文件名推算失败
- **THEN** `utm_easting`、`utm_northing` 和 `utm_zone` 设为 `NaN` 或 `0`，并输出警告

### Requirement: 批量导入 Qdrant
系统 SHALL 将逐像素解析的数据以批次（默认 10,000 条/批）通过 Qdrant upsert 接口写入向量数据库，并采用批量化快速路径（减少逐像素 Python 层构造开销、按需控制同步等待）以显著提升大目录导入速度；导入完成后可触发全量 HNSW 向量索引重建。

#### Scenario: 批量导入单张影像
- **WHEN** 导入一张 `(128, 128)` 影像的 16,384 个像素，批次大小 10,000
- **THEN** 分为 2 个批次完成导入，显示进度条，总计插入 16,384 条记录

#### Scenario: 导入进度显示
- **WHEN** 导入过程运行中
- **THEN** 显示当前进度（已处理像素数/总像素数）、已用时间和预估剩余时间

#### Scenario: 批量化快速路径
- **WHEN** 导入包含多张影像的数据目录
- **THEN** 使用批量化路径构造与写入向量，单位时间导入的像素数显著高于逐像素逐批写入，且断点续传、像素 ID 生成、统计输出等行为与既往一致

### Requirement: 断点续传
系统 SHALL 在导入前检查 Qdrant 中已存在的 `image_id`，跳过已导入的影像，确保导入过程可安全重复执行不产生重复数据。

#### Scenario: 重复导入跳过
- **WHEN** 对已导入过 `image_id` 的影像再次执行导入
- **THEN** 跳过该影像，输出 "已跳过 N 个已导入影像" 信息

#### Scenario: 部分导入后继续
- **WHEN** 导入过程在 7 张影像中的第 4 张之后中断，重新运行导入命令
- **THEN** 跳过前 4 张已完成影像，从第 5 张继续导入

### Requirement: 导入统计输出
系统 SHALL 在导入完成后输出统计摘要：总像素数、每类土地覆盖标签的像素数量及占比、总耗时、平均导入速率（像素/秒）。

#### Scenario: 导入完成统计输出
- **WHEN** 7 对 demo 文件全部导入完成
- **THEN** 输出总像素数 114,688、各类别像素计数、总耗时和速率

### Requirement: 像素 ID 生成
系统 SHALL 使用 `{image_id}_{row}_{col}` 格式生成像素唯一标识，其中 `image_id` 为 SE 文件名（不含扩展名），`row` 和 `col` 为像素在 128×128 影像中的行列号。

#### Scenario: 像素 ID 格式
- **WHEN** 处理 `all_mean_E121.4025_N25.1947_2024` 影像中 row=0, col=0 的像素
- **THEN** 生成 ID 为 `all_mean_E121.4025_N25.1947_2024_0_0`

### Requirement: WebUI 待导入数据分页预览
系统 SHALL 在 WebUI 数据导入区对待导入文件预览列表分页显示，每页最多显示 20 条，并提供"上一页/下一页"翻页按钮查看其余条目。

#### Scenario: 初始只显示前 20 条
- **WHEN** 数据目录中待导入文件对超过 20 条
- **THEN** 预览列表初始只显示前 20 条，页面不渲染全部条目

#### Scenario: 翻页查看其余条目
- **WHEN** 用户点击"下一页"按钮
- **THEN** 预览列表切换到后续 20 条，当前页信息随之更新

#### Scenario: 首页无上一页
- **WHEN** 用户处于第一页时查看翻页控件
- **THEN** "上一页"按钮不可用（禁用态），"下一页"在未到末页时可用

### Requirement: WebUI 导入全部按钮位置
系统 SHALL 将 WebUI 数据导入区的"导入全部"按钮置于"数据目录"输入框所在栏上方（或同一栏），用户无需滚动越过预览列表即可点击。

#### Scenario: 按钮位于数据目录栏上方
- **WHEN** 数据目录栏与预览列表同时显示
- **THEN** "导入全部"按钮出现在数据目录输入框上方或同栏，用户无需向下滚动预览条目即可点击

### Requirement: 导入进度条按像素推进
系统 SHALL 在 WebUI 向 Qdrant 导入数据期间显示线性进度条，每成功导入一个像素进度条前进相应比例，并显示当前进度文本（已导入/总数）。

#### Scenario: 导入期间显示进度
- **WHEN** 用户点击"导入全部"开始导入
- **THEN** 显示线性进度条与进度文本，进度按成功导入的像素数单调推进

#### Scenario: 导入完成进度到满
- **WHEN** 全部像素导入完成
- **THEN** 进度条到达满值并显示完成提示

### Requirement: 全量 HNSW 向量索引重建
系统 SHALL 在批量导入完成后支持对全量数据重建 HNSW 向量索引，使新导入的向量快速进入可检索状态，可在导入完成后显式触发。

#### Scenario: 导入完成后触发 HNSW 重建
- **WHEN** 批量导入完成且用户/CLI 指定重建索引（`--reindex`）
- **THEN** 系统对全量已导入数据执行 HNSW 向量索引重建（通过 `indexing_threshold=0` 强制重建），重建期间不影响已存在数据的检索

### Requirement: image_id 精确匹配索引
系统 SHALL 为 `image_id` 字段建立 keyword 索引（而非 text 索引），使 `MatchValue` 精确值匹配的 count/scroll 查询走索引路径，避免全量 payload 扫描导致的逐影像慢查询。

#### Scenario: keyword 索引加速精确 count
- **WHEN** 对 `image_id` 用 `MatchValue` 精确值执行 count 查询
- **THEN** 查询通过 keyword 索引 O(log n) 完成，不再全量扫描（实测 1.4s → 0.01s 量级）

#### Scenario: 既有 collection 索引迁移
- **WHEN** collection 已存在且 `image_id` 是 text 索引
- **THEN** 提供迁移路径：删除 text 索引并重建为 keyword 索引，保证既有数据可享受加速

### Requirement: 导入失败重试机制
系统 SHALL 在向 Qdrant 导入数据时对瞬时失败（网络抖动、超时、服务端繁忙）提供指数退避重试，避免单次失败导致整个导入中断。

#### Scenario: upsert 失败重试
- **WHEN** 某批 upsert 因瞬时错误失败
- **THEN** 系统按指数退避（如 1s, 2s, 4s）重试该批，达到最大重试次数仍失败才抛出

#### Scenario: count 失败重试
- **WHEN** 断点续传的 count 查询因瞬时错误失败
- **THEN** 系统重试该查询，避免把误判为"已导入"而跳过影像

### Requirement: 导入成功后同步更新 manifest
系统 SHALL 在 `import_directory` 每张影像导入/跳过完成后同步更新导入 manifest（见 `import-manifest` capability），使 CLI 与 WebUI 所有导入入口共享同一清单同步点。

#### Scenario: 单张导入后更新清单
- **WHEN** `import_directory` 完成某张影像的导入（或跳过已导入影像）
- **THEN** 该 image_id 及其已导入像素数写入 manifest

#### Scenario: 全部导入后清单完整
- **WHEN** 目录中全部影像导入完成
- **THEN** manifest 中包含目录中所有已导入影像的 image_id 与像素数

### Requirement: 迁移命令复用断点续传
系统 SHALL 使 `migrate` 命令复用 `import_directory` 的断点续传逻辑（按 image_id count 判定）重建 Collection 并重导数据，保证迁移可中断重试、不产生重复数据。

#### Scenario: 迁移中断后重试
- **WHEN** `migrate` 过程中断，重新运行
- **THEN** 已完整导入的影像被跳过，未完成的继续导入

#### Scenario: 迁移后数据完整
- **WHEN** `migrate` 全部完成
- **THEN** Collection 包含全量影像，`check_image_count` 对每张影像返回 16384

