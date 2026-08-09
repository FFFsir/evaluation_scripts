# similarity-heatmap-compare Specification

## Purpose
在 KNN 评估基础上提供与标签无关的 embedding 对比能力：随机采样 N 个 UTM 坐标，从两个 embedding 集合按位置提取向量，各自计算 N×N 余弦相似度矩阵并并排生成热力图，用于直观对比两个集合的局部邻域结构。
## Requirements
### Requirement: 随机采样 N 个 UTM 坐标
系统 SHALL 支持两种方式随机抽取 N 个 UTM 坐标（不按标签分层，纯随机）：
- **数据库模式**：从目标 Qdrant Collection 全库随机抽取 N 个点，取其 UTM 坐标（`utm_easting` / `utm_northing` / `utm_zone`）与对应 point_id；
- **单张图片模式**：用户指定一个 `image_id`，在该影像 128×128 像素网格内随机抽取 N 个像素（不重复），取各像素的 UTM 坐标与 point_id。

N 为可配置参数，默认 200，最大 600；超过上限时 MUST 报错拒绝执行。采样 MUST 支持随机种子（seed）保证可复现。

#### Scenario: 数据库模式正常采样
- **WHEN** 调用数据库模式，N=200，seed=42，目标 Collection 总点数 ≥ 200
- **THEN** 返回 200 个不重复的采样点，每点含 UTM 坐标与 point_id，且同一 seed 重复执行结果完全一致

#### Scenario: 单张图片模式正常采样
- **WHEN** 指定 image_id（如 `E121.4025_N25.1947`），N=200，seed=1
- **THEN** 返回该影像内 200 个不重复像素（row,col∈[0,127]）的 UTM 坐标与 point_id，且同一 seed 可复现

#### Scenario: N 超过上限
- **WHEN** 传入 N=1000（上限 600）
- **THEN** 执行被拒绝并返回明确错误信息，不产生部分结果

#### Scenario: 样本不足
- **WHEN** 单张图片模式 N=200，但该 image_id 在数据库中像素不足（如集合中缺失该影像）
- **THEN** 返回明确错误（影像不存在或无足够像素），不静默降采样

#### Scenario: Collection 为空
- **WHEN** 数据库模式且目标 Collection 总点数为 0
- **THEN** 返回明确错误，提示 Collection 为空

### Requirement: 双集合按位置提取 embedding
系统 SHALL 基于同一组采样 point_id，从 `google_aef_embedding` 与 `xian_aef_embedding` 两个 Collection 分别检索对应位置的 embedding 向量。两个集合的 point_id 通过 `uuid5(uuid5(DNS, image_id), "row_col")` 确定性生成且 image 集一致，因此同一 UTM 位置在两集合中具有相同 point_id。任一集合缺少某个 point_id 时 MUST 跳过该点（两侧同时剔除），避免矩阵维度不一致。

#### Scenario: 双集合提取成功
- **WHEN** 采样得到 N 个 point_id，两个集合均包含全部 N 个点
- **THEN** 返回两个 (N, 64) 的 embedding 矩阵，行序与采样顺序一致

#### Scenario: 单侧缺失点剔除
- **WHEN** 采样点 P 在 google 集合存在但在 xian 集合缺失
- **THEN** P 被剔除，两侧矩阵均不含 P，行序保持一致（维度 N' < N），并记录剔除点数量

### Requirement: N×N 余弦相似度矩阵计算
系统 SHALL 分别对两个集合的 embedding 矩阵计算 N×N 余弦相似度矩阵（与 Qdrant COSINE 距离度量一致，对角元为 1.0，值域 [-1, 1]）。两个矩阵 MUST 使用相同的行/列顺序与统一色阶映射，保证并排对比的视觉可比性。

#### Scenario: 矩阵正确性
- **WHEN** 对 (N, 64) 矩阵计算余弦相似度
- **THEN** 输出 (N, N) 对称矩阵，对角元恒为 1.0，其余元素值域 [-1, 1]

#### Scenario: 矩阵维度
- **WHEN** 提取后每侧各 N' 个向量
- **THEN** 两矩阵均为 (N', N')，维度一致可并排渲染

### Requirement: 并排热力图生成
系统 SHALL 将两个相似度矩阵分别渲染为热力图（同一色图与统一颜色标尺 vmin/vmax），并排（1×2）合并为一张对比 PNG 输出。两幅子图标题 MUST 标注各自集合名称。

#### Scenario: 对比图输出
- **WHEN** 完成两个矩阵计算并请求输出
- **THEN** 生成一张 1×2 并排的对比 PNG，左/右子图分别为 google / xian 集合热力图，使用统一色阶与色图，文件保存到指定路径

### Requirement: CLI 子命令
系统 SHALL 在 CLI 提供 `similarity-heatmap` 子命令，支持参数：`--n`（默认 200）、`--seed`（默认 42）、`--image-id`（指定后进入单张图片模式，缺省为数据库模式）、`--output`（PNG 输出路径）、`--google-collection` 与 `--xian-collection`（可覆盖默认集合名）、`--qdrant-url`。执行成功后打印输出路径；任一步骤失败时输出错误并返回非零退出码。

#### Scenario: 数据库模式 CLI
- **WHEN** 运行 `python -m KNN_evaluation.cli similarity-heatmap --n 200 --output compare.png`
- **THEN** 从 google 集合全库随机抽 200 点，双集合提取并生成并排热力图 `compare.png`，打印输出路径，退出码 0

#### Scenario: 单张图片模式 CLI
- **WHEN** 运行 `python -m KNN_evaluation.cli similarity-heatmap --n 200 --image-id E121.4025_N25.1947 --output compare.png`
- **THEN** 在该影像内抽 200 像素并生成并排热力图，退出码 0

#### Scenario: Qdrant 不可达
- **WHEN** Qdrant 服务未启动或不可达
- **THEN** 输出明确错误信息并返回非零退出码

### Requirement: WebUI 对比面板

系统 SHALL 在 WebUI 提供"相似度热力图对比"面板：支持输入 N（默认 200）、seed、采样模式（数据库/单张图片）、图片选择（单张图片模式时从已导入影像下拉选择）、集合名（沿用当前 collection 选择器的 google/xian 预置与自定义集合），异步执行并内嵌展示两张并排热力图；执行失败时展示错误信息。

#### Scenario: 面板执行并展示
- **WHEN** 用户在面板设置 N=200、数据库模式并点击执行
- **THEN** 系统异步执行，完成后在面板内显示 1×2 并排对比热力图，两张子图标题标注集合名

#### Scenario: 单张图片模式下拉
- **WHEN** 用户选择单张图片模式
- **THEN** 出现影像下拉列表（来自已导入影像清单），选择后按该影像执行采样

#### Scenario: 参数校验失败提示
- **WHEN** 用户输入 N=1000（超上限）或目标集合为空
- **THEN** 面板展示明确错误信息，不执行

#### Scenario: 面板导出按钮
- **WHEN** 用户在 SimilarityMatrix 页面完成一次相似度对比
- **THEN** 页面出现「导出 JSON」「导出图片图表」与「输出 npy 文件」三个按钮，且此前不可见（未对比时隐藏）
- **AND** 点击「导出 JSON」后在 `outputs/evaluation/similarity` 生成带时间戳的 JSON 文件（含采样参数与保留像素信息），文件名以 `full_col_`（数据库全库）或 `single_img_`（单张图片）开头
- **AND** 点击「导出图片图表」后在 `outputs/evaluation/similarity` 生成带时间戳的热力图 PNG 文件，文件名以 `full_col_` 或 `single_img_` 开头
- **AND** 点击「输出 npy 文件」后在 `outputs/evaluation/similarity` 生成两个带时间戳的 npy 文件（`{前缀}_similarity_{时间戳}.npy`，google/xian 各一），重复导出生成新时间戳文件、不覆盖旧文件
- **AND** 目录不存在时自动创建，成功后提示文件路径；未执行对比时点击提示「请先生成热力图对比」

### Requirement: 导出相似度矩阵与采样信息

系统 SHALL 在相似度对比执行时导出数据：将两个 N×N 余弦相似度矩阵分别导出为以对应 collection 命名的 `.npy` 文件（命名规则 `<collection>_similarity.npy`，如 `google_aef_embedding_similarity.npy`）；将采样参数与保留的采样像素信息总结为一个 JSON 文件。**默认导出到 `outputs/` 目录**（相对项目根，自动创建）；用户可覆盖为其他目录或留空/`None` 禁用导出。CLI 与 WebUI 双入口均支持。导出函数 SHALL 支持可选 `prefix` 参数（非空时文件名带前缀，为空时原名）与可选 `export_npy` 参数（默认 True 自动导出 npy；False 时只写 sampling json，不写 npy，供 WebUI 手动导出场景使用）。

#### Scenario: 默认导出两个矩阵与 JSON
- **WHEN** 用户执行相似度对比且未指定导出目录（CLI 缺省 / WebUI 输入框默认 `outputs`）
- **THEN** 在 `outputs/` 目录生成 `google_aef_embedding_similarity.npy`、`xian_aef_embedding_similarity.npy`（均为 (N',N') 余弦相似度矩阵）与 `similarity_sampling.json`（含采样参数与每个保留像素的 point_id/image_id/row/col/utm_easting/utm_northing/utm_zone），热力图 PNG 照常生成

#### Scenario: 指定导出目录
- **WHEN** 用户指定导出目录（如 CLI `--export-dir out/` 或 WebUI 输入框改为 `out/`）
- **THEN** 在指定目录生成上述文件，不再使用默认 `outputs/`

#### Scenario: 留空禁用导出
- **WHEN** 用户留空导出目录（CLI `--export-dir ""` 或 WebUI 输入框清空）
- **THEN** 只生成热力图 PNG，不生成任何 npy/json 文件

#### Scenario: 单侧剔除后导出
- **WHEN** 采样点存在单侧缺失被剔除（kept=N'<N）
- **THEN** 导出的 npy 矩阵为 (N',N')，JSON 的 pixels 数组仅包含保留的 N' 个像素，与矩阵行序一致

#### Scenario: WebUI 带前缀导出
- **WHEN** 用户从 SimilarityMatrix 页面执行对比（数据库全库或单张图片模式）
- **THEN** sampling json 文件名带 `full_col_` 或 `single_img_` 前缀，且 npy 不再自动落盘（WebUI 传 `export_npy=False`），由「输出 npy 文件」按钮手动导出

### Requirement: 分页默认数据导入目录
系统 SHALL 按 collection 提供默认数据导入目录：`google_aef_embedding` 分页默认 `data_google`，`xian_aef_embedding` 分页默认 `data_xian`（相对项目根）。切换到对应分页时，数据目录输入框与当前数据目录自动指向该默认目录；自定义 collection 分页保持现有默认数据目录不变。导入执行时 MUST 使用数据目录输入框的当前值（与「浏览」按钮同源），不得使用切换分页前残留的旧目录。

#### Scenario: 切换到 xian 分页默认目录
- **WHEN** 用户切换到 `xian_aef_embedding` 分页
- **THEN** 数据目录输入框显示 `data_xian`，导入使用该目录

#### Scenario: 切换到 google 分页默认目录
- **WHEN** 用户切换到 `google_aef_embedding` 分页
- **THEN** 数据目录输入框显示 `data_google`，导入使用该目录

#### Scenario: 修改输入框后直接导入
- **WHEN** 用户将数据目录输入框改为其他路径（未点击「浏览」）并点击「导入全部」
- **THEN** 导入使用输入框当前值指向的目录（与浏览按钮同源），不使用旧的 `state["data_dir"]`

#### Scenario: 目录不存在
- **WHEN** 数据目录输入框指向的目录不存在时点击「导入全部」
- **THEN** 显示明确错误信息，不执行导入

### Requirement: 坐标段数值匹配配对
系统 SHALL 在 `scan_directory` 配对 SE/DW/GeoTIFF 文件时，将文件名坐标段解析为数值坐标（经度、纬度浮点数）进行匹配，而非字符串精确相等；同一数值坐标的文件即使坐标段字符串精度不同（如 `N25.137` 与 `N25.1370`）MUST 配对为同一影像。配对成功后 `ImagePair.image_id` MUST 使用**数值归一化坐标段**（经纬度 round 到 4 位小数、去掉尾随零），而非 SE 文件坐标段的原始字符串，以保证不同精度文件产生的 image_id（及派生 point_id）一致。

#### Scenario: 精度不同的 SE/DW 配对
- **WHEN** 目录中 SE 文件坐标段为 `E121.4033_N25.1370`（4 位小数）而 DW 文件坐标段为 `E121.4033_N25.137`（3 位小数）
- **THEN** `scan_directory` 返回 1 对，`ImagePair.image_id` 为数值归一化串 `E121.4033_N25.137`（尾随零去除）

#### Scenario: 数值相同字符串不同的多个文件
- **WHEN** SE 与 DW 各有多个文件其坐标段字符串不同但解析后的数值坐标相同（如 `E121.4030_N25.1601` 与 `E121.403_N25.1601`）
- **THEN** 按数值坐标配对，数量与「数值坐标集合的交集」一致；`image_id` 为归一化串 `E121.403_N25.1601`，且同一数值坐标的所有文件产生相同 image_id

#### Scenario: 孤儿文件仍被跳过
- **WHEN** 某 SE 文件的数值坐标在 DW 中无对应
- **THEN** 该文件不产生配对，保持既有跳过语义

### Requirement: 可视化探索按检索 collection 定位影像文件
系统 SHALL 在 WebUI 检索结果的「可视化探索」中，按**产生该检索结果的 collection** 对应的数据目录（`COLLECTION_DATA_DIRS` 映射，如 `google_aef_embedding → data_google`、`xian_aef_embedding → data_xian`）扫描 SE 文件构建影像映射，而非使用跨 collection 共享的全局浏览目录。由于 image_id 归一化后两 collection 的 image_id 完全重叠（相同坐标段），全局共享的 `se_paths_map` 会取到另一 collection 的 SE 文件导致背景图串集；本需求保证可视化背景图与检索 collection 一致。

#### Scenario: google 检索可视化用 google 数据
- **WHEN** 用户在 `google_aef_embedding` collection 检索并打开可视化探索
- **THEN** 背景图加载自 `data_google` 目录的 SE 文件（即使 XIAN 页最后浏览过 data_xian，也不会串集）

#### Scenario: xian 检索可视化用 xian 数据
- **WHEN** 用户在 `xian_aef_embedding` collection 检索并打开可视化探索
- **THEN** 背景图加载自 `data_xian` 目录的 SE 文件

#### Scenario: 自定义 collection 回退默认目录
- **WHEN** 检索的 collection 不在 `COLLECTION_DATA_DIRS` 映射中（自定义 collection）
- **THEN** 可视化背景图回退使用 `_CLI_DATA_DIR`（启动默认数据目录）扫描的 SE 文件

### Requirement: payload 索引自动补齐
系统 SHALL 在 WebUI 分页切换与页面加载时，对当前 collection 幂等补齐缺失的 payload 索引（label / label_name / utm_easting / utm_northing / image_id）。collection 已存在但缺索引（如历史创建/重建时未建）时，UTM 范围过滤或标签过滤会走全量 payload 扫描导致检索超时；自动补齐保证过滤检索走索引路径。

#### Scenario: 分页切换补齐缺失索引
- **WHEN** 用户切换到缺 utm_easting/utm_northing 索引的 collection（如历史 xian collection）
- **THEN** 系统幂等补齐缺失索引（已有索引跳过），UTM 过滤检索不再超时

#### Scenario: 索引已齐全不重复创建
- **WHEN** collection 的 payload 索引已齐全（如 google collection）
- **THEN** 不重复创建任何索引，无额外开销

