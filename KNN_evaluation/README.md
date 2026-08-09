# Qdrant KNN 像素评估系统 — 使用说明

## 环境准备

### 1. 启动 Qdrant 向量数据库（挂 volume 持久化）

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

> **数据持久化**：Qdrant 数据挂载到命名 volume `qdrant_data`，容器重建后数据保留。系统内置
> `_start_qdrant()`（`cli.py` / `webui.py` 一致）幂等启动容器，日常无需手动启动；若手动管理，
> 请务必使用上述带 volume 的命令，避免删除重建容器后数据丢失。

### 2. 确认数据目录

默认数据目录：`data_demo/`

目录结构：
```
data_demo/
├── SE/           # Satellite Embedding V1 (.npy / .npz + .tif)
└── DW/           # Dynamic World V1 (.npy + .tif)
```

### 3. 启动 WebUI

```bash
cd .\evaluation_scripts
uv run python KNN_evaluation/webui.py --port 8003 --dir data_demo
```

浏览器访问：**http://127.0.0.1:8003**

可通过命令行参数自定义：
```bash
uv run python KNN_evaluation/webui.py --port 8003 --dir data_demo --qdrant-url http://localhost:6333
```

---

## 界面操作指南

### 一、Qdrant 连接状态

打开页面后自动连接 Qdrant。显示：
- ✅ **Qdrant 连接正常** — 已连接到 Qdrant
- 显示 Collection 名称、总点数和分段数
- 如果 Collection 不存在会显示「创建 Collection」按钮，**点击一次即可**

> **注意**：Qdrant 不可达时显示 ❌ 红色警告，请确认 Docker 容器已启动。

### 二、数据导入

1. 展开「**数据导入**」面板
2. 确认「数据目录」路径正确（默认 `data_demo`）
   - 切换预置分页（`google_aef_embedding` / `xian_aef_embedding`）时，数据目录输入框
     会自动切换到该分页的默认数据目录（`data_google` / `data_xian`，相对项目根）；
     自定义 collection 分页保持 `--dir` 指定的目录
   - 直接修改输入框后无需点「浏览」，「**导入全部**」会按输入框当前值导入（同步
     `state`，与「浏览」按钮同源）；目录不存在时明确提示且不启动导入
3. 点击「**浏览**」按钮扫描 SE/DW 文件对
   - 📦 待导入 — 尚未导入
   - ⏳ 部分已导入 — 已导入部分数据
   - ✅ 已导入 — 已完整导入（16,384 像素）
4. 点击「**导入全部**」按钮，等待导入完成
   - 导入在后台进行，不会阻塞 UI
   - 完成后弹出统计对话框（像素数、标签分布、耗时等）

### 三、向量检索

导入完成后即可进行检索。

#### A. 随机选取（推荐新手使用）

1. 选择「**随机选取**」模式
2. 点击「**从 Collection 随机获取**」按钮
3. 系统会从已导入数据中随机选取一个像素作为查询向量
4. 设置检索参数：
   - **K**：返回 Top-K 个最近邻（默认 10）
   - **标签过滤**：多选筛选（水/树/草/建筑等），选「无过滤」则不限制
   - **UTM 范围过滤**：可选，按地理坐标过滤
   - **精确搜索**：开启使用暴力全量对比（慢但精确），关闭使用近似搜索（快）
5. 点击「**执行检索**」

#### B. 指定像素

1. 选择「**指定像素**」模式
2. 在「影像」下拉中选择一个影像标识
3. 输入行 (0-127) 和列 (0-127) 坐标
4. 点击「**获取向量**」
5. 设置检索参数后点击「**执行检索**」

> **提示**：「影像」下拉列表在**浏览数据目录**或**导入数据**后自动填充。

### 四、检索结果

检索完成后弹出结果对话框：

- **元数据**：搜索模式（ANN/Exact）、耗时（ms）、命中数量
- **标签分布**：可展开查看命中结果中各类标签的计数
- **命中表格**：每行显示一个命中像素的 ID、相似度分数、标签、坐标等

### 五、可视化探索

在检索结果对话框中点击「**可视化探索**」：
- 显示 128×128 灰度图（可选择通道和缩放级别）
- **红色十字准线**：查询像素位置
- **编号彩色圆点**：命中的最近邻像素
- 点击图中任意位置可将其设为新查询点
- 右侧面板显示每个命中像素的详情和图例

---

## 命令行替代方案

如果不使用 WebUI，也可以通过 CLI 操作：

```bash
# 导入数据
uv run python -m KNN_evaluation.cli import data_demo

# 随机搜索
uv run python -m KNN_evaluation.cli search --random --k 5

# 查看统计
uv run python -m KNN_evaluation.cli stats

# 重建 Collection（切换存储配置，默认 disk）
uv run python -m KNN_evaluation.cli migrate [--storage disk|ram] [--dir data_demo]
```

---

## Embedding 质量评估

系统内置了两类评估指标，用于衡量像素 embedding 向量的质量。

### CLI 快速评估

```bash
# 基础评估 (F1 + F2)
uv run python -m KNN_evaluation.cli evaluate --samples-per-class 200 --k-values 10,50,100

# 含图表输出
uv run python -m KNN_evaluation.cli evaluate --samples-per-class 200 --k-values 10,50,100 --plot --plot-dir ./eval_plots/

# 导出 JSON
uv run python -m KNN_evaluation.cli evaluate --samples-per-class 200 --k-values 10,50,100 --output result.json
```

### WebUI 评估面板

在 WebUI 中展开「**评估面板**」expansion（位于"向量检索"下方），设置参数后点击「开始评估」：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 每类采样数 | 500 | 从每个地表类别随机采样的像素数，共 9 类 |
| K-F1 | 10 | KNN 分类器邻居数，使用留一法 |
| K 值序列 | 10,20,50,100,300,1000 | Purity/Recall 曲线的采样点 |
| Seed | 42 | 随机种子，保证可复现 |

展开「**参数说明**」查看各参数的功能原理解释。

### 双集合相似度热力图对比

系统内置双集合 embedding 相似度热力图对比功能：从两个集合（默认
`google_aef_embedding` × `xian_aef_embedding`，point_id 确定性一致，天然按位置
对齐）随机采样 N 个点，按同一批 point_id 分别取 embedding，各自计算 N×N 余弦
相似度矩阵，并排（1×2）统一色阶输出 PNG。

> **image_id 全链路归一化**：坐标段字符串（如 `E121.4033_N25.1370`）在扫描时会
> 统一 round 4 位小数并去尾随零（`E121.4033_N25.137`），保证 google（混合精度）与
> xian（全 4 位）两集合的 image_id / point_id 一致，双集合对比不丢点。**升级后须
> 清空并重新导入两个 collection**，旧数据中的 image_id 仍为归一化前的原始字符串。

CLI 方式：

```bash
# 数据库全库模式
uv run python -m KNN_evaluation.cli similarity-heatmap --n 200 --seed 42 --output similarity_heatmap.png

# 单张图片模式（指定 image_id）
uv run python -m KNN_evaluation.cli similarity-heatmap --n 200 --seed 42 --image-id E121.4794_N25.1378 --output heatmap.png

# 自定义双集合
uv run python -m KNN_evaluation.cli similarity-heatmap --google-collection google_aef_embedding --xian-collection xian_aef_embedding

# 导出相似度矩阵与采样信息（默认导出到 outputs/，--export-dir "" 可禁用）
uv run python -m KNN_evaluation.cli similarity-heatmap --n 200 --seed 42
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n` | 200 | 采样点数，范围 1..600 |
| `--seed` | 42 | 随机种子，保证可复现 |
| `--image-id` | 无 | 指定影像则进入图片模式，否则数据库全库模式 |
| `--output` | similarity_heatmap.png | 输出 PNG 路径 |
| `--export-dir` | outputs | 导出目录（不存在时自动创建）；显式传空串 `--export-dir ""` 禁用导出 |
| `--google-collection` / `--xian-collection` | google_aef_embedding / xian_aef_embedding | 对比双集合名称 |
| `--qdrant-url` | http://localhost:6333 | Qdrant 服务地址 |

**导出文件**（默认导出到 `outputs/`，`--export-dir` 可自定义，目录不存在时自动创建）：
- `{google_collection}_similarity.npy`：google 侧 N'×N' 余弦相似度矩阵
- `{xian_collection}_similarity.npy`：xian 侧 N'×N' 余弦相似度矩阵
- `similarity_sampling.json`：`{"params": ..., "pixels": [...]}`，params 为
  `{n, seed, image_id, collections, sampled, kept, dropped, elapsed_sec}`（image_id
  为 None 时记 null）；pixels 为保留像素信息列表，每项含
  point_id/image_id/pixel_row/pixel_col/utm_easting/utm_northing/utm_zone，与
  相似度矩阵行序严格一致（按保留 ids 顺序，单侧缺失已剔除）。

WebUI 方式：展开「**相似度热力图对比**」expansion（位于"评估面板"下方），设置
采样数 N、seed，选择模式（数据库全库 / 单张图片），在「导出目录」输入框填写
导出目录（默认 `outputs`，留空则不导出），点击「生成热力图对比」即可在面板内嵌
展示并排热力图，导出时面板会展示导出文件路径。WebUI 固定对比预置对
`google_aef_embedding` × `xian_aef_embedding`，与当前 collection 选择器无关。

### 两类评估指标

#### F1 — KNN 分类准确率

评估 K 近邻作为分类器的效果。对每个采样像素：

1. 检索 K+1 个最近邻（Leave-One-Out，剔除自身）
2. 取前 K 个有效邻居按多数投票预测标签
3. 平票时逐次减小 K 重新计票直至打破平局

**输出**：Overall Accuracy、Per-class Precision/Recall/F1、Confusion Matrix（9×9）。

> 9 类随机基线约 11%，显著高于此值说明 embedding 有区分能力。

#### F2 — 邻居纯度与 Recall@K 曲线

衡量不同 K 值下的邻居质量：

- **Purity@K**：Top-K 邻居中与查询像素同标签的比例。随 K 增大单调递减（邻居越多越可能混入异类）。
- **Recall@K**：前 K 个邻居中能召回全量同类像素的比例。随 K 增大单调递增（邻居越多越能覆盖同类）。分母为全局同类像素总数（通过 Qdrant count 精确统计），因此 K 较小时 Recall 天然很小——这是标准信息检索 Recall@K 的预期行为。

> Purity 和 Recall 曲线方向相反是正常的：Purity↓, Recall↑。恰好说明指标在正常工作。

### 性能说明

- F1：4500 个查询 × K+1 个邻居检索 ≈ 4,500 次 exact 检索
- F2：**关键优化** — 每个查询像素仅检索一次 `K=max(k_values)+1`，从结果中递增取不同 K 值计算，避免重复检索。4500 个查询 × 1001 个结果检索 ≈ 4,500 次 exact 检索

所有检索默认使用 `exact=True`（暴力精确搜索），评估的是 embedding 质量而非 HNSW 近似误差。

---

## 常见问题

**Q: 页面显示 Qdrant 不可达？**
A: 确认 Docker 容器已启动 `docker ps | grep qdrant`

**Q: 点击导入报错？**
A: 检查 `data_demo/SE/` 和 `data_demo/DW/` 目录下是否有相匹配的 `.npy` 文件对

**Q: 影像下拉没有选项？**
A: 先展开「数据导入」面板，点击「浏览」按钮扫描数据目录，或在导入数据后自动刷新

**Q: 检索结果为空？**
A: 可能过滤条件过严（如选择了 Collection 中不存在的标签），尝试去掉标签过滤后再搜索
