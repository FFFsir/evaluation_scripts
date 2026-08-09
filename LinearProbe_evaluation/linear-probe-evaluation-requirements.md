# 基于 Qdrant 的像素级 Linear Probe（MLP_label）评估系统 — 需求方案

## 背景

KNN 评估已完成：`pixel_embeddings` Collection 已保存约 1022 万像素，
每点含 64 维 SE embedding 向量与 DW 硬分类 label（0-8）。本阶段在其之上
训练一个 **64 维输入、9 维输出的线性分类 MLP 网络（MLP_label）**：

- 64 维输入 = 1 个像素的 embedding；
- 9 维输出 = 9 个标签的硬分类**独热码**（当前 Qdrant 保存的是像素 DW 数据集的
  label 信息，因此对应硬分类 one-hot 语义，训练用 CrossEntropyLoss）；
- **后续扩展**：若引入像素 DW 数据集的 prob（软分类概率）信息，将新增
  64 维输入、9 维输出的软分类概率网络 **MLP_prob**（输出概率分布，软标签训练）。
  本方案中 MLP_label 与 MLP_prob 命名与结构互相独立，便于并行演进。

## 数据模型（复用 KNN 的 Collection）

| 字段 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `id` | string/uuid | KNN import | 像素唯一标识 |
| `vector` | float[64] | SE V1 | 64 维 embedding（本模块输入） |
| `label` | int 0-8 | DW V1 | 土地覆盖硬分类（本模块监督目标） |
| `label_name` | string | 查表映射 | water/trees/grass/flooded_vegetation/crops/shrub_and_scrub/built/bare/snow_and_ice |
| `utm_easting`/`utm_northing`/`utm_zone` | float/int | 坐标推算 | 本模块不使用 |
| `image_id`/`pixel_row`/`pixel_col` | str/int | 影像元数据 | 本模块不使用 |

数据导入由 `KNN_evaluation` 完成，本模块**只读**。

### 类别分布（实测，1022 万点，严重不平衡）

| label | 名称 | 像素数 |
|---|---|---|
| 0 | water | 533,687 |
| 1 | trees | 6,209,827 |
| 2 | grass | 74,887 |
| 3 | flooded_vegetation | 4,319 |
| 4 | crops | 51,816 |
| 5 | shrub_and_scrub | 5,362 |
| 6 | built | 3,332,335 |
| 7 | bare | 11,143 |
| 8 | snow_and_ice | 240 |

不平衡比约 25,874:1 → 必须**分层采样**，并采用 **macro-F1** 作为主指标。

## 网络定义

**两种结构可选**（CLI `--model` / WebUI「模型结构」下拉，默认 `mlp`）：

1. **mlp（LeNet-5 规模多层 MLP，约 15 万参数）**：

```
MLP_label(x) = fc_out( ReLU(Dropout(... ReLU(Dropout( fc1(x) )) ...)) )
   x ∈ R^64 → 隐含层 [256, 256, 256]（ReLU + Dropout(0.2)）→ 9 维 logits
```

2. **linear（标准 Linear Probe，585 参数）**：

```
MLP_label(x) = W @ x + b,   x ∈ R^64,  W ∈ R^(9×64),  b ∈ R^9
```

- 输入为 64 维像素 embedding（1 维向量而非图像），故采用全连接 MLP 而非 CNN；
  隐含层维度 `hidden_dims` 可配置（`()` = 单层 linear）；
- 输出 9 维 logits → `argmax` 得硬分类类别 → one-hot 编码即 9 维硬分类独热码；
- 损失：CrossEntropyLoss（真实标签 0-8，直接 class index）；
- `linear` 结构用于检测 embedding 线性可分性（学术惯例：Alain & Bengio 2016 等）。

## 功能需求

### F1：分层数据采样与 train/val 划分

- 复用 KNN 采样地图（`qdrant_sampling_map.json`，point_id→label），本地随机选
  ID 后按 ID 精确 `retrieve` 向量，只下载样本量（MB 级）；
- 每类先按 `val_ratio`（默认 0.2）切出验证集，再分别受 `train_per_class`
  （默认 2000）/ `val_per_class`（默认 500）上限约束；
- **稀有类保证**：某类样本数少于训练上限时（如 snow_and_ice=240），仍保证
  验证集非空（至少 1 个）且 train/val 不重叠；
- 相同 `seed` 完全可复现。

### F2：训练循环

- 支持 `--epochs` / `--batch-size` / `--lr` / `--weight-decay` / `--optimizer`（adam|sgd）/ `--model`（mlp|linear）/ `--seed`；
  **训练设备固定 CUDA**（`resolve_device` 已禁用 CPU fallback：CUDA 不可用直接报错，
  绝不静默回退 CPU）；训练结果输出 GPU 设备名与 CUDA kernel 累计耗时（CUDA events 计时）；
- 每个 epoch 输出：train loss / train acc / val loss / val acc / val macro-F1；
- 进度回调（`event` dict）逐 batch / 逐 epoch 上报，供 CLI 与 WebUI 共用；
- `threading.Event` 协作式取消（逐 batch 检查）；
- checkpoint：按 val_acc 保存 best，训练结束保存 final（state_dict + 架构/标签元数据）；
- 训练元数据 JSON（超参 + 每 epoch 历史 + 最终分类报告）。

### F3：评估指标

- accuracy、macro-F1、weighted-F1、per-class precision/recall/F1/support、混淆矩阵；
- 稀有类指标分母为 0 时记 0（不除零）；
- `evaluate` 子命令：加载 checkpoint，**默认按采样地图分层采样**（每类 1000，
  只下载样本量）；`--samples-per-class 0` 才显式全量读取（耗时长，不推荐）。

### F4：可视化

- matplotlib（Agg + 中文字体，与 KNN 风格一致）：训练/验证 loss 与 accuracy/
  macro-F1 双面板曲线、混淆矩阵热力图；
- 图表 PNG 落盘（CLI）或 base64 data URI（WebUI 实时刷新）。

### F5：WebUI 训练监测（NiceGUI，与 KNN_evaluation/webui.py 同风格）

- Qdrant 连接 & Collection 状态区；
- 训练参数配置区（含参数说明）；
- 分层采样进度 + 训练进度（进度条 + 状态文本）；
- 逐 epoch 实时更新的 loss/accuracy/macro-F1 曲线；
- 训练结果：指标摘要、per-class 表、验证集混淆矩阵图；
- 训练结果 JSON 导出、「中止训练」协作式取消。

## 模块结构

```
LinearProbe_evaluation/
  config.py / label_mapping.py / qdrant_client.py   # 配置与数据访问
  dataset.py    # 分层采样 / train-val 划分 / 全量读取
  model.py      # MLPLabel（Linear(64,9)），预留 MLPProb 扩展注释
  trainer.py    # 训练循环 / 回调 / checkpoint
  metrics.py    # 指标与 one-hot 工具
  visualization.py  # 曲线 / 混淆矩阵
  cli.py        # train / evaluate / stats
  webui.py      # NiceGUI 训练监测界面
  tests/        # 单元测试（内存 Fake Qdrant）
```

## 验收标准

1. `uv run pytest LinearProbe_evaluation/tests/ -v` 全绿（无需真实 Qdrant）；
2. `cli stats` 输出总点数与每类像素数，与 Qdrant count API 一致；
3. `cli train` 在真实 Qdrant 上完成采样 → 训练 → checkpoint/图/报告全流程；
4. `cli evaluate --checkpoint ...` 加载 checkpoint 输出指标与混淆矩阵；
5. `webui.py` 启动后可打开页面，训练过程曲线逐 epoch 刷新，可中止。
