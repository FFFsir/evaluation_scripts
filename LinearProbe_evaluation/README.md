# Qdrant Linear Probe 评估系统（MLP_label）

基于 Qdrant 中已有的像素 embedding（64 维）与 DW 硬分类标签（0-8），训练
**线性分类 MLP 网络 MLP_label**（64 维输入 → 9 维输出），并提供 CLI 训练/评估
与 NiceGUI Web 训练过程监测界面。与 `KNN_evaluation/` 共享同一
Qdrant Collection 与采样地图基础设施。

> **Collection 选择**：默认读取 `xian_aef_embedding` Collection；可在 CLI
> 用 `--collection`、WebUI 用选择器在预置 `google_aef_embedding` /
> `xian_aef_embedding` 之间切换，或输入自定义 collection 名称。采样地图按
> collection 自动隔离（`qdrant_sampling_map_<collection>.json`）。

## 网络结构：MLP_label（两种结构可选）

- **输入**：64 维像素 embedding（Qdrant 所选 collection 的向量，1 行 = 1 个像素）
- **输出**：9 维 logits，对应 9 个 DW 土地覆盖标签的**硬分类独热码**
  （water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice）
- **损失**：CrossEntropyLoss（真实标签为 0-8 硬分类；预测取 logits 的 argmax）
- **两种结构（CLI `--model` / WebUI「模型结构」下拉选择，默认 `mlp`）**：
  - **`mlp`（LeNet-5 规模多层 MLP，约 15 万参数）**：
    `Linear(64,256) → ReLU → Dropout(0.2) → Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,9)`，
    可拟合 embedding 中的非线性、追求更高精度；
  - **`linear`（标准 Linear Probe，585 参数）**：单一 `Linear(64,9)`，无隐藏层、
    无激活，学术惯例（Alain & Bengio 2016 等）用于检测 embedding 的**线性可分性**、
    纵向对比不同 embedding 版本。
- **扩展预留**：后续若引入像素 DW 数据集的 **prob**（软分类概率）信息，
  将新增 `MLPProb`（同为 64→9，输出概率分布、软标签训练），本模块结构可复用。

## 环境要求

- Python >= 3.12（项目使用 `.venv/` 虚拟环境 + [uv](https://docs.astral.sh/uv/) 管理）
- Docker（Qdrant 服务，与 KNN 评估共用）
- 依赖均已写入 `pyproject.toml`：`torch` / `qdrant-client` / `nicegui` / `matplotlib` / `numpy` 等

```bash
uv sync   # 同步 .venv 与 lock（依赖已在 pyproject.toml，无需新增）
```

## 数据来源

训练/评估数据直接读取 Qdrant Collection（默认 `xian_aef_embedding`，约 1022 万
像素点，每点 64 维 embedding + label 0-8）。数据导入由 `KNN_evaluation` 完成，
本模块只读；切换 collection 时采样地图按 collection 隔离重建。

**分层采样**：DW 标签严重不平衡（trees ≈ 621 万 vs snow_and_ice ≈ 240），
本模块复用 KNN 的采样地图（`qdrant_sampling_map.json`，point_id→label 地图）
做**分层随机采样**，只下载样本量（MB 级），避免全量下载 2.6GB 向量；
稀有类即使少于训练上限也保证验证集非空。

> **阶段说明**：训练分两个阶段——① 分层采样（**CPU 密集**：加载约 480MB
> 采样地图 + 组装 numpy 数据，与设备无关，占用 CPU/内存属正常）；② CUDA 训练
> （固定 GPU，`resolve_device` 禁用 CPU fallback，CUDA 不可用直接报错）。
> 训练循环已优化为**数据一次性驻留 GPU、epoch 内 shuffle/索引全在 GPU 上做**
> （消除 DataLoader 逐样本 collate 与每 batch CPU→GPU 搬运），实测训练阶段
> CPU 占用率由 ~9 核降至 ~1 核。训练结果会输出 GPU 设备名与 CUDA kernel
> 累计耗时（CUDA events 计时）作为 GPU 真实参与的证据。

## CLI 用法

```bash
# 训练 MLP_label（固定 CUDA，禁用 CPU fallback；默认分层采样：每类 train 2000 / val 500）
uv run python -m LinearProbe_evaluation.cli train --output-dir outputs/mlp_label

# 训练（自定义超参与采样规模；--model 选结构：mlp=LeNet-5 多层 / linear=单层 Linear Probe）
uv run python -m LinearProbe_evaluation.cli train \
  --epochs 50 --batch-size 4096 --lr 1e-3 --optimizer adam \
  --train-per-class 5000 --val-per-class 1000 --seed 42 \
  --model mlp --plot-dir outputs/mlp_label/plots

# 全量训练（每类不设上限）
uv run python -m LinearProbe_evaluation.cli train --train-per-class 0 --val-per-class 0

# 用 checkpoint 评估（默认按采样地图分层采样，每类 1000，只下载样本量）
uv run python -m LinearProbe_evaluation.cli evaluate \
  --checkpoint outputs/mlp_label/checkpoints/mlp_label_best.pt \
  --samples-per-class 1000 --plot --output eval_result.json

# 显式全量评估（--samples-per-class 0，会下载全库 2.6GB 向量，耗时长，不推荐）
uv run python -m LinearProbe_evaluation.cli evaluate \
  --checkpoint outputs/mlp_label/checkpoints/mlp_label_best.pt --samples-per-class 0

# Collection 统计（总点数 + 每类像素数）
uv run python -m LinearProbe_evaluation.cli stats [--json]

# 指定 Collection（默认 xian_aef_embedding；train / evaluate / stats 均支持 --collection）
uv run python -m LinearProbe_evaluation.cli train --collection google_aef_embedding --output-dir outputs/mlp_label
uv run python -m LinearProbe_evaluation.cli stats --collection google_aef_embedding --json
```

`train` 输出：
- `outputs/<dir>/checkpoints/mlp_label_best.pt` / `mlp_label_final.pt`（best 按 val_acc 保存）
- `outputs/<dir>/mlp_label_meta.json`（超参 + 每 epoch 历史 + 分类报告）
- `plots/train_curves.png`、`plots/confusion_matrix.png`
- 控制台报告：accuracy / macro-F1 / weighted-F1 / per-class precision-recall-F1

## 启动 WebUI（训练过程监测）

```bash
uv run python LinearProbe_evaluation/webui.py --port 8004
```

打开浏览器访问 `http://localhost:8004`。功能与 `KNN_evaluation/webui.py` 同风格：

- **Qdrant 连接 & Collection 状态**：实时显示可达性、总点数、状态；顶部提供
  **Collection 选择器**（预置 `google_aef_embedding` / `xian_aef_embedding` +
  自定义名称输入，切换后整个会话的 manager / 采样 / 训练跟随新 collection，
  localStorage 记忆上次选择）
- **训练参数配置**：epochs / batch size / lr / 优化器 / **模型结构（mlp | linear）** / 每类训练·验证样本数 / 验证比例 / seed（**训练设备固定 CUDA**，无 CPU fallback，参数区显示 GPU 设备名）
- **实时监测**：分层采样进度条 → 训练进度条 + 当前 loss → 逐 epoch 更新的
  loss / accuracy / macro-F1 曲线（matplotlib PNG）
- **结果展示**：best epoch、accuracy / macro-F1 / weighted-F1、per-class 指标表、
  验证集混淆矩阵图、训练结果 JSON 导出
- **协作式取消**：训练中可点击「中止训练」（threading.Event，逐 batch 检查）

## 测试

```bash
uv run pytest LinearProbe_evaluation/tests/ -v
```

测试使用内存 Fake Qdrant（`tests/helpers.py`），无需真实服务；覆盖模型线性性、
指标手算对照、分层采样（含稀有类）、训练循环、checkpoint 往返、WebUI 全流程。

## 目录结构

```
LinearProbe_evaluation/
  config.py         # Qdrant URL / 默认 Collection 与预置列表 / 64 维 / 9 类 / 默认超参
  label_mapping.py  # 复用 KNN 的 9 类标签映射（单一事实源）
  qdrant_client.py  # Qdrant 薄封装：健康检查/统计/scroll 读取
  dataset.py        # 像素数据读取 + 分层采样 + train/val 划分
  model.py          # MLPLabel：Linear(64, 9) 线性分类器
  trainer.py        # 训练循环 / 进度回调 / checkpoint / 取消
  metrics.py        # accuracy / macro-F1 / per-class / 混淆矩阵 / one-hot
  visualization.py  # 训练曲线与混淆矩阵（matplotlib）
  cli.py            # train / evaluate / stats 命令行入口
  webui.py          # NiceGUI Web 训练监测界面
  tests/            # 单元测试（Fake Qdrant，无需真实服务）
```

## 与 KNN 评估的关系

| | KNN_evaluation | LinearProbe_evaluation |
|---|---|---|
| 任务 | 向量检索 + embedding 质量评估 | 64→9 线性分类训练/评估 |
| 数据 | collection 选择器（google/xian，默认 google） | collection 选择器（默认 xian_aef_embedding，只读） |
| 采样 | 采样地图分层采样 | 复用同一采样地图 |
| 标签 | LABEL_NAMES（9 类） | re-export 同一映射 |
| 界面 | NiceGUI（8003） | NiceGUI（8004，训练监测） |
