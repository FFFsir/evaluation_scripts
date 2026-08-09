# LinearProbe_eval 教程

> 面向 `LinearProbe_evaluation/` 模块的项目结构分析、端到端工作流程与 WebUI 使用指南。
> 核验时间：2026-08-09。文中所有 `file:line` 引用均指向 `D:\Project\光机所项目\evaluation_scripts` 下的源码。
> ⚠️ **CLI 已弃用**：本模块仅通过 WebUI（端口 8004）提供使用入口，`cli.py` 保留仅作代码参考。

## 目录

1. [项目概述](#1-项目概述)
2. [环境与依赖](#2-环境与依赖)
3. [WebUI 使用方法](#3-webui-使用方法)
4. [项目结构分析](#4-项目结构分析)
5. [数据模型与分层采样](#5-数据模型与分层采样)
6. [工作流程](#6-工作流程)
7. [输出物说明](#7-输出物说明)
8. [测试与诊断](#8-测试与诊断)
9. [常见问题（FAQ）](#9-常见问题faq)
10. [附录](#10-附录)

---

## 1. 项目概述

`LinearProbe_evaluation` 是一个基于 **Qdrant** 的 **Linear Probe（线性探针）评估系统**，核心任务是：从 Qdrant 中读取已有的**像素级 embedding（64 维）**与 **DW 硬分类标签（0-8）**，训练一个线性分类网络 **`MLP_label`**（64 维输入 → 9 维输出），以评估 embedding 的线性可分性 / 分类精度，通过 **NiceGUI Web**（训练过程监测）提供使用入口。

要点：

- **对 Qdrant 只读**：数据导入与 collection 生命周期由 `KNN_evaluation/` 负责，本模块只消费数据（`qdrant_client.py:1-5` 的 docstring 明确说明）。
- **共享基础设施**：与 `KNN_evaluation/` 共享同一个 Qdrant Collection、采样地图（`qdrant_sampling_map_<collection>.json`）与 9 类标签映射。
- **两类模型结构**（WebUI「模型结构」，默认 `mlp`）：
  - **`mlp`（LeNet-5 规模多层 MLP，约 15 万参数）**：`Linear(64,256) → ReLU → Dropout(0.2) → Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,256) → ReLU → Dropout(0.2) → Linear(256,9)`，可拟合 embedding 中的非线性、追求更高精度（`model.py:47-107`，`README.md:20-22`）；
  - **`linear`（标准 Linear Probe，585 参数）**：单一 `Linear(64,9)`（`hidden_dims=()` 时由 `model.py:83` 的 `nn.Linear(prev, num_classes)` 构建），无隐藏层、无激活，学术惯例（Alain & Bengio 2016 等）用于检测 embedding 的**线性可分性**、纵向对比不同 embedding 版本（`model.py:9-14` docstring，`README.md:23-25`）。
- **训练设备固定 CUDA**：`resolve_device` 禁用 CPU fallback（`trainer.py:80-99`），CUDA 不可用时直接报错，不静默降级。
- **主指标为 macro-F1**：数据严重类别不平衡，accuracy 会被大类主导，以 macro-F1 为主要评估指标。

---

## 2. 环境与依赖

**环境要求**（`README.md:29-37`）：

- Python >= 3.12（项目使用 `.venv/` + [uv](https://docs.astral.sh/uv/) 管理，`pyproject.toml` 指定版本）
- Docker（运行 Qdrant 服务）
- 依赖均已写入根目录 `pyproject.toml`：`torch` / `qdrant-client` / `nicegui` / `matplotlib` / `numpy` / `tqdm` 等；测试依赖 `pytest`

```bash
# 同步虚拟环境与依赖
cd D:\Project\光机所项目\evaluation_scripts
uv sync
```

**启动 Qdrant（与 KNN 评估共用）**：

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
# 默认连接地址 http://localhost:6333（config.py:5）
```

> Qdrant 的 collection 创建与数据导入由 `KNN_evaluation` 完成（通过其 WebUI 的「数据导入」面板）。
> 本模块在首次使用时若发现采样地图缺失/过期，会自动重建（见第 5 节）。

---

## 3. WebUI 使用方法

### 3.1 启动

```bash
uv run python LinearProbe_evaluation/webui.py --port 8004
# 可选 --qdrant-url http://localhost:6333
```

浏览器访问 **http://localhost:8004**。

### 3.2 ① Qdrant 连接 & Collection 状态（默认展开）

- **状态区**：实时显示 Qdrant 可达性、collection 是否存在、总点数、向量维度、状态；点击 **「刷新状态」** 重新获取。
- **Collection 选择器**：预置 `google_aef_embedding` / `xian_aef_embedding` 下拉；选择后**必须点击「切换 collection」**才生效（切换后整个会话的 manager / 采样 / 训练跟随新 collection）。
- **自定义 collection**：在输入框填写名称后点 **「使用自定义 Collection」**（校验：非空、不含 `/` 或 `\`），新名称会追加到下拉列表。
- **记忆功能**：上次选择的 collection 存入浏览器 `localStorage`（key `comet.linearprobe.current_collection`，`webui.py:77`），下次打开页面自动恢复，非法记录回退到默认。

### 3.3 ② MLP_label 训练（默认展开）

**参数面板**（「参数说明」子展开中有详细说明）：

| 控件 | 默认值 | 说明 |
|---|---|---|
| Epochs | 500 | 训练轮数（1-1000） |
| Batch Size | 40960 | 批大小（16-65536） |
| Learning Rate | 0.01000 | 学习率（步进 0.0001） |
| 优化器 | adam | adam / sgd |
| 模型结构 | mlp | mlp=LeNet-5 规模多层 / linear=单层 Linear Probe |
| 每类训练样本数 | **10000** | 分层采样上限，0=不限制 |
| 每类验证样本数 | **1000** | 分层采样上限，0=不限制 |
| 验证比例 | 0.20 | 每类先按比例切出验证集 |
| Seed | 42 | 随机种子 |

**训练过程**：

1. 点 **「开始训练」**：先显示采样进度条（逐类上报），随后训练进度条 + 当前 loss。
2. **实时曲线区**：每个 epoch 结束后刷新「loss + accuracy/macro-F1」双面板图（matplotlib PNG）。
3. 训练中可随时点 **「中止训练」**（协作式取消，`threading.Event` 逐 batch 检查，退出码/提示为「取消」）。

### 3.4 ③ 结果展示

训练完成后显示：

- **best epoch 摘要**：best 按 val_acc 保存的轮次及其指标；
- **指标区**：Accuracy / Macro-F1 / Weighted-F1；
- **per-class 表格**：类别 / Precision / Recall / F1 / Support；
- **验证集混淆矩阵图**（heatmap）；
- **导出按钮**：「导出 JSON」「导出图片图表」（训练完成前隐藏）。

### 3.5 ④ 导出（写入 `outputs/evaluation/linearprobe/`）

文件名遵循 `<collection缩写>_<variant>_<内容>_<时间戳>` 规则（`webui.py:630-711`）：

- **导出 JSON**：如 `xian_mlp_result_20260809_123456.json` —— 含 model/architecture（variant、hidden_dims、num_parameters）、label_names、每 epoch history、best epoch 与指标、val+train 分类报告、elapsed、device、gpu_kernel_seconds。
- **导出图片图表**：如 `xian_mlp_curves_<ts>.png` 与 `xian_mlp_cm_<ts>.png`（训练曲线 + 混淆矩阵）。
- collection 缩写：`google_aef_embedding → google`、`xian_aef_embedding → xian`，自定义名称原样使用。

### 3.6 使用说明区

页面底部有折叠的「使用说明」markdown，包含模块简介与操作指引。

---

## 4. 项目结构分析

### 4.1 目录树

```
LinearProbe_evaluation/
├── __init__.py            # 包标记
├── config.py              # 全局常量：Qdrant URL / collection / 超参默认值 / 输出目录
├── label_mapping.py       # 9 类标签映射（re-export KNN 单一事实源）
├── qdrant_client.py       # QdrantManager 薄封装：健康检查 / 统计 / scroll 读取
├── dataset.py             # 像素数据读取 + 分层采样 + train/val 划分
├── model.py               # MLPLabel 网络（mlp 多层 / linear 单层）
├── trainer.py             # CUDA 训练循环 / 进度回调 / checkpoint / 协作式取消
├── metrics.py             # accuracy / macro-F1 / per-class / 混淆矩阵 / one-hot
├── visualization.py       # 训练曲线与混淆矩阵图（matplotlib Agg）
├── webui.py               # NiceGUI Web 训练监测界面（端口 8004）
├── diagnose_cpu.py        # 独立诊断脚本：各阶段 CPU/墙钟耗时与 GPU 参与验证
├── linear-probe-evaluation-requirements.md  # 需求规格文档
├── README.md              # 模块自带 README
├── cli.py                 # 命令行入口（已弃用，仅 WebUI 入口）
└── tests/                 # 单元测试（内存 Fake Qdrant，无需真实服务）
    ├── conftest.py        # fake_manager fixture
    ├── helpers.py         # FakeClient / FakeQdrantManager / 合成数据构造
    ├── test_cli.py        # 5 个
    ├── test_dataset.py    # 16 个
    ├── test_metrics.py    # 8 个
    ├── test_model.py      # 8 个
    ├── test_trainer.py    # 15 个（部分需真实 CUDA GPU，见第 8 节）
    └── test_webui.py      # 27 个
```

### 4.2 模块职责与关键符号

| 模块 | 职责 | 关键类 / 函数 / 常量 |
|---|---|---|
| `config.py` | 全局常量，无环境变量读取 | `QDRANT_URL = "http://localhost:6333"` (:5)、`DEFAULT_COLLECTION = "xian_aef_embedding"` (:11)、`PRESET_COLLECTIONS = ["google_aef_embedding", "xian_aef_embedding"]` (:12)、`VECTOR_SIZE = 64` (:19)、`NUM_CLASSES = 9` (:20)、`DEFAULT_EPOCHS = 500` / `DEFAULT_BATCH_SIZE = 40960` / `DEFAULT_LR = 1e-2` (:24-26)、`DEFAULT_TRAIN_PER_CLASS = 10000` / `DEFAULT_VAL_PER_CLASS = 1000` (:29-30)、`DEFAULT_VAL_RATIO = 0.2` (:31)、`DEFAULT_OUTPUT_DIR = Path("outputs/mlp_label")` (:36) |
| `label_mapping.py` | 标签映射 re-export（7 行） | `from KNN_evaluation.label_mapping import LABEL_NAMES, LABEL_IDS` (:7)，单一事实源在 KNN 侧 |
| `qdrant_client.py` | Qdrant 薄封装 | `QdrantManager(url, collection_name, timeout)` (:12)：`client` 懒加载 (:27)、`health_check()` (:32)、`collection_exists()` (:40)、`collection_info()` (:47)、`scroll_vectors_and_labels(batch_size, limit)` (:61) |
| `dataset.py` | 数据读取与采样 | `PixelDataset` dataclass（`X:(N,64) float32`、`y:(N,) int64 0-8`、`point_ids`）(:24-45)、`empty_dataset()` (:48)、`_retrieve_batch()` (:56)、`stratified_train_val_split(manager, *, train_per_class, val_per_class, val_ratio, seed, ...)` (:109)、`sample_dataset(manager, *, samples_per_class, seed, ...)` (:190)、`load_full_dataset(manager, *, max_points, ...)` (:230)、`CancelledError` (:305) |
| `model.py` | 网络结构 | `DEFAULT_HIDDEN_DIMS = (256,256,256)` (:35)、`DEFAULT_DROPOUT = 0.2` (:36)、`VARIANT_HIDDEN_DIMS` (:41-44)、`MLPLabel(nn.Module)` (:47，`hidden_dims=()` 即单层 linear)、`build_model(variant, in_features=64, num_classes=9, dropout=0.2)` (:110)、`count_parameters()` (:133) |
| `trainer.py` | 训练循环 | `TrainConfig` dataclass (:44，`device="cuda"` 固定)、`TrainResult` (:62，含 `best_val_macro_f1`、`gpu_kernel_seconds`)、`resolve_device()` (:80，拒绝 cpu)、`predict_labels()` (:110)、`load_mlp_label()` (:191，`weights_only=True`)、`train_mlp(train_ds, val_ds, config, *, progress_callback, cancel_event)` (:212)、`_save_meta()` (:394) |
| `metrics.py` | 评估指标（纯 numpy） | `confusion_matrix()` (:10)、`accuracy()` (:23)、`per_class_accuracy()` (:32)、`macro_f1()` (:71)、`weighted_f1()` (:79)、`classification_report() -> dict` (:88)、`to_one_hot()` (:126) |
| `visualization.py` | 绘图（matplotlib Agg + 中文字体） | `plot_training_curves(history, save_path=None)` (:50，双面板：loss + acc/macro-F1)、`plot_confusion_matrix(cm, label_names, save_path=None, title=...)` (:100)、`training_curves_base64()` / `confusion_matrix_base64()`（WebUI 用）(:36-47) |
| `webui.py` | NiceGUI 界面（端口 8004） | `DEFAULT_PORT = 8004` (:55)；页面 `@ui.page("/")` `index()` (:182-618)；训练 `do_train()` (:360-516)；导出 `_export_lp_results()` (:630-711) |
| `diagnose_cpu.py` | 独立诊断脚本 | `main()` (:42-93)：分 4 阶段测量墙钟/CPU 耗时并验证 GPU 参与 |
| `cli.py` | 已弃用 | 命令行入口（仅作代码参考，不提供使用说明） |
| `tests/` | 单元测试 | 共 79 个；内存 Fake Qdrant（`tests/helpers.py`），无需真实服务（trainer 测试部分需 CUDA，见第 8 节） |

### 4.3 关键设计要点

- **全 GPU 训练**：train/val 数据一次性拷贝驻留 GPU（`trainer.py:267-268`），GPU 随机数生成器 + epoch 内 shuffle（`torch.randperm`，`trainer.py:272-273, 289`）与 batch 索引（`trainer.py:297-300`）全在 GPU 上完成，不使用 DataLoader，消除逐样本 collate 与每 batch CPU→GPU 搬运。实测训练阶段 CPU 占用率由 ~9 核降至 ~1 核（`README.md:50-56`）。
- **原子写 checkpoint**：先写 `.tmp` 再 `os.replace`（`trainer.py:170-171, 187-188`），避免断电/中断产生损坏文件。
- **checkpoint 保存架构元数据**：`architecture` 字段含 `class / in_features / num_classes / hidden_dims / dropout`（`trainer.py:173-179`），无 `variant` 字段（variant 由 `hidden_dims` 推导）；`load_mlp_label` 从该字段重建模型，`hidden_dims=[]` 表示 linear 单层（与「空 = 默认」区分开，`trainer.py:203-204`）。
- **取消是协作式的**：训练与采样均接受 `threading.Event` 作为 cancel token，逐 batch 检查（epoch 层 `trainer.py:281-282`、batch 层 `trainer.py:295-296`；采样侧 `dataset.py:81-82, 178-179, 225-226, 271-272`）。

---

## 5. 数据模型与分层采样

### 5.1 标签映射（9 类 DW 土地覆盖）

单一事实源在 `KNN_evaluation/label_mapping.py`，`LinearProbe_evaluation` 仅 re-export（`label_mapping.py:7`）：

| ID | 标签（英文） | 中文 |
|---|---|---|
| 0 | water | 水 |
| 1 | trees | 树 |
| 2 | grass | 草 |
| 3 | flooded_vegetation | 被淹植被 |
| 4 | crops | 农作物 |
| 5 | shrub_and_scrub | 灌木与矮树丛 |
| 6 | built | 建筑物 |
| 7 | bare | 空地 |
| 8 | snow_and_ice | 冰雪 |

### 5.2 类别极度不平衡

默认 collection `xian_aef_embedding` 约 **1022 万像素点**，每点 64 维 embedding + label 0-8。类别分布严重失衡（`README.md:45-48`，`linear-probe-evaluation-requirements.md:31-41`）：

- `trees` ≈ 621 万 vs `snow_and_ice` ≈ 240，**不平衡比约 25874:1**。
- 因此以 **macro-F1**（各类 F1 的算术平均）为主指标，accuracy 会被大类主导。

### 5.3 分层采样算法（`stratified_train_val_split`，`dataset.py:109-188`）

为避免全量下载 2.6GB 向量，复用 KNN 的**采样地图**（`qdrant_sampling_map_<collection>.json`，point_id→label 地图）在本地做随机抽样，再按精确 ID 通过 `retrieve` 下载向量（MB 级）。算法要点：

1. 前置校验：Qdrant 健康、collection 存在且非空（`dataset.py:144-150`）。
2. 读取采样地图（`ensure_sampling_map(manager)`，来自 `KNN_evaluation.sampling_map`；地图缺失或与 collection 指纹不符时自动重建，`dataset.py:142, 152`）。
3. 对每个类别：用 `random.Random(seed)` 打乱该类所有 point_id（可复现，`dataset.py:153, 159`）。
4. **先按 `val_ratio`（默认 0.2）切出验证集**，再用 `val_per_class` 上限截断；**稀有类保证验证集至少 1 个样本**（`dataset.py:163-168`）。
5. 其余样本按 `train_per_class` 上限截断进训练集；**train/val 无重叠**（`dataset.py:171-174`）。
6. 最后通过 `_retrieve_batch` 批量 `client.retrieve(ids, with_vectors=True, with_payload=["label"])` 下载向量与标签（`dataset.py:181-186`）。

> 示例：稀有类 `snow_and_ice`（240 个）→ val 48 / train 192（`dataset.py:125-126`）。训练集约 16k 样本 × 64 维 × 4B ≈ 4MB，可整体驻留 GPU。

### 5.4 两种数据读取路径

| 路径 | 函数 | 用途 | 下载量 |
|---|---|---|---|
| 采样路径（默认） | `stratified_train_val_split` / `sample_dataset` | 训练 / 评估 | MB 级 |
| 全量路径 | `load_full_dataset(manager, max_points=...)`（scroll 整个 collection） | 显式全量评估 | 2.6GB（`dataset.py:230-303`） |

---

## 6. 工作流程

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. 环境：uv sync + Docker 启动 Qdrant（:6333）                        │
│ 2. 数据：由 KNN_evaluation 完成 collection 创建与数据导入（本模块只读）│
│ 3. 采样地图：首次使用时 ensure_sampling_map 自动构建/校验              │
│    qdrant_sampling_map_<collection>.json（写入当前工作目录）          │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────── WebUI 训练管线 ────────────────────────────┐
│                                                                      │
│  QdrantManager（qdrant_client.py，连接 localhost:6333）              │
│    │                                                                │
│    ├─ stratified_train_val_split() → (train_ds, val_ds)              │
│    │   （分层采样，CPU 密集阶段，进度回调逐类上报）                    │
│    │                    │                                           │
│    │                    ▼                                           │
│    │           train_mlp(train_ds, val_ds, TrainConfig)              │
│    │           · CrossEntropyLoss + Adam/SGD（固定 CUDA）             │
│    │           · 全 GPU 训练循环，逐 batch 进度回调                  │
│    │           · best（按 val_acc）与 final checkpoint 原子保存       │
│    │           · classification_report（val + train，用 best 权重）  │
│    │                    │                                           │
│    │                    ▼                                           │
│    │           WebUI：实时曲线 + 结果区 + 导出 JSON/图片              │
│    │                                                                │
└──────────────────────────────────────────────────────────────────────┘
```

各阶段输出物汇总见[第 7 节「输出物说明」](#7-输出物说明)。

---

## 7. 输出物说明

### 7.1 各阶段输出物总览

| 阶段 | 产物 |
|---|---|
| WebUI 训练 | `outputs/mlp_label/checkpoints/mlp_label_best.pt`、`mlp_label_final.pt`、`outputs/mlp_label/mlp_label_meta.json`、`outputs/mlp_label/plots/train_curves.png`、`confusion_matrix.png` |
| WebUI 结果区 | 浏览器内实时曲线与结果表 |
| WebUI 导出 | 「导出 JSON / 导出图片图表」写入 `outputs/evaluation/linearprobe/`（见 3.5） |
| `diagnose_cpu.py` | CPU/GPU 性能画像输出（`outputs/diag_cpu/`） |

### 7.2 Checkpoints（`trainer.py:159-188, 355-390`）

- `checkpoints/mlp_label_best.pt`：按 **val_acc** 最优的 epoch 权重；
- `checkpoints/mlp_label_final.pt`：最后 epoch 的权重（记录了该 epoch 自身的 val 指标）；
- 内容：`model_state_dict` + `architecture`（`class / in_features / num_classes / hidden_dims / dropout`，`trainer.py:173-179`）+ `label_names` + `epoch` + `val_accuracy` + `val_macro_f1` + `optimizer` + `lr` + `seed`。加载用 `load_mlp_label`（`weights_only=True` 安全加载）。

### 7.3 `mlp_label_meta.json`（`trainer.py:_save_meta` :394-422）

完整超参（`asdict(TrainConfig)`）+ 架构信息（variant、dims、dropout、参数量）+ `label_names` + 每 epoch `history`（epoch/train_loss/train_acc/val_loss/val_acc/val_macro_f1）+ best 指标 + `val_report` / `train_report`（classification_report）+ 训练耗时（墙钟 + CUDA kernel 累计耗时）+ 设备信息。

### 7.4 Plots（`visualization.py`）

| 文件 | 生成时机 | 内容 |
|---|---|---|
| `plots/train_curves.png` | 训练 | 左：train/val loss；右：train/val accuracy + val macro-F1 |
| `plots/confusion_matrix.png` | 训练（val 非空时） | 验证集混淆矩阵 heatmap（Blues） |
| WebUI 导出 `outputs/evaluation/linearprobe/*.png` | WebUI 导出 | 曲线 + 混淆矩阵 |

---

## 8. 测试与诊断

### 8.1 单元测试

```bash
uv run pytest LinearProbe_evaluation/tests/ -v
```

- 共 **79 个测试**：test_cli 5 / test_dataset 16 / test_metrics 8 / test_model 8 / test_trainer 15 / test_webui 27。
- **无需真实 Qdrant**：测试使用内存 Fake Qdrant（`tests/helpers.py` 的 `_FakeClient` / `FakeQdrantManager`）；`test_dataset.py` 还 monkeypatch 掉 480MB 的真实采样地图（`_patch_map`）。
- **trainer 测试部分需要真实 CUDA GPU**：`test_trainer.py` 共 15 个测试，其中约 10 个需要真实 GPU；`test_resolve_device` / `test_resolve_device_cuda_unavailable`（模拟 CUDA 不可用断言拒绝 cpu）、`test_predict_labels_shape`（显式用 `device="cpu"`）、`test_empty_train_raises` / `test_invalid_optimizer`（在 `resolve_device` 之前就抛 `ValueError`）在无 GPU 环境仍可运行。这是项目「无 CPU fallback」策略的预期行为。
- WebUI 测试用 `_FakeUI` 驱动整个页面（按钮、回调、导出、collection 切换），不启动真实 NiceGUI 服务。

### 8.2 诊断脚本 `diagnose_cpu.py`

```bash
uv run python LinearProbe_evaluation/diagnose_cpu.py
```

独立运行，需要 CUDA + 可达的 Qdrant。分 4 阶段测量墙钟 / 进程 CPU 时间并输出 CPU 占用率（cpu/wall，>1 表示多核参与）：

1. 采样地图加载（~480MB JSON）；
2. 分层划分（train 2000 / val 500，seed 42）；
3. CUDA 训练（epochs=10, batch=2048，输出到 `outputs/diag_cpu`）；
4. best checkpoint 在验证集上的推理。

用于验证「采样阶段 CPU 密集、训练阶段 GPU 主导」的性能画像（`README.md:50-56`）。

---

## 9. 常见问题（FAQ）

| # | 问题 | 说明 |
|---|---|---|
| 1 | **强制 CUDA，无 CPU fallback** | 训练固定 CUDA（`trainer.py:80-99`）。无 GPU 环境无法训练，会直接报错而非降级到 CPU。 |
| 2 | **WebUI「参数说明」中 batch size 文案过时** | 文案写「默认 2048，GPU 上建议 2048-8192」（`webui.py:303`），但实际默认值已是 **40960**（`config.py:25`，输入框默认值）。以输入框默认值为准。 |
| 3 | **WebUI 未暴露全部超参** | `weight_decay`、`log_interval` 不在 WebUI 参数面板中，走 `TrainConfig` 默认值（0.0 / 20）。 |
| 4 | **采样地图写在当前工作目录** | `qdrant_sampling_map_<collection>.json` 生成于运行命令时的工作目录，按 collection 隔离（`KNN_evaluation/sampling_map.py:20-25`）；首次使用某 collection 时自动构建，耗时较长（解析约 480MB 地图属正常）。 |
| 5 | **`data` 层的点 ID 类型** | `load_full_dataset` 预分配 `point_ids` 用 `dtype=object` 再转 str，避免 numpy 2.x 中 `np.empty(dtype=str)` 变成 `<U1` 截断 UUID（`dataset.py:264-266`）—— 属于已修复的坑，如遇到 UUID 被截断请确认 numpy 版本。 |
| 6 | **WebUI 无文件上传** | 所有数据来自 Qdrant collection（预置 + 自定义名称），页面上没有上传入口。 |
| 7 | **checkpoint 只能由本模块加载** | `mlp_label_best.pt` 内含架构元数据，加载需 `load_mlp_label`（`trainer.py:191`），直接 `torch.load` 拿不到可推理模型。 |
| 8 | **「中止训练」是协作式取消** | 需要训练循环内部检查 cancel token（epoch 层 `trainer.py:281-282`、batch 层 `trainer.py:295-296`），点击后不会立即硬中断，而是在下一个检查点优雅退出。 |

---

## 10. 附录

### 10.1 关键函数索引

（行号以 2026-08-09 代码为准，供代码阅读定位。）

| 功能 | 类 / 函数 / 常量 | 位置 |
|---|---|---|
| 全局配置 | `QDRANT_URL` / `DEFAULT_COLLECTION` / `PRESET_COLLECTIONS` / `VECTOR_SIZE` / `NUM_CLASSES` / `DEFAULT_*` | `config.py:5-36` |
| 标签映射 re-export | `LABEL_NAMES` / `LABEL_IDS` | `label_mapping.py:7` |
| Qdrant 薄封装 | `QdrantManager`（`health_check` / `collection_exists` / `collection_info` / `scroll_vectors_and_labels`） | `qdrant_client.py:12`（:32 / :40 / :47 / :61） |
| 数据容器 / 采样 / 全量读取 | `PixelDataset` / `empty_dataset` / `_retrieve_batch` / `stratified_train_val_split` / `sample_dataset` / `load_full_dataset` | `dataset.py:24` / :48 / :56 / :109 / :190 / :230 |
| 网络结构 | `MLPLabel` / `build_model` / `count_parameters` | `model.py:47` / :110 / :133 |
| 设备解析 / 训练 / 加载 / 预测 | `resolve_device` / `train_mlp` / `load_mlp_label` / `predict_labels` | `trainer.py:80` / :212 / :191 / :110 |
| 评估指标 | `confusion_matrix` / `accuracy` / `macro_f1` / `weighted_f1` / `classification_report` / `to_one_hot` | `metrics.py:10` / :23 / :71 / :79 / :88 / :126 |
| 绘图 | `plot_training_curves` / `plot_confusion_matrix` / `training_curves_base64` / `confusion_matrix_base64` | `visualization.py:50` / :100 / :36 / :41 |
| WebUI | `index` / `do_train` / `_export_lp_results` / `run` | `webui.py:182` / :360 / :630 / :717 |
| 诊断脚本 | `main`（4 阶段测量） | `diagnose_cpu.py:42` |
| CLI 入口（已弃用） | `cmd_train` / `cmd_evaluate` / `cmd_stats` / `_build_parser` / `main` | `cli.py:59` / :194 / :252 / :292 / :368 |

### 10.2 与 KNN 评估的关系速查

| 维度 | KNN_evaluation | LinearProbe_evaluation |
|---|---|---|
| 任务 | 向量检索 + embedding 质量评估 | 64→9 线性分类训练/评估 |
| 数据 | collection 选择器（google/xian，默认 google） | collection 选择器（默认 xian_aef_embedding，只读） |
| 采样 | 采样地图分层采样 | 复用同一采样地图 |
| 标签 | LABEL_NAMES（9 类） | re-export 同一映射 |
| 界面 | NiceGUI（8003） | NiceGUI（8004，训练监测） |
| 模型 | 向量索引（如 HNSW） | MLP_label（mlp / linear） |
