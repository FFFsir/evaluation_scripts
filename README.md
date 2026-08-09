# Evaluation Scripts — 卫星影像像素级 embedding 质量评估系统

基于 **Qdrant 向量数据库**的像素级 embedding 质量评估项目。数据为卫星影像的
**SE（Satellite Embedding）** 与 **DW（Dynamic World 标签）** 文件：每张影像
128×128 像素，每像素一个 **64 维** embedding 向量与一个 **0–8 地物类别**标签
（共 9 类）。系统围绕同一套 Qdrant Collection 提供两套独立评估模块：

- **KNN 评估**（`KNN_evaluation/`）：KNN 检索 + embedding 质量指标（F1 / Purity@K /
  Recall@K）+ 双集合相似度热力图对比；
- **Linear Probe 评估**（`LinearProbe_evaluation/`）：训练 `MLP_label` 线性分类网络，
  评估 embedding 的线性可分性（macro-F1 为主指标）。

> ⚠️ **CLI 已弃用**：本系统仅通过 **WebUI** 提供使用入口（KNN 评估 `:8003`、
> LP 评估 `:8004`）。`cli.py` 文件保留仅作代码参考，不再作为使用方式。

---

## 📚 文档导航（建议从这里开始）

> **详细使用说明一律以教程为准**，本 README 仅作项目总览与入口导航。

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [**KNN_eval 教程**](docs/KNN_eval教程.md) | `KNN_evaluation` 项目结构分析、数据模型、工作流、WebUI 完整使用说明、输出物、测试与 FAQ | 使用 KNN 检索与评估、相似度热力图对比 |
| [**LinearProbe_eval 教程**](docs/LinearProbe_eval教程.md) | `LinearProbe_evaluation` 结构分析、分层采样、训练/评估工作流、WebUI 指南、输出物、FAQ | 训练 MLP_label 线性探针、评估线性可分性 |
| [qdrant-init-tutorial.md](qdrant-init-tutorial.md) | Qdrant 环境初始化教程（创建容器、默认 Collection、payload 索引） | 首次从零搭建环境 |

> 两篇教程均按当前代码核验（2026-08-09）。历史版本与本 README 相关的文档差异说明见
> [KNN_eval 教程附录 10.2](docs/KNN_eval教程.md#102-现有文档与代码差异说明)。

---

## 模块速览

| 维度 | KNN_evaluation | LinearProbe_evaluation |
|------|----------------|------------------------|
| **定位** | 向量检索 + embedding 质量评估 | 线性分类（探针）训练与评估 |
| **核心指标** | F1（KNN 分类准确率）、Purity@K、Recall@K | macro-F1（主指标）、Accuracy、Weighted-F1 |
| **WebUI** | NiceGUI，端口 **8003**（`webui.py`） | NiceGUI，端口 **8004**（`webui.py`） |
| **默认 Collection** | `google_aef_embedding` | `xian_aef_embedding` |
| **预置 Collection** | `google_aef_embedding` / `xian_aef_embedding`（相似度对比固定 google × xian） | `google_aef_embedding` / `xian_aef_embedding` |
| **模型** | HNSW 索引 + GPU/CPU 精确 KNN 引擎 | `MLPLabel`：`mlp`（多层，~15 万参数）/ `linear`（单层 Linear Probe，585 参数） |
| **数据访问** | 只读 + 可重建缓存（manifest / 采样地图 / 语料缓存） | 对 Qdrant 只读，复用采样地图 |
| **训练设备** | 评估可 GPU/CPU | **固定 CUDA**（无 CPU fallback） |

---

## 环境要求

- **Python ≥ 3.12.12**（`.python-version` 锁定）
- **[uv](https://docs.astral.sh/uv/)**：依赖与运行管理
- **Docker**：运行 Qdrant 向量数据库服务

```bash
# 安装依赖（含 CUDA 12.6 版 PyTorch，来自 pyproject.toml 指定 wheel 源）
uv sync
```

### 启动 Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

- REST `:6333` / gRPC `:6334`，数据挂载命名卷 `qdrant_data`（容器重建后数据保留）。
- Collection 创建与数据导入由 **KNN_evaluation** 负责；`LinearProbe_evaluation` 只消费数据，不建 Collection。
- **容器不会自动启动**：`cli.py` 仅在 `migrate` 路径调用幂等的 `_start_qdrant()`（`cli.py:679`），
  `webui.py` 定义了 `_start_qdrant()`（`webui.py:401`）但页面运行时不调用；导入/评估/WebUI 均需
  按上述命令先行启动容器，否则显示 Qdrant 不可达（❌）。
- 首次从零搭建请参考 [qdrant-init-tutorial.md](qdrant-init-tutorial.md)。

---

## 快速上手

### ① KNN 评估（`KNN_evaluation`）

启动 WebUI（http://localhost:8003），在页面内完成导入、检索、评估与相似度对比：

```bash
uv run python KNN_evaluation/webui.py --port 8003
```

### ② Linear Probe 评估（`LinearProbe_evaluation`）

启动 WebUI（http://localhost:8004），在页面内完成训练、评估与结果导出：

```bash
uv run python LinearProbe_evaluation/webui.py --port 8004
```

> ⚠️ Linear Probe 训练/评估**固定 CUDA**，无 GPU 环境会直接报错而非降级 CPU。

详细操作步骤见对应教程。

---

## 目录结构

```
evaluation_scripts/
├── KNN_evaluation/            # KNN 检索与 embedding 质量评估（WebUI :8003）
│   ├── webui.py               # NiceGUI 三页界面（GOOGLE / XIAN / SimilarityMatrix）
│   ├── data_loader.py         # SE/DW/TIF 扫描、加载与配对
│   ├── importer.py            # 批量导入 + 断点续传 + 重试
│   ├── qdrant_client.py       # QdrantManager：Collection 生命周期与索引管理
│   ├── searcher.py            # 向量检索（exact / ANN + 标签/UTM 过滤）
│   ├── gpu_knn.py             # GPU/CPU 分块精确 KNN 引擎
│   ├── metrics.py             # F1 / Purity@K / Recall@K 评估
│   ├── similarity_compare.py  # 双集合相似度热力图对比
│   ├── corpus_cache.py / manifest.py / sampling_map.py   # 三类可重建缓存
│   ├── cli.py                 # 命令行入口（已弃用，仅 WebUI 入口）
│   └── tests/                 # 355 个测试
├── LinearProbe_evaluation/    # 线性探针训练与评估（WebUI :8004）
│   ├── webui.py               # NiceGUI 训练监测界面
│   ├── dataset.py             # 分层采样 + train/val 划分
│   ├── model.py               # MLPLabel（mlp 多层 / linear 单层）
│   ├── trainer.py             # CUDA 训练循环 / checkpoint / 协作式取消
│   ├── cli.py                 # 命令行入口（已弃用，仅 WebUI 入口）
│   └── tests/                 # 79 个测试
├── src/                       # 早期工具模块（embedding 加载器、AEF 转换等）
├── docs/                      # 权威教程文档（见「文档导航」）
├── data_google/ data_xian/    # 两套集合的 SE/DW 数据目录
├── data_demo/                 # 示例导入目录（当前为空，需自行放入 SE/ 与 DW/ 数据）
├── tests/                     # src/ 顶层模块测试
├── outputs/                   # 评估结果 / 图表 / 相似度导出（见「输出说明」）
├── pyproject.toml             # 依赖声明（uv 管理，torch 走 CUDA 12.6 wheel 源）
└── qdrant-init-tutorial.md    # Qdrant 环境初始化教程
```

---

## 测试

```bash
# KNN 评估全量测试（355 个，含需要真实 Qdrant 的 integration 用例）
uv run pytest KNN_evaluation/tests/ -v

# 跳过需要 Qdrant 服务的用例
uv run pytest KNN_evaluation/tests/ -v -m "not integration"

# Linear Probe 测试（79 个，内存 Fake Qdrant；trainer 测试部分需真实 CUDA GPU）
uv run pytest LinearProbe_evaluation/tests/ -v
```

---

## 输出说明

| 目录 / 产物 | 来源 | 内容 |
|------|------|------|
| `outputs/mlp_label/` | LP 训练 | `checkpoints/mlp_label_best.pt` / `mlp_label_final.pt`、`mlp_label_meta.json`、`plots/`（训练曲线 + 混淆矩阵） |
| `outputs/evaluation/knn_eval/` | KNN WebUI 评估导出 | `{google\|xian}_knn_result_{时间戳}.json`、`{google\|xian}_knn_cm_<ts>.png`、`{google\|xian}_knn_pr_<ts>.png` |
| `outputs/evaluation/linearprobe/` | LP WebUI 导出 | `{collection缩写}_{variant}_result/curves/cm_<ts>.*`（JSON + PNG） |
| `outputs/evaluation/similarity/` | KNN WebUI 相似度对比 | 热力图 PNG、采样 JSON、google/xian 相似度矩阵 `.npy` |
| `outputs/diag_cpu/` | `diagnose_cpu.py` 诊断脚本 | CPU/GPU 性能画像输出 |

---

## 版本说明

本 README 为 2026-08-09 重写的项目总览入口（按 8 节骨架与 download_scripts 侧 README 统一排版），
替代旧版 KNN-only 总览文档。旧 README 与当前代码之间的已知差异（如 WebUI `--dir` 参数、K-F1 默认值、
Collection 分页等）已由 [KNN_eval 教程附录 10.2](docs/KNN_eval教程.md#102-现有文档与代码差异说明)
统一记录。

**后续变更（同日期）**：CLI 已弃用，使用方式统一为 WebUI（详见各模块教程第 3 章）。
