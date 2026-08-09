# Outcome

基于 Qdrant `pixel_embeddings` Collection（约 1022 万像素点，每点 64 维 SE embedding + DW 硬分类 label 0-8）实现 **MLP_label** 线性分类评估系统，代码全部位于 `LinearProbe_evaluation/`：

- 提供 64 维输入 → 9 维输出的分类网络（两种结构可选：`mlp` = LeNet-5 规模多层全连接 MLP 约 15 万参数；`linear` = 标准 Linear Probe 单层 585 参数），9 维输出对应 9 个 DW 标签的硬分类独热码语义（CrossEntropyLoss，argmax 取类别）。
- 训练/评估数据**只按采样地图（`qdrant_sampling_map.json`）分层随机采样、分批 retrieve 下载样本量（MB 级），绝不全库下载**；类别严重不平衡（trees ≈ 621 万 vs snow_and_ice = 240）下每类先按 `val_ratio` 切出验证集再受 `train_per_class`/`val_per_class` 上限约束，稀有类保证验证集非空。
- **训练设备固定 CUDA**：`resolve_device` 禁用 CPU fallback（CUDA 不可用直接报错），训练循环为全 GPU 模式（数据一次性驻留 GPU、epoch 内 shuffle/索引在 GPU 上做、loss 在 GPU 计算），并输出 GPU 设备名与 CUDA kernel 累计耗时作为 GPU 真实参与证据。
- 提供 CLI（`train` / `evaluate` / `stats`）与 NiceGUI WebUI（端口 8004）：Qdrant 状态、训练参数配置（含模型结构选择）、采样/训练进度、逐 epoch 实时曲线、per-class 指标表、混淆矩阵、结果 JSON 导出、协作式取消、并发训练保护。
- 单元测试 59 项全绿（内存 Fake Qdrant，无需真实服务）；真实 Qdrant 端到端验证通过（10 epoch、每类 2000/500：val_acc 0.844 / macro-F1 0.859）。

# Scope

- `LinearProbe_evaluation/` 新增模块：`config.py`、`label_mapping.py`（re-export KNN 权威 9 类标签映射）、`qdrant_client.py`（薄封装：健康检查/统计/scroll 读取）、`dataset.py`（分层采样/train-val 划分/全量读取）、`model.py`（MLPLabel 双结构 + build_model 工厂）、`trainer.py`（训练循环/进度回调/checkpoint/取消/CUDA 计时）、`metrics.py`（accuracy/macro-F1/weighted-F1/per-class/混淆矩阵/one-hot）、`visualization.py`（训练曲线/混淆矩阵 matplotlib 图）、`cli.py`（train/evaluate/stats）、`webui.py`（NiceGUI 训练监测）、`diagnose_cpu.py`（分阶段 CPU 诊断）、`tests/`（59 项）、`README.md`、`linear-probe-evaluation-requirements.md`。
- 训练管线：分层采样（复用 KNN `ensure_sampling_map`）→ CUDA 训练 → checkpoint（best/final，按 `hidden_dims` 重建结构）→ 元数据 JSON → 图表/报告/JSON 导出。
- 修改 `.comet/config.yaml` 的 `native.snapshot.exclude`：排除运行时大数据（`.venv/`、`qdrant_corpus_cache/`、`qdrant_sampling_map.json`、`data/`、`data_demo/`、`outputs/`、`result/`），快照只保留代码与文档。

# Non-goals

- 不修改 `KNN_evaluation/` 现有实现（仅复用其 `label_mapping` 与 `sampling_map`）。
- 不引入像素 DW 数据集的 **prob（软分类概率）** 信息；`MLPProb`（64→9 软标签网络）仅在后续引入 prob 时新增，本次不实现。
- 不做每 epoch 在线重采样（训练数据固定为训练开始前采样的一次性数据集）。
- 不做分布式训练、多 GPU 或 mixed precision。
- 不将本模块作为独立 pip 包发布（依赖与 KNN 同仓库部署）。

# Acceptance examples

- `uv run pytest LinearProbe_evaluation/tests/ -q` 全部通过（59 项，无需真实 Qdrant）。
- `uv run python -m LinearProbe_evaluation.cli stats` 输出总点数（1022 万）与每类像素数，与 Qdrant count API 一致。
- `uv run python -m LinearProbe_evaluation.cli train --model mlp`（真实 Qdrant）完成 采样→CUDA 训练→checkpoint/图/报告 全流程，输出 GPU 设备名与 GPU kernel 耗时；`--model linear` 同样可训练（585 参数单层）。
- `uv run python -m LinearProbe_evaluation.cli evaluate --checkpoint ... --samples-per-class 1000` 默认分层采样评估（约 8240 样本），秒级完成，不触发全库下载。
- `--device cpu`（若仍有该参数）被拒绝；CUDA 不可用时 `train` 直接报错而非静默回退 CPU。
- WebUI（`uv run python LinearProbe_evaluation/webui.py --port 8004`）可打开，训练过程中曲线逐 epoch 刷新、可中止；模型结构下拉（mlp/linear）生效。
- 训练阶段 CPU 占用率约 1 核（`diagnose_cpu.py` 实测，非多核飙升）。

# Constraints and invariants

- 网络接口：64 维输入 → 9 维输出；`hidden_dims=()` 为单层 linear、非空为多层 MLP；checkpoint 记录 `hidden_dims` 且加载时正确重建（`[]` 不能被 `or DEFAULT` 误判为多层）。
- 数据读取绝不全库下载：采样/训练/评估只经采样地图按 ID 分批 `retrieve` 样本量。
- 设备约束：`resolve_device` 只接受 `cuda`/`auto`（均解析为 cuda）；`cpu` 与未知值抛 `ValueError`；CUDA 不可用抛 `RuntimeError`；训练/评估数据与模型均有 `assert .is_cuda` 防御断言。
- 标签单一事实源：`label_mapping.py` re-export `KNN_evaluation.label_mapping.LABEL_NAMES`，不重复定义。
- 可复现：采样（seed）、shuffle（GPU generator 显式 seed）、权重初始化均受 `--seed` 控制。
- 协作式取消：采样/训练循环逐批检查 `threading.Event`；WebUI 同时只允许一个训练任务（`state["training"]` 保护）。
- 训练阶段 CPU 预算：数据驻留 GPU、epoch 内零 CPU 搬运，`_evaluate` 分块 forward 限制激活显存。

# Decisions

- 网络结构采用**两种可选**（用户 2026-08-03 确认）：`mlp`（LeNet-5 规模，`(256,256,256)` + ReLU + Dropout(0.2)，150,537 参数，默认）与 `linear`（标准 Linear Probe 单层 `Linear(64,9)`，585 参数，测 embedding 线性可分性）；CLI `--model` 与 WebUI「模型结构」下拉一致。
- 设备策略（用户确认）：**只提供 CUDA，禁用 CPU fallback**；CUDA 不可用直接报错。
- 训练数据来源（用户确认）：不进行全库下载，每次按采样地图（manifest map）流水线式分批下载部分内容进行训练。
- 训练循环优化（诊断驱动）：数据一次性驻留 GPU、GPU `randperm` shuffle + GPU gather 索引，实测训练阶段 CPU 占用由 ~9 核降至 ~1 核。
- 指标以 macro-F1 为主（类别严重不平衡），同时输出 accuracy / weighted-F1 / per-class precision-recall-F1 / 混淆矩阵。
- 用 `.venv/` + uv 管理依赖（`uv sync`；新增依赖用 `uv add`）。

# Open questions

- 已确认（2026-08-03）：change 名称 `linear-probe-evaluation`，共享理解以本 brief 的 Outcome/Scope/Non-goals/Acceptance examples 为准；用户确认将已完成的 `LinearProbe_evaluation/` 开发纳入 Comet Native 流程并推进 Build → Verify → Archive。

# Verification expectations

- 单元测试：`uv run pytest LinearProbe_evaluation/tests/ -q` → 59 passed。
- 真实端到端：stats / train（mlp 与 linear 两结构）/ evaluate / WebUI 冒烟（HTTP 200）。
- GPU 证据：`diagnose_cpu.py` 显示训练阶段 CPU 占用率 ≈ 1 核、GPU kernel 耗时与 wall 接近。
- 归档前确认所有验收项均以真实命令结果作为证据（不把未运行的检查写成通过）。
