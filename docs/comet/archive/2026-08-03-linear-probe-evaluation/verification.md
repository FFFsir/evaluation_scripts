# Acceptance evidence

<!-- comet-native:acceptance-evidence:start -->
[
  {
    "acceptance_id": "acceptance-28b2ecb4447276abdd7ff4e1e17064d14d9de3ff51d165b3e36e491345a8e0ae",
    "evidence_refs": [
      "LinearProbe_evaluation/diagnose_cpu.py"
    ]
  },
  {
    "acceptance_id": "acceptance-50e0396d7e5262f74760bfaad7f59402ba1ac724fc03284947b99b42982e892f",
    "evidence_refs": [
      "LinearProbe_evaluation/cli.py",
      "LinearProbe_evaluation/dataset.py"
    ]
  },
  {
    "acceptance_id": "acceptance-a64795acf54d57971f6806cc2afb23fa55f81daee51fb931c51a53d2c5ced4e1",
    "evidence_refs": [
      "LinearProbe_evaluation/trainer.py"
    ]
  },
  {
    "acceptance_id": "acceptance-a8bdba7bb56921639fb0b53c695d80ca28988f0fd4f27f00c7e1576f5fb32604",
    "evidence_refs": [
      "LinearProbe_evaluation/tests/test_trainer.py",
      "LinearProbe_evaluation/tests/test_webui.py"
    ]
  },
  {
    "acceptance_id": "acceptance-d38bf59952499db8b530d02f628f73cd816b7048bb9dbe70f471668a6fba32f3",
    "evidence_refs": [
      "LinearProbe_evaluation/cli.py",
      "LinearProbe_evaluation/qdrant_client.py"
    ]
  },
  {
    "acceptance_id": "acceptance-d5990e5f3a77dd1f8ed2b1d137e0ebbb2258b1a5e491986b080dad96ff10056b",
    "evidence_refs": [
      "LinearProbe_evaluation/webui.py"
    ]
  },
  {
    "acceptance_id": "acceptance-e1f2cb57cc160a29d2ec0ecd904be3c9d1f363bb29f473795786e6b89977a6ff",
    "evidence_refs": [
      "LinearProbe_evaluation/cli.py",
      "LinearProbe_evaluation/model.py",
      "LinearProbe_evaluation/trainer.py"
    ]
  }
]
<!-- comet-native:acceptance-evidence:end -->

# Commands and results

- `uv run pytest LinearProbe_evaluation/tests/ -q` → `59 passed in 4.93s`（覆盖模型双结构、指标手算对照、分层采样/稀有类、训练循环/checkpoint 往返/取消、WebUI 全流程、设备策略；内存 Fake Qdrant，无需真实服务）。
- `uv run python -m LinearProbe_evaluation.cli stats` → `Collection: pixel_embeddings | total_points=10223616 | status=green`，每类像素数（water 533687 / trees 6209827 / grass 74887 / flooded_vegetation 4319 / crops 51816 / shrub_and_scrub 5362 / built 3332335 / bare 11143 / snow_and_ice 240）与 Qdrant count API 一致。
- `uv run python -m LinearProbe_evaluation.cli train --epochs 10 --train-per-class 2000 --val-per-class 500 --batch-size 2048 --lr 0.01 --model mlp --output-dir outputs/mlp_label_final` → 采样 16192 训练 + 4048 验证 → CUDA 训练完成，`Best epoch: 10 | val_acc=0.8439 | val_macro_f1=0.8593`，checkpoint/图/报告落盘；`--model linear` 同样训练成功（585 参数单层）。
- `uv run python -m LinearProbe_evaluation.cli evaluate --checkpoint outputs/mlp_label_smoke/checkpoints/mlp_label_best.pt`（默认 --samples-per-class 1000）→ 分层采样 8240 样本（8 类×1000 + snow_and_ice 240），秒级完成，exit=0，不触发全库下载。
- `uv run python LinearProbe_evaluation/webui.py --port 8006` + HTTP 请求 → `HTTP 200 | title: Qdrant Linear Probe 评估系统 — MLP_label`，页面含「模型结构」下拉（MLP（LeNet / Linear Probe））。
- `uv run python LinearProbe_evaluation/diagnose_cpu.py` → `[阶段2 CUDA训练] wall=1.51s | cpu=1.47s | CPU占用率=0.98 核`（训练阶段 CPU 约 1 核，非多核飙升；GPU kernel 1.51s 与 wall 接近）。
- CUDA 环境：`nvidia-smi` → `NVIDIA GeForce RTX 4090, 24564 MiB, driver 610.88`；`torch 2.13.0+cu126 | cuda available: True`。
- 设备策略验证：`resolve_device("cpu")` 抛 `ValueError`（CPU fallback 已禁用），测试 `test_resolve_device` / `test_resolve_device_cuda_unavailable` 覆盖。

# Skipped checks

- 无跳过项；所有 Acceptance examples 均已通过真实命令验证。

# Spec consistency

- 网络结构（spec「网络」节）：`hidden_dims=()` 单层 585 参数 / `(256,256,256)` 多层 150,537 参数，`build_model` 工厂与 CLI `--model`、WebUI「模型结构」下拉一致；checkpoint 记录 `hidden_dims` 且加载正确重建（`[]` 不误判为多层），`test_checkpoint_roundtrip` / `test_linear_variant_training` 覆盖。
- 设备约束（spec「训练」节）：`resolve_device` 只接受 cuda/auto，`cpu` 抛 `ValueError`，CUDA 不可用抛 `RuntimeError`；训练/评估数据与模型 `assert .is_cuda`；训练循环全 GPU（数据驻留、GPU randperm/gather、GPU loss）。
- 数据访问（spec「数据访问」节）：采样/训练/评估只经采样地图分层采样分批 retrieve，不全库下载；`stratified_train_val_split` 稀有类保证验证集非空（`test_rare_class_keeps_val`）。
- 指标（spec「评估与指标」节）：accuracy / macro-F1 / weighted-F1 / per-class / 混淆矩阵与手算对照一致（`test_metrics.py`）。
- WebUI（spec「WebUI」节）：进度阶段文本、实时曲线、中止、并发保护均有测试覆盖（`test_webui.py`）。

# Known limitations and risks

- WebUI 冒烟仅验证页面可打开与关键控件渲染（HTTP 200），未在浏览器中完成一次完整训练交互（需要真实浏览器会话）；训练逻辑本身经 FakeUI 驱动的 do_train 全流程测试覆盖。
- GPU 可复现性：`randperm` 使用显式 seed 的 CUDA generator，同机可复现；跨 GPU 型号/驱动不做复现保证（PyTorch CUDA 通用限制）。
- 采样阶段（读约 480MB 采样地图 JSON + 组装 numpy）为单核 CPU 密集，与训练设备无关；已通过 `diagnose_cpu.py` 量化（阶段1 约 7s、0.9 核）。
- 训练循环假设数据量在单机显存可容纳（训练集 MB 级）；超大采样（`--train-per-class 0` 全量）时由 `_evaluate` 分块 forward 限制激活显存，但训练数据一次性驻留 GPU 的显存占用需用户按需控制采样规模。

# Conclusion

- 7/7 Acceptance examples 全部通过真实命令验证；59 项单元测试全绿；真实 Qdrant 端到端（stats / train mlp+linear / evaluate / WebUI 冒烟 / CPU 诊断）全部符合规格。
- 验证结果：**pass**。
