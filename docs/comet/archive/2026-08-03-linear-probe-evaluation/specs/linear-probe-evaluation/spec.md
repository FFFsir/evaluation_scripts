# linear-probe-evaluation

基于 Qdrant `pixel_embeddings` Collection 的像素级 MLP_label 线性分类评估系统（`LinearProbe_evaluation/`）。

## 数据访问

- 只读 Qdrant `pixel_embeddings` Collection（约 1022 万点，每点 64 维 embedding + payload `label` 0-8）。
- **绝不全库下载**：训练/评估数据均通过 KNN 采样地图（`qdrant_sampling_map.json`）按标签分层随机选取 point_id，再按 ID 分批 `retrieve`（`batch_size=5000`）只下载样本量（MB 级）。
- 分层采样语义（每类独立）：组内按 seed 打乱 → 先按 `val_ratio`（默认 0.2）切出验证集并受 `val_per_class` 上限约束（`val_ratio>0` 且该类有点时至少 1 个进验证集，保证稀有类验证集非空）→ 剩余部分取前 `train_per_class` 个进训练集（`<=0` 表示不限制）。
- `sample_dataset(samples_per_class)`：评估用独立采样，每类最多 N 个（`<=0` 全取）。

## 网络

- `MLPLabel(in_features=64, num_classes=9, hidden_dims, dropout=0.2)`：
  - `hidden_dims=()`：单层 `Linear(64,9)`，585 参数（标准 Linear Probe，检测 embedding 线性可分性）。
  - `hidden_dims=(256,256,256)`（默认）：`Linear(64,256)→ReLU→Dropout(0.2)→Linear(256,256)→ReLU→Dropout(0.2)→Linear(256,256)→ReLU→Dropout(0.2)→Linear(256,9)`，150,537 参数（LeNet-5 规模）。
- 输出 9 维 logits，对应 9 个 DW 标签硬分类独热码语义；损失 CrossEntropyLoss（真实标签 0-8 直接 class index）；预测取 `argmax`。
- `build_model(variant)` 工厂：`variant ∈ {"mlp", "linear"}`，未知值抛 `ValueError`。
- 后续 `MLPProb`（软分类概率网络）不在本规格范围内。

## 训练

- `train_mlp(train_ds, val_ds, config, progress_callback, cancel_event)`：纯内存训练，不接收 Qdrant 连接。
- 超参：epochs / batch_size / lr / weight_decay / optimizer（adam|sgd）/ model_variant（mlp|linear）/ seed / log_interval。
- **设备固定 CUDA**：`resolve_device` 只接受 `cuda`/`auto`/None（均解析为 `cuda`）；`cpu` 与未知值抛 `ValueError`；CUDA 不可用抛 `RuntimeError`。训练/评估数据与模型带 `assert .is_cuda` 防御断言。
- 全 GPU 训练循环：数据一次性驻留 GPU；每 epoch 用 GPU `torch.randperm`（显式 seed 的 CUDA generator）shuffle；batch 用 GPU gather 索引；loss/backward/step 全在 GPU；`_evaluate` 分块 forward（chunk 8192）在 GPU 累积 logits 后一次计算 loss 与 argmax，仅最终 pred 一次性 `.cpu()`。
- 进度回调事件：`epoch_start` / `batch`（`(batch_idx+1) % log_interval == 0` 触发，字段 epoch/total_epochs/batch/total_batches/loss）/ `epoch_end`（train_loss/train_acc/val_loss/val_acc/val_macro_f1；空验证集时 val_* 为 None，消费端需 None 容错）。
- 协作式取消：每 epoch 开始与每 batch 检查 `threading.Event`，置位抛 `CancelledError`。
- checkpoint：按 val_acc 保存 best、结束保存 final（state_dict + architecture{hidden_dims, dropout} + 元数据）；加载按 `hidden_dims` 重建（`[]` 不能误判为默认多层）；`torch.load(weights_only=True)`。
- 元数据 JSON：config + architecture（variant/hidden_dims/num_parameters）+ history + 分类报告 + device/device_name/gpu_kernel_seconds。
- 分类报告用 best checkpoint 权重计算（与 best 指标一致）。

## 评估与指标

- `evaluate`：加载 checkpoint，默认按采样地图分层采样（`--samples-per-class 1000`，只下载样本量）；`--samples-per-class 0` 才显式全量（耗时长，文档标注不推荐）。
- 指标：accuracy、macro-F1、weighted-F1、per-class precision/recall/F1/support、混淆矩阵；稀有类分母为 0 时记 0（不除零）。

## CLI

`uv run python -m LinearProbe_evaluation.cli <command>`

- `train`：--epochs --batch-size --lr --weight-decay --optimizer --model{mlp|linear} --train-per-class --val-per-class --val-ratio --seed --log-interval --output-dir --plot-dir --output --verbose --qdrant-url。训练前打印模型结构；完成输出 GPU 设备名与 GPU kernel 耗时。
- `evaluate`：--checkpoint --samples-per-class --max-points --seed --plot --plot-dir --output --qdrant-url。
- `stats`：--json --qdrant-url（总点数 + 每类像素数，label count API）。

## WebUI

`uv run python LinearProbe_evaluation/webui.py --port 8004`

- Qdrant 连接 & Collection 状态区；训练参数区（epochs/batch/lr/优化器/**模型结构下拉**/每类训练·验证样本数/验证比例/seed；训练设备固定 CUDA 并显示 GPU 设备名）；进度区（进度条 + 阶段文本「1/2 分层采样(CPU 密集) → 2/2 CUDA 训练」）；逐 epoch 实时曲线（matplotlib → base64 → ui.image）；结果区（best 指标、per-class 表、验证集混淆矩阵、导出 JSON 含 architecture）。
- 「中止训练」协作式取消；并发训练保护（`state["training"]`，进行中再次点击返回 warning）。

## 可视化

- `plot_training_curves`：loss / accuracy+macro-F1 双面板曲线；`plot_confusion_matrix`：混淆矩阵热力图（matplotlib Agg + 中文字体，与 KNN 风格一致）；base64 data URI 供 WebUI。

## 测试

- `tests/` 59 项，内存 Fake Qdrant（`tests/helpers.py`），无需真实服务；覆盖模型双结构/线性性、指标手算对照、分层采样（含稀有类）、训练循环/checkpoint 往返/取消、WebUI 全流程（FakeUI 驱动）、设备策略。
