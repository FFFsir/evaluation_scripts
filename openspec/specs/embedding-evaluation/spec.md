# embedding-evaluation Specification

## Purpose
对 Qdrant 中存储的像素 embedding 向量进行系统的质量评估，输出三类指标（KNN 分类准确率、邻居纯度/召回、类内/类间距离分布），支持 CLI 和 WebUI 两种使用方式。默认使用 PyTorch CUDA 做 GPU 加速批量精确 KNN 检索。
## Requirements
### Requirement: 分层随机采样

系统 SHALL 从 Qdrant Collection 中按标签（label ∈ [0..8]）分层随机采样像素，每类采样数可配置。采样像素 MUST 包含向量、标签、标签名、影像 ID、像素行列和点 ID。采样前 SHALL 检查向量索引状态：当 `indexed_vectors_count < total_points`（HNSW 索引构建未完成）时，明确提示索引构建中及已索引比例，不静默等待或超时失败。

#### Scenario: 正常采样

- **WHEN** 调用 `sample_queries_by_label(manager, samples_per_class=500, seed=42)`，Collection 中各类像素数均 ≥ 500
- **THEN** 返回 4500 个采样像素，每类恰好 500 个，且可复现

#### Scenario: 某类像素不足

- **WHEN** 某类标签（如 label=6 "built"）只有 120 个像素，但 `samples_per_class=500`
- **THEN** 该类只采样 120 个，结果中记录 actual_count=120，其他类仍采样 500 个

#### Scenario: Collection 为空

- **WHEN** Collection 中无任何像素（total_points=0）
- **THEN** 抛出明确异常，提示 Collection 为空

#### Scenario: 索引未就绪预警

- **WHEN** Collection 的 `indexed_vectors_count` 小于 `total_points`（向量索引构建未完成）
- **THEN** 采样步骤明确提示「向量索引构建中（已索引 X / 总点数 Y）」并继续采样，不再因客户端超时静默失败；索引未就绪的提示优先于采样结果返回

### Requirement: KNN 分类准确率评估（F1）
系统 SHALL 使用 Leave-One-Out 策略评估 KNN 分类准确率：对每个查询像素检索 K+1 个最近邻，剔除查询像素自身后，取前 K 个有效邻居按多数投票预测标签。

#### Scenario: 正常分类
- **WHEN** 对 4500 个查询像素执行 `compute_knn_accuracy(manager, queries, k=10)`（exact=True）
- **THEN** 返回包含 overall_accuracy、per_class_metrics（每个类别的 precision/recall/f1/support）、confusion_matrix（9×9）、有效的 K 值、查询总数和耗时的字典

#### Scenario: 平票处理
- **WHEN** Top-K 有效邻居中出现平票（如 K=10 时 5 票 vs 5 票）
- **THEN** 逐次减小 K（K-1, K-3, …）重新计票直到打破平局；若最终至 K=1 仍平票（不可能），则按余弦距离最近邻居的标签预测

### Requirement: 邻居纯度与 Recall@K 曲线（F2）
系统 SHALL 计算不同 K 值下的邻居纯度（Purity@K）和召回率（Recall@K）。Purity(K) = (1/N) × Σᵢ (Kᵢ_same / K)。Recall@K = (1/N) × Σᵢ (Kᵢ_same / N_same_class(i))，其中 N_same_class(i) 为查询像素同类别在全集中的总数。

#### Scenario: 单次检索优化
- **WHEN** 计算多个 K 值的 Purity/Recall（如 K∈[10,20,50,100,300,1000]）
- **THEN** 对每个查询像素仅执行一次 `search(k=max(k_values)+1, exact=True)`，从结果中递增取不同 K 值计算，而非多次独立检索

#### Scenario: Leave-One-Out 剔除
- **WHEN** 查询像素自身出现在搜索结果中
- **THEN** 按 point_id 剔除自身后计入统计，Purity(K=10) 不恒为 1.0

#### Scenario: Recall@K 分母
- **WHEN** 计算 Recall@K
- **THEN** 分母使用全量同类像素总数（通过 `QdrantManager.client.count()` 按 label 过滤获取），而非 `min(K, N_same_class)`

### Requirement: GPU 批量精确 KNN 检索
系统 SHALL 默认使用 PyTorch CUDA 做分块批量精确 KNN 检索（GPU 分块矩阵乘 + CPU 多线程聚合），替代逐条 Qdrant exact 检索与 numpy 整体物化路径。GPU 不可用时自动回退 torch CPU 分块路径（同一 `KnnEngine`，device="cpu"）。

#### Scenario: GPU 正常路径
- **WHEN** `torch.cuda.is_available()` 返回 True 且 `device="cuda"`（默认）
- **THEN** 从 Qdrant 分片 scroll 全量向量并以 float32 常驻 GPU 显存，查询分块在 GPU 上做矩阵乘 + `torch.topk` 取精确 Top-K，逐块回传 CPU 由线程池并行聚合，完成 LOO + 多数投票 + Purity/Recall

#### Scenario: GPU 不可用自动回退
- **WHEN** `torch.cuda.is_available()` 返回 False 且用户未显式指定 device
- **THEN** 输出警告信息，自动回退 torch CPU 分块路径（同一 `KnnEngine` 在 cpu 上运行），结果与 GPU 路径一致

#### Scenario: 显式 device 指定
- **WHEN** 用户通过 CLI/WebUI 指定 `device=cuda` 或 `device=cpu`
- **THEN** 系统按指定设备执行；指定 `cuda` 但 CUDA 不可用时输出明确错误而非静默回退

#### Scenario: 分块计算与显存预算
- **WHEN** 全量向量与相似度矩阵占用超过显存预算 (`max_gpu_mem`)
- **THEN** 系统按显存预算自动推导查询分块大小，每块单独 GPU matmul + topk，逐块回传 CPU 聚合，跨块结果与全量 GPU 计算一致，峰值显存 < 预算值

<!-- 主 spec 的旧 scenario：`--no-gpu 强制回退` 与 `分块计算` 已被本 change 取代（--no-gpu/gpu_memory_gb 参数移除，分块语义精确化）。OpenSpec 合并需显式覆盖它们，避免丢弃。 -->
#### Scenario: --no-gpu 强制回退
- **WHEN** 用户指定 `--no-gpu` 参数
- **THEN** 该参数已移除，由 `--device cuda|cpu|auto` 取代；不再支持 `--no-gpu`

#### Scenario: 分块计算
- **WHEN** 全量向量占用超过 GPU 显存预算 (`gpu_memory_gb`)
- **THEN** 由 `分块计算与显存预算` 精确化；`gpu_memory_gb` 参数已改名 `max_gpu_mem`，分块大小按预算自动推导

### Requirement: CLI evaluate 子命令

系统 SHALL 在 CLI 中新增 `evaluate` 子命令，支持 `--samples-per-class`、`--k-f1`、`--k-values`、`--device`、`--gpu-batch-q`、`--max-gpu-mem`、`--seed`、`--output`、`--plot`、`--plot-dir`、`--qdrant-url` 等参数。

#### Scenario: 基础评估

- **WHEN** 运行 `python -m KNN_evaluation.cli evaluate --samples-per-class 200 --k-values 10,50,100`
- **THEN** 默认使用 GPU 加速（如可用），输出 F1 和 F2 的文本格式结果，退出码为 0

#### Scenario: GPU 不可用回退

- **WHEN** 运行 evaluate 且 CUDA 不可用
- **THEN** 输出警告 "CUDA not available, falling back to CPU"，自动使用 torch CPU 分块路径

#### Scenario: GPU 参数指定

- **WHEN** 运行 `evaluate --device cuda --gpu-batch-q 3000 --max-gpu-mem 16`
- **THEN** 使用指定的查询块大小与显存预算执行 GPU 分块 KNN，峰值显存不超预算

#### Scenario: JSON 导出

- **WHEN** 运行 `python -m KNN_evaluation.cli evaluate --output result.json`
- **THEN** 生成结构化的 JSON 文件，包含所有指标数据

#### Scenario: 图表生成

- **WHEN** 运行 `python -m KNN_evaluation.cli evaluate --plot --plot-dir ./eval_plots/`
- **THEN** 在指定目录生成混淆矩阵 PNG、Purity / Per-class Recall 曲线 PNG 和距离分布直方图 PNG

#### Scenario: Qdrant 不可达

- **WHEN** Qdrant 服务未启动或不可达
- **THEN** 输出明确错误信息并返回非零退出码

### Requirement: WebUI 评估面板

系统 SHALL 在 WebUI 中提供评估面板 expansion，含 GPU 参数配置（device/samples_per_class/k_values/max_gpu_mem）、异步执行（不阻塞事件循环）、进度反馈（采样→显存驻留→逐块 KNN→聚合→出图）和结果展示。状态栏 SHALL 显示向量索引进度（已索引向量数 / 总点数），索引未就绪时以警告提示。

#### Scenario: 参数配置与触发

- **WHEN** 用户在评估面板中设置 device、每类采样数、K 值序列、显存预算并点击"开始评估"
- **THEN** 系统异步执行评估流程，显示进度文本（"正在加载全量向量到 GPU..."、"正在计算 KNN Accuracy (block 12/300)..."等）

#### Scenario: 结果展示

- **WHEN** 评估完成后
- **THEN** 展示 F1 指标表格（Overall Accuracy + Per-class Precision/Recall/F1）、F2 Purity/Recall 交互式折线图

#### Scenario: JSON 导出

- **WHEN** 用户点击"导出 JSON"按钮
- **THEN** 在 `outputs/evaluation/knn_eval` 生成文件名 `{集合缩写}_knn_result_{时间戳}.json`（集合缩写 `google`/`xian`），含完整评估结果，不再走浏览器下载

#### Scenario: 图片图表导出

- **WHEN** 用户点击"导出图片图表"按钮
- **THEN** 在 `outputs/evaluation/knn_eval` 生成 `{集合缩写}_knn_cm_{时间戳}.png`（混淆矩阵）与 `{集合缩写}_knn_pr_{时间戳}.png`（Purity & Per-class Recall 曲线，右面板为 Per-class Recall，与 Web 页面一致）

#### Scenario: 状态栏显示索引进度

- **WHEN** 状态栏刷新且 Collection 存在
- **THEN** 显示 `已索引向量数 / 总点数` 索引进度；`indexed_vectors_count < points_count` 时以警告样式提示"向量索引构建中"

#### Scenario: 一键构建向量索引

- **WHEN** 用户在「Qdrant 连接 & Collection 状态」面板点击"构建向量索引"按钮且 Collection 存在
- **THEN** 系统触发全量 HNSW 向量索引重建（`indexing_threshold=0`），提示"向量索引重建已触发，Qdrant 后台构建中"，并刷新状态显示索引进度；Collection 不存在时提示先创建或导入

### Requirement: 评估前预估告警
系统 SHALL 在 CLI/WebUI 评估前打印/显示预估内存与查询量提示，帮助用户判断资源占用。

#### Scenario: CLI 评估打印预估
- **WHEN** 运行 `evaluate` 子命令
- **THEN** 评估前打印预估内存与查询像素数提示

#### Scenario: WebUI 评估面板显示预估
- **WHEN** 用户在 WebUI 评估面板配置参数
- **THEN** 显示基于当前参数的预估内存提示

