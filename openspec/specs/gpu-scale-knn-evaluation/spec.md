# gpu-scale-knn-evaluation Specification

## Purpose
在 16GB 内存 / 20GB 显存约束下对 Qdrant 中存储的全量像素 embedding 做大规模精确 KNN 统计分析：全库向量常驻显存、查询分块矩阵乘取 Top-K、CPU 多线程聚合指标，支持 90 万查询 × top-10001 的量级，输出 KNN 分类准确率（F1）、邻居纯度/召回（F2）与距离分布（F3）三类指标。
## Requirements
### Requirement: GPU 分块矩阵乘精确 KNN
系统 SHALL 将全库向量以 float32 常驻 GPU 显存，查询向量分块在 GPU 上计算 `query_block @ corpus.T` 相似度矩阵并 `torch.topk` 取精确 Top-K，逐块回传 CPU 后即时丢弃，不物化全部查询的结果矩阵。

#### Scenario: 全库向量驻留显存
- **WHEN** 用户请求大规模评估且 CUDA 可用
- **THEN** 系统从 Qdrant 分片 scroll 全量向量并转为 float32 常驻 GPU 显存（N=1,000,000 时约 230MB），供所有查询分块复用

#### Scenario: 分块取 Top-K
- **WHEN** 查询像素总量为 Q、目标 K 为 `max_k+1`（含 Leave-One-Out 剔除位）
- **THEN** 系统把查询向量按块（每块 `block_q` 条）在 GPU 上计算精确 Top-K，每块结果回传 CPU 后即时聚合并丢弃，全程不物化 (Q, N) 相似度矩阵或 (Q, K) 结果矩阵

#### Scenario: 显存预算受控
- **WHEN** GPU 可用显存上限为 `max_gpu_mem`（GB）
- **THEN** 系统按预算自动推导查询块大小 `block_q`，使单块相似度矩阵（block_q × N × 4B）与 corpus 驻留占用合计不超过预算，峰值显存 < `max_gpu_mem`

### Requirement: CPU 多线程聚合
系统 SHALL 将 GPU 回传的 Top-K 标签/ID 结果在 CPU 上多线程并行聚合，聚合器（混淆矩阵、Purity/Recall 累加器、类计数）不持有全量 Top-K，峰值内存恒定。

#### Scenario: F1 聚合
- **WHEN** 处理一块 Top-K 标签矩阵
- **THEN** 按行执行 Leave-One-Out 剔除自身 + 多数投票（平票时递减 K 重计），累加到 9×9 混淆矩阵，内存不随查询总量增长

#### Scenario: F2 聚合
- **WHEN** 处理一块 Top-K 结果
- **THEN** 对每个 K 值递增取邻居子集累加 Purity/Recall 累加器，不重复检索；Recall@K 分母使用 Qdrant count 的全量同类总数

#### Scenario: 峰值内存受控
- **WHEN** 90 万查询 × top-10001 全流程运行
- **THEN** 客户端峰值内存 < 4GB，只保留当前块结果与累加器

### Requirement: CUDA 不可用回退
系统 SHALL 在 CUDA 不可用、显存不足或数据量过小时自动回退 torch CPU 分块路径（同一 `KnnEngine` 在 device="cpu" 运行），功能不中断，结果一致。

#### Scenario: CUDA 不可用自动回退
- **WHEN** `torch.cuda.is_available()` 返回 False 且用户未显式指定 device
- **THEN** 输出警告并自动回退 torch CPU 分块路径，采样、指标返回结构与 GPU 路径一致

#### Scenario: 显式指定 device
- **WHEN** 用户通过 `--device cuda|cpu` 显式指定计算设备
- **THEN** 系统按指定设备执行；显式 `--device cuda` 但 CUDA 不可用时输出明确错误

### Requirement: 多 K 值零额外检索成本
系统 SHALL 对每个查询像素仅计算一次 `max_k+1` 的精确 Top-K，从同一结果中递增取不同 K 值计算 F1 分类准确率与 Purity/Recall 曲线，避免重复检索。

#### Scenario: 单次检索多 K 统计
- **WHEN** K 值序列为 [10,100,300,1000,3000,10000]
- **THEN** 每个查询只检索一次 top-10001，F1 各 K 多数投票与 Purity/Recall 各 K 均从该结果递增取值，无额外检索开销

### Requirement: F1/F2 联合评估共享检索（verify 缺陷修复）
系统 SHALL 提供联合评估函数 `evaluate_knn`，对同一查询集单次全量 scroll + 单次 top-(max_k+1)，同时聚合 F1（KNN 分类准确率多 K + 混淆矩阵）与 F2（Purity/Recall 曲线），F1/F2 共享同一份 Top-K 结果，零重复下载、零重复检索；CLI/WebUI 使用该函数替代对两个独立函数的顺序调用。`KnnEngine.close()` SHALL 释放 PyTorch GPU 缓存（`torch.cuda.empty_cache()`），确保评估完成后显存回落。

#### Scenario: 联合评估单次检索
- **WHEN** 用户在 WebUI 设置每类采样 5000、k_f1=100、k_values=[10,20,50,100,300,1000] 并触发评估
- **THEN** 系统单次 scroll 全量向量 + 单次 top-1001 计算（max_k=1000），F1 各 K 与 F2 各 K 全部从该结果聚合，不重复 scroll、不重复 topk

#### Scenario: 联合结果与独立调用一致
- **WHEN** 同一查询集分别用 `evaluate_knn` 与 `compute_knn_accuracy`+`compute_purity_recall_curve` 计算
- **THEN** F1/F2 结果完全一致

#### Scenario: 显存释放
- **WHEN** 评估完成后 engine 关闭
- **THEN** 调用 `torch.cuda.empty_cache()` 释放 PyTorch GPU 缓存，显存回落到评估前水平，后续 F2/其他任务不会叠加未释放显存导致 OOM

