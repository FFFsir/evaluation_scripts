---
comet_change: gpu-knn-scale-evaluation
role: technical-design
canonical_spec: openspec
archived-with: 2026-08-03-gpu-knn-scale-evaluation
status: final
---

# GPU 分块 KNN 大规模评估 — 技术设计

## 1. 背景与目标

现有 `KNN_evaluation/metrics.py` 的批量精确 KNN 存在硬性内存上限：

- `_batch_exact_knn`（numpy CPU 批量）把全部 Top-K 结果物化为 `(Q, K_eff)` 的 `idxs`/`scores`/`labels` 三个数组。目标规模 Q=900,000、K_eff=10,001 时各 ≈72GB，16GB RAM 不可行。
- `_scroll_full_vectors` 以 float64 全量载入客户端内存，N 越大越不可行。
- 目标：16GB RAM / 20GB VRAM 约束下，对全量数据完成 top-K=[10,100,300,1000,3000,10000] 的精确 KNN 统计分析（9 类各采样 10 万查询，共 90 万查询）。

本设计用 **GPU 分块矩阵乘 + CPU 多线程聚合** 重建批量精确 KNN，替代 numpy 整体物化路径，并作为默认路径。

## 2. 范围（Goals / Non-Goals）

**Goals:**
- GPU 分块矩阵乘核心：全库向量 float32 常驻 GPU 显存，查询分块 `query_block @ corpus.T` + `torch.topk`，边算边回传 CPU，不物化 `(Q, N)` 或 `(Q, K)` 全量结果。
- 统一 device 抽象：同一 `KnnEngine` 在 cuda 或 cpu 上运行；`device="auto"` 默认 CUDA 可用则 GPU，否则 torch CPU 分块。
- CPU 多线程聚合：F1 混淆矩阵 / F2 Purity/Recall 累加器，峰值内存恒定 < 4GB。
- 显存预算推导：按 `max_gpu_mem` 自动推导 `block_q`，峰值显存 < 预算。
- 多 K 零额外检索：单次 top-(max_k+1)，对 k_values 递增取值；F1 输出多 K 准确率（`accuracy_by_k`）。
- CUDA 不可用自动回退 torch CPU 分块路径，结果一致。
- CLI/WebUI 暴露 `--device`/`--gpu-batch-q`/`--max-gpu-mem` 参数。
- 移除 `use_batch` 参数与 `--batch` 标志（被 `--device` 取代），调用方同步更新。

**Non-Goals:**
- **F3 距离分布（Intra/Inter-class 距离统计）不在本 change 范围**：`compute_distance_distribution` 尚未实现、无 spec Requirement 支撑，留待后续 change。
- 不引入 faiss/annoy 等外部 ANN 索引库。
- 不做分布式/多机 KNN。
- 不改变 Collection schema、payload 结构、`searcher.py` 单条交互检索行为。
- 不支持带过滤的精确 KNN 批量评估（仅在全集上计算）。
- 不做导入侧优化（属另一 change `data-import-optimization`）。

## 3. 架构

### 3.1 总体数据流

```
queries (Q×64, CPU)
      │
      ▼
┌─── KnnEngine (corpus float32 常驻 device) ───┐
│  device = cuda | cpu (torch)                 │
│  for q_block in chunks(queries, block_q):    │
│    sim = q_block @ corpus.T   (分块 matmul)   │
│    topk → (scores, labels, ids) 回传 CPU      │
└──────────────┬────────────────────────────────┘
               ▼
CPU ThreadPoolExecutor 聚合（每块结果即时丢弃）
     ├─ F1: LOO + 多数投票(平票递减K) → 9×9 混淆矩阵
     │      + accuracy_by_k 多 K 准确率
     └─ F2: 按 k_values 递增 → Purity/Recall 累加器
```

### 3.2 组件划分

**`gpu_knn.py`（新增）— `KnnEngine`**：device-agnostic 分块精确 KNN 核心。
- `__init__(device="auto")`：延迟 import torch；解析 device（cuda/cpu/auto）。
- `set_corpus(all_vectors: np.ndarray)`: float32 (N,64) 常驻 device，L2 归一化。
- `knn_chunk(query_block: np.ndarray, k: int) -> tuple`: 单块 `q_block @ corpus.T` + `torch.topk`，返回 CPU (scores, labels, ids)。
- `estimate_block_q(max_mem_gb)`: 显存预算 → block_q。
- `close()`: 释放 device 内存。

**`metrics.py`（修改）— 批量聚合路径**：
- `_scroll_full_vectors`：保留分页 scroll 语义，改为返回 float32 向量（供 GPU/cpu 常驻）。
- 新增 GPU/CPU 分块聚合逻辑，`compute_knn_accuracy` / `compute_purity_recall_curve` 内部使用 `KnnEngine`。
- `use_batch` 参数移除，新增 `device="auto"`、`gpu_batch_q=None`、`max_gpu_mem=16`、`max_eval_ram=6.0`。

**`cli.py`（修改）— evaluate 子命令**：
- 移除 `--batch`/`--use-batch`；新增 `--device cuda|cpu|auto`（默认 auto）、`--gpu-batch-q`、`--max-gpu-mem`（默认 16）、`--max-eval-ram`（默认 6，CPU 回退 RAM 预算）。
- 输出 F1 多 K 表（`accuracy_by_k`）与 F2 曲线，JSON 导出结构兼容。

**`webui.py`（修改）— 评估面板**：
- 批量 checkbox → device 选择器（cuda/cpu/auto）+ max_gpu_mem 输入。
- 进度反馈覆盖：采样 → 显存驻留 → 逐块 KNN → 聚合 → 出图。

**`pyproject.toml`**：新增 `torch` 依赖。

## 4. 详细设计

### D1: KnnEngine 核心（device-agnostic 分块）

**类 `KnnEngine`**：

```python
class KnnEngine:
    def __init__(self, device="auto"):
        import torch  # 延迟 import
        self.device = resolve_device(device)  # "cuda" | "cpu"
    def set_corpus(self, all_vectors: np.ndarray):
        # float32 (N,64) 常驻 device，L2 归一化
        self.corpus = torch.from_numpy(all_vectors.astype(np.float32)).to(self.device)
        self.corpus /= self.corpus.norm(dim=1, keepdim=True).clamp(min=1e-12)
    def knn_chunk(self, query_block: np.ndarray, k: int) -> tuple:
        q = torch.from_numpy(query_block.astype(np.float32)).to(self.device)
        q /= q.norm(dim=1, keepdim=True).clamp(min=1e-12)
        sim = q @ self.corpus.T                    # (block_q, N) float32
        scores, idx = torch.topk(sim, k=min(k, N), dim=1)
        return scores.cpu().numpy(), self.labels[idx].cpu().numpy(), self.ids[idx]  # ids 为 numpy str 索引
```

- `labels`/`ids` 预取为 CPU numpy（`all_labels`, `all_point_ids`），`knn_chunk` 用 `self.labels[idx]` 花式索引取回，避免 GPU 到 CPU 传输标签数组。
- L2 归一化语义与 `_batch_exact_knn` 一致（cosine 相似度）。
- **备选 B（双引擎：GPU torch + numpy CPU 回退）弃**：双代码路径维护成本高，且 numpy CPU 整体物化仍在 90 万查询下不可行。torch CPU 分块在 N 大时 matmul 有线程优化且与 GPU 共享代码。
- **备选 C（旧形态：查询驻留、corpus 分块迭代合并 topk）弃**：Q=90 万时 GPU matmul 次数多 N/chunk 倍，且需在 GPU 上持有 (Q, k) 累加器。

### D2: 显存/内存模型

**公式**（沿用 design D1，corpus 占用精确计）：

```
block_q = floor(0.8 × (max_gpu_mem_GB - corpus_GB) × 1e9 / (N × 4))
block_q = clamp(block_q, 1, Q)
corpus_GB = N × 64 × 4 / 1e9   # float32 (N,64)
```

- `gpu_batch_q` 显式给定时直接用作 block_q，跳过推导。
- CPU 回退同理：`max_eval_ram`（RAM 预算）代入公式，corpus_GB 计 float32 占用。
- 块大小推导结果在 CLI/WebUI 打印（device、corpus 大小、块大小、预估显存/内存）。

### D3: CPU 回退

- `device="auto"`：`torch.cuda.is_available()` True → cuda，否则 cpu + 警告。
- `device="cuda"` 但 CUDA 不可用 → 抛明确错误（不静默回退）。
- `device="cpu"` → torch CPU 分块（RAM 预算 `max_eval_ram`），结果与 GPU 一致。
- **不再存在 numpy 整体物化路径**（`_batch_exact_knn` 移除）。

### D4: metrics API 兼容与 use_batch 移除

**签名变更**：

```python
def compute_knn_accuracy(manager, queries, k=10, exact=True,
                         device="auto", gpu_batch_q=None, max_gpu_mem=16,
                         max_eval_ram=6.0, progress_callback=None) -> dict:
    # 返回结构保持: overall_accuracy, per_class_metrics, confusion_matrix,
    #               k, num_queries, elapsed_sec
    # 新增: accuracy_by_k: {k: float}  多 K 准确率

def compute_purity_recall_curve(manager, queries, k_values=None, exact=True,
                                device="auto", gpu_batch_q=None, max_gpu_mem=16,
                                max_eval_ram=6.0, progress_callback=None) -> dict:
    # 返回结构保持: k_values, global_purity, global_recall,
    #               per_class_purity, per_class_recall, num_queries, elapsed_sec
```

- `exact=True` → `KnnEngine` 分块（device 参数决定）。
- `exact=False` → 保留 PixelSearcher ANN 逐条（供 `--ann` 对比）。
- **`use_batch` 参数移除**：CLI/WebUI/tests 调用方同步更新（本 change 唯一破坏性变更）。

### D5: 多 K F1（accuracy_by_k）

- GPU 单次计算 top-(max_k+1)，max_k = max(k_values + [k_f1])。
- 聚合阶段对每个 K ∈ sorted_k_values 分别做多数投票（LOO + 平票递减 K），得到 `accuracy_by_k: {k: acc}`。
- `overall_accuracy` 仍 = k_f1 的单 K 准确率（既有调用方兼容）。
- 零额外检索：一次 top-(max_k+1) 覆盖全部 K。

### D6: CLI / WebUI

**CLI `evaluate` 参数**：
- `--device cuda|cpu|auto`（默认 auto）
- `--gpu-batch-q`（默认 None → 按预算推导）
- `--max-gpu-mem`（默认 16）
- `--max-eval-ram`（默认 6.0，CPU 回退 RAM 预算）
- 移除 `--batch`/`--use-batch`
- 打印 GPU 路径信息（device、corpus 大小、块大小、预估显存）
- 输出 F1 多 K 表 + F2 曲线，JSON 导出结构兼容

**WebUI 评估面板**：
- 批量 checkbox → device 选择器（cuda/cpu/auto）+ max_gpu_mem 输入
- 进度反馈：采样 → 显存驻留 → 逐块 KNN（当前块/总块数）→ 聚合 → 出图

### D7: 依赖

- `pyproject.toml` 新增 `torch`。
- CPU 回退路径延迟 import torch（`KnnEngine.__init__` 内），无 CUDA 环境启动不失败。
- `--device cpu` 时 torch 仍需要（torch CPU 分块）；如 torch 未安装且需 GPU，抛明确错误。

### D8: 联合评估与显存释放（verify 阶段缺陷修复，用户实测反馈）

**问题**：CLI/WebUI 顺序调用 `compute_knn_accuracy` 与 `compute_purity_recall_curve` 两个独立函数，导致：
- **两次全量 scroll**（每次下载 10.2M 点，各数分钟）
- **两次独立 topk**（F1 的 top-(max_k+1) 结果在函数内聚合后丢弃，F2 重算）
- **显存不释放**：`KnnEngine.close()` 只置 `corpus=None`，无 `torch.cuda.empty_cache()`，F2 重新 `set_corpus` 叠加未释放显存 → OOM 外溢

**方案：新增联合评估函数 `evaluate_knn`**（metrics.py）：

```python
def evaluate_knn(manager, queries, k_f1, k_values, exact=True, device="auto",
                 gpu_batch_q=None, max_gpu_mem=16, max_eval_ram=6.0,
                 progress_callback=None) -> dict:
    # 单次 scroll + 单次 top-(max_k+1)，max_k = max(k_values + [k_f1])
    # 同时聚合 F1（accuracy_by_k + 混淆矩阵）与 F2（Purity/Recall）
    # 返回 {"f1": {...}, "f2": {...}}，结构分别与 compute_knn_accuracy / compute_purity_recall_curve 兼容
```

- 内部一次 `_scroll_full_vectors` + 一次 `KnnEngine.set_corpus` + 一次 `engine.knn_chunk(query_block, max_k+1)`，F1/F2 复用同一份 Top-K 结果聚合。
- `KnnEngine.close()` 增加 `torch.cuda.empty_cache()`（仅 cuda 设备），释放 PyTorch GPU 缓存。
- CLI `cmd_evaluate` 与 WebUI `do_evaluate` 改用 `evaluate_knn`，单次调用同时得 F1/F2，移除顺序调用。
- 默认 `--k-f1` 提到 100（`evaluate_knn` 内部 `max_k` 天然覆盖 k_values 默认序列 [10,20,50,100,300,1000]）。
- `compute_knn_accuracy` / `compute_purity_recall_curve` 保留为独立 API（单次调用仍可用，供测试/单指标场景）。

**理由**：消除重复下载与重复 topk（实测 F1+F2 各数分钟 → 合并后单次），修复显存外溢；F1 多 K 与 F2 共享单次 top-1001 实现真正零额外检索。
**备选**：缓存层复用（侵入 metrics 内部状态，并发复杂）弃；仅局部修复（不解决重复耗时）弃。

### D9: 可用性改进（归档前用户反馈）

**问题**（用户实测反馈）：
1. WebUI K-F1 输入上限硬编码 100（`ui.number(max=100)`），无法设置到 1000
2. Web 评估运行后无法中止/关闭，线程持续占用内存/显存
3. 采样耗时过长（`sample_queries_by_label` 逐类 9 次带 filter 的 scroll，每次传回 15000×64 向量）

**方案**：
- **D9.1 K-F1 上限 1000**：`webui.py` 的 `kf1_input` `max=100` → `max=1000`，与 `evaluate_knn` 的 max_k=1000（top-1001）预算一致；CLI `--k-f1` 同步校验范围。
- **D9.2 评估可中止**：`do_evaluate` 引入取消令牌。方案：`asyncio.to_thread` 外维护 `threading.Event`（或 asyncio Future 取消）；「中止评估」按钮触发事件；`evaluate_knn` 的分块循环每块检查事件（`engine.knn_chunk` 前后），命中则抛出 `CancelledError`/`RuntimeError`，`engine.close()`（try/finally）确保显存释放，进度 UI 复位。CLI 侧 `Ctrl+C` 已由 KeyboardInterrupt 天然支持，不额外处理。
- **D9.3 采样提速**：`sample_queries_by_label` 改为**单次 scroll 全量（无 label filter）+ 客户端按 label 分组 + 每类 rng.sample**。一次 scroll 取全量（10.2M 点若一次取完内存大；改为分页 scroll 累积到客户端后分组采样）。替代逐类 9 次带 filter scroll。保留分层语义（每类 samples_per_class）与 seed 可复现（`random.Random(seed)` 每类独立采样）。
  - **权衡**：单次全量 scroll 客户端内存 = 全库向量（10.2M×64×4B ≈ 2.6GB），与逐类 scroll 相比传输总量相同但 RPC 次数从 9 次降到 1 次（分页 N/50000 次）。逐类 scroll 服务端过滤 + 只取 15000 条，单类数据小时更快；全量 scroll 适合全库均匀采样。保留分页（每批 50000）避免单次内存峰值。

**理由**：K-F1 1000 使 WebUI 与 top-1001 预算一致；中止机制释放长时运行的资源占用；采样提速减少用户等待。
**备选**：并行类 scroll（ThreadPoolExecutor 每类独立 scroll）弃——与逐类串行相比 RPC 总量相同，复杂度更高；单次全量 scroll + 分组采样最简。

### D10: 采样地图 manifest（归档前用户反馈深化）

**问题**（用户实测反馈）：Section 9.3 的单次全量 scroll 采样仍需下载全量向量（10.2M 点 × 64 维 ≈ 2.6GB），即使每类只需几千个样本。用户提出"地图式采样"：在数据库外维护一个 point→label 地图，采样时本地随机选 ID 再按 ID 精确取向量。

**方案**：
- **D10.1 采样地图 manifest**（`qdrant_sampling_map.json`）：`{"collection": str, "total_points": int, "updated_at": str, "by_label": {label_id: [point_id, ...]}}`。与现有 `qdrant_import_manifest.json` 同模式：可重建、原子写（tmp + os.replace）。
- **D10.2 地图构建** `build_sampling_map(manager)`：Qdrant scroll（`with_vectors=False, with_payload=["label"]`）按 label 分组只取 point_id，**不下载向量**。10.2M 点下载量从 2.6GB（全向量）降到 ~150MB（纯 ID+label，每点约 15B）。
- **D10.3 自动对账重建** `ensure_sampling_map(manager)`：比较 manifest 的 collection 名称 + `total_points` 与当前 collection 信息，不一致或文件缺失/损坏时自动重建。
- **D10.4 采样改用地图**：`sample_queries_by_label` 读地图 → 每类 `rng.sample` 选 point_id → `client.retrieve(ids)` 批量取向量（只下载样本量，如 9 类 × 5000 = 45000 个 ≈ 11MB）。保留分层语义、seed 可复现、空类标记、返回结构兼容。

**理由**：把"全量下载 + 抽样"改为"地图定位 + 精确取样本"，采样从下载 2.6GB 降到下载样本量（MB 级）；地图一次性构建、后续零全量扫描。与导入 manifest 同模式，可重建、不引入持久化风险。
**备选**：无缓存每次扫（每次全库 ID 仍慢）弃；复用导入 manifest（impoter 需存坐标，改动大）弃。

### D11: 全库向量磁盘缓存（归档前用户反馈深化）

**问题**（用户实测反馈）：`evaluate_knn` 每次评估都调用 `_scroll_full_vectors` 下载全库向量（10.2M 点 ≈ 2.6GB，约 205 批 scroll，3-4 分钟）。同一 collection 重复评估（不同采样/参数）时每次都重下。日志确认评估阶段连续 16MB × 0.8s scroll 请求。

**方案**：
- **D11.1 全库向量磁盘缓存**（`qdrant_corpus_cache/{collection}.npz`）：`np.savez` 存 `vectors (N,64) float32` + `labels (N,) int64` + `point_ids (N,) str`。与采样地图/导入 manifest 同模式：可重建、原子写（tmp + os.replace）。
- **D11.2 惰性构建**：`_scroll_full_vectors` 首次下载全量后写入缓存；后续调用先查缓存（collection 指纹一致则直接 `np.load` 加载，避免重复下载 2.6GB）。CLI/WebUI 无需新命令。
- **D11.3 指纹对账** `ensure_corpus_cache(manager)`：缓存文件存 collection 名称 + total_points 元数据（npz 的 `allow_pickle` 或单独 json 侧车），与当前 collection 不一致/缺失/损坏时重新下载构建。

**理由**：把"每次评估全量下载"改为"首次下载 + 后续加载"。NVMe 读 2.6GB npz 约 5-10s，对比每次 3-4 分钟下载，重复评估收益巨大。与采样地图（ID 地图）互补：地图管采样定位，缓存管评估全量。
**备选**：仅内存缓存（进程重启失效）弃；服务端查询（每查询 RPC，大库慢）弃。

## 5. 边界条件

- **空 Collection**：`_scroll_full_vectors` 返回空 → 回退 Qdrant 服务端逐条（现有语义保留）。
- **某类无像素**：采样跳过（`sample_queries_by_label` 现有行为）；聚合时该类 support=0，per-class 指标返回 0。
- **K > N**：`torch.topk(k=min(k, N))`，结果取全部 N；LOO 后 effective_labels 可能不足 k，按现有逻辑处理。
- **N 极小**：engine 仍分块（block_q clamp 到 Q），结果与全量一致。
- **显存 OOM**：`estimate_block_q` 保守 ×0.8 余量；预留 OOM 异常捕获后按更小块重试（CLI 提示用户调低 `--max-gpu-mem`）。

## 6. 测试策略

- **差分测试**（`test_gpu_knn.py`）：小数据（N=1000, Q=100, K=100）上 GPU 分块 vs torch CPU 分块 vs Qdrant 服务端逐条 三者 Top-K 标签/顺序完全一致。
- `estimate_block_q` 单测：不同 N / max_gpu_mem 下返回合理块大小。
- LOO 剔除：查询自身出现在 Top-K 时被正确剔除（单测）。
- 平票 `_resolve_tie` 递减 K 正确（小规模 mock 数据）。
- F1 多 K（`accuracy_by_k`）与单 K `overall_accuracy` 一致（mock 数据）。
- F2 Purity/Recall 累加正确，Recall@K 用全量同类总数作分母。
- CUDA 回退：mock `torch.cuda.is_available` 验证 auto 回退 / 显式 cuda 抛错。
- `device="cpu"` 显式：torch CPU 分块结果正确。
- 边界：某类无像素、空 Collection、K > N 行为明确。
- `evaluate_knn` 与独立调用 `compute_knn_accuracy`+`compute_purity_recall_curve` 结果一致（差分）。
- `close()` 后显存释放（mock `torch.cuda.empty_cache` 断言调用）。
- F2 复用 F1 结果不重复 scroll（mock `_scroll_full_vectors` 计数为 1）。
- 集成验证：`data_demo` 小采样 GPU vs CPU 结果一致；WebUI 面板触发评估。

## 7. Migration Plan

1. 合并后直接可用：新增 `gpu_knn.py` 与 metrics GPU 路径为增量能力，numpy CPU 路径移除（`_batch_exact_knn` 删除）。
2. 回滚：`git revert` 本 change 回到 numpy 批量行为（若需恢复）。
3. CUDA 环境首次运行前确认 `uv sync` 安装 torch；无 GPU 机器自动走 CPU 分块路径。

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 显存溢出 | `estimate_block_q` ×0.8 保守余量 + `max_gpu_mem` 可调 + OOM 更小块重试；`close()` 加 `empty_cache` 释放缓存 |
| F1/F2 重复计算 | `evaluate_knn` 单次 scroll + 单次 topk，F1/F2 共享结果 |
| torch 依赖体积 | 延迟 import，无 CUDA 环境启动不失败 |
| GIL 限速聚合 | 多数投票用 Counter 原生路径；必要时退化串行 |
| 结果一致性 | 差分测试（GPU/CPU/服务端逐条三者一致） |
| float32 精度 | cosine 相似度下可接受；差分测试兜底 |
| use_batch 移除破坏性 | 调用方 CLI/WebUI/tests 同步更新；spec 已声明返回结构兼容 |

## 9. Open Questions（build 阶段定夺）

- `accuracy_by_k` 字段名与 CLI 表格布局细节（不改变本 design 结构）。
- `max_eval_ram` 默认值（6.0 沿用现状 vs 更高）——保守沿用现状，可在 CLI 调。
