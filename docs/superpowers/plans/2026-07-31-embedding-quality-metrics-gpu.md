---
change: embedding-quality-metrics
design-doc: docs/superpowers/specs/2026-07-31-embedding-quality-metrics-gpu-design.md
base-ref: e1e4b8f7430097c1705d666c844b14ff0db8b62f
archived-with: 2026-07-31-embedding-quality-metrics
---

# Embedding 质量评估 GPU 加速实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 为现有 F1/F2 评估模块新增 PyTorch CUDA GPU 加速路径，替代逐条 Qdrant exact 检索作为默认计算方式。

**Architecture:** Qdrant 负责数据层（scroll 向量+标签），PyTorch GPU 负责计算层（分块矩阵乘法做精确 KNN），searcher.py 不改动。

**Tech Stack:** Python 3.12+, numpy, torch>=2.0, qdrant-client, tqdm

## Global Constraints

- GPU 路径使用 float32 计算 cosine distance（Qdrant float64 → cast to float32）
- GPU 显存预算默认 16GB，作为可调参数 (`--gpu-memory-gb`)
- torch.cuda.is_available()==False 时自动回退 numpy CPU 路径
- --no-gpu 强制回退 Qdrant exact 逐条检索
- searcher.py / qdrant_client.py 不改动
- 现有单元测试 + 集成测试继续通过
- 保持中文错误信息

---

### Task 1: 添加 torch 依赖 + GPU 工具函数

**Files:**
- Modify: `pyproject.toml`
- Modify: `KNN_evaluation/metrics.py`

**Interfaces:**
- Produces: `_scroll_full_vectors(manager) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- Produces: `_gpu_exact_knn(query_vectors, all_vectors, all_labels, all_point_ids, k, gpu_memory_gb) -> tuple`

- [x] **Step 1: 添加 torch 依赖**

在 `pyproject.toml` 的 `dependencies` 中添加 `"torch>=2.0"`。

```bash
uv add torch
```

- [x] **Step 2: 更新 metrics.py imports**

```python
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False
```

- [x] **Step 3: 实现 `_scroll_full_vectors`**

```python
def _scroll_full_vectors(manager: QdrantManager) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 Qdrant 全量 scroll 向量、标签、point_id 到客户端内存.
    
    Returns:
        vectors: (N, 64) float64
        labels: (N,) int64
        point_ids: (N,) str
    """
    all_vectors = []
    all_labels = []
    all_point_ids = []
    offset = None
    while True:
        records, offset = manager.client.scroll(
            collection_name=manager.collection_name,
            scroll_filter=None,
            limit=50000,
            offset=offset,
            with_payload=["label"],
            with_vectors=True,
        )
        if not records:
            break
        for rec in records:
            all_vectors.append(np.array(rec.vector, dtype=np.float64))
            p = rec.payload or {}
            all_labels.append(int(p.get("label", -1)))
            all_point_ids.append(str(rec.id))
        if offset is None:
            break
    return (
        np.stack(all_vectors),
        np.array(all_labels, dtype=np.int64),
        np.array(all_point_ids),
    )
```

- [x] **Step 4: 实现 `_gpu_exact_knn`**

```python
def _gpu_exact_knn(
    query_vectors: np.ndarray,
    all_vectors: np.ndarray,
    all_labels: np.ndarray,
    all_point_ids: np.ndarray,
    k: int,
    gpu_memory_gb: float = 16,
) -> tuple:
    """GPU 批量精确 KNN.
    
    Args:
        query_vectors: (Q, 64) float64 查询向量矩阵
        all_vectors: (N, 64) float64 全集向量矩阵
        all_labels: (N,) int 标签数组
        all_point_ids: (N,) str point_id 数组
        k: Top-K
        gpu_memory_gb: GPU 显存预算
    
    Returns:
        topk_scores: (Q, K) float32  Top-K cosine similarity
        topk_labels:  (Q, K) int32    Top-K 标签
        topk_ids:     (Q, K) str      Top-K point_id
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch 未安装，无法使用 GPU 路径")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = torch.cuda.is_available()
    
    if not use_gpu:
        import warnings
        warnings.warn("CUDA not available, falling back to CPU")
    
    # 每GB显存约可容纳 16M 个 float32×64 向量，预留一半给queries
    chunk_size = int(gpu_memory_gb * 1e9 / (64 * 4) / 2)
    
    N = all_vectors.shape[0]
    Q = query_vectors.shape[0]
    
    # 转换为 float32 torch tensor, 上传设备
    queries_t = torch.from_numpy(query_vectors.astype(np.float32)).to(device)
    # L2 normalize queries
    queries_t = queries_t / torch.clamp(torch.norm(queries_t, dim=1, keepdim=True), min=1e-12)
    
    # 初始化 Top-K 累加器
    topk_scores = torch.full((Q, 0), float('-inf'), device=device)
    topk_indices = torch.full((Q, 0), -1, dtype=torch.long, device=device)
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = torch.from_numpy(all_vectors[start:end].astype(np.float32)).to(device)
        chunk = chunk / torch.clamp(torch.norm(chunk, dim=1, keepdim=True), min=1e-12)
        
        # Cosine similarity matrix: (Q, chunk_size)
        sim = queries_t @ chunk.T
        
        # 合并当前块的 Top-K 与之前的 Top-K
        combined_scores = torch.cat([topk_scores, sim], dim=1)
        combined_indices = torch.cat([
            topk_indices,
            torch.arange(start, end, device=device).unsqueeze(0).expand(Q, -1),
        ], dim=1)
        
        # 取全局 Top-K
        topk_scores, local_idx = torch.topk(combined_scores, k=k, dim=1)
        topk_indices = torch.gather(combined_indices, dim=1, index=local_idx)
    
    # 转换为 numpy
    scores_np = topk_scores.cpu().numpy()
    idx_np = topk_indices.cpu().numpy()
    labels_np = all_labels[idx_np]
    ids_np = all_point_ids[idx_np]
    
    return scores_np, labels_np, ids_np
```

- [x] **Step 5: 提交**

```bash
git add pyproject.toml KNN_evaluation/metrics.py
uv add torch
git add uv.lock  # if exists
git commit -m "feat(metrics): add torch GPU dependency and _gpu_exact_knn helper functions"
```

### Task 2: 修改 compute_knn_accuracy 支持 GPU 路径

**Files:**
- Modify: `KNN_evaluation/metrics.py`

- [x] **Step 1: 更新 compute_knn_accuracy 签名和 GPU 分支**

```python
def compute_knn_accuracy(
    manager: QdrantManager,
    queries: list[dict],
    k: int = 10,
    exact: bool = True,
    use_gpu: bool = True,
    gpu_memory_gb: float = 16,
    progress_callback: ProgressCallback = None,
) -> dict:
    """计算 KNN 分类准确率, 默认使用 GPU 加速."""
    num_labels = len(LABEL_NAMES)
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    total_queries = len(queries)
    start = time.perf_counter()
    
    # 过滤出有效查询像素
    valid_queries = [q for q in queries if q.get("vectors") is None]
    query_vectors = np.array([q["vector"] for q in valid_queries], dtype=np.float64)
    query_point_ids = [q["point_id"] for q in valid_queries]
    query_labels = [q["label"] for q in valid_queries]
    Q = len(query_vectors)
    
    if use_gpu and _TORCH_AVAILABLE:
        if progress_callback:
            progress_callback(0, 1)
        all_vecs, all_lbls, all_ids = _scroll_full_vectors(manager)
        if progress_callback:
            progress_callback(1, 2)
        topk_scores, topk_labels, topk_ids_np = _gpu_exact_knn(
            query_vectors, all_vecs, all_lbls, all_ids.astype(str), k + 1, gpu_memory_gb,
        )
        if progress_callback:
            progress_callback(2, 2)
        # LOO + majority vote on CPU (results are small)
        for i in range(Q):
            effective_labels = []
            for j in range(k + 1):
                if str(topk_ids_np[i][j]) != query_point_ids[i]:
                    effective_labels.append(int(topk_labels[i][j]))
                if len(effective_labels) >= k:
                    break
            if not effective_labels:
                continue
            predicted = max(set(effective_labels), key=effective_labels.count)
            confusion[query_labels[i]][predicted] += 1
        elapsed = time.perf_counter() - start
        # compute per_class metrics (same as before)...
        return {...}
    elif use_gpu and not _TORCH_AVAILABLE:
        import warnings
        warnings.warn("torch not available, falling back to Qdrant exact")
    # ... existing Qdrant exact fallback ...
```

- [x] **Step 2: 运行现有测试确保无回归**

```bash
uv run pytest KNN_evaluation/tests/test_metrics.py -v -k "not integration"
```

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/metrics.py
git commit -m "feat(metrics): add GPU-accelerated path to compute_knn_accuracy"
```

### Task 3: 修改 compute_purity_recall_curve 支持 GPU 路径

**Files:**
- Modify: `KNN_evaluation/metrics.py`

- [x] **Step 1: 更新 compute_purity_recall_curve 签名和 GPU 分支**

```python
def compute_purity_recall_curve(
    manager: QdrantManager,
    queries: list[dict],
    k_values: list[int] | None = None,
    exact: bool = True,
    use_gpu: bool = True,
    gpu_memory_gb: float = 16,
    progress_callback: ProgressCallback = None,
) -> dict:
    """计算 Purity/Recall@K 曲线, 默认使用 GPU 加速."""
    # GPU path: scroll once, compute once with max_k+1
    # Use topk_scores/labels from _gpu_exact_knn
    # Slice different k to compute purity/recall
```

- [x] **Step 2: 运行测试**

```bash
uv run pytest KNN_evaluation/tests/test_metrics.py -v -k "not integration"
```

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/metrics.py
git commit -m "feat(metrics): add GPU-accelerated path to compute_purity_recall_curve"
```

### Task 4: CLI 新增 GPU 参数

**Files:**
- Modify: `KNN_evaluation/cli.py`

- [x] **Step 1: 新增 --gpu-memory-gb 和 --no-gpu 参数**

在 evaluate subparser 中添加:
```python
p_eval.add_argument("--gpu-memory-gb", type=float, default=16,
                    help="GPU 显存预算 (GB, 默认: 16)")
p_eval.add_argument("--no-gpu", action="store_true",
                    help="强制回退 Qdrant exact 逐条检索")
```

- [x] **Step 2: 在 cmd_evaluate 中传递 GPU 参数**

```python
use_gpu = not args.no_gpu
f1 = compute_knn_accuracy(manager, queries, k=args.k_f1, use_gpu=use_gpu,
                          gpu_memory_gb=args.gpu_memory_gb, ...)
f2 = compute_purity_recall_curve(manager, queries, k_values, use_gpu=use_gpu,
                                 gpu_memory_gb=args.gpu_memory_gb, ...)
```

- [x] **Step 3: 验证 CLI help**

```bash
uv run python -m KNN_evaluation.cli evaluate --help
```

- [x] **Step 4: 提交**

```bash
git add KNN_evaluation/cli.py
git commit -m "feat(cli): add --gpu-memory-gb and --no-gpu arguments to evaluate"
```

### Task 5: WebUI 新增 GPU 控件

**Files:**
- Modify: `KNN_evaluation/webui.py`

- [x] **Step 1: 在评估面板参数区新增 GPU 输入**

```python
gpu_memory_input = ui.number(label="GPU 显存预算 (GB)", value=16, min=1, max=48).classes("w-32")
no_gpu_switch = ui.switch("禁用 GPU", value=False)
```

- [x] **Step 2: 传递 GPU 参数到 metrics 调用**

```python
use_gpu = not no_gpu_switch.value
gpu_memory_gb_val = float(gpu_memory_input.value)
f1 = await asyncio.to_thread(compute_knn_accuracy, manager, queries, k_f1,
                              True, use_gpu, gpu_memory_gb_val, ...
```

- [x] **Step 3: 验证导入**

```bash
uv run python -c "from KNN_evaluation.webui import index; print('import OK')"
```

- [x] **Step 4: 提交**

```bash
git add KNN_evaluation/webui.py
git commit -m "feat(webui): add GPU memory budget and no-GPU controls"
```

### Task 6: GPU 路径单元测试 + 集成验证

**Files:**
- Modify: `KNN_evaluation/tests/test_metrics.py`

- [x] **Step 1: 新增 GPU mock 测试**

```python
class TestGpuExactKnn:
    def test_gpu_knn_matches_numpy(self):
        """验证 GPU KNN 结果与 numpy brute force 一致."""
        ...
    def test_chunked_knn_matches_full(self):
        """验证分块结果与全量单块一致."""
        ...
    def test_fallback_to_cpu_when_no_cuda(self):
        """验证 CUDA 不可用时自动回退."""
        ...
```

- [x] **Step 2: 运行全量测试**

```bash
uv run pytest KNN_evaluation/tests/ -v -k "not integration"
```

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/tests/test_metrics.py
git commit -m "test(metrics): add GPU KNN correctness and fallback tests"
```
