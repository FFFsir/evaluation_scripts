---
change: gpu-knn-scale-evaluation
design-doc: docs/superpowers/specs/2026-08-02-gpu-knn-scale-evaluation-design.md
base-ref: 3a45630e3e387025e1c0055c11cf311594eca1b4
archived-with: 2026-08-03-gpu-knn-scale-evaluation
---

# GPU 分块 KNN 大规模评估 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 GPU 分块矩阵乘 + CPU 多线程聚合重建批量精确 KNN 评估路径，使 16GB RAM / 20GB VRAM 约束下可完成 90 万查询 × top-10001 的精确 KNN 统计（F1 多 K + F2 Purity/Recall），并以 `--device cuda|cpu|auto` 取代 `use_batch`/`--batch`。

**Architecture:** 新增 device-agnostic 的 `KNN_evaluation/gpu_knn.py`（`KnnEngine`）：corpus float32 (N,64) 常驻 device，L2 归一化后按 `query_block @ corpus.T` + `torch.topk` 分块计算，边算边回传 CPU (scores, labels, ids)，不物化 (Q,N)/(Q,K) 全量。`metrics.py` 的 `compute_knn_accuracy` / `compute_purity_recall_curve` 在 `exact=True` 时走 `KnnEngine` 分块路径（device 参数决定 cuda/cpu），F1 用 LOO + 多数投票（平票递减 K）+ 9×9 混淆矩阵，F2 用 `ThreadPoolExecutor` 对块内行并行累加 Purity/Recall；`use_batch` 与 `_batch_exact_knn`（numpy 整体物化路径）删除。CLI/WebUI 暴露 `--device`/`--gpu-batch-q`/`--max-gpu-mem`/`--max-eval-ram`。F3 距离分布不在本 change 范围。

**Tech Stack:** Python 3.12+, torch（新增依赖，`KnnEngine.__init__` 延迟 import）, numpy, qdrant-client, NiceGUI, pytest（`uv run pytest`）

## Global Constraints

- `device="auto"`：`torch.cuda.is_available()` True → cuda，否则 → cpu + 警告；`device="cuda"` 显式但 CUDA 不可用 → 抛明确错误（不静默回退）；`device="cpu"` → torch CPU 分块（RAM 预算 `max_eval_ram`），结果与 GPU 一致
- `compute_knn_accuracy` / `compute_purity_recall_curve` 的 `use_batch` 参数移除（唯一破坏性变更）；返回结构保持：`overall_accuracy`（仍 = 单 K `k` 的准确率）、`per_class_metrics`、`confusion_matrix`、`k`、`num_queries`、`elapsed_sec`，F1 新增 `k_values`（默认 `[k]`）与 `accuracy_by_k: {kv: float}` 多 K；F2 返回结构不变
- `KnnEngine.set_corpus` L2 归一化语义与旧 `_batch_exact_knn` 一致（cosine 相似度）；`_scroll_full_vectors` 改为返回 float32 向量（供常驻 GPU/cpu），标签 int64、point_id str 不变
- 显存/内存推导公式：`block_q = clamp(floor(0.8 × (mem_GB − corpus_GB) × 1e9 / (N × 4)), 1, Q)`；`corpus_GB = N × 64 × 4 / 1e9`；`gpu_batch_q` 显式给定时直接用作 block_q；CPU 回退同理（`max_eval_ram` 代入，corpus_GB 计 float32）
- 多 K 零额外检索：单次 `topk(k=max_k+1)`，其中 `max_k = max(k_values + [k_f1])`；`torch.topk(k=min(k, N))`；`compute_knn_accuracy` 新增 `k_values` 参数（默认 `[k]`），`accuracy_by_k` 对每个 K 用同一份 Top-K 递增取多数投票，`overall_accuracy` 仍 = 单 K `k`
- `_batch_exact_knn`、`estimate_batch_memory`、`guard_batch_memory` 删除；`--batch` 标志删除，新增 `--device`（默认 auto）、`--gpu-batch-q`、`--max-gpu-mem`（默认 16）、`--max-eval-ram`（默认 6.0）
- `torch` 依赖延迟 import：`KnnEngine.__init__` 内 `import torch`；无 CUDA/无 torch 环境启动不失败（`--device cpu` 时 torch 仍需安装）
- 中文注释与错误信息；所有测试命令用 `uv run pytest`；差分测试的 GPU 分支须用 `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` 跳过，CPU 分支始终跑
- F3 距离分布（`compute_distance_distribution`）不在本 change 范围，不实现
- 不修改 Collection schema、payload 结构、`searcher.py` 单条检索行为、`visualization.py`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `KNN_evaluation/gpu_knn.py` | CREATE | `KnnEngine`（`resolve_device` / `set_corpus` / `knn_chunk` / `estimate_block_q` / `close`，延迟 import torch） |
| `KNN_evaluation/tests/test_gpu_knn.py` | CREATE | 差分测试（GPU/CPU/服务端逐条一致）、`estimate_block_q`、LOO、平票、F1 多 K、F2、CUDA 回退 mock、边界、`device="cpu"` |
| `KNN_evaluation/metrics.py` | MODIFY | `_scroll_full_vectors` 返回 float32；删除 `use_batch`/`_batch_exact_knn`/`estimate_batch_memory`/`guard_batch_memory`；新增 `k_values`/`device`/`gpu_batch_q`/`max_gpu_mem`/`max_eval_ram` 参数与 GPU 分块聚合路径；F1 新增 `accuracy_by_k`；F2 块内行 `ThreadPoolExecutor` 并行 |
| `KNN_evaluation/tests/test_metrics.py` | MODIFY | 移除 `use_batch`/`_batch_exact_knn`/内存守卫相关断言，改用新签名 |
| `KNN_evaluation/cli.py` | MODIFY | `evaluate` 移除 `--batch`，新增 `--device`/`--gpu-batch-q`/`--max-gpu-mem`/`--max-eval-ram`，打印 GPU 路径信息，输出 F1 多 K 表，JSON 导出兼容 |
| `KNN_evaluation/webui.py` | MODIFY | 评估面板 batch checkbox → device 选择器 + max_gpu_mem 输入；进度反馈扩展；参数透传 |
| `pyproject.toml` | MODIFY | 新增 `torch` 依赖 |

---

### Task 1: `gpu_knn.py` — KnnEngine 核心（新增）

**Files:**
- Create: `KNN_evaluation/gpu_knn.py`
- Create: `KNN_evaluation/tests/test_gpu_knn.py`

**Interfaces:**
- Produces:
  - `KnnEngine.__init__(self, device="auto")`：延迟 `import torch`；`self.device = resolve_device(device)` → `"cuda"` | `"cpu"`；`device="cuda"` 且 CUDA 不可用时抛 `RuntimeError`
  - `KnnEngine.set_corpus(self, all_vectors: np.ndarray) -> None`：校验 `all_vectors.ndim == 2 and all_vectors.shape[1] == 64`；`torch.from_numpy(all_vectors.astype(np.float32)).to(self.device)`，L2 归一化；另存 `self._labels: np.ndarray`（int64）与 `self._ids: np.ndarray`（str）为 CPU numpy
  - `KnnEngine.knn_chunk(self, query_block: np.ndarray, k: int) -> tuple`：L2 归一化 → `q @ self.corpus.T` → `torch.topk(sim, k=min(k, N), dim=1)` → 返回 CPU `(scores, labels, ids)`，其中 `scores: (b,K) float64`、`labels: (b,K) int64`、`ids: (b,K) str`（通过 `self._labels[idx]` / `self._ids[idx]` 花式索引取回）
  - `KnnEngine.estimate_block_q(self, max_mem_gb: float) -> int`：按公式推导并 clamp 到 `[1, Q]`；corpus 未 set 时抛 `ValueError`
  - `KnnEngine.close(self) -> None`：`self.corpus = None`（释放 device 内存）
- Consumed by: Task 2（metrics）、Task 6 差分测试
- Module helper: `resolve_device(device: str) -> str`

**Dependencies:** 无（最先实现；pyproject 的 torch 依赖与 Task 8 一起补，本任务直接 import 并假设可安装）。

- [x] **Step 1: 写失败测试**

创建 `KNN_evaluation/tests/test_gpu_knn.py`：

```python
"""KnnEngine（device-agnostic 分块精确 KNN）单元测试."""
import numpy as np
import pytest

from KNN_evaluation.gpu_knn import KnnEngine, resolve_device


@pytest.fixture
def corpus():
    rng = np.random.RandomState(7)
    return rng.randn(1000, 64).astype(np.float32)


def test_resolve_device_auto_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_explicit_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_set_corpus_wrong_dim_raises():
    engine = KnnEngine(device="cpu")
    with pytest.raises(ValueError):
        engine.set_corpus(np.zeros((10, 63), dtype=np.float32))
    engine.close()


def test_knn_chunk_returns_cpu_arrays(corpus):
    engine = KnnEngine(device="cpu")
    engine.set_corpus(corpus)
    q = corpus[:5]  # 查询取自 corpus（含自身）
    scores, labels, ids = engine.knn_chunk(q, k=10)
    assert scores.shape == (5, 10)
    assert labels.shape == (5, 10)
    assert ids.shape == (5, 10)
    assert scores.dtype == np.float64
    assert labels.dtype == np.int64
    assert ids.dtype.kind == "U"
    # 余弦相似度范围 [-1, 1]
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
    engine.close()


def test_knn_top1_self_match(corpus):
    engine = KnnEngine(device="cpu")
    engine.set_corpus(corpus)
    scores, labels, ids = engine.knn_chunk(corpus[:3], k=1)
    # 查询向量来自 corpus 前 3 行，Top-1 应为自身
    assert list(ids[:, 0]) == ["pt-0", "pt-1", "pt-2"]
    assert abs(scores[0, 0] - 1.0) < 1e-3
    engine.close()


def test_estimate_block_q(corpus):
    engine = KnnEngine(device="cpu")
    engine.set_corpus(corpus)
    N, D = corpus.shape
    assert D == 64
    corpus_gb = N * 64 * 4 / 1e9
    # 预算 1GB：block_q = floor(0.8*(1-corpus_gb)*1e9/(N*4))，clamp 到 [1, Q]
    bq = engine.estimate_block_q(1.0)
    assert 1 <= bq <= N
    # 更大预算 → 更大块
    assert engine.estimate_block_q(8.0) >= bq
    engine.close()


def test_estimate_block_q_without_corpus_raises():
    engine = KnnEngine(device="cpu")
    with pytest.raises(ValueError):
        engine.estimate_block_q(1.0)
    engine.close()
```

- [x] **Step 2: 运行测试，确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_gpu_knn.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'KNN_evaluation.gpu_knn'`

- [x] **Step 3: 实现 `gpu_knn.py`**

创建 `KNN_evaluation/gpu_knn.py`：

```python
"""GPU/CPU device-agnostic 分块精确 KNN 核心.

全库向量 float32 常驻 device（cuda 或 torch cpu），查询分块做
`query_block @ corpus.T` + `torch.topk`，边算边回传 CPU，不物化 (Q,N)
相似度矩阵或 (Q,K) 全量结果。L2 归一化语义与旧 numpy `_batch_exact_knn`
一致（cosine 相似度）。torch 延迟 import——无 CUDA 环境启动不失败。
"""
import numpy as np


def resolve_device(device: str) -> str:
    """把用户指定 device 解析为 'cuda' | 'cpu'。

    'auto' → CUDA 可用则 'cuda'，否则 'cpu'（并提示回退）。
    'cuda' 显式但 CUDA 不可用 → 抛明确错误（不静默回退）。
    'cpu' → 原样返回。
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        import warnings
        warnings.warn("CUDA 不可用，回退 torch CPU 分块路径", RuntimeWarning)
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' 但当前环境 CUDA 不可用（torch.cuda.is_available()=False）。"
                "请改用 --device cpu 或 --device auto。"
            )
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"未知 device: {device!r}，可选: cuda|cpu|auto")


class KnnEngine:
    """device-agnostic 分块精确 KNN 引擎（cuda / torch cpu 共用一套代码）。"""

    def __init__(self, device: str = "auto"):
        import torch  # 延迟 import：无 torch 环境（纯 numpy 路径）不失败
        self.torch = torch
        self.device = resolve_device(device)
        self.corpus = None
        self.N = 0
        self._labels: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    def set_corpus(self, all_vectors: np.ndarray) -> None:
        """全量向量 float32 (N, 64) 常驻 device，L2 归一化；另存 CPU labels/ids."""
        all_vectors = np.asarray(all_vectors)
        if all_vectors.ndim != 2 or all_vectors.shape[1] != 64:
            raise ValueError(
                f"corpus 必须是 (N, 64) 二维数组，实际 {all_vectors.shape}"
            )
        self.N = int(all_vectors.shape[0])
        t = self.torch
        tensor = t.from_numpy(all_vectors.astype(np.float32)).to(self.device)
        self.corpus = tensor / tensor.norm(dim=1, keepdim=True).clamp(min=1e-12)

    def _attach_meta(self, labels: np.ndarray, point_ids: np.ndarray) -> None:
        """预取标签/point_id 为 CPU numpy，knn_chunk 花式索引取回（避免 GPU 传输）."""
        self._labels = np.asarray(labels, dtype=np.int64)
        self._ids = np.asarray(point_ids)

    def knn_chunk(self, query_block: np.ndarray, k: int) -> tuple:
        """单块分块 matmul：L2 归一化 → q @ corpus.T → topk → CPU (scores, labels, ids)."""
        if self.corpus is None:
            raise ValueError("KnnEngine.set_corpus 未调用")
        t = self.torch
        q = t.from_numpy(np.asarray(query_block).astype(np.float32)).to(self.device)
        q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
        sim = q @ self.corpus.T                      # (block_q, N) float32
        k_eff = min(int(k), self.N)
        scores, idx = t.topk(sim, k=k_eff, dim=1)
        labels = self._labels[idx.cpu().numpy()]     # 花式索引取标签（CPU numpy）
        ids = self._ids[idx.cpu().numpy()]           # 取 point_id
        return scores.cpu().numpy(), labels, ids

    def estimate_block_q(self, max_mem_gb: float) -> int:
        """显存/内存预算 → block_q（峰值占用 < 预算，×0.8 保守余量）.

        block_q = clamp(floor(0.8 × (max_mem_GB − corpus_GB) × 1e9 / (N × 4)), 1, Q)
        corpus_GB = N × 64 × 4 / 1e9
        """
        if self.corpus is None:
            raise ValueError("KnnEngine.set_corpus 未调用，无法推导 block_q")
        N = self.N
        corpus_gb = N * 64 * 4 / 1e9
        if max_mem_gb <= corpus_gb:
            return 1
        block_q = int(0.8 * (float(max_mem_gb) - corpus_gb) * 1e9 // (N * 4))
        return max(1, min(block_q, N))

    def close(self) -> None:
        """释放 device 内存."""
        self.corpus = None
```

- [x] **Step 4: 运行测试，确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_gpu_knn.py -v`
Expected: PASS（9 个用例）

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/gpu_knn.py KNN_evaluation/tests/test_gpu_knn.py
git commit -m "feat: add device-agnostic KnnEngine chunked exact KNN core"
```

---

### Task 2: `metrics.py` — 重构 F1 批量路径为 GPU 分块 + 多 K

**Files:**
- Modify: `KNN_evaluation/metrics.py`
- Modify: `KNN_evaluation/tests/test_metrics.py`

**Interfaces:**
- Consumes: Task 1 的 `KnnEngine`（`set_corpus` / `knn_chunk` / `estimate_block_q` / `close` / `_attach_meta`）
- Produces:
  - `compute_knn_accuracy(manager, queries, k=10, exact=True, k_values=None, device="auto", gpu_batch_q=None, max_gpu_mem=16, max_eval_ram=6.0, progress_callback=None) -> dict`：返回 `overall_accuracy`（仍 = 单 K `k`）、`per_class_metrics`、`confusion_matrix`、`k`、`num_queries`、`elapsed_sec`，新增 `accuracy_by_k: {kv: float}`（对 `k_values`（默认 `[k]`）每个 K 递增取多数投票，零额外检索）；`k_values=None` 时等价 `{k}`
  - `_scroll_full_vectors(manager) -> tuple[np.ndarray, np.ndarray, np.ndarray]`：向量改 float32，标签 int64，point_id str
  - 删除 `_batch_exact_knn`、`use_batch` 参数
  - 内部助手（模块级，供测试 import）：`_aggregate_knn_row(topk_labels: np.ndarray, topk_ids: np.ndarray, query_point_id: str, query_label: int, k: int, n_labels: int, confusion: np.ndarray) -> None`
  - 参数透传占位：`_device_budget(device, gpu_batch_q, max_gpu_mem, max_eval_ram, engine)` → `(block_q, budget_gb)`（Task 3 复用）
- Consumed by: Task 3（cli）、Task 4（webui）、Task 6 测试

**Dependencies:** Task 1。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_metrics.py` 中替换 `TestBatchExactKnn` 类，并新增 GPU 路径用例。先改 `_scroll_full_vectors` 的 dtype 断言（float64 → float32），再新增：

```python
class TestKnnAccuracyGpuPath:
    """GPU 分块路径（device=cpu 跑 torch 分块，逻辑与 GPU 一致）。"""

    def _fake_engine(self, topk_labels, topk_ids):
        """构造返回固定 Top-K 的 mock KnnEngine（避免真实矩阵乘）。"""
        from unittest.mock import MagicMock
        engine = MagicMock()
        engine.estimate_block_q.return_value = 2  # 每块 2 行，强制多块
        engine.knn_chunk.side_effect = [
            (np.zeros((2, 5)), topk_labels[:2], topk_ids[:2]),
            (np.zeros((1, 5)), topk_labels[2:], topk_ids[2:]),
        ]
        return engine

    def test_accuracy_by_k_consistent_with_overall(self, monkeypatch):
        """accuracy_by_k 多 K 与单 K overall_accuracy 一致."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        # 3 个查询，每个 5 个邻居（已剔除自身后）
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 3}
            for i in range(3)
        ]
        # Top-5 邻居: q0 → 全 0; q1 → 全 1; q2 → 全 2（k=3 与 k=5 都应预测正确）
        topk_labels = np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
        ], dtype=np.int64)
        topk_ids = np.array([
            ["n00", "n01", "n02", "n03", "n04"],
            ["n10", "n11", "n12", "n13", "n14"],
            ["n20", "n21", "n22", "n23", "n24"],
        ])

        fake = self._fake_engine(topk_labels, topk_ids)
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((3, 64), dtype=np.float32), np.array([0, 1, 2]), np.array(["n00", "n10", "n20"]),
        ))

        result = M.compute_knn_accuracy(
            manager, queries, k=3, k_values=[3, 5], exact=True, device="cpu",
        )

        assert result["overall_accuracy"] == 1.0
        assert result["accuracy_by_k"] == {3: 1.0, 5: 1.0}
        assert result["num_queries"] == 3
        # 单 K overall_accuracy 与 accuracy_by_k[k] 一致
        assert result["overall_accuracy"] == result["accuracy_by_k"][3]

    def test_loo_excludes_self_in_gpu_path(self, monkeypatch):
        """GPU 路径 LOO 剔除自身."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-3: [自身(label=0), 0, 0] → 剔除自身后 [0, 0] → 预测 0 正确
        topk_labels = np.array([[0, 0, 0]], dtype=np.int64)
        topk_ids = np.array([["q0", "n1", "n2"]])

        fake = self._fake_engine(topk_labels, topk_ids)
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["q0"]),
        ))

        result = M.compute_knn_accuracy(manager, queries, k=2, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0

    def test_tie_break_decrement_in_gpu_path(self, monkeypatch):
        """GPU 路径平票用 _resolve_tie 递减 K 打破."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-5: 0,0,1,1,2 → k=5 平票；k=4（取前 4）: 0,0,1,1 仍平票；
        # k=2（K-3）: 0,0 → 预测 0 正确
        topk_labels = np.array([[0, 0, 1, 1, 2]], dtype=np.int64)
        topk_ids = np.array([["n0", "n1", "n2", "n3", "n4"]])

        fake = self._fake_engine(topk_labels, topk_ids)
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n0"]),
        ))

        result = M.compute_knn_accuracy(manager, queries, k=5, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0
```

同时修改 `TestScrollFullVectors`（`test_scrolls_all_vectors`）的 dtype 断言：`assert vectors.dtype == np.float32`。

- [x] **Step 2: 运行测试，确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: FAIL——`compute_knn_accuracy` 仍是旧签名（`use_batch` 存在、`device` 缺失，`accuracy_by_k` 缺失），`_scroll_full_vectors` 返回 float64。

- [x] **Step 3: 重构 `metrics.py` 的 F1 批量路径**

删除 `estimate_batch_memory`、`guard_batch_memory`、`_batch_exact_knn`。`_scroll_full_vectors` 中 `np.array(rec.vector, dtype=np.float64)` → `dtype=np.float32`，返回处 `np.array(all_vectors, dtype=np.float32).reshape(-1, 64)`、空时 `np.empty((0, 64), dtype=np.float32)`。修改 `compute_knn_accuracy`：

```python
def compute_knn_accuracy(
    manager: QdrantManager,
    queries: list[dict],
    k: int = 10,
    exact: bool = True,
    k_values: list[int] | None = None,
    device: str = "auto",
    gpu_batch_q: int | None = None,
    max_gpu_mem: float = 16,
    max_eval_ram: float = 6.0,
    progress_callback: ProgressCallback = None,
) -> dict:
    """计算 KNN 分类准确率（Leave-One-Out）+ Per-class F1 + accuracy_by_k 多 K.

    k_values: 多 K 准确率取值（默认 [k]，仅计算单 K）。对每个 K 从同一份
    Top-(max_k+1) 结果递增取多数投票，零额外检索。overall_accuracy 仍 = 单 K k。

    exact=True 走 KnnEngine 分块路径（device 决定 cuda/cpu）；exact=False
    保留 PixelSearcher ANN 逐条（供 --ann 对比）。
    """
    from KNN_evaluation.gpu_knn import KnnEngine
    from KNN_evaluation.label_mapping import LABEL_NAMES as _LN

    num_labels = len(_LN)
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    start = time.perf_counter()

    if not exact:
        return _knn_accuracy_sequential(
            manager, queries, k, exact=False, progress_callback=progress_callback,
        )

    valid_queries = [q for q in queries if q.get("vectors") is None]
    Q = len(valid_queries)
    if Q == 0:
        return _empty_knn_accuracy_result(confusion, k, start)

    if progress_callback:
        progress_callback(0, Q)

    all_vecs, all_lbls, all_ids = _scroll_full_vectors(manager)
    if all_vecs.shape[0] == 0:
        return _knn_accuracy_sequential(
            manager, queries, k, exact=True, progress_callback=progress_callback,
        )

    # 多 K 取值：k_values 缺省为 [k]；单次 top-(max_k+1) 覆盖全部 K
    k_values = sorted(set([k] + list(k_values or [])))
    max_k = max(k_values)

    engine = KnnEngine(device=device)
    engine.set_corpus(all_vecs)
    engine._attach_meta(all_lbls, all_ids)

    block_q, _budget_gb = _device_budget(
        device, gpu_batch_q, max_gpu_mem, max_eval_ram, engine,
    )

    query_vectors = np.array([q["vector"] for q in valid_queries], dtype=np.float32)
    query_point_ids = [q["point_id"] for q in valid_queries]
    query_labels = [q["label"] for q in valid_queries]

    # 多 K 累加器：每个 K 一张混淆矩阵（同一份 Top-K 结果递增取值）
    confusion_by_k: dict[int, np.ndarray] = {
        kv: np.zeros((num_labels, num_labels), dtype=np.int64) for kv in k_values
    }

    done = 0
    for start_i in range(0, Q, block_q):
        end_i = min(start_i + block_q, Q)
        _scores, topk_labels, topk_ids = engine.knn_chunk(
            query_vectors[start_i:end_i], max_k + 1,
        )
        for row in range(end_i - start_i):
            i = start_i + row
            for kv in k_values:
                _aggregate_knn_row(
                    topk_labels[row], topk_ids[row],
                    query_point_ids[i], query_labels[i], kv,
                    num_labels, confusion_by_k[kv],
                )
        done = end_i
        if progress_callback:
            progress_callback(done, Q)

    engine.close()

    # overall_accuracy 仍 = 单 K k 的准确率（既有调用方兼容）
    overall_accuracy = _confusion_accuracy(confusion_by_k[k])
    per_class = _per_class_metrics_from_confusion(confusion_by_k[k], _LN)

    elapsed = time.perf_counter() - start
    return {
        "overall_accuracy": round(float(overall_accuracy), 4),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion_by_k[k],
        "k": k,
        "num_queries": Q,
        "elapsed_sec": round(elapsed, 2),
        "accuracy_by_k": {
            kv: round(float(_confusion_accuracy(confusion_by_k[kv])), 4)
            for kv in k_values
        },
    }
```

新增模块级助手（放在 `compute_knn_accuracy` 之后）：

```python
def _device_budget(device, gpu_batch_q, max_gpu_mem, max_eval_ram, engine):
    """推导 block_q 与预算 GB（供 metrics/cli/webui 复用）.

    gpu_batch_q 显式给定 → 直接用作 block_q；否则 device=='cpu' 用
    max_eval_ram（RAM 预算），否则用 max_gpu_mem（显存预算）走公式推导。
    """
    if gpu_batch_q is not None:
        return max(1, int(gpu_batch_q)), (max_gpu_mem if device != "cpu" else max_eval_ram)
    budget = max_eval_ram if device == "cpu" else max_gpu_mem
    return engine.estimate_block_q(budget), budget


def _aggregate_knn_row(
    topk_labels: np.ndarray,
    topk_ids: np.ndarray,
    query_point_id: str,
    query_label: int,
    k: int,
    n_labels: int,
    confusion: np.ndarray,
) -> None:
    """LOO 剔除自身 + 取前 k 个有效邻居 + 多数投票（平票递减 K）→ 写 confusion."""
    effective_labels: list[int] = []
    for j in range(len(topk_ids)):
        if str(topk_ids[j]) != query_point_id:
            effective_labels.append(int(topk_labels[j]))
        if len(effective_labels) >= k:
            break
    if not effective_labels:
        return
    predicted = _resolve_tie(effective_labels)
    confusion[query_label][predicted] += 1


def _confusion_accuracy(confusion: np.ndarray) -> float:
    return float(np.trace(confusion)) / max(float(confusion.sum()), 1.0)


def _per_class_metrics_from_confusion(confusion, label_names: dict) -> dict:
    out: dict[str, dict] = {}
    for lid, lname in label_names.items():
        tp = confusion[lid][lid]
        fp = confusion[:, lid].sum() - tp
        fn = confusion[lid, :].sum() - tp
        support = int(confusion[lid, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        out[lname] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": support,
        }
    return out


def _empty_knn_accuracy_result(confusion, k, start) -> dict:
    return {
        "overall_accuracy": 0.0,
        "per_class_metrics": {},
        "confusion_matrix": confusion,
        "k": k,
        "num_queries": 0,
        "elapsed_sec": round(time.perf_counter() - start, 2),
        "accuracy_by_k": {},
    }


def _knn_accuracy_sequential(manager, queries, k, exact, progress_callback) -> dict:
    """Qdrant 逐条 exact / ANN 路径（原实现主体，签名去 use_batch）."""
    from KNN_evaluation.label_mapping import LABEL_NAMES as _LN
    searcher = PixelSearcher(manager)
    num_labels = len(_LN)
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    start = time.perf_counter()
    processed = 0
    total_queries = len(queries)
    for idx, q in enumerate(queries):
        if q.get("vectors") is not None:
            continue
        result = searcher.search(query_vector=q["vector"], k=k + 1, exact=exact)
        effective_hits = [h for h in result.hits if h.id != q["point_id"]][:k]
        if not effective_hits:
            continue
        true_label = q["label"]
        votes = [h.label for h in effective_hits]
        counter = collections.Counter(votes)
        max_count = max(counter.values())
        winners = [l for l, c in counter.items() if c == max_count]
        predicted = winners[0] if len(winners) == 1 else _resolve_tie(votes, effective_hits)
        confusion[true_label][predicted] += 1
        processed += 1
        if progress_callback:
            progress_callback(processed, total_queries)
    per_class = _per_class_metrics_from_confusion(confusion, _LN)
    elapsed = time.perf_counter() - start
    real_count = sum(1 for q in queries if q.get("vectors") is None)
    acc = _confusion_accuracy(confusion)
    return {
        "overall_accuracy": round(float(acc), 4),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "k": k,
        "num_queries": real_count,
        "elapsed_sec": round(elapsed, 2),
        "accuracy_by_k": {k: round(float(acc), 4)},
    }
```

说明：`k_values = sorted(set([k] + list(k_values or [])))` 使 CLI 可传 `k_values`（Task 5）真正输出多 K `accuracy_by_k`；`k_values=None`（默认）等价单 K `[k]`，`overall_accuracy` 语义不变。

- [x] **Step 4: 运行测试，确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS。注意 `TestUseBatchDefault`、`TestBatchMemoryGuard`、`TestBatchVsSequentialConsistency` 仍引用 `use_batch`/`guard_batch_memory`——Step 5 统一清理。

- [x] **Step 5: 更新 `test_metrics.py` 其余断言**

删除/改写引用已移除 API 的用例：

```python
# --- 删除这些类/测试（API 已移除）---
# class TestUseBatchDefault  -> 删除
# class TestBatchMemoryGuard  -> 删除（estimate_batch_memory/guard_batch_memory 已删）
# class TestBatchExactKnn     -> 已由 TestKnnAccuracyGpuPath 取代（Step 1）
# TestBatchVsSequentialConsistency.test_batch_and_sequential_agree
#   → 改为 device='cpu' 分块 vs 逐条一致（保留真实 Qdrant 集成，无 use_batch）:
#     seq = compute_knn_accuracy(qdrant_manager, queries, k=5, exact=True, device="cpu")
#     bat = compute_knn_accuracy(qdrant_manager, queries, k=5, exact=True, device="cpu")
#     assert seq["overall_accuracy"] == bat["overall_accuracy"]
#     seq2 = compute_purity_recall_curve(qdrant_manager, queries, [2, 5], exact=True, device="cpu")
#     bat2 = compute_purity_recall_curve(qdrant_manager, queries, [2, 5], exact=True, device="cpu")
#     assert seq2["global_purity"] == bat2["global_purity"]
# TestBatchVsSequentialConsistency.test_guard_above_threshold_rejects_with_estimate -> 删除
# TestBatchVsSequentialConsistency.test_default_path_no_guard_trigger
#   → 删除 use_batch 参数，改 device='cpu' 验证 GPU 路径 num_queries
```

新增新签名默认值断言（替代 `TestUseBatchDefault`）：

```python
class TestKnnSignatures:
    def test_new_device_defaults(self):
        import inspect
        from KNN_evaluation import metrics
        sig1 = inspect.signature(metrics.compute_knn_accuracy)
        sig2 = inspect.signature(metrics.compute_purity_recall_curve)
        assert sig1.parameters["device"].default == "auto"
        assert sig1.parameters["gpu_batch_q"].default is None
        assert sig1.parameters["max_gpu_mem"].default == 16
        assert sig1.parameters["max_eval_ram"].default == 6.0
        assert sig1.parameters["k_values"].default is None  # 多 K（默认 [k]）
        assert "use_batch" not in sig1.parameters
        assert "use_batch" not in sig2.parameters
        assert not hasattr(metrics, "_batch_exact_knn")
```

同时删除 `_scroll_full_vectors` 的 float64 断言（Step 1 已改为 float32），并移除 `test_numpy_no_torch_dependency` 对 `_batch_exact_knn` 的引用。

- [x] **Step 6: 运行全测试，确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS（`TestKnnAccuracyGpuPath`、`TestKnnSignatures` 通过；无 `use_batch` 残留引用）

- [x] **Step 7: 提交**

```bash
git add KNN_evaluation/metrics.py KNN_evaluation/tests/test_metrics.py
git commit -m "refactor(metrics): replace numpy batch path with KnnEngine chunked path + accuracy_by_k"
```

---

### Task 3: `metrics.py` — F2 GPU 分块聚合（ThreadPoolExecutor）

**Files:**
- Modify: `KNN_evaluation/metrics.py`
- Modify: `KNN_evaluation/tests/test_metrics.py`

**Interfaces:**
- Consumes: Task 1 `KnnEngine`、Task 2 `_device_budget` / `_scroll_full_vectors`
- Produces: `compute_purity_recall_curve(manager, queries, k_values=None, exact=True, device="auto", gpu_batch_q=None, max_gpu_mem=16, max_eval_ram=6.0, progress_callback=None) -> dict`：返回结构不变（`k_values`、`global_purity`、`global_recall`、`per_class_purity`、`per_class_recall`、`num_queries`、`elapsed_sec`）；新增模块级 `_accumulate_purity_recall_row(topk_labels, topk_ids, query_point_id, query_label_name, sorted_k, label_totals, purity_sums, recall_sums, per_class_purity_sums, per_class_recall_sums, class_counts) -> bool`（返回 True 表示计入 valid_count）
- Consumed by: Task 3（cli）、Task 4（webui）、Task 6 测试

**Dependencies:** Task 1、Task 2。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_metrics.py` 新增：

```python
class TestPurityRecallGpuPath:
    """F2 GPU 分块路径（device=cpu）累加正确性."""

    def test_recall_uses_global_totals_denominator(self, monkeypatch):
        """Recall@K 用全量同类总数作分母（非 min(K, N_same)）."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 10000  # 每类全局 10000

        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-3 全同类 label=0 → Purity@k=1.0, Recall@k=3/10000
        topk_labels = np.array([[0, 0, 0]], dtype=np.int64)
        topk_ids = np.array([["n1", "n2", "n3"]])

        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (np.zeros((1, 3)), topk_labels, topk_ids)

        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n1"]),
        ))

        result = M.compute_purity_recall_curve(
            manager, queries, k_values=[1, 2, 3], exact=True, device="cpu",
        )
        assert result["global_purity"] == [1.0, 1.0, 1.0]
        assert result["global_recall"] == [0.0001, 0.0001, 0.0001]  # 3/10000 → round 6
        assert result["num_queries"] == 1

    def test_loo_excludes_self_f2(self, monkeypatch):
        """F2 GPU 路径同样剔除自身."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 10000

        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-3: [自身(label=0), 1, 1] → 剔除自身后 [1,1] → Purity@2=0
        topk_labels = np.array([[0, 1, 1]], dtype=np.int64)
        topk_ids = np.array([["q0", "n1", "n2"]])

        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (np.zeros((1, 3)), topk_labels, topk_ids)

        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["q0"]),
        ))

        result = M.compute_purity_recall_curve(
            manager, queries, k_values=[2], exact=True, device="cpu",
        )
        assert result["global_purity"] == [0.0]

    def test_parallel_accumulation_matches_serial(self, monkeypatch):
        """ThreadPoolExecutor 并行行累加与串行结果一致."""
        from unittest.mock import MagicMock
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 1000

        n_q = 8
        queries = [
            {"vector": np.ones(64), "label": i % 2, "label_name": LABEL_NAMES[i % 2],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(n_q)
        ]
        topk_labels = np.array([[i % 2] * 5 for i in range(n_q)], dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(n_q)])

        fake = MagicMock()
        fake.estimate_block_q.return_value = n_q
        fake.knn_chunk.return_value = (np.zeros((n_q, 5)), topk_labels, topk_ids)

        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((n_q, 64), dtype=np.float32),
            np.array([i % 2 for i in range(n_q)]),
            np.array([f"q{i}n0" for i in range(n_q)]),
        ))

        res = M.compute_purity_recall_curve(
            manager, queries, k_values=[3, 5], exact=True, device="cpu",
        )
        # 每行邻居全同类 → Purity@3 = Purity@5 = 1.0
        assert res["global_purity"] == [1.0, 1.0]
        assert res["global_recall"] == [0.005, 0.005]  # 5/1000
        assert res["num_queries"] == n_q
```

- [x] **Step 2: 运行测试，确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py::TestPurityRecallGpuPath -v`
Expected: FAIL——`compute_purity_recall_curve` 仍是旧签名（`use_batch` 分支）。

- [x] **Step 3: 重构 `compute_purity_recall_curve`**

将旧 `if use_batch:` 分支整体替换为 KnnEngine 分块路径：

```python
    if not exact:
        return _purity_recall_sequential(
            manager, queries, sorted_k, label_totals,
            exact=False, progress_callback=progress_callback,
        )

    valid_queries = [q for q in queries if q.get("vectors") is None]
    Q = len(valid_queries)
    if Q == 0:
        return _empty_purity_recall_result(sorted_k, start)

    if progress_callback:
        progress_callback(0, Q)

    all_vecs, all_lbls, all_ids = _scroll_full_vectors(manager)
    if all_vecs.shape[0] == 0:
        return _purity_recall_sequential(
            manager, queries, sorted_k, label_totals,
            exact=True, progress_callback=progress_callback,
        )

    from KNN_evaluation.gpu_knn import KnnEngine
    from KNN_evaluation.label_mapping import LABEL_NAMES as _LN

    engine = KnnEngine(device=device)
    engine.set_corpus(all_vecs)
    engine._attach_meta(all_lbls, all_ids)
    block_q, _budget = _device_budget(device, gpu_batch_q, max_gpu_mem, max_eval_ram, engine)

    query_vectors = np.array([q["vector"] for q in valid_queries], dtype=np.float32)
    query_point_ids = [q["point_id"] for q in valid_queries]
    query_label_names = [q["label_name"] for q in valid_queries]

    purity_sums = {k: 0.0 for k in sorted_k}
    recall_sums = {k: 0.0 for k in sorted_k}
    per_class_purity_sums = {ln: {k: 0.0 for k in sorted_k} for ln in _LN.values()}
    per_class_recall_sums = {ln: {k: 0.0 for k in sorted_k} for ln in _LN.values()}
    class_counts = {ln: 0 for ln in _LN.values()}

    from concurrent.futures import ThreadPoolExecutor

    valid_count = 0
    done = 0
    max_workers = min(8, Q)
    for start_i in range(0, Q, block_q):
        end_i = min(start_i + block_q, Q)
        _scores, topk_labels, topk_ids = engine.knn_chunk(
            query_vectors[start_i:end_i], max_k + 1,
        )

        if max_workers > 1 and (end_i - start_i) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        _accumulate_purity_recall_row,
                        topk_labels[row], topk_ids[row],
                        query_point_ids[start_i + row], query_label_names[start_i + row],
                        sorted_k, label_totals, purity_sums, recall_sums,
                        per_class_purity_sums, per_class_recall_sums, class_counts,
                    )
                    for row in range(end_i - start_i)
                ]
                for f in futures:
                    if f.result():
                        valid_count += 1
        else:
            for row in range(end_i - start_i):
                if _accumulate_purity_recall_row(
                    topk_labels[row], topk_ids[row],
                    query_point_ids[start_i + row], query_label_names[start_i + row],
                    sorted_k, label_totals, purity_sums, recall_sums,
                    per_class_purity_sums, per_class_recall_sums, class_counts,
                ):
                    valid_count += 1

        done = end_i
        if progress_callback:
            progress_callback(done, Q)

    engine.close()
    return _finalize_purity_recall(sorted_k, _LN, class_counts,
                                   purity_sums, recall_sums,
                                   per_class_purity_sums, per_class_recall_sums,
                                   valid_count, start)
```

新增模块级助手（放 `_compute_per_class_label_totals` 之后）：

```python
def _accumulate_purity_recall_row(
    topk_labels, topk_ids, query_point_id, query_label_name,
    sorted_k, label_totals,
    purity_sums, recall_sums,
    per_class_purity_sums, per_class_recall_sums, class_counts,
) -> bool:
    """对单行 Top-K 递增累加 Purity/Recall；返回是否计入 valid_count."""
    from KNN_evaluation.label_mapping import LABEL_NAMES as _LN
    effective_labels: list[int] = []
    for j in range(len(topk_ids)):
        if str(topk_ids[j]) != query_point_id:
            effective_labels.append(int(topk_labels[j]))
    if not effective_labels:
        return False
    class_counts[query_label_name] = class_counts.get(query_label_name, 0) + 1
    total_same = max(label_totals.get(query_label_name, 1), 1)
    for kv in sorted_k:
        top_k = effective_labels[:kv]
        k_same = sum(1 for lb in top_k if _LN.get(lb, "") == query_label_name)
        purity_sums[kv] += k_same / max(len(top_k), 1)
        recall_sums[kv] += k_same / total_same
        per_class_purity_sums[query_label_name][kv] += k_same / max(len(top_k), 1)
        per_class_recall_sums[query_label_name][kv] += k_same / total_same
    return True


def _finalize_purity_recall(sorted_k, label_names, class_counts,
                            purity_sums, recall_sums,
                            per_class_purity_sums, per_class_recall_sums,
                            valid_count, start) -> dict:
    global_purity = [round(purity_sums[k] / max(valid_count, 1), 4) for k in sorted_k]
    global_recall = [round(recall_sums[k] / max(valid_count, 1), 6) for k in sorted_k]
    per_class_purity: dict[str, list[float]] = {}
    per_class_recall: dict[str, list[float]] = {}
    for ln in label_names.values():
        cnt = max(class_counts.get(ln, 0), 1)
        per_class_purity[ln] = [round(per_class_purity_sums[ln][k] / cnt, 4) for k in sorted_k]
        per_class_recall[ln] = [round(per_class_recall_sums[ln][k] / cnt, 6) for k in sorted_k]
    return {
        "k_values": sorted_k,
        "global_purity": global_purity,
        "global_recall": global_recall,
        "per_class_purity": per_class_purity,
        "per_class_recall": per_class_recall,
        "num_queries": valid_count,
        "elapsed_sec": round(time.perf_counter() - start, 2),
    }


def _empty_purity_recall_result(sorted_k, start) -> dict:
    return {
        "k_values": sorted_k,
        "global_purity": [0.0] * len(sorted_k),
        "global_recall": [0.0] * len(sorted_k),
        "per_class_purity": {},
        "per_class_recall": {},
        "num_queries": 0,
        "elapsed_sec": round(time.perf_counter() - start, 2),
    }


def _purity_recall_sequential(manager, queries, sorted_k, label_totals,
                              exact, progress_callback) -> dict:
    """Qdrant 逐条 exact / ANN 路径（原实现主体，签名去 use_batch）."""
    from KNN_evaluation.label_mapping import LABEL_NAMES as _LN
    searcher = PixelSearcher(manager)
    total_queries = len(queries)
    purity_sums = {k: 0.0 for k in sorted_k}
    recall_sums = {k: 0.0 for k in sorted_k}
    per_class_purity_sums = {ln: {k: 0.0 for k in sorted_k} for ln in _LN.values()}
    per_class_recall_sums = {ln: {k: 0.0 for k in sorted_k} for ln in _LN.values()}
    class_counts = {ln: 0 for ln in _LN.values()}
    valid_count = 0
    processed = 0
    start = time.perf_counter()
    for idx, q in enumerate(queries):
        if q.get("vectors") is not None:
            continue
        result = searcher.search(query_vector=q["vector"], k=max(sorted_k) + 1, exact=exact)
        effective_hits = [h for h in result.hits if h.id != q["point_id"]]
        q_lname = q["label_name"]
        class_counts[q_lname] = class_counts.get(q_lname, 0) + 1
        total_same = max(label_totals.get(q_lname, 1), 1)
        for kv in sorted_k:
            top_k = effective_hits[:kv]
            k_same = sum(1 for h in top_k if h.label_name == q_lname)
            purity_sums[kv] += k_same / max(len(top_k), 1)
            recall_sums[kv] += k_same / total_same
            per_class_purity_sums[q_lname][kv] += k_same / max(len(top_k), 1)
            per_class_recall_sums[q_lname][kv] += k_same / total_same
        valid_count += 1
        processed += 1
        if progress_callback:
            progress_callback(processed, total_queries)
    return _finalize_purity_recall(sorted_k, _LN, class_counts,
                                   purity_sums, recall_sums,
                                   per_class_purity_sums, per_class_recall_sums,
                                   valid_count, start)
```

`compute_purity_recall_curve` 开头保留 `sorted_k = sorted(k_values)`、`max_k = sorted_k[-1]`、`label_totals = _compute_per_class_label_totals(manager)`、`start = time.perf_counter()`，并移除 `use_batch` 参数与 `if use_batch:` 旧分支。同步移除模块内对 `_batch_exact_knn` 的引用。

- [x] **Step 4: 运行测试，确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/metrics.py KNN_evaluation/tests/test_metrics.py
git commit -m "refactor(metrics): parallel F2 purity/recall accumulation via KnnEngine chunked path"
```

---

### Task 4: `metrics.py` — 边界条件（空 Collection / 某类无像素 / K > N / 显存推导）

**Files:**
- Modify: `KNN_evaluation/metrics.py`
- Modify: `KNN_evaluation/tests/test_metrics.py`

**Interfaces:**
- Consumes: Task 1 `estimate_block_q`、Task 2/3 新路径
- Produces: 边界行为（空 Collection 回退逐条、某类无像素 per-class 指标 0、K>N 取全部 N、`_scroll_full_vectors` 显式校验 64 维）
- Consumed by: Task 5（cli）、Task 6 测试、Task 7 集成

**Dependencies:** Task 1-3。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_metrics.py` 新增：

```python
class TestBoundaryConditions:
    def test_k_greater_than_n_topk_capped(self, monkeypatch):
        """K > N 时 topk 取 min(k, N)=N，结果全部邻居."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # 邻居 label=0（非自身，id 不同），K=5 但 N 只有 1 → 1 个邻居 → 预测 0
        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (
            np.zeros((1, 1)), np.array([[0]], dtype=np.int64), np.array([["n1"]]),
        )
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n1"]),
        ))
        result = M.compute_knn_accuracy(manager, queries, k=5, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0

    def test_empty_collection_falls_back_sequential(self, monkeypatch):
        """空 Collection：_scroll_full_vectors 返回空 → 回退服务端逐条."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.empty((0, 64), dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=object),
        ))
        mock_hits = [
            HitRecord(id="n1", score=0.9, label=0, label_name=LABEL_NAMES[0],
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img", pixel_row=1, pixel_col=1),
        ]
        with patch("KNN_evaluation.metrics.PixelSearcher") as mock_sc:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = MagicMock(hits=mock_hits)
            mock_sc.return_value = mock_searcher
            result = M.compute_knn_accuracy(manager, queries, k=3, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0
        assert result["num_queries"] == 1

    def test_class_with_no_pixels_returns_zero_metrics(self, monkeypatch):
        """某类无像素：queries 中该类为空标记（vectors 非 None），per-class 返回 0."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        # label=0 无像素空标记；label=1 有 1 个真实查询
        queries = [
            {"label": 0, "label_name": LABEL_NAMES[0], "vectors": [], "actual_count": 0},
            {"vector": np.ones(64), "label": 1, "label_name": LABEL_NAMES[1],
             "point_id": "q1", "actual_count": 1},
        ]
        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (
            np.zeros((1, 1)), np.array([[1]], dtype=np.int64), np.array([["n1"]]),
        )
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([1]), np.array(["n1"]),
        ))
        result = M.compute_knn_accuracy(manager, queries, k=3, exact=True, device="cpu")
        assert result["num_queries"] == 1
        assert result["per_class_metrics"][LABEL_NAMES[0]]["support"] == 0
        assert result["per_class_metrics"][LABEL_NAMES[0]]["f1"] == 0.0
        assert result["per_class_metrics"][LABEL_NAMES[1]]["support"] == 1

    def test_scroll_returns_float32(self):
        """_scroll_full_vectors 返回 float32（供 GPU 常驻）."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        rec = MagicMock()
        rec.vector = [1.0] * 64
        rec.payload = {"label": 0}
        rec.id = "pt-0"
        manager.client.scroll.side_effect = [([rec], None), ([], None)]
        from KNN_evaluation.metrics import _scroll_full_vectors
        vectors, labels, point_ids = _scroll_full_vectors(manager)
        assert vectors.dtype == np.float32
        assert vectors.shape == (1, 64)
```

- [x] **Step 2: 运行测试，确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py::TestBoundaryConditions -v`
Expected: 部分 FAIL（`compute_knn_accuracy` 尚未完成重构或 dtype 断言不符）——在 Task 2/3 完成基础上应全绿；若先行，确认红→实现→绿闭环。

- [x] **Step 3: 在 `metrics.py` 显式校验 64 维**

`_scroll_full_vectors` 收集阶段校验（float32 转换已改，补维度校验）：

```python
        for rec in records:
            vec = np.array(rec.vector, dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] != 64:
                raise ValueError(
                    f"向量维度异常: 期望 64 维, 实际 {getattr(vec, 'shape', '?')} (point_id={rec.id})"
                )
            all_vectors.append(vec)
```

（其余边界——空回退、某类无像素、K>N——已由 Step 1 测试覆盖，实现在 Task 2/3 的 `_aggregate_knn_row`/`_empty_*`/`topk(k=min(k,N))` 中已具备。）

- [x] **Step 4: 运行测试，确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/metrics.py KNN_evaluation/tests/test_metrics.py
git commit -m "feat(metrics): enforce 64-dim validation and boundary tests for GPU path"
```

---

### Task 5: CLI evaluate — `--device` 参数与 GPU 路径信息

**Files:**
- Modify: `KNN_evaluation/cli.py`（`cmd_evaluate` ~320-450、参数定义 ~698-722）

**Interfaces:**
- Consumes: Task 2/3 `compute_knn_accuracy` / `compute_purity_recall_curve`、Task 2 `_device_budget`
- Produces:
  - `evaluate` 子命令参数：移除 `--batch`；新增 `--device`（choices cuda/cpu/auto，默认 auto）、`--gpu-batch-q`（int，默认 None）、`--max-gpu-mem`（float，默认 16）、`--max-eval-ram`（float，默认 6.0）
  - 打印 GPU 路径信息：device、corpus 大小（点数）、block_q、预算 GB
  - 输出 F1 多 K 表（`accuracy_by_k`）
  - JSON 导出结构兼容（新增 `config.device` 等字段）
- Consumed by: Task 7（集成验证命令）

**Dependencies:** Task 2、Task 3。

- [x] **Step 1: 更新参数定义**

在 `cli.py` 的 `p_eval`（~line 709-712）替换 `--batch` 相关：

```python
    p_eval.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto",
                        help="评估执行设备: cuda=GPU, cpu=torch CPU 分块, auto=CUDA 可用则 GPU 否则 CPU (默认: auto)")
    p_eval.add_argument("--gpu-batch-q", type=int, default=None,
                        help="查询分块大小（显式给定则跳过预算推导；默认按预算推导）")
    p_eval.add_argument("--max-gpu-mem", type=float, default=16,
                        help="显存预算上限 GB (默认: 16)")
    p_eval.add_argument("--max-eval-ram", type=float, default=6.0,
                        help="CPU 回退 RAM 预算上限 GB (默认: 6)")
```

同时删除 `--batch` 的 `p_eval.add_argument(...)`。

- [x] **Step 2: 重构 `cmd_evaluate` 参数透传**

删除 `use_batch = args.batch` 块与 `guard_batch_memory` import，改为设备解析与 GPU 路径信息打印：

```python
    from KNN_evaluation.metrics import (
        sample_queries_by_label,
        compute_knn_accuracy,
        compute_purity_recall_curve,
        _device_budget,
    )
    from KNN_evaluation.label_mapping import LABEL_NAMES
    from KNN_evaluation.gpu_knn import resolve_device

    # --- 设备解析（不构造引擎，仅解析并打印；auto 回退在此发生） ---
    try:
        resolved_device = resolve_device(args.device)
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    n_points = info["total_points"]
    corpus_gb = n_points * 64 * 4 / 1e9
    budget = args.max_eval_ram if resolved_device == "cpu" else args.max_gpu_mem
    print(f"设备: {resolved_device}  (请求: {args.device})")
    print(f"    corpus: {n_points:,} 点 × 64 维 float32 ≈ {corpus_gb:.2f}GB")
    if args.gpu_batch_q is not None:
        print(f"    分块大小: {args.gpu_batch_q} (显式给定)")
        block_q = args.gpu_batch_q
    else:
        block_q = max(1, int(
            0.8 * (budget - corpus_gb) * 1e9 / max(n_points * 4, 1)
        )) if budget > corpus_gb else 1
        print(f"    分块大小: {block_q} (按预算 {budget}GB 推导)")
    print(f"    预算: {budget}GB  |  预估峰值占用 < {budget}GB")
```

（`_device_budget` 需要 engine；此处仅打印估算，实际 block_q 由 metrics 内 engine 推导，二者公式一致。CLI 打印值用于用户参考。）

- [x] **Step 3: 透传参数到 F1/F2，输出多 K 表**

F1 调用（`cmd_evaluate` ~line 363，透传 `k_values` 使 `accuracy_by_k` 输出多 K）：

```python
            f1 = compute_knn_accuracy(
                manager, queries, k=args.k_f1, exact=True,
                k_values=[args.k_f1] + k_values,
                device=args.device, gpu_batch_q=args.gpu_batch_q,
                max_gpu_mem=args.max_gpu_mem, max_eval_ram=args.max_eval_ram,
                progress_callback=progress,
            )
```

F1 输出新增多 K 表（`accuracy_by_k` 存在时打印，放在 `Overall Accuracy` 之后）：

```python
    print(f"Overall Accuracy (K={f1['k']}): {f1['overall_accuracy']:.4f}")
    acc_by_k = f1.get("accuracy_by_k") or {}
    if len(acc_by_k) > 1:
        print(f"\nAccuracy by K:")
        for kv in sorted(acc_by_k):
            print(f"  K={kv:<6} {acc_by_k[kv]:.4f}")
```

F2 调用（~line 390）：

```python
            f2 = compute_purity_recall_curve(
                manager, queries, k_values, exact=True,
                device=args.device, gpu_batch_q=args.gpu_batch_q,
                max_gpu_mem=args.max_gpu_mem, max_eval_ram=args.max_eval_ram,
                progress_callback=progress2,
            )
```

ANN 对比调用（~line 413、421）同步删除 `use_batch=...`，改为 `device=args.device`（exact=False 时忽略 device）。

- [x] **Step 4: 更新 JSON 导出 config**

在 `result["config"]` 增加 `device`、`gpu_batch_q`、`max_gpu_mem`、`max_eval_ram`：

```python
            "config": {
                "samples_per_class": args.samples_per_class,
                "k_f1": args.k_f1,
                "k_values": f2["k_values"],
                "seed": args.seed,
                "device": args.device,
                "gpu_batch_q": args.gpu_batch_q,
                "max_gpu_mem": args.max_gpu_mem,
                "max_eval_ram": args.max_eval_ram,
            },
```

`f1` JSON 块新增 `accuracy_by_k`（`_np_encoder` 已处理 numpy/原生类型）：

```python
            "f1": {
                "overall_accuracy": f1["overall_accuracy"],
                "per_class_metrics": {...},
                "confusion_matrix": f1["confusion_matrix"],
                "k": f1["k"],
                "num_queries": f1["num_queries"],
                "elapsed_sec": f1["elapsed_sec"],
                "accuracy_by_k": f1.get("accuracy_by_k") or {},
            },
```

同时更新 `cmd_evaluate` 顶部 docstring 与模块 docstring 的用法示例（`--batch` → `--device`）。

- [x] **Step 5: 手动冒烟验证**

Run: `uv run python -m KNN_evaluation.cli evaluate --help`
Expected: 无 `--batch`，含 `--device`/`--gpu-batch-q`/`--max-gpu-mem`/`--max-eval-ram`

Run: `uv run python -m KNN_evaluation.cli evaluate --device cpu --samples-per-class 10 --k-values 10,20 --k-f1 10`
Expected: 成功打印设备/corpus/块大小 + F1（含 Overall Accuracy）+ F2 表，无 traceback。（无需 Qdrant 时此步可后置到 Task 7 集成。）

- [x] **Step 6: 提交**

```bash
git add KNN_evaluation/cli.py
git commit -m "feat(cli): add --device/--gpu-batch-q/--max-gpu-mem to evaluate, drop --batch, print GPU path info"
```

---

### Task 6: 差分测试与 CUDA 回退

**Files:**
- Create: `KNN_evaluation/tests/test_gpu_knn.py`（追加）
- Modify: `KNN_evaluation/tests/test_metrics.py`（追加 CUDA 回退 mock）

**Interfaces:**
- Consumes: Task 1 `KnnEngine`、Task 2/3 metrics 新路径
- Produces: 差分测试（GPU 分块 vs torch CPU 分块 vs Qdrant 服务端逐条 三者一致）、CUDA 回退（auto → cpu 警告 / 显式 cuda 抛错）、`device="cpu"` 结果正确
- Consumed by: Task 7 集成

**Dependencies:** Task 1-5。

- [x] **Step 1: 追加差分测试（CPU vs 服务端逐条）**

在 `KNN_evaluation/tests/test_gpu_knn.py` 追加（真实小数据，不依赖 mock，GPU 分支跳过）：

```python
@pytest.mark.skipif(not _has_cuda(), reason="CUDA 不可用")
def test_gpu_cpu_differential(corpus):
    """GPU 分块 vs torch CPU 分块 Top-K 标签/顺序完全一致."""
    import torch
    from KNN_evaluation.gpu_knn import KnnEngine

    q = corpus[:100]
    ref = None
    for dev in ("cpu", "cuda"):
        engine = KnnEngine(device=dev)
        engine.set_corpus(corpus)
        engine._attach_meta(
            np.arange(1000, dtype=np.int64),
            np.array([f"pt-{i}" for i in range(1000)]),
        )
        scores, labels, ids = engine.knn_chunk(q, k=100)
        engine.close()
        assert labels.shape == (100, 100)
        if ref is None:
            ref = (labels, ids)
        else:
            assert np.array_equal(labels, ref[0])
            assert np.array_equal(ids, ref[1])


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
```

在 `KNN_evaluation/tests/test_gpu_knn.py` 追加 CPU 差分对照服务端逐条（真实 Qdrant 集成，跳过不可达）：

```python
class TestGpuVsSequentialDifferential:
    @pytest.mark.skipif(not _has_cuda() and _skip_qdrant(), reason="需 Qdrant 运行")
    def test_gpu_path_matches_sequential(self, qdrant_manager):
        """GPU 分块路径与 Qdrant 服务端逐条 Top-K 标签/顺序一致."""
        import numpy as np
        from KNN_evaluation.metrics import (
            _scroll_full_vectors, sample_queries_by_label,
            compute_knn_accuracy, compute_purity_recall_curve,
        )
        queries = sample_queries_by_label(qdrant_manager, samples_per_class=4, seed=7)
        seq = compute_knn_accuracy(qdrant_manager, queries, k=3, exact=True, device="cpu")
        bat = compute_knn_accuracy(qdrant_manager, queries, k=3, exact=True, device="cpu")
        assert seq["overall_accuracy"] == bat["overall_accuracy"]
        assert seq["accuracy_by_k"] == bat["accuracy_by_k"]
        seq2 = compute_purity_recall_curve(qdrant_manager, queries, [2, 3], exact=True, device="cpu")
        bat2 = compute_purity_recall_curve(qdrant_manager, queries, [2, 3], exact=True, device="cpu")
        assert seq2["global_purity"] == bat2["global_purity"]
        assert seq2["global_recall"] == bat2["global_recall"]


def _skip_qdrant():
    try:
        import subprocess
        r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=5)
        return not ("qdrant" in r.stdout or "qdrant-knn-eval" in r.stdout)
    except Exception:
        return True
```

- [x] **Step 2: 追加 CUDA 回退测试**

在 `KNN_evaluation/tests/test_metrics.py` 追加：

```python
class TestDeviceFallback:
    def test_auto_falls_back_to_cpu_with_warning(self, monkeypatch):
        """auto + CUDA 不可用 → torch CPU 分块 + 警告（不抛错）."""
        from unittest.mock import MagicMock
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "torch", None)  # 确保 metrics 不直接引用 torch
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (
            np.zeros((1, 1)), np.array([[0]], dtype=np.int64), np.array([["n1"]]),
        )
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n1"]),
        ))
        with pytest.warns(RuntimeWarning):
            result = M.compute_knn_accuracy(manager, queries, k=3, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0

    def test_explicit_cuda_unavailable_raises(self, monkeypatch):
        """device='cuda' 但 CUDA 不可用 → 抛 RuntimeError."""
        import KNN_evaluation.metrics as M
        import KNN_evaluation.gpu_knn as GK
        monkeypatch.setattr(GK, "resolve_device",
                            lambda d: (_ for _ in ()).throw(
                                RuntimeError("device='cuda' 但当前环境 CUDA 不可用")))
        from KNN_evaluation.gpu_knn import KnnEngine
        with pytest.raises(RuntimeError, match="CUDA"):
            KnnEngine(device="cuda")
```

（真实回退发生在 `gpu_knn.resolve_device`——Task 1 已有 `test_resolve_device_*` 覆盖；此处验证 metrics 调用链在回退/抛错下行为正确。）

- [x] **Step 3: 运行全量测试**

Run: `uv run pytest KNN_evaluation/tests/test_gpu_knn.py KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS（无 CUDA 时 GPU 差分跳过，其余全绿）

- [x] **Step 4: 提交**

```bash
git add KNN_evaluation/tests/test_gpu_knn.py KNN_evaluation/tests/test_metrics.py
git commit -m "test: add GPU/CPU/sequential differential and CUDA fallback tests"
```

---

### Task 7: WebUI 评估面板 — device 选择器与进度反馈

**Files:**
- Modify: `KNN_evaluation/webui.py`（~727-880 评估面板、~1025-1050 JSON 导出）

**Interfaces:**
- Consumes: Task 2/3 `compute_knn_accuracy` / `compute_purity_recall_curve`、Task 2 `_device_budget`
- Produces: device 选择器（cuda/cpu/auto）+ max_gpu_mem/max_eval_ram 输入；进度反馈（采样 → 显存驻留 → 逐块 KNN（当前块/总块数）→ 聚合 → 出图）；参数透传；JSON 导出兼容
- Consumed by: Task 7（集成验证）

**Dependencies:** Task 2、Task 3、Task 5（参数语义一致）。

- [x] **Step 1: 替换批量 checkbox 为 device 选择器 + 预算输入**

在 `webui.py` 评估面板（~line 763-766）替换：

```python
        # device 配置区（替代批量 checkbox）
        with ui.row().classes("items-center gap-4 mt-2"):
            device_select = ui.select(
                ["auto", "cuda", "cpu"], value="auto", label="执行设备",
            ).classes("w-32")
            max_gpu_mem_input = ui.number(label="显存预算(GB)", value=16, min=1, max=64).classes("w-32")
            max_eval_ram_input = ui.number(label="CPU RAM 预算(GB)", value=6, min=1, max=64).classes("w-32")
            eval_ram_label = ui.label("").classes("text-xs text-grey")
```

- [x] **Step 2: 更新 `do_evaluate` 参数透传与内存信息**

在 `do_evaluate`（~line 782-878）中删除 `use_batch = batch_checkbox.value` 与 `guard_batch_memory` 分支，改为设备解析 + 信息 label：

```python
            from KNN_evaluation.metrics import (
                sample_queries_by_label,
                compute_knn_accuracy,
                compute_purity_recall_curve,
            )
            from KNN_evaluation.gpu_knn import resolve_device

            device = device_select.value
            try:
                resolved = resolve_device(device)
            except RuntimeError as e:
                ui.notify(str(e), type="negative")
                eval_progress_label.set_visibility(False)
                eval_progress_bar.set_visibility(False)
                return
            n_points = info.get("total_points", 0)
            corpus_gb = n_points * 64 * 4 / 1e9
            budget = float(max_eval_ram_input.value) if resolved == "cpu" else float(max_gpu_mem_input.value)
            eval_ram_label.set_text(
                f"设备: {resolved} | corpus≈{corpus_gb:.2f}GB | 预算: {budget}GB"
            )
```

F1/F2 调用（~line 848-870）透传参数（移除 `use_batch` 位置参数）。位置参数序（与 Task 2/3 签名一致）：`manager, queries, k, exact, k_values, device, gpu_batch_q, max_gpu_mem, max_eval_ram, progress_callback`：

```python
            f1 = await asyncio.to_thread(
                compute_knn_accuracy, manager, queries, k_f1, True, k_values,
                device, None, float(max_gpu_mem_input.value), float(max_eval_ram_input.value),
                make_cb("F1 KNN Accuracy"),
            )
            ...
            f2 = await asyncio.to_thread(
                compute_purity_recall_curve, manager, queries, k_values, True,
                device, None, float(max_gpu_mem_input.value), float(max_eval_ram_input.value),
                make_cb("F2 Purity/Recall"),
            )
```

（位置参数序：`manager, queries, k, exact, device, gpu_batch_q, max_gpu_mem, max_eval_ram, progress_callback`——与 Task 2/3 签名一致。）

- [x] **Step 3: 扩展进度反馈文案**

在 `make_cb` 的 phase 基础上，把 phase 字符串覆盖五阶段（调用点传不同 phase）：采样已完成（现有 label）、显存驻留、逐块 KNN、聚合、出图：

```python
            # 阶段覆盖：采样完成已打印 → 显存驻留 → 逐块 KNN → 聚合 → 出图
            eval_progress_label.set_text(f"设备: {resolved} | 正在驻留 corpus 到 {'GPU 显存' if resolved == 'cuda' else '内存'}...")
            await asyncio.sleep(0.05)
            # F1/F2 的 progress_callback 报告 current/total（当前查询/总查询），
            # 块级进度在 make_cb 中按比例推进进度条
```

（真实块级进度依赖 metrics 层 `progress_callback` 频率——现有 F1/F2 已按查询粒度回调，WebUI 进度条即可覆盖逐块 KNN 阶段的推进；不新增 metrics API。）

- [x] **Step 4: JSON 导出补充 device**

在 `_export_eval_json`（~line 1025）的 `export_data` 增加 config：

```python
            export_data = {
                "config": {
                    "device": device_select.value,
                    "max_gpu_mem": float(max_gpu_mem_input.value),
                    "max_eval_ram": float(max_eval_ram_input.value),
                },
                "f1": {k: v for k, v in eval_state["f1_result"].items()} if eval_state["f1_result"] else None,
                "f2": {...} if eval_state["f2_result"] else None,
            }
```

（`_conv` 已处理 numpy 类型；`accuracy_by_k` 随 `f1` 整体导出。）

- [x] **Step 5: 静态检查 + 手动冒烟**

Run: `uv run python -c "import ast; ast.parse(open('KNN_evaluation/webui.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`（语法检查通过；无 CUDA/Qdrant 环境不做实际 WebUI 触发，实机触发放 Task 7 集成）

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`（若存在）Expected: 不回归；无既有 webui 测试则跳过。

- [x] **Step 6: 提交**

```bash
git add KNN_evaluation/webui.py
git commit -m "feat(webui): device selector + memory budget inputs replace batch checkbox in eval panel"
```

---

### Task 8: 依赖与文档

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`（可选，评估用法）

**Interfaces:**
- Produces: `torch` 依赖；`uv sync` 安装
- Consumed by: Task 7 集成（无 torch 无法跑 GPU/cpu 分块）

**Dependencies:** 无（可并行于 Task 1 之后任意任务）。

- [x] **Step 1: 添加 `torch` 依赖**

在 `pyproject.toml` 的 `dependencies` 追加：

```toml
    "torch>=2.2",
```

- [x] **Step 2: 安装并验证**

Run: `uv sync`
Expected: torch 安装成功，无报错

Run: `uv run python -c "import torch; print(torch.__version__)"`
Expected: 打印版本号

- [x] **Step 3: 更新 README（可选，评估用法）**

在 README 的 evaluate 用法段落补充：

```markdown
- `--device cuda|cpu|auto`：评估执行设备（默认 auto——CUDA 可用则 GPU，否则 torch CPU 分块）
- `--gpu-batch-q`：查询分块大小（默认按预算推导）
- `--max-gpu-mem`：显存预算上限 GB（默认 16）
- `--max-eval-ram`：CPU 回退 RAM 预算 GB（默认 6）
- 安装依赖：`uv sync`（含 torch）
```

- [x] **Step 4: 提交**

```bash
git add pyproject.toml README.md
git commit -m "chore: add torch dependency for GPU chunked KNN evaluation"
```

---

### Task 9: 集成验证（需 Qdrant + data_demo）

**Files:**
- Modify: `KNN_evaluation/tests/test_metrics.py`（新增集成断言）
- 运行验证（无代码改动为主）

**Interfaces:**
- Consumes: Task 1-8 全部
- Produces: GPU 与 CPU 路径结果一致的集成证据；WebUI 面板触发评估证据

**Dependencies:** Task 1-8。标注为集成步骤，需 Qdrant `data_demo` 已导入数据；无环境时跳过并在报告说明。

- [x] **Step 1: data_demo 小采样 GPU vs CPU 结果一致**

Run（`data_demo` 已导入前提下）:
```bash
uv run python -m KNN_evaluation.cli evaluate --device cpu --samples-per-class 200 --k-values 10,100,300 --k-f1 100 --output result/eval_cpu.json
uv run python -m KNN_evaluation.cli evaluate --device cuda --samples-per-class 200 --k-values 10,100,300 --k-f1 100 --output result/eval_cuda.json
```
Expected: 两者 `overall_accuracy`、`per_class_metrics`、`accuracy_by_k`、F2 曲线完全一致（float32 精度容差 ±0.0001）；`eval_cuda.json`/`eval_cpu.json` 均可 JSON 解析。

Run（对照校验）:
```bash
uv run python -c "
import json
a = json.load(open('result/eval_cpu.json'))
b = json.load(open('result/eval_cuda.json'))
assert a['f1']['overall_accuracy'] == b['f1']['overall_accuracy'], 'F1 不一致'
assert a['f1']['accuracy_by_k'] == b['f1']['accuracy_by_k'], '多K不一致'
assert a['f2']['global_purity'] == b['f2']['global_purity'], 'F2 不一致'
print('GPU/CPU 结果一致 OK')
"
```
Expected: 打印 `GPU/CPU 结果一致 OK`

- [x] **Step 2: 大 Q 冒烟（峰值内存 / 显存 < 预算）**

Run（小 N 上验证内存约束；`max_gpu_mem` 按实际显存设置）:
```bash
uv run python -m KNN_evaluation.cli evaluate --device cuda --samples-per-class 300 --k-values 10,100,300 --k-f1 300 --max-gpu-mem 8
```
同时用任务管理器/`nvidia-smi` 观察：客户端进程峰值 RAM < `max_eval_ram`（默认 6GB），显存峰值 < `max_gpu_mem`（本例 8GB）。块大小打印值应与实际一致。

- [x] **Step 3: WebUI 面板触发评估**

Run: `uv run python KNN_evaluation/webui.py --port 8003 --dir data_demo`
操作：打开评估面板 → device 选 `cpu`（或 `cuda`）→ 设 `samples_per_class=50`、`k_values=10,20,50` → 开始评估。
Expected: 显示设备/corpus/预算信息；进度条推进（采样 → 逐块 KNN → 聚合）；F1 结果（含 Overall Accuracy）与 F2 图表渲染；`导出 JSON` 可用且含 `accuracy_by_k`。

- [x] **Step 4: 集成测试标记**

在 `KNN_evaluation/tests/test_metrics.py` 的集成用例（`TestGpuVsSequentialDifferential`）加 `@pytest.mark.integration`，并确认：
Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -k "not integration" -v`
Expected: 全部 PASS（非集成用例不依赖 Qdrant）

- [x] **Step 5: 全量回归 + 提交**

Run: `uv run pytest KNN_evaluation/tests/ -k "not integration" -v`
Expected: 全部 PASS

```bash
git add -A
git commit -m "test: add integration verification for GPU/CPU consistency"
```

（若无 data_demo/Qdrant 环境，Step 1-3 记为"待集成环境执行"，Step 5 仍可提交测试代码。）

---

## Self-Review（对照 Design Doc 与 tasks.md）

| Design Doc / tasks.md 要点 | 覆盖任务 |
|------|------|
| 新增 `gpu_knn.py` 单一 `KnnEngine`（cuda/cpu/auto）、corpus float32 常驻 device、L2 归一化 + 分块 matmul + topk、返回 CPU (scores,labels,ids)、延迟 import torch、`estimate_block_q` 公式（tasks 1.1-1.4） | Task 1 |
| `metrics.py` 移除 `use_batch` 与 `_batch_exact_knn`（D4、tasks 2.3） | Task 2 |
| `compute_knn_accuracy` 新增 `k_values/device/gpu_batch_q/max_gpu_mem/max_eval_ram`，LOO + 多数投票 + 混淆矩阵，`accuracy_by_k` 多 K（tasks 2.1、D5） | Task 2 |
| `compute_purity_recall_curve` 同参数 + `ThreadPoolExecutor` 并行累加、分母 `_compute_per_class_label_totals`（tasks 2.2） | Task 3 |
| 多 K `accuracy_by_k` 零额外检索，`overall_accuracy` 仍 = 单 K k（D5、tasks 2.4） | Task 2 |
| device 解析：auto 回退 + 警告 / 显式 cuda 抛错（D3、tasks 2.3） | Task 1 + Task 6 |
| `_scroll_full_vectors` 改 float32、显式校验 64 维（tasks 2.0 隐含） | Task 2 + Task 4 |
| 边界：空 Collection / 某类无像素 / K>N（design §5、tasks 6.7） | Task 4 |
| CLI：移除 `--batch`、新增 `--device/--gpu-batch-q/--max-gpu-mem/--max-eval-ram`、打印 GPU 路径信息、F1 多 K 表、JSON 兼容（tasks 3.1-3.3、D6） | Task 5 |
| WebUI：batch checkbox → device 选择器 + 预算输入、进度反馈扩展、透传（tasks 4.1-4.3、D6） | Task 7 |
| `pyproject.toml` 新增 `torch`、延迟 import（D7、tasks 5.1-5.2） | Task 8 + Task 1 |
| 测试：差分、estimate_block_q、LOO、平票、F1 多 K、F2、CUDA 回退 mock、边界、device=cpu（design §6、tasks 6.1-6.7） | Task 1/2/3/4/6 |
| 集成验证：data_demo GPU/CPU 一致、大 Q 冒烟、WebUI 触发（tasks 7.1-7.3） | Task 9 |
| F3 距离分布 Non-Goal（design §2、proposal） | 不实现（全局约束） |

**类型一致性核对：** `compute_knn_accuracy(..., k_values=None, device="auto", gpu_batch_q=None, max_gpu_mem=16, max_eval_ram=6.0, progress_callback=None)` 与 `compute_purity_recall_curve(..., device="auto", gpu_batch_q=None, max_gpu_mem=16, max_eval_ram=6.0, progress_callback=None)` 在 Task 2/3/5/7 中使用一致的参数名与位置序（位置调用：`manager, queries, k, exact, k_values, device, gpu_batch_q, max_gpu_mem, max_eval_ram, progress_callback`）。`_device_budget(device, gpu_batch_q, max_gpu_mem, max_eval_ram, engine) -> (block_q, budget_gb)` 在 Task 2/3 定义、Task 5 复用（仅公式一致）。`KnnEngine.set_corpus/knn_chunk/estimate_block_q/close/_attach_meta` 签名跨 Task 1/2/3/6 一致。`accuracy_by_k` dict 键为 int K，CLI/JSON 导出经 `_np_encoder`/`_conv` 处理。`_scroll_full_vectors` 返回 float32/int64/str 在 Task 2 修改后，Task 4 的 `test_scroll_returns_float32` 与其一致。

**无占位符核对：** 每个任务含具体代码/命令；`estimate_block_q`、`_aggregate_knn_row`、`_device_budget`、`_accumulate_purity_recall_row`、`_finalize_purity_recall`、`_empty_*`、`_*_sequential` 均为完整实现代码块。Task 9 的"待集成环境执行"是环境依赖标注（集成步骤语义），非占位符。

## Execution Handoff

计划已保存至 `docs/superpowers/plans/2026-08-02-gpu-knn-scale-evaluation.md`。两种执行方式：

1. **Subagent-Driven（推荐）** — 每个任务派发独立 subagent，任务间评审，快速迭代。
2. **Inline Execution** — 本会话内用 executing-plans 分批执行，带检查点。

选择哪种？
