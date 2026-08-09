---
change: embedding-quality-metrics
design-doc: docs/superpowers/specs/2026-07-31-embedding-quality-metrics-design.md
base-ref: 3e1061d3117eb5e3e1903b299a1eeca62ce825c6
---

# Embedding 质量评估模块实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 为 Qdrant KNN 像素分类系统新增 embedding 质量评估能力，实现 F1（KNN 分类准确率）、F2（邻居纯度/Recall@K 曲线）、F3（Intra/Inter-class 距离分布）三类指标。

**Architecture:** `metrics.py` 为四个纯函数的集合，`visualization.py` 负责 matplotlib 图表生成，`cli.py` 和 `webui.py` 各自编排调用。指标计算复用现有 `PixelSearcher.search()` 接口和 `QdrantManager.client.scroll()/count()`。进度通过 `Callable[[int, int], None]` 回调传递。

**Tech Stack:** Python 3.11+, numpy, qdrant-client, matplotlib, NiceGUI, tqdm, pytest

## Global Constraints

- 所有公开函数输出纯数据结构（dict / np.ndarray），调用方负责格式化/可视化
- 进度回调签名：`Callable[[int, int], None] | None`，CLI 端传 tqdm 包装，WebUI 端传 NiceGUI label 更新
- 默认 exact=True（暴力精确搜索），评估 embedding 质量而非 HNSW 近似
- F1 和 F2 复用同一查询像素集（`sample_queries_by_label` 返回），F3 独立采样
- 中文字符使用 UTF-8，matplotlib 配置中文字体
- 所有 np.ndarray 在 JSON 序列化时递归转换为 Python 原生类型
- 错误信息使用中文

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `KNN_evaluation/metrics.py` | CREATE | 四个纯函数：采样、F1 分类、F2 纯度/召回、F3 距离分布 |
| `KNN_evaluation/visualization.py` | CREATE | 三个 matplotlib 函数：混淆矩阵热力图、Purity/Recall 曲线、距离箱线图 |
| `KNN_evaluation/tests/test_metrics.py` | CREATE | 单元测试（mock）+ 边界测试 + 集成测试 |
| `KNN_evaluation/cli.py` | MODIFY | 新增 `evaluate` 子命令和 `cmd_evaluate` 函数 |
| `KNN_evaluation/webui.py` | MODIFY | 新增"评估面板"expansion，异步执行 + 结果展示 |

---

### Task 1: metrics.py — 采样与 F1 KNN 分类准确率

**Files:**
- Create: `KNN_evaluation/metrics.py`
- Test: `KNN_evaluation/tests/test_metrics.py`（部分）

**Interfaces:**
- Produces: `sample_queries_by_label(manager, samples_per_class, seed) -> list[dict]`
- Produces: `compute_knn_accuracy(manager, queries, k, exact, progress_callback) -> dict`
- Consumes: `QdrantManager.client.scroll()`, `PixelSearcher.search()`, `HitRecord`, `SearchResult`
- Imports: `numpy`, `qdrant_client.models`, `random`, `time`, `collections`
- Internal: `_resolve_tie(votes, hits) -> int`

- [x] **Step 1: 创建文件骨架**

在 `KNN_evaluation/metrics.py` 中写入模块文档、导入和类型定义：

```python
"""KNN Embedding 质量评估指标模块.

提供分层采样、KNN 分类准确率 (F1)、邻居纯度/Recall@K (F2)、
Intra/Inter-class 距离分布 (F3) 四类评估函数。
"""
import collections
import random
import time
from typing import Callable

import numpy as np
from qdrant_client import models

from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.searcher import PixelSearcher, HitRecord
from KNN_evaluation.label_mapping import LABEL_NAMES

ProgressCallback = Callable[[int, int], None] | None
```

- [x] **Step 2: 实现 `_resolve_tie` 内部函数**

```python
def _resolve_tie(votes: list[int], hits: list[HitRecord]) -> int:
    """递减 K 打破平票。

    按 K, K-1, K-3, K-5, K-7, K-9 依次重新计票，
    若至 K<=0 仍未打破，回退到最近邻居的标签。
    """
    for step in (0, 1, 3, 5, 7, 9):
        k_sub = len(votes) - step
        if k_sub <= 0:
            return hits[0].label
        counter = collections.Counter(votes[:k_sub])
        max_count = max(counter.values())
        winners = [l for l, c in counter.items() if c == max_count]
        if len(winners) == 1:
            return winners[0]
    return hits[0].label
```

- [x] **Step 3: 实现 `sample_queries_by_label`**

```python
def sample_queries_by_label(
    manager: QdrantManager,
    samples_per_class: int = 500,
    seed: int = 42,
) -> list[dict]:
    """按标签分层随机采样查询像素。

    Args:
        manager: QdrantManager 实例。
        samples_per_class: 每类目标采样数。
        seed: 随机种子，保证可复现。

    Returns:
        查询像素列表，每项 dict 含 vector, label, label_name,
        image_id, pixel_row, pixel_col, point_id, actual_count。

    Raises:
        ConnectionError: Qdrant 不可达。
        ValueError: Collection 为空。
    """
    # 检查连接
    if not manager.health_check():
        raise ConnectionError(f"Qdrant 不可达: {manager.url}")

    if not manager.collection_exists():
        raise ValueError(f"Collection '{manager.collection_name}' 不存在，请先执行 import")

    info = manager.collection_info()
    if info.get("total_points", 0) == 0:
        raise ValueError("Collection 为空，请先导入数据")

    rng = random.Random(seed)
    queries: list[dict] = []

    for label_id, label_name in LABEL_NAMES.items():
        # 逐类 scroll 获取像素（含向量）
        records, _ = manager.client.scroll(
            collection_name=manager.collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="label", match=models.MatchValue(value=label_id),
                )],
            ),
            limit=samples_per_class * 3,  # 多取一些用于随机采样
            with_payload=True,
            with_vectors=True,
        )

        actual_count = min(len(records), samples_per_class)
        if actual_count == 0:
            queries.append({
                "label": label_id,
                "label_name": label_name,
                "actual_count": 0,
                "vectors": [],
            })
            continue

        # 随机采样
        selected = rng.sample(records, actual_count)
        for rec in selected:
            p = rec.payload or {}
            queries.append({
                "vector": np.array(rec.vector, dtype=np.float64),
                "label": label_id,
                "label_name": label_name,
                "image_id": str(p.get("image_id", "")),
                "pixel_row": int(p.get("pixel_row", -1)),
                "pixel_col": int(p.get("pixel_col", -1)),
                "point_id": str(rec.id),
                "actual_count": actual_count,
            })

    return queries
```

- [x] **Step 4: 实现 `compute_knn_accuracy`**

```python
def compute_knn_accuracy(
    manager: QdrantManager,
    queries: list[dict],
    k: int = 10,
    exact: bool = True,
    progress_callback: ProgressCallback = None,
) -> dict:
    """计算 KNN 分类准确率（Leave-One-Out）+ Per-class F1。

    Args:
        manager: QdrantManager 实例。
        queries: sample_queries_by_label 返回的查询像素列表。
        k: 有效邻居数（剔除自身后）。
        exact: True 使用暴力精确搜索。
        progress_callback: 进度回调 callback(current, total)。

    Returns:
        dict with keys: overall_accuracy, per_class_metrics,
        confusion_matrix, k, num_queries, elapsed_sec.
    """
    searcher = PixelSearcher(manager)
    num_labels = len(LABEL_NAMES)
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    total_queries = len(queries)
    start = time.perf_counter()

    for idx, q in enumerate(queries):
        if q.get("vectors") is not None:
            continue  # 跳过无像素类别的空标记

        result = searcher.search(
            query_vector=q["vector"],
            k=k + 1,
            exact=exact,
        )

        # Leave-One-Out: 剔除自身
        effective_hits = [h for h in result.hits if h.id != q["point_id"]][:k]

        if not effective_hits:
            continue  # 无有效邻居，跳过

        true_label = q["label"]
        votes = [h.label for h in effective_hits]
        counter = collections.Counter(votes)
        max_count = max(counter.values())
        winners = [l for l, c in counter.items() if c == max_count]

        if len(winners) == 1:
            predicted = winners[0]
        else:
            predicted = _resolve_tie(votes, effective_hits)

        confusion[true_label][predicted] += 1

        if progress_callback:
            progress_callback(idx + 1, total_queries)

    # 计算指标
    per_class: dict[str, dict] = {}
    total_correct = np.trace(confusion)
    total_all = confusion.sum()
    overall_accuracy = total_correct / max(total_all, 1)

    for lid, lname in LABEL_NAMES.items():
        tp = confusion[lid][lid]
        fp = confusion[:, lid].sum() - tp
        fn = confusion[lid, :].sum() - tp
        support = int(confusion[lid, :].sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        per_class[lname] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": support,
        }

    elapsed = time.perf_counter() - start

    return {
        "overall_accuracy": round(float(overall_accuracy), 4),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "k": k,
        "num_queries": total_queries,
        "elapsed_sec": round(elapsed, 2),
    }
```

- [x] **Step 5: 写采样单元测试**

在 `KNN_evaluation/tests/test_metrics.py`:

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from KNN_evaluation.metrics import (
    sample_queries_by_label,
    compute_knn_accuracy,
    _resolve_tie,
)
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.searcher import HitRecord


class TestSampleQueriesByLabel:
    def test_raises_connection_error_when_unreachable(self):
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = False
        with pytest.raises(ConnectionError):
            sample_queries_by_label(manager, samples_per_class=10)

    def test_raises_value_error_when_collection_missing(self):
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = True
        manager.collection_exists.return_value = False
        with pytest.raises(ValueError, match="不存在"):
            sample_queries_by_label(manager, samples_per_class=10)

    def test_raises_value_error_when_collection_empty(self):
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True
        manager.collection_info.return_value = {"total_points": 0}
        with pytest.raises(ValueError, match="为空"):
            sample_queries_by_label(manager, samples_per_class=10)
```

- [x] **Step 6: 写 F1 单元测试（mock PixelSearcher）**

```python
class TestComputeKnnAccuracy:
    def test_all_neighbors_same_label_gives_accuracy_one(self):
        manager = MagicMock(spec=QdrantManager)
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": "water",
             "image_id": "img1", "pixel_row": 0, "pixel_col": 0,
             "point_id": "uuid-self", "actual_count": 10},
        ]

        # 构造 mock hits: 全部与查询同 label
        mock_hits = [
            HitRecord(id="uuid-1", score=0.95, label=0, label_name="water",
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img1", pixel_row=1, pixel_col=1),
        ] * 10

        mock_result = MagicMock()
        mock_result.hits = mock_hits

        with patch("KNN_evaluation.metrics.PixelSearcher") as mock_searcher_cls:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = mock_result
            mock_searcher_cls.return_value = mock_searcher

            result = compute_knn_accuracy(manager, queries, k=3)

        assert result["overall_accuracy"] == 1.0
        assert result["per_class_metrics"]["water"]["f1"] == 1.0
        assert result["k"] == 3
        assert result["num_queries"] == 1

    def test_tie_break_decrement(self):
        """验证 _resolve_tie 递减 K 打破平票。"""
        votes = [0, 0, 1, 1, 2, 2, 0, 1, 2, 3]
        hits = [
            HitRecord(id=f"uuid-{i}", score=1.0 - i * 0.01,
                      label=votes[i], label_name="test",
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img", pixel_row=i, pixel_col=0)
            for i in range(10)
        ]
        winner = _resolve_tie(votes, hits)
        assert winner == 0  # K-1=9: 3 votes for 0, 可打破
```

- [x] **Step 7: 运行测试确保通过**

```bash
uv run pytest KNN_evaluation/tests/test_metrics.py::TestSampleQueriesByLabel -v
uv run pytest KNN_evaluation/tests/test_metrics.py::TestComputeKnnAccuracy -v
```

- [x] **Step 8: 提交**

```bash
git add KNN_evaluation/metrics.py KNN_evaluation/tests/test_metrics.py
git commit -m "feat(metrics): add sample_queries_by_label and compute_knn_accuracy with tie-breaking"
```

---

### Task 2: metrics.py — F2 纯度/Recall 曲线

**Files:**
- Modify: `KNN_evaluation/metrics.py`
- Test: `KNN_evaluation/tests/test_metrics.py`（追加）

**Interfaces:**
- Produces: `compute_purity_recall_curve(manager, queries, k_values, exact, progress_callback) -> dict`
- Internal: `_compute_per_class_label_totals(manager) -> dict[str, int]`

- [x] **Step 1: 实现 `_compute_per_class_label_totals`**

在 `metrics.py` 中添加：

```python
def _compute_per_class_label_totals(manager: QdrantManager) -> dict[str, int]:
    """按 label 统计 Qdrant 中各类别的全局像素总数。

    Returns:
        {label_name: count} 字典。
    """
    totals: dict[str, int] = {}
    for label_id, label_name in LABEL_NAMES.items():
        count_result = manager.client.count(
            collection_name=manager.collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(
                    key="label", match=models.MatchValue(value=label_id),
                )],
            ),
            exact=True,
        )
        totals[label_name] = count_result.count
    return totals
```

- [x] **Step 2: 实现 `compute_purity_recall_curve`**

```python
DEFAULT_K_VALUES = [10, 20, 50, 100, 300, 1000]


def compute_purity_recall_curve(
    manager: QdrantManager,
    queries: list[dict],
    k_values: list[int] | None = None,
    exact: bool = True,
    progress_callback: ProgressCallback = None,
) -> dict:
    """计算不同 K 值下的邻居纯度（Purity@K）和召回率（Recall@K）。

    对每个查询像素仅执行一次 search(k=max(k_values)+1, exact=True)，
    Leave-One-Out 剔除自身后递增取 hits[:k] 计算。

    Args:
        manager: QdrantManager 实例。
        queries: 查询像素列表。
        k_values: K 值列表，默认 [10, 20, 50, 100, 300, 1000]。
        exact: True 使用暴力精确搜索。
        progress_callback: 进度回调。

    Returns:
        dict with keys: k_values, global_purity, global_recall,
        per_class_purity, per_class_recall, num_queries, elapsed_sec.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    if not k_values:
        raise ValueError("k_values 不能为空")

    sorted_k = sorted(k_values)
    max_k = sorted_k[-1]

    # 获取全局各类像素数（Recall@K 分母）
    label_totals = _compute_per_class_label_totals(manager)

    searcher = PixelSearcher(manager)
    total_queries = len(queries)
    start = time.perf_counter()

    # 累加器
    num_k = len(sorted_k)
    purity_sums: dict[int, float] = {k: 0.0 for k in sorted_k}
    recall_sums: dict[int, float] = {k: 0.0 for k in sorted_k}
    per_class_purity_sums: dict[str, dict[int, float]] = {
        ln: {k: 0.0 for k in sorted_k} for ln in LABEL_NAMES.values()
    }
    per_class_recall_sums: dict[str, dict[int, float]] = {
        ln: {k: 0.0 for k in sorted_k} for ln in LABEL_NAMES.values()
    }
    class_counts: dict[str, int] = {ln: 0 for ln in LABEL_NAMES.values()}
    valid_count = 0

    for idx, q in enumerate(queries):
        if q.get("vectors") is not None:
            continue  # 跳过无像素类别的空标记

        result = searcher.search(
            query_vector=q["vector"],
            k=max_k + 1,
            exact=exact,
        )

        # Leave-One-Out
        effective_hits = [h for h in result.hits if h.id != q["point_id"]]

        query_label_name = q["label_name"]
        class_counts[query_label_name] = class_counts.get(query_label_name, 0) + 1
        total_same = max(label_totals.get(query_label_name, 1), 1)

        for kv in sorted_k:
            top_k = effective_hits[:kv]
            k_same = sum(1 for h in top_k if h.label_name == query_label_name)
            purity_sums[kv] += k_same / max(len(top_k), 1)
            recall_sums[kv] += k_same / total_same
            per_class_purity_sums[query_label_name][kv] += k_same / max(len(top_k), 1)
            per_class_recall_sums[query_label_name][kv] += k_same / total_same

        valid_count += 1

        if progress_callback:
            progress_callback(idx + 1, total_queries)

    # 聚合
    global_purity = [round(purity_sums[k] / max(valid_count, 1), 4) for k in sorted_k]
    global_recall = [round(recall_sums[k] / max(valid_count, 1), 6) for k in sorted_k]
    per_class_purity: dict[str, list[float]] = {}
    per_class_recall: dict[str, list[float]] = {}
    for ln in LABEL_NAMES.values():
        cnt = max(class_counts.get(ln, 0), 1)
        per_class_purity[ln] = [
            round(per_class_purity_sums[ln][k] / cnt, 4) for k in sorted_k
        ]
        per_class_recall[ln] = [
            round(per_class_recall_sums[ln][k] / cnt, 6) for k in sorted_k
        ]

    elapsed = time.perf_counter() - start

    return {
        "k_values": sorted_k,
        "global_purity": global_purity,
        "global_recall": global_recall,
        "per_class_purity": per_class_purity,
        "per_class_recall": per_class_recall,
        "num_queries": valid_count,
        "elapsed_sec": round(elapsed, 2),
    }
```

- [x] **Step 3: 写 F2 单元测试**

在 `test_metrics.py` 中追加：

```python
class TestComputePurityRecallCurve:
    def test_loo_excludes_self(self):
        """验证 Leave-One-Out 剔除自身。"""
        manager = MagicMock(spec=QdrantManager)
        # 模拟 count 返回每个 label 10000 个全局像素
        manager.client.count.return_value.count = 10000

        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": "water",
             "image_id": "img1", "pixel_row": 0, "pixel_col": 0,
             "point_id": "uuid-self", "actual_count": 10},
        ]

        # 构造 hits: 第一个就是自身
        mock_hits = [
            HitRecord(id="uuid-self", score=1.0, label=0, label_name="water",
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img1", pixel_row=0, pixel_col=0),
            HitRecord(id="uuid-1", score=0.99, label=1, label_name="trees",
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img1", pixel_row=1, pixel_col=1),
        ] * 50  # 足够覆盖 max(k_values)

        mock_result = MagicMock()
        mock_result.hits = mock_hits

        with patch("KNN_evaluation.metrics.PixelSearcher") as mock_sc:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = mock_result
            mock_sc.return_value = mock_searcher

            result = compute_purity_recall_curve(
                manager, queries, k_values=[5, 10],
            )

        # 自身被剔除，所有 neighbor 都是 trees(label=1)
        # Purity@5: 0/5 = 0，非 1.0
        assert result["global_purity"][0] == 0.0

    def test_purity_decreases_with_k(self):
        """验证 Purity(K₁) >= Purity(K₂) when K₁ < K₂. (Purity 应随 K 增大单调递减)"""
        # 使用 mock 构造 mixed neighbors
        manager = MagicMock(spec=QdrantManager)
        manager.client.count.return_value.count = 10000

        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": "water",
             "image_id": "img1", "pixel_row": 0, "pixel_col": 0,
             "point_id": "uuid-self", "actual_count": 10},
        ]

        # 前 5 个 neighbor 与查询同 label，后面全是不同 label
        mock_hits = []
        for i in range(5):
            mock_hits.append(HitRecord(
                id=f"uuid-{i}", score=1.0 - i * 0.01, label=0, label_name="water",
                utm_easting=0, utm_northing=0, utm_zone=51,
                image_id="img1", pixel_row=i, pixel_col=0,
            ))
        for i in range(5, 100):
            mock_hits.append(HitRecord(
                id=f"uuid-{i}", score=1.0 - i * 0.01, label=1, label_name="trees",
                utm_easting=0, utm_northing=0, utm_zone=51,
                image_id="img1", pixel_row=i, pixel_col=0,
            ))

        mock_result = MagicMock()
        mock_result.hits = mock_hits

        with patch("KNN_evaluation.metrics.PixelSearcher") as mock_sc:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = mock_result
            mock_sc.return_value = mock_searcher

            result = compute_purity_recall_curve(
                manager, queries, k_values=[5, 10],
            )

        # Purity@5: 5/5=1.0, Purity@10: 5/10=0.5
        assert result["global_purity"][0] >= result["global_purity"][1]

    def test_empty_k_values_raises(self):
        manager = MagicMock(spec=QdrantManager)
        with pytest.raises(ValueError, match="不能为空"):
            compute_purity_recall_curve(manager, [], k_values=[])
```

- [x] **Step 4: 运行测试确保通过**

```bash
uv run pytest KNN_evaluation/tests/test_metrics.py::TestComputePurityRecallCurve -v
```

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/metrics.py KNN_evaluation/tests/test_metrics.py
git commit -m "feat(metrics): add compute_purity_recall_curve with single-search optimization"
```

---

### Task 3: metrics.py — F3 距离分布

**Files:**
- Modify: `KNN_evaluation/metrics.py`
- Test: `KNN_evaluation/tests/test_metrics.py`（追加）

**Interfaces:**
- Produces: `compute_distance_distribution(manager, samples_per_class, seed) -> dict`

- [x] **Step 1: 实现 `compute_distance_distribution`**

在 `metrics.py` 中添加：

```python
def compute_distance_distribution(
    manager: QdrantManager,
    samples_per_class: int = 200,
    seed: int = 42,
) -> dict:
    """计算 Intra/Inter-class 余弦距离分布。

    逐类 scroll 获取向量，numpy 层计算余弦距离。
    余弦距离 = 1 - (v1·v2) / (‖v1‖·‖v2‖)

    Args:
        manager: QdrantManager 实例。
        samples_per_class: 每类采样数。
        seed: 随机种子。

    Returns:
        dict with keys: intra_stats, inter_stats,
        global_separability_ratio, most_confused_pairs,
        samples_per_class, elapsed_sec.
    """
    if not manager.health_check():
        raise ConnectionError(f"Qdrant 不可达: {manager.url}")
    if not manager.collection_exists():
        raise ValueError(f"Collection '{manager.collection_name}' 不存在")

    rng = random.Random(seed)
    start = time.perf_counter()

    # 逐类采样向量
    class_vectors: dict[str, np.ndarray] = {}
    for label_id, label_name in LABEL_NAMES.items():
        records, _ = manager.client.scroll(
            collection_name=manager.collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="label", match=models.MatchValue(value=label_id),
                )],
            ),
            limit=samples_per_class * 3,
            with_vectors=True,
            with_payload=False,
        )
        selected = rng.sample(records, min(len(records), samples_per_class))
        vectors = np.array([rec.vector for rec in selected], dtype=np.float64)
        class_vectors[label_name] = vectors

    # 归一化所有向量（用于余弦距离计算）
    norms = {ln: np.linalg.norm(v, axis=1, keepdims=True) for ln, v in class_vectors.items()}
    normalized = {ln: v / np.maximum(norms[ln], 1e-12) for ln, v in class_vectors.items()}

    # Intra-class 距离
    intra_stats: dict[str, dict] = {}
    all_intra_dists: list[float] = []
    for label_name, v in normalized.items():
        sim_matrix = v @ v.T
        # 取上三角（排除对角线的 0）
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        dists = 1.0 - sim_matrix[triu_indices]
        dists = np.clip(dists, 0.0, 2.0)  # 余弦距离范围 [0, 2]
        if len(dists) == 0:
            intra_stats[label_name] = {"mean": 0.0, "std": 0.0, "median": 0.0}
        else:
            intra_stats[label_name] = {
                "mean": round(float(np.mean(dists)), 4),
                "std": round(float(np.std(dists)), 4),
                "median": round(float(np.median(dists)), 4),
            }
            all_intra_dists.extend(dists.tolist())

    # Inter-class 距离
    label_list = list(LABEL_NAMES.values())
    inter_stats: dict[str, dict] = {}
    all_inter_dists: list[float] = []
    pair_means: list[tuple[str, str, float]] = []

    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            ln1, ln2 = label_list[i], label_list[j]
            sim_matrix = normalized[ln1] @ normalized[ln2].T
            dists = 1.0 - sim_matrix
            dists = np.clip(dists, 0.0, 2.0)
            flat_dists = dists.ravel()
            key = f"{ln1}-{ln2}"
            inter_stats[key] = {
                "mean": round(float(np.mean(flat_dists)), 4),
                "std": round(float(np.std(flat_dists)), 4),
            }
            inter_mean = float(np.mean(flat_dists))
            pair_means.append((ln1, ln2, inter_mean))
            all_inter_dists.extend(flat_dists.tolist())

    # 全局可分性比率
    global_intra_mean = np.mean(all_intra_dists) if all_intra_dists else 0.0
    global_inter_mean = np.mean(all_inter_dists) if all_inter_dists else 0.0
    separability_ratio = (
        global_inter_mean / global_intra_mean if global_intra_mean > 0 else 0.0
    )

    # Top-3 最混淆类对
    pair_means.sort(key=lambda x: x[2])
    most_confused_pairs = [
        (ln1, ln2, round(mean_val, 4)) for ln1, ln2, mean_val in pair_means[:3]
    ]

    elapsed = time.perf_counter() - start

    return {
        "intra_stats": intra_stats,
        "inter_stats": inter_stats,
        "global_separability_ratio": round(float(separability_ratio), 2),
        "most_confused_pairs": most_confused_pairs,
        "samples_per_class": samples_per_class,
        "elapsed_sec": round(elapsed, 2),
    }
```

- [x] **Step 2: 写 F3 单元测试**

在 `test_metrics.py` 中追加：

```python
class TestComputeDistanceDistribution:
    def test_same_vector_distance_zero(self):
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True

        # Mock scroll 返回两个相同向量（每类）
        mock_record = MagicMock()
        mock_record.vector = list(np.ones(64, dtype=np.float64))

        manager.client.scroll.return_value = ([mock_record, mock_record], None)

        result = compute_distance_distribution(manager, samples_per_class=2, seed=42)

        # 所有 intra distance 应为 0（相同向量）
        for ln, stats in result["intra_stats"].items():
            assert stats["mean"] == 0.0, f"{ln} intra mean should be 0"

    def test_orthogonal_vectors_distance_one(self):
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True

        # 对 water 返回正交向量
        v1 = np.zeros(64, dtype=np.float64)
        v1[0] = 1.0
        v2 = np.zeros(64, dtype=np.float64)
        v2[1] = 1.0

        rec1 = MagicMock()
        rec1.vector = v1.tolist()
        rec2 = MagicMock()
        rec2.vector = v2.tolist()

        # 所有类别的 scroll 都返回相同数据
        manager.client.scroll.return_value = ([rec1, rec2], None)

        result = compute_distance_distribution(manager, samples_per_class=2, seed=42)

        # 正交向量余弦距离 = 1
        assert abs(result["intra_stats"]["water"]["mean"] - 1.0) < 0.01

    def test_intra_less_than_inter_for_separated_data(self):
        """验证类内距离 < 类间距离（构造分离数据）。"""
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True

        # 为 water 和 trees 构造不同聚类中心的向量
        def make_records(base, noise_scale=0.01):
            rng = np.random.RandomState(42)
            recs = []
            for _ in range(10):
                vec = base + rng.randn(64) * noise_scale
                rec = MagicMock()
                rec.vector = vec.tolist()
                recs.append(rec)
            return recs

        water_base = np.zeros(64)
        water_base[0] = 1.0
        trees_base = np.zeros(64)
        trees_base[1] = 1.0

        water_recs = make_records(water_base)
        trees_recs = make_records(trees_base)

        # 按 label 返回不同数据
        def scroll_side_effect(collection_name, scroll_filter, limit, with_vectors, with_payload):
            must = scroll_filter.must if hasattr(scroll_filter, 'must') else []
            return ([], None)

        # 简化：对所有 label 使用 scroll 返回值列表
        call_count = [0]

        def scroll_side_effect(**kwargs):
            label_val = None
            if "scroll_filter" in kwargs and kwargs["scroll_filter"]:
                f = kwargs["scroll_filter"]
                if hasattr(f, 'must') and f.must:
                    label_val = f.must[0].match.value
            # 模拟 alternation — label 0: water vectors, label 1: trees vectors
            rng = np.random.RandomState(42 + (label_val or 0))
            base = np.zeros(64)
            if label_val == 0:
                base[0] = 1.0
            elif label_val == 1:
                base[1] = 1.0
            recs = []
            for _ in range(10):
                vec = base + rng.randn(64) * 0.01
                rec = MagicMock()
                rec.vector = vec.tolist()
                recs.append(rec)
            return (recs, None)

        manager.client.scroll = MagicMock(side_effect=scroll_side_effect)

        result = compute_distance_distribution(manager, samples_per_class=10, seed=42)

        assert result["global_separability_ratio"] > 1.0
```

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/metrics.py KNN_evaluation/tests/test_metrics.py
git commit -m "feat(metrics): add compute_distance_distribution for intra/inter-class cosine distances"
```

---

### Task 4: visualization.py — matplotlib 图表

**Files:**
- Create: `KNN_evaluation/visualization.py`

**Interfaces:**
- Produces: `plot_confusion_matrix(cm, label_names, save_path, title) -> None`
- Produces: `plot_purity_recall_curve(purity_data, save_path) -> None`
- Produces: `plot_distance_histogram(distance_data, save_path) -> None`

- [x] **Step 1: 创建 visualization.py**

```python
"""评估结果可视化模块 — matplotlib 图表生成."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体配置
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """绘制混淆矩阵热力图。

    Args:
        cm: shape (N, N) 混淆矩阵，C[i][j] = 真实 i 预测为 j。
        label_names: 标签名列表。
        save_path: 输出 PNG 路径。
        title: 图表标题。
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")

    # 标注
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            val = cm[i][j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_purity_recall_curve(
    purity_data: dict,
    save_path: str | Path,
) -> None:
    """绘制 Purity/Recall vs K 双面板图。

    Args:
        purity_data: compute_purity_recall_curve 的返回值。
        save_path: 输出 PNG 路径。
    """
    k_values = purity_data["k_values"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左面板: Purity
    for ln, purity_list in purity_data["per_class_purity"].items():
        ax1.plot(k_values, purity_list, marker="o", markersize=3,
                 linewidth=1, alpha=0.6, label=ln)
    ax1.plot(k_values, purity_data["global_purity"], marker="o",
             linewidth=2.5, color="black", label="Global")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Purity@K")
    ax1.set_title("Neighborhood Purity vs K")
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # 右面板: Recall@K
    for ln, recall_list in purity_data["per_class_recall"].items():
        ax2.plot(k_values, recall_list, marker="s", markersize=3,
                 linewidth=1, alpha=0.6, label=ln)
    ax2.plot(k_values, purity_data["global_recall"], marker="s",
             linewidth=2.5, color="black", label="Global")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Recall@K")
    ax2.set_title("Recall@K vs K")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    fig.suptitle("Embedding Quality: Purity & Recall@K Curves", fontsize=14)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_histogram(
    distance_data: dict,
    save_path: str | Path,
) -> None:
    """绘制 Intra/Inter-class 距离箱线图。

    Args:
        distance_data: compute_distance_distribution 的返回值。
        save_path: 输出 PNG 路径。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左面板: Intra-class 箱线图
    intra_means = [s["mean"] for s in distance_data["intra_stats"].values()]
    intra_labels = list(distance_data["intra_stats"].keys())

    colors = plt.cm.tab10(range(len(intra_labels)))
    ax1.bar(range(len(intra_labels)), intra_means, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(intra_labels)))
    ax1.set_xticklabels(intra_labels, rotation=45, ha="right")
    ax1.set_ylabel("Mean Cosine Distance")
    ax1.set_title("Intra-class Mean Distance")
    ax1.grid(True, alpha=0.3, axis="y")

    # 右面板: Inter-class 箱线图
    inter_items = sorted(
        distance_data["inter_stats"].items(), key=lambda x: x[1]["mean"],
    )[:15]
    inter_means = [s["mean"] for _, s in inter_items]
    inter_labels = [l for l, _ in inter_items]

    ax2.barh(range(len(inter_labels)), inter_means,
             color=plt.cm.tab20(range(len(inter_labels))), alpha=0.8)
    ax2.set_yticks(range(len(inter_labels)))
    ax2.set_yticklabels(inter_labels, fontsize=8)
    ax2.set_xlabel("Mean Cosine Distance")
    ax2.set_title("Inter-class Mean Distance (Top-15 Closest Pairs)")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        f"Distance Distribution (Separability Ratio: "
        f"{distance_data['global_separability_ratio']})",
        fontsize=14,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [x] **Step 2: 提交**

```bash
git add KNN_evaluation/visualization.py
git commit -m "feat(visualization): add matplotlib chart functions for confusion matrix, purity/recall, distance"
```

---

### Task 5: CLI — evaluate 子命令

**Files:**
- Modify: `KNN_evaluation/cli.py`

**Interfaces:**
- Produces: `cmd_evaluate(args) -> int`
- Consumes: `metrics.sample_queries_by_label`, `compute_knn_accuracy`, `compute_purity_recall_curve`, `compute_distance_distribution`
- Consumes: `visualization.plot_confusion_matrix`, `plot_purity_recall_curve`, `plot_distance_histogram`

- [x] **Step 1: 新增 evaluate 子命令参数**

在 `cli.py` 的 `main()` 函数中添加（放在 `p_stats` 定义之后）：

```python
    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="评估 embedding 质量指标 (F1/F2/F3)")
    p_eval.add_argument("--samples-per-class", type=int, default=500,
                        help="每类采样查询像素数 (默认: 500)")
    p_eval.add_argument("--k-f1", type=int, default=10,
                        help="KNN 分类器 K 值 (默认: 10)")
    p_eval.add_argument("--k-values", type=str, default="10,20,50,100,300,1000",
                        help="Purity/Recall 曲线的 K 值序列，逗号分隔 (默认: 10,20,50,100,300,1000)")
    p_eval.add_argument("--distance-samples", type=int, default=200,
                        help="F3 距离分析的每类采样数 (默认: 200)")
    p_eval.add_argument("--with-distance", action="store_true",
                        help="执行 F3 距离分布分析")
    p_eval.add_argument("--ann", action="store_true",
                        help="额外用 ANN 模式跑一遍，输出 exact vs ANN 对比")
    p_eval.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    p_eval.add_argument("--output", type=str, default=None,
                        help="结果 JSON 输出路径")
    p_eval.add_argument("--plot", action="store_true",
                        help="生成 matplotlib 图表")
    p_eval.add_argument("--plot-dir", type=str, default="./eval_plots",
                        help="图表输出目录 (默认: ./eval_plots)")
    p_eval.add_argument("--qdrant-url", default=QDRANT_URL,
                        help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")
```

在 `main()` 底部的 dispatch 中添加：

```python
    elif args.command == "evaluate":
        return cmd_evaluate(args)
```

- [x] **Step 2: 实现 `cmd_evaluate`**

在 `cli.py` 中添加（放在现有 `cmd_*` 函数之后、`main()` 之前）：

```python
def _np_encoder(obj):
    """JSON encoder: 将 numpy 类型转换为 Python 原生类型."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def cmd_evaluate(args) -> int:
    """执行 evaluate 子命令."""
    from KNN_evaluation.metrics import (
        sample_queries_by_label,
        compute_knn_accuracy,
        compute_purity_recall_curve,
        compute_distance_distribution,
    )
    from KNN_evaluation.label_mapping import LABEL_NAMES

    # 连接 Qdrant
    manager = QdrantManager(url=args.qdrant_url)

    if not manager.health_check():
        print(f"错误: Qdrant 不可达 ({args.qdrant_url})", file=sys.stderr)
        return 1

    if not manager.collection_exists():
        print(f"错误: Collection '{manager.collection_name}' 不存在，请先执行 import", file=sys.stderr)
        return 1

    info = manager.collection_info()
    if info.get("total_points", 0) == 0:
        print("错误: Collection 为空，请先导入数据", file=sys.stderr)
        return 1

    # 解析 K 值
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    print(f"{'='*60}")
    print(f"KNN Embedding 质量评估")
    print(f"{'='*60}")
    print(f"Collection: {manager.collection_name}  |  总像素: {info['total_points']:,}")
    total_start = time.perf_counter()

    # --- 采样 ---
    print(f"\n⏳ 正在采样查询像素 (每类 {args.samples_per_class})...")
    try:
        queries = sample_queries_by_label(manager, args.samples_per_class, args.seed)
    except (ConnectionError, ValueError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    num_queries = sum(1 for q in queries if "point_id" in q)
    print(f"   已采样 {num_queries} 个查询像素")

    # --- F1: KNN Classification ---
    print(f"\n--- F1: KNN 分类准确率 (K={args.k_f1}, exact) ---")
    with _tqdm_context(num_queries, "F1 KNN Accuracy") as pbar:
        def progress(c, t):
            pbar.update(c - pbar.n)

        try:
            f1 = compute_knn_accuracy(
                manager, queries, k=args.k_f1, exact=True,
                progress_callback=progress,
            )
        except Exception as e:
            print(f"错误: F1 计算失败: {e}", file=sys.stderr)
            return 1

    print(f"Overall Accuracy: {f1['overall_accuracy']:.4f}")
    print(f"\nPer-class Metrics:")
    print(f"  {'Label':<22} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    label_names = list(LABEL_NAMES.values())
    for ln in label_names:
        m = f1["per_class_metrics"].get(ln, {})
        print(f"  {ln:<22} {m.get('precision', 0):>6.4f} {m.get('recall', 0):>6.4f} "
              f"{m.get('f1', 0):>6.4f} {m.get('support', 0):>8}")
    print(f"\n  耗时: {f1['elapsed_sec']:.1f}s")

    # --- F2: Purity & Recall@K ---
    print(f"\n--- F2: Purity & Recall@K (exact) ---")
    with _tqdm_context(num_queries, "F2 Purity/Recall") as pbar:
        def progress2(c, t):
            pbar.update(c - pbar.n)

        try:
            f2 = compute_purity_recall_curve(
                manager, queries, k_values, exact=True,
                progress_callback=progress2,
            )
        except Exception as e:
            print(f"错误: F2 计算失败: {e}", file=sys.stderr)
            return 1

    print(f"  {'K':>6} {'Purity':>8} {'Recall@K':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*12}")
    for i, kv in enumerate(f2["k_values"]):
        print(f"  {kv:>6} {f2['global_purity'][i]:>8.4f} {f2['global_recall'][i]:>12.6f}")
    print(f"\n  耗时: {f2['elapsed_sec']:.1f}s")

    # --- F3: 距离分布 ---
    f3 = None
    if args.with_distance:
        print(f"\n--- F3: 距离分布 (每类 {args.distance_samples} 采样) ---")
        try:
            f3 = compute_distance_distribution(
                manager, args.distance_samples, args.seed,
            )
        except Exception as e:
            print(f"错误: F3 计算失败: {e}", file=sys.stderr)
            return 1

        print(f"可分性比率 (mean_inter / mean_intra): {f3['global_separability_ratio']}")
        print(f"\nIntra-class Mean Distance:")
        for ln in label_names:
            s = f3["intra_stats"].get(ln, {})
            print(f"  {ln:<22} mean={s.get('mean', 0):.4f}  std={s.get('std', 0):.4f}  median={s.get('median', 0):.4f}")
        print(f"\n最混淆类对 (Top-3):")
        for ln1, ln2, dist in f3["most_confused_pairs"]:
            print(f"  {ln1} ↔ {ln2}: {dist:.4f}")
        print(f"\n  耗时: {f3['elapsed_sec']:.1f}s")

    # --- ANN 对比 (可选) ---
    ann_f1 = None
    ann_f2 = None
    if args.ann:
        print(f"\n--- ANN vs Exact ---")
        with _tqdm_context(num_queries, "ANN F1 Accuracy") as pbar:
            def progress_a1(c, t):
                pbar.update(c - pbar.n)
            ann_f1 = compute_knn_accuracy(
                manager, queries, k=args.k_f1, exact=False,
                progress_callback=progress_a1,
            )
        with _tqdm_context(num_queries, "ANN F2 Purity/Recall") as pbar:
            def progress_a2(c, t):
                pbar.update(c - pbar.n)
            ann_f2 = compute_purity_recall_curve(
                manager, queries, k_values, exact=False,
                progress_callback=progress_a2,
            )

        print(f"{'Metric':<20} {'Exact':>8} {'ANN':>8} {'Δ':>10}")
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*10}")
        print(f"{'F1 Accuracy':<20} {f1['overall_accuracy']:>8.4f} "
              f"{ann_f1['overall_accuracy']:>8.4f} "
              f"{ann_f1['overall_accuracy'] - f1['overall_accuracy']:>+10.4f}")
        if f2["k_values"]:
            last_k = len(f2["k_values"]) - 1
            print(f"{'Purity@' + str(f2['k_values'][last_k]):<20} "
                  f"{f2['global_purity'][last_k]:>8.4f} "
                  f"{ann_f2['global_purity'][last_k]:>8.4f} "
                  f"{ann_f2['global_purity'][last_k] - f2['global_purity'][last_k]:>+10.4f}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"总耗时: {total_elapsed:.1f}s")
    print(f"{'='*60}")

    # --- JSON 导出 ---
    if args.output:
        result = {
            "config": {
                "samples_per_class": args.samples_per_class,
                "k_f1": args.k_f1,
                "k_values": f2["k_values"],
                "distance_samples": args.distance_samples,
                "seed": args.seed,
            },
            "f1": {
                "overall_accuracy": f1["overall_accuracy"],
                "per_class_metrics": {
                    ln: {k: v for k, v in m.items()}
                    for ln, m in f1["per_class_metrics"].items()
                },
                "confusion_matrix": f1["confusion_matrix"],
                "k": f1["k"],
                "num_queries": f1["num_queries"],
                "elapsed_sec": f1["elapsed_sec"],
            },
            "f2": {
                "k_values": f2["k_values"],
                "global_purity": f2["global_purity"],
                "global_recall": f2["global_recall"],
                "per_class_purity": f2["per_class_purity"],
                "per_class_recall": f2["per_class_recall"],
                "num_queries": f2["num_queries"],
                "elapsed_sec": f2["elapsed_sec"],
            },
            "f3": f3,
            "total_elapsed_sec": round(total_elapsed, 2),
        }
        if ann_f1:
            result["ann_f1"] = ann_f1
        if ann_f2:
            result["ann_f2"] = ann_f2
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=_np_encoder)
        print(f"\n结果已导出到: {args.output}")

    # --- 图表 ---
    if args.plot:
        from KNN_evaluation.visualization import (
            plot_confusion_matrix, plot_purity_recall_curve,
            plot_distance_histogram,
        )
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            f1["confusion_matrix"], label_names,
            plot_dir / "confusion_matrix.png",
        )
        print(f"  ✓ {plot_dir / 'confusion_matrix.png'}")
        plot_purity_recall_curve(f2, plot_dir / "purity_recall_curve.png")
        print(f"  ✓ {plot_dir / 'purity_recall_curve.png'}")
        if f3:
            plot_distance_histogram(f3, plot_dir / "distance_histogram.png")
            print(f"  ✓ {plot_dir / 'distance_histogram.png'}")
        print(f"\n图表已保存到: {plot_dir}")

    return 0
```

在 `cli.py` 顶部添加 import：

```python
import time
from tqdm import tqdm
```

并在文件顶部添加 tqdm 上下文管理器辅助函数（放在 `_parse_label` 之后）：

```python
class _tqdm_context:
    """tqdm 上下文管理器，确保 pbar.close() 被调用."""
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc

    def __enter__(self):
        self.pbar = tqdm(total=self.total, desc=self.desc, unit="query")
        return self.pbar

    def __exit__(self, *args):
        self.pbar.close()
```

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/cli.py
git commit -m "feat(cli): add evaluate subcommand with F1/F2/F3 metrics, JSON export, and ANN comparison"
```

---

### Task 6: WebUI — 评估面板

**Files:**
- Modify: `KNN_evaluation/webui.py`

**Interfaces:**
- Consumes: `metrics.sample_queries_by_label`, `compute_knn_accuracy`, `compute_purity_recall_curve`, `compute_distance_distribution`

- [x] **Step 1: 在"向量检索" expansion 后新增评估面板 expansion**

在 `webui.py` 中找到 `</expansion>`（向量检索 expansion 结束标记，约第 510 行），在其后添加：

```python
    # ===== 评估面板 =====
    with ui.expansion("评估面板", value=False).classes("w-full mt-2"):
        eval_state = {
            "f1_result": None,
            "f2_result": None,
            "f3_result": None,
        }

        ui.label("Embedding 质量评估").classes("text-h6")

        # 参数配置区
        with ui.row().classes("items-center gap-4 mt-2"):
            spc_input = ui.number(label="每类采样数", value=500, min=10, max=10000).classes("w-32")
            kf1_input = ui.number(label="K-F1", value=10, min=1, max=100).classes("w-20")
            kvalues_input = ui.input(
                label="K 值序列", value="10,20,50,100,300,1000",
            ).classes("w-64")
            with_distance_switch = ui.switch("F3 距离分析", value=False)
            seed_input = ui.number(label="Seed", value=42).classes("w-20")

        # 进度区
        eval_progress_label = ui.label("").classes("text-sm text-grey mt-2")
        eval_progress_label.set_visibility(False)

        # 结果区
        eval_result_container = ui.column().classes("w-full mt-4")
        eval_result_container.set_visibility(False)

        eval_export_btn = ui.button("导出 JSON", on_click=lambda: _export_eval_json())
        eval_export_btn.set_visibility(False)

        async def do_evaluate():
            if state["manager"] is None or not state["manager"].health_check():
                ui.notify("Qdrant 不可达", type="negative")
                return
            if not state["manager"].collection_exists():
                ui.notify(f"Collection '{COLLECTION_NAME}' 不存在", type="negative")
                return
            info = state["manager"].collection_info()
            if info.get("total_points", 0) == 0:
                ui.notify("Collection 为空", type="negative")
                return

            manager = state["manager"]
            spc = int(spc_input.value)
            k_f1 = int(kf1_input.value)
            k_values = [int(x.strip()) for x in kvalues_input.value.split(",") if x.strip()]
            with_dist = with_distance_switch.value
            seed_val = int(seed_input.value)

            from KNN_evaluation.metrics import (
                sample_queries_by_label,
                compute_knn_accuracy,
                compute_purity_recall_curve,
                compute_distance_distribution,
            )

            eval_progress_label.set_visibility(True)
            eval_progress_label.set_text("⏳ 正在采样查询像素...")
            await asyncio.sleep(0.05)

            try:
                queries = await asyncio.to_thread(
                    sample_queries_by_label, manager, spc, seed_val,
                )
            except Exception as e:
                ui.notify(f"采样失败: {e}", type="negative")
                eval_progress_label.set_visibility(False)
                return

            num_q = sum(1 for q in queries if "point_id" in q)
            eval_progress_label.set_text(f"⏳ 采样完成，共 {num_q} 个查询像素 | 正在计算 KNN Accuracy...")

            # F1
            def make_cb(phase):
                def cb(current, total):
                    eval_progress_label.set_text(f"⏳ {phase} ({current}/{total})")
                return cb

            try:
                f1 = await asyncio.to_thread(
                    compute_knn_accuracy, manager, queries, k_f1, True, make_cb("F1 KNN Accuracy"),
                )
            except Exception as e:
                ui.notify(f"F1 计算失败: {e}", type="negative")
                eval_progress_label.set_visibility(False)
                return

            eval_progress_label.set_text(f"⏳ F1 完成 | 正在计算 Purity/Recall...")

            # F2
            try:
                f2 = await asyncio.to_thread(
                    compute_purity_recall_curve, manager, queries, k_values, True,
                    make_cb("F2 Purity/Recall"),
                )
            except Exception as e:
                ui.notify(f"F2 计算失败: {e}", type="negative")
                eval_progress_label.set_visibility(False)
                return

            # F3 (optional)
            f3 = None
            if with_dist:
                eval_progress_label.set_text("⏳ F1+F2 完成 | 正在计算距离分布...")
                try:
                    f3 = await asyncio.to_thread(
                        compute_distance_distribution, manager, 200, seed_val,
                    )
                except Exception as e:
                    ui.notify(f"F3 计算失败: {e}", type="negative")

            eval_state["f1_result"] = f1
            eval_state["f2_result"] = f2
            eval_state["f3_result"] = f3

            eval_progress_label.set_visibility(False)
            _show_eval_results(f1, f2, f3)
            eval_export_btn.set_visibility(True)

        def _show_eval_results(f1, f2, f3):
            eval_result_container.clear()
            eval_result_container.set_visibility(True)

            # F1 结果
            with eval_result_container:
                ui.label(f"Overall Accuracy: {f1['overall_accuracy']:.4f}").classes("text-h6 mt-2")

                from KNN_evaluation.label_mapping import LABEL_NAMES
                label_names = list(LABEL_NAMES.values())

                f1_rows = []
                for ln in label_names:
                    m = f1["per_class_metrics"].get(ln, {})
                    f1_rows.append({
                        "label": ln,
                        "precision": f"{m.get('precision', 0):.4f}",
                        "recall": f"{m.get('recall', 0):.4f}",
                        "f1": f"{m.get('f1', 0):.4f}",
                        "support": str(m.get("support", 0)),
                    })

                ui.label("Per-class Metrics").classes("text-subtitle2 mt-2")
                ui.table(
                    columns=[
                        {"name": "label", "label": "Label", "field": "label"},
                        {"name": "precision", "label": "Precision", "field": "precision"},
                        {"name": "recall", "label": "Recall", "field": "recall"},
                        {"name": "f1", "label": "F1", "field": "f1"},
                        {"name": "support", "label": "Support", "field": "support"},
                    ],
                    rows=f1_rows, row_key="label",
                ).classes("w-full")

                # F2 图表: ECharts 双轴折线图
                ui.label("Purity & Recall@K").classes("text-subtitle2 mt-4")

                purity_series = [
                    {"name": ln, "type": "line", "data": vals}
                    for ln, vals in f2["per_class_purity"].items()
                ]
                purity_series.append({
                    "name": "Global Purity", "type": "line",
                    "data": f2["global_purity"],
                    "lineStyle": {"width": 3, "color": "#000"},
                })

                recall_series = [
                    {"name": ln, "type": "line", "data": vals}
                    for ln, vals in f2["per_class_recall"].items()
                ]
                recall_series.append({
                    "name": "Global Recall", "type": "line",
                    "data": f2["global_recall"],
                    "lineStyle": {"width": 3, "color": "#000"},
                })

                ui.echart({
                    "title": {"text": "Purity & Recall@K", "left": "center"},
                    "tooltip": {"trigger": "axis"},
                    "legend": {"type": "scroll", "bottom": 0},
                    "grid": {"left": "3%", "right": "4%", "bottom": "15%", "containLabel": True},
                    "xAxis": {
                        "type": "log",
                        "data": [str(k) for k in f2["k_values"]],
                        "name": "K",
                    },
                    "yAxis": [
                        {"type": "value", "name": "Purity", "min": 0, "max": 1},
                        {"type": "value", "name": "Recall@K"},
                    ],
                    "series": [
                        {**s, "yAxisIndex": 0} for s in purity_series
                    ] + [
                        {**s, "yAxisIndex": 1} for s in recall_series
                    ],
                })

                # F3 结果
                if f3:
                    ui.label("距离分布").classes("text-subtitle2 mt-4")
                    f3_rows = []
                    for ln in label_names:
                        s = f3.get("intra_stats", {}).get(ln, {})
                        f3_rows.append({
                            "label": ln,
                            "intra_mean": f"{s.get('mean', 0):.4f}",
                            "intra_std": f"{s.get('std', 0):.4f}",
                            "intra_median": f"{s.get('median', 0):.4f}",
                        })
                    ui.table(
                        columns=[
                            {"name": "label", "label": "Label", "field": "label"},
                            {"name": "intra_mean", "label": "Intra Mean", "field": "intra_mean"},
                            {"name": "intra_std", "label": "Intra Std", "field": "intra_std"},
                            {"name": "intra_median", "label": "Intra Median", "field": "intra_median"},
                        ],
                        rows=f3_rows, row_key="label",
                    ).classes("w-full")

                    ui.label(
                        f"可分性比率: {f3.get('global_separability_ratio', 'N/A')}"
                    ).classes("text-sm mt-1")
                    ui.label("最混淆类对:").classes("text-sm")
                    for ln1, ln2, dist in f3.get("most_confused_pairs", []):
                        ui.label(f"  {ln1} ↔ {ln2}: {dist:.4f}").classes("text-xs text-grey")

        def _export_eval_json():
            import json as _json

            def _conv(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return str(obj)

            export_data = {
                "f1": {k: v for k, v in eval_state["f1_result"].items()} if eval_state["f1_result"] else None,
                "f2": {
                    "k_values": eval_state["f2_result"]["k_values"],
                    "global_purity": eval_state["f2_result"]["global_purity"],
                    "global_recall": eval_state["f2_result"]["global_recall"],
                } if eval_state["f2_result"] else None,
                "f3": eval_state["f3_result"],
            }
            json_str = _json.dumps(export_data, indent=2, ensure_ascii=False, default=_conv)
            ui.download(
                json_str.encode("utf-8"),
                filename="evaluation_result.json",
                media_type="application/json",
            )

        with ui.row().classes("gap-2 mt-4"):
            ui.button("开始评估", on_click=do_evaluate).props("flat")
```

- [x] **Step 2: 提交**

```bash
git add KNN_evaluation/webui.py
git commit -m "feat(webui): add evaluation panel with async execution and ECharts visualization"
```

---

### Task 7: 集成测试与验证

**Files:**
- Modify: `KNN_evaluation/tests/test_metrics.py`（追加集成测试）

- [x] **Step 1: 追加集成测试**

```python
@pytest.mark.integration
class TestIntegration:
    """需要运行中的 Qdrant + data_demo 数据。"""

    def test_evaluate_end_to_end(self):
        """使用 data_demo 数据运行完整评估流程。"""
        from KNN_evaluation.qdrant_client import QdrantManager
        from KNN_evaluation.metrics import (
            sample_queries_by_label,
            compute_knn_accuracy,
            compute_purity_recall_curve,
        )

        manager = QdrantManager()
        if not manager.health_check():
            pytest.skip("Qdrant 不可达")
        if not manager.collection_exists():
            pytest.skip("Collection 不存在")

        info = manager.collection_info()
        if info.get("total_points", 0) == 0:
            pytest.skip("Collection 为空")

        # 小规模采样评估
        queries = sample_queries_by_label(manager, samples_per_class=50, seed=42)
        num_q = sum(1 for q in queries if "point_id" in q)
        assert num_q > 0

        f1 = compute_knn_accuracy(manager, queries, k=10)
        assert 0.0 <= f1["overall_accuracy"] <= 1.0
        assert f1["confusion_matrix"].shape == (9, 9)
        assert len(f1["per_class_metrics"]) == 9

        f2 = compute_purity_recall_curve(manager, queries, k_values=[10, 30, 50])
        assert len(f2["k_values"]) == 3
        assert len(f2["global_purity"]) == 3
        assert len(f2["global_recall"]) == 3

        # Purity 单调递减验证
        for i in range(len(f2["k_values"]) - 1):
            assert f2["global_purity"][i] >= f2["global_purity"][i + 1] - 0.01, \
                f"Purity@{f2['k_values'][i]}: {f2['global_purity'][i]} < Purity@{f2['k_values'][i+1]}: {f2['global_purity'][i+1]}"
```

- [x] **Step 2: 运行完整测试套件**

```bash
uv run pytest KNN_evaluation/tests/test_metrics.py -v
uv run python -m KNN_evaluation.cli evaluate --samples-per-class 50 --k-values 10,30,50 --seed 42
```

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/tests/test_metrics.py
git commit -m "test(metrics): add integration test for end-to-end evaluation flow"
```
