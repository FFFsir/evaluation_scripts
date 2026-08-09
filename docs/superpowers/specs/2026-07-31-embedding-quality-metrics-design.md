---
comet_change: embedding-quality-metrics
role: technical-design
canonical_spec: openspec
---

# Embedding 质量评估模块 — 技术设计

## 1. 架构概览

```
                    ┌──────────────────┐
                    │   cli.py / webui.py    │  ← 调用方，各自编排
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │ metrics.py │ │ visualization.py │ │ searcher.py │ ← 不改动
     └─────┬──────┘ └────────────┘ └─────┬──────┘
           │                             │
           ▼                             ▼
     ┌────────────┐             ┌────────────────┐
     │ qdrant_client.py │       │  Qdrant Server  │
     └────────────┘             └────────────────┘
```

- `metrics.py`：四个纯函数，无副作用，输入 manager + queries → 输出 dict/np.ndarray
- `visualization.py`：三个 matplotlib 函数，输入指标数据 → 写入 PNG
- `cli.py` / `webui.py`：各自编排上述调用

## 2. API 设计

### 2.1 进度回调

```python
from typing import Callable

ProgressCallback = Callable[[int, int], None] | None
# 签名: callback(current: int, total: int) -> None
```

- CLI 端：传入 `tqdm.tqdm(...).update` 的包装函数
- WebUI 端：传入更新 NiceGUI label text 的 async 安全函数
- `None` 时不输出任何进度（测试模式）

### 2.2 公开函数

#### `sample_queries_by_label`

```python
def sample_queries_by_label(
    manager: QdrantManager,
    samples_per_class: int = 500,
    seed: int = 42,
) -> list[dict]:
```

**流程**：
1. 对每个 label ∈ [0..8]，调用 `manager.client.scroll(collection_name=..., scroll_filter=Filter(label=c), with_vectors=True, limit=samples_per_class)`
2. 客户端 `random.sample()` 选取 `min(available, samples_per_class)` 个
3. 返回 list[dict]，每个 dict = `{"vector": np.ndarray(64,), "label": int, "label_name": str, "image_id": str, "pixel_row": int, "pixel_col": int, "point_id": str, "actual_count": int}`

**错误处理**：Qdrant 不可达 → raise `ConnectionError`；空 Collection → raise `ValueError("Collection 为空")`

#### `compute_knn_accuracy`

```python
def compute_knn_accuracy(
    manager: QdrantManager,
    queries: list[dict],
    k: int = 10,
    exact: bool = True,
    progress_callback: ProgressCallback = None,
) -> dict:
```

**流程**：
1. 对每个 query pixel，`searcher.search(query_vector, k=k+1, exact=exact)`
2. 从 hits 中剔除 `hit.id == query["point_id"]` 的记录，取前 k 个
3. 多数投票 → 平票时递减 K（K-1, K-3, K-5, K-7, K-9）重新计票
4. 构建 confusion_matrix (9×9)，计算 per-class P/R/F1 和 overall accuracy

**返回值**：
```python
{
    "overall_accuracy": float,
    "per_class_metrics": {
        "water": {"precision": float, "recall": float, "f1": float, "support": int},
        ...
    },
    "confusion_matrix": np.ndarray,  # shape (9, 9)
    "k": int,            # 有效 K 值
    "num_queries": int,
    "elapsed_sec": float,
}
```

#### `compute_purity_recall_curve`

```python
def compute_purity_recall_curve(
    manager: QdrantManager,
    queries: list[dict],
    k_values: list[int] | None = None,
    exact: bool = True,
    progress_callback: ProgressCallback = None,
) -> dict:
```

**默认 k_values**：`[10, 20, 50, 100, 300, 1000]`

**流程**：
1. 调用内部函数 `_compute_per_class_label_totals(manager)` 获取各类全局像素数
2. 对每个 query pixel，`searcher.search(query_vector, k=max(k_values)+1, exact=exact)`
3. Leave-One-Out 自身剔除后，对排序后的 k_values 递增取 `hits[:k]`
4. 分别计算 per-class 和 global 的 purity / recall

**内部函数 `_compute_per_class_label_totals`**：
```python
def _compute_per_class_label_totals(manager: QdrantManager) -> dict[str, int]:
    totals = {}
    for label_id, label_name in LABEL_NAMES.items():
        count_result = manager.client.count(
            collection_name=manager.collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="label", match=models.MatchValue(value=label_id))]
            ),
            exact=True,
        )
        totals[label_name] = count_result.count
    return totals
```

**返回值**：
```python
{
    "k_values": [10, 20, ...],
    "global_purity": [0.78, 0.71, ...],
    "global_recall": [0.002, 0.005, ...],
    "per_class_purity": {"water": [0.79, ...], ...},
    "per_class_recall": {"water": [0.001, ...], ...},
    "num_queries": int,
    "elapsed_sec": float,
}
```

#### `compute_distance_distribution`

```python
def compute_distance_distribution(
    manager: QdrantManager,
    samples_per_class: int = 200,
    seed: int = 42,
) -> dict:
```

**流程**：
1. 逐类 scroll（含 `with_vectors=True`），每类取 `min(available, samples_per_class)` 个
2. numpy 向量化计算：
   - Intra-class：对每类 c，计算 `1 - vectors_c @ vectors_c.T` 的上三角
   - Inter-class：对每对 (c1, c2)，计算 `1 - vectors_c1 @ vectors_c2.T` 的全部值
3. 汇总 per-class intra mean/std/median、per-pair inter mean/std
4. 计算 `global_separability_ratio = mean(all_inter) / mean(all_intra)`
5. 排序出 top-3 `most_confused_pairs`（inter mean 最小）

**返回值**：
```python
{
    "intra_stats": {"water": {"mean": 0.12, "std": 0.05, "median": 0.11}, ...},
    "inter_stats": {"water-trees": {"mean": 0.45, "std": 0.12}, ...},
    "global_separability_ratio": float,
    "most_confused_pairs": [("grass", "crops", 0.28), ...],
    "samples_per_class": int,
    "elapsed_sec": float,
}
```

## 3. 平票处理算法

```python
import collections

def _resolve_tie(votes: list[int], hits: list[HitRecord]) -> int:
    """递减 K 打破平票。"""
    for step in (0, 1, 3, 5, 7, 9):
        k_sub = len(votes) - step
        if k_sub <= 0:
            return hits[0].label  # fallback: 最近邻居的标签
        counter = collections.Counter(votes[:k_sub])
        max_count = max(counter.values())
        winners = [l for l, c in counter.items() if c == max_count]
        if len(winners) == 1:
            return winners[0]
    return hits[0].label  # 理论上不可达
```

## 4. CLI 设计

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--samples-per-class` | int | 500 | 每类采样数 |
| `--k-f1` | int | 10 | KNN 分类器 K 值 |
| `--k-values` | str | "10,20,50,100,300,1000" | Purity/Recall K 序列 |
| `--distance-samples` | int | 200 | F3 每类采样数 |
| `--with-distance` | flag | False | 执行 F3 |
| `--ann` | flag | False | ANN vs exact 对比 |
| `--seed` | int | 42 | 随机种子 |
| `--output` | str | - | JSON 输出路径 |
| `--plot` | flag | False | 生成 PNG 图表 |
| `--plot-dir` | str | "./eval_plots" | 图表输出目录 |
| `--qdrant-url` | str | "http://localhost:6333" | Qdrant 地址 |

### 执行流程

```
连接 & 健康检查 → 采样 (sample_queries_by_label) → F1 (compute_knn_accuracy)
→ F2 (compute_purity_recall_curve) → [F3 (compute_distance_distribution)]
→ [ANN 对比] → [JSON 导出] → [PNG 图表]
```

### JSON 导出

`np.ndarray` 和 `np.generic` 类型递归转换为 Python 原生类型。JSON 顶层结构：

```json
{
  "config": {"samples_per_class": 500, "k_f1": 10, "k_values": [...], "seed": 42},
  "f1": { ... },
  "f2": { ... },
  "f3": { ... } | null,
  "total_elapsed_sec": 142.3
}
```

## 5. WebUI 设计

### 位置

新增 expansion 面板，放在"向量检索" expansion 下方

### 异步执行

```python
async def do_evaluate():
    # 参数从 UI 控件读取
    # 所有 metrics 调用包装在 asyncio.to_thread() 中
    queries = await asyncio.to_thread(sample_queries_by_label, manager, spc, seed)
    f1 = await asyncio.to_thread(compute_knn_accuracy, manager, queries, k_f1, True,
                                  progress_callback)
    f2 = await asyncio.to_thread(compute_purity_recall_curve, manager, queries,
                                  k_values, True, progress_callback)
    # ...结果展示
```

### 进度回调（WebUI 侧）

```python
def _make_progress_callback(label: ui.label, phase: str):
    def cb(current: int, total: int):
        label.set_text(f"{phase} ({current}/{total})")
    return cb
```

### 结果展示

- **F1**：`ui.aggrid` — 列：Label / Precision / Recall / F1 / Support，行高亮 Overall Accuracy
- **F2**：`ui.echart` — 双 Y 轴折线图，X 轴 K 值（log），左轴 Purity，右轴 Recall@K
- **F3**：`ui.table` — 左侧 per-class intra stats，右侧 most confused pairs
- **导出**：`ui.button` + `ui.download()` 生成 JSON blob

## 6. 可视化模块

```python
# visualization.py
def plot_confusion_matrix(cm: np.ndarray, label_names: list[str],
                          save_path: str | Path, title: str = "Confusion Matrix") -> None
def plot_purity_recall_curve(purity_data: dict, save_path: str | Path) -> None
def plot_distance_histogram(distance_data: dict, save_path: str | Path) -> None
```

- 中文字体：`matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]`
- `plot_purity_recall_curve` 生成双面板（上下或左右）：Purity vs K + Recall@K vs K
- `plot_distance_histogram` 生成 intra 箱线图 + inter 箱线图对比

## 7. 测试策略

### 单元测试 Mock 策略

```python
# 构造 mock SearchResult 替代 PixelSearcher.search()
mock_hits = [
    HitRecord(id="uuid-1", score=0.95, label=0, label_name="water", ...),
    HitRecord(id="uuid-2", score=0.90, label=0, label_name="water", ...),
    ...
]
mock_result = SearchResult(hits=mock_hits, elapsed_ms=1.0,
                           label_distribution={"water": 2},
                           search_mode="exact", query_params={})
with patch.object(PixelSearcher, "search", return_value=mock_result):
    result = compute_knn_accuracy(manager, queries, k=3)
```

### 测试覆盖矩阵

| 测试场景 | 验证点 |
|----------|--------|
| `sample_queries_by_label` 正常 | 总数 = class×samples_per_class，label 分布均匀，shape=(64,) |
| `sample_queries_by_label` 某类不足 | actual_count < samples_per_class 的记录存在 |
| `compute_knn_accuracy` 全邻居同 label | accuracy=1.0 |
| `compute_knn_accuracy` 平票 | 递减 K 打破，结果可复现 |
| `compute_knn_accuracy` 某类 support=0 | per_class 中 support=0，P/R/F1=NaN 或 0 |
| `compute_purity_recall_curve` LOO | 自身被剔除，Purity(K=1) 可能不是 1.0 |
| `compute_purity_recall_curve` 单调性 | Purity(K₁) ≥ Purity(K₂) when K₁ < K₂ |
| `compute_purity_recall_curve` Recall@K 分母 | 非 min(K, N_same)，随 K 增长接近 1.0 |
| `compute_distance_distribution` 相同向量 | cos_dist=0 |
| `compute_distance_distribution` 正交向量 | cos_dist=1 |
| `compute_distance_distribution` 分离数据 | intra_mean < inter_mean |
| 边界：空 Collection | ValueError 含明确消息 |
| 边界：某类无像素 | 该类 actual_count=0，不阻塞其他类 |
| 边界：空 k_values | ValueError |
| 集成：data_demo | `evaluate --samples-per-class 50 --k-values 10,30,50` 完整运行 |
