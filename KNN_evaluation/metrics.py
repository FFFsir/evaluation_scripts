"""KNN Embedding 质量评估指标模块.

提供分层采样、KNN 分类准确率 (F1)、邻居纯度/Recall@K (F2) 评估函数。
默认使用 GPU 分块矩阵乘（KnnEngine）替代逐条 Qdrant exact 检索——
全库向量 float32 常驻 device，查询分块 matmul 取精确 Top-K，边算边回传
CPU 聚合（F1 混淆矩阵 / F2 Purity/Recall 累加器），不物化 (Q,N) 或 (Q,K)
全量结果。CUDA 不可用时自动回退 torch CPU 分块 / Qdrant 逐条路径。
"""
import collections
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from qdrant_client import models

from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.searcher import PixelSearcher, HitRecord
from KNN_evaluation.label_mapping import LABEL_NAMES
from KNN_evaluation.gpu_knn import KnnEngine  # torch 延迟 import；模块级便于测试 monkeypatch
from KNN_evaluation.sampling_map import ensure_sampling_map  # 采样地图：采样时精确取样本

ProgressCallback = Callable[[int, int], None] | None


class EvaluationCancelled(Exception):
    """评估被用户中止（协作式取消令牌命中）时抛出.

    与 asyncio.CancelledError 区分：由 `cancel_event`（threading.Event）触发，
    WebUI 捕获后通知用户并复位进度 UI；engine.close() 由 try/finally 保证执行。
    """


def _check_cancel(cancel_event: threading.Event | None) -> None:
    """分块循环每块检查取消令牌；命中即抛 EvaluationCancelled.

    缺省 None（旧调用方）时直接返回，保持向后兼容。
    """
    if cancel_event is not None and cancel_event.is_set():
        raise EvaluationCancelled("评估已中止")


def _resolve_tie(votes: list[int], hits: list[HitRecord] | None = None) -> int:
    """递减 K 打破平票。

    按 K, K-1, K-3, K-5, K-7, K-9 依次重新计票，
    若至 K<=0 仍未打破，回退到最近邻居的标签。
    hits 仅用于回退取最近邻居标签；批量路径无 HitRecord 时传 None，
    回退到 votes[0]（votes 按相似度降序，第一项即最近邻居）。
    """
    fallback = votes[0] if hits is None else hits[0].label
    for step in (0, 1, 3, 5, 7, 9):
        k_sub = len(votes) - step
        if k_sub <= 0:
            return fallback
        counter = collections.Counter(votes[:k_sub])
        max_count = max(counter.values())
        winners = [l for l, c in counter.items() if c == max_count]
        if len(winners) == 1:
            return winners[0]
    return fallback


def sample_queries_by_label(
    manager: QdrantManager,
    samples_per_class: int = 500,
    seed: int = 42,
    warn_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """按标签分层随机采样查询像素。

    优化（Section 10.4）：改用**采样地图 manifest**（`qdrant_sampling_map.json`）
    替代全量 scroll 下载全向量。流程：读地图（`ensure_sampling_map`，自动对账
    重建）→ 每类 `rng.sample` 选 point_id → `client.retrieve(ids)` 批量取向量。
    只下载样本量（如 9 类 × 5000 = 45000 个 ≈ 11MB），不再全量下载 2.6GB 向量。

    timeout 修复：采样前检查向量索引状态（`indexed_vectors_count < total_points`
    表示 HNSW 索引构建未完成，读操作退化为磁盘扫描导致服务端响应极慢、旧客户端
    5s 超时）；未就绪时经 warn_callback 明确提示已索引比例，不静默等待/超时失败。

    Args:
        manager: QdrantManager 实例。
        samples_per_class: 每类目标采样数。
        seed: 随机种子，保证可复现。
        warn_callback: 可选警告回调；索引未就绪等非致命告警经它输出。
            缺省 None 时用 print 输出（CLI 行为）。

    Returns:
        查询像素列表，每项 dict 含 vector, label, label_name,
        image_id, pixel_row, pixel_col, point_id, actual_count。
        某类在地图中无 ID 时保留 {"label", "label_name", "actual_count": 0, "vectors": []} 空标记。

    Raises:
        ConnectionError: Qdrant 不可达。
        ValueError: Collection 为空。
        RuntimeError: 采样地图缺失且构建失败（by_label 全空但 collection 非空）。
    """
    # 检查连接
    if not manager.health_check():
        raise ConnectionError(f"Qdrant 不可达: {manager.url}")

    if not manager.collection_exists():
        raise ValueError(f"Collection '{manager.collection_name}' 不存在，请先执行 import")

    info = manager.collection_info()
    if info.get("total_points", 0) == 0:
        raise ValueError("Collection 为空，请先导入数据")

    # 索引未就绪预警：HNSW 索引构建未完成时读操作极慢，明确提示已索引比例
    indexed = info.get("vectors_count", 0) or 0
    total = info.get("total_points", 0) or 0
    if total > 0 and indexed < total:
        warn = warn_callback or print
        warn(
            f"向量索引构建中（已索引 {indexed:,} / 总点数 {total:,}），"
            f"未索引部分读操作较慢，采样/评估可能耗时较长"
        )

    # 读采样地图（自动对账：指纹一致返回缓存，否则重建）
    sampling_map = ensure_sampling_map(manager)
    by_label: dict[int, list[str]] = sampling_map.get("by_label") or {}

    # 地图构建失败且 collection 非空（by_label 全空）→ 抛明确错误
    if not any(by_label.values()) and info.get("total_points", 0) > 0:
        raise RuntimeError(
            "采样地图为空且构建失败：无法从 Qdrant 读取 point_id→label 地图，"
            f"请检查 collection '{manager.collection_name}' 与本地 qdrant_sampling_map.json"
        )

    rng = random.Random(seed)
    queries: list[dict] = []

    for label_id, label_name in LABEL_NAMES.items():
        class_ids = by_label.get(label_id, [])
        actual_count = min(len(class_ids), samples_per_class)
        if actual_count == 0:
            queries.append({
                "label": label_id,
                "label_name": label_name,
                "actual_count": 0,
                "vectors": [],
            })
            continue

        # 每类独立随机采样（同一 rng 顺序推进，seed 可复现）
        selected_ids = rng.sample(class_ids, actual_count)
        # 按 ID 精确取向量（只下载样本量，不下载全量）
        records = manager.client.retrieve(
            collection_name=manager.collection_name,
            ids=selected_ids,
            with_payload=True,
            with_vectors=True,
        )
        for rec in records:
            payload = rec.payload or {}
            queries.append({
                "vector": np.array(rec.vector, dtype=np.float64),
                "label": label_id,
                "label_name": label_name,
                "image_id": str(payload.get("image_id", "")),
                "pixel_row": int(payload.get("pixel_row", -1)),
                "pixel_col": int(payload.get("pixel_col", -1)),
                "point_id": str(rec.id),
                "actual_count": actual_count,
            })

    return queries


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
    cancel_event: threading.Event | None = None,
) -> dict:
    """计算 KNN 分类准确率（Leave-One-Out）+ Per-class F1 + accuracy_by_k 多 K.

    k_values: 多 K 准确率取值（默认 [k]，仅计算单 K）。对每个 K 从同一份
    Top-(max_k+1) 结果递增取多数投票，零额外检索。overall_accuracy 仍 = 单 K k。

    exact=True 走 KnnEngine 分块路径（device 决定 cuda/cpu）；exact=False
    保留 PixelSearcher ANN 逐条（供 --ann 对比）。
    """
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

    # 用已解析的 engine.device（'cuda'|'cpu'）而非原始 device 参数：
    # device='auto' 在 CUDA 不可用时 KnnEngine 内部已回退 'cpu'，若仍传 'auto'
    # 会让 _device_budget 的 device=='cpu' 判断为 False，误用 max_gpu_mem 推导 block_q。
    block_q, _budget_gb = _device_budget(
        engine.device, gpu_batch_q, max_gpu_mem, max_eval_ram, engine,
    )

    query_vectors = np.array([q["vector"] for q in valid_queries], dtype=np.float32)
    query_point_ids = [q["point_id"] for q in valid_queries]
    query_labels = [q["label"] for q in valid_queries]

    # 多 K 累加器：每个 K 一张混淆矩阵（同一份 Top-K 结果递增取值）
    confusion_by_k: dict[int, np.ndarray] = {
        kv: np.zeros((num_labels, num_labels), dtype=np.int64) for kv in k_values
    }

    done = 0
    try:
        for start_i in range(0, Q, block_q):
            end_i = min(start_i + block_q, Q)
            _check_cancel(cancel_event)  # 每块开始前检查
            _scores, topk_labels, topk_ids = engine.knn_chunk(
                query_vectors[start_i:end_i], max_k + 1,
            )
            _check_cancel(cancel_event)  # 每块计算后检查
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
    finally:
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


def _compute_per_class_label_totals(manager: QdrantManager) -> dict[str, int]:
    """按 label 统计 Qdrant 中各类别的全局像素总数，用于 Recall@K 分母。

    调用 Qdrant count API 逐类精确统计。

    Returns:
        {label_name: count} 字典，如 {"water": 20000, "trees": 15000, ...}
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


def _accumulate_purity_recall_row(
    topk_labels, topk_ids, query_point_id, query_label_name,
    sorted_k, label_totals,
    purity_sums, recall_sums,
    per_class_purity_sums, per_class_recall_sums, class_counts,
) -> bool:
    """对单行 Top-K 递增累加 Purity/Recall；返回是否计入 valid_count.

    调用方可能用 ThreadPoolExecutor 并行调用本函数（每行一个任务），因此
    只写入各累加器 dict 内不同 key，不做跨行共享的原地增量。
    """
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


# K 值序列默认值：从 10 起步避免 Recall@K 在 N_same 极大时几乎为零
DEFAULT_K_VALUES = [10, 20, 50, 100, 300, 1000]


def compute_purity_recall_curve(
    manager: QdrantManager,
    queries: list[dict],
    k_values: list[int] | None = None,
    exact: bool = True,
    device: str = "auto",
    gpu_batch_q: int | None = None,
    max_gpu_mem: float = 16,
    max_eval_ram: float = 6.0,
    progress_callback: ProgressCallback = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """计算不同 K 值下的邻居纯度（Purity@K）和召回率（Recall@K）。

    Purity(K) = (1/N) × Σᵢ (Kᵢ_same / K)
      衡量 Top-K 邻居的标签同质性。随 K 增大单调递减。

    Recall@K = (1/N) × Σᵢ (Kᵢ_same / N_same_class(i))
      衡量前 K 个邻居能覆盖全量同类像素的比例。随 K 增大单调递增。
      分母为全局同类总数（Qdrant count exact），非 min(K, N_same)。

    优化策略：对每个查询像素仅检索一次 max(k_values)+1 个最近邻，
    从结果中递增取不同 K 值计算，避免重复检索。GPU 路径用 KnnEngine
    分块 matmul 得到精确 Top-K，逐块回传 CPU 累加器。

    Args:
        manager: QdrantManager 实例。
        queries: 查询像素列表。
        k_values: K 值列表，默认 [10, 20, 50, 100, 300, 1000]。
        exact: True 使用暴力精确搜索（GPU 分块 / Qdrant 回退路径）；False 走 ANN 逐条。
        device: 计算设备 "auto" | "cuda" | "cpu"。
        gpu_batch_q: 显式查询块大小；None 按预算推导。
        max_gpu_mem: 显存预算（GB）。
        max_eval_ram: CPU 路径内存预算（GB）。
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

    label_totals = _compute_per_class_label_totals(manager)

    start = time.perf_counter()

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

    from KNN_evaluation.label_mapping import LABEL_NAMES as _LN

    engine = KnnEngine(device=device)
    engine.set_corpus(all_vecs)
    engine._attach_meta(all_lbls, all_ids)

    # 用已解析的 engine.device（'cuda'|'cpu'）而非原始 device 参数：
    # device='auto' 在 CUDA 不可用时 KnnEngine 内部已回退 'cpu'，若仍传 'auto'
    # 会让 _device_budget 的 device=='cpu' 判断为 False，误用 max_gpu_mem 推导 block_q。
    block_q, _budget_gb = _device_budget(
        engine.device, gpu_batch_q, max_gpu_mem, max_eval_ram, engine,
    )

    query_vectors = np.array([q["vector"] for q in valid_queries], dtype=np.float32)
    query_point_ids = [q["point_id"] for q in valid_queries]
    query_label_names = [q["label_name"] for q in valid_queries]

    purity_sums: dict[int, float] = {kv: 0.0 for kv in sorted_k}
    recall_sums: dict[int, float] = {kv: 0.0 for kv in sorted_k}
    per_class_purity_sums: dict[str, dict[int, float]] = {
        ln: {kv: 0.0 for kv in sorted_k} for ln in _LN.values()
    }
    per_class_recall_sums: dict[str, dict[int, float]] = {
        ln: {kv: 0.0 for kv in sorted_k} for ln in _LN.values()
    }
    class_counts: dict[str, int] = {ln: 0 for ln in _LN.values()}

    valid_count = 0
    done = 0
    max_workers = min(8, Q)
    try:
        for start_i in range(0, Q, block_q):
            end_i = min(start_i + block_q, Q)
            _check_cancel(cancel_event)  # 每块开始前检查
            _scores, topk_labels, topk_ids = engine.knn_chunk(
                query_vectors[start_i:end_i], max_k + 1,
            )
            _check_cancel(cancel_event)  # 每块计算后检查

            if max_workers > 1 and (end_i - start_i) > 1:
                # 每行独立累加器：避免共享 dict 在多线程下并发 += 丢失更新，
                # 各线程只写自己那行的 dict（无共享可变状态），结束后主线程串行合并。
                row_accs = []
                for _row in range(end_i - start_i):
                    row_accs.append((
                        {kv: 0.0 for kv in sorted_k},           # row purity_sums
                        {kv: 0.0 for kv in sorted_k},           # row recall_sums
                        {ln: {kv: 0.0 for kv in sorted_k} for ln in _LN.values()},
                        {ln: {kv: 0.0 for kv in sorted_k} for ln in _LN.values()},
                        {ln: 0 for ln in _LN.values()},          # row class_counts
                    ))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [
                        ex.submit(
                            _accumulate_purity_recall_row,
                            topk_labels[row], topk_ids[row],
                            query_point_ids[start_i + row], query_label_names[start_i + row],
                            sorted_k, label_totals, *(row_accs[row]),
                        )
                        for row in range(end_i - start_i)
                    ]
                    for f, (rp, rr, rpp, rpr, rcc) in zip(futures, row_accs):
                        if f.result():
                            valid_count += 1
                        # 主线程串行合并每行累加器（无并发，安全）
                        for kv in sorted_k:
                            purity_sums[kv] += rp[kv]
                            recall_sums[kv] += rr[kv]
                            for ln in _LN.values():
                                per_class_purity_sums[ln][kv] += rpp[ln][kv]
                                per_class_recall_sums[ln][kv] += rpr[ln][kv]
                        for ln in _LN.values():
                            class_counts[ln] += rcc[ln]
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
    finally:
        engine.close()
    return _finalize_purity_recall(sorted_k, _LN, class_counts,
                                   purity_sums, recall_sums,
                                   per_class_purity_sums, per_class_recall_sums,
                                   valid_count, start)


def evaluate_knn(
    manager: QdrantManager,
    queries: list[dict],
    k_f1: int,
    k_values: list[int],
    exact: bool = True,
    device: str = "auto",
    gpu_batch_q: int | None = None,
    max_gpu_mem: float = 16,
    max_eval_ram: float = 6.0,
    progress_callback: ProgressCallback = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """联合评估：单次 scroll + 单次 top-(max_k+1)，同时聚合 F1（多 K）+ F2.

    verify 缺陷修复：CLI/WebUI 原本顺序调用 compute_knn_accuracy 与
    compute_purity_recall_curve，导致两次全量 scroll（各数分钟）与两次独立 topk，
    且 F2 重新 set_corpus 叠加未释放显存 → OOM。本函数对同一查询集单次 scroll
    全量向量 + 单次 `knn_chunk(query_block, max_k+1)`（max_k = max(k_values + [k_f1])），
    每块 Top-K 结果同时聚合 F1（多 K 混淆矩阵）与 F2（Purity/Recall 累加器），
    零重复下载、零重复检索。`engine.close()` 置于 try/finally，异常时也释放显存。

    Args:
        manager: QdrantManager 实例。
        queries: 查询像素列表。
        k_f1: F1 KNN 分类器邻居数（overall_accuracy 对应的单 K）。
        k_values: F2 Purity/Recall 曲线 K 值序列（缺省 DEFAULT_K_VALUES）。
            f2 输出只用本序列（与 compute_purity_recall_curve 一致）；k_f1 只并入
            F1 多 K 与 max_k 计算（单次 top-(max_k+1) 复用），不并入 F2 输出。
        exact: True 走 KnnEngine 分块精确路径；False 走 Qdrant 服务端 ANN 逐条。
        device / gpu_batch_q / max_gpu_mem / max_eval_ram: 与 compute_* 一致。
        progress_callback: 进度回调 (current, total)。
        cancel_event: 可选取消令牌（threading.Event）。分块循环每块检查
            is_set()，命中即抛 EvaluationCancelled（engine.close() 由 try/finally
            保证释放显存）。缺省 None 时不检查，保持向后兼容。

    Returns:
        {"f1": <compute_knn_accuracy 结构>, "f2": <compute_purity_recall_curve 结构>}
    """
    from KNN_evaluation.label_mapping import LABEL_NAMES as _LN

    num_labels = len(_LN)
    start = time.perf_counter()

    # F2 输出/累加器只用用户传入 k_values（与 compute_purity_recall_curve 一致，
    # 缺省用 DEFAULT_K_VALUES）；k_f1 不并入 F2，避免 f2 多出 k_f1 点违反差分一致性。
    f2_ks = sorted(k_values) if k_values is not None else sorted(DEFAULT_K_VALUES)
    # F1 多 K + max_k 用合并集：覆盖 k_f1 与用户 k_values，单次 top-(max_k+1) 全部复用
    all_ks = sorted(set([k_f1] + f2_ks))
    max_k = max(all_ks)

    if not exact:
        # ANN 逐条路径无共享 topk：退化为两个独立逐条函数
        label_totals = _compute_per_class_label_totals(manager)
        f1 = _knn_accuracy_sequential(manager, queries, k_f1, False, progress_callback)
        f2 = _purity_recall_sequential(
            manager, queries, f2_ks, label_totals, False, progress_callback,
        )
        return {"f1": f1, "f2": f2}

    valid_queries = [q for q in queries if q.get("vectors") is None]
    Q = len(valid_queries)
    if Q == 0:
        confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
        return {
            "f1": _empty_knn_accuracy_result(confusion, k_f1, start),
            "f2": _empty_purity_recall_result(f2_ks, start),
        }

    if progress_callback:
        progress_callback(0, Q)

    label_totals = _compute_per_class_label_totals(manager)

    all_vecs, all_lbls, all_ids = _scroll_full_vectors(manager)
    if all_vecs.shape[0] == 0:
        # 空 Collection 回退 Qdrant 服务端逐条（与两个独立函数的行为一致）
        f1 = _knn_accuracy_sequential(manager, queries, k_f1, True, progress_callback)
        f2 = _purity_recall_sequential(
            manager, queries, f2_ks, label_totals, True, progress_callback,
        )
        return {"f1": f1, "f2": f2}

    # F1 累加器：每 K 一张混淆矩阵（同一份 Top-K 结果递增取值，含 k_f1）
    confusion_by_k: dict[int, np.ndarray] = {
        kv: np.zeros((num_labels, num_labels), dtype=np.int64) for kv in all_ks
    }
    # F2 累加器：只用用户 k_values（f2_ks）
    purity_sums: dict[int, float] = {kv: 0.0 for kv in f2_ks}
    recall_sums: dict[int, float] = {kv: 0.0 for kv in f2_ks}
    per_class_purity_sums: dict[str, dict[int, float]] = {
        ln: {kv: 0.0 for kv in f2_ks} for ln in _LN.values()
    }
    per_class_recall_sums: dict[str, dict[int, float]] = {
        ln: {kv: 0.0 for kv in f2_ks} for ln in _LN.values()
    }
    class_counts: dict[str, int] = {ln: 0 for ln in _LN.values()}

    engine = KnnEngine(device=device)
    engine.set_corpus(all_vecs)
    engine._attach_meta(all_lbls, all_ids)

    # 用已解析的 engine.device（'cuda'|'cpu'）而非原始 device 参数（同 compute_* 的说明）
    block_q, _budget_gb = _device_budget(
        engine.device, gpu_batch_q, max_gpu_mem, max_eval_ram, engine,
    )

    query_vectors = np.array([q["vector"] for q in valid_queries], dtype=np.float32)
    query_point_ids = [q["point_id"] for q in valid_queries]
    query_labels = [q["label"] for q in valid_queries]
    query_label_names = [q["label_name"] for q in valid_queries]

    valid_count = 0
    done = 0
    max_workers = min(8, Q)
    try:
        for start_i in range(0, Q, block_q):
            end_i = min(start_i + block_q, Q)
            _check_cancel(cancel_event)  # 每块开始前检查（knn_chunk 前）
            _scores, topk_labels, topk_ids = engine.knn_chunk(
                query_vectors[start_i:end_i], max_k + 1,
            )
            _check_cancel(cancel_event)  # 每块计算后检查（knn_chunk 后）
            block_n = end_i - start_i

            # F1：逐行多 K 多数投票 → 每 K 一张混淆矩阵（串行写共享数组，避免并发 += 丢失）
            for row in range(block_n):
                i = start_i + row
                for kv in all_ks:
                    _aggregate_knn_row(
                        topk_labels[row], topk_ids[row],
                        query_point_ids[i], query_labels[i], kv,
                        num_labels, confusion_by_k[kv],
                    )

            # F2：逐行 Purity/Recall 累加（并行时每行独立累加器，主线程串行合并）
            if max_workers > 1 and block_n > 1:
                row_accs = []
                for _row in range(block_n):
                    row_accs.append((
                        {kv: 0.0 for kv in f2_ks},           # row purity_sums
                        {kv: 0.0 for kv in f2_ks},           # row recall_sums
                        {ln: {kv: 0.0 for kv in f2_ks} for ln in _LN.values()},
                        {ln: {kv: 0.0 for kv in f2_ks} for ln in _LN.values()},
                        {ln: 0 for ln in _LN.values()},          # row class_counts
                    ))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [
                        ex.submit(
                            _accumulate_purity_recall_row,
                            topk_labels[row], topk_ids[row],
                            query_point_ids[start_i + row], query_label_names[start_i + row],
                            f2_ks, label_totals, *(row_accs[row]),
                        )
                        for row in range(block_n)
                    ]
                    for f, (rp, rr, rpp, rpr, rcc) in zip(futures, row_accs):
                        if f.result():
                            valid_count += 1
                        for kv in f2_ks:
                            purity_sums[kv] += rp[kv]
                            recall_sums[kv] += rr[kv]
                            for ln in _LN.values():
                                per_class_purity_sums[ln][kv] += rpp[ln][kv]
                                per_class_recall_sums[ln][kv] += rpr[ln][kv]
                        for ln in _LN.values():
                            class_counts[ln] += rcc[ln]
            else:
                for row in range(block_n):
                    if _accumulate_purity_recall_row(
                        topk_labels[row], topk_ids[row],
                        query_point_ids[start_i + row], query_label_names[start_i + row],
                        f2_ks, label_totals, purity_sums, recall_sums,
                        per_class_purity_sums, per_class_recall_sums, class_counts,
                    ):
                        valid_count += 1

            done = end_i
            if progress_callback:
                progress_callback(done, Q)
    finally:
        engine.close()

    # 多 K per-class Recall：对每张 K 混淆矩阵取各类别召回率（tp/(tp+fn)），
    # 供 WebUI 绘制"不同 K 下各类别 Recall"曲线（区分度高于全量分母的 Recall@K）。
    per_class_recall_by_k: dict[int, dict[str, float]] = {}
    for kv in all_ks:
        per_class_recall_by_k[kv] = {
            ln: _per_class_metrics_from_confusion(confusion_by_k[kv], _LN)[ln]["recall"]
            for ln in _LN.values()
        }

    f1_result = {
        "overall_accuracy": round(float(_confusion_accuracy(confusion_by_k[k_f1])), 4),
        "per_class_metrics": _per_class_metrics_from_confusion(confusion_by_k[k_f1], _LN),
        "confusion_matrix": confusion_by_k[k_f1],
        "k": k_f1,
        "num_queries": Q,
        "elapsed_sec": round(time.perf_counter() - start, 2),
        "accuracy_by_k": {
            kv: round(float(_confusion_accuracy(confusion_by_k[kv])), 4)
            for kv in all_ks
        },
        "per_class_recall_by_k": per_class_recall_by_k,
    }
    f2_result = _finalize_purity_recall(
        f2_ks, _LN, class_counts, purity_sums, recall_sums,
        per_class_purity_sums, per_class_recall_sums, valid_count, start,
    )
    return {"f1": f1_result, "f2": f2_result}


def _finalize_purity_recall(sorted_k, label_names, class_counts,
                            purity_sums, recall_sums,
                            per_class_purity_sums, per_class_recall_sums,
                            valid_count, start) -> dict:
    """汇总 Purity/Recall 累加器为结果 dict（分块与逐条路径共用）."""
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
    """无有效查询时的空结果 dict."""
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
    max_k = sorted_k[-1]
    total_queries = len(queries)
    processed = 0
    start = time.perf_counter()

    purity_sums: dict[int, float] = {k: 0.0 for k in sorted_k}
    recall_sums: dict[int, float] = {k: 0.0 for k in sorted_k}
    per_class_purity_sums: dict[str, dict[int, float]] = {
        ln: {k: 0.0 for k in sorted_k} for ln in _LN.values()
    }
    per_class_recall_sums: dict[str, dict[int, float]] = {
        ln: {k: 0.0 for k in sorted_k} for ln in _LN.values()
    }
    class_counts: dict[str, int] = {ln: 0 for ln in _LN.values()}
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
        processed += 1

        if progress_callback:
            progress_callback(processed, total_queries)

    return _finalize_purity_recall(sorted_k, _LN, class_counts,
                                   purity_sums, recall_sums,
                                   per_class_purity_sums, per_class_recall_sums,
                                   valid_count, start)


def _scroll_full_vectors(manager: QdrantManager) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 Qdrant 全量 scroll 向量、标签、point_id 到客户端内存（磁盘缓存加速）.

    通过分页 scroll（每批 50000 条）遍历整个 Collection，返回所有向量排列成
    (N, 64) 数组及对应的标签和 ID。向量转 float32 以适配 KnnEngine 显存驻留。

    Section 11：**惰性磁盘缓存**——首次下载全量后写入 `qdrant_corpus_cache/`
    npz 缓存；后续调用先查缓存（collection 名称 + total_points 指纹一致则直接
    `np.load` 加载），避免同一 collection 重复评估时重复下载 2.6GB（约 205 批
    scroll，3-4 分钟）。指纹不一致 / 文件缺失 / 损坏时自动重建。CLI/WebUI
    无需新命令，本函数惰性触发。与采样地图（采样定位）互补：缓存管评估全量。

    Returns:
        vectors: (N, 64) float32  全量向量矩阵，空时 (0, 64)
        labels: (N,) int64        全量标签数组
        point_ids: (N,) str       全量 point_id 数组
    """
    from KNN_evaluation.corpus_cache import CORPUS_CACHE_DIR, ensure_corpus_cache

    return ensure_corpus_cache(manager, CORPUS_CACHE_DIR)
