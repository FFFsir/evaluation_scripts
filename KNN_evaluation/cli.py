"""Qdrant KNN 评估系统 — 命令行入口.

用法:
    python -m KNN_evaluation.cli import <directory> [--batch-size N] [--no-resume] [--reindex]
    python -m KNN_evaluation.cli search --query-file <path> [--k N] [--label ...] [--utm-range ...] [--exact]
    python -m KNN_evaluation.cli stats [--json]
    python -m KNN_evaluation.cli evaluate [--samples-per-class N] [--k-f1 N] [--k-values ...] [--ann] [--seed N] [--output PATH] [--plot] [--plot-dir DIR] [--device cuda|cpu|auto] [--gpu-batch-q N] [--max-gpu-mem GB] [--max-eval-ram GB]
    python -m KNN_evaluation.cli migrate [--dir DIR] [--storage disk|ram] [--no-resume] [--qdrant-url URL]
    python -m KNN_evaluation.cli similarity-heatmap [--n N] [--seed N] [--image-id ID] [--output PATH] [--export-dir DIR] [--google-collection NAME] [--xian-collection NAME] [--qdrant-url URL]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from qdrant_client.http import exceptions as qdrant_http_exceptions

from KNN_evaluation.config import QDRANT_URL, DEFAULT_COLLECTION, BATCH_SIZE
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.searcher import PixelSearcher


def _parse_utm_range(utm_str: str) -> dict:
    """解析 UTM 范围字符串.

    Args:
        utm_str: 格式 "min_e,max_e,min_n,max_n" 如 "500000,501000,4000000,4001000".

    Returns:
        dict with min_e/max_e/min_n/max_n keys.

    Raises:
        ValueError: 格式不正确.
    """
    parts = utm_str.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"UTM 范围格式应为 'min_e,max_e,min_n,max_n'，实际: {utm_str}"
        )
    return {
        "min_e": float(parts[0]),
        "max_e": float(parts[1]),
        "min_n": float(parts[2]),
        "max_n": float(parts[3]),
    }


def _parse_positive_n(value: str) -> int:
    """argparse type 校验：采样数 N 必须在 1..600（与 similarity_compare.MAX_N 同步）."""
    from KNN_evaluation.similarity_compare import MAX_N

    n = int(value)
    if not (1 <= n <= MAX_N):
        raise argparse.ArgumentTypeError(f"n 必须在 1..{MAX_N} 之间，实际: {n}")
    return n


def _parse_label(label_str: str) -> list[int]:
    """解析标签字符串.

    Args:
        label_str: 逗号分隔的标签值或名称，如 "0,1,2" 或 "water,trees".

    Returns:
        标签整数列表.
    """
    from KNN_evaluation.label_mapping import LABEL_IDS

    values: list[int] = []
    for part in label_str.split(","):
        part = part.strip()
        if part.isdigit():
            values.append(int(part))
        elif part in LABEL_IDS:
            values.append(LABEL_IDS[part])
        else:
            raise ValueError(f"未知标签: {part}")
    return values


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


def cmd_import(args) -> int:
    """执行 import 子命令."""
    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)

    if not manager.collection_exists():
        print(f"Collection '{manager.collection_name}' 不存在，正在创建...")
        manager.create_collection()
        manager.create_payload_indices()
        print("Collection 创建完成.")

    importer = PixelImporter(manager, batch_size=args.batch_size)

    # 像素级进度条：首个回调时创建（此时才知道总像素数），结束时关闭
    pbar = None

    def progress(imported, total):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="导入像素", unit="px")
        pbar.n = imported
        pbar.refresh()

    try:
        stats = importer.import_directory(
            data_dir=Path(args.directory),
            no_resume=args.no_resume,
            reindex=args.reindex,
            progress_callback=progress,
        )
    except ConnectionError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except qdrant_http_exceptions.UnexpectedResponse as e:
        print(f"错误: Qdrant 返回异常 (HTTP {e.status_code}): {e}", file=sys.stderr)
        return 1
    except qdrant_http_exceptions.ResponseHandlingException as e:
        print(f"错误: Qdrant 连接异常: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — 兜底：任何未预期异常输出干净错误而非裸 traceback
        print(f"错误: 导入失败: {e}", file=sys.stderr)
        return 1
    finally:
        if pbar is not None:
            pbar.close()

    # 输出统计
    print(f"\n{'='*50}")
    print(f"导入完成")
    print(f"{'='*50}")
    print(f"总像素数:     {stats['total_pixels']:,}")
    print(f"总影像数:     {stats['total_images']}")
    print(f"新导入影像:   {stats['imported_images']}")
    print(f"跳过(已存在):  {stats['skipped_images']}")
    print(f"总耗时:       {stats['elapsed_sec']:.1f}s")
    print(f"平均速率:     {stats['rate_pps']:.0f} 像素/秒")
    print(f"\n标签分布:")
    for name in sorted(stats["label_counts"].keys()):
        cnt = stats["label_counts"][name]
        pct = cnt / max(stats["total_pixels"], 1) * 100
        print(f"  {name:<22} {cnt:>10,}  ({pct:>5.1f}%)")
    print(f"\nCollection 当前总量: {manager.collection_info()['total_points']:,}")

    return 0


def cmd_search(args) -> int:
    """执行 search 子命令."""
    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)

    if not manager.collection_exists():
        print(f"错误: Collection '{manager.collection_name}' 不存在，请先执行 import", file=sys.stderr)
        return 1

    # 获取 query vector
    if args.query_file:
        query_vector = np.load(args.query_file)
        if query_vector.ndim == 2:
            query_vector = query_vector.squeeze(axis=0)
    elif args.random:
        # 从 collection 中随机取一个 vector
        scroll_result = manager.client.scroll(
            collection_name=manager.collection_name,
            limit=1,
            with_vectors=True,
        )
        if not scroll_result[0]:
            print("错误: Collection 为空，无法随机获取 query vector", file=sys.stderr)
            return 1
        query_vector = np.array(scroll_result[0][0].vector, dtype=np.float64)
        print(f"随机选取 query point: {scroll_result[0][0].id}")
    elif args.query_spec:
        image_id, row, col = args.query_spec
        from qdrant_client import models
        scroll_result = manager.client.scroll(
            collection_name=manager.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="image_id", match=models.MatchValue(value=image_id)),
                    models.FieldCondition(key="pixel_row", match=models.MatchValue(value=int(row))),
                    models.FieldCondition(key="pixel_col", match=models.MatchValue(value=int(col))),
                ],
            ),
            limit=1,
            with_vectors=True,
        )
        if not scroll_result[0]:
            print(f"错误: 未找到像素 {image_id}_{row}_{col}", file=sys.stderr)
            return 1
        query_vector = np.array(scroll_result[0][0].vector, dtype=np.float64)
    else:
        print("错误: 必须指定 --query-file、--random 或 --query-spec", file=sys.stderr)
        return 1

    # 解析过滤条件
    label_filter = None
    if args.label:
        label_filter = _parse_label(args.label)

    utm_range = None
    if args.utm_range:
        utm_range = _parse_utm_range(args.utm_range)

    # 执行搜索
    searcher = PixelSearcher(manager)
    try:
        result = searcher.search(
            query_vector=query_vector,
            k=args.k,
            label_filter=label_filter,
            utm_range=utm_range,
            exact=args.exact,
            ef_search=args.ef_search,
        )
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    # 输出结果
    if args.output == "json":
        hits_json = [
            {
                "id": h.id, "score": h.score, "label": h.label,
                "label_name": h.label_name, "utm_easting": h.utm_easting,
                "utm_northing": h.utm_northing, "utm_zone": h.utm_zone,
                "image_id": h.image_id, "pixel_row": h.pixel_row,
                "pixel_col": h.pixel_col,
            }
            for h in result.hits
        ]
        print(json.dumps({
            "hits": hits_json,
            "elapsed_ms": result.elapsed_ms,
            "label_distribution": result.label_distribution,
            "search_mode": result.search_mode,
            "query_params": result.query_params,
        }, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*70}")
        print(f"检索结果 (mode={result.search_mode}, k={result.query_params['k']}, "
              f"elapsed={result.elapsed_ms:.1f}ms)")
        print(f"{'='*70}")
        print(f"{'#':>4} {'ID':<40} {'Score':>8} {'Label':<22} {'Easting':>12} {'Northing':>12}")
        print(f"{'-'*4} {'-'*40} {'-'*8} {'-'*22} {'-'*12} {'-'*12}")
        for i, h in enumerate(result.hits, 1):
            print(f"{i:>4} {h.id:<40} {h.score:>8.4f} {h.label_name:<22} "
                  f"{h.utm_easting:>12.1f} {h.utm_northing:>12.1f}")

        if result.label_distribution:
            print(f"\n标签分布: {result.label_distribution}")

    return 0


def cmd_stats(args) -> int:
    """执行 stats 子命令."""
    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)

    if not manager.collection_exists():
        print(f"错误: Collection '{manager.collection_name}' 不存在", file=sys.stderr)
        return 1

    info = manager.collection_info()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"Collection: {manager.collection_name}")
        print(f"  总点数:    {info['total_points']:,}")
        print(f"  向量数:    {info['vectors_count']:,}")
        print(f"  分段数:    {info['segments_count']}")
        print(f"  状态:      {info['status']}")

    return 0


def cmd_similarity_heatmap(args) -> int:
    """执行 similarity-heatmap 子命令：双集合相似度热力图对比.

    流程：健康检查 → 双集合存在性检查 → 调 `compare_similarity_heatmaps`
    （数据库/图片模式由 --image-id 决定）→ 打印输出路径与 sampled/kept/dropped。
    """
    from KNN_evaluation.similarity_compare import compare_similarity_heatmaps

    g_manager = QdrantManager(url=args.qdrant_url, collection_name=args.google_collection)
    x_manager = QdrantManager(url=args.qdrant_url, collection_name=args.xian_collection)

    if not g_manager.health_check():
        print(f"错误: Qdrant 不可达 ({args.qdrant_url})", file=sys.stderr)
        return 1
    if not g_manager.collection_exists():
        print(f"错误: Collection '{args.google_collection}' 不存在，请先执行 import", file=sys.stderr)
        return 1
    if not x_manager.collection_exists():
        print(f"错误: Collection '{args.xian_collection}' 不存在，请先执行 import", file=sys.stderr)
        return 1

    try:
        # Minor 3：--export-dir 纯空白串先 strip（空串禁用由 compare 层 if export_dir: 归一化），
        # 避免把 "  " 当 truthy 目录创建带空格目录。
        export_dir = args.export_dir.strip() if args.export_dir else args.export_dir
        result = compare_similarity_heatmaps(
            g_manager, x_manager,
            n=args.n, seed=args.seed, image_id=args.image_id,
            output=args.output,
            collection_names=(args.google_collection, args.xian_collection),
            export_dir=export_dir,
        )
    except (ValueError, RuntimeError, ConnectionError, OSError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except qdrant_http_exceptions.UnexpectedResponse as e:
        print(f"错误: Qdrant 返回异常 (HTTP {e.status_code}): {e}", file=sys.stderr)
        return 1
    except qdrant_http_exceptions.ResponseHandlingException as e:
        print(f"错误: Qdrant 连接异常: {e}", file=sys.stderr)
        return 1

    print(f"{'='*60}")
    print("相似度热力图对比完成")
    print(f"{'='*60}")
    print(f"采样点:   {result['sampled']}")
    print(f"保留点:   {result['kept']}  (剔除 {result['dropped']})")
    print(f"矩阵:     {result['matrix_shape'][0]}×{result['matrix_shape'][1]}")
    print(f"耗时:     {result['elapsed_sec']:.2f}s")
    print(f"输出:     {result['output_path']}")
    for path in result.get("exported_files", []):
        print(f"导出:     {path}")
    return 0


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
    """执行 evaluate 子命令.

    --device 决定评估执行设备（cuda/cpu/auto）；auto 在 CUDA 不可用时回退 CPU
    分块路径。F1 输出 Overall Accuracy + 多 K 表（accuracy_by_k）。JSON 导出
    的 config 块包含 device/gpu_batch_q/max_gpu_mem/max_eval_ram。
    Section 8: 改用 evaluate_knn 单次联合评估（单次 scroll + 单次 topk，F1/F2 共享）。
    """
    from KNN_evaluation.metrics import (
        sample_queries_by_label,
        evaluate_knn,
    )
    from KNN_evaluation.label_mapping import LABEL_NAMES
    from KNN_evaluation.gpu_knn import resolve_device

    # 连接 Qdrant
    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)

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
    print(f"\n正在采样查询像素 (每类 {args.samples_per_class})...")
    try:
        queries = sample_queries_by_label(manager, args.samples_per_class, args.seed)
    except (ConnectionError, ValueError, RuntimeError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    num_queries = sum(1 for q in queries if "point_id" in q)
    print(f"   已采样 {num_queries} 个查询像素")

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

    # --- F1 + F2: 联合评估（单次 scroll + 单次 topk，F1/F2 共享结果） ---
    print(f"\n--- F1: KNN 分类准确率 (K={args.k_f1}, exact) ---")
    print(f"--- F2: Purity & Recall@K (exact, K={k_values}) ---")
    with _tqdm_context(num_queries, "联合评估 (F1/F2)") as pbar:
        def progress(c, t):
            pbar.update(c - pbar.n)

        try:
            combined = evaluate_knn(
                manager, queries, k_f1=args.k_f1, k_values=k_values, exact=True,
                device=resolved_device, gpu_batch_q=args.gpu_batch_q,
                max_gpu_mem=args.max_gpu_mem, max_eval_ram=args.max_eval_ram,
                progress_callback=progress,
            )
        except Exception as e:
            print(f"错误: 评估失败: {e}", file=sys.stderr)
            return 1
        f1 = combined["f1"]
        f2 = combined["f2"]

    print(f"Overall Accuracy (K={f1['k']}): {f1['overall_accuracy']:.4f}")
    acc_by_k = f1.get("accuracy_by_k") or {}
    if len(acc_by_k) > 1:
        print(f"\nAccuracy by K:")
        for kv in sorted(acc_by_k):
            print(f"  K={kv:<6} {acc_by_k[kv]:.4f}")
    print(f"\nPer-class Metrics:")
    print(f"  {'Label':<22} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    label_names = list(LABEL_NAMES.values())
    for ln in label_names:
        m = f1["per_class_metrics"].get(ln, {})
        print(f"  {ln:<22} {m.get('precision', 0):>6.4f} {m.get('recall', 0):>6.4f} "
              f"{m.get('f1', 0):>6.4f} {m.get('support', 0):>8}")
    print(f"  F1 耗时: {f1['elapsed_sec']:.1f}s")

    print(f"\n  {'K':>6} {'Purity':>8} {'Recall@K':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*12}")
    for i, kv in enumerate(f2["k_values"]):
        print(f"  {kv:>6} {f2['global_purity'][i]:>8.4f} {f2['global_recall'][i]:>12.6f}")
    print(f"  F2 耗时: {f2['elapsed_sec']:.1f}s")

    # --- ANN 对比 (可选) ---
    ann_f1 = None
    ann_f2 = None
    if args.ann:
        print(f"\n--- ANN vs Exact ---")
        from KNN_evaluation.metrics import compute_knn_accuracy, compute_purity_recall_curve
        with _tqdm_context(num_queries, "ANN F1 Accuracy") as pbar:
            def progress_a1(c, t):
                pbar.update(c - pbar.n)
            ann_f1 = compute_knn_accuracy(
                manager, queries, k=args.k_f1, exact=False,
                device=resolved_device,
                progress_callback=progress_a1,
            )
        with _tqdm_context(num_queries, "ANN F2 Purity/Recall") as pbar:
            def progress_a2(c, t):
                pbar.update(c - pbar.n)
            ann_f2 = compute_purity_recall_curve(
                manager, queries, k_values, exact=False,
                device=resolved_device,
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
                "seed": args.seed,
                "device": args.device,
                "gpu_batch_q": args.gpu_batch_q,
                "max_gpu_mem": args.max_gpu_mem,
                "max_eval_ram": args.max_eval_ram,
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
                "accuracy_by_k": f1.get("accuracy_by_k") or {},
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
        )
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            f1["confusion_matrix"], label_names,
            plot_dir / "confusion_matrix.png",
        )
        print(f"  [OK] {plot_dir / 'confusion_matrix.png'}")
        plot_purity_recall_curve(f2, plot_dir / "purity_recall_curve.png", f1)
        print(f"  [OK] {plot_dir / 'purity_recall_curve.png'}")
        print(f"\n图表已保存到: {plot_dir}")

    return 0


def _start_qdrant() -> bool:
    """幂等启动 Qdrant Docker 容器（挂 volume 持久化）.

    按需自检 + 启动，可安全重试：
    1. `docker ps -a` 全量列出（含停止容器）：
       - 运行中 → 复用；
       - 存在但停止 → `docker start`；
       - 不存在 → `docker run -v qdrant_data:/qdrant/storage` 创建并启动.
    2. 任何异常（Docker 未安装/守护进程未启动等）捕获并返回 False.

    Returns:
        True 表示容器已处于运行状态，False 表示无法确保（调用方应重试健康检查）.
    """
    name = "qdrant"
    try:
        ps = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        names = {line.strip() for line in ps.stdout.splitlines() if line.strip()}
        if name in names:
            # 容器已存在：检查是否运行中，停止则启动
            running = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10,
            )
            if name in {
                line.strip() for line in running.stdout.splitlines() if line.strip()
            }:
                return True
            start = subprocess.run(
                ["docker", "start", name],
                capture_output=True, text=True, timeout=30,
            )
            return start.returncode == 0
        run = subprocess.run(
            [
                "docker", "run", "-d", "--name", name,
                "-p", "6333:6333", "-p", "6334:6334",
                "-v", "qdrant_data:/qdrant/storage",
                "qdrant/qdrant:latest",
            ],
            capture_output=True, text=True, timeout=60,
        )
        return run.returncode == 0
    except Exception:
        return False


def _wait_for_qdrant(manager, attempts: int = 10, delay: float = 1.0) -> bool:
    """短轮询等待 Qdrant 服务就绪（docker run/start 后冷启动需数秒）."""
    for _ in range(attempts):
        if manager.health_check():
            return True
        time.sleep(delay)
    return manager.health_check()


def _container_has_volume(name: str = "qdrant") -> bool:
    """检查 Qdrant Docker 容器是否挂载 qdrant_data:/qdrant/storage 卷.

    迁移是删除重建高危操作：若删除发生在无持久化卷的容器上，迁移中断后
    容器重建即数据丢失。删除前必须确认容器带 named volume（容器重建数据
    保留）。容器不存在/未运行/无卷/命令失败均返回 False（保守失败）.

    Returns:
        True 表示容器存在且挂载了 qdrant_data 卷，False 表示无卷保障.
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Mounts}}", name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
        mounts = json.loads(result.stdout)
        for m in mounts:
            if (
                m.get("Type") == "volume"
                and (m.get("Name") or "") == "qdrant_data"
                and m.get("Destination") == "/qdrant/storage"
            ):
                return True
        return False
    except Exception:
        return False


def cmd_migrate(args) -> int:
    """重建 Collection 为指定存储配置并重导数据（幂等可重试）.

    流程：备份旧统计 → 删除重建 Collection → 重导 → 重建 manifest.
    数据安全前提：删除前必须确认 Qdrant 容器挂载持久化 volume
    （先挂卷再迁移）；无卷时 fail-fast 中止，绝不执行删除.
    """
    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)

    # 1) 确保 Qdrant 容器运行 + 挂 volume（未运行则幂等启动）
    if not manager.health_check():
        _start_qdrant()  # 幂等：确保容器挂 volume 就绪
        if not _wait_for_qdrant(manager):
            print("Qdrant 不可达", file=sys.stderr)
            return 1

    # 2) 删除前必须确认容器带持久化 volume（先挂卷再迁移，防数据丢失）
    if not _container_has_volume():
        print(
            "错误: Qdrant 容器未挂载 qdrant_data:/qdrant/storage 卷，"
            "删除重建将导致数据丢失，已中止。\n"
            "请先以带 volume 的容器重建并重跑 migrate:\n"
            "  docker rm -f qdrant\n"
            "  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \\\n"
            "    -v qdrant_data:/qdrant/storage qdrant/qdrant:latest",
            file=sys.stderr,
        )
        return 1

    try:
        old_info = manager.collection_info() if manager.collection_exists() else None
        if manager.collection_exists():
            manager.client.delete_collection(manager.collection_name)
        manager.create_collection(storage=args.storage)
        manager.create_payload_indices()
        manager.migrate_image_id_index()
        importer = PixelImporter(manager)
        importer.import_directory(
            Path(args.dir), no_resume=args.no_resume, reindex=True,
        )
        manager.reconcile_manifest()  # 重建 manifest
        new_info = manager.collection_info()
    except ConnectionError as e:
        print(f"错误: 迁移失败: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"错误: 迁移失败: {e}", file=sys.stderr)
        return 1
    except qdrant_http_exceptions.UnexpectedResponse as e:
        print(f"错误: Qdrant 返回异常 (HTTP {e.status_code}): {e}", file=sys.stderr)
        return 1
    except qdrant_http_exceptions.ResponseHandlingException as e:
        print(f"错误: Qdrant 连接异常: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — 兜底：任何未预期异常输出干净错误而非裸 traceback
        print(f"错误: 迁移失败: {e}", file=sys.stderr)
        return 1

    print(
        f"迁移完成: {old_info['total_points'] if old_info else 0:,} "
        f"→ {new_info['total_points']:,}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器（供 main 与测试复用）."""
    parser = argparse.ArgumentParser(
        prog="knn-eval",
        description="Qdrant KNN 评估系统 — 像素级向量检索与评估",
    )
    sub = parser.add_subparsers(dest="command")

    # --- import ---
    p_import = sub.add_parser("import", help="批量导入 SE/DW 像素数据到 Qdrant")
    p_import.add_argument("directory", help="包含 SE/ 和 DW/ 子目录的数据根目录")
    p_import.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                          help=f"每批 upsert 的点数 (默认: {BATCH_SIZE})")
    p_import.add_argument("--no-resume", action="store_true",
                          help="不检查断点续传，强制重新导入")
    p_import.add_argument("--reindex", action="store_true",
                          help="导入完成后重建全量 HNSW 向量索引（indexing_threshold=0）")
    p_import.add_argument("--collection", default=DEFAULT_COLLECTION,
                          help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_import.add_argument("--qdrant-url", default=QDRANT_URL,
                          help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- search ---
    p_search = sub.add_parser("search", help="执行向量检索")
    group = p_search.add_mutually_exclusive_group(required=True)
    group.add_argument("--query-file", type=str, help="query vector 的 .npy 文件路径")
    group.add_argument("--random", action="store_true", help="从 collection 随机选取一个 vector")
    group.add_argument("--query-spec", nargs=3, metavar=("IMAGE_ID", "ROW", "COL"),
                       help="按 image_id、row、col 指定 query 像素")
    p_search.add_argument("--k", type=int, default=10, help="返回 Top-K 结果 (默认: 10)")
    p_search.add_argument("--label", type=str, help="标签过滤，逗号分隔，如 '0,1' 或 'water,trees'")
    p_search.add_argument("--utm-range", type=str,
                          help="UTM 范围过滤，格式: min_e,max_e,min_n,max_n")
    p_search.add_argument("--exact", action="store_true", help="使用暴力精确搜索")
    p_search.add_argument("--ef-search", type=int, default=64,
                          help="ANN 搜索的 ef 参数 (默认: 64)")
    p_search.add_argument("--output", choices=["table", "json"], default="table",
                          help="输出格式 (默认: table)")
    p_search.add_argument("--collection", default=DEFAULT_COLLECTION,
                          help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_search.add_argument("--qdrant-url", default=QDRANT_URL,
                          help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- stats ---
    p_stats = sub.add_parser("stats", help="查看 Collection 统计信息")
    p_stats.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    p_stats.add_argument("--collection", default=DEFAULT_COLLECTION,
                         help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_stats.add_argument("--qdrant-url", default=QDRANT_URL,
                         help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="评估 embedding 质量指标 (F1/F2)")
    p_eval.add_argument("--samples-per-class", type=int, default=500,
                        help="每类采样查询像素数 (默认: 500)")
    p_eval.add_argument("--k-f1", type=int, default=100,
                        choices=range(1, 1001), metavar="{1..1000}",
                        help="KNN 分类器 K 值 (默认: 100，范围 1..1000，与 F2 默认 K 序列上限对齐，单次 top-1001 覆盖)")
    p_eval.add_argument("--k-values", type=str, default="10,20,50,100,300,1000",
                        help="Purity/Recall 曲线的 K 值序列，逗号分隔 (默认: 10,20,50,100,300,1000)")
    p_eval.add_argument("--ann", action="store_true",
                        help="额外用 ANN 模式跑一遍，输出 exact vs ANN 对比")
    p_eval.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto",
                        help="评估执行设备: cuda=GPU, cpu=torch CPU 分块, auto=CUDA 可用则 GPU 否则 CPU (默认: auto)")
    p_eval.add_argument("--gpu-batch-q", type=int, default=None,
                        help="查询分块大小（显式给定则跳过预算推导；默认按预算推导）")
    p_eval.add_argument("--max-gpu-mem", type=float, default=16,
                        help="显存预算上限 GB (默认: 16)")
    p_eval.add_argument("--max-eval-ram", type=float, default=6.0,
                        help="CPU 回退 RAM 预算上限 GB (默认: 6)")
    p_eval.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    p_eval.add_argument("--output", type=str, default=None,
                        help="结果 JSON 输出路径")
    p_eval.add_argument("--plot", action="store_true",
                        help="生成 matplotlib 图表")
    p_eval.add_argument("--plot-dir", type=str, default="./eval_plots",
                        help="图表输出目录 (默认: ./eval_plots)")
    p_eval.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_eval.add_argument("--qdrant-url", default=QDRANT_URL,
                        help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- similarity-heatmap ---
    p_sim = sub.add_parser("similarity-heatmap",
                           help="双集合 embedding 相似度热力图对比")
    p_sim.add_argument("--n", type=_parse_positive_n, default=200,
                       help="采样点数（1..600，默认: 200）")
    p_sim.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    p_sim.add_argument("--image-id", type=str, default=None,
                       help="指定影像（图片模式）；缺省为数据库全库模式")
    p_sim.add_argument("--output", type=str, default="similarity_heatmap.png",
                       help="输出 PNG 路径 (默认: similarity_heatmap.png)")
    p_sim.add_argument("--export-dir", type=str, default="outputs",
                       help="导出目录（默认 outputs，自动创建；显式传空串禁用导出）")
    p_sim.add_argument("--google-collection", default=DEFAULT_COLLECTION,
                       help=f"google 侧 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_sim.add_argument("--xian-collection", default="xian_aef_embedding",
                       help="xian 侧 Qdrant Collection 名称 (默认: xian_aef_embedding)")
    p_sim.add_argument("--qdrant-url", default=QDRANT_URL,
                       help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- migrate ---
    p_migrate = sub.add_parser("migrate", help="重建 Collection 为指定存储配置并重导数据")
    p_migrate.add_argument("--dir", default="data_demo", help="数据根目录")
    p_migrate.add_argument("--storage", choices=["disk", "ram"], default="disk",
                           help="新 Collection 存储预设 (默认: disk)")
    p_migrate.add_argument("--no-resume", action="store_true", help="强制重新导入")
    p_migrate.add_argument("--collection", default=DEFAULT_COLLECTION,
                           help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_migrate.add_argument("--qdrant-url", default=QDRANT_URL,
                           help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "import":
        return cmd_import(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "migrate":
        return cmd_migrate(args)
    elif args.command == "similarity-heatmap":
        return cmd_similarity_heatmap(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
