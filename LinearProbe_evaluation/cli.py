"""LinearProbe_evaluation 命令行入口：train / evaluate / stats.

用法示例:
    # 训练 MLP_label（默认分层采样，每类 train 2000 / val 500）
    uv run python -m LinearProbe_evaluation.cli train --output-dir outputs/mlp_label

    # 全量训练（每类不限制采样上限）
    uv run python -m LinearProbe_evaluation.cli train --train-per-class 0 --val-per-class 0

    # 用已有 checkpoint 在 Qdrant 数据上评估
    uv run python -m LinearProbe_evaluation.cli evaluate --checkpoint outputs/mlp_label/checkpoints/mlp_label_best.pt

    # Collection 统计（每类像素数）
    uv run python -m LinearProbe_evaluation.cli stats --json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from LinearProbe_evaluation.config import (
    QDRANT_URL, DEFAULT_COLLECTION, NUM_CLASSES, DEFAULT_OUTPUT_DIR,
)
from LinearProbe_evaluation.label_mapping import LABEL_NAMES
from LinearProbe_evaluation.qdrant_client import QdrantManager
from LinearProbe_evaluation.dataset import (
    stratified_train_val_split, sample_dataset, load_full_dataset, CancelledError,
)
from LinearProbe_evaluation.trainer import (
    TrainConfig, train_mlp, load_mlp_label, predict_labels, resolve_device,
)
from LinearProbe_evaluation.model import build_model, VARIANT_HIDDEN_DIMS
from LinearProbe_evaluation.metrics import classification_report, confusion_matrix
from LinearProbe_evaluation.visualization import plot_training_curves, plot_confusion_matrix


def _np_encoder(obj):
    """JSON 序列化 numpy 标量/数组（参考 KNN cli._np_encoder）."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _print_class_counts(ds, title: str) -> None:
    print(f"{title}: 共 {ds.size} 个像素")
    for lid, name in LABEL_NAMES.items():
        cnt = int((ds.y == lid).sum())
        if cnt:
            print(f"  label {lid} {name}: {cnt}")


def cmd_train(args) -> int:
    """训练 MLP_label：分层采样 → 训练 → checkpoint / 图表 / 报告."""
    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)
    if not manager.health_check():
        print(f"[错误] Qdrant 不可达: {args.qdrant_url}", file=sys.stderr)
        return 1
    if not manager.collection_exists():
        print(f"[错误] Collection '{manager.collection_name}' 不存在，请先执行 KNN import", file=sys.stderr)
        return 1

    print("正在从 Qdrant 分层采样训练/验证数据...")
    try:
        train_ds, val_ds = stratified_train_val_split(
            manager,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    except CancelledError:
        print("已取消", file=sys.stderr)
        return 130
    _print_class_counts(train_ds, "训练集")
    _print_class_counts(val_ds, "验证集")
    if train_ds.size == 0:
        print("[错误] 训练集为空", file=sys.stderr)
        return 1

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        model_variant=args.model,
        seed=args.seed,
        log_interval=args.log_interval,
        output_dir=Path(args.output_dir),
    )
    print(f"模型结构: {build_model(args.model).describe()}")

    from tqdm import tqdm

    bar = tqdm(total=args.epochs, desc="Training", unit="epoch")

    def progress(event: dict):
        if event["event"] == "epoch_start":
            bar.set_description(f"Epoch {event['epoch']}/{event['total_epochs']}")
        elif event["event"] == "batch":
            if args.verbose:
                bar.set_postfix(batch=f"{event['batch']}/{event['total_batches']}",
                                loss=f"{event['loss']:.4f}")
        elif event["event"] == "epoch_end":
            bar.update(1)
            # 空验证集时 val_* 为 None（不参与 best），用 0.0 占位避免格式化崩溃
            val_acc = event["val_acc"] if event["val_acc"] is not None else 0.0
            val_mf1 = event["val_macro_f1"] if event["val_macro_f1"] is not None else 0.0
            bar.set_postfix(
                train_loss=f"{event['train_loss']:.4f}",
                val_acc=f"{val_acc:.4f}",
                val_mf1=f"{val_mf1:.4f}",
            )
            if args.verbose:
                val_loss = event["val_loss"]
                val_loss_s = f"{val_loss:.4f}" if val_loss is not None else "n/a"
                print(
                    f"\n[epoch {event['epoch']}/{event['total_epochs']}] "
                    f"train_loss={event['train_loss']:.4f} train_acc={event['train_acc']:.4f} | "
                    f"val_loss={val_loss_s} val_acc={val_acc:.4f} "
                    f"val_macro_f1={val_mf1:.4f}"
                )

    try:
        result = train_mlp(train_ds, val_ds, config, progress_callback=progress)
    except CancelledError:
        print("\n训练已取消", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n[错误] 训练失败: {e}", file=sys.stderr)
        return 1
    finally:
        bar.close()

    print(f"\n训练完成，耗时 {result.elapsed_seconds:.1f}s"
          f"（设备: {result.device_name}，GPU kernel {result.gpu_kernel_seconds:.3f}s）")
    print(f"Best epoch: {result.best_epoch} | val_acc={result.best_val_accuracy:.4f} "
          f"| val_macro_f1={result.best_val_macro_f1:.4f}")

    # 图表
    plot_dir = Path(args.plot_dir)
    plot_training_curves(result.history, plot_dir / "train_curves.png")
    ckpt_path = result.best_checkpoint or result.final_checkpoint
    if val_ds.size > 0 and ckpt_path is not None:
        best_model, _ = load_mlp_label(ckpt_path, result.device)
        final_pred = predict_labels(best_model, val_ds.X, result.device)
        cm = confusion_matrix(val_ds.y, final_pred, NUM_CLASSES)
        plot_confusion_matrix(cm, [LABEL_NAMES[i] for i in range(NUM_CLASSES)],
                              plot_dir / "confusion_matrix.png")
    print(f"图表已保存到 {plot_dir}/")

    _print_report(result.val_report, "验证集")
    if args.output:
        out = {
            "best_epoch": result.best_epoch,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_macro_f1": result.best_val_macro_f1,
            "val_report": result.val_report,
            "train_report": result.train_report,
            "elapsed_seconds": result.elapsed_seconds,
            "device": result.device,
            "best_checkpoint": str(result.best_checkpoint) if result.best_checkpoint else None,
            "final_checkpoint": str(result.final_checkpoint) if result.final_checkpoint else None,
        }
        Path(args.output).write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=_np_encoder), encoding="utf-8",
        )
        print(f"结果 JSON 已保存到 {args.output}")
    return 0


def _print_report(report: dict, title: str) -> None:
    if not report:
        print(f"{title}: 无评估数据")
        return
    print(f"\n{title}指标:")
    print(f"  accuracy      = {report['accuracy']:.4f}")
    print(f"  macro-F1      = {report['macro_f1']:.4f}")
    print(f"  weighted-F1   = {report['weighted_f1']:.4f}")
    print("  per-class:")
    for lid, m in report["per_class"].items():
        name = LABEL_NAMES.get(int(lid), "?")
        print(f"    {lid} {name}: precision={m['precision']:.4f} recall={m['recall']:.4f} "
              f"f1={m['f1']:.4f} support={m['support']}")


def cmd_evaluate(args) -> int:
    """用已有 checkpoint 在 Qdrant 数据上评估（默认分层采样，只下载样本量）."""
    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)
    if not manager.health_check():
        print(f"[错误] Qdrant 不可达: {args.qdrant_url}", file=sys.stderr)
        return 1
    device = resolve_device("cuda")  # 只支持 cuda（禁用 CPU fallback）
    model, meta = load_mlp_label(args.checkpoint, device)
    print(f"已加载 checkpoint: {args.checkpoint}")
    print(f"  架构: {meta['architecture']} | best_val_acc={meta.get('val_accuracy', '?')}"
          f" | GPU: {torch.cuda.get_device_name(0)}")

    # 默认按采样地图分层采样（只下载样本量）；--samples-per-class 0 才显式全量。
    try:
        if args.samples_per_class and args.samples_per_class > 0:
            print(f"正在分层采样评估数据（每类 {args.samples_per_class}）...")
            ds = sample_dataset(manager, samples_per_class=args.samples_per_class,
                                seed=args.seed)
            _print_class_counts(ds, "评估集")
        else:
            print("正在从 Qdrant 读取全量数据（--samples-per-class 0，可能耗时较长）...")
            ds = load_full_dataset(manager, max_points=args.max_points)
    except (ConnectionError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if ds.size == 0:
        print("[错误] 评估数据为空", file=sys.stderr)
        return 1
    print(f"推理 {ds.size} 个像素（GPU: {torch.cuda.get_device_name(0)}）...")
    pred = predict_labels(model, ds.X, device)
    report = classification_report(ds.y, pred, NUM_CLASSES)
    _print_report(report, "评估集")

    cm = confusion_matrix(ds.y, pred, NUM_CLASSES)
    if args.plot:
        plot_dir = Path(args.plot_dir)
        plot_confusion_matrix(
            cm, [LABEL_NAMES[i] for i in range(NUM_CLASSES)],
            plot_dir / "evaluate_confusion_matrix.png",
            title="MLP_label 混淆矩阵（评估集）",
        )
        print(f"混淆矩阵已保存到 {plot_dir}/evaluate_confusion_matrix.png")
    if args.output:
        out = {
            "checkpoint": str(args.checkpoint),
            "n_samples": int(ds.size),
            "report": report,
            "confusion_matrix": cm.tolist(),
            "device": device,
        }
        Path(args.output).write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=_np_encoder), encoding="utf-8",
        )
        print(f"结果 JSON 已保存到 {args.output}")
    return 0


def cmd_stats(args) -> int:
    """Collection 统计：总点数 + 每类像素数（label count API）."""
    from qdrant_client import models

    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)
    if not manager.health_check():
        print(f"[错误] Qdrant 不可达: {args.qdrant_url}", file=sys.stderr)
        return 1
    if not manager.collection_exists():
        print(f"[错误] Collection '{manager.collection_name}' 不存在", file=sys.stderr)
        return 1

    info = manager.collection_info()
    per_label: dict[str, int] = {}
    for lid in range(NUM_CLASSES):
        cnt = manager.client.count(
            collection_name=manager.collection_name, exact=True,
            count_filter=models.Filter(must=[
                models.FieldCondition(key="label", match=models.MatchValue(value=lid)),
            ]),
        ).count
        per_label[str(lid)] = int(cnt)

    result = {
        "collection": manager.collection_name,
        "total_points": int(info["total_points"]),
        "status": info["status"],
        "per_label": per_label,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_np_encoder))
    else:
        print(f"Collection: {manager.collection_name} | total_points={result['total_points']} "
              f"| status={result['status']}")
        for lid in range(NUM_CLASSES):
            name = LABEL_NAMES[lid]
            print(f"  label {lid} {name}: {per_label[str(lid)]}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linear-probe-eval",
        description="Qdrant Linear Probe 评估系统 — 像素 embedding 线性分类 MLP_label",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    p_train = sub.add_parser("train", help="训练 MLP_label（64 维 embedding → 9 类硬标签）")
    p_train.add_argument("--epochs", type=int, default=TrainConfig.epochs,
                         help=f"训练轮数 (默认: {TrainConfig.epochs})")
    p_train.add_argument("--batch-size", type=int, default=TrainConfig.batch_size,
                         help=f"批大小 (默认: {TrainConfig.batch_size})")
    p_train.add_argument("--lr", type=float, default=TrainConfig.lr,
                         help=f"学习率 (默认: {TrainConfig.lr})")
    p_train.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay,
                         help=f"权重衰减 (默认: {TrainConfig.weight_decay})")
    p_train.add_argument("--optimizer", choices=["adam", "sgd"], default=TrainConfig.optimizer,
                         help=f"优化器 (默认: {TrainConfig.optimizer})")
    p_train.add_argument("--model", choices=list(VARIANT_HIDDEN_DIMS), default="mlp",
                         help="模型结构: mlp=LeNet-5 规模多层 (15万参数), "
                              "linear=标准 Linear Probe 单层 (585参数) (默认: mlp)")
    p_train.add_argument("--train-per-class", type=int, default=2000,
                         help="每类最多进入训练集的样本数，0=不限制 (默认: 2000)")
    p_train.add_argument("--val-per-class", type=int, default=500,
                         help="每类最多进入验证集的样本数，0=不限制 (默认: 500)")
    p_train.add_argument("--val-ratio", type=float, default=0.2,
                         help="每类先按该比例切出验证集 (默认: 0.2)")
    p_train.add_argument("--seed", type=int, default=TrainConfig.seed,
                         help=f"随机种子 (默认: {TrainConfig.seed})")
    p_train.add_argument("--log-interval", type=int, default=TrainConfig.log_interval,
                         help=f"每 N 个 batch 上报一次训练进度 (默认: {TrainConfig.log_interval})")
    p_train.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                         help=f"输出目录（checkpoint / 元数据）(默认: {DEFAULT_OUTPUT_DIR})")
    p_train.add_argument("--plot-dir", type=str, default="outputs/mlp_label/plots",
                         help="图表输出目录 (默认: outputs/mlp_label/plots)")
    p_train.add_argument("--output", type=str, default=None,
                         help="训练结果 JSON 输出路径")
    p_train.add_argument("--verbose", action="store_true", help="打印每个 epoch 的详细指标")
    p_train.add_argument("--collection", default=DEFAULT_COLLECTION,
                         help=f"Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_train.add_argument("--qdrant-url", default=QDRANT_URL,
                         help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="用 checkpoint 在 Qdrant 数据上评估")
    p_eval.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径 (.pt)")
    p_eval.add_argument("--samples-per-class", type=int, default=1000,
                        help="每类最多评估样本数（默认 1000，按采样地图分层采样、"
                             "只下载样本量；0 = 显式全量读取，耗时长）")
    p_eval.add_argument("--max-points", type=int, default=None,
                        help="全量读取时最多读取点数（默认不限）")
    p_eval.add_argument("--seed", type=int, default=42,
                        help=f"随机种子 (默认: 42)")
    p_eval.add_argument("--plot", action="store_true", help="生成混淆矩阵图")
    p_eval.add_argument("--plot-dir", type=str, default="outputs/mlp_label/plots",
                        help="图表输出目录 (默认: outputs/mlp_label/plots)")
    p_eval.add_argument("--output", type=str, default=None,
                        help="评估结果 JSON 输出路径")
    p_eval.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_eval.add_argument("--qdrant-url", default=QDRANT_URL,
                        help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- stats ---
    p_stats = sub.add_parser("stats", help="Collection 统计（总点数 + 每类像素数）")
    p_stats.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    p_stats.add_argument("--collection", default=DEFAULT_COLLECTION,
                         help=f"Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_stats.add_argument("--qdrant-url", default=QDRANT_URL,
                         help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "stats":
        return cmd_stats(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
