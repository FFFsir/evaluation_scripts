"""训练器：MLP_label 线性分类器的训练循环 / 进度回调 / checkpoint 管理.

训练目标：给定 64 维像素 embedding，用 CrossEntropyLoss 学习 64→9 的
线性映射，输出 9 维 logits（硬分类独热码语义）。进度经
``progress_callback(event: dict)`` 逐批/逐 epoch 上报，供 CLI 与 WebUI
共用；``cancel_event`` 支持协作式取消。
"""
import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from LinearProbe_evaluation.config import (
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
    DEFAULT_OPTIMIZER, DEFAULT_SEED, DEFAULT_LOG_INTERVAL, DEFAULT_OUTPUT_DIR,
    VECTOR_SIZE, NUM_CLASSES,
)
from LinearProbe_evaluation.label_mapping import LABEL_NAMES
from LinearProbe_evaluation.metrics import (
    accuracy as calc_accuracy,
    macro_f1 as calc_macro_f1,
    classification_report,
    confusion_matrix,
)
from LinearProbe_evaluation.model import (
    MLPLabel, build_model, DEFAULT_HIDDEN_DIMS, DEFAULT_DROPOUT, VARIANT_MLP,
)
from LinearProbe_evaluation.dataset import PixelDataset

CHECKPOINT_BEST_NAME = "mlp_label_best.pt"
CHECKPOINT_FINAL_NAME = "mlp_label_final.pt"
META_JSON_NAME = "mlp_label_meta.json"


class CancelledError(Exception):
    """训练被用户主动取消."""


@dataclass
class TrainConfig:
    """训练超参与输出配置."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer: str = DEFAULT_OPTIMIZER  # adam | sgd
    device: str = "cuda"               # 只支持 cuda（禁用 CPU fallback）
    model_variant: str = VARIANT_MLP   # 'mlp' = LeNet-5 多层 | 'linear' = 单层 Linear Probe
    seed: int = DEFAULT_SEED
    log_interval: int = DEFAULT_LOG_INTERVAL  # 每 N 个 batch 上报一次进度
    output_dir: Path = DEFAULT_OUTPUT_DIR
    checkpoint_best: bool = True
    save_final: bool = True


@dataclass
class TrainResult:
    """训练产物汇总."""

    history: list[dict] = field(default_factory=list)   # 每 epoch 指标
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    best_val_macro_f1: float = 0.0
    best_checkpoint: Path | None = None
    final_checkpoint: Path | None = None
    val_report: dict = field(default_factory=dict)      # 最终 val 分类报告
    train_report: dict = field(default_factory=dict)    # 最终 train 分类报告
    elapsed_seconds: float = 0.0
    device: str = "cuda"
    device_name: str = ""            # 实际 GPU 设备名（如 NVIDIA GeForce RTX 4090）
    gpu_kernel_seconds: float = 0.0  # CUDA kernel 累计耗时（CUDA events 计时）


def resolve_device(device: str = "cuda") -> str:
    """解析执行设备：**只支持 CUDA，禁用 CPU fallback**.

    ``cuda`` / ``auto`` / None 一律解析为 ``cuda``；CUDA 不可用直接抛错，
    绝不静默回退到 CPU（避免用户以为走了 GPU 实际却在 CPU 上训练）。

    Raises:
        RuntimeError: CUDA 不可用（未检测到 NVIDIA GPU / torch 非 CUDA 版）.
        ValueError: 传入未知设备名（cpu 已不再支持）.
    """
    if device not in (None, "cuda", "auto"):
        raise ValueError(
            f"设备 {device!r} 不受支持：本模块已禁用 CPU fallback，只支持 cuda"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用：未检测到可用的 NVIDIA GPU（torch "
            f"{torch.__version__}）。本模块只支持 cuda 训练，请检查驱动/环境后重试。"
        )
    return "cuda"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_labels(
    model: nn.Module,
    X: np.ndarray,
    device: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    """批量推理：X (N, 64) → 类别索引 (N,)（模型输出的 9 维 logits 取 argmax）."""
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size].astype(np.float32))
            logits = model(batch.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)


def _evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str,
              chunk_size: int = 8192) -> dict:
    """无梯度评估：GPU 分块 forward，loss 在 GPU 上计算，pred 一次性拷回 CPU.

    分块（而非单次全量 forward）以限制激活显存：默认采样下验证集仅数千样本，
    但 ``--val-per-class 0`` / 全量验证可能达数百万样本，单次 forward 会 OOM。
    logits 在 GPU 上累积（(N, 9) 极小），cat / cross_entropy / argmax 均在 GPU，
    只有最终 pred 一次性 .cpu()。

    Returns:
        loss / accuracy / macro_f1 / confusion matrix（后两者用 numpy 计算）.
    """
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)
    assert X_t.is_cuda, "评估数据必须驻留 GPU（禁用 CPU fallback）"
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, X_t.size(0), chunk_size):
            chunk = X_t[start:start + chunk_size]  # view，不拷贝
            logits_parts.append(model(chunk))
        logits = torch.cat(logits_parts, dim=0)
        loss = nn.functional.cross_entropy(logits, y_t, reduction="mean").item()
        pred = logits.argmax(dim=1).cpu().numpy()
    return {
        "loss": float(loss),
        "accuracy": calc_accuracy(y, pred),
        "macro_f1": calc_macro_f1(y, pred, NUM_CLASSES),
        "confusion": confusion_matrix(y, pred, NUM_CLASSES),
    }


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    epoch: int,
    config: TrainConfig,
    val_accuracy: float,
    val_macro_f1: float,
) -> None:
    """保存模型 checkpoint（state_dict + 元数据），原子写（tmp + replace）."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": {
            "class": "MLPLabel",
            "in_features": model.in_features,
            "num_classes": model.num_classes,
            "hidden_dims": list(model.hidden_dims),
            "dropout": model.dropout,
        },
        "label_names": LABEL_NAMES,
        "epoch": epoch,
        "val_accuracy": float(val_accuracy),
        "val_macro_f1": float(val_macro_f1),
        "optimizer": config.optimizer,
        "lr": config.lr,
        "seed": config.seed,
    }, tmp)
    tmp.replace(path)


def load_mlp_label(checkpoint: str | Path, device: str = "cuda") -> tuple[MLPLabel, dict]:
    """加载 checkpoint → (模型, 元数据 dict).

    checkpoint 为本模块 `_save_checkpoint` 生成（仅含张量与基本类型），
    使用 ``weights_only=True`` 安全加载（避免 pickle 任意代码执行）。
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    arch = ckpt["architecture"]
    hidden = arch.get("hidden_dims")
    model = MLPLabel(
        in_features=int(arch["in_features"]),
        num_classes=int(arch["num_classes"]),
        # 空列表 [] 表示单层 linear（不能用 `or DEFAULT`，会把 [] 误判为默认多层）
        hidden_dims=tuple(hidden) if hidden is not None else DEFAULT_HIDDEN_DIMS,
        dropout=float(arch.get("dropout", DEFAULT_DROPOUT)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, ckpt


def train_mlp(
    train_ds: PixelDataset,
    val_ds: PixelDataset,
    config: TrainConfig | None = None,
    *,
    progress_callback=None,
    cancel_event=None,
) -> TrainResult:
    """执行 MLP_label 训练.

    Args:
        train_ds / val_ds: 分层划分后的训练/验证集（PixelDataset）.
        config: 训练超参（None 用默认）.
        progress_callback: ``cb(event: dict)``，事件见模块 docstring（可空）.
        cancel_event: threading.Event，每个 batch 检查，置位时抛 ``CancelledError``.

    Returns:
        TrainResult：history / best / checkpoint 路径 / 分类报告 / 耗时.

    Raises:
        ValueError: 训练集为空或参数非法.
        CancelledError: 用户取消.
    """
    if config is None:
        config = TrainConfig()
    if train_ds.size == 0:
        raise ValueError("训练集为空，无法训练（请检查采样参数与 Qdrant 数据）")
    if config.epochs < 1:
        raise ValueError(f"epochs 必须 >= 1，实际 {config.epochs}")
    if config.optimizer not in ("adam", "sgd"):
        raise ValueError(f"optimizer 必须是 adam|sgd，实际 {config.optimizer!r}")

    device = resolve_device(config.device)
    _set_seed(config.seed)
    start_time = time.perf_counter()

    # CUDA events 统计 GPU kernel 累计耗时（证明训练真实发生在 GPU 上）
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()

    model = build_model(config.model_variant, VECTOR_SIZE, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9,
        )

    # ---- 数据一次性驻留 GPU（训练/验证集都搬上去，epoch 内零 CPU 搬运）----
    # 数据量级：训练集 1.6 万 × 64 × 4B ≈ 4MB，一次 PCIe 拷贝后全部计算留在 GPU；
    # 避免 DataLoader 逐样本 collate（select/stack）与每 batch .to(device) 的 CPU 开销。
    train_X = torch.from_numpy(train_ds.X.astype(np.float32)).to(device)
    train_y = torch.from_numpy(train_ds.y.astype(np.int64)).to(device)
    n_train = train_X.size(0)
    assert train_X.is_cuda and train_y.is_cuda, "训练数据必须驻留 GPU（禁用 CPU fallback）"
    # GPU 上的随机数生成器：shuffle 置换也在 GPU 生成，保证可复现且零 CPU 参与
    shuffle_gen = torch.Generator(device=device)
    shuffle_gen.manual_seed(config.seed)

    result = TrainResult(device=device)
    result.device_name = torch.cuda.get_device_name(0)
    output_dir = Path(config.output_dir)
    best_val_acc = -1.0

    for epoch in range(1, config.epochs + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("训练已取消")
        if progress_callback is not None:
            progress_callback({"event": "epoch_start", "epoch": epoch,
                               "total_epochs": config.epochs})

        # ---- 训练阶段（全 GPU：shuffle 置换 + 索引 + 前向/反向都在 cuda 上）----
        model.train()
        perm = torch.randperm(n_train, device=device, generator=shuffle_gen)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        total_batches = (n_train + config.batch_size - 1) // config.batch_size
        for batch_idx in range(total_batches):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("训练已取消")
            idx = perm[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size]
            xb = train_X[idx]   # GPU gather（index_select kernel），无 CPU 搬运
            yb = train_y[idx]
            assert xb.is_cuda, "batch 必须驻留 GPU（禁用 CPU fallback）"
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += xb.size(0)
            if progress_callback is not None and (batch_idx + 1) % config.log_interval == 0:
                progress_callback({
                    "event": "batch",
                    "epoch": epoch, "total_epochs": config.epochs,
                    "batch": batch_idx + 1, "total_batches": total_batches,
                    "loss": loss.item(),
                })

        train_epoch = {
            "loss": total_loss / max(total_seen, 1),
            "accuracy": total_correct / max(total_seen, 1),
        }

        # ---- 验证阶段（全量 val；空验证集时记录 None，不参与 best 选择）----
        if val_ds.size > 0:
            val_metrics = _evaluate(model, val_ds.X, val_ds.y, device)
        else:
            val_metrics = {"loss": None, "accuracy": None, "macro_f1": None}

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_epoch["loss"],
            "train_acc": train_epoch["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        result.history.append(epoch_record)

        if val_ds.size > 0 and val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            result.best_epoch = epoch
            result.best_val_accuracy = val_metrics["accuracy"]
            result.best_val_macro_f1 = val_metrics["macro_f1"]
            if config.checkpoint_best:
                result.best_checkpoint = output_dir / "checkpoints" / CHECKPOINT_BEST_NAME
                _save_checkpoint(
                    model, result.best_checkpoint, epoch=epoch, config=config,
                    val_accuracy=val_metrics["accuracy"],
                    val_macro_f1=val_metrics["macro_f1"],
                )

        if progress_callback is not None:
            progress_callback({"event": "epoch_end", **epoch_record,
                               "total_epochs": config.epochs})

    # ---- 收尾：final checkpoint + 报告 + 元数据 JSON ----
    # final checkpoint 记录**最后一个 epoch** 自身的 val 指标
    # （与最终权重对应，而非 best epoch 的指标）。
    last = result.history[-1] if result.history else {}
    last_val_acc = last.get("val_acc") or 0.0
    last_val_mf1 = last.get("val_macro_f1") or 0.0
    if config.save_final:
        result.final_checkpoint = output_dir / "checkpoints" / CHECKPOINT_FINAL_NAME
        _save_checkpoint(
            model, result.final_checkpoint, epoch=config.epochs, config=config,
            val_accuracy=last_val_acc, val_macro_f1=last_val_mf1,
        )
    # 若未启用 best checkpoint，则回填 best 为最终权重对应指标
    if result.best_checkpoint is None and result.history:
        result.best_epoch = last["epoch"]
        result.best_val_accuracy = last_val_acc
        result.best_val_macro_f1 = last_val_mf1

    # 分类报告：优先用 best checkpoint 权重（报告头条指标与 best 一致）；
    # 未启用 best checkpoint 时用最终权重。
    report_model = model
    if val_ds.size > 0 and result.best_checkpoint is not None:
        report_model, _ = load_mlp_label(result.best_checkpoint, device)
    if val_ds.size > 0:
        final_pred = predict_labels(report_model, val_ds.X, device)
        result.val_report = classification_report(val_ds.y, final_pred, NUM_CLASSES)
    if train_ds.size > 0:
        train_pred = predict_labels(report_model, train_ds.X, device)
        result.train_report = classification_report(train_ds.y, train_pred, NUM_CLASSES)

    result.elapsed_seconds = time.perf_counter() - start_time
    # GPU kernel 累计耗时（CUDA events，已同步）
    gpu_end.record()
    torch.cuda.synchronize()
    result.gpu_kernel_seconds = gpu_start.elapsed_time(gpu_end) / 1000.0
    _save_meta(output_dir, config, result, model)
    return result


def _save_meta(output_dir: Path, config: TrainConfig, result: TrainResult, model: nn.Module) -> None:
    """训练元数据 JSON（参数 + 每 epoch 历史 + 最终报告），原子写."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "architecture": {
            "class": "MLPLabel",
            "variant": model.variant,
            "in_features": model.in_features,
            "num_classes": model.num_classes,
            "hidden_dims": list(model.hidden_dims),
            "dropout": model.dropout,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        },
        "label_names": LABEL_NAMES,
        "history": result.history,
        "best_epoch": result.best_epoch,
        "best_val_accuracy": result.best_val_accuracy,
        "best_val_macro_f1": result.best_val_macro_f1,
        "val_report": result.val_report,
        "train_report": result.train_report,
        "elapsed_seconds": result.elapsed_seconds,
        "device": result.device,
        "device_name": result.device_name,
        "gpu_kernel_seconds": result.gpu_kernel_seconds,
    }
    tmp = output_dir / (META_JSON_NAME + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output_dir / META_JSON_NAME)
