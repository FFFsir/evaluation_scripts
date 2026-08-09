"""训练过程可视化 — matplotlib 图表生成（与 KNN_evaluation/visualization.py 同风格）.

图表以 PNG 保存到磁盘（CLI 用），或序列化为 base64 data URI（WebUI 用）.
"""
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体配置（与 KNN_evaluation/visualization.py 一致）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_to_base64(fig) -> str:
    """matplotlib Figure → base64 PNG data URI（WebUI ui.image 直接可用）."""
    return "data:image/png;base64," + base64.b64encode(_fig_to_png_bytes(fig)).decode()


def _png_to_base64(png: bytes) -> str:
    """PNG bytes → base64 data URI."""
    return "data:image/png;base64," + base64.b64encode(png).decode()


def training_curves_base64(history: list[dict]) -> str:
    """训练曲线 → base64 data URI（WebUI 实时刷新用）."""
    return _png_to_base64(plot_training_curves(history))


def confusion_matrix_base64(
    cm: np.ndarray,
    label_names: list[str],
    title: str = "MLP_label 混淆矩阵（验证集）",
) -> str:
    """混淆矩阵 → base64 data URI（WebUI 展示用）."""
    return _png_to_base64(plot_confusion_matrix(cm, label_names, title=title))


def plot_training_curves(
    history: list[dict],
    save_path: str | Path | None = None,
) -> bytes:
    """绘制训练/验证 loss 与 accuracy 双面板曲线.

    Args:
        history: trainer.TrainResult.history（每 epoch 一条记录）.
        save_path: 输出 PNG 路径（None 则只返回 bytes）.

    Returns:
        PNG bytes.
    """
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", markersize=3,
             linewidth=1.5, label="train loss")
    if all(h.get("val_loss") is not None for h in history):
        ax1.plot(epochs, [h["val_loss"] for h in history], marker="s", markersize=3,
                 linewidth=1.5, label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [h["train_acc"] for h in history], marker="o", markersize=3,
             linewidth=1.5, label="train acc")
    if all(h.get("val_acc") is not None for h in history):
        ax2.plot(epochs, [h["val_acc"] for h in history], marker="s", markersize=3,
                 linewidth=1.5, label="val acc")
        if all(h.get("val_macro_f1") is not None for h in history):
            ax2.plot(epochs, [h["val_macro_f1"] for h in history], marker="^",
                     markersize=3, linewidth=1.5, label="val macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy / F1")
    ax2.set_title("Accuracy & Macro-F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("MLP_label 训练曲线（64 维 embedding → 9 类硬标签）", fontsize=13)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return Path(save_path).read_bytes()
    return _fig_to_png_bytes(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    save_path: str | Path | None = None,
    title: str = "MLP_label 混淆矩阵（验证集）",
) -> bytes:
    """绘制混淆矩阵热力图（复刻 KNN_evaluation.visualization.plot_confusion_matrix 风格）.

    Args:
        cm: (N, N) 混淆矩阵，C[i][j] = 真实 i 预测为 j.
        label_names: 标签名列表（顺序与类别索引一致）.
        save_path: 输出 PNG 路径（None 则只返回 bytes）.

    Returns:
        PNG bytes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")

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

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return Path(save_path).read_bytes()
    return _fig_to_png_bytes(fig)
