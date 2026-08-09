"""评估结果可视化模块 — matplotlib 图表生成."""
import base64
import io
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
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> bytes | None:
    """绘制混淆矩阵热力图。

    Args:
        cm: shape (N, N) 混淆矩阵，C[i][j] = 真实 i 预测为 j。
        label_names: 标签名列表。
        save_path: 输出 PNG 路径；None 时返回 PNG bytes（WebUI 图片展示/导出用，
            与 LinearProbe_evaluation.visualization.plot_confusion_matrix 一致）。
        title: 图表标题。

    Returns:
        save_path 非 None 时返回 None（图已落盘）；否则返回 PNG bytes。
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

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def confusion_matrix_base64(
    cm: np.ndarray,
    label_names: list[str],
    title: str = "Confusion Matrix",
) -> str:
    """混淆矩阵 → base64 PNG data URI（WebUI 图片展示用，对齐 LinearProbe 模式）.

    Args:
        cm: shape (N, N) 混淆矩阵。
        label_names: 标签名列表。
        title: 图表标题。

    Returns:
        "data:image/png;base64,..." data URI，可直接用于 ui.image 的 source。
    """
    png = plot_confusion_matrix(cm, label_names, title=title)
    return "data:image/png;base64," + base64.b64encode(png).decode()


def plot_purity_recall_curve(
    purity_data: dict,
    save_path: str | Path,
    f1_result: dict | None = None,
) -> None:
    """绘制 Purity/Per-class Recall vs K 双面板图。

    右面板与 Web 页面「Per-class Recall (不同 K 下各类别召回率)」一致：
    优先用 KNN 分类器 per-class Recall（f1.per_class_recall_by_k，区分度更高），
    ANN 逐条模式无该键时回退 f2.per_class_recall；均只画各类别折线，无 Global 线。

    Args:
        purity_data: compute_purity_recall_curve 的返回值（f2）。
        save_path: 输出 PNG 路径。
        f1_result: evaluate_knn 的 f1 结果；含 per_class_recall_by_k 时右面板
            使用该数据（K 序列取 sorted(prbk)，与 Web 页面一致）。
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

    # 右面板: Per-class Recall（镜像 Web 页面：优先 f1.per_class_recall_by_k，
    # ANN 回退 f2.per_class_recall；无 Global 线）
    prbk = (f1_result or {}).get("per_class_recall_by_k") or {}
    if prbk:
        recall_ks = sorted(prbk)
        recall_items = {
            ln: [prbk[k][ln] for k in recall_ks]
            for ln in prbk[recall_ks[0]]
        }
    else:
        recall_ks = k_values
        recall_items = purity_data["per_class_recall"]
    for ln, recall_list in recall_items.items():
        ax2.plot(recall_ks, recall_list, marker="s", markersize=3,
                 linewidth=1, alpha=0.6, label=ln)
    ax2.set_xlabel("K")
    ax2.set_ylabel("Recall")
    ax2.set_title("Per-class Recall vs K")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    fig.suptitle("Embedding Quality: Purity & Per-class Recall Curves", fontsize=14)
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


def plot_similarity_heatmap_pair(
    mat_g: np.ndarray,
    mat_x: np.ndarray,
    save_path: str | Path | io.BytesIO,
    collection_names: tuple[str, str] = ("google_aef_embedding", "xian_aef_embedding"),
) -> None:
    """并排（1×2）渲染两个余弦相似度热力图，统一 vmin/vmax 色阶.

    Args:
        mat_g: google 集合 N'×N' 相似度矩阵。
        mat_x: xian 集合 N'×N' 相似度矩阵。
        save_path: PNG 输出路径（str/Path，自动建父目录）或类文件对象（BytesIO）。
        collection_names: 左/右子图标题集合名（默认 google × xian 预置对）。
    """
    vmin = min(float(mat_g.min()), float(mat_x.min()))
    vmax = max(float(mat_g.max()), float(mat_x.max()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    for ax, mat, name in (
        (ax1, mat_g, collection_names[0]),
        (ax2, mat_x, collection_names[1]),
    ):
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name} 相似度热力图")
        ax.set_xlabel("样本索引")
        ax.set_ylabel("样本索引")
        fig.colorbar(im, ax=ax, shrink=0.8)
    n = mat_g.shape[0]
    fig.suptitle(f"N={n}×{n} 余弦相似度矩阵对比（统一色阶）")
    if isinstance(save_path, (str, Path)):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
