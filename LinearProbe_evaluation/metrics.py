"""分类指标：accuracy / macro-F1 / per-class 指标 / 混淆矩阵.

面向类别严重不平衡的 DW 数据（trees 621 万 vs snow_and_ice 240），
除整体 accuracy 外重点提供 macro-F1（各类 F1 的算术平均，稀有类
与大类别等权）与 per-class recall。
"""
import numpy as np


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """混淆矩阵 C，C[i][j] = 真实 i 预测为 j 的样本数."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """整体准确率（分子 = 对角线）."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """每类 recall：C[i][i] / sum_j C[i][j]（该类无真实样本时记 0）."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    row_sum = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(row_sum > 0, cm.diagonal() / np.maximum(row_sum, 1), 0.0)
    return acc


def _per_class_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """每类 precision / recall / f1 / support（support = 该类真实样本数）.

    该类在预测或真实中均未出现时分母为 0，指标记 0（避免除零）.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = cm.diagonal().astype(np.float64)
    fp = cm.sum(axis=0) - tp  # 预测为该类但真实不是
    fn = cm.sum(axis=1) - tp  # 真实为该类但预测不是
    support = cm.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / np.maximum(precision + recall, 1e-12),
            0.0,
        )
    return precision, recall, f1, support


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """macro-F1：各类 F1 的算术平均（类别不平衡场景的主指标）."""
    _, _, f1, _ = _per_class_precision_recall_f1(y_true, y_pred, num_classes)
    if num_classes == 0:
        return 0.0
    return float(f1.mean())


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """weighted-F1：按各类真实样本占比加权的 F1 平均."""
    _, _, f1, support = _per_class_precision_recall_f1(y_true, y_pred, num_classes)
    total = float(support.sum())
    if total == 0:
        return 0.0
    return float((f1 * support).sum() / total)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict:
    """汇总指标：accuracy / macro_f1 / weighted_f1 / per_class 明细.

    Returns:
        dict，形如::

            {
              "accuracy": 0.93,
              "macro_f1": 0.71,
              "weighted_f1": 0.92,
              "per_class": {label_id: {"precision":..., "recall":..., "f1":..., "support":...}},
            }
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    precision, recall, f1, support = _per_class_precision_recall_f1(
        y_true, y_pred, num_classes,
    )
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "weighted_f1": weighted_f1(y_true, y_pred, num_classes),
        "per_class": {
            int(i): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
    }


def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """标签索引 → 独热码矩阵 (N, num_classes)（硬分类独热码语义）.

    对应模型 9 维输出的硬分类 one-hot 表示：预测时对 logits 取 argmax
    后同样编码为 one-hot，两者可逐位比较。
    """
    y = np.asarray(y).ravel()
    n = len(y)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    valid = (y >= 0) & (y < num_classes)
    one_hot[valid, y[valid].astype(np.int64)] = 1.0
    return one_hot
