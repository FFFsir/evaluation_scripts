"""Tests for classification metrics."""
import numpy as np

from LinearProbe_evaluation.metrics import (
    accuracy, confusion_matrix, per_class_accuracy, macro_f1,
    weighted_f1, classification_report, to_one_hot,
)


def test_confusion_matrix_basic():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1   # 真实 0 预测 0
    assert cm[0, 2] == 1   # 真实 0 预测 2
    assert cm[1, 1] == 2   # 真实 1 预测 1
    assert cm[1, 2] == 0
    assert cm[2, 1] == 1   # 真实 2 预测 1


def test_accuracy():
    assert accuracy(np.array([0, 1, 2]), np.array([0, 1, 1])) == 2 / 3
    assert accuracy(np.array([], dtype=int), np.array([], dtype=int)) == 0.0


def test_per_class_accuracy():
    y_true = np.array([0, 0, 1, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    acc = per_class_accuracy(y_true, y_pred, 3)
    assert acc[0] == 0.5      # 0 类：1/2
    assert acc[1] == 2 / 3    # 1 类：2/3
    assert acc[2] == 1.0      # 2 类：1/1


def test_macro_f1_hand_calculated():
    """二分类对照：TP=4 FP=1 FN=2 → P=0.8 R=2/3 F1=0.7272..."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])
    mf1 = macro_f1(y_true, y_pred, 2)
    # 类 0: TP=4 FP=1 FN=2 → P=0.8, R=2/3, F1=8/11≈0.7273
    # 类 1: TP=2 FP=2 FN=1 → P=0.5, R=2/3, F1=4/7≈0.5714
    # macro = (8/11 + 4/7)/2 ≈ 0.64935
    assert abs(mf1 - (8 / 11 + 4 / 7) / 2) < 1e-9


def test_macro_f1_empty_class():
    """某类在预测中完全未出现：该类的指标记 0，不除零崩溃."""
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])
    mf1 = macro_f1(y_true, y_pred, 3)  # 类 2 无样本也无预测
    assert 0.0 <= mf1 <= 1.0


def test_weighted_f1():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])
    wf1 = weighted_f1(y_true, y_pred, 2)
    # 类 0: TP=2 FP=1 FN=0 → P=2/3, R=1, F1=0.8，support=2
    # 类 1: F1=0，support=1
    # weighted = (0.8*2 + 0*1)/3 = 1.6/3 ≈ 0.5333
    assert abs(wf1 - 1.6 / 3) < 1e-9


def test_classification_report_structure():
    y_true = np.array([0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 2, 2, 2, 2])
    rep = classification_report(y_true, y_pred, 3)
    assert set(rep) == {"accuracy", "macro_f1", "weighted_f1", "per_class"}
    assert abs(rep["accuracy"] - 5 / 6) < 1e-9
    assert set(rep["per_class"]) == {0, 1, 2}
    m = rep["per_class"][2]
    assert m["support"] == 3
    assert abs(m["recall"] - 1.0) < 1e-9


def test_to_one_hot():
    y = np.array([0, 2, 8, -1, 99])
    oh = to_one_hot(y, 9)
    assert oh.shape == (5, 9)
    assert oh[0, 0] == 1 and oh[0, 1] == 0
    assert oh[1, 2] == 1
    assert oh[2, 8] == 1
    # 越界/非法标签行保持全 0
    assert oh[3].sum() == 0 and oh[4].sum() == 0
