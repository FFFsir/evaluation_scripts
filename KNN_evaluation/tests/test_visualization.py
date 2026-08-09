"""visualization.plot_purity_recall_curve 冒烟测试（导出 Per-class Recall 右面板）."""
import pytest

from KNN_evaluation.visualization import plot_purity_recall_curve


def _f2_fixture():
    """f2 结构（与 test_webui._eval_fixture 同构，含 9 类折线数据）. """
    labels = [f"class_{i}" for i in range(9)]
    return {
        "k_values": [10, 20, 50],
        "global_purity": [0.9, 0.8, 0.7],
        "global_recall": [0.4, 0.5, 0.6],
        "per_class_purity": {ln: [0.9, 0.8, 0.7] for ln in labels},
        "per_class_recall": {ln: [0.3, 0.4, 0.5] for ln in labels},
        "num_queries": 100,
        "elapsed_sec": 1.0,
    }


def _f1_fixture():
    """f1 结构，含 per_class_recall_by_k（KNN 分类器各类别 Recall）. """
    labels = [f"class_{i}" for i in range(9)]
    return {
        "per_class_recall_by_k": {
            k: {ln: 0.5 + 0.05 * k for ln in labels}
            for k in (10, 20, 50)
        },
    }


class TestPlotPurityRecallCurve:
    def test_classifier_path_renders_png(self, tmp_path):
        """传 f1（含 per_class_recall_by_k）：右面板走分类器 per-class Recall，落盘 PNG."""
        out = tmp_path / "pr.png"
        plot_purity_recall_curve(_f2_fixture(), out, _f1_fixture())
        assert out.exists() and out.stat().st_size > 0
        assert out.read_bytes().startswith(b"\x89PNG")

    def test_ann_fallback_renders_png(self, tmp_path):
        """仅传 f2（ANN 模式无 per_class_recall_by_k）：回退 f2.per_class_recall，落盘 PNG."""
        out = tmp_path / "pr_fallback.png"
        plot_purity_recall_curve(_f2_fixture(), out)
        assert out.exists() and out.stat().st_size > 0
        assert out.read_bytes().startswith(b"\x89PNG")

    def test_mixed_f1_without_prbk_falls_back(self, tmp_path):
        """f1 存在但不含 per_class_recall_by_k（空结果路径）：仍回退 f2，不抛异常."""
        out = tmp_path / "pr_empty_f1.png"
        plot_purity_recall_curve(_f2_fixture(), out, {})
        assert out.exists() and out.stat().st_size > 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
