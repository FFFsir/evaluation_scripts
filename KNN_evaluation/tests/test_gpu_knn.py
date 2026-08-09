"""KnnEngine（device-agnostic 分块精确 KNN）单元测试."""
import numpy as np
import pytest

from KNN_evaluation.gpu_knn import KnnEngine, resolve_device


@pytest.fixture
def corpus():
    rng = np.random.RandomState(7)
    return rng.randn(1000, 64).astype(np.float32)


def test_resolve_device_auto_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_explicit_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_set_corpus_wrong_dim_raises():
    engine = KnnEngine(device="cpu")
    with pytest.raises(ValueError):
        engine.set_corpus(np.zeros((10, 63), dtype=np.float32))
    engine.close()


def test_knn_chunk_returns_cpu_arrays(corpus):
    engine = KnnEngine(device="cpu")
    engine.set_corpus(corpus)
    q = corpus[:5]  # 查询取自 corpus（含自身）
    scores, labels, ids = engine.knn_chunk(q, k=10)
    assert scores.shape == (5, 10)
    assert labels.shape == (5, 10)
    assert ids.shape == (5, 10)
    assert scores.dtype == np.float64
    assert labels.dtype == np.int64
    assert ids.dtype.kind == "U"
    # 余弦相似度范围 [-1, 1]
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
    engine.close()


def test_knn_top1_self_match(corpus):
    engine = KnnEngine(device="cpu")
    engine.set_corpus(corpus)
    scores, labels, ids = engine.knn_chunk(corpus[:3], k=1)
    # 查询向量来自 corpus 前 3 行，Top-1 应为自身
    assert list(ids[:, 0]) == ["pt-0", "pt-1", "pt-2"]
    assert abs(scores[0, 0] - 1.0) < 1e-3
    engine.close()


def test_estimate_block_q(corpus):
    engine = KnnEngine(device="cpu")
    engine.set_corpus(corpus)
    N, D = corpus.shape
    assert D == 64
    corpus_gb = N * 64 * 4 / 1e9
    # 预算 1GB：block_q = floor(0.8*(1-corpus_gb)*1e9/(N*4))，clamp 到 [1, Q]
    bq = engine.estimate_block_q(1.0)
    assert 1 <= bq <= N
    # 更大预算 → 更大块
    assert engine.estimate_block_q(8.0) >= bq
    engine.close()


def test_estimate_block_q_without_corpus_raises():
    engine = KnnEngine(device="cpu")
    with pytest.raises(ValueError):
        engine.estimate_block_q(1.0)
    engine.close()


def _has_cuda():
    """探测 CUDA 是否可用（差分测试的 GPU 分支跳过条件）."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _skip_qdrant():
    """本地无运行中的 Qdrant 容器时跳过服务端逐条对照."""
    try:
        import subprocess
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        return not ("qdrant" in r.stdout or "qdrant-knn-eval" in r.stdout)
    except Exception:
        return True


@pytest.mark.skipif(not _has_cuda(), reason="CUDA 不可用")
def test_gpu_cpu_differential(corpus):
    """GPU 分块 vs torch CPU 分块 Top-K 标签/顺序完全一致."""
    import torch
    from KNN_evaluation.gpu_knn import KnnEngine

    q = corpus[:100]
    ref = None
    for dev in ("cpu", "cuda"):
        engine = KnnEngine(device=dev)
        engine.set_corpus(corpus)
        engine._attach_meta(
            np.arange(1000, dtype=np.int64),
            np.array([f"pt-{i}" for i in range(1000)]),
        )
        scores, labels, ids = engine.knn_chunk(q, k=100)
        engine.close()
        assert labels.shape == (100, 100)
        if ref is None:
            ref = (labels, ids)
        else:
            assert np.array_equal(labels, ref[0])
            assert np.array_equal(ids, ref[1])


@pytest.mark.integration
class TestGpuVsSequentialDifferential:
    @pytest.mark.skipif(not _has_cuda() and _skip_qdrant(), reason="需 Qdrant 运行")
    def test_gpu_path_matches_sequential(self, qdrant_manager):
        """GPU 分块路径（KnnEngine torch CPU）与 Qdrant 服务端逐条 Top-K 标签/顺序一致."""
        import numpy as np
        from KNN_evaluation.metrics import (
            sample_queries_by_label,
            compute_knn_accuracy, compute_purity_recall_curve,
            _knn_accuracy_sequential, _purity_recall_sequential,
            _compute_per_class_label_totals,
        )
        queries = sample_queries_by_label(qdrant_manager, samples_per_class=4, seed=7)
        # 分块路径（KnnEngine，torch CPU） vs Qdrant 服务端逐条 exact
        bat = compute_knn_accuracy(qdrant_manager, queries, k=3, exact=True, device="cpu")
        seq = _knn_accuracy_sequential(qdrant_manager, queries, 3, True, None)
        assert seq["overall_accuracy"] == bat["overall_accuracy"]
        assert seq["accuracy_by_k"] == bat["accuracy_by_k"]
        label_totals = _compute_per_class_label_totals(qdrant_manager)
        bat2 = compute_purity_recall_curve(qdrant_manager, queries, [2, 3], exact=True, device="cpu")
        seq2 = _purity_recall_sequential(qdrant_manager, queries, [2, 3], label_totals, True, None)
        assert seq2["global_purity"] == bat2["global_purity"]
        assert seq2["global_recall"] == bat2["global_recall"]
