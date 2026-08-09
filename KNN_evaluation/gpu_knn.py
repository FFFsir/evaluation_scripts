"""GPU/CPU device-agnostic 分块精确 KNN 核心.

全库向量 float32 常驻 device（cuda 或 torch cpu），查询分块做
`query_block @ corpus.T` + `torch.topk`，边算边回传 CPU，不物化 (Q,N)
相似度矩阵或 (Q,K) 全量结果。L2 归一化语义与旧 numpy `_batch_exact_knn`
一致（cosine 相似度）。torch 延迟 import——无 CUDA 环境启动不失败。
"""
import numpy as np


def resolve_device(device: str) -> str:
    """把用户指定 device 解析为 'cuda' | 'cpu'.

    'auto' → CUDA 可用则 'cuda'，否则 'cpu'（并提示回退）。
    'cuda' 显式但 CUDA 不可用 → 抛明确错误（不静默回退）。
    'cpu' → 原样返回。
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        import warnings
        warnings.warn("CUDA 不可用，回退 torch CPU 分块路径", RuntimeWarning)
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' 但当前环境 CUDA 不可用（torch.cuda.is_available()=False）。"
                "请改用 --device cpu 或 --device auto。"
            )
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"未知 device: {device!r}，可选: cuda|cpu|auto")


class KnnEngine:
    """device-agnostic 分块精确 KNN 引擎（cuda / torch cpu 共用一套代码）。"""

    def __init__(self, device: str = "auto"):
        import torch  # 延迟 import：无 torch 环境（纯 numpy 路径）不失败
        self.torch = torch
        self.device = resolve_device(device)
        self.corpus = None
        self.N = 0
        self._labels: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    def set_corpus(self, all_vectors: np.ndarray) -> None:
        """全量向量 float32 (N, 64) 常驻 device，L2 归一化；另存 CPU labels/ids.

        未显式 attach 元数据时生成默认 labels（全 0）与 ids（`pt-{i}`），
        保证 knn_chunk 立即可用；真实元数据可由 `_attach_meta` 覆盖。
        """
        all_vectors = np.asarray(all_vectors)
        if all_vectors.ndim != 2 or all_vectors.shape[1] != 64:
            raise ValueError(
                f"corpus 必须是 (N, 64) 二维数组，实际 {all_vectors.shape}"
            )
        self.N = int(all_vectors.shape[0])
        t = self.torch
        tensor = t.from_numpy(all_vectors.astype(np.float32)).to(self.device)
        self.corpus = tensor / tensor.norm(dim=1, keepdim=True).clamp(min=1e-12)
        if self._labels is None:
            self._labels = np.zeros(self.N, dtype=np.int64)
        if self._ids is None:
            self._ids = np.array([f"pt-{i}" for i in range(self.N)])

    def _attach_meta(self, labels: np.ndarray, point_ids: np.ndarray) -> None:
        """预取标签/point_id 为 CPU numpy，knn_chunk 花式索引取回（避免 GPU 传输）."""
        self._labels = np.asarray(labels, dtype=np.int64)
        self._ids = np.asarray(point_ids)

    def knn_chunk(self, query_block: np.ndarray, k: int) -> tuple:
        """单块分块 matmul：L2 归一化 → q @ corpus.T → topk → CPU (scores, labels, ids)."""
        if self.corpus is None:
            raise ValueError("KnnEngine.set_corpus 未调用")
        t = self.torch
        q = t.from_numpy(np.asarray(query_block).astype(np.float32)).to(self.device)
        q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
        sim = q @ self.corpus.T                      # (block_q, N) float32
        k_eff = min(int(k), self.N)
        scores, idx = t.topk(sim, k=k_eff, dim=1)
        idx_np = idx.cpu().numpy()
        labels = self._labels[idx_np]                # 花式索引取标签（CPU numpy）
        ids = self._ids[idx_np]                      # 取 point_id
        # float32 matmul 舍入可能让自相似度略超 1.0（如 1.0000001），
        # cosine 相似度数学上界 [-1, 1]，clip 到该范围并转 float64 输出。
        scores_np = np.clip(scores.cpu().numpy(), -1.0, 1.0).astype(np.float64)
        return scores_np, labels, ids

    def estimate_block_q(self, max_mem_gb: float) -> int:
        """显存/内存预算 → block_q（峰值占用 < 预算，×0.8 保守余量）.

        block_q = clamp(floor(0.8 × (max_mem_GB − corpus_GB) × 1e9 / (N × 4)), 1, Q)
        corpus_GB = N × 64 × 4 / 1e9
        """
        if self.corpus is None:
            raise ValueError("KnnEngine.set_corpus 未调用，无法推导 block_q")
        N = self.N
        corpus_gb = N * 64 * 4 / 1e9
        if max_mem_gb <= corpus_gb:
            return 1
        block_q = int(0.8 * (float(max_mem_gb) - corpus_gb) * 1e9 // (N * 4))
        return max(1, min(block_q, N))

    def close(self) -> None:
        """释放 device 内存.

        cuda 设备额外调用 `torch.cuda.empty_cache()` 归还 PyTorch 申请的 GPU
        缓存块，避免评估完成后显存不回落（verify 缺陷：F1 完成后 F2 重新
        set_corpus 叠加未释放显存导致 OOM）。
        """
        self.corpus = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
