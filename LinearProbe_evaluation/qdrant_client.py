"""Qdrant 连接管理与像素数据读取（Linear Probe 专用薄封装）.

复用与 KNN 评估共享的 Qdrant Collection（默认 ``xian_aef_embedding``），
只暴露本模块需要的连接 / 统计 / scroll 读取能力；Collection 的创建、
索引维护等生命周期操作仍由 KNN_evaluation 负责，此处不重复实现。
"""
from qdrant_client import QdrantClient

from LinearProbe_evaluation.config import QDRANT_URL, COLLECTION_NAME, QDRANT_TIMEOUT


class QdrantManager:
    """封装 Qdrant 连接、健康检查、Collection 统计与逐批数据读取."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
        timeout: int = QDRANT_TIMEOUT,
    ):
        self.url = url
        self.collection_name = collection_name
        self.timeout = timeout
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.url, timeout=self.timeout)
        return self._client

    def health_check(self) -> bool:
        """检查 Qdrant 服务是否可达."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        """检查目标 Collection 是否已存在."""
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception:
            return False

    def collection_info(self) -> dict:
        """获取 Collection 统计信息.

        Returns:
            含 total_points / vectors_count / segments_count / status 的字典.
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "total_points": info.points_count,
            "vectors_count": info.indexed_vectors_count or 0,
            "segments_count": info.segments_count,
            "status": str(info.status),
        }

    def scroll_vectors_and_labels(
        self,
        batch_size: int,
        limit: int | None = None,
    ) -> tuple[list[list[float]], list[int]]:
        """逐批 scroll 读取全部点的向量与 label 标签.

        Args:
            batch_size: 每批读取点数.
            limit: 最多读取的点数（None 表示全量）.

        Returns:
            (vectors, labels)：vectors 为 list[list[float]]（每点 64 维），
            labels 为 list[int]（0-8，缺失 label 的点跳过）.
        """
        vectors: list[list[float]] = []
        labels: list[int] = []
        next_offset = None
        seen = 0
        while True:
            if limit is not None and seen >= limit:
                break
            batch_limit = batch_size
            if limit is not None:
                batch_limit = min(batch_size, limit - seen)
            resp = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_limit,
                offset=next_offset,
                with_vectors=True,
                with_payload=["label"],
            )
            points = resp[0]
            next_offset = resp[1]
            for pt in points:
                label = (pt.payload or {}).get("label")
                if label is None or pt.vector is None:
                    continue
                vectors.append(pt.vector)
                labels.append(int(label))
                seen += 1
            if next_offset is None or not points:
                break
        return vectors, labels
