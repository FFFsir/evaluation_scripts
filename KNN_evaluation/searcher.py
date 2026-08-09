"""联合检索接口：向量检索 + 标量过滤."""
import time
from dataclasses import dataclass, field
import numpy as np
from qdrant_client import models


@dataclass
class HitRecord:
    """单条检索命中记录."""
    id: str
    score: float
    label: int
    label_name: str
    utm_easting: float
    utm_northing: float
    utm_zone: int
    image_id: str
    pixel_row: int
    pixel_col: int


@dataclass
class SearchResult:
    """检索结果集."""
    hits: list[HitRecord]
    elapsed_ms: float
    label_distribution: dict[str, int] = field(default_factory=dict)
    search_mode: str = "ann"
    query_params: dict = field(default_factory=dict)


class PixelSearcher:
    """像素嵌入向量检索器.

    封装 Qdrant search API，支持 label 过滤、UTM 范围过滤、exact/ANN 模式切换.
    """

    def __init__(self, manager):
        self.manager = manager

    def _build_filter(
        self,
        label_filter: list[int] | None,
        utm_range: dict | None,
    ) -> models.Filter | None:
        """构建 Qdrant 过滤条件（AND 语义）.

        Args:
            label_filter: 允许的标签值列表，如 [0, 1].
            utm_range: UTM 范围，含 min_e/max_e/min_n/max_n 键.

        Returns:
            Filter 对象或 None.
        """
        conditions = []

        if label_filter:
            conditions.append(
                models.FieldCondition(
                    key="label",
                    match=models.MatchAny(any=label_filter),
                )
            )

        if utm_range:
            conditions.extend([
                models.FieldCondition(
                    key="utm_easting",
                    range=models.Range(
                        gte=utm_range["min_e"],
                        lte=utm_range["max_e"],
                    ),
                ),
                models.FieldCondition(
                    key="utm_northing",
                    range=models.Range(
                        gte=utm_range["min_n"],
                        lte=utm_range["max_n"],
                    ),
                ),
            ])

        if not conditions:
            return None
        return models.Filter(must=conditions)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        label_filter: list[int] | None = None,
        utm_range: dict | None = None,
        exact: bool = False,
        ef_search: int = 64,
    ) -> SearchResult:
        """执行向量检索.

        Args:
            query_vector: (64,) float64 查询向量.
            k: 返回 Top-K 结果.
            label_filter: 标签过滤列表.
            utm_range: UTM 坐标范围过滤.
            exact: True 时使用暴力精确搜索.
            ef_search: ANN 搜索的 ef 参数.

        Returns:
            SearchResult 包含命中记录、耗时、标签分布等.

        Raises:
            ValueError: query_vector 维度不是 64.
        """
        if query_vector.shape != (64,):
            raise ValueError(
                f"query_vector 维度应为 (64,)，实际: {query_vector.shape}"
            )

        qdrant_filter = self._build_filter(label_filter, utm_range)

        start = time.perf_counter()
        hits = self.manager.client.query_points(
            collection_name=self.manager.collection_name,
            query=query_vector.tolist(),
            query_filter=qdrant_filter,
            limit=k,
            search_params=models.SearchParams(
                exact=exact,
                hnsw_ef=None if exact else ef_search,
            ),
            with_payload=True,
        )
        # 新版 SDK 返回 QueryResponse, .points 是结果列表
        points = hits.points if hasattr(hits, 'points') else hits
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 解析 hits 并计算 label 分布
        hit_records: list[HitRecord] = []
        label_dist: dict[str, int] = {}
        for h in points:
            p = h.payload or {}
            record = HitRecord(
                id=str(h.id),
                score=float(h.score),
                label=int(p.get("label", -1)),
                label_name=str(p.get("label_name", "unknown")),
                utm_easting=float(p.get("utm_easting", float("nan"))),
                utm_northing=float(p.get("utm_northing", float("nan"))),
                utm_zone=int(p.get("utm_zone", -1)),
                image_id=str(p.get("image_id", "")),
                pixel_row=int(p.get("pixel_row", -1)),
                pixel_col=int(p.get("pixel_col", -1)),
            )
            hit_records.append(record)
            label_dist[record.label_name] = label_dist.get(record.label_name, 0) + 1

        return SearchResult(
            hits=hit_records,
            elapsed_ms=elapsed_ms,
            label_distribution=label_dist,
            search_mode="exact" if exact else "ann",
            query_params={
                "k": k,
                "label_filter": label_filter,
                "utm_range": utm_range,
                "exact": exact,
                "ef_search": ef_search,
            },
        )
