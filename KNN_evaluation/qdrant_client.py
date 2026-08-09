"""Qdrant 连接管理与 Collection 生命周期."""
from pathlib import Path
from qdrant_client import QdrantClient, models
from KNN_evaluation.config import (
    QDRANT_URL, COLLECTION_NAME, VECTOR_SIZE, QDRANT_TIMEOUT,
    HNSW_M, HNSW_EF_CONSTRUCT,
)
from KNN_evaluation.manifest import manifest_path, load_manifest, save_manifest


class QdrantManager:
    """封装 Qdrant 连接创建、健康检查、Collection 管理."""

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
            self._client = QdrantClient(
                url=self.url,
                timeout=self.timeout,
            )
        return self._client

    def health_check(self) -> bool:
        """检查 Qdrant 服务是否可达.

        Returns:
            True 表示健康，False 表示不可达.
        """
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        """检查目标 Collection 是否已存在."""
        return self.client.collection_exists(self.collection_name)

    def delete_collection(self) -> bool:
        """删除当前 Collection 及其全部数据（不可恢复）.

        Returns:
            True 表示删除成功；Qdrant 侧删除失败时向上抛异常，由调用方决定处理.
        """
        self.client.delete_collection(collection_name=self.collection_name)
        return True

    def create_collection(
        self,
        vector_size: int = VECTOR_SIZE,
        m: int = HNSW_M,
        ef_construct: int = HNSW_EF_CONSTRUCT,
        storage: str = "disk",
    ) -> None:
        """创建 Collection 并配置 HNSW 索引参数.

        若 Collection 已存在则跳过.
        storage: "disk" 时向量与 payload 落盘（on_disk=True/on_disk_payload=True，不量化），
                 常驻内存 ≈2-3GB；"ram" 保持全内存（on_disk=False，现状）.
        """
        if storage not in ("disk", "ram"):
            raise ValueError(f"storage 必须是 'disk' 或 'ram'，实际: {storage!r}")
        if self.collection_exists():
            return

        disk = storage == "disk"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=disk,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=m,
                ef_construct=ef_construct,
            ),
            on_disk_payload=disk,
            quantization_config=None,
        )

    def create_payload_indices(self) -> None:
        """为标量字段创建 payload 索引.

        索引字段: label, label_name, utm_easting, utm_northing, image_id.
        utm_zone, pixel_row, pixel_col 不需要过滤/聚合，不建索引.
        """
        indices = [
            ("label", models.IntegerIndexParams(
                type=models.IntegerIndexType.INTEGER, lookup=True, range=True,
            )),
            ("label_name", models.TextIndexParams(
                type=models.TextIndexType.TEXT,
            )),
            ("utm_easting", models.FloatIndexParams(
                type=models.FloatIndexType.FLOAT,
            )),
            ("utm_northing", models.FloatIndexParams(
                type=models.FloatIndexType.FLOAT,
            )),
            ("image_id", models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            )),
        ]

        for field_name, field_schema in indices:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

    def ensure_payload_indices(self) -> None:
        """自动补齐缺失的 payload 索引（幂等，防 UTM 过滤全量扫描超时）.

        读取 collection 的 payload_schema，对过滤/聚合字段
        (label, label_name, utm_easting, utm_northing, image_id) 中缺失索引的
        逐个 create_payload_index；全部已存在时不发起任何创建请求。
        字段列表与 create_payload_indices 保持一致；缺失判定按
        payload_schema 缺 key 或值为 None 两种形式处理。

        背景：xian_aef_embedding 缺 utm_easting/utm_northing 索引时，UTM 过滤
        走全量扫描 1023 万点耗时 5.03s > QDRANT_TIMEOUT=5s 触发超时；补建索引后
        降至约 274ms。本方法在分页切换/页面加载时调用以自愈，防止再次发生。
        """
        indices = [
            ("label", models.IntegerIndexParams(
                type=models.IntegerIndexType.INTEGER, lookup=True, range=True,
            )),
            ("label_name", models.TextIndexParams(
                type=models.TextIndexType.TEXT,
            )),
            ("utm_easting", models.FloatIndexParams(
                type=models.FloatIndexType.FLOAT,
            )),
            ("utm_northing", models.FloatIndexParams(
                type=models.FloatIndexType.FLOAT,
            )),
            ("image_id", models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            )),
        ]
        info = self.client.get_collection(self.collection_name)
        schema = info.payload_schema or {}
        for field_name, field_schema in indices:
            if schema.get(field_name) is not None:
                continue  # 已有索引，跳过
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

    def migrate_image_id_index(self) -> None:
        """迁移 image_id 索引：删除旧 text 索引并重建为 keyword 索引.

        旧版本对 image_id 使用 text 索引（word tokenizer），无法高效回答
        含 . 和 _ 的字符串精确匹配；check_image_count 的 MatchValue 查询
        因此触发全量 payload 扫描（实测 count 约 1.41s）。重建为 keyword
        索引后精确 count 降至约 0.01s（约 140× 快）。

        幂等且廉价：先检查当前 schema —— 已是 keyword 索引则直接跳过
        （避免每次导入都触发删除/重建窗口）；仅当 text 索引或索引缺失时
        才删除并重建。旧索引不存在时删除步骤会抛异常，此处吞掉以容忍
        首次迁移，可安全重试.
        """
        # 0) 检查当前 schema：已是 keyword 索引则无需迁移
        info = self.client.get_collection(self.collection_name)
        index_info = (info.payload_schema or {}).get("image_id")
        if (
            index_info is not None
            and index_info.data_type == models.PayloadSchemaType.KEYWORD
        ):
            return  # 已满足目标形态，跳过重建

        # 1) 删除旧索引（可能不存在，容忍失败以保持幂等）
        try:
            self.client.delete_payload_index(
                collection_name=self.collection_name,
                field_name="image_id",
            )
        except Exception:
            # 旧索引不存在时 Qdrant 抛 404，忽略并继续重建
            pass

        # 2) 重建为 keyword 索引
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="image_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

    def _manifest_file(self) -> Path:
        """当前 collection 的 manifest 文件路径（按 collection 隔离，D4）."""
        return manifest_path(self.collection_name)

    def get_imported_image_ids(self) -> set[str]:
        """从本地 manifest 读取已导入 image_id（毫秒级，Qdrant 离线可用）.

        无 manifest（空清单）时回退 facet 重建。
        """
        data = load_manifest(self._manifest_file())
        ids = set((data.get("images") or {}).keys())
        if not ids:
            ids = self._facet_image_ids()
        return ids

    def _facet_image_ids(self, limit: int = 1000) -> set[str]:
        """用 facet 取 Collection 内去重 image_id 集合（单次请求，keyword 索引毫秒级）."""
        resp = self.client.facet(
            collection_name=self.collection_name,
            key="image_id",
            limit=limit,
            exact=True,
        )
        return {v.value for v in resp.hits}

    def reconcile_manifest(self) -> dict:
        """用 facet 对账 manifest：一致不改、不一致刷新、缺失重建.

        对差异项用 check_image_count 补精确像素数（facet 只返回去重集合，不含像素数）。
        Returns: 对账后的 manifest dict.
        """
        from KNN_evaluation.data_loader import PixelDataLoader  # 函数内 import 防循环
        db_ids = self._facet_image_ids()
        current = load_manifest(self._manifest_file())
        manifest_ids = set((current.get("images") or {}).keys())

        if db_ids == manifest_ids:
            return current

        images: dict[str, int] = {}
        for iid in db_ids:
            if iid in manifest_ids:
                images[iid] = int((current.get("images") or {}).get(iid, 0))
            else:
                # 新增差异项：精确 count（复用 check_image_count）；0 像素为幻影项，不写入
                count = PixelDataLoader.check_image_count(iid, self)
                if count:
                    images[iid] = count
        save_manifest({
            "collection": self.collection_name,
            "images": images,
            "updated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }, self._manifest_file())
        return load_manifest(self._manifest_file())

    def collection_info(self) -> dict:
        """获取 Collection 统计信息.

        Returns:
            包含 total_points, segments_count 等字段的字典.
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "total_points": info.points_count,
            "vectors_count": info.indexed_vectors_count or 0,
            "segments_count": info.segments_count,
            "status": str(info.status),
        }

    def reindex_vectors(self) -> None:
        """触发全量 HNSW 向量索引重建（一键构建索引）.

        通过 `update_collection(indexing_threshold=0)` 强制 Qdrant 对全部向量
        重建 HNSW 索引，解决索引构建停滞（indexed_vectors_count < points_count）
        导致读操作退化为磁盘扫描、采样/评估超时的问题。重建在 Qdrant 后台异步
        执行，调用方可通过 `collection_info()` 的 `vectors_count` 轮询进度。
        仅重建 HNSW 向量索引，不重建 payload 标量索引（与 importer reindex 一致）。
        """
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
        )
