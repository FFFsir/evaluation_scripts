"""Tests for QdrantManager."""
import pytest
from unittest.mock import MagicMock, patch
from KNN_evaluation import qdrant_client as qc
from KNN_evaluation.config import HNSW_M, HNSW_EF_CONSTRUCT
from KNN_evaluation.qdrant_client import QdrantManager


class TestQdrantManagerInit:
    def test_init_stores_parameters(self):
        manager = QdrantManager(
            url="http://localhost:6333",
            collection_name="test_collection",
            timeout=5,
        )
        assert manager.url == "http://localhost:6333"
        assert manager.collection_name == "test_collection"
        assert manager.timeout == 5


class TestHealthCheck:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_health_check_returns_true_when_healthy(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_collections.return_value = "ok"
        mock_client_class.return_value = mock_client

        manager = QdrantManager()
        result = manager.health_check()
        assert result is True

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_health_check_returns_false_on_timeout(self, mock_client_class):
        from qdrant_client.http.exceptions import ResponseHandlingException
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = ResponseHandlingException("timeout")
        mock_client_class.return_value = mock_client

        manager = QdrantManager()
        result = manager.health_check()
        assert result is False


class TestCollectionExists:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_collection_exists_returns_true(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        assert manager.collection_exists() is True


class TestCreateCollection:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_create_collection_calls_api_with_correct_params(self, mock_client_class):
        from qdrant_client import models
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        manager.create_collection(vector_size=64, m=16, ef_construct=100)

        call_args = mock_client.create_collection.call_args
        assert call_args is not None, "create_collection was not called"
        assert call_args.kwargs["collection_name"] == "test_collection"
        vectors_config = call_args.kwargs["vectors_config"]
        assert vectors_config.size == 64
        assert vectors_config.distance == models.Distance.COSINE


class TestCreatePayloadIndices:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_create_payload_indices_calls_api(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        manager.create_payload_indices()

        # 应调用 5 次 create_payload_index
        assert mock_client.create_payload_index.call_count == 5

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_image_id_uses_keyword_index(self, mock_client_class):
        """image_id 应使用 keyword 索引而非 text 索引.

        根因回归：text 索引（word tokenizer）无法高效回答含 . 和 _ 的
        字符串精确匹配，check_image_count 因此触发全量扫描.
        """
        from qdrant_client import models
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        manager.create_payload_indices()

        # 取出 image_id 对应的 create_payload_index 调用参数
        image_id_calls = [
            call
            for call in mock_client.create_payload_index.call_args_list
            if call.kwargs["field_name"] == "image_id"
        ]
        assert len(image_id_calls) == 1
        field_schema = image_id_calls[0].kwargs["field_schema"]
        assert isinstance(field_schema, models.KeywordIndexParams)
        assert field_schema.type == models.KeywordIndexType.KEYWORD


class TestMigrateImageIdIndex:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_deletes_old_index_then_creates_keyword(self, mock_client_class):
        """迁移应先删除旧 text 索引，再重建为 keyword 索引."""
        from qdrant_client import models
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        manager.migrate_image_id_index()

        # 先删除旧索引
        mock_client.delete_payload_index.assert_called_once_with(
            collection_name="test_collection",
            field_name="image_id",
        )
        # 再创建 keyword 索引
        create_call = mock_client.create_payload_index.call_args
        assert create_call is not None
        assert create_call.kwargs["collection_name"] == "test_collection"
        assert create_call.kwargs["field_name"] == "image_id"
        field_schema = create_call.kwargs["field_schema"]
        assert isinstance(field_schema, models.KeywordIndexParams)
        assert field_schema.type == models.KeywordIndexType.KEYWORD

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_idempotent_when_old_index_missing(self, mock_client_class):
        """旧索引不存在（首次迁移）时删除失败不应中断重建."""
        from qdrant_client import models
        mock_client = MagicMock()
        # 模拟旧索引不存在：delete_payload_index 抛异常（如 404）
        mock_client.delete_payload_index.side_effect = Exception("404")
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        manager.migrate_image_id_index()  # 不应抛异常

        create_call = mock_client.create_payload_index.call_args
        assert create_call is not None
        field_schema = create_call.kwargs["field_schema"]
        assert isinstance(field_schema, models.KeywordIndexParams)
        assert field_schema.type == models.KeywordIndexType.KEYWORD


class TestEnsurePayloadIndices:
    """Task 13: ensure_payload_indices 幂等自动补齐缺失的 payload 索引."""

    @staticmethod
    def _schema(*fields: str):
        """构建 payload_schema dict（含指定字段，其余缺失）."""
        return {f: object() for f in fields}

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_creates_only_missing_fields(self, mock_client_class):
        """schema 缺失部分字段时，只对缺失字段 create_payload_index."""
        from qdrant_client import models
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # 仅 label / label_name 已有索引，utm_easting/utm_northing/image_id 缺失
        info = MagicMock()
        info.payload_schema = self._schema("label", "label_name")
        mock_client.get_collection.return_value = info

        manager = QdrantManager(collection_name="test_collection")
        manager.ensure_payload_indices()

        created = [
            c.kwargs["field_name"]
            for c in mock_client.create_payload_index.call_args_list
        ]
        assert set(created) == {"utm_easting", "utm_northing", "image_id"}
        # 每次 create 都指向当前 collection
        for call in mock_client.create_payload_index.call_args_list:
            assert call.kwargs["collection_name"] == "test_collection"
        # utm 字段使用 float 索引（与 create_payload_indices 一致）
        utm_call = next(
            c for c in mock_client.create_payload_index.call_args_list
            if c.kwargs["field_name"] == "utm_easting"
        )
        assert isinstance(utm_call.kwargs["field_schema"], models.FloatIndexParams)
        assert utm_call.kwargs["field_schema"].type == models.FloatIndexType.FLOAT

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_idempotent_when_schema_complete(self, mock_client_class):
        """schema 齐全时不得发起任何 create_payload_index."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        info = MagicMock()
        info.payload_schema = self._schema(
            "label", "label_name", "utm_easting", "utm_northing", "image_id",
        )
        mock_client.get_collection.return_value = info

        manager = QdrantManager(collection_name="test_collection")
        manager.ensure_payload_indices()

        mock_client.create_payload_index.assert_not_called()

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_none_value_treated_as_missing(self, mock_client_class):
        """schema 中某字段值为 None（缺 key 之外的形式）也视为缺失."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        info = MagicMock()
        info.payload_schema = {
            "label": object(), "label_name": object(),
            "utm_easting": None, "utm_northing": object(),
            "image_id": object(),
        }
        mock_client.get_collection.return_value = info

        manager = QdrantManager(collection_name="test_collection")
        manager.ensure_payload_indices()

        created = [
            c.kwargs["field_name"]
            for c in mock_client.create_payload_index.call_args_list
        ]
        assert created == ["utm_easting"]

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_none_schema_creates_all(self, mock_client_class):
        """payload_schema 为 None（空 collection 无 payload）时补齐全部 5 个字段."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        info = MagicMock()
        info.payload_schema = None
        mock_client.get_collection.return_value = info

        manager = QdrantManager(collection_name="test_collection")
        manager.ensure_payload_indices()

        created = {
            c.kwargs["field_name"]
            for c in mock_client.create_payload_index.call_args_list
        }
        assert created == {"label", "label_name", "utm_easting", "utm_northing", "image_id"}


def _manager() -> QdrantManager:
    return QdrantManager(url="http://localhost:1", collection_name="c")


def _seed_manifest(monkeypatch, images: dict, collection="c"):
    """替换 qdrant_client 命名空间内的 load_manifest/save_manifest，避免真实文件."""
    state = {"collection": collection, "images": dict(images), "updated_at": "x"}
    monkeypatch.setattr(qc, "load_manifest", lambda path=None: dict(state))
    def _save(data, path=None):
        state.clear(); state.update(data)
    monkeypatch.setattr(qc, "save_manifest", _save)
    return state


class TestGetImportedImageIdsFromManifest:
    def test_reads_manifest(self, monkeypatch):
        _seed_manifest(monkeypatch, {"A": 16384, "B": 16384})
        m = _manager()
        assert m.get_imported_image_ids() == {"A", "B"}

    def test_missing_manifest_falls_back_to_facet(self, monkeypatch):
        _seed_manifest(monkeypatch, {})          # 空 manifest → 回退 facet
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"X", "Y"}
        assert m.get_imported_image_ids() == {"X", "Y"}


class TestReconcileManifest:
    def test_consistent_no_op(self, monkeypatch):
        _seed_manifest(monkeypatch, {"A": 16384})
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A"}
        out = m.reconcile_manifest()
        assert set(out["images"]) == {"A"}

    def test_adds_missing_manually_counted(self, monkeypatch):
        from KNN_evaluation.data_loader import PixelDataLoader
        _seed_manifest(monkeypatch, {})
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A", "B"}
        # 对差异项精确 count：A 完整导入补 16384；B 无像素（count=0）不写入
        monkeypatch.setattr(
            PixelDataLoader, "check_image_count",
            lambda iid, mgr: 16384 if iid == "A" else 0,
        )
        out = m.reconcile_manifest()
        assert out["images"].get("A") == 16384
        assert "B" not in out["images"]

    def test_removes_stale(self, monkeypatch):
        _seed_manifest(monkeypatch, {"A": 16384, "GHOST": 16384})
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A"}
        out = m.reconcile_manifest()
        assert set(out["images"]) == {"A"}

    def test_rebuild_from_empty(self, monkeypatch):
        from KNN_evaluation.data_loader import PixelDataLoader
        _seed_manifest(monkeypatch, {})
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A"}
        monkeypatch.setattr(PixelDataLoader, "check_image_count", lambda iid, mgr: 16384)
        out = m.reconcile_manifest()
        assert set(out["images"]) == {"A"}


class TestReindexVectors:
    """一键构建向量索引：reindex_vectors 触发全量 HNSW 重建."""

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_reindex_calls_update_collection_with_indexing_threshold_zero(self, mock_client_class):
        from qdrant_client import models
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        manager.reindex_vectors()

        call_args = mock_client.update_collection.call_args
        assert call_args is not None, "update_collection was not called"
        assert call_args.kwargs["collection_name"] == "test_collection"
        opt = call_args.kwargs["optimizer_config"]
        assert isinstance(opt, models.OptimizersConfigDiff)
        assert opt.indexing_threshold == 0


class TestCreateCollectionStorage:
    """create_collection 的 storage 参数（默认 disk 磁盘化）."""

    def _make_manager(self, monkeypatch):
        from KNN_evaluation.qdrant_client import QdrantManager
        m = QdrantManager(url="http://localhost:1", collection_name="c")
        monkeypatch.setattr(m, "collection_exists", lambda: False)
        calls: dict = {}

        def fake_create_collection(**kwargs):
            calls.update(kwargs)

        class _C:
            create_collection = staticmethod(fake_create_collection)

        # client 是只读 @property，实例级 setattr 会抛 AttributeError；
        # 改用类级 patch：把 QdrantManager.client 替换为返回 _C() 的 property。
        monkeypatch.setattr(QdrantManager, "client", property(lambda self: _C()))
        return m, calls

    def test_default_is_disk(self, monkeypatch):
        m, calls = self._make_manager(monkeypatch)
        m.create_collection()
        vp = calls["vectors_config"]
        assert vp.on_disk is True
        assert calls.get("on_disk_payload") is True
        assert calls.get("quantization_config") is None
        # 磁盘化不应改变 HNSW 参数：与现状一致
        hnsw = calls["hnsw_config"]
        assert hnsw.m == HNSW_M
        assert hnsw.ef_construct == HNSW_EF_CONSTRUCT

    def test_ram_preset(self, monkeypatch):
        m, calls = self._make_manager(monkeypatch)
        m.create_collection(storage="ram")
        vp = calls["vectors_config"]
        assert vp.on_disk is False
        assert calls.get("on_disk_payload") is False
        assert calls.get("quantization_config") is None
        # ram 与改造前完全一致：on_disk=False + on_disk_payload=False
        assert calls.get("on_disk_payload") is False
        assert vp.on_disk is False

    def test_invalid_storage(self, monkeypatch):
        m, _ = self._make_manager(monkeypatch)
        with pytest.raises(ValueError):
            m.create_collection(storage="ssd")


class TestDeleteCollection:
    """Task 9: 删除自定义 collection — QdrantManager.delete_collection() 封装."""

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_delete_collection_calls_client(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        result = manager.delete_collection()

        mock_client.delete_collection.assert_called_once_with(
            collection_name="test_collection",
        )
        assert result is True

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_delete_collection_propagates_failure(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = RuntimeError("Qdrant 不可达")
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        with pytest.raises(RuntimeError):
            manager.delete_collection()


class TestManifestPathPerCollection:
    """Task: QdrantManager 的 manifest 读写按当前 collection 派生路径（D4）."""

    def test_get_imported_image_ids_uses_collection_path(self, monkeypatch):
        calls: list = []
        state = {"collection": "c", "images": {"A": 16384}, "updated_at": "x"}
        monkeypatch.setattr(qc, "manifest_path", lambda collection: f"path/{collection}.json")
        monkeypatch.setattr(qc, "load_manifest",
                            lambda path=None: (calls.append(path), dict(state))[1])
        m = _manager()
        m.get_imported_image_ids()
        assert calls and calls[0] == "path/c.json"

    def test_reconcile_manifest_saves_to_collection_path(self, monkeypatch):
        state = {"collection": "c", "images": {"A": 16384, "GHOST": 16384}, "updated_at": "x"}
        load_calls: list = []
        save_calls: list = []
        monkeypatch.setattr(qc, "manifest_path", lambda collection: f"path/{collection}.json")
        monkeypatch.setattr(qc, "load_manifest",
                            lambda path=None: (load_calls.append(path), dict(state))[1])

        def _save(data, path=None):
            save_calls.append(path)
            state.clear()
            state.update(data)

        monkeypatch.setattr(qc, "save_manifest", _save)
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A"}
        out = m.reconcile_manifest()
        assert save_calls and save_calls[0] == "path/c.json"
        assert set(out["images"]) == {"A"}
