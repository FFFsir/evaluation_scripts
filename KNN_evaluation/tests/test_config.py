"""Tests for KNN_evaluation.config collection constants."""
from KNN_evaluation import config


class TestCollectionConstants:
    def test_default_collection(self):
        assert config.DEFAULT_COLLECTION == "google_aef_embedding"

    def test_preset_collections(self):
        assert config.PRESET_COLLECTIONS == ["google_aef_embedding", "xian_aef_embedding"]

    def test_collection_name_alias(self):
        assert config.COLLECTION_NAME == config.DEFAULT_COLLECTION

    def test_qdrant_timeout_accommodates_slow_indexing(self):
        """QDRANT_TIMEOUT 足够覆盖索引未就绪时的慢读（实测 10-15s）."""
        assert config.QDRANT_TIMEOUT >= 60
