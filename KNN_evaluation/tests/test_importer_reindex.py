"""Tests for import_directory reindex behavior (HNSW rebuild)."""
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.data_loader import ImagePair
from KNN_evaluation.qdrant_client import QdrantManager


def _make_pair() -> ImagePair:
    return ImagePair(
        image_id="E121.4025_N25.1947",
        se_path=Path("/fake/se.npy"),
        dw_path=Path("/fake/dw.npy"),
        tif_path=None,
    )


def _make_manager():
    manager = MagicMock(spec=QdrantManager)
    manager.health_check.return_value = True
    manager.collection_exists.return_value = True
    manager.url = "http://localhost:6333"
    manager.collection_name = "test"
    manager.client.upsert.return_value = None
    return manager


@patch("KNN_evaluation.importer.compute_utm_grid")
@patch("KNN_evaluation.importer.PixelDataLoader")
class TestReindex:
    def _setup(self, mock_loader, mock_utm):
        mock_loader.scan_directory.return_value = [_make_pair()]
        mock_loader.check_image_count.return_value = 0
        mock_loader.load_se.return_value = np.zeros((64, 128, 128), dtype=np.float64)
        mock_loader.load_dw.return_value = np.zeros((128, 128), dtype=np.uint8)
        mock_utm.return_value = (
            np.zeros((128, 128), dtype=np.float64),
            np.zeros((128, 128), dtype=np.float64),
            51,
        )

    def test_reindex_triggers_hnsw_rebuild(self, mock_loader, mock_utm):
        self._setup(mock_loader, mock_utm)
        manager = _make_manager()
        importer = PixelImporter(manager, batch_size=10000)

        stats = importer.import_directory(Path("/fake/data"), reindex=True)

        manager.client.update_collection.assert_called_once()
        call_kwargs = manager.client.update_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "test"
        assert call_kwargs.kwargs["optimizer_config"].indexing_threshold == 0
        assert stats["imported_images"] == 1

    def test_no_reindex_skips_rebuild(self, mock_loader, mock_utm):
        self._setup(mock_loader, mock_utm)
        manager = _make_manager()
        importer = PixelImporter(manager, batch_size=10000)

        importer.import_directory(Path("/fake/data"), reindex=False)

        manager.client.update_collection.assert_not_called()
