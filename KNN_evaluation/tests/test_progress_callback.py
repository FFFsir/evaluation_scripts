"""Tests for import_directory progress_callback."""
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.data_loader import ImagePair, PixelDataLoader
from KNN_evaluation.qdrant_client import QdrantManager


def _make_pair(image_id: str) -> ImagePair:
    return ImagePair(
        image_id=image_id,
        se_path=Path(f"/fake/{image_id}/se.npy"),
        dw_path=Path(f"/fake/{image_id}/dw.npy"),
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
class TestProgressCallback:
    def _setup(self, mock_loader, mock_utm, pairs, check_count=0):
        mock_loader.scan_directory.return_value = pairs
        mock_loader.check_image_count.return_value = check_count
        mock_loader.load_se.return_value = np.zeros((64, 128, 128), dtype=np.float64)
        mock_loader.load_dw.return_value = np.zeros((128, 128), dtype=np.uint8)
        mock_utm.return_value = (
            np.zeros((128, 128), dtype=np.float64),
            np.zeros((128, 128), dtype=np.float64),
            51,
        )
        return _make_manager()

    def test_callback_advances_per_batch(self, mock_loader, mock_utm):
        pairs = [
            _make_pair("E121.4025_N25.1947"),
            _make_pair("E122.4025_N26.1947"),
        ]
        manager = self._setup(mock_loader, mock_utm, pairs, check_count=0)

        importer = PixelImporter(manager, batch_size=10000)
        calls: list[tuple[int, int]] = []

        def cb(imported, total):
            calls.append((imported, total))

        stats = importer.import_directory(Path("/fake/data"), progress_callback=cb)

        # 2 张影像 × 16384，每张按 batch_size=10000 分 2 批 (10000, 6384)
        assert calls == [
            (10000, 32768),
            (16384, 32768),
            (26384, 32768),
            (32768, 32768),
        ]
        assert stats["total_pixels"] == 32768
        assert stats["imported_images"] == 2

    def test_callback_none_behaves_identically(self, mock_loader, mock_utm):
        pairs = [_make_pair("E121.4025_N25.1947")]
        manager = self._setup(mock_loader, mock_utm, pairs, check_count=0)

        importer = PixelImporter(manager, batch_size=10000)
        stats = importer.import_directory(Path("/fake/data"))

        assert stats["total_pixels"] == 16384
        assert stats["imported_images"] == 1
        manager.client.upsert.assert_called()

    def test_all_skipped_total_is_zero(self, mock_loader, mock_utm):
        pairs = [_make_pair("E121.4025_N25.1947")]
        manager = self._setup(mock_loader, mock_utm, pairs, check_count=16384)

        importer = PixelImporter(manager, batch_size=10000)
        calls: list[tuple[int, int]] = []

        def cb(imported, total):
            calls.append((imported, total))

        stats = importer.import_directory(Path("/fake/data"), progress_callback=cb)

        assert stats["skipped_images"] == 1
        assert stats["imported_images"] == 0
        assert stats["total_pixels"] == 0
        assert calls == []
        manager.client.upsert.assert_not_called()


@patch("KNN_evaluation.importer.compute_utm_grid")
@patch("KNN_evaluation.importer.compute_utm_grid_from_name")
class TestUtmPreferredPath:
    """文件名解析成功时，UTM 应优先从文件名推算，而非回退 GeoTIFF.

    这里只 patch PixelDataLoader 的扫描/加载方法，保留真实的
    parse_location_coord，使 image_id 坐标段真正参与解析。
    """

    def _setup(self, mock_from_name, mock_utm, image_id="E121.4025_N25.1947"):
        manager = _make_manager()
        with (
            patch.object(PixelDataLoader, "scan_directory", return_value=[_make_pair(image_id)]),
            patch.object(PixelDataLoader, "check_image_count", return_value=0),
            patch.object(PixelDataLoader, "load_se",
                         return_value=np.zeros((64, 128, 128), dtype=np.float64)),
            patch.object(PixelDataLoader, "load_dw",
                         return_value=np.zeros((128, 128), dtype=np.uint8)),
        ):
            mock_from_name.return_value = (
                np.zeros((128, 128), dtype=np.float64),
                np.zeros((128, 128), dtype=np.float64),
                51,
            )
            mock_utm.return_value = (
                np.zeros((128, 128), dtype=np.float64),
                np.zeros((128, 128), dtype=np.float64),
                51,
            )
            importer = PixelImporter(manager, batch_size=10000)
            importer.import_directory(Path("/fake/data"))
        return mock_from_name, mock_utm

    def test_filename_parsed_prefers_from_name(self, mock_from_name, mock_utm):
        """image_id 坐标段可解析时：调用 compute_utm_grid_from_name，不调用 compute_utm_grid."""
        mock_from_name, mock_utm = self._setup(mock_from_name, mock_utm)
        # 主路径走文件名推算
        mock_from_name.assert_called_once()
        # 不应回退 GeoTIFF 路径
        mock_utm.assert_not_called()

    def test_unparsable_filename_falls_back(self, mock_from_name, mock_utm):
        """image_id 坐标段无法解析时：回退 compute_utm_grid（GeoTIFF 路径）."""
        mock_from_name, mock_utm = self._setup(mock_from_name, mock_utm, image_id="no_coord_here")
        mock_from_name.assert_not_called()
        mock_utm.assert_called_once()
