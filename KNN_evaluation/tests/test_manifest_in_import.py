"""Tests: import_directory 每张影像（导入/跳过）后同步更新 manifest（IMP-1 修复）.

背景：Design Doc §4.4 / import-manifest spec 要求 `import_directory` 每张影像
导入或跳过完成后同步调用 `update_manifest`，保证 manifest 与数据库同步咽喉点
唯一、CLI 与 WebUI 导入（共用本入口）都不遗漏。本文件以 mock 断言为主，
不依赖真实 Qdrant。
"""
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.data_loader import ImagePair
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
class TestImportUpdatesManifest:
    def _setup(self, mock_loader, mock_utm, pairs, counts):
        mock_loader.scan_directory.return_value = pairs
        mock_loader.check_image_count.side_effect = lambda iid, mgr: counts.get(iid, 0)
        mock_loader.load_se.return_value = np.zeros((64, 128, 128), dtype=np.float64)
        mock_loader.load_dw.return_value = np.zeros((128, 128), dtype=np.uint8)
        mock_utm.return_value = (
            np.zeros((128, 128), dtype=np.float64),
            np.zeros((128, 128), dtype=np.float64),
            51,
        )
        return _make_manager()

    @patch("KNN_evaluation.importer.update_manifest")
    def test_calls_update_manifest_for_import_and_skip(self, mock_update, mock_loader, mock_utm):
        """导入与跳过两路径：每张影像后都调用 update_manifest，像素数为 16384."""
        pairs = [
            _make_pair("E121.4025_N25.1947"),  # 未导入 → 导入分支
            _make_pair("E122.4025_N26.1947"),  # 已完整导入 → 跳过分支
        ]
        counts = {
            "E121.4025_N25.1947": 0,
            "E122.4025_N26.1947": 16384,
        }
        manager = self._setup(mock_loader, mock_utm, pairs, counts)
        importer = PixelImporter(manager, batch_size=10000)

        stats = importer.import_directory(Path("/fake/data"))

        assert mock_update.call_count == 2
        assert mock_update.call_args_list == [
            call("E121.4025_N25.1947", 16384, "test"),
            call("E122.4025_N26.1947", 16384, "test"),
        ]
        assert stats["imported_images"] == 1
        assert stats["skipped_images"] == 1
        assert stats.get("manifest_updated") is True

    @patch("KNN_evaluation.importer.update_manifest")
    def test_calls_update_manifest_with_progress_callback(self, mock_update, mock_loader, mock_utm):
        """WebUI 导入路径（progress_callback 非 None）同样无条件更新 manifest."""
        pairs = [_make_pair("E121.4025_N25.1947")]
        counts = {"E121.4025_N25.1947": 0}
        manager = self._setup(mock_loader, mock_utm, pairs, counts)
        importer = PixelImporter(manager, batch_size=10000)

        progress: list = []
        stats = importer.import_directory(
            Path("/fake/data"), progress_callback=lambda a, b: progress.append((a, b)),
        )

        assert mock_update.call_count == 1
        assert mock_update.call_args_list == [call("E121.4025_N25.1947", 16384, "test")]
        assert stats.get("manifest_updated") is True

    @patch("KNN_evaluation.importer.update_manifest")
    def test_skip_path_records_count_value(self, mock_update, mock_loader, mock_utm):
        """跳过路径：update_manifest 记录 count 值（此处为 16384），不写 0."""
        pairs = [_make_pair("E121.4025_N25.1947")]
        counts = {"E121.4025_N25.1947": 16384}
        manager = self._setup(mock_loader, mock_utm, pairs, counts)
        importer = PixelImporter(manager, batch_size=10000)

        stats = importer.import_directory(Path("/fake/data"))

        assert mock_update.call_args_list == [call("E121.4025_N25.1947", 16384, "test")]
        assert stats["imported_images"] == 0
        assert stats["skipped_images"] == 1
