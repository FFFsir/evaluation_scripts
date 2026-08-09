"""Tests for PixelImporter."""
import numpy as np
import uuid
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.data_loader import ImagePair
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.label_mapping import LABEL_NAMES


class TestBuildPoints:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_build_points_creates_correct_count(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")

        importer = PixelImporter(manager)
        se_data = np.random.randn(64, 128, 128).astype(np.float64)
        dw_data = np.full((128, 128), 0, dtype=np.uint8)
        easting = np.full((128, 128), 500000.0, dtype=np.float64)
        northing = np.full((128, 128), 4000000.0, dtype=np.float64)

        points = importer.build_points(
            se_data, dw_data, easting, northing, 51,
            "E121.4025_N25.1947",
        )
        assert len(points) == 16384  # 128*128

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_build_points_has_correct_fields(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")

        importer = PixelImporter(manager)
        se_data = np.random.randn(64, 128, 128).astype(np.float64)
        dw_data = np.full((128, 128), 3, dtype=np.uint8)
        easting = np.full((128, 128), 500000.0, dtype=np.float64)
        northing = np.full((128, 128), 4000000.0, dtype=np.float64)

        points = importer.build_points(
            se_data, dw_data, easting, northing, 51,
            "E121.4025_N25.1947",
        )

        # 检查第一个 point 的字段
        p0 = points[0]
        # ID 现在是 UUID 格式（36 字符，4 个连字符）
        assert len(p0.id) == 36
        assert p0.id.count("-") == 4
        assert len(p0.vector) == 64
        assert p0.payload["label"] == 3
        assert p0.payload["label_name"] == LABEL_NAMES[3]
        assert p0.payload["utm_easting"] == 500000.0
        assert p0.payload["utm_northing"] == 4000000.0
        assert p0.payload["utm_zone"] == 51
        assert p0.payload["image_id"] == "E121.4025_N25.1947"
        assert p0.payload["pixel_row"] == 0
        assert p0.payload["pixel_col"] == 0

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_build_points_with_nan_utm(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")

        importer = PixelImporter(manager)
        se_data = np.random.randn(64, 128, 128).astype(np.float64)
        dw_data = np.full((128, 128), 0, dtype=np.uint8)
        easting = np.full((128, 128), np.nan, dtype=np.float64)
        northing = np.full((128, 128), np.nan, dtype=np.float64)

        points = importer.build_points(
            se_data, dw_data, easting, northing, None,
            "E121.4025_N25.1947",
        )
        p0 = points[0]
        assert p0.payload["utm_zone"] == -1
        assert np.isnan(p0.payload["utm_easting"])


class TestImportImagePair:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    @patch("KNN_evaluation.importer.PixelDataLoader")
    @patch("KNN_evaluation.importer.compute_utm_grid")
    def test_skips_already_imported(self, mock_utm, mock_loader, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")

        # check_image_count is a @staticmethod; mock its return value
        mock_loader.check_image_count.return_value = 16384

        importer = PixelImporter(manager)
        pair = ImagePair(
            image_id="E121.4025_N25.1947",
            se_path=Path("/fake/se.npy"),
            dw_path=Path("/fake/dw.npy"),
            tif_path=None,
        )

        imported, skipped = importer.import_image_pair(pair)
        assert imported == 0
        assert skipped == 16384


class TestBuildPointsBatching:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_large_batch_splits_into_chunks(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")

        importer = PixelImporter(manager, batch_size=5000)
        se_data = np.random.randn(64, 128, 128).astype(np.float64)
        dw_data = np.full((128, 128), 0, dtype=np.uint8)
        easting = np.full((128, 128), 0.0, dtype=np.float64)
        northing = np.full((128, 128), 0.0, dtype=np.float64)

        points = importer.build_points(
            se_data, dw_data, easting, northing, None,
            "test_image",
        )
        importer._batch_upsert(points)

        # 16384 points with batch_size 5000 → 4 batches
        assert mock_client.upsert.call_count == 4


class TestBuildPointsVectorization:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_point_order_is_row_major(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")

        importer = PixelImporter(manager)
        se_data = np.random.randn(64, 128, 128).astype(np.float64)
        dw_data = np.zeros((128, 128), dtype=np.uint8)
        easting = np.zeros((128, 128), dtype=np.float64)
        northing = np.zeros((128, 128), dtype=np.float64)

        points = importer.build_points(
            se_data, dw_data, easting, northing, 51, "img1",
        )

        assert points[0].payload["pixel_row"] == 0
        assert points[0].payload["pixel_col"] == 0
        assert points[1].payload["pixel_row"] == 0
        assert points[1].payload["pixel_col"] == 1
        assert points[127].payload["pixel_row"] == 0
        assert points[127].payload["pixel_col"] == 127
        assert points[128].payload["pixel_row"] == 1
        assert points[128].payload["pixel_col"] == 0
        assert points[-1].payload["pixel_row"] == 127
        assert points[-1].payload["pixel_col"] == 127

        # 向量与逐像素切片等价（row-major 索引 k = row*128 + col）
        assert points[5].vector == se_data[:, 0, 5].tolist()
        assert points[128 + 7].vector == se_data[:, 1, 7].tolist()

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_id_is_deterministic(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")

        importer = PixelImporter(manager)
        se_data = np.random.randn(64, 128, 128).astype(np.float64)
        dw_data = np.zeros((128, 128), dtype=np.uint8)
        easting = np.zeros((128, 128), dtype=np.float64)
        northing = np.zeros((128, 128), dtype=np.float64)

        ns = uuid.uuid5(uuid.NAMESPACE_DNS, "img1")
        p1 = importer.build_points(se_data, dw_data, easting, northing, 51, "img1")
        p2 = importer.build_points(se_data, dw_data, easting, northing, 51, "img1")

        assert p1[0].id == p2[0].id
        assert p1[0].id == str(uuid.uuid5(ns, "0_0"))
        assert p1[-1].id == str(uuid.uuid5(ns, "127_127"))
