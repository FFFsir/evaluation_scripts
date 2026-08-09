"""Tests for PixelSearcher."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from KNN_evaluation.searcher import PixelSearcher, HitRecord, SearchResult
from KNN_evaluation.qdrant_client import QdrantManager


class TestBuildFilter:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_no_filters_returns_none(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")
        searcher = PixelSearcher(manager)

        result = searcher._build_filter(None, None)
        assert result is None

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_label_filter_only(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")
        searcher = PixelSearcher(manager)

        result = searcher._build_filter(label_filter=[0, 1], utm_range=None)
        assert result is not None
        assert len(result.must) == 1

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_utm_range_filter(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")
        searcher = PixelSearcher(manager)

        utm_range = {"min_e": 500000, "max_e": 501000, "min_n": 4000000, "max_n": 4001000}
        result = searcher._build_filter(label_filter=None, utm_range=utm_range)
        assert result is not None
        assert len(result.must) == 2  # easting + northing

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_combined_filter(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")
        searcher = PixelSearcher(manager)

        result = searcher._build_filter(
            label_filter=[0, 1],
            utm_range={"min_e": 0, "max_e": 100, "min_n": 0, "max_n": 100},
        )
        assert result is not None
        assert len(result.must) == 3  # label + easting + northing


class TestSearch:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_search_returns_search_result(self, mock_client_class):
        mock_client = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "img_0_0"
        mock_hit.score = 0.95
        mock_hit.payload = {
            "label": 0, "label_name": "water",
            "utm_easting": 500000.0, "utm_northing": 4000000.0,
            "utm_zone": 51, "image_id": "img",
            "pixel_row": 0, "pixel_col": 0,
        }
        # New SDK uses query_points which returns QueryResponse with .points
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_client.query_points.return_value = mock_response
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test")
        searcher = PixelSearcher(manager)

        query = np.random.randn(64).astype(np.float64)
        result = searcher.search(query, k=5)

        assert isinstance(result, SearchResult)
        assert len(result.hits) == 1
        assert result.hits[0].label_name == "water"
        assert result.search_mode == "ann"

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_exact_search_mode(self, mock_client_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test")
        searcher = PixelSearcher(manager)

        query = np.random.randn(64).astype(np.float64)
        result = searcher.search(query, k=5, exact=True)

        assert result.search_mode == "exact"
        # 验证搜索参数包含 exact=True
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["search_params"].exact is True
