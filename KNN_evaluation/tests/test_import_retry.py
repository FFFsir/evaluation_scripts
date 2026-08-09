"""Tests for import retry with exponential backoff (Qdrant upsert/count).

覆盖：
- `_retry_call` 指数退避、参数透传、重试上限
- 仅瞬时失败（网络/超时、429 服务端繁忙、5xx）重试；4xx 持久错误不重试
- `_batch_upsert` 的 client.upsert 接入重试
- `import_directory` 的 check_image_count（断点续传 count）接入重试
"""
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http import exceptions as qexc

from KNN_evaluation.importer import PixelImporter, _retry_call
from KNN_evaluation.data_loader import ImagePair
from KNN_evaluation.qdrant_client import QdrantManager


def _make_pair(image_id="E121.4025_N25.1947") -> ImagePair:
    return ImagePair(
        image_id=image_id,
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


def _build_points(importer: PixelImporter, image_id: str = "img_retry"):
    se_data = np.zeros((64, 128, 128), dtype=np.float64)
    dw_data = np.zeros((128, 128), dtype=np.uint8)
    easting = np.zeros((128, 128), dtype=np.float64)
    northing = np.zeros((128, 128), dtype=np.float64)
    return importer.build_points(se_data, dw_data, easting, northing, 51, image_id)


# ---------- _retry_call ----------

class TestRetryCall:
    def test_retries_transient_failure_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise qexc.ResponseHandlingException(ConnectionError("boom"))
            return "ok"

        assert _retry_call(flaky, retries=3, base_delay=0) == "ok"
        assert calls["n"] == 2

    def test_raises_after_retries_exhausted(self):
        calls = {"n": 0}

        def always_fail():
            calls["n"] += 1
            raise qexc.ResponseHandlingException(ConnectionError("boom"))

        with pytest.raises(qexc.ResponseHandlingException):
            _retry_call(always_fail, retries=3, base_delay=0)
        # retries=3 → 首次调用 + 3 次重试 = 4 次尝试
        assert calls["n"] == 4

    @patch("KNN_evaluation.importer.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        calls = {"n": 0}

        def always_fail():
            calls["n"] += 1
            raise qexc.ResponseHandlingException(ConnectionError("boom"))

        with pytest.raises(qexc.ResponseHandlingException):
            _retry_call(always_fail, retries=3, base_delay=1.0)

        # 每次失败后按 1s→2s→4s 指数退避（共 3 次 sleep），随后抛出
        assert mock_sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]
        assert calls["n"] == 4

    def test_passes_args_and_kwargs(self):
        seen = {}

        def target(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return "done"

        result = _retry_call(target, 1, 2, key="val", retries=3, base_delay=0)
        assert result == "done"
        assert seen == {"args": (1, 2), "kwargs": {"key": "val"}}

    def test_does_not_retry_persistent_4xx(self):
        calls = {"n": 0}

        def persistent():
            calls["n"] += 1
            raise qexc.UnexpectedResponse(400, "Bad Request", b"", {})

        with pytest.raises(qexc.UnexpectedResponse):
            _retry_call(persistent, retries=3, base_delay=0)
        assert calls["n"] == 1

    def test_retries_429_rate_limit(self):
        calls = {"n": 0}

        def busy():
            calls["n"] += 1
            if calls["n"] == 1:
                raise ResourceExhaustedResponse("busy", retry_after_s=1)
            return "ok"

        assert _retry_call(busy, retries=3, base_delay=0) == "ok"
        assert calls["n"] == 2

    def test_retries_server_5xx(self):
        calls = {"n": 0}

        def server_error():
            calls["n"] += 1
            if calls["n"] == 1:
                raise qexc.UnexpectedResponse(500, "Internal Server Error", b"", {})
            return "ok"

        assert _retry_call(server_error, retries=3, base_delay=0) == "ok"
        assert calls["n"] == 2


# ---------- _batch_upsert ----------

class TestBatchUpsertRetry:
    @patch("KNN_evaluation.importer.time.sleep")
    def test_upsert_retries_transient_failure_then_succeeds(self, mock_sleep):
        manager = _make_manager()
        manager.client.upsert.side_effect = [
            qexc.ResponseHandlingException(ConnectionError("boom")),
            qexc.ResponseHandlingException(ConnectionError("boom")),
            None,
        ]
        importer = PixelImporter(manager, batch_size=16384)
        importer._batch_upsert(_build_points(importer))
        assert manager.client.upsert.call_count == 3

    @patch("KNN_evaluation.importer.time.sleep")
    def test_upsert_raises_after_retries_exhausted(self, mock_sleep):
        manager = _make_manager()
        manager.client.upsert.side_effect = qexc.ResponseHandlingException(
            ConnectionError("boom")
        )
        importer = PixelImporter(manager, batch_size=16384)
        with pytest.raises(qexc.ResponseHandlingException):
            importer._batch_upsert(_build_points(importer))
        # retries=3 → 首次调用 + 3 次重试 = 4 次 upsert 尝试
        assert manager.client.upsert.call_count == 4

    @patch("KNN_evaluation.importer.time.sleep")
    def test_upsert_does_not_retry_persistent_error(self, mock_sleep):
        manager = _make_manager()
        manager.client.upsert.side_effect = qexc.UnexpectedResponse(400, "Bad Request", b"", {})
        importer = PixelImporter(manager, batch_size=16384)
        with pytest.raises(qexc.UnexpectedResponse):
            importer._batch_upsert(_build_points(importer))
        assert manager.client.upsert.call_count == 1


# ---------- import_directory count 路径 ----------

class TestCountRetry:
    @patch("KNN_evaluation.importer.time.sleep")
    @patch("KNN_evaluation.importer.PixelDataLoader")
    @patch("KNN_evaluation.importer.compute_utm_grid_from_name")
    def test_count_retries_transient_failure_then_skips(self, mock_utm, mock_loader, mock_sleep):
        mock_loader.scan_directory.return_value = [_make_pair()]
        # 瞬时失败 2 次后恢复；返回 16384 表示影像已完整导入 → 走跳过分支
        mock_loader.check_image_count.side_effect = [
            qexc.ResponseHandlingException(ConnectionError("boom")),
            qexc.ResponseHandlingException(ConnectionError("boom")),
            16384,
        ]
        mock_loader.load_dw.return_value = np.zeros((128, 128), dtype=np.uint8)
        mock_utm.return_value = (
            np.zeros((128, 128), dtype=np.float64),
            np.zeros((128, 128), dtype=np.float64),
            51,
        )
        manager = _make_manager()
        importer = PixelImporter(manager, batch_size=16384)

        stats = importer.import_directory(Path("/fake/data"))

        # 1 次逻辑查询 + 2 次重试 = 3 次调用
        assert mock_loader.check_image_count.call_count == 3
        assert stats["skipped_images"] == 1
        assert stats["imported_images"] == 0

    @patch("KNN_evaluation.importer.time.sleep")
    @patch("KNN_evaluation.importer.PixelDataLoader")
    @patch("KNN_evaluation.importer.compute_utm_grid_from_name")
    def test_count_exhausts_retries_then_raises(self, mock_utm, mock_loader, mock_sleep):
        mock_loader.scan_directory.return_value = [_make_pair()]
        mock_loader.check_image_count.side_effect = qexc.ResponseHandlingException(
            ConnectionError("boom")
        )
        mock_utm.return_value = (
            np.zeros((128, 128), dtype=np.float64),
            np.zeros((128, 128), dtype=np.float64),
            51,
        )
        manager = _make_manager()
        importer = PixelImporter(manager, batch_size=16384)

        with pytest.raises(qexc.ResponseHandlingException):
            importer.import_directory(Path("/fake/data"))

        # retries=3 → 1 次逻辑查询 + 3 次重试 = 4 次调用
        assert mock_loader.check_image_count.call_count == 4

    @patch("KNN_evaluation.importer.time.sleep")
    @patch("KNN_evaluation.importer.PixelDataLoader")
    @patch("KNN_evaluation.importer.compute_utm_grid_from_name")
    def test_count_precompute_retries_with_progress_callback(self, mock_utm, mock_loader, mock_sleep):
        mock_loader.scan_directory.return_value = [_make_pair()]
        # 预计算 count：瞬时失败 2 次后返回 0（未导入）；导入路径内 line129 再次 count → 0
        mock_loader.check_image_count.side_effect = [
            qexc.ResponseHandlingException(ConnectionError("boom")),
            qexc.ResponseHandlingException(ConnectionError("boom")),
            0,
            0,
        ]
        mock_loader.load_se.return_value = np.zeros((64, 128, 128), dtype=np.float64)
        mock_loader.load_dw.return_value = np.zeros((128, 128), dtype=np.uint8)
        mock_utm.return_value = (
            np.zeros((128, 128), dtype=np.float64),
            np.zeros((128, 128), dtype=np.float64),
            51,
        )
        manager = _make_manager()
        importer = PixelImporter(manager, batch_size=16384)
        progress = []

        stats = importer.import_directory(
            Path("/fake/data"),
            progress_callback=lambda a, b: progress.append((a, b)),
        )

        # 预计算 count（1 次逻辑 + 2 次重试 = 3 次）+ import_image_pair 内 count 1 次 = 4 次
        assert mock_loader.check_image_count.call_count == 4
        assert stats["imported_images"] == 1
        assert stats["total_pixels"] == 128 * 128
        assert manager.client.upsert.call_count == 1
