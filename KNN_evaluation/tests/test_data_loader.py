"""Tests for PixelDataLoader."""
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from KNN_evaluation.data_loader import PixelDataLoader, ImagePair


class TestExtractLocationKey:
    def test_extracts_coordinate_segment_from_se_filename(self):
        key = PixelDataLoader.extract_location_key(
            "all_mean_E121.4025_N25.1947_2024.npy"
        )
        assert key == "E121.4025_N25.1947"

    def test_extracts_coordinate_segment_from_dw_filename(self):
        key = PixelDataLoader.extract_location_key(
            "label_mode_E121.4025_N25.1947_2024-01-01_2024-12-31.npy"
        )
        assert key == "E121.4025_N25.1947"

    def test_returns_none_for_unmatched_filename(self):
        key = PixelDataLoader.extract_location_key("random_file.txt")
        assert key is None


class TestLoadSE:
    @patch("KNN_evaluation.data_loader.load_embedding")
    def test_load_se_calls_load_embedding(self, mock_load):
        mock_load.return_value = np.zeros((64, 128, 128), dtype=np.float64)
        result = PixelDataLoader.load_se(Path("/fake/se.npy"))
        assert result.shape == (64, 128, 128)
        assert result.dtype == np.float64
        mock_load.assert_called_once()

    @patch("KNN_evaluation.data_loader.load_embedding")
    def test_load_se_raises_on_wrong_shape(self, mock_load):
        mock_load.return_value = np.zeros((32, 128, 128), dtype=np.float64)
        with pytest.raises(ValueError, match="期望 SE 数据维度为 64"):
            PixelDataLoader.load_se(Path("/fake/se.npy"))


class TestLoadDW:
    def test_load_dw_structured_dtype(self):
        import tempfile
        raw = np.zeros((128, 128), dtype=[("label", "u1")])
        raw["label"] = 5
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, raw)
            path = Path(f.name)
        try:
            result = PixelDataLoader.load_dw(path)
            assert result.shape == (128, 128)
            assert result.dtype == np.uint8
            assert np.all(result == 5)
        finally:
            path.unlink()

    def test_load_dw_plain_array(self):
        import tempfile
        raw = np.full((128, 128), 3, dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, raw)
            path = Path(f.name)
        try:
            result = PixelDataLoader.load_dw(path)
            assert result.shape == (128, 128)
            assert result.dtype == np.uint8
            assert np.all(result == 3)
        finally:
            path.unlink()

    def test_load_dw_wrong_shape_raises(self):
        import tempfile
        raw = np.zeros((64, 64), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, raw)
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="期望 DW 数据 shape 为"):
                PixelDataLoader.load_dw(path)
        finally:
            path.unlink()

    def test_load_dw_unsupported_suffix(self, tmp_path):
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not numpy")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            PixelDataLoader.load_dw(bad_file)


class TestScanDirectory:
    def test_matches_se_dw_pairs(self, tmp_path):
        se_dir = tmp_path / "SE"
        dw_dir = tmp_path / "DW"
        se_dir.mkdir()
        dw_dir.mkdir()

        # 创建一组匹配的 SE/DW 文件
        (se_dir / "all_mean_E121.4025_N25.1947_2024.npy").write_text("")
        (dw_dir / "label_mode_E121.4025_N25.1947_2024-01-01_2024-12-31.npy").write_text("")

        pairs = PixelDataLoader.scan_directory(tmp_path)

        assert len(pairs) == 1
        assert pairs[0].image_id == "E121.4025_N25.1947"

    def test_skips_orphan_se_without_dw(self, tmp_path):
        se_dir = tmp_path / "SE"
        dw_dir = tmp_path / "DW"
        se_dir.mkdir()
        dw_dir.mkdir()

        (se_dir / "all_mean_E121.4025_N25.1947_2024.npy").write_text("")
        # 不创建对应的 DW 文件

        pairs = PixelDataLoader.scan_directory(tmp_path)
        assert len(pairs) == 0

    def test_missing_subdirs_warns(self, tmp_path):
        # 没有 SE/ 或 DW/ 子目录
        pairs = PixelDataLoader.scan_directory(tmp_path)
        assert pairs == []


class TestNormalizeLocationKey:
    """normalize_location_key：round 4 位小数 + 去尾随零的坐标段归一化."""

    def test_strips_trailing_zero_in_lat(self):
        assert (
            PixelDataLoader.normalize_location_key("E121.4033_N25.1370")
            == "E121.4033_N25.137"
        )

    def test_strips_trailing_zero_in_lon(self):
        assert (
            PixelDataLoader.normalize_location_key("E121.4030_N25.1601")
            == "E121.403_N25.1601"
        )

    def test_no_trailing_zero_unchanged(self):
        assert (
            PixelDataLoader.normalize_location_key("E121.4025_N25.1947")
            == "E121.4025_N25.1947"
        )

    def test_all_zero_fraction_keeps_one_decimal(self):
        # 小数全为 0 时保留至少 1 位，保持坐标段合法（E\d+\.\d+_N\d+\.\d+ 可匹配）
        assert (
            PixelDataLoader.normalize_location_key("E121.0000_N25.0000")
            == "E121.0_N25.0"
        )

    def test_rounds_to_4_decimal_places(self):
        # 超过 4 位小数时 round 到 4 位
        assert (
            PixelDataLoader.normalize_location_key("E121.40339_N25.13709")
            == "E121.4034_N25.1371"
        )

    def test_numeric_equivalents_unify_to_same_string(self):
        # 数值相同但字符串不同的坐标段应归一化为同一串（双集合 image_id 一致的关键）
        assert PixelDataLoader.normalize_location_key(
            "E121.4033_N25.1370"
        ) == PixelDataLoader.normalize_location_key("E121.4033_N25.137")


class TestScanDirectoryNumericMatch:
    """坐标段数值匹配配对：SE/DW 精度不一致时按解析后的数值 (lon, lat) 配对."""

    @staticmethod
    def _make_dirs(tmp_path):
        se_dir = tmp_path / "SE"
        dw_dir = tmp_path / "DW"
        se_dir.mkdir()
        dw_dir.mkdir()
        return se_dir, dw_dir

    def test_se_4digit_dw_3digit_pairs_numerically(self, tmp_path):
        se_dir, dw_dir = self._make_dirs(tmp_path)
        # SE 坐标段 4 位小数、DW 坐标段 3 位小数，数值相等：应配对成功
        (se_dir / "all_mean_E121.4033_N25.1370_2024.npy").write_text("")
        (dw_dir / "label_mode_E121.4033_N25.137_2024-01-01_2024-12-31.npy").write_text("")

        pairs = PixelDataLoader.scan_directory(tmp_path)

        assert len(pairs) == 1
        # image_id 取 SE 文件坐标段归一化字符串（去尾随零）
        assert pairs[0].image_id == "E121.4033_N25.137"

    def test_multi_precision_files_match_numerically(self, tmp_path):
        se_dir, dw_dir = self._make_dirs(tmp_path)
        # 多个文件：SE 4 位 / DW 3 位 与 SE 4 位 / DW 3 位（不同坐标），应全部配对
        (se_dir / "all_mean_E121.4033_N25.1370_2024.npy").write_text("")
        (dw_dir / "label_mode_E121.4033_N25.137_2024-01-01.npy").write_text("")
        (se_dir / "all_mean_E116.3970_N39.9040_2024.npy").write_text("")
        (dw_dir / "label_mode_E116.397_N39.904_2024-01-01.npy").write_text("")

        pairs = PixelDataLoader.scan_directory(tmp_path)

        assert len(pairs) == 2
        assert {p.image_id for p in pairs} == {
            "E121.4033_N25.137",
            "E116.397_N39.904",
        }

    def test_orphan_se_skipped_in_numeric_mode(self, tmp_path):
        se_dir, dw_dir = self._make_dirs(tmp_path)
        # 孤儿 SE：DW 侧无对应坐标文件，应跳过
        (se_dir / "all_mean_E121.4033_N25.1370_2024.npy").write_text("")

        pairs = PixelDataLoader.scan_directory(tmp_path)

        assert len(pairs) == 0
