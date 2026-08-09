"""Tests for coordinate_utils."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from KNN_evaluation.coordinate_utils import compute_utm_grid, compute_utm_grid_from_name, read_geotiff_meta
from KNN_evaluation.data_loader import PixelDataLoader
from KNN_evaluation.config import UTM_RESOLUTION_M


class TestReadGeotiffMeta:
    @patch("KNN_evaluation.coordinate_utils.rasterio")
    def test_returns_crs_transform_zone(self, mock_rio):
        mock_crs = MagicMock()
        mock_crs.__str__ = MagicMock(return_value="EPSG:32651")
        mock_crs.is_epsg_code = False

        mock_src = MagicMock()
        mock_src.crs = mock_crs
        mock_src.transform = MagicMock()
        mock_src.bounds = MagicMock()
        # Correctly mock rasterio.open as a context manager
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        mock_rio.open.return_value = mock_src

        result = read_geotiff_meta(Path("/fake/test.tif"))
        assert result is not None
        assert result["crs"] == "EPSG:32651"
        assert "transform" in result

    @patch("KNN_evaluation.coordinate_utils.rasterio")
    def test_returns_none_on_error(self, mock_rio):
        mock_rio.open.side_effect = Exception("cannot open")
        result = read_geotiff_meta(Path("/fake/missing.tif"))
        assert result is None


class TestComputeUtmGrid:
    def test_none_tif_returns_nan(self):
        easting, northing, zone = compute_utm_grid(None)
        assert easting.shape == (128, 128)
        assert northing.shape == (128, 128)
        assert np.all(np.isnan(easting))
        assert np.all(np.isnan(northing))
        assert zone is None

    @patch("KNN_evaluation.coordinate_utils.read_geotiff_meta")
    def test_produces_128x128_grid(self, mock_read_meta):
        mock_read_meta.return_value = {
            "crs": "EPSG:32651",
            "utm_zone": 51,
            "transform": MagicMock(),
            "bounds": None,
        }
        # pixel 0,0 center → easting = c + a*0.5, northing = f + e*0.5
        mock_read_meta.return_value["transform"].c = 500000.0
        mock_read_meta.return_value["transform"].a = 10.0   # pixel width
        mock_read_meta.return_value["transform"].b = 0.0
        mock_read_meta.return_value["transform"].f = 4000000.0
        mock_read_meta.return_value["transform"].d = 0.0
        mock_read_meta.return_value["transform"].e = -10.0  # pixel height (negative)

        easting, northing, zone = compute_utm_grid(Path("/fake/test.tif"))
        assert easting.shape == (128, 128)
        assert northing.shape == (128, 128)
        assert zone == 51
        # 第一个像素中心点的坐标: c + a*0.5, f + e*0.5
        assert easting[0, 0] == 500005.0
        assert northing[0, 0] == 3999995.0


def test_resolution_constant():
    assert UTM_RESOLUTION_M == 10


def test_parse_location_coord():
    key = "E121.4025_N25.1947"
    lon, lat = PixelDataLoader.parse_location_coord(key)
    assert lon == pytest.approx(121.4025)
    assert lat == pytest.approx(25.1947)


def test_parse_location_coord_invalid():
    with pytest.raises(ValueError):
        PixelDataLoader.parse_location_coord("no_coord_here")


class TestComputeUtmGridFromName:
    def test_grid_shape(self):
        easting, northing, zone = compute_utm_grid_from_name(121.4025, 25.1947)
        assert easting.shape == (128, 128)
        assert northing.shape == (128, 128)
        assert isinstance(zone, int)

    def test_nw_origin_modulo_semantics(self):
        # 验证文件名推算网格的 NW 原点模语义（不依赖 GeoTIFF）：
        # NW 角按 scale 对齐，像素中心落在 scale 网格上产生 +scale/2 偏移
        easting, northing, zone = compute_utm_grid_from_name(121.4025, 25.1947, scale=10)
        # NW 原点应在 scale 网格上
        assert easting[0, 0] % 10 == pytest.approx(5, abs=1e-9)  # +scale/2 偏移
        assert northing[0, 0] % 10 == pytest.approx(5, abs=1e-9)
        assert zone == 51  # 东经121.4 → zone int((121.4+180)/6)+1 = 51

    def test_out_of_utm_range_raises(self):
        # 纬度超出 UTM 有效范围（>84 或 <-80）应提前抛 ValueError，
        # 避免构造无效的 EPSG:32600/32700 后由外层宽 except 掩盖
        with pytest.raises(ValueError, match="超出 UTM 范围"):
            compute_utm_grid_from_name(121.4025, 85.0)
        with pytest.raises(ValueError, match="超出 UTM 范围"):
            compute_utm_grid_from_name(121.4025, -81.0)

    def test_southern_hemisphere_zone(self):
        _, _, zone = compute_utm_grid_from_name(121.4025, -25.1947)
        assert zone == -51  # 南半球用负带号表示（与 read_geotiff_meta 一致）


# demo GeoTIFF 路径（相对仓库根目录）。TIF 缺失时跳过等价性测试，
# 避免测试对数据文件的硬依赖。
DEMO_TIF = (
    Path(__file__).resolve().parents[2]
    / "data_demo" / "DW" / "label_mode_E121.4025_N25.1947_2024-01-01_2024-12-31.tif"
)


@pytest.mark.skipif(not DEMO_TIF.exists(), reason="demo GeoTIFF 缺失，跳过 UTM 等价性测试")
class TestUtmNameVsTifEquivalence:
    """文件名推算 UTM 与 GeoTIFF 精确网格的等价性.

    demo 数据 TIF 原点与 10m 网格对齐，故文件名推算网格（NW 角对齐 scale
    整数倍）应与 TIF 逐像素坐标完全一致（max abs diff == 0）。当 TIF 原点
    未对齐 10m 网格时，两者最多偏差一个像素（10m），此为有意取舍
    （用户要求不加载 TIF，见 design.md D7）。
    """

    def test_max_abs_diff_is_zero(self):
        from KNN_evaluation.data_loader import PixelDataLoader

        lon, lat = PixelDataLoader.parse_location_coord("E121.4025_N25.1947")
        e_name, n_name, z_name = compute_utm_grid_from_name(lon, lat)
        e_tif, n_tif, z_tif = compute_utm_grid(DEMO_TIF)

        assert np.max(np.abs(e_name - e_tif)) == 0.0
        assert np.max(np.abs(n_name - n_tif)) == 0.0
        assert z_name == z_tif
