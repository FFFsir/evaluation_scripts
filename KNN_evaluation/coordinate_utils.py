"""UTM 坐标计算：从 GeoTIFF Affine Transform 或文件名经纬度推算逐像素坐标."""
import warnings
from pathlib import Path
import numpy as np

from KNN_evaluation.config import UTM_RESOLUTION_M

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    rasterio = None  # type: ignore
    HAS_RASTERIO = False


def read_geotiff_meta(tif_path: Path) -> dict | None:
    """读取 GeoTIFF 文件的元数据.

    Returns:
        含 crs、transform、utm_zone、bounds 的字典，读取失败返回 None.
    """
    if not HAS_RASTERIO:
        warnings.warn("rasterio 未安装，无法读取 GeoTIFF 元数据")
        return None

    try:
        with rasterio.open(tif_path) as src:
            crs = str(src.crs)
            utm_zone = None
            if src.crs and src.crs.is_epsg_code:
                code = src.crs.to_epsg()
                if code and 32601 <= code <= 32660:
                    utm_zone = code - 32600
                elif code and 32701 <= code <= 32760:
                    utm_zone = -(code - 32700)

            return {
                "crs": crs,
                "transform": src.transform,
                "utm_zone": utm_zone,
                "bounds": src.bounds,
            }
    except Exception as e:
        warnings.warn(f"读取 GeoTIFF 失败: {tif_path}，{e}")
        return None


def compute_utm_grid(tif_path: Path | None) -> tuple[np.ndarray, np.ndarray, int | None]:
    """根据 GeoTIFF Affine transform 计算每个像素的 UTM 坐标.

    Args:
        tif_path: GeoTIFF 文件路径，None 时返回全 NaN.

    Returns:
        (easting_grid, northing_grid, utm_zone)
        - easting_grid: (128, 128) float64，UTM 东向坐标
        - northing_grid: (128, 128) float64，UTM 北向坐标
        - utm_zone: int 或 None
    """
    if tif_path is None:
        return (
            np.full((128, 128), np.nan, dtype=np.float64),
            np.full((128, 128), np.nan, dtype=np.float64),
            None,
        )

    meta = read_geotiff_meta(tif_path)
    if meta is None:
        return (
            np.full((128, 128), np.nan, dtype=np.float64),
            np.full((128, 128), np.nan, dtype=np.float64),
            None,
        )

    transform = meta["transform"]
    utm_zone = meta.get("utm_zone")

    # 使用 Affine transform 计算每个像素中心点的坐标
    # x = c + a*col + b*row (easting)
    # y = f + d*col + e*row (northing)
    rows = np.arange(128)
    cols = np.arange(128)
    col_grid, row_grid = np.meshgrid(cols, rows)

    easting = transform.c + transform.a * (col_grid + 0.5) + transform.b * (row_grid + 0.5)
    northing = transform.f + transform.d * (col_grid + 0.5) + transform.e * (row_grid + 0.5)

    return easting.astype(np.float64), northing.astype(np.float64), utm_zone


def compute_utm_grid_from_name(
    lon: float,
    lat: float,
    scale: int = UTM_RESOLUTION_M,
    grid_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """从文件名经纬度坐标推算 (grid_size, grid_size) UTM 坐标网格.

    坐标模型与 DW 下载脚本 _create_square_roi 一致：
    文件名坐标段 E{lon}_N{lat} 为影像中心点；NW 角对齐 scale 整数倍向东南展开。

    Args:
        lon: 中心点经度.
        lat: 中心点纬度.
        scale: 像素分辨率（米/像素）.
        grid_size: 网格边长像素数（默认 128）.

    Returns:
        (easting, northing, utm_zone)
        - easting: (grid_size, grid_size) float64
        - northing: (grid_size, grid_size) float64
        - utm_zone: int（北半球正、南半球负）
    """
    from pyproj import Transformer

    zone = _lonlat_to_utm_zone(lon, lat)
    if zone == 0:
        # 超出 UTM 有效纬度范围（lat > 84 或 lat < -80），EPSG:32600/32700 不存在，
        # 提前抛错避免构造无效 CRS 后由外层宽 except 掩盖
        raise ValueError(f"经纬度超出 UTM 范围: lon={lon}, lat={lat}（UTM 有效纬度 -80~84）")
    epsg = 32600 + abs(zone) if zone >= 0 else 32700 + abs(zone)
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = trans.transform(lon, lat)

    half = (grid_size * scale) / 2.0
    nw_x = float(np.floor((cx - half) / scale) * scale)
    nw_y = float(np.ceil((cy + half) / scale) * scale)

    rows = np.arange(grid_size)
    cols = np.arange(grid_size)
    col_grid, row_grid = np.meshgrid(cols, rows)
    easting = nw_x + col_grid * scale + scale / 2.0
    northing = nw_y - row_grid * scale - scale / 2.0
    return easting.astype(np.float64), northing.astype(np.float64), zone


def _lonlat_to_utm_zone(lon: float, lat: float) -> int:
    """经纬度 → UTM 带号（与 DW 下载脚本 _get_utm_epsg 一致）."""
    if lat > 84 or lat < -80:
        return 0  # 超出 UTM 范围
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(60, zone))
    return zone if lat >= 0 else -zone
