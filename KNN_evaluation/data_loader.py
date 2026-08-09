"""数据加载模块：SE/DW 文件读取、文件配对与坐标段提取."""
import re
import warnings
from pathlib import Path
from typing import NamedTuple
import numpy as np
from src.satellite_embedding_loader import load_embedding


class ImagePair(NamedTuple):
    """一对匹配的 SE + DW 文件（可选 GeoTIFF）."""
    image_id: str        # 坐标段，如 "E121.4025_N25.1947"
    se_path: Path
    dw_path: Path
    tif_path: Path | None


class PixelDataLoader:
    """卫星影像数据加载器：扫描、配对、加载 SE/DW 数据."""

    # 文件名中 E{lon}_N{lat} 坐标段的正则
    COORDINATE_PATTERN = re.compile(r"(E\d+\.\d+_N\d+\.\d+)")
    # 坐标段内经纬度数字的显式捕获（带符号，供 parse_location_coord 使用）
    LON_LAT_PATTERN = re.compile(r"E(?P<lon>-?\d+\.\d+)_N(?P<lat>-?\d+\.\d+)")

    @staticmethod
    def extract_location_key(filename: str) -> str | None:
        """从文件名中提取坐标段作为配对 key.

        Args:
            filename: 文件名（不含路径）.

        Returns:
            坐标段字符串（如 "E121.4025_N25.1947"），匹配失败返回 None.
        """
        match = PixelDataLoader.COORDINATE_PATTERN.search(filename)
        if match is None:
            return None
        return match.group(1)

    @staticmethod
    def parse_location_coord(location_key: str) -> tuple[float, float]:
        """从坐标段（如 E121.4025_N25.1947）解析经纬度.

        Args:
            location_key: extract_location_key 返回的坐标段字符串.

        Returns:
            (lon, lat) 浮点经纬度.

        Raises:
            ValueError: 坐标段格式不合法.
        """
        m = PixelDataLoader.LON_LAT_PATTERN.search(location_key)
        if m is None:
            raise ValueError(f"无法从坐标段解析经纬度: {location_key}")
        # 用命名捕获组显式提取 lon/lat 数字，避免对匹配串做切片处理
        return float(m.group("lon")), float(m.group("lat"))

    @staticmethod
    def normalize_location_key(raw_key: str) -> str:
        """将坐标段字符串归一化为 round 4 位 + 去尾随零的形式.

        双集合（google 混合精度 / xian 全 4 位）同名坐标的坐标段字符串可能不同
        （如 "E121.4033_N25.1370" vs "E121.4033_N25.137"），统一归一化后得到一致
        的 image_id，保证 point_id 双集合对齐.

        Args:
            raw_key: 原始坐标段字符串，如 "E121.4033_N25.1370".

        Returns:
            归一化坐标段字符串，如 "E121.4033_N25.137"；小数全为 0 时保留至少
            1 位小数（如 "E121.0_N25.0"），保证坐标段合法（E\\d+\\.\\d+_N\\d+\\.\\d+ 可匹配）.

        Raises:
            ValueError: 坐标段格式不合法.
        """
        lon, lat = PixelDataLoader.parse_location_coord(raw_key)

        def _trim(value: float) -> str:
            text = format(round(value, 4), ".4f").rstrip("0").rstrip(".")
            if "." not in text:
                text += ".0"
            return text

        return f"E{_trim(lon)}_N{_trim(lat)}"

    @staticmethod
    def load_se(se_path: Path) -> np.ndarray:
        """加载 SE embedding 文件，返回 (64, 128, 128) float64 数组.

        Raises:
            ValueError: 维度不是 64 时抛出.
        """
        data = load_embedding(se_path)
        if data.shape[0] != 64:
            raise ValueError(
                f"期望 SE 数据维度为 64，实际: {data.shape[0]}，文件: {se_path}"
            )
        return data

    @staticmethod
    def load_dw(dw_path: Path) -> np.ndarray:
        """加载 DW label 文件，返回 (128, 128) uint8 标签矩阵.

        支持结构化 dtype [('label', 'u1')] 和普通 uint8 数组.
        """
        suffix = dw_path.suffix.lower()
        if suffix not in (".npy"):
            raise ValueError(f"不支持的文件格式: {suffix}，DW 仅支持 .npy")

        raw = np.load(dw_path)

        # 结构化 dtype: 提取 'label' 字段
        if raw.dtype.names is not None:
            if "label" not in raw.dtype.names:
                raise KeyError(
                    f"DW 文件中缺少 'label' 字段，可用字段: {list(raw.dtype.names)}"
                )
            label_array = raw["label"].astype(np.uint8)
        else:
            label_array = raw.astype(np.uint8)

        if label_array.shape != (128, 128):
            raise ValueError(
                f"期望 DW 数据 shape 为 (128, 128)，实际: {label_array.shape}，文件: {dw_path}"
            )

        return label_array

    @staticmethod
    def scan_directory(data_dir: Path) -> list[ImagePair]:
        """扫描目录，匹配 SE/DW/GeoTIFF 文件对.

        Args:
            data_dir: 包含 SE/ 和 DW/ 子目录的数据根目录.

        Returns:
            按 image_id 排序的 ImagePair 列表.
        """
        se_dir = data_dir / "SE"
        dw_dir = data_dir / "DW"

        if not se_dir.exists() or not dw_dir.exists():
            warnings.warn(f"缺少 SE/ 或 DW/ 子目录: {data_dir}")
            return []

        # 扫描 SE 文件：数值坐标段作为配对 key，另存原始坐标段字符串供 image_id
        se_numeric: dict[tuple[float, float], Path] = {}
        se_raw: dict[tuple[float, float], str] = {}
        for fpath in list(se_dir.glob("*.npy")) + list(se_dir.glob("*.npz")):
            raw_key = PixelDataLoader.extract_location_key(fpath.name)
            if raw_key is not None:
                num_key = PixelDataLoader.parse_location_coord(raw_key)
                se_numeric[num_key] = fpath
                se_raw[num_key] = raw_key

        # 扫描 DW 文件
        dw_numeric: dict[tuple[float, float], Path] = {}
        for fpath in list(dw_dir.glob("*.npy")):
            raw_key = PixelDataLoader.extract_location_key(fpath.name)
            if raw_key is not None:
                num_key = PixelDataLoader.parse_location_coord(raw_key)
                dw_numeric[num_key] = fpath

        # 扫描 GeoTIFF 文件
        tif_numeric: dict[tuple[float, float], Path] = {}
        for fpath in list(se_dir.glob("*.tif")) + list(dw_dir.glob("*.tif")):
            raw_key = PixelDataLoader.extract_location_key(fpath.name)
            if raw_key is not None:
                num_key = PixelDataLoader.parse_location_coord(raw_key)
                tif_numeric[num_key] = fpath

        # 配对：按解析后的数值 (lon, lat) 取交集，image_id 用 SE 侧坐标段归一化字符串
        pairs: list[ImagePair] = []
        common_keys = set(se_numeric.keys()) & set(dw_numeric.keys())
        for key in sorted(common_keys):
            pairs.append(ImagePair(
                image_id=PixelDataLoader.normalize_location_key(se_raw[key]),
                se_path=se_numeric[key],
                dw_path=dw_numeric[key],
                tif_path=tif_numeric.get(key),
            ))

        return pairs

    @staticmethod
    def check_image_count(image_id: str, manager) -> int:
        """检查指定 image_id 在 Qdrant 中已导入的 point 数量.

        Args:
            image_id: 影像标识.
            manager: QdrantManager 实例.

        Returns:
            已导入的 point 数（0 或 16384 表示完整导入）.
        """
        from qdrant_client import models
        count_result = manager.client.count(
            collection_name=manager.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="image_id",
                        match=models.MatchValue(value=image_id),
                    ),
                ],
            ),
            exact=True,
        )
        return count_result.count
