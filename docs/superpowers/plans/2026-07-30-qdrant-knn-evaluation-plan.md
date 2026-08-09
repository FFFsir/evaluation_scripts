---
archived-with: 2026-07-30-qdrant-knn-evaluation
status: final
---
# Qdrant KNN 评估系统 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 构建一个基于 Qdrant 向量数据库的卫星遥感像素级 KNN 评估系统，支持 64 维 embedding 向量的批量导入、标量过滤联合检索和 CLI 命令行操作。

**Architecture:** 在现有 `KNN_evaluation/` 目录下创建模块化 Python 包，通过 `QdrantManager` 管理 Qdrant 连接与 Collection 生命周期，通过 `PixelDataLoader` 复用已有的 `src.satellite_embedding_loader` 加载 SE/DW 数据，通过 `PixelImporter` 编排逐文件/逐像素的批量 upsert 导入管线，通过 `PixelSearcher` 封装向量检索与标量过滤的组合查询，最后通过 `cli.py` 暴露出 `import`/`search`/`stats` 三个子命令。

**Tech Stack:** Python 3.12+, Qdrant (Docker), qdrant-client>=1.12, numpy>=1.26, tqdm>=4.66, matplotlib>=3.8, pyproj>=3.6, rasterio>=1.3

## Global Constraints

- Python >= 3.12.12（与项目 `pyproject.toml` 一致）
- 所有新增依赖追加到 `pyproject.toml` 的 `dependencies` 列表中
- 单元测试使用 mock Qdrant client，不依赖外部服务；集成测试在 Docker Qdrant 环境执行
- 错误信息使用中文，与现有代码 `src/satellite_embedding_loader.py` 保持一致
- 导入管线支持断点续传：以 `image_id` 为粒度，已完整导入的影像跳过
- 向量维度固定为 64，point ID 格式为 `{image_id}_{row}_{col}`
- GeoTIFF 缺失时不阻塞导入，UTM 字段填 NaN 并输出警告

---
---

### Task 1: 项目基础设施与依赖配置

**Files:**
- Modify: `pyproject.toml`
- Create: `KNN_evaluation/__init__.py`
- Create: `KNN_evaluation/config.py`
- Create: `KNN_evaluation/label_mapping.py`
- Create: `KNN_evaluation/tests/__init__.py`

**Interfaces:**
- Consumes: 现有的 `pyproject.toml`、`src/satellite_embedding_loader.py`
- Produces:
  - `KNN_evaluation.config.QDRANT_URL: str = "http://localhost:6333"`
  - `KNN_evaluation.config.COLLECTION_NAME: str = "pixel_embeddings"`
  - `KNN_evaluation.config.BATCH_SIZE: int = 10000`
  - `KNN_evaluation.config.VECTOR_SIZE: int = 64`
  - `KNN_evaluation.label_mapping.LABEL_NAMES: dict[int, str]` — 0-8 到名称的映射
  - `KNN_evaluation.label_mapping.LABEL_IDS: dict[str, int]` — 名称到 0-8 的反向映射

- [x] **Step 1: 在 `pyproject.toml` 中添加新依赖**

在 `dependencies` 列表末尾追加 `qdrant-client`、`tqdm`、`matplotlib`、`pyproj`：

```toml
dependencies = [
    "nicegui>=3.15.0",
    "numpy>=2.5.1",
    "pillow>=12.3.0",
    "rasterio>=1.5.0",
    "qdrant-client>=1.12",
    "tqdm>=4.66",
    "matplotlib>=3.8",
    "pyproj>=3.6",
]
```

- [x] **Step 2: 安装依赖**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && pip install -e ".[dev]"
```

- [x] **Step 3: 创建 `KNN_evaluation/__init__.py`**

```python
"""Qdrant KNN 评估系统 — 像素级 embedding 向量数据库检索与评估."""
```

- [x] **Step 4: 创建 `KNN_evaluation/tests/__init__.py`**

```python
"""Tests for KNN_evaluation package."""
```

- [x] **Step 5: 创建 `KNN_evaluation/config.py`**

```python
"""Qdrant KNN 评估系统配置."""

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pixel_embeddings"
BATCH_SIZE = 10000
VECTOR_SIZE = 64
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
EF_SEARCH_DEFAULT = 64
QDRANT_TIMEOUT = 5
```

- [x] **Step 6: 创建 `KNN_evaluation/label_mapping.py`**

```python
"""土地覆盖标签映射表 (0-8 <-> 名称)."""

LABEL_NAMES: dict[int, str] = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}

LABEL_IDS: dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}
```

- [x] **Step 7: 验证导入**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -c "
import KNN_evaluation
from KNN_evaluation.config import QDRANT_URL, COLLECTION_NAME, BATCH_SIZE, VECTOR_SIZE
from KNN_evaluation.label_mapping import LABEL_NAMES, LABEL_IDS
print('config:', QDRANT_URL, COLLECTION_NAME, BATCH_SIZE, VECTOR_SIZE)
print('labels:', LABEL_NAMES)
print('reverse:', LABEL_IDS)
"
```

- [x] **Step 8: Commit**

```bash
git add pyproject.toml KNN_evaluation/__init__.py KNN_evaluation/tests/__init__.py KNN_evaluation/config.py KNN_evaluation/label_mapping.py
git commit -m "feat: add KNN_evaluation package skeleton, config, and label mapping"
```

---

### Task 2: QdrantManager — 连接管理与 Collection 创建

**Files:**
- Create: `KNN_evaluation/qdrant_client.py`
- Create: `KNN_evaluation/tests/test_qdrant_client.py`

**Interfaces:**
- Consumes: `KNN_evaluation.config`（所有常量）、`KNN_evaluation.label_mapping`（不直接使用，但 `create_collection` 文档中提及）
- Produces:
  - `QdrantManager.__init__(url, collection_name, timeout)` — 创建连接，不立即验证
  - `QdrantManager.health_check() -> bool` — 检查 Qdrant 是否可达，超时 5s
  - `QdrantManager.collection_exists() -> bool` — 检查 Collection 是否存在
  - `QdrantManager.create_collection(vector_size, m, ef_construct)` — 创建 Collection + HNSW 配置
  - `QdrantManager.create_payload_indices()` — 为 label、label_name、utm_easting、utm_northing、image_id 创建标量索引
  - `QdrantManager.get_imported_image_ids() -> set[str]` — 查询已导入的 image_id 集合
  - `QdrantManager.collection_info() -> dict` — 返回 collection 统计信息

- [x] **Step 1: 编写失败的测试**

创建 `KNN_evaluation/tests/test_qdrant_client.py`：

```python
"""Tests for QdrantManager."""
import pytest
from unittest.mock import MagicMock, patch
from KNN_evaluation.qdrant_client import QdrantManager


class TestQdrantManagerInit:
    def test_init_stores_parameters(self):
        manager = QdrantManager(
            url="http://localhost:6333",
            collection_name="test_collection",
            timeout=5,
        )
        assert manager.url == "http://localhost:6333"
        assert manager.collection_name == "test_collection"
        assert manager.timeout == 5


class TestHealthCheck:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_health_check_returns_true_when_healthy(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.health.return_value = "ok"
        mock_client_class.return_value = mock_client

        manager = QdrantManager()
        result = manager.health_check()
        assert result is True

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_health_check_returns_false_on_timeout(self, mock_client_class):
        from qdrant_client.http.exceptions import ResponseHandlingException
        mock_client = MagicMock()
        mock_client.health.side_effect = ResponseHandlingException("timeout")
        mock_client_class.return_value = mock_client

        manager = QdrantManager()
        result = manager.health_check()
        assert result is False


class TestCollectionExists:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_collection_exists_returns_true(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        assert manager.collection_exists() is True


class TestCreateCollection:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_create_collection_calls_api_with_correct_params(self, mock_client_class):
        from qdrant_client import models
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        manager.create_collection(vector_size=64, m=16, ef_construct=100)

        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        vectors_config = call_args.kwargs["vectors_config"]
        assert vectors_config.size == 64
        assert vectors_config.distance == models.Distance.COSINE


class TestCreatePayloadIndices:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_create_payload_indices_calls_api(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        manager.create_payload_indices()

        # 应调用 5 次 create_payload_index
        assert mock_client.create_payload_index.call_count == 5


class TestGetImportedImageIds:
    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_returns_set_of_image_ids(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.scroll.return_value = (
            [
                MagicMock(payload={"image_id": "img_001"}),
                MagicMock(payload={"image_id": "img_002"}),
                MagicMock(payload={"image_id": "img_001"}),
            ],
            None,
        )
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        result = manager.get_imported_image_ids()
        assert result == {"img_001", "img_002"}

    @patch("KNN_evaluation.qdrant_client.QdrantClient")
    def test_empty_collection_returns_empty_set(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test_collection")
        result = manager.get_imported_image_ids()
        assert result == set()
```

- [x] **Step 2: 运行测试验证失败**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_qdrant_client.py -v
```
预期：所有测试 FAIL（模块/类不存在）

- [x] **Step 3: 实现 `KNN_evaluation/qdrant_client.py`**

```python
"""Qdrant 连接管理与 Collection 生命周期."""
from qdrant_client import QdrantClient, models
from KNN_evaluation.config import (
    QDRANT_URL, COLLECTION_NAME, VECTOR_SIZE, QDRANT_TIMEOUT,
    HNSW_M, HNSW_EF_CONSTRUCT,
)


class QdrantManager:
    """封装 Qdrant 连接创建、健康检查、Collection 管理."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
        timeout: int = QDRANT_TIMEOUT,
    ):
        self.url = url
        self.collection_name = collection_name
        self.timeout = timeout
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.url,
                timeout=self.timeout,
            )
        return self._client

    def health_check(self) -> bool:
        """检查 Qdrant 服务是否可达.

        Returns:
            True 表示健康，False 表示不可达.
        """
        try:
            self.client.health()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        """检查目标 Collection 是否已存在."""
        return self.client.collection_exists(self.collection_name)

    def create_collection(
        self,
        vector_size: int = VECTOR_SIZE,
        m: int = HNSW_M,
        ef_construct: int = HNSW_EF_CONSTRUCT,
    ) -> None:
        """创建 Collection 并配置 HNSW 索引参数.

        若 Collection 已存在则跳过.
        """
        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=False,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=m,
                ef_construct=ef_construct,
            ),
            quantization_config=None,
        )

    def create_payload_indices(self) -> None:
        """为标量字段创建 payload 索引.

        索引字段: label, label_name, utm_easting, utm_northing, image_id.
        utm_zone, pixel_row, pixel_col 不需要过滤/聚合，不建索引.
        """
        indices = [
            ("label", models.IntegerIndexParams(
                type=models.IntegerIndexType.INTEGER, lookup=True, range=True,
            )),
            ("label_name", models.TextIndexParams(
                type=models.TextIndexType.KEYWORD, is_tenant=False,
            )),
            ("utm_easting", models.FloatIndexParams(
                type=models.FloatIndexType.FLOAT, lookup=False, range=True,
            )),
            ("utm_northing", models.FloatIndexParams(
                type=models.FloatIndexType.FLOAT, lookup=False, range=True,
            )),
            ("image_id", models.TextIndexParams(
                type=models.TextIndexType.KEYWORD, is_tenant=False,
            )),
        ]

        for field_name, field_schema in indices:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

    def get_imported_image_ids(self) -> set[str]:
        """获取当前 Collection 中已导入的所有 image_id.

        通过 scroll 遍历全量，提取 payload 中 image_id 去重.
        """
        image_ids: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=10000,
                offset=offset,
                with_payload=["image_id"],
                with_vectors=False,
            )
            for record in records:
                if record.payload and "image_id" in record.payload:
                    image_ids.add(record.payload["image_id"])
            if offset is None:
                break
        return image_ids

    def collection_info(self) -> dict:
        """获取 Collection 统计信息.

        Returns:
            包含 total_points, segments_count 等字段的字典.
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "total_points": info.points_count,
            "vectors_count": info.vectors_count or 0,
            "segments_count": info.segments_count,
            "status": str(info.status),
        }
```

- [x] **Step 4: 运行测试验证通过**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_qdrant_client.py -v
```
预期：全部 PASS

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/qdrant_client.py KNN_evaluation/tests/test_qdrant_client.py
git commit -m "feat: implement QdrantManager with connection, collection, and payload index management"
```

---

### Task 3: PixelDataLoader — SE/DW 数据加载与文件配对

**Files:**
- Create: `KNN_evaluation/data_loader.py`
- Create: `KNN_evaluation/tests/test_data_loader.py`

**Interfaces:**
- Consumes: `src.satellite_embedding_loader.load_embedding`、`KNN_evaluation.label_mapping.LABEL_NAMES`
- Produces:
  - `ImagePair = namedtuple("ImagePair", ["image_id", "se_path", "dw_path", "tif_path"])`
  - `PixelDataLoader.scan_directory(data_dir: Path) -> list[ImagePair]` — 扫描目录，按 E{lon}_N{lat} 坐标段匹配 SE/DW/GeoTIFF 文件对
  - `PixelDataLoader.extract_location_key(filename: str) -> str` — 从文件名提取坐标段（如 `E121.4025_N25.1947`）作为配对 key
  - `PixelDataLoader.load_se(se_path: Path) -> np.ndarray` — 返回 `(64, 128, 128) float64`
  - `PixelDataLoader.load_dw(dw_path: Path) -> np.ndarray` — 返回 `(128, 128) uint8` 标签矩阵
  - `PixelDataLoader.check_image_count(image_id: str, manager: QdrantManager) -> int` — 检查该 image 已导入的 point 数量

- [x] **Step 1: 编写失败的测试**

创建 `KNN_evaluation/tests/test_data_loader.py`：

```python
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
```

- [x] **Step 2: 运行测试验证失败**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_data_loader.py -v
```
预期：全部 FAIL

- [x] **Step 3: 实现 `KNN_evaluation/data_loader.py`**

```python
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
        if suffix not in (".npy", ".npz"):
            raise ValueError(f"不支持的文件格式: {suffix}，DW 仅支持 .npy / .npz")

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

        # 扫描 SE 文件
        se_files: dict[str, Path] = {}
        for fpath in list(se_dir.glob("*.npy")) + list(se_dir.glob("*.npz")):
            key = PixelDataLoader.extract_location_key(fpath.name)
            if key is not None:
                se_files[key] = fpath

        # 扫描 DW 文件
        dw_files: dict[str, Path] = {}
        for fpath in list(dw_dir.glob("*.npy")):
            key = PixelDataLoader.extract_location_key(fpath.name)
            if key is not None:
                dw_files[key] = fpath

        # 扫描 GeoTIFF 文件
        tif_files: dict[str, Path] = {}
        for fpath in list(se_dir.glob("*.tif")) + list(dw_dir.glob("*.tif")):
            key = PixelDataLoader.extract_location_key(fpath.name)
            if key is not None:
                tif_files[key] = fpath

        # 配对
        pairs: list[ImagePair] = []
        common_keys = set(se_files.keys()) & set(dw_files.keys())
        for key in sorted(common_keys):
            pairs.append(ImagePair(
                image_id=key,
                se_path=se_files[key],
                dw_path=dw_files[key],
                tif_path=tif_files.get(key),
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
```

- [x] **Step 4: 运行测试验证通过**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_data_loader.py -v
```
预期：全部 PASS（mock 相关测试）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/data_loader.py KNN_evaluation/tests/test_data_loader.py
git commit -m "feat: implement PixelDataLoader with SE/DW loading and file pair matching"
```

---

### Task 4: UTM 坐标计算模块

**Files:**
- Create: `KNN_evaluation/coordinate_utils.py`
- Create: `KNN_evaluation/tests/test_coordinate_utils.py`

**Interfaces:**
- Consumes: rasterio、pyproj
- Produces:
  - `compute_utm_grid(tif_path: Path | None) -> tuple[np.ndarray, np.ndarray, int | None]` — 返回 `(easting_grid, northing_grid, utm_zone)`，shape 均为 `(128, 128)`；tif_path 为 None 时返回全 NaN + None zone
  - `read_geotiff_meta(tif_path: Path) -> dict` — 返回 `{crs, transform, utm_zone, bounds}`

- [x] **Step 1: 编写失败的测试**

创建 `KNN_evaluation/tests/test_coordinate_utils.py`：

```python
"""Tests for coordinate_utils."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from KNN_evaluation.coordinate_utils import compute_utm_grid, read_geotiff_meta


class TestReadGeotiffMeta:
    @patch("KNN_evaluation.coordinate_utils.rasterio")
    def test_returns_crs_transform_zone(self, mock_rio):
        mock_src = MagicMock()
        mock_src.crs = "EPSG:32651"
        mock_src.transform = MagicMock()
        mock_src.bounds = MagicMock()
        mock_rio.open.return_value.__enter__ = MagicMock(return_value=mock_src)
        mock_rio.open.return_value.__exit__ = MagicMock(return_value=False)

        # rasterio.open 作为 context manager
        mock_rio.open.return_value = mock_src

        result = read_geotiff_meta(Path("/fake/test.tif"))
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
        # Mock affine transform: (c, a, b, f, d, e)
        # pixel 0,0 → easting=c, northing=f
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
        # 第一个像素的坐标
        assert easting[0, 0] == 500000.0
        assert northing[0, 0] == 4000000.0
```

- [x] **Step 2: 运行测试验证失败**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_coordinate_utils.py -v
```
预期：全部 FAIL

- [x] **Step 3: 实现 `KNN_evaluation/coordinate_utils.py`**

```python
"""UTM 坐标计算：从 GeoTIFF Affine Transform 逐像素计算坐标."""
import warnings
from pathlib import Path
import numpy as np

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
```

- [x] **Step 4: 运行测试验证通过**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_coordinate_utils.py -v
```
预期：全部 PASS

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/coordinate_utils.py KNN_evaluation/tests/test_coordinate_utils.py
git commit -m "feat: implement UTM coordinate grid computation from GeoTIFF transform"
```

---

### Task 5: PixelImporter — 批量导入管线（含断点续传）

**Files:**
- Create: `KNN_evaluation/importer.py`
- Create: `KNN_evaluation/tests/test_importer.py`

**Interfaces:**
- Consumes: `QdrantManager`、`PixelDataLoader`、`compute_utm_grid`、`KNN_evaluation.config`、`KNN_evaluation.label_mapping`
- Produces:
  - `PixelImporter.__init__(manager: QdrantManager, batch_size: int)` — 初始化导入器
  - `PixelImporter.import_directory(data_dir: Path, no_resume: bool, reindex: bool) -> dict` — 完整导入流程，返回统计字典
  - `PixelImporter.import_image_pair(pair: ImagePair) -> tuple[int, int]` — 导入单个影像对，返回 `(已导入点数, 已跳过数)`
  - `PixelImporter.build_points(se_data, dw_data, easting, northing, utm_zone, image_id) -> list[PointStruct]` — 构建该影像的全部 16,384 个 point
  - `PixelImporter._batch_upsert(points: list[PointStruct]) -> None` — 分批 upsert

- [x] **Step 1: 编写失败的测试**

创建 `KNN_evaluation/tests/test_importer.py`：

```python
"""Tests for PixelImporter."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.data_loader import ImagePair
from KNN_evaluation.qdrant_client import QdrantManager


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
        assert p0.id == "E121.4025_N25.1947_0_0"
        assert len(p0.vector) == 64
        assert p0.payload["label"] == 3
        assert p0.payload["label_name"] == "flooded_vegetation"
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
        mock_client.count.return_value = MagicMock(count=16384)
        mock_client_class.return_value = mock_client
        manager = QdrantManager(collection_name="test")

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
```

- [x] **Step 2: 运行测试验证失败**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_importer.py -v
```
预期：全部 FAIL

- [x] **Step 3: 实现 `KNN_evaluation/importer.py`**

```python
"""批量导入管线：将 SE/DW 像素数据批量导入 Qdrant."""
import time
import warnings
from pathlib import Path
import numpy as np
from qdrant_client import models

from KNN_evaluation.config import BATCH_SIZE, VECTOR_SIZE
from KNN_evaluation.label_mapping import LABEL_NAMES
from KNN_evaluation.data_loader import PixelDataLoader, ImagePair
from KNN_evaluation.coordinate_utils import compute_utm_grid


class PixelImporter:
    """像素数据批量导入器.

    编排 scan → load → compute coords → build points → upsert 流程.
    """

    def __init__(self, manager, batch_size: int = BATCH_SIZE):
        self.manager = manager
        self.batch_size = batch_size
        self._stats: dict = {}

    def build_points(
        self,
        se_data: np.ndarray,
        dw_data: np.ndarray,
        easting: np.ndarray,
        northing: np.ndarray,
        utm_zone: int | None,
        image_id: str,
    ) -> list[models.PointStruct]:
        """将单张影像的所有像素构造为 Qdrant PointStruct 列表.

        Args:
            se_data: (64, 128, 128) float64 embedding 数据.
            dw_data: (128, 128) uint8 标签矩阵.
            easting: (128, 128) float64 UTM 东向坐标.
            northing: (128, 128) float64 UTM 北向坐标.
            utm_zone: UTM 投影带号，None 时记为 -1.
            image_id: 影像标识.

        Returns:
            长度为 16384 的 PointStruct 列表.
        """
        points: list[models.PointStruct] = []
        zone = utm_zone if utm_zone is not None else -1

        for row in range(128):
            for col in range(128):
                label = int(dw_data[row, col])
                point = models.PointStruct(
                    id=f"{image_id}_{row}_{col}",
                    vector=se_data[:, row, col].tolist(),
                    payload={
                        "label": label,
                        "label_name": LABEL_NAMES.get(label, "unknown"),
                        "utm_easting": float(easting[row, col]),
                        "utm_northing": float(northing[row, col]),
                        "utm_zone": zone,
                        "image_id": image_id,
                        "pixel_row": row,
                        "pixel_col": col,
                    },
                )
                points.append(point)

        return points

    def _batch_upsert(self, points: list[models.PointStruct]) -> None:
        """分批 upsert points 到 Qdrant.

        Args:
            points: PointStruct 列表.
        """
        for i in range(0, len(points), self.batch_size):
            batch = points[i:i + self.batch_size]
            self.manager.client.upsert(
                collection_name=self.manager.collection_name,
                points=batch,
                wait=True,
            )

    def import_image_pair(self, pair: ImagePair) -> tuple[int, int]:
        """导入一对 SE + DW 文件的所有像素.

        Args:
            pair: 匹配的 ImagePair.

        Returns:
            (imported, skipped) — 新导入的点数和跳过的点数.
        """
        # 断点续传检查
        existing_count = PixelDataLoader.check_image_count(pair.image_id, self.manager)
        total = 128 * 128

        if existing_count >= total:
            return 0, total

        if existing_count > 0:
            warnings.warn(
                f"{pair.image_id}: 已存在 {existing_count} 条记录（共 {total}），将覆盖重传"
            )

        # 加载数据
        se_data = PixelDataLoader.load_se(pair.se_path)
        dw_data = PixelDataLoader.load_dw(pair.dw_path)
        easting, northing, utm_zone = compute_utm_grid(pair.tif_path)

        if pair.tif_path is None:
            warnings.warn(f"{pair.image_id}: 未找到 GeoTIFF，UTM 坐标设为 NaN")

        # 构建并导入
        points = self.build_points(se_data, dw_data, easting, northing, utm_zone, pair.image_id)
        self._batch_upsert(points)

        return total, existing_count

    def import_directory(
        self,
        data_dir: Path,
        no_resume: bool = False,
        reindex: bool = False,
    ) -> dict:
        """完整导入流程：扫描目录 → 配对 → 逐文件导入.

        Args:
            data_dir: 数据根目录（含 SE/ 和 DW/ 子目录）.
            no_resume: True 时不检查断点续传，强制重新导入.
            reindex: True 时导入完成后触发 HNSW 全量索引重建.

        Returns:
            统计字典:
            {
                "total_pixels": int,
                "total_images": int,
                "skipped_images": int,
                "imported_images": int,
                "label_counts": dict[str, int],
                "elapsed_sec": float,
                "rate_pps": float,  // pixels per second
            }
        """
        if not self.manager.health_check():
            raise ConnectionError(f"Qdrant 不可达: {self.manager.url}")

        if not self.manager.collection_exists():
            raise RuntimeError(
                f"Collection '{self.manager.collection_name}' 不存在，请先创建 Collection"
            )

        start_time = time.perf_counter()

        pairs = PixelDataLoader.scan_directory(Path(data_dir))
        if not pairs:
            print("警告: 未找到匹配的 SE/DW 文件对")
            return {
                "total_pixels": 0, "total_images": 0,
                "skipped_images": 0, "imported_images": 0,
                "label_counts": {}, "elapsed_sec": 0, "rate_pps": 0,
            }

        total_pixels = 0
        total_images = 0
        skipped_images = 0
        label_counts: dict[str, int] = {}
        imported_ids = set() if no_resume else self.manager.get_imported_image_ids()

        # 使用 tqdm 进度条
        try:
            from tqdm import tqdm
            pair_iter = tqdm(pairs, desc="导入影像", unit="img")
        except ImportError:
            pair_iter = pairs

        for pair in pair_iter:
            total_images += 1
            pair_label_counts: dict[str, int] = {}

            if not no_resume and pair.image_id in imported_ids:
                skipped_images += 1
                # 计算该 image 的 label 统计（需要加载 DW）
                dw_data = PixelDataLoader.load_dw(pair.dw_path)
                unique, counts = np.unique(dw_data, return_counts=True)
                for lbl, cnt in zip(unique, counts):
                    name = LABEL_NAMES.get(int(lbl), "unknown")
                    pair_label_counts[name] = int(cnt)
            else:
                imported, skip = self.import_image_pair(pair)
                total_pixels += imported

                # 统计 label 分布
                dw_data = PixelDataLoader.load_dw(pair.dw_path)
                unique, counts = np.unique(dw_data, return_counts=True)
                for lbl, cnt in zip(unique, counts):
                    name = LABEL_NAMES.get(int(lbl), "unknown")
                    pair_label_counts[name] = int(cnt)

            # 合并 label 统计
            for name, cnt in pair_label_counts.items():
                label_counts[name] = label_counts.get(name, 0) + cnt

        elapsed = time.perf_counter() - start_time

        # 索引重建
        if reindex:
            self.manager.client.update_collection(
                collection_name=self.manager.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )

        stats = {
            "total_pixels": total_pixels,
            "total_images": total_images,
            "skipped_images": skipped_images,
            "imported_images": total_images - skipped_images,
            "label_counts": label_counts,
            "elapsed_sec": elapsed,
            "rate_pps": total_pixels / elapsed if elapsed > 0 else 0,
        }
        self._stats = stats
        return stats
```

- [x] **Step 4: 运行测试验证通过**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_importer.py -v
```
预期：全部 PASS

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/importer.py KNN_evaluation/tests/test_importer.py
git commit -m "feat: implement PixelImporter with resume, batch upsert, and import stats"
```

---

### Task 6: PixelSearcher — 联合检索接口

**Files:**
- Create: `KNN_evaluation/searcher.py`
- Create: `KNN_evaluation/tests/test_searcher.py`

**Interfaces:**
- Consumes: `QdrantManager`、`KNN_evaluation.config`、`KNN_evaluation.label_mapping`
- Produces:
  - `HitRecord = dataclass` — 单条查询结果（id, score, label, label_name, utm_easting, utm_northing, utm_zone, image_id, pixel_row, pixel_col）
  - `SearchResult = dataclass` — 搜索结果集（hits, elapsed_ms, label_distribution, search_mode, query_params）
  - `PixelSearcher.__init__(manager: QdrantManager)` — 初始化检索器
  - `PixelSearcher.search(query_vector, k, label_filter, utm_range, exact, ef_search) -> SearchResult` — 执行检索
  - `PixelSearcher._build_filter(label_filter, utm_range) -> models.Filter | None` — 构建过滤条件（AND 组合）

- [x] **Step 1: 编写失败的测试**

创建 `KNN_evaluation/tests/test_searcher.py`：

```python
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
        from qdrant_client import models
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
        mock_client.search.return_value = [mock_hit]
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
        mock_client.search.return_value = []
        mock_client_class.return_value = mock_client

        manager = QdrantManager(collection_name="test")
        searcher = PixelSearcher(manager)

        query = np.random.randn(64).astype(np.float64)
        result = searcher.search(query, k=5, exact=True)

        assert result.search_mode == "exact"
        # 验证搜索参数包含 exact=True
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["search_params"].exact is True
```

- [x] **Step 2: 运行测试验证失败**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_searcher.py -v
```
预期：全部 FAIL

- [x] **Step 3: 实现 `KNN_evaluation/searcher.py`**

```python
"""联合检索接口：向量检索 + 标量过滤."""
import time
from dataclasses import dataclass, field
import numpy as np
from qdrant_client import models


@dataclass
class HitRecord:
    """单条检索命中记录."""
    id: str
    score: float
    label: int
    label_name: str
    utm_easting: float
    utm_northing: float
    utm_zone: int
    image_id: str
    pixel_row: int
    pixel_col: int


@dataclass
class SearchResult:
    """检索结果集."""
    hits: list[HitRecord]
    elapsed_ms: float
    label_distribution: dict[str, int] = field(default_factory=dict)
    search_mode: str = "ann"
    query_params: dict = field(default_factory=dict)


class PixelSearcher:
    """像素嵌入向量检索器.

    封装 Qdrant search API，支持 label 过滤、UTM 范围过滤、exact/ANN 模式切换.
    """

    def __init__(self, manager):
        self.manager = manager

    def _build_filter(
        self,
        label_filter: list[int] | None,
        utm_range: dict | None,
    ) -> models.Filter | None:
        """构建 Qdrant 过滤条件（AND 语义）.

        Args:
            label_filter: 允许的标签值列表，如 [0, 1].
            utm_range: UTM 范围，含 min_e/max_e/min_n/max_n 键.

        Returns:
            Filter 对象或 None.
        """
        conditions = []

        if label_filter:
            conditions.append(
                models.FieldCondition(
                    key="label",
                    match=models.MatchAny(any=label_filter),
                )
            )

        if utm_range:
            conditions.extend([
                models.FieldCondition(
                    key="utm_easting",
                    range=models.Range(
                        gte=utm_range["min_e"],
                        lte=utm_range["max_e"],
                    ),
                ),
                models.FieldCondition(
                    key="utm_northing",
                    range=models.Range(
                        gte=utm_range["min_n"],
                        lte=utm_range["max_n"],
                    ),
                ),
            ])

        if not conditions:
            return None
        return models.Filter(must=conditions)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        label_filter: list[int] | None = None,
        utm_range: dict | None = None,
        exact: bool = False,
        ef_search: int = 64,
    ) -> SearchResult:
        """执行向量检索.

        Args:
            query_vector: (64,) float64 查询向量.
            k: 返回 Top-K 结果.
            label_filter: 标签过滤列表.
            utm_range: UTM 坐标范围过滤.
            exact: True 时使用暴力精确搜索.
            ef_search: ANN 搜索的 ef 参数.

        Returns:
            SearchResult 包含命中记录、耗时、标签分布等.

        Raises:
            ValueError: query_vector 维度不是 64.
        """
        if query_vector.shape != (64,):
            raise ValueError(
                f"query_vector 维度应为 (64,)，实际: {query_vector.shape}"
            )

        qdrant_filter = self._build_filter(label_filter, utm_range)

        start = time.perf_counter()
        hits = self.manager.client.search(
            collection_name=self.manager.collection_name,
            query_vector=query_vector.tolist(),
            query_filter=qdrant_filter,
            limit=k,
            search_params=models.SearchParams(
                exact=exact,
                hnsw_ef=None if exact else ef_search,
            ),
            with_payload=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 解析 hits 并计算 label 分布
        hit_records: list[HitRecord] = []
        label_dist: dict[str, int] = {}
        for h in hits:
            p = h.payload or {}
            record = HitRecord(
                id=str(h.id),
                score=float(h.score),
                label=int(p.get("label", -1)),
                label_name=str(p.get("label_name", "unknown")),
                utm_easting=float(p.get("utm_easting", float("nan"))),
                utm_northing=float(p.get("utm_northing", float("nan"))),
                utm_zone=int(p.get("utm_zone", -1)),
                image_id=str(p.get("image_id", "")),
                pixel_row=int(p.get("pixel_row", -1)),
                pixel_col=int(p.get("pixel_col", -1)),
            )
            hit_records.append(record)
            label_dist[record.label_name] = label_dist.get(record.label_name, 0) + 1

        return SearchResult(
            hits=hit_records,
            elapsed_ms=elapsed_ms,
            label_distribution=label_dist,
            search_mode="exact" if exact else "ann",
            query_params={
                "k": k,
                "label_filter": label_filter,
                "utm_range": utm_range,
                "exact": exact,
                "ef_search": ef_search,
            },
        )
```

- [x] **Step 4: 运行测试验证通过**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/test_searcher.py -v
```
预期：全部 PASS

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/searcher.py KNN_evaluation/tests/test_searcher.py
git commit -m "feat: implement PixelSearcher with label/UTM filter, exact/ANN modes"
```

---

### Task 7: CLI 命令行入口

**Files:**
- Create: `KNN_evaluation/cli.py`

**Interfaces:**
- Consumes: `QdrantManager`、`PixelImporter`、`PixelSearcher`、`KNN_evaluation.config`、`KNN_evaluation.label_mapping`
- Produces: `knn-eval` CLI，含 `import`、`search`、`stats` 三个子命令

- [x] **Step 1: 创建 `KNN_evaluation/cli.py`**

```python
"""Qdrant KNN 评估系统 — 命令行入口.

用法:
    python -m KNN_evaluation.cli import <directory> [--batch-size N] [--no-resume] [--reindex]
    python -m KNN_evaluation.cli search --query-file <path> [--k N] [--label ...] [--utm-range ...] [--exact]
    python -m KNN_evaluation.cli stats [--json]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

from KNN_evaluation.config import QDRANT_URL, COLLECTION_NAME, BATCH_SIZE
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.searcher import PixelSearcher


def _parse_utm_range(utm_str: str) -> dict:
    """解析 UTM 范围字符串.

    Args:
        utm_str: 格式 "min_e,max_e,min_n,max_n" 如 "500000,501000,4000000,4001000".

    Returns:
        dict with min_e/max_e/min_n/max_n keys.

    Raises:
        ValueError: 格式不正确.
    """
    parts = utm_str.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"UTM 范围格式应为 'min_e,max_e,min_n,max_n'，实际: {utm_str}"
        )
    return {
        "min_e": float(parts[0]),
        "max_e": float(parts[1]),
        "min_n": float(parts[2]),
        "max_n": float(parts[3]),
    }


def _parse_label(label_str: str) -> list[int]:
    """解析标签字符串.

    Args:
        label_str: 逗号分隔的标签值或名称，如 "0,1,2" 或 "water,trees".

    Returns:
        标签整数列表.
    """
    from KNN_evaluation.label_mapping import LABEL_IDS

    values: list[int] = []
    for part in label_str.split(","):
        part = part.strip()
        if part.isdigit():
            values.append(int(part))
        elif part in LABEL_IDS:
            values.append(LABEL_IDS[part])
        else:
            raise ValueError(f"未知标签: {part}")
    return values


def cmd_import(args) -> int:
    """执行 import 子命令."""
    manager = QdrantManager(url=args.qdrant_url)

    if not manager.collection_exists():
        print(f"Collection '{manager.collection_name}' 不存在，正在创建...")
        manager.create_collection()
        manager.create_payload_indices()
        print("Collection 创建完成.")

    importer = PixelImporter(manager, batch_size=args.batch_size)

    try:
        stats = importer.import_directory(
            data_dir=Path(args.directory),
            no_resume=args.no_resume,
            reindex=args.reindex,
        )
    except ConnectionError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    # 输出统计
    print(f"\n{'='*50}")
    print(f"导入完成")
    print(f"{'='*50}")
    print(f"总像素数:     {stats['total_pixels']:,}")
    print(f"总影像数:     {stats['total_images']}")
    print(f"新导入影像:   {stats['imported_images']}")
    print(f"跳过(已存在):  {stats['skipped_images']}")
    print(f"总耗时:       {stats['elapsed_sec']:.1f}s")
    print(f"平均速率:     {stats['rate_pps']:.0f} 像素/秒")
    print(f"\n标签分布:")
    for name in sorted(stats["label_counts"].keys()):
        cnt = stats["label_counts"][name]
        pct = cnt / max(stats["total_pixels"], 1) * 100
        print(f"  {name:<22} {cnt:>10,}  ({pct:>5.1f}%)")
    print(f"\nCollection 当前总量: {manager.collection_info()['total_points']:,}")

    return 0


def cmd_search(args) -> int:
    """执行 search 子命令."""
    manager = QdrantManager(url=args.qdrant_url)

    if not manager.collection_exists():
        print(f"错误: Collection '{manager.collection_name}' 不存在，请先执行 import", file=sys.stderr)
        return 1

    # 获取 query vector
    if args.query_file:
        query_vector = np.load(args.query_file)
        if query_vector.ndim == 2:
            query_vector = query_vector.squeeze(axis=0)
    elif args.random:
        # 从 collection 中随机取一个 vector
        scroll_result = manager.client.scroll(
            collection_name=manager.collection_name,
            limit=1,
            with_vectors=True,
        )
        if not scroll_result[0]:
            print("错误: Collection 为空，无法随机获取 query vector", file=sys.stderr)
            return 1
        query_vector = np.array(scroll_result[0][0].vector, dtype=np.float64)
        print(f"随机选取 query point: {scroll_result[0][0].id}")
    elif args.query_spec:
        image_id, row, col = args.query_spec
        from qdrant_client import models
        scroll_result = manager.client.scroll(
            collection_name=manager.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="image_id", match=models.MatchValue(value=image_id)),
                    models.FieldCondition(key="pixel_row", match=models.MatchValue(value=int(row))),
                    models.FieldCondition(key="pixel_col", match=models.MatchValue(value=int(col))),
                ],
            ),
            limit=1,
            with_vectors=True,
        )
        if not scroll_result[0]:
            print(f"错误: 未找到像素 {image_id}_{row}_{col}", file=sys.stderr)
            return 1
        query_vector = np.array(scroll_result[0][0].vector, dtype=np.float64)
    else:
        print("错误: 必须指定 --query-file、--random 或 --query-spec", file=sys.stderr)
        return 1

    # 解析过滤条件
    label_filter = None
    if args.label:
        label_filter = _parse_label(args.label)

    utm_range = None
    if args.utm_range:
        utm_range = _parse_utm_range(args.utm_range)

    # 执行搜索
    searcher = PixelSearcher(manager)
    try:
        result = searcher.search(
            query_vector=query_vector,
            k=args.k,
            label_filter=label_filter,
            utm_range=utm_range,
            exact=args.exact,
            ef_search=args.ef_search,
        )
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    # 输出结果
    if args.output == "json":
        hits_json = [
            {
                "id": h.id, "score": h.score, "label": h.label,
                "label_name": h.label_name, "utm_easting": h.utm_easting,
                "utm_northing": h.utm_northing, "utm_zone": h.utm_zone,
                "image_id": h.image_id, "pixel_row": h.pixel_row,
                "pixel_col": h.pixel_col,
            }
            for h in result.hits
        ]
        print(json.dumps({
            "hits": hits_json,
            "elapsed_ms": result.elapsed_ms,
            "label_distribution": result.label_distribution,
            "search_mode": result.search_mode,
            "query_params": result.query_params,
        }, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*70}")
        print(f"检索结果 (mode={result.search_mode}, k={result.query_params['k']}, "
              f"elapsed={result.elapsed_ms:.1f}ms)")
        print(f"{'='*70}")
        print(f"{'#':>4} {'ID':<40} {'Score':>8} {'Label':<22} {'Easting':>12} {'Northing':>12}")
        print(f"{'-'*4} {'-'*40} {'-'*8} {'-'*22} {'-'*12} {'-'*12}")
        for i, h in enumerate(result.hits, 1):
            print(f"{i:>4} {h.id:<40} {h.score:>8.4f} {h.label_name:<22} "
                  f"{h.utm_easting:>12.1f} {h.utm_northing:>12.1f}")

        if result.label_distribution:
            print(f"\n标签分布: {result.label_distribution}")

    return 0


def cmd_stats(args) -> int:
    """执行 stats 子命令."""
    manager = QdrantManager(url=args.qdrant_url)

    if not manager.collection_exists():
        print(f"错误: Collection '{manager.collection_name}' 不存在", file=sys.stderr)
        return 1

    info = manager.collection_info()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"Collection: {manager.collection_name}")
        print(f"  总点数:    {info['total_points']:,}")
        print(f"  向量数:    {info['vectors_count']:,}")
        print(f"  分段数:    {info['segments_count']}")
        print(f"  状态:      {info['status']}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="knn-eval",
        description="Qdrant KNN 评估系统 — 像素级向量检索与评估",
    )
    sub = parser.add_subparsers(dest="command")

    # --- import ---
    p_import = sub.add_parser("import", help="批量导入 SE/DW 像素数据到 Qdrant")
    p_import.add_argument("directory", help="包含 SE/ 和 DW/ 子目录的数据根目录")
    p_import.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                          help=f"每批 upsert 的点数 (默认: {BATCH_SIZE})")
    p_import.add_argument("--no-resume", action="store_true",
                          help="不检查断点续传，强制重新导入")
    p_import.add_argument("--reindex", action="store_true",
                          help="导入完成后触发 HNSW 全量索引重建")
    p_import.add_argument("--qdrant-url", default=QDRANT_URL,
                          help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- search ---
    p_search = sub.add_parser("search", help="执行向量检索")
    group = p_search.add_mutually_exclusive_group(required=True)
    group.add_argument("--query-file", type=str, help="query vector 的 .npy 文件路径")
    group.add_argument("--random", action="store_true", help="从 collection 随机选取一个 vector")
    group.add_argument("--query-spec", nargs=3, metavar=("IMAGE_ID", "ROW", "COL"),
                       help="按 image_id、row、col 指定 query 像素")
    p_search.add_argument("--k", type=int, default=10, help="返回 Top-K 结果 (默认: 10)")
    p_search.add_argument("--label", type=str, help="标签过滤，逗号分隔，如 '0,1' 或 'water,trees'")
    p_search.add_argument("--utm-range", type=str,
                          help="UTM 范围过滤，格式: min_e,max_e,min_n,max_n")
    p_search.add_argument("--exact", action="store_true", help="使用暴力精确搜索")
    p_search.add_argument("--ef-search", type=int, default=64,
                          help="ANN 搜索的 ef 参数 (默认: 64)")
    p_search.add_argument("--output", choices=["table", "json"], default="table",
                          help="输出格式 (默认: table)")
    p_search.add_argument("--qdrant-url", default=QDRANT_URL,
                          help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    # --- stats ---
    p_stats = sub.add_parser("stats", help="查看 Collection 统计信息")
    p_stats.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    p_stats.add_argument("--qdrant-url", default=QDRANT_URL,
                         help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")

    args = parser.parse_args()

    if args.command == "import":
        return cmd_import(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "stats":
        return cmd_stats(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 2: 验证 CLI 帮助输出**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m KNN_evaluation.cli --help
```
预期：显示 import/search/stats 子命令帮助

- [x] **Step 3: 验证各子命令帮助**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m KNN_evaluation.cli import --help
cd "D:\Project\光机所项目\evaluation_scripts" && python -m KNN_evaluation.cli search --help
cd "D:\Project\光机所项目\evaluation_scripts" && python -m KNN_evaluation.cli stats --help
```
预期：每个子命令显示各自的参数帮助

- [x] **Step 4: Commit**

```bash
git add KNN_evaluation/cli.py
git commit -m "feat: implement CLI with import, search, and stats subcommands"
```

---

### Task 8: 集成测试 — 端到端验证

**Files:**
- Create: `KNN_evaluation/tests/conftest.py`

**Interfaces:**
- Consumes: 所有模块、`data_demo/` 测试数据、Qdrant Docker 环境
- Produces: 端到端测试，在 Docker Qdrant 中验证导入 + 检索完整流程

- [x] **Step 1: 创建 `KNN_evaluation/tests/conftest.py`**

```python
"""Fixtures for KNN_evaluation integration tests."""
import pytest
import subprocess
import time
from KNN_evaluation.qdrant_client import QdrantManager


def _qdrant_is_running() -> bool:
    """检查本地 Qdrant Docker 是否运行."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        return "qdrant" in result.stdout or "qdrant-knn-eval" in result.stdout
    except Exception:
        return False


def _start_qdrant():
    """启动 Qdrant Docker 容器."""
    subprocess.run(
        [
            "docker", "run", "-d", "--name", "qdrant-knn-eval",
            "-p", "6333:6333", "-p", "6334:6334",
            "qdrant/qdrant:latest",
        ],
        capture_output=True, timeout=30,
    )
    time.sleep(2)  # 等待启动


@pytest.fixture(scope="session")
def qdrant_manager():
    """提供连接到本地 Qdrant 的 QdrantManager.

    若 Qdrant 未运行则自动启动 Docker 容器.
    """
    if not _qdrant_is_running():
        _start_qdrant()

    manager = QdrantManager(
        url="http://localhost:6333",
        collection_name="test_pixel_embeddings",
        timeout=10,
    )

    # 确保 collection 存在
    if not manager.collection_exists():
        manager.create_collection()
        manager.create_payload_indices()

    yield manager

    # 清理测试 collection
    try:
        manager.client.delete_collection("test_pixel_embeddings")
    except Exception:
        pass
```

- [x] **Step 2: 运行集成测试**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/ -v --timeout=120
```
预期：全部单元测试 PASS，集成测试在有 Docker 时 PASS，无 Docker 时 SKIP

- [x] **Step 3: Commit**

```bash
git add KNN_evaluation/tests/conftest.py
git commit -m "test: add integration test fixtures for Qdrant Docker environment"
```

---

### Task 9: 最终质量检查

**行尾配置**
**Files:**
- Check: `KNN_evaluation/` 全部 Python 文件

- [x] **Step 1: 确认所有 Import 正常**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -c "
import KNN_evaluation
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.data_loader import PixelDataLoader
from KNN_evaluation.coordinate_utils import compute_utm_grid
from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.searcher import PixelSearcher
from KNN_evaluation.config import QDRANT_URL, COLLECTION_NAME
from KNN_evaluation.label_mapping import LABEL_NAMES
print('All imports OK')
"
```

- [x] **Step 2: 运行所有测试**

```bash
cd "D:\Project\光机所项目\evaluation_scripts" && python -m pytest KNN_evaluation/tests/ -v
```

- [x] **Step 3: Commit**

```bash
git add -A KNN_evaluation/
git commit -m "chore: finalize KNN_evaluation package with all modules wired"
```
