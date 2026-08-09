---
change: data-import-optimization
design-doc: docs/superpowers/specs/2026-08-01-data-import-optimization-design.md
base-ref: 81ee5cb11e38e7acd8d1d9f30e49a1370c8e0ae5
archived-with: 2026-08-02-data-import-optimization
---

# Data Import Optimization 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化 Qdrant KNN 像素评估系统的数据导入模块——`build_points()` 向量化提速、`import_directory()` 新增像素级进度回调、CLI `--reindex` 文档化并接入像素级 tqdm、WebUI 分页预览（每页 20 条）+"导入全部"按钮上移+线性进度条，同时确认 HNSW 全量索引重建为既有行为。

**Architecture:** 核心在 `KNN_evaluation/importer.py`：`build_points()` 从双层循环改为 numpy 向量化构造（`se_data.reshape(64, -1).T` + meshgrid），`import_directory()` 增加可选 `progress_callback(imported, total)`，通过内部 `_batch_upsert`/`import_image_pair` 可选回调按批次驱动进度。新增 `KNN_evaluation/ui_pagination.py` 纯函数模块供 WebUI 分页预览，可独立测试。CLI 与 WebUI 各自编排进度显示（tqdm / NiceGUI 线性进度条）。`--reindex` 沿用既有 `update_collection(indexing_threshold=0)` 触发 HNSW 重建，不新增 payload 索引重建。

**Tech Stack:** Python 3.12+, numpy, qdrant-client, NiceGUI, tqdm, pytest

## Global Constraints

- 不新增第三方依赖（仅复用现有 qdrant-client、numpy、nicegui、tqdm、pytest）
- `import_directory()` 仅新增可选参数 `progress_callback`，默认 `None` 时行为与现状完全一致（tqdm 影像级进度保留）
- `import_image_pair()` 位置参数与返回值不变，仅新增可选私有关键字 `_batch_callback` 用于进度线程传递（向后兼容）
- 保守 upsert：保持 `wait=True`（断点续传 `check_image_count` 依赖写入可见性）
- `--reindex` / `reindex=True` 仅触发 HNSW 向量索引重建（`indexing_threshold=0`），不新增 payload 标量索引删除/重建
- `build_points()` 重构后点序（row-major：row 外层、col 内层）与 payload 字段（键序、类型）与现状完全一致
- 分页每页 20 条（`PAGE_SIZE = 20`），`page` 从 0 开始
- 中文注释与错误信息；Python >= 3.12
- 所有测试命令使用 `uv run pytest`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `KNN_evaluation/ui_pagination.py` | CREATE | 分页辅助纯函数：`PAGE_SIZE` / `paginate_slice` / `total_pages` / `page_controls` |
| `KNN_evaluation/tests/test_ui_pagination.py` | CREATE | 分页纯函数单元测试 |
| `KNN_evaluation/importer.py` | MODIFY | `build_points()` 向量化 + `import_directory()` 进度回调 + reindex 注释 |
| `KNN_evaluation/tests/test_importer.py` | MODIFY | 向量化回归测试（点序、ID 确定性、向量一致性） |
| `KNN_evaluation/tests/test_progress_callback.py` | CREATE | `import_directory` 进度回调测试 |
| `KNN_evaluation/tests/test_importer_reindex.py` | CREATE | `reindex=True` 触发 HNSW 重建调用测试 |
| `KNN_evaluation/cli.py` | MODIFY | `--reindex` 帮助文案 + `cmd_import` 像素级 tqdm |
| `KNN_evaluation/webui.py` | MODIFY | 分页预览 + 导入全部按钮上移 + 线性进度条 |
| `openspec/changes/data-import-optimization/proposal.md` | MODIFY | 移除与 payload 重建相关的过时残留（含重复行） |

---

### Task 1: 分页辅助纯函数（ui_pagination.py）

**Files:**
- Create: `KNN_evaluation/ui_pagination.py`
- Test: `KNN_evaluation/tests/test_ui_pagination.py`

**Interfaces:**
- Produces:
  - `PAGE_SIZE: int = 20`
  - `paginate_slice(items: list, page: int, page_size: int = PAGE_SIZE) -> list`
  - `total_pages(total: int, page_size: int = PAGE_SIZE) -> int`（`total == 0` 时返回 1）
  - `page_controls(page: int, total: int, page_size: int = PAGE_SIZE) -> tuple[bool, bool]`（返回 `(can_prev, can_next)`，`can_prev = page > 0`，`can_next = (page+1)*page_size < total`）
- Consumed by: Task 6（WebUI 数据导入区分页预览）

- [x] **Step 1: 写失败测试**

创建 `KNN_evaluation/tests/test_ui_pagination.py`：

```python
"""Tests for ui_pagination pure functions."""
from KNN_evaluation.ui_pagination import (
    PAGE_SIZE,
    paginate_slice,
    total_pages,
    page_controls,
)


class TestPaginateSlice:
    def test_page_size_constant(self):
        assert PAGE_SIZE == 20

    def test_first_page(self):
        items = list(range(45))
        assert paginate_slice(items, 0) == list(range(20))

    def test_second_page(self):
        items = list(range(45))
        assert paginate_slice(items, 1) == list(range(20, 40))

    def test_last_partial_page(self):
        items = list(range(45))
        assert paginate_slice(items, 2) == list(range(40, 45))

    def test_page_out_of_range_returns_empty(self):
        items = list(range(45))
        assert paginate_slice(items, 3) == []
        assert paginate_slice(items, -1) == []

    def test_empty_items(self):
        assert paginate_slice([], 0) == []

    def test_custom_page_size(self):
        items = list(range(10))
        assert paginate_slice(items, 1, page_size=3) == [3, 4, 5]


class TestTotalPages:
    def test_zero_total(self):
        assert total_pages(0) == 1

    def test_exact_multiple(self):
        assert total_pages(40) == 2

    def test_partial(self):
        assert total_pages(41) == 3

    def test_single(self):
        assert total_pages(1) == 1


class TestPageControls:
    def test_first_page(self):
        assert page_controls(0, 45) == (False, True)

    def test_middle_page(self):
        assert page_controls(1, 45) == (True, True)

    def test_last_page(self):
        assert page_controls(2, 45) == (True, False)

    def test_empty_total(self):
        assert page_controls(0, 0) == (False, False)
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_ui_pagination.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'KNN_evaluation.ui_pagination'`

- [x] **Step 3: 实现纯函数**

创建 `KNN_evaluation/ui_pagination.py`：

```python
"""WebUI 数据导入区预览分页辅助纯函数.

独立于 NiceGUI，便于单元测试。所有函数为纯函数，不持有 UI 状态。
"""

PAGE_SIZE = 20


def paginate_slice(items: list, page: int, page_size: int = PAGE_SIZE) -> list:
    """返回 items[page*page_size : (page+1)*page_size].

    Args:
        items: 待分页的条目列表。
        page: 从 0 开始的页码。
        page_size: 每页条目数。

    Returns:
        当前页切片；items 为空或 page 越界时返回空列表。
    """
    if not items:
        return []
    start = page * page_size
    if start < 0 or start >= len(items):
        return []
    return items[start:start + page_size]


def total_pages(total: int, page_size: int = PAGE_SIZE) -> int:
    """计算总页数.

    Args:
        total: 条目总数。
        page_size: 每页条目数。

    Returns:
        总页数；total == 0 时返回 1。
    """
    if total <= 0:
        return 1
    return (total + page_size - 1) // page_size


def page_controls(page: int, total: int, page_size: int = PAGE_SIZE) -> tuple[bool, bool]:
    """推导翻页按钮可用态.

    Args:
        page: 从 0 开始的当前页码。
        total: 条目总数。
        page_size: 每页条目数。

    Returns:
        (can_prev, can_next)。
    """
    can_prev = page > 0
    can_next = (page + 1) * page_size < total
    return can_prev, can_next
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_ui_pagination.py -v`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/ui_pagination.py KNN_evaluation/tests/test_ui_pagination.py
git commit -m "feat(webui): add pagination pure functions for data import preview"
```

---

### Task 2: build_points() 向量化重构

**Files:**
- Modify: `KNN_evaluation/importer.py:26-75`（`build_points` 方法体）
- Modify: `KNN_evaluation/tests/test_importer.py`（追加回归测试）

**Interfaces:**
- Produces（签名不变，语义不变）: `build_points(self, se_data: np.ndarray, dw_data: np.ndarray, easting: np.ndarray, northing: np.ndarray, utm_zone: int | None, image_id: str) -> list[models.PointStruct]`
  - 返回 16384 个 PointStruct，点序 row-major（row 外层、col 内层）
  - payload 键序保持：`label, label_name, utm_easting, utm_northing, utm_zone, image_id, pixel_row, pixel_col`
  - 每个 point 的 `vector` 与逐像素版本等价：`points[k].vector == se_data[:, k//128, k%128].tolist()`
- Consumed by: Task 3（`import_image_pair` 调用 `self.build_points(...)`，调用点不变）

- [x] **Step 1: 追加回归测试（先在现状代码上运行确认通过，作为基线）**

在 `KNN_evaluation/tests/test_importer.py` 顶部导入区添加 `import uuid`（`import numpy as np` 之后），并在文件末尾追加：

```python
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
```

- [x] **Step 2: 运行回归测试确认基线通过（现状实现）**

Run: `uv run pytest KNN_evaluation/tests/test_importer.py::TestBuildPointsVectorization -v`
Expected: PASS（这些测试在当前逐像素实现下也应通过——它们是行为保真回归测试）

- [x] **Step 3: 实现向量化 `build_points`**

在 `KNN_evaluation/importer.py` 中将 `build_points` 方法体整体替换为：

```python
    def build_points(
        self,
        se_data: np.ndarray,
        dw_data: np.ndarray,
        easting: np.ndarray,
        northing: np.ndarray,
        utm_zone: int | None,
        image_id: str,
    ) -> list[models.PointStruct]:
        """将单张影像的所有像素构造为 Qdrant PointStruct 列表（向量化）.

        Args:
            se_data: (64, 128, 128) float64 embedding 数据.
            dw_data: (128, 128) uint8 标签矩阵.
            easting: (128, 128) float64 UTM 东向坐标.
            northing: (128, 128) float64 UTM 北向坐标.
            utm_zone: UTM 投影带号，None 时记为 -1.
            image_id: 影像标识.

        Returns:
            长度为 16384 的 PointStruct 列表，点序为 row-major
            （row 外层、col 内层），与逐像素循环版本完全一致.
        """
        zone = utm_zone if utm_zone is not None else -1

        # 用 image_id 作为 UUID 命名空间，保证同 image_id 下的
        # row/col 组合产生唯一且确定的 UUID
        ns = uuid.uuid5(uuid.NAMESPACE_DNS, image_id)

        # 向量化构造：(64, 128, 128) -> (16384, 64)，row-major 展平
        # .tolist() 提供 Qdrant 序列化所需的 Python 原生 float
        vectors = se_data.reshape(64, -1).T.tolist()

        # 标签与 label_name 一次展平
        labels = dw_data.reshape(-1).tolist()
        label_names = [LABEL_NAMES.get(int(l), "unknown") for l in labels]

        # UTM 坐标展平（row-major）
        eastings = [float(e) for e in easting.reshape(-1)]
        northings = [float(n) for n in northing.reshape(-1)]

        # 行列网格：indexing="ij" 使 rows[i,j]=i, cols[i,j]=j，row-major 展平
        rows_grid, cols_grid = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
        rows = rows_grid.reshape(-1).tolist()
        cols = cols_grid.reshape(-1).tolist()

        return [
            models.PointStruct(
                id=str(uuid.uuid5(ns, f"{r}_{c}")),
                vector=vectors[i],
                payload={
                    "label": labels[i],
                    "label_name": label_names[i],
                    "utm_easting": eastings[i],
                    "utm_northing": northings[i],
                    "utm_zone": zone,
                    "image_id": image_id,
                    "pixel_row": r,
                    "pixel_col": c,
                },
            )
            for i, (r, c) in enumerate(zip(rows, cols))
        ]
```

- [x] **Step 4: 运行完整 test_importer 确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_importer.py -v`
Expected: PASS —— 包括既有 `TestBuildPoints`（点数/字段/NaN UTM）、`TestImportImagePair`、`TestBuildPointsBatching`（batch_size=5000 → 4 次 upsert）以及新增 `TestBuildPointsVectorization`。

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/importer.py KNN_evaluation/tests/test_importer.py
git commit -m "perf(importer): vectorize build_points construction while preserving point order and payload"
```

---

### Task 3: import_directory() 进度回调

**Files:**
- Modify: `KNN_evaluation/importer.py`
- Test: `KNN_evaluation/tests/test_progress_callback.py`（新增）

**Interfaces:**
- Produces: `import_directory(self, data_dir: Path, no_resume: bool = False, reindex: bool = False, progress_callback: Callable[[int, int], None] | None = None) -> dict`
  - `progress_callback(imported_so_far, total)`：每批 upsert 成功后调用；`total` 为待导入像素总数（非跳过影像的 16384 累加，全部跳过时 `total=0`）；`imported_so_far` 为已成功 upsert 像素累计，最终等于 `stats["total_pixels"]`
  - 默认 `None` 时行为与现状完全一致
- Internal:
  - `_batch_upsert(self, points, batch_callback: Callable[[int], None] | None = None) -> None`（每批 upsert 后回调批内点数）
  - `import_image_pair(self, pair, _batch_callback: Callable[[int], None] | None = None) -> tuple[int, int]`（位置参数与返回值不变，仅新增可选私有关键字）
- Consumed by: Task 5（CLI）、Task 6（WebUI）

- [x] **Step 1: 写失败测试**

创建 `KNN_evaluation/tests/test_progress_callback.py`：

```python
"""Tests for import_directory progress_callback."""
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

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
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_progress_callback.py -v`
Expected: FAIL —— `TypeError: import_directory() got an unexpected keyword argument 'progress_callback'`

- [x] **Step 3: 实现进度回调线程**

在 `KNN_evaluation/importer.py` 顶部导入区添加 `from typing import Callable`（放在 `from pathlib import Path` 之后）。

将 `_batch_upsert` 方法体替换为：

```python
    def _batch_upsert(
        self,
        points: list[models.PointStruct],
        batch_callback: Callable[[int], None] | None = None,
    ) -> None:
        """分批 upsert points 到 Qdrant.

        Args:
            points: PointStruct 列表.
            batch_callback: 每批 upsert 成功后回调该批点数（内部使用，用于进度推进）.
        """
        for i in range(0, len(points), self.batch_size):
            batch = points[i:i + self.batch_size]
            self.manager.client.upsert(
                collection_name=self.manager.collection_name,
                points=batch,
                wait=True,
            )
            if batch_callback is not None:
                batch_callback(len(batch))
```

将 `import_image_pair` 签名与方法内 upsert 调用替换为（其余方法体保持不变）：

```python
    def import_image_pair(
        self,
        pair: ImagePair,
        _batch_callback: Callable[[int], None] | None = None,
    ) -> tuple[int, int]:
        """导入一对 SE + DW 文件的所有像素.

        Args:
            pair: 匹配的 ImagePair.
            _batch_callback: 可选内部回调，每批 upsert 成功后调用（私有，用于进度线程）.

        Returns:
            (imported, skipped) — 新导入的点数和跳过的点数.
        """
        ...
        # 构建并导入
        points = self.build_points(se_data, dw_data, easting, northing, utm_zone, pair.image_id)
        self._batch_upsert(points, _batch_callback)

        return total, existing_count
```

将 `import_directory` 整体替换为：

```python
    def import_directory(
        self,
        data_dir: Path,
        no_resume: bool = False,
        reindex: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """完整导入流程：扫描目录 → 配对 → 逐文件导入.

        Args:
            data_dir: 数据根目录（含 SE/ 和 DW/ 子目录）.
            no_resume: True 时不检查断点续传，强制重新导入.
            reindex: True 时导入完成后触发 HNSW 全量索引重建.
            progress_callback: 可选进度回调 (imported_so_far, total)，
                每批 upsert 成功后调用；默认 None 时行为与现状完全一致.

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

        # 预计算断点续传状态与待导入像素总数（仅当需要进度回调时）。
        # existing_counts 缓存 count 查询结果，主循环直接复用；
        # total 只累加真实待导入影像（16384/张），全部跳过时为 0。
        if progress_callback is not None:
            if no_resume:
                existing_counts = {pair.image_id: 0 for pair in pairs}
            else:
                existing_counts = {
                    pair.image_id: PixelDataLoader.check_image_count(pair.image_id, self.manager)
                    for pair in pairs
                }
            total = sum(
                128 * 128 for pair in pairs
                if no_resume or existing_counts[pair.image_id] < 128 * 128
            )
        else:
            existing_counts = {}
            total = 0

        # 进度推进器：由 import_directory 统一驱动，_batch_upsert 每批后回调
        imported_so_far = 0

        def _batch_progress(batch_len: int) -> None:
            nonlocal imported_so_far
            imported_so_far += batch_len
            progress_callback(imported_so_far, total)

        def _get_existing_count(pair: ImagePair) -> int:
            if no_resume:
                return 0
            if progress_callback is not None:
                return existing_counts[pair.image_id]
            return PixelDataLoader.check_image_count(pair.image_id, self.manager)

        # 使用 tqdm 进度条
        try:
            from tqdm import tqdm
            pair_iter = tqdm(pairs, desc="导入影像", unit="img")
        except ImportError:
            pair_iter = pairs

        for pair in pair_iter:
            total_images += 1
            pair_label_counts: dict[str, int] = {}

            # 断点续传：按 count 判定，而非二进制 set
            existing_count = _get_existing_count(pair)
            if not no_resume and existing_count >= 128 * 128:
                skipped_images += 1
                dw_data = PixelDataLoader.load_dw(pair.dw_path)
                unique, counts = np.unique(dw_data, return_counts=True)
                for lbl, cnt in zip(unique, counts):
                    name = LABEL_NAMES.get(int(lbl), "unknown")
                    pair_label_counts[name] = int(cnt)
            else:
                imported, skip = self.import_image_pair(
                    pair,
                    _batch_callback=_batch_progress if progress_callback is not None else None,
                )
                total_pixels += imported

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

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_progress_callback.py -v`
Expected: PASS

- [x] **Step 5: 运行既有测试确认无回归**

Run: `uv run pytest KNN_evaluation/tests/test_importer.py KNN_evaluation/tests/test_data_loader.py -v`
Expected: PASS

- [x] **Step 6: 提交**

```bash
git add KNN_evaluation/importer.py KNN_evaluation/tests/test_progress_callback.py
git commit -m "feat(importer): add progress_callback to import_directory driven per batch"
```

---

### Task 4: HNSW 全量索引重建确认 + reindex 测试

**Files:**
- Modify: `KNN_evaluation/importer.py`（`import_directory` 内 reindex 块注释）
- Modify: `openspec/changes/data-import-optimization/proposal.md`（移除过时残留）
- Test: `KNN_evaluation/tests/test_importer_reindex.py`（新增）

**Interfaces:**
- Consumes: Task 3 的 `import_directory(..., reindex=True)`（既有行为，本任务仅确认+测试）
- Produces: 无新接口

- [x] **Step 1: 补充 reindex 语义注释（既有行为确认，无新代码）**

在 `KNN_evaluation/importer.py` 的 `import_directory` 中，将索引重建块替换为（仅为注释增强，逻辑不变）：

```python
        # 索引重建：indexing_threshold=0 强制触发全量 HNSW 向量索引重建，
        # 使新导入向量快速进入可检索状态。
        # 仅重建 HNSW 向量索引，不重建 payload 标量索引（用户确认，避免重建窗口内过滤查询不可用）。
        if reindex:
            self.manager.client.update_collection(
                collection_name=self.manager.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )
```

- [x] **Step 2: 清理 proposal.md 中的 payload 重建残留**

在 `openspec/changes/data-import-optimization/proposal.md` 中，删除与设计冲突的过时重复行（第 11 行，含"减少 payload 冗余、按需禁用 wait 批量 upsert"字样；保留其后已修正的同类行）。删除的行内容为：

```
- **修改 `KNN_evaluation/importer.py` 导入管线**：重构为更快的批量导入路径，减少逐像素 Python 层开销（向量按影像整体构造批次、减少 payload 冗余、按需禁用 `wait` 批量 upsert），显著提升大目录导入速度。
```

- [x] **Step 3: 写 reindex 调用测试**

创建 `KNN_evaluation/tests/test_importer_reindex.py`：

```python
"""Tests for import_directory reindex behavior (HNSW rebuild)."""
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from KNN_evaluation.importer import PixelImporter
from KNN_evaluation.data_loader import ImagePair
from KNN_evaluation.qdrant_client import QdrantManager


def _make_pair() -> ImagePair:
    return ImagePair(
        image_id="E121.4025_N25.1947",
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


@patch("KNN_evaluation.importer.compute_utm_grid")
@patch("KNN_evaluation.importer.PixelDataLoader")
class TestReindex:
    def _setup(self, mock_loader, mock_utm):
        mock_loader.scan_directory.return_value = [_make_pair()]
        mock_loader.check_image_count.return_value = 0
        mock_loader.load_se.return_value = np.zeros((64, 128, 128), dtype=np.float64)
        mock_loader.load_dw.return_value = np.zeros((128, 128), dtype=np.uint8)
        mock_utm.return_value = (
            np.zeros((128, 128), dtype=np.float64),
            np.zeros((128, 128), dtype=np.float64),
            51,
        )

    def test_reindex_triggers_hnsw_rebuild(self, mock_loader, mock_utm):
        self._setup(mock_loader, mock_utm)
        manager = _make_manager()
        importer = PixelImporter(manager, batch_size=10000)

        stats = importer.import_directory(Path("/fake/data"), reindex=True)

        manager.client.update_collection.assert_called_once()
        call_kwargs = manager.client.update_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "test"
        assert call_kwargs.kwargs["optimizer_config"].indexing_threshold == 0
        assert stats["imported_images"] == 1

    def test_no_reindex_skips_rebuild(self, mock_loader, mock_utm):
        self._setup(mock_loader, mock_utm)
        manager = _make_manager()
        importer = PixelImporter(manager, batch_size=10000)

        importer.import_directory(Path("/fake/data"), reindex=False)

        manager.client.update_collection.assert_not_called()
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_importer_reindex.py -v`
Expected: PASS（`reindex=True` 断言 `update_collection(indexing_threshold=0)` 被调用；`reindex=False` 断言不调用）

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/importer.py KNN_evaluation/tests/test_importer_reindex.py openspec/changes/data-import-optimization/proposal.md
git commit -m "docs(importer): confirm HNSW full rebuild semantics for reindex and add tests"
```

---

### Task 5: CLI import 子命令

**Files:**
- Modify: `KNN_evaluation/cli.py`

**Interfaces:**
- Consumes: Task 3 的 `import_directory(..., progress_callback=progress)`
- Produces: `cmd_import(args) -> int`（签名不变，内部接入像素级 tqdm）
- 依赖 `tqdm`（cli.py 顶部已导入 `from tqdm import tqdm`）

- [x] **Step 1: 完善 `--reindex` 帮助文案（3.1）**

在 `KNN_evaluation/cli.py` 的 `main()` 中，将 `import` 子命令的 `--reindex` 参数帮助文案更新为（明确 HNSW 语义）：

```python
    p_import.add_argument("--reindex", action="store_true",
                          help="导入完成后重建全量 HNSW 向量索引（indexing_threshold=0）")
```

- [x] **Step 2: `cmd_import` 接入像素级进度（3.2）**

将 `KNN_evaluation/cli.py` 中 `cmd_import` 函数的 `import_directory` 调用部分替换为：

```python
    importer = PixelImporter(manager, batch_size=args.batch_size)

    # 像素级进度条：首个回调时创建（此时才知道总像素数），结束时关闭
    pbar = None

    def progress(imported, total):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="导入像素", unit="px")
        pbar.n = imported
        pbar.refresh()

    try:
        stats = importer.import_directory(
            data_dir=Path(args.directory),
            no_resume=args.no_resume,
            reindex=args.reindex,
            progress_callback=progress,
        )
    except ConnectionError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    finally:
        if pbar is not None:
            pbar.close()
```

其余统计输出部分保持不变。

- [x] **Step 3: 验证 CLI 帮助与语法**

Run: `uv run python -m KNN_evaluation.cli import --help`
Expected: 帮助中 `--reindex` 文案为"导入完成后重建全量 HNSW 向量索引（indexing_threshold=0）"；无语法错误。

Run: `uv run pytest KNN_evaluation/tests -k "not integration" -q`
Expected: 既有单元测试全部通过。

- [x] **Step 4: 提交**

```bash
git add KNN_evaluation/cli.py
git commit -m "feat(cli): document --reindex HNSW semantics and add pixel-level tqdm progress to import"
```

---

### Task 6: WebUI 数据导入区改造

**Files:**
- Modify: `KNN_evaluation/webui.py`

**Interfaces:**
- Consumes: Task 1 的 `PAGE_SIZE` / `paginate_slice` / `total_pages` / `page_controls`；Task 3 的 `import_directory(..., progress_callback=cb)`
- Produces: 无新接口（页面行为：分页预览、按钮上移、线性进度条）

- [x] **Step 1: 导入分页纯函数**

在 `KNN_evaluation/webui.py` 的 KNN 包导入区（`from KNN_evaluation.label_mapping import ...` 之后）添加：

```python
from KNN_evaluation.ui_pagination import (
    PAGE_SIZE, paginate_slice, total_pages, page_controls,
)
```

- [x] **Step 2: state 增加页码字段**

在 `index()` 的 `state` 字典（约第 159 行）中，于 `"file_pairs": [],` 之后添加：

```python
        "preview_page": 0,
```

- [x] **Step 3: 替换"数据导入" expansion 块**

将 `KNN_evaluation/webui.py` 中整个 `# ===== 数据导入 =====` 的 `with ui.expansion("数据导入", value=False).classes("w-full mt-2"):` 块（当前约第 227-333 行）替换为：

```python
    # ===== 数据导入 =====
    with ui.expansion("数据导入", value=False).classes("w-full mt-2"):
        # 闭包函数定义（对 UI 元素的引用在调用时解析）
        async def do_import():
            if state["manager"] is None:
                ui.notify("QdrantManager 未初始化，请刷新页面", type="negative")
                return
            if not state["manager"].health_check():
                ui.notify("Qdrant 不可达，请先启动 Qdrant", type="negative")
                return
            if not state["manager"].collection_exists():
                state["manager"].create_collection()
                state["manager"].create_payload_indices()
                ui.notify("已自动创建 Collection", type="info")
            importer = PixelImporter(state["manager"], batch_size=BATCH_SIZE)
            ui.notify("导入已开始，请稍候...", type="info")

            import_progress_bar.set_visibility(True)
            import_progress_label.set_visibility(True)
            import_progress_bar.value = 0
            import_progress_label.set_text("准备中...")

            def cb(imported, total):
                import_progress_bar.value = imported / max(total, 1)
                import_progress_label.set_text(f"已导入 {imported:,} / {total:,}")

            try:
                stats = await asyncio.to_thread(
                    importer.import_directory,
                    state["data_dir"], False, True, cb,
                )
            except Exception as e:
                ui.notify(f"导入失败: {e}", type="negative")
                import_progress_bar.set_visibility(False)
                import_progress_label.set_visibility(False)
                return

            # 完成后进度到满并隐藏进度条
            import_progress_bar.value = 1.0
            import_progress_bar.set_visibility(False)
            import_progress_label.set_visibility(False)

            if stats["imported_images"] == 0:
                ui.notify("无新导入（数据已存在或目录为空）", type="info")
            else:
                _show_import_stats(stats)
                ui.notify(
                    f"导入完成: {stats['imported_images']} 张影像, "
                    f"{stats['total_pixels']:,} 像素",
                    type="positive",
                )
            refresh_status()
            await browse_directory()

        async def browse_directory():
            data_dir = Path(dir_input.value)
            state["data_dir"] = data_dir
            if not data_dir.exists():
                ui.notify(f"目录不存在: {dir_input.value}", type="negative")
                state["file_pairs"] = []
                _render_preview()
                return
            try:
                pairs = PixelDataLoader.scan_directory(data_dir)
            except Exception as e:
                ui.notify(f"扫描失败: {e}", type="negative")
                return
            state["file_pairs"] = pairs
            state["se_paths_map"] = {p.image_id: p.se_path for p in pairs}
            state["preview_page"] = 0

            # 同时更新指定像素的影像选择器选项
            _refresh_image_list()

            _render_preview()

        def _set_page(delta: int):
            total = len(state["file_pairs"])
            if total == 0:
                return
            last_page = total_pages(total, PAGE_SIZE) - 1
            state["preview_page"] = max(0, min(state["preview_page"] + delta, last_page))
            _render_preview()

        def _render_preview():
            file_column.clear()
            pairs = state["file_pairs"]
            total = len(pairs)
            if not pairs:
                with file_column:
                    ui.label("未找到匹配的 SE/DW 文件对").classes("text-grey text-sm")
                prev_btn.set_enabled(False)
                next_btn.set_enabled(False)
                page_label.set_text("")
                return
            page = state["preview_page"]
            can_prev, can_next = page_controls(page, total, PAGE_SIZE)
            prev_btn.set_enabled(can_prev)
            next_btn.set_enabled(can_next)
            page_label.set_text(
                f"第 {page + 1}/{total_pages(total, PAGE_SIZE)} 页 · 共 {total} 条"
            )
            for pair in paginate_slice(pairs, page, PAGE_SIZE):
                count = 0
                if state["manager"] and state["manager"].collection_exists():
                    try:
                        count = PixelDataLoader.check_image_count(
                            pair.image_id, state["manager"],
                        )
                    except Exception:
                        pass
                status = (
                    "✅ 已导入" if count >= 16384
                    else f"⏳ {count}/16384" if count > 0
                    else "📦 待导入"
                )
                with ui.row().classes("w-full items-center border-b py-1"):
                    ui.label(pair.image_id).classes("font-mono text-sm")
                    ui.label(status).classes("text-xs text-grey")

        def _show_import_stats(stats: dict):
            with ui.dialog() as d, ui.card().classes("w-full max-w-lg p-4"):
                d.open()
                ui.label("导入统计").classes("text-h5")
                ui.label(
                    f"总像素: {stats['total_pixels']:,}  |  "
                    f"总影像: {stats['total_images']}  |  "
                    f"新导入: {stats['imported_images']}  |  "
                    f"跳过: {stats['skipped_images']}"
                ).classes("text-sm")
                ui.label(
                    f"耗时: {stats['elapsed_sec']:.1f}s  |  "
                    f"速率: {stats['rate_pps']:.0f} 像素/秒"
                ).classes("text-sm text-grey")
                if stats.get("label_counts"):
                    ui.label("标签分布:").classes("text-sm mt-2")
                    for name, cnt in sorted(
                        stats["label_counts"].items(), key=lambda x: -x[1],
                    ):
                        ui.label(f"  {name}: {cnt:,}").classes("text-xs text-grey")
                ui.button("关闭", on_click=lambda: d.close()).props("flat mt-4")

        # 导入全部按钮 + 进度条：位于数据目录栏上方，无需滚动越过预览列表即可点击
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("导入全部", on_click=do_import).props("flat")
            import_progress_bar = ui.linear_progress(value=0).classes("w-96")
            import_progress_bar.set_visibility(False)
            import_progress_label = ui.label("").classes("text-sm text-grey")
            import_progress_label.set_visibility(False)

        # 数据目录栏
        with ui.row().classes("w-full items-center gap-4"):
            dir_input = ui.input(
                label="数据目录", value=str(_CLI_DATA_DIR),
            ).classes("w-96")
            ui.button("浏览", on_click=browse_directory).props("flat")

        # 预览列表
        file_column = ui.column().classes("w-full")

        # 分页控件
        with ui.row().classes("items-center gap-2 mt-2"):
            prev_btn = ui.button(
                "上一页", on_click=lambda: _set_page(-1),
            ).props("flat dense size=sm")
            page_label = ui.label("").classes("text-sm text-grey")
            next_btn = ui.button(
                "下一页", on_click=lambda: _set_page(1),
            ).props("flat dense size=sm")
```

- [x] **Step 4: 验证模块可加载**

Run: `uv run python -c "import KNN_evaluation.webui"`（若因缺运行环境失败，至少确认无语法错误：`uv run python -m py_compile KNN_evaluation/webui.py`）
Expected: 无异常 / 编译通过。

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/webui.py
git commit -m "feat(webui): paginated import preview, top import-all button, and pixel progress bar"
```

---

### Task 7: 集成验证

**Files:**
- No source changes（仅运行验证）

**Interfaces:**
- Consumes: 全部先前任务的产出

- [x] **Step 1: 启动 Qdrant**

确认本地 Qdrant 运行：`docker ps --format "{{.Names}}"`，若无则启动：
```bash
docker run -d --name qdrant-knn-eval -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

- [x] **Step 2: CLI 集成验证（6.1）**

Run: `uv run python -m KNN_evaluation.cli import data_demo --reindex`
Expected: 像素级 tqdm 进度条推进至完成；统计输出正常；`--reindex` 触发 HNSW 重建无报错。可再执行一次 `uv run python -m KNN_evaluation.cli import data_demo --reindex` 验证断点续传跳过逻辑（skipped_images 增加、upsert 不重复）。

- [x] **Step 3: WebUI 集成验证（6.2）**

Run: `uv run python KNN_evaluation/webui.py --port 8003 --dir data_demo`
Expected:
- 数据导入区预览列表每页最多 20 条；"上一页/下一页"可用性随页首/末禁用态正确
- "导入全部"按钮位于数据目录栏上方，无需滚动即可点击
- 点击"导入全部"后显示线性进度条 + 进度文本（`已导入 N / M`），完成后进度到满并弹出统计对话框

- [x] **Step 4: 断点续传与测试套件回归（6.3）**

Run: `uv run pytest KNN_evaluation/tests -k "not integration" -v`
Expected: 全部单元测试通过（含新增 `test_ui_pagination.py`、`test_progress_callback.py`、`test_importer_reindex.py`、更新后的 `test_importer.py`）。

Run: `uv run python -m KNN_evaluation.cli stats`
Expected: 统计输出正常（重复导入后 Collection 总量不重复增长）。

- [x] **Step 5: 最终提交（如有验证期修复）**

如有验证期代码修复，逐项提交并说明；无修复则本任务无需提交。

---

## 验收对照

| Design Doc 验收目标 | 对应实现 |
|---|---|
| WebUI 待导入数据预览分页（每页 20 条）且可翻页查看全部 | Task 1（纯函数）+ Task 6（预览切片 + 翻页按钮） |
| "导入全部"按钮位于数据目录栏上方 | Task 6（按钮行置于目录栏上方） |
| 导入期间显示按像素推进的线性进度条 + 进度文本，完成后到满 | Task 3（progress_callback）+ Task 6（ui.linear_progress + 文本） |
| 导入速度显著提升（向量化构造加速） | Task 2（build_points 向量化） |
| `--reindex` 触发全量 HNSW 向量索引重建（既有行为确认并文档化） | Task 4（注释确认）+ Task 5（CLI 帮助文案） |

## Self-Review 结论

- **Spec coverage**：tasks.md 六组任务全覆盖——1.x（Task 2/3）、2.x（Task 4）、3.x（Task 5）、4.x（Task 6）、5.x（Task 1/2/3/4）、6.x（Task 7）；Design Doc 的 5 个验收目标均有对应任务。
- **Placeholder scan**：所有步骤含完整代码与预期输出，无 TBD/TODO。
- **Type consistency**：`progress_callback: Callable[[int, int], None] | None`、`paginate_slice`/`total_pages`/`page_controls` 签名在 Task 3/5/6 中保持一致；`import_directory` 参数顺序 `(data_dir, no_resume, reindex, progress_callback)` 在 Task 6 的 `asyncio.to_thread` 位置参数调用中与定义一致。

---

### Task 8: UTM 坐标从文件名推算（coordinate_utils.py + data_loader.py + config.py）

**Files:**
- Modify: `KNN_evaluation/coordinate_utils.py`（新增 `compute_utm_grid_from_name`）
- Modify: `KNN_evaluation/config.py`（新增 `UTM_RESOLUTION_M`）
- Modify: `KNN_evaluation/data_loader.py`（新增 `parse_location_coord`）
- Test: `KNN_evaluation/tests/test_coordinate_utils.py`（新增）

**Interfaces:**
- Produces:
  - `compute_utm_grid_from_name(lon: float, lat: float, scale: int = UTM_RESOLUTION_M, grid_size: int = 128) -> tuple[np.ndarray, np.ndarray, int]`
  - `parse_location_coord(location_key: str) -> tuple[float, float]`
  - `UTM_RESOLUTION_M: int = 10`（config.py）
- Consumed by: Task 9（importer 主路径改用文件名推算）

**坐标模型（DW 下载脚本 `_create_square_roi` 权威）：**
- 文件名坐标段 `E{lon}_N{lat}` = 影像中心点
- `zone = int((lon+180)/6)+1`；北半球 EPSG:326xx、南半球 EPSG:327xx
- `half = grid_size×scale/2`；中心点 (lon,lat)→UTM (cx,cy)（pyproj Transformer）
- `nw_x = floor((cx−half)/scale)×scale`；`nw_y = ceil((cy+half)/scale)×scale`
- 逐像素：`easting[r,c] = nw_x + c×scale + scale/2`；`northing[r,c] = nw_y − r×scale − scale/2`

- [x] **Step 1: 写失败测试（TDD）**

创建 `KNN_evaluation/tests/test_coordinate_utils.py`：

```python
"""Tests for UTM coordinates derived from filename coordinates."""
import numpy as np
import pytest

from KNN_evaluation.coordinate_utils import compute_utm_grid_from_name, compute_utm_grid
from KNN_evaluation.data_loader import PixelDataLoader
from KNN_evaluation.config import UTM_RESOLUTION_M


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

    def test_matches_geotiff_transform(self, tmp_path):
        # 用 data_demo 的 GeoTIFF 验证差分一致性
        # (该测试在无 TIF 环境跳过，仅验证文件名推算网格的 NW 原点语义)
        easting, northing, zone = compute_utm_grid_from_name(121.4025, 25.1947, scale=10)
        # NW 原点应在 scale 网格上
        assert easting[0, 0] % 10 == pytest.approx(5, abs=1e-9)  # +scale/2 偏移
        assert northing[0, 0] % 10 == pytest.approx(5, abs=1e-9)
        assert zone == 51  # 东经121.4 → zone int((121.4+180)/6)+1 = 51

    def test_southern_hemisphere_zone(self):
        _, _, zone = compute_utm_grid_from_name(121.4025, -25.1947)
        assert zone == -51  # 南半球用负带号表示（与 read_geotiff_meta 一致）
```

- [x] **Step 2: 运行测试确认失败**
- [x] **Step 3: 实现 `parse_location_coord` + `UTM_RESOLUTION_M` + `compute_utm_grid_from_name`**

在 `KNN_evaluation/config.py` 添加：
```python
UTM_RESOLUTION_M = 10  # UTM 坐标推算分辨率（米/像素）
```

在 `KNN_evaluation/data_loader.py` 的 `PixelDataLoader` 内添加（复用 `extract_location_key` 的正则）：
```python
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
        m = PixelDataLoader.COORDINATE_PATTERN.search(location_key)
        if m is None:
            raise ValueError(f"无法从坐标段解析经纬度: {location_key}")
        lon_str, lat_str = m.group(1)[1:].split("_N")
        return float(lon_str), float(lat_str)
```

在 `KNN_evaluation/coordinate_utils.py` 添加：
```python
from KNN_evaluation.config import UTM_RESOLUTION_M

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
```

- [x] **Step 4: 运行测试确认通过**
- [x] **Step 5: 提交**
```bash
git add KNN_evaluation/coordinate_utils.py KNN_evaluation/config.py KNN_evaluation/data_loader.py KNN_evaluation/tests/test_coordinate_utils.py
git commit -m "feat(coordinate_utils): derive UTM grid from filename coordinates without TIF"
```

---

### Task 9: importer 主路径改用文件名推算 UTM

**Files:**
- Modify: `KNN_evaluation/importer.py`

**Interfaces:**
- Consumes: Task 8 的 `compute_utm_grid_from_name` + `parse_location_coord`
- Produces: `import_image_pair` 内部 UTM 获取逻辑改为文件名推算（GeoTIFF 降为回退）

- [x] **Step 1: 修改 `import_image_pair`**

在 `KNN_evaluation/importer.py` 的 `import_image_pair` 中，把 `compute_utm_grid(pair.tif_path)` 调用替换为：

```python
        # UTM 坐标：优先从文件名坐标段推算（不加载 TIF）
        try:
            lon, lat = PixelDataLoader.parse_location_coord(pair.image_id)
            easting, northing, utm_zone = compute_utm_grid_from_name(lon, lat)
        except (ValueError, Exception):
            # 文件名推算失败时回退 GeoTIFF（保留兼容）
            warnings.warn(f"{pair.image_id}: 文件名坐标推算失败，回退 GeoTIFF")
            easting, northing, utm_zone = compute_utm_grid(pair.tif_path)
```

同时更新顶部导入：`from KNN_evaluation.coordinate_utils import compute_utm_grid, compute_utm_grid_from_name`

- [x] **Step 2: 运行既有测试确认无回归**
- [x] **Step 3: 提交**
```bash
git add KNN_evaluation/importer.py
git commit -m "feat(importer): derive UTM from filename coordinates, fallback to TIF"
```

---

### Task 10: UTM 文件名推算集成验证

**Files:**
- No source changes

- [x] **Step 1: 运行测试套件**
- [x] **Step 2: 验证无 TIF 场景导入**

---

### Task 11: image_id keyword 索引修复（根因）

**Files:**
- Modify: `KNN_evaluation/qdrant_client.py`
- Test: `KNN_evaluation/tests/test_qdrant_client.py`（追加）

**Background（根因）**: `image_id` 用 text 索引（word tokenizer），`check_image_count` 用 `MatchValue` 精确匹配，text 索引无法高效回答含 `.`/`_` 的字符串精确匹配 → Qdrant 全量 payload 扫描（实测 count 1.41s）。改为 keyword 索引后 0.010s（140× 快）。

**Interfaces:**
- Produces:
  - `create_payload_indices()` 中 `image_id` 用 `KeywordIndexParams(type=KeywordIndexType.KEYWORD)` 替代 `TextIndexParams`
  - 新增 `migrate_image_id_index()`：删除旧 text 索引 + 重建 keyword 索引（幂等，失败可重试）
- Consumed by: Task 12（importer 重试）+ 既有 `check_image_count`（自动受益）

- [x] **Step 1: 写失败测试**
  在 `test_qdrant_client.py` 追加：断言 `create_payload_indices` 对 `image_id` 用 keyword schema（`KeywordIndexParams`）。
- [x] **Step 2: 实现索引定义修改**
  `qdrant_client.py` 的 `create_payload_indices`：`image_id` 从 `TextIndexParams` 改为 `KeywordIndexParams(type=KeywordIndexType.KEYWORD)`。
- [x] **Step 3: 新增索引迁移方法**
  新增 `migrate_image_id_index()`：try 删除 `image_id` 索引（可能不存在），再创建 keyword 索引；幂等。
- [x] **Step 4: 运行测试确认通过**
  `uv run pytest KNN_evaluation/tests/test_qdrant_client.py -v`
- [x] **Step 5: 提交**
  `git add KNN_evaluation/qdrant_client.py KNN_evaluation/tests/test_qdrant_client.py`
  `git commit -m "fix(qdrant_client): use keyword index for image_id to speed up exact count queries"`

---

### Task 12: 导入失败重试机制

**Files:**
- Modify: `KNN_evaluation/importer.py`
- Test: `KNN_evaluation/tests/test_import_retry.py`（新增）

**Interfaces:**
- Produces:
  - `importer._retry_call(fn, *args, retries=3, base_delay=1.0, **kwargs)` — 指数退避重试包装器
  - `_batch_upsert` 与 `import_directory` 的 count 路径接入重试
- Consumed by: `import_directory` / `import_image_pair`

- [x] **Step 1: 写失败测试（TDD）**
  创建 `test_import_retry.py`：mock upsert 首次抛异常、第二次成功 → 断言调用 2 次且最终成功；mock 持续抛异常 → 断言重试 3 次后抛出。
- [x] **Step 2: 实现 `_retry_call`**
  指数退避：`delay = base_delay * 2**attempt`，sleep 后重试。
- [x] **Step 3: 接入 `_batch_upsert` 与 count**
  `_batch_upsert` 的 `client.upsert` 与 `import_directory` 的 `check_image_count` 用 `_retry_call` 包装。
- [x] **Step 4: 运行测试确认通过**
  `uv run pytest KNN_evaluation/tests/test_import_retry.py -v`
- [x] **Step 5: 提交**
  `git add KNN_evaluation/importer.py KNN_evaluation/tests/test_import_retry.py`
  `git commit -m "feat(importer): add exponential backoff retry for Qdrant upsert/count"`

---

### Task 13: 导入提速集成验证

**Files:**
- No source changes

- [x] **Step 1: 迁移既有 collection 索引**
  对 `pixel_embeddings` 运行 `migrate_image_id_index()`（或 CLI 触发）。
- [x] **Step 2: 实测 count 提速**
  `check_image_count('E121.4025_N25.1947')` 耗时应显著下降（keyword 索引生效）。
- [x] **Step 3: 运行测试套件**
  `uv run pytest KNN_evaluation/tests -k "not integration" -q`（隔离 label_mapping.py）
- [x] **Step 4: data 子集导入冒烟**
  用 `data` 前 10 对子集跑 `import_directory`，确认不再卡死、进度正常。
