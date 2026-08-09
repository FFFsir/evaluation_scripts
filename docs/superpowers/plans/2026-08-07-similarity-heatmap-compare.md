---
change: similarity-heatmap-compare
design-doc: docs/superpowers/specs/2026-08-07-similarity-heatmap-compare-design.md
base-ref: 5f2209b23044d869b05a8982591e8d1d4899d2a7
archived-with: 2026-08-08-similarity-heatmap-compare
---

# 双集合 Embedding 相似度热力图对比 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 CLI 与 WebUI 双入口提供"双集合（`google_aef_embedding` × `xian_aef_embedding`）embedding 相似度热力图对比"能力：随机采样 N 个点，按同一批 point_id 从两个集合分别取 embedding，各自计算 N×N 余弦相似度矩阵，并排（1×2）统一色阶输出 PNG。

**Architecture:** 单一核心模块 `KNN_evaluation/similarity_compare.py` 承载采样、提取、矩阵、编排（CLI/WebUI 复用，行为一致，与 `metrics.py` 抽公共层同模式）；渲染进 `visualization.py`（复用既有 Agg 后端与中文字体配置）；CLI 新增 `similarity-heatmap` 子命令，WebUI 新增「相似度热力图对比」expansion（固定预置对）。数据库模式复用 `sampling_map.ensure_sampling_map` 候选池；图片模式按 `importer.build_points` 完全一致的 uuid5 公式换算 point_id。

**Tech Stack:** Python 3.12+ / argparse / numpy / matplotlib（Agg）/ qdrant-client / pytest / uv

## Global Constraints

- 全部注释与用户可见输出使用中文（与现有代码一致）。
- **不引入新第三方依赖**：只复用 numpy / matplotlib / qdrant-client / pytest / NiceGUI（Deep Design Non-Goals）。
- N 默认 200、上限 600；越界抛 `ValueError`（CLI 层用 `argparse.ArgumentTypeError`），模块函数层与 parser 层双保险（D6）。
- seed 默认 42，`random.Random(seed)` 保证可复现。
- 采样点 point_id 换算必须与 `importer.build_points` 完全一致：`uuid5(uuid5(uuid.NAMESPACE_DNS, image_id), f"{row}_{col}")`（D2 / `importer.py:106,129`）。
- 双集合按**同一 ids 列表**批量 `client.retrieve(with_payload=True, with_vectors=True)`；单侧缺失点剔除，行序 = 采样顺序的保留子序列；`dropped` 统计透出（D3）。
- 提取阶段统一以 `retrieve` 返回的 payload 为 UTM 数据源，采样点 dict 不携带 UTM 坐标（D2 决策）。
- 余弦度量与 Qdrant COSINE 一致：归一化后 `V @ V.T`（D4）。
- WebUI 固定对比预置对 `PRESET_COLLECTIONS[0] × [1]`，**与 `_current_collection` 选择器无关**（D7，用户确认）。
- 不修改 manifest / 采样地图 / corpus 缓存文件格式；不修改 F1/F2 评估、导入、检索、迁移流程（Non-Goals）。
- 全量测试命令：`uv run pytest KNN_evaluation/tests/ -v`（Windows 平台，Git Bash）。
- base-ref 为 `5f2209b23044d869b05a8982591e8d1d4899d2a7`；每个任务结束必须 commit。

## 文件结构

| 文件 | 职责 | 操作 |
|------|------|------|
| `KNN_evaluation/similarity_compare.py` | 采样 / 提取 / 余弦矩阵 / 编排（单一事实源） | 新建（Task 1/2/4 分步写入） |
| `KNN_evaluation/visualization.py` | 新增 `plot_similarity_heatmap_pair` 并排热力图 | 修改（Task 3） |
| `KNN_evaluation/cli.py` | 新增 `similarity-heatmap` parser 与 `cmd_similarity_heatmap` | 修改（Task 5） |
| `KNN_evaluation/webui.py` | 新增「相似度热力图对比」面板（异步 `asyncio.to_thread`，base64 内嵌展示） | 修改（Task 6） |
| `KNN_evaluation/tests/test_similarity_compare.py` | 核心模块 + 可视化单元测试（fake manager 模式） | 新建（Task 1/2/3/4） |
| `KNN_evaluation/tests/test_cli.py` | `similarity-heatmap` 子命令测试 | 修改（Task 5） |
| `KNN_evaluation/tests/test_webui.py` | WebUI 面板测试（`_FakeUI` 扩展 radio/image） | 修改（Task 6） |
| `KNN_evaluation/README.md` + 根 `README.md` | CLI 用法 + WebUI 面板说明 | 修改（Task 7） |

**任务与 openspec tasks.md 六组任务的对应关系：**
- Task 1 ↔ 组 1（1.1 采样 / 1.2 校验）+ 组 5 测试 5.1
- Task 2 ↔ 组 1（1.3 提取 / 1.4 矩阵）+ 组 5 测试 5.2、5.3 矩阵部分
- Task 3 ↔ 组 2（2.1 可视化）+ 组 5 测试 5.3 绘图部分
- Task 4 ↔ 组 1（1.5 编排函数）
- Task 5 ↔ 组 3（3.1 / 3.2 CLI）+ 组 5 测试 5.4
- Task 6 ↔ 组 4（4.1 / 4.2 WebUI）+ 组 5 测试 5.5
- Task 7 ↔ 组 6（6.1 文档）

---

### Task 1: `similarity_compare.py` — 随机采样（数据库 / 图片模式）

**Files:**
- Create: `KNN_evaluation/similarity_compare.py`
- Create: `KNN_evaluation/tests/test_similarity_compare.py`

**Interfaces:**
- Consumes: `sampling_map.ensure_sampling_map(manager)`（返回 `{"by_label": {label_id: [point_id...]}}`）；`data_loader.PixelDataLoader.check_image_count(image_id, manager)`（图片模式存在性校验）。
- Produces: `similarity_compare.MAX_N = 600`、`IMAGE_GRID = 128`、`IMAGE_CELLS = 16384`、`_validate_n(n)`、`_point_id(image_id, row, col) -> str`、`sample_random_points(manager, n, seed, image_id=None) -> list[dict]`（数据库模式每项 `{"point_id": pid}`；图片模式每项 `{"point_id", "image_id", "pixel_row", "pixel_col"}`）。

- [x] **Step 1: 写失败测试**

创建 `KNN_evaluation/tests/test_similarity_compare.py`：

```python
"""Tests for KNN_evaluation.similarity_compare（双集合相似度热力图对比核心模块）."""
import io
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation import similarity_compare as SC


def _manager(collection="test_collection", total_points=0):
    manager = MagicMock(spec=QdrantManager)
    manager.collection_name = collection
    manager.url = "http://localhost:6333"
    manager.collection_exists.return_value = True
    manager.collection_info.return_value = {"total_points": total_points}
    return manager


def _by_label_map(by_label: dict, total_points: int) -> dict:
    return {
        "collection": "test_collection", "total_points": total_points,
        "updated_at": "", "by_label": dict(by_label),
    }


class TestSampleRandomPoints:
    def test_db_mode_uses_map_and_seed_reproducible(self, monkeypatch):
        by_label = {0: [f"pt-{i}" for i in range(50)], 1: [f"pt-{i + 50}" for i in range(50)]}
        mgr = _manager(total_points=100)
        monkeypatch.setattr(SC, "ensure_sampling_map",
                            lambda m, path=None: _by_label_map(by_label, 100))
        pts1 = SC.sample_random_points(mgr, 20, seed=42)
        pts2 = SC.sample_random_points(mgr, 20, seed=42)
        assert [p["point_id"] for p in pts1] == [p["point_id"] for p in pts2]
        assert len(pts1) == 20
        all_ids = {pid for ids in by_label.values() for pid in ids}
        assert all(p["point_id"] in all_ids for p in pts1)
        assert pts1 == [{"point_id": p["point_id"]} for p in pts1]

    def test_n_out_of_range_raises(self):
        mgr = _manager(total_points=100)
        for bad in (0, -1, 601):
            with pytest.raises(ValueError, match="n 必须在 1..600"):
                SC.sample_random_points(mgr, bad, seed=42)

    def test_insufficient_candidates_sampled_actual(self, monkeypatch):
        by_label = {0: [f"pt-{i}" for i in range(5)]}
        mgr = _manager(total_points=5)
        monkeypatch.setattr(SC, "ensure_sampling_map",
                            lambda m, path=None: _by_label_map(by_label, 5))
        pts = SC.sample_random_points(mgr, 200, seed=1)
        assert len(pts) == 5

    def test_empty_collection_raises(self):
        mgr = _manager(total_points=0)
        with pytest.raises(ValueError, match="为空"):
            SC.sample_random_points(mgr, 10, seed=42)

    def test_by_label_empty_collection_nonempty_raises(self, monkeypatch):
        mgr = _manager(total_points=100)
        monkeypatch.setattr(SC, "ensure_sampling_map",
                            lambda m, path=None: _by_label_map({}, 100))
        with pytest.raises(RuntimeError, match="采样地图为空"):
            SC.sample_random_points(mgr, 10, seed=42)

    def test_image_mode_point_ids_deterministic(self, monkeypatch):
        mgr = _manager(total_points=16384)
        monkeypatch.setattr(
            "KNN_evaluation.data_loader.PixelDataLoader.check_image_count",
            lambda iid, m: 16384,
        )
        pts1 = SC.sample_random_points(mgr, 5, seed=7, image_id="E121.4_N25.1")
        pts2 = SC.sample_random_points(mgr, 5, seed=7, image_id="E121.4_N25.1")
        assert pts1 == pts2
        assert len(pts1) == 5
        ns = uuid.uuid5(uuid.NAMESPACE_DNS, "E121.4_N25.1")
        for p in pts1:
            assert p["point_id"] == str(uuid.uuid5(ns, f"{p['pixel_row']}_{p['pixel_col']}"))
            assert p["image_id"] == "E121.4_N25.1"

    def test_image_mode_rows_cols_unique(self, monkeypatch):
        mgr = _manager(total_points=16384)
        monkeypatch.setattr(
            "KNN_evaluation.data_loader.PixelDataLoader.check_image_count",
            lambda iid, m: 16384,
        )
        pts = SC.sample_random_points(mgr, 100, seed=3, image_id="IMG")
        cells = {(p["pixel_row"], p["pixel_col"]) for p in pts}
        assert len(cells) == 100

    def test_image_mode_unknown_image_raises(self, monkeypatch):
        mgr = _manager(total_points=16384)
        monkeypatch.setattr(
            "KNN_evaluation.data_loader.PixelDataLoader.check_image_count",
            lambda iid, m: 0,
        )
        with pytest.raises(ValueError, match="不存在"):
            SC.sample_random_points(mgr, 5, seed=1, image_id="NO_SUCH")
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'KNN_evaluation.similarity_compare'`

- [x] **Step 3: 写最小实现**

创建 `KNN_evaluation/similarity_compare.py`：

```python
"""双集合 embedding 相似度热力图对比 — 核心计算模块.

CLI 与 WebUI 共用单一事实源：采样 → 双集合批量提取 → 余弦相似度矩阵 →
并排热力图渲染。与 `metrics.py` 抽公共层的模式一致（CLI/WebUI 复用）。

采样模式：
- 数据库模式：复用 `sampling_map.ensure_sampling_map` 全库 point_id 候选池随机抽取；
- 图片模式：在指定 image_id 的 128×128 网格内随机抽不重复 (row, col) 像素，
  point_id 用与 `importer.build_points` 完全一致的 uuid5 公式换算。
"""
import random
import uuid

import numpy as np

from KNN_evaluation.sampling_map import ensure_sampling_map

MAX_N = 600
IMAGE_GRID = 128
IMAGE_CELLS = IMAGE_GRID * IMAGE_GRID  # 16384


def _validate_n(n: int) -> None:
    """校验采样数 N（1..600，Deep Design D6 双保险之一）."""
    if not (1 <= int(n) <= MAX_N):
        raise ValueError(f"n 必须在 1..{MAX_N} 之间，实际: {n}")


def _point_id(image_id: str, row: int, col: int) -> str:
    """与 importer.build_points 完全一致的 point_id 换算公式（importer.py:106,129）."""
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, image_id)
    return str(uuid.uuid5(ns, f"{row}_{col}"))


def sample_random_points(
    manager,
    n: int,
    seed: int,
    image_id: str | None = None,
) -> list[dict]:
    """随机采样 N 个点（数据库全库模式 / 单张图片模式）.

    Args:
        manager: google 集合的 QdrantManager（作为采样侧；两集合 point_id 集合一致）。
        n: 目标采样数（1..600）。
        seed: 随机种子（random.Random(seed)，保证可复现）。
        image_id: None → 数据库模式（采样地图全库候选池）；
                  非 None → 图片模式（该 image_id 的 128×128 网格内随机不重复像素）。

    Returns:
        点字典列表；数据库模式每项 {point_id}；图片模式每项
        {point_id, image_id, pixel_row, pixel_col}。UTM 坐标不在采样阶段携带，
        提取阶段统一以 retrieve 返回的 payload 为准（Deep Design D2）。

    Raises:
        ValueError: n 越界；collection 不存在/为空；image_id 在 google 集合不存在。
        RuntimeError: 采样地图构建失败（by_label 全空但集合非空）。
    """
    _validate_n(n)
    rng = random.Random(seed)

    if image_id is None:
        # 数据库模式：复用采样地图候选池（按 label 合并即全库 point_id 池）
        if not manager.collection_exists():
            raise ValueError(
                f"Collection '{manager.collection_name}' 不存在，请先执行 import"
            )
        info = manager.collection_info()
        if info.get("total_points", 0) == 0:
            raise ValueError("Collection 为空，请先导入数据")
        sampling_map = ensure_sampling_map(manager)
        by_label = sampling_map.get("by_label") or {}
        candidates = [pid for ids in by_label.values() for pid in ids]
        if not candidates and info.get("total_points", 0) > 0:
            raise RuntimeError(
                "采样地图为空且构建失败：无法从 Qdrant 读取 point_id→label 地图，"
                f"请检查 collection '{manager.collection_name}' 与本地 qdrant_sampling_map.json"
            )
        n_actual = min(n, len(candidates))
        picked = rng.sample(candidates, n_actual)
        return [{"point_id": pid} for pid in picked]

    # 图片模式：128×128 网格内随机抽不重复像素
    from KNN_evaluation.data_loader import PixelDataLoader  # 函数内 import 防循环
    if PixelDataLoader.check_image_count(image_id, manager) <= 0:
        raise ValueError(
            f"image_id '{image_id}' 在 collection '{manager.collection_name}' 中不存在"
        )
    n_actual = min(n, IMAGE_CELLS)
    cells = rng.sample(
        [(r, c) for r in range(IMAGE_GRID) for c in range(IMAGE_GRID)],
        n_actual,
    )
    return [
        {
            "point_id": _point_id(image_id, r, c),
            "image_id": image_id,
            "pixel_row": r,
            "pixel_col": c,
        }
        for r, c in cells
    ]
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py -v`
Expected: PASS（7 个采样相关测试）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/similarity_compare.py KNN_evaluation/tests/test_similarity_compare.py
git commit -m "feat: add similarity_compare sample_random_points (db/image modes)"
```

---

### Task 2: `similarity_compare.py` — 批量提取与余弦相似度矩阵

**Files:**
- Modify: `KNN_evaluation/similarity_compare.py`（追加两个函数）
- Modify: `KNN_evaluation/tests/test_similarity_compare.py`（追加测试类）

**Interfaces:**
- Consumes: Task 1 的 `sample_random_points` 返回的点 dict（只读 `point_id` 字段）；两个 `QdrantManager.client.retrieve`。
- Produces: `extract_embeddings(points, google_manager, xian_manager) -> tuple[np.ndarray, np.ndarray, int]`（`(mat_g, mat_x, dropped)`，矩阵 shape `(N', 64)`，行序 = 采样顺序保留子序列）；`cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray`（`(N', N')`，对角 1.0）。Task 4 编排函数与 Task 5 CLI / Task 6 WebUI 依赖。

- [x] **Step 1: 写失败测试**

追加到 `KNN_evaluation/tests/test_similarity_compare.py` 末尾：

```python
class TestExtractEmbeddings:
    @staticmethod
    def _record(pid: str, value: float):
        rec = MagicMock()
        rec.id = pid
        rec.vector = [value] * 64
        rec.payload = {}
        return rec

    def test_aligned_matrices_zero_dropped(self):
        g, x = _manager("g"), _manager("x")
        ids = ["a", "b", "c"]
        g.client.retrieve.return_value = [
            self._record(i, float(n)) for n, i in enumerate(ids)
        ]
        x.client.retrieve.return_value = [
            self._record(i, float(n) * 10) for n, i in enumerate(ids)
        ]
        mat_g, mat_x, dropped = SC.extract_embeddings(
            [{"point_id": i} for i in ids], g, x,
        )
        assert dropped == 0
        assert mat_g.shape == (3, 64) and mat_x.shape == (3, 64)
        np.testing.assert_allclose(mat_g[:, 0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(mat_x[:, 0], [0.0, 10.0, 20.0])

    def test_retrieve_called_with_same_ids_and_vectors(self):
        g, x = _manager("g"), _manager("x")
        ids = ["a", "b"]
        g.client.retrieve.return_value = [self._record(i, 1.0) for i in ids]
        x.client.retrieve.return_value = [self._record(i, 2.0) for i in ids]
        SC.extract_embeddings([{"point_id": i} for i in ids], g, x)
        g.client.retrieve.assert_called_once_with(
            collection_name="g", ids=["a", "b"],
            with_payload=True, with_vectors=True,
        )
        x.client.retrieve.assert_called_once_with(
            collection_name="x", ids=["a", "b"],
            with_payload=True, with_vectors=True,
        )

    def test_single_side_missing_dropped_and_row_alignment(self):
        g, x = _manager("g"), _manager("x")
        g.client.retrieve.return_value = [
            self._record(i, float(n)) for n, i in enumerate(("a", "b", "c"))
        ]
        x.client.retrieve.return_value = [
            self._record(i, float(n) * 10)
            for n, i in enumerate(("a", "b", "c")) if i != "b"
        ]
        mat_g, mat_x, dropped = SC.extract_embeddings(
            [{"point_id": i} for i in ("a", "b", "c")], g, x,
        )
        assert dropped == 1
        assert mat_g.shape == (2, 64) and mat_x.shape == (2, 64)
        # 行序 = ids 原始顺序的保留子序列 (a, c)
        np.testing.assert_allclose(mat_g[:, 0], [0.0, 2.0])
        np.testing.assert_allclose(mat_x[:, 0], [0.0, 20.0])

    def test_all_missing_raises(self):
        g, x = _manager("g"), _manager("x")
        g.client.retrieve.return_value = [self._record("a", 1.0)]
        x.client.retrieve.return_value = []  # xian 侧全缺
        with pytest.raises(RuntimeError, match="无任何对齐点"):
            SC.extract_embeddings([{"point_id": "a"}], g, x)

    def test_wrong_dimension_raises(self):
        g, x = _manager("g"), _manager("x")
        rec = MagicMock()
        rec.id = "a"
        rec.vector = [1.0] * 32  # 非 64 维
        rec.payload = {}
        g.client.retrieve.return_value = [rec]
        x.client.retrieve.return_value = [rec]
        with pytest.raises(ValueError, match="维度应为 64"):
            SC.extract_embeddings([{"point_id": "a"}], g, x)


class TestCosineSimilarityMatrix:
    def test_symmetric_diagonal_one_range(self):
        rng = np.random.default_rng(0)
        v = rng.normal(0, 1, (10, 64))
        m = SC.cosine_similarity_matrix(v)
        assert m.shape == (10, 10)
        np.testing.assert_allclose(m, m.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-12)
        assert m.min() >= -1.0 - 1e-9 and m.max() <= 1.0 + 1e-9

    def test_zero_vector_defense(self):
        v = np.zeros((2, 64))
        m = SC.cosine_similarity_matrix(v)
        assert np.isfinite(m).all()
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-12)
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py -v`
Expected: FAIL — `AttributeError: module 'KNN_evaluation.similarity_compare' has no attribute 'extract_embeddings'`

- [x] **Step 3: 写最小实现**

追加到 `KNN_evaluation/similarity_compare.py` 末尾：

```python
def extract_embeddings(points, google_manager, xian_manager):
    """按同一 ids 列表从两个集合批量 retrieve，单侧缺失剔除并保持行序对齐.

    Args:
        points: sample_random_points 的返回（每项含 point_id）。
        google_manager: google 集合 QdrantManager。
        xian_manager: xian 集合 QdrantManager。

    Returns:
        (mat_g, mat_x, dropped)：mat_g/mat_x 为 (N', 64) float64 矩阵，
        行序 = points 原始顺序的保留子序列（两侧逐行像素级对应）；
        dropped 为单侧缺失剔除数。

    Raises:
        RuntimeError: kept == 0（两侧无任何对齐点）。
        ValueError: 向量维度不是 64（与 searcher.py:113 防御一致）。
        ConnectionError / qdrant 连接异常：向上传播，由 CLI/WebUI 各自捕获。
    """
    ids = [p["point_id"] for p in points]
    g_recs = google_manager.client.retrieve(
        collection_name=google_manager.collection_name,
        ids=ids, with_payload=True, with_vectors=True,
    )
    x_recs = xian_manager.client.retrieve(
        collection_name=xian_manager.collection_name,
        ids=ids, with_payload=True, with_vectors=True,
    )
    g_by_id = {str(r.id): np.array(r.vector, dtype=np.float64) for r in g_recs}
    x_by_id = {str(r.id): np.array(r.vector, dtype=np.float64) for r in x_recs}
    kept_ids = [pid for pid in ids if pid in g_by_id and pid in x_by_id]
    if not kept_ids:
        raise RuntimeError("两侧集合无任何对齐点：请检查双集合 image 集是否一致")
    mat_g = np.stack([g_by_id[pid] for pid in kept_ids])
    mat_x = np.stack([x_by_id[pid] for pid in kept_ids])
    if mat_g.shape[1] != 64:
        raise ValueError(f"google 集合向量维度应为 64，实际: {mat_g.shape[1]}")
    if mat_x.shape[1] != 64:
        raise ValueError(f"xian 集合向量维度应为 64，实际: {mat_x.shape[1]}")
    return mat_g, mat_x, len(ids) - len(kept_ids)


def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    """N×N 余弦相似度矩阵（numpy 向量化，对角恰为 1.0）.

    与 Qdrant COSINE 度量一致（cos_sim = dot / (|a| |b|)）。
    零向量防御：范数为 0 时置 1 避免 NaN（Deep Design D4）。
    """
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    v = vecs / norm
    return v @ v.T
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py -v`
Expected: PASS（新增 7 个测试通过，Task 1 的 7 个测试仍通过）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/similarity_compare.py KNN_evaluation/tests/test_similarity_compare.py
git commit -m "feat: add extract_embeddings and cosine_similarity_matrix"
```

---

### Task 3: `visualization.py` — 并排热力图 `plot_similarity_heatmap_pair`

**Files:**
- Modify: `KNN_evaluation/visualization.py`（顶部加 `import io`，追加函数）
- Modify: `KNN_evaluation/tests/test_similarity_compare.py`（追加测试类）

**Interfaces:**
- Consumes: Task 2 的 `cosine_similarity_matrix` 返回的两个 `(N', N')` 矩阵（仅用于构造测试数据）；matplotlib（Agg 后端已在模块顶部配置）。
- Produces: `plot_similarity_heatmap_pair(mat_g, mat_x, save_path, collection_names=("google_aef_embedding", "xian_aef_embedding")) -> None`；`save_path` 支持 `str | Path | io.BytesIO`（matplotlib `savefig` 原生支持类文件对象）。Task 4 编排函数、Task 5 CLI、Task 6 WebUI（BytesIO）依赖。

- [x] **Step 1: 写失败测试**

追加到 `KNN_evaluation/tests/test_similarity_compare.py` 末尾：

```python
class TestPlotSimilarityHeatmapPair:
    @staticmethod
    def _matrix(n=10):
        rng = np.random.default_rng(1)
        v = rng.normal(0, 1, (n, 64))
        return SC.cosine_similarity_matrix(v)

    def test_renders_png_file(self, tmp_path):
        from KNN_evaluation.visualization import plot_similarity_heatmap_pair
        out = tmp_path / "h.png"
        plot_similarity_heatmap_pair(self._matrix(), self._matrix(), out)
        assert out.exists() and out.stat().st_size > 0

    def test_renders_to_bytesio(self):
        from KNN_evaluation.visualization import plot_similarity_heatmap_pair
        buf = io.BytesIO()
        plot_similarity_heatmap_pair(self._matrix(), self._matrix(), buf)
        assert buf.getvalue().startswith(b"\x89PNG")

    def test_unified_color_scale(self, tmp_path, monkeypatch):
        import matplotlib.pyplot as plt
        from KNN_evaluation.visualization import plot_similarity_heatmap_pair
        captured = []
        orig = plt.Axes.imshow

        def spy(self, data, **kw):
            captured.append((kw.get("vmin"), kw.get("vmax")))
            return orig(self, data, **kw)

        monkeypatch.setattr(plt.Axes, "imshow", spy)
        plot_similarity_heatmap_pair(
            self._matrix(5), self._matrix(8), tmp_path / "h.png",
        )
        assert len(captured) == 2
        assert captured[0] == captured[1], "两子图必须统一 vmin/vmax 色阶"
        assert captured[0][0] is not None and captured[0][1] is not None
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py::TestPlotSimilarityHeatmapPair -v`
Expected: FAIL — `ImportError: cannot import name 'plot_similarity_heatmap_pair' from 'KNN_evaluation.visualization'`

- [x] **Step 3: 写最小实现**

在 `KNN_evaluation/visualization.py` 顶部 `from pathlib import Path` 之后加一行：

```python
import io
```

在文件末尾追加：

```python
def plot_similarity_heatmap_pair(
    mat_g: np.ndarray,
    mat_x: np.ndarray,
    save_path: str | Path | io.BytesIO,
    collection_names: tuple[str, str] = ("google_aef_embedding", "xian_aef_embedding"),
) -> None:
    """并排（1×2）渲染两个余弦相似度热力图，统一 vmin/vmax 色阶.

    Args:
        mat_g: google 集合 N'×N' 相似度矩阵。
        mat_x: xian 集合 N'×N' 相似度矩阵。
        save_path: PNG 输出路径（str/Path，自动建父目录）或类文件对象（BytesIO）。
        collection_names: 左/右子图标题集合名（默认 google × xian 预置对）。
    """
    vmin = min(float(mat_g.min()), float(mat_x.min()))
    vmax = max(float(mat_g.max()), float(mat_x.max()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    for ax, mat, name in (
        (ax1, mat_g, collection_names[0]),
        (ax2, mat_x, collection_names[1]),
    ):
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name} 相似度热力图")
        ax.set_xlabel("样本索引")
        ax.set_ylabel("样本索引")
        fig.colorbar(im, ax=ax, shrink=0.8)
    n = mat_g.shape[0]
    fig.suptitle(f"N={n}×{n} 余弦相似度矩阵对比（统一色阶）")
    if isinstance(save_path, (str, Path)):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py -v`
Expected: PASS（绘图 3 个测试通过；该文件全部测试通过）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/visualization.py KNN_evaluation/tests/test_similarity_compare.py
git commit -m "feat: add plot_similarity_heatmap_pair to visualization"
```

---

### Task 4: `similarity_compare.py` — 编排函数 `compare_similarity_heatmaps`

**Files:**
- Modify: `KNN_evaluation/similarity_compare.py`（顶部加 `import io`、`import time`、`from KNN_evaluation.visualization import plot_similarity_heatmap_pair`，末尾追加编排函数）
- Modify: `KNN_evaluation/tests/test_similarity_compare.py`（追加测试类）

**Interfaces:**
- Consumes: Task 1 `sample_random_points`、Task 2 `extract_embeddings` / `cosine_similarity_matrix`、Task 3 `plot_similarity_heatmap_pair`。
- Produces: `compare_similarity_heatmaps(google_manager, xian_manager, n=200, seed=42, image_id=None, output="similarity_heatmap.png", collection_names=("google_aef_embedding", "xian_aef_embedding")) -> dict`，返回 `{"sampled", "kept", "dropped", "matrix_shape": [N', N'], "elapsed_sec", "output_path"}`（D8 契约）。Task 5 CLI / Task 6 WebUI 依赖。

- [x] **Step 1: 写失败测试**

追加到 `KNN_evaluation/tests/test_similarity_compare.py` 末尾：

```python
class TestCompareSimilarityHeatmaps:
    def test_returns_metadata_and_writes_file(self, tmp_path, monkeypatch):
        out = tmp_path / "h.png"
        points = [{"point_id": f"p{i}"} for i in range(3)]
        monkeypatch.setattr(
            SC, "sample_random_points",
            lambda mgr, n, seed, image_id=None: points,
        )
        monkeypatch.setattr(
            SC, "extract_embeddings",
            lambda pts, gm, xm: (np.eye(3), np.eye(3), 0),
        )
        monkeypatch.setattr(
            SC, "plot_similarity_heatmap_pair",
            lambda a, b, out, collection_names=None: out.write_bytes(b"PNG"),
        )
        g, x = _manager("g"), _manager("x")
        result = SC.compare_similarity_heatmaps(g, x, n=3, seed=42, output=str(out))
        assert result["sampled"] == 3
        assert result["kept"] == 3
        assert result["dropped"] == 0
        assert result["matrix_shape"] == [3, 3]
        assert result["output_path"] == str(out)
        assert out.exists()

    def test_bytesio_output_path_empty(self, monkeypatch):
        buf = io.BytesIO()
        monkeypatch.setattr(
            SC, "sample_random_points",
            lambda mgr, n, seed, image_id=None: [{"point_id": "p0"}, {"point_id": "p1"}],
        )
        monkeypatch.setattr(
            SC, "extract_embeddings",
            lambda pts, gm, xm: (np.eye(2), np.eye(2), 0),
        )
        monkeypatch.setattr(
            SC, "plot_similarity_heatmap_pair",
            lambda a, b, out, collection_names=None: out.write(b"PNG"),
        )
        g, x = _manager("g"), _manager("x")
        result = SC.compare_similarity_heatmaps(g, x, n=2, seed=1, output=buf)
        assert result["output_path"] == ""
        assert buf.getvalue() == b"PNG"
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py::TestCompareSimilarityHeatmaps -v`
Expected: FAIL — `AttributeError: module 'KNN_evaluation.similarity_compare' has no attribute 'compare_similarity_heatmaps'`

- [x] **Step 3: 写最小实现**

在 `KNN_evaluation/similarity_compare.py` 顶部导入区改为：

```python
import io
import random
import time
import uuid

import numpy as np

from KNN_evaluation.sampling_map import ensure_sampling_map
from KNN_evaluation.visualization import plot_similarity_heatmap_pair
```

在文件末尾追加：

```python
def compare_similarity_heatmaps(
    google_manager,
    xian_manager,
    n: int = 200,
    seed: int = 42,
    image_id: str | None = None,
    output="similarity_heatmap.png",
    collection_names: tuple[str, str] = ("google_aef_embedding", "xian_aef_embedding"),
) -> dict:
    """编排：采样 → 双集合提取 → 双矩阵 → 并排热力图渲染 → 元数据（D8 契约）.

    Args:
        google_manager: google 集合 QdrantManager（采样侧）。
        xian_manager: xian 集合 QdrantManager。
        n: 目标采样数（1..600，默认 200）。
        seed: 随机种子（默认 42）。
        image_id: None → 数据库模式；非 None → 图片模式。
        output: PNG 输出路径（CLI 落盘）；WebUI 传 io.BytesIO 时图已渲染进 buffer，
                output_path 返回空字符串。
        collection_names: (google 名, xian 名)，用于热力图标题。

    Returns:
        {"sampled", "kept", "dropped", "matrix_shape": [N', N'],
         "elapsed_sec", "output_path"}。

    Raises:
        ValueError / RuntimeError / ConnectionError：透传下层函数异常，
        由 CLI/WebUI 各自捕获转错误信息。
    """
    _validate_n(n)
    start = time.perf_counter()
    points = sample_random_points(google_manager, n, seed, image_id=image_id)
    sampled = len(points)
    mat_g, mat_x, dropped = extract_embeddings(points, google_manager, xian_manager)
    sim_g = cosine_similarity_matrix(mat_g)
    sim_x = cosine_similarity_matrix(mat_x)
    plot_similarity_heatmap_pair(sim_g, sim_x, output, collection_names=collection_names)
    elapsed_sec = round(time.perf_counter() - start, 3)
    return {
        "sampled": sampled,
        "kept": mat_g.shape[0],
        "dropped": dropped,
        "matrix_shape": [mat_g.shape[0], mat_g.shape[0]],
        "elapsed_sec": elapsed_sec,
        "output_path": "" if isinstance(output, io.BytesIO) else output,
    }
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py -v`
Expected: PASS（全部测试通过）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/similarity_compare.py KNN_evaluation/tests/test_similarity_compare.py
git commit -m "feat: add compare_similarity_heatmaps orchestrator"
```

---

### Task 5: CLI — `similarity-heatmap` 子命令

**Files:**
- Modify: `KNN_evaluation/cli.py`（新增 `_parse_positive_n`、`cmd_similarity_heatmap`、parser 子命令、`main` 分发）
- Modify: `KNN_evaluation/tests/test_cli.py`（追加测试类）

**Interfaces:**
- Consumes: Task 4 `similarity_compare.compare_similarity_heatmaps`；`config.QDRANT_URL` / `DEFAULT_COLLECTION`；`qdrant_client.QdrantManager`。
- Produces: CLI 子命令 `similarity-heatmap`，参数 `--n`（默认 200，1..600）、`--seed`（42）、`--image-id`（可选→图片模式）、`--output`（`similarity_heatmap.png`）、`--google-collection`（`DEFAULT_COLLECTION`）、`--xian-collection`（`xian_aef_embedding`）、`--qdrant-url`。成功返回 0 并打印输出路径与 sampled/kept/dropped；不可达/集合不存在/参数超限/业务异常返回 1（D6）。

- [x] **Step 1: 写失败测试**

追加到 `KNN_evaluation/tests/test_cli.py` 末尾：

```python
class TestSimilarityHeatmap:
    """Task 3: similarity-heatmap 子命令 — 参数、两类模式、错误码、输出文件."""

    def _manager(self, exists=True, total_points=100):
        mgr = MagicMock()
        mgr.health_check.return_value = True
        mgr.collection_exists.return_value = exists
        mgr.collection_info.return_value = {"total_points": total_points}
        mgr.collection_name = "c"
        return mgr

    def _args(self, **overrides):
        base = dict(
            qdrant_url="http://localhost:6333",
            google_collection="google_aef_embedding",
            xian_collection="xian_aef_embedding",
            n=200, seed=42, image_id=None, output="similarity_heatmap.png",
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_parser_defaults(self):
        parser = cli._build_parser()
        args = parser.parse_args(["similarity-heatmap"])
        assert args.n == 200
        assert args.seed == 42
        assert args.image_id is None
        assert args.output == "similarity_heatmap.png"
        assert args.google_collection == "google_aef_embedding"
        assert args.xian_collection == "xian_aef_embedding"
        assert args.qdrant_url == "http://localhost:6333"

    def test_n_out_of_range_rejected(self):
        parser = cli._build_parser()
        for bad in ("0", "601", "-1"):
            with pytest.raises(SystemExit):
                parser.parse_args(["similarity-heatmap", "--n", bad])
        assert parser.parse_args(["similarity-heatmap", "--n", "600"]).n == 600

    def test_image_id_selects_image_mode(self):
        parser = cli._build_parser()
        args = parser.parse_args(["similarity-heatmap", "--image-id", "E121.4_N25.1"])
        assert args.image_id == "E121.4_N25.1"

    def test_qdrant_unreachable_returns_1(self, capsys, monkeypatch):
        mgr = self._manager()
        mgr.health_check.return_value = False
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        assert cli.cmd_similarity_heatmap(self._args()) == 1
        assert "Qdrant 不可达" in capsys.readouterr().err

    def test_missing_collection_returns_1(self, capsys, monkeypatch):
        mgr = self._manager(exists=False)
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        assert cli.cmd_similarity_heatmap(self._args()) == 1
        assert "不存在" in capsys.readouterr().err

    def test_builds_two_managers(self, monkeypatch):
        captured: list = []

        def fake_factory(url=None, collection_name=None, timeout=5):
            captured.append(collection_name)
            mgr = self._manager()
            mgr.collection_name = collection_name
            return mgr

        monkeypatch.setattr(cli, "QdrantManager", fake_factory)
        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps",
            lambda gm, xm, **kw: {
                "sampled": 1, "kept": 1, "dropped": 0,
                "matrix_shape": [1, 1], "elapsed_sec": 0.1, "output_path": kw["output"],
            },
        )
        assert cli.cmd_similarity_heatmap(self._args()) == 0
        assert captured == ["google_aef_embedding", "xian_aef_embedding"]

    def test_output_file_generated_and_printed(self, tmp_path, capsys, monkeypatch):
        out = tmp_path / "heatmap.png"
        args = self._args(output=str(out))
        managers = iter([self._manager(), self._manager()])

        def fake_factory(url=None, collection_name=None, timeout=5):
            m = next(managers)
            m.collection_name = collection_name
            return m

        monkeypatch.setattr(cli, "QdrantManager", fake_factory)

        def fake_compare(gm, xm, **kw):
            out.write_bytes(b"PNGDATA")
            return {
                "sampled": 200, "kept": 199, "dropped": 1,
                "matrix_shape": [199, 199], "elapsed_sec": 1.0,
                "output_path": str(out),
            }

        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps",
            fake_compare,
        )
        assert cli.cmd_similarity_heatmap(args) == 0
        assert out.exists()
        captured_out = capsys.readouterr().out
        assert "相似度热力图对比完成" in captured_out
        assert "剔除 1" in captured_out
        assert str(out) in captured_out
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_cli.py::TestSimilarityHeatmap -v`
Expected: FAIL — `AttributeError: 'ArgumentParser' object has no attribute` 或 `cmd_similarity_heatmap` 不存在

- [x] **Step 3: 写最小实现**

在 `KNN_evaluation/cli.py` 的 `_parse_label` 函数之后（`_tqdm_context` 之前）插入：

```python
def _parse_positive_n(value: str) -> int:
    """argparse type 校验：采样数 N 必须在 1..600（与 similarity_compare.MAX_N 同步）."""
    from KNN_evaluation.similarity_compare import MAX_N

    n = int(value)
    if not (1 <= n <= MAX_N):
        raise argparse.ArgumentTypeError(f"n 必须在 1..{MAX_N} 之间，实际: {n}")
    return n
```

在 `cmd_stats` 函数之后（`_np_encoder` 之前）插入：

```python
def cmd_similarity_heatmap(args) -> int:
    """执行 similarity-heatmap 子命令：双集合相似度热力图对比.

    流程：健康检查 → 双集合存在性检查 → 调 `compare_similarity_heatmaps`
    （数据库/图片模式由 --image-id 决定）→ 打印输出路径与 sampled/kept/dropped。
    """
    from KNN_evaluation.similarity_compare import compare_similarity_heatmaps

    g_manager = QdrantManager(url=args.qdrant_url, collection_name=args.google_collection)
    x_manager = QdrantManager(url=args.qdrant_url, collection_name=args.xian_collection)

    if not g_manager.health_check():
        print(f"错误: Qdrant 不可达 ({args.qdrant_url})", file=sys.stderr)
        return 1
    if not g_manager.collection_exists():
        print(f"错误: Collection '{args.google_collection}' 不存在，请先执行 import", file=sys.stderr)
        return 1
    if not x_manager.collection_exists():
        print(f"错误: Collection '{args.xian_collection}' 不存在，请先执行 import", file=sys.stderr)
        return 1

    try:
        result = compare_similarity_heatmaps(
            g_manager, x_manager,
            n=args.n, seed=args.seed, image_id=args.image_id,
            output=args.output,
            collection_names=(args.google_collection, args.xian_collection),
        )
    except (ValueError, RuntimeError, ConnectionError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except qdrant_http_exceptions.UnexpectedResponse as e:
        print(f"错误: Qdrant 返回异常 (HTTP {e.status_code}): {e}", file=sys.stderr)
        return 1
    except qdrant_http_exceptions.ResponseHandlingException as e:
        print(f"错误: Qdrant 连接异常: {e}", file=sys.stderr)
        return 1

    print(f"{'='*60}")
    print("相似度热力图对比完成")
    print(f"{'='*60}")
    print(f"采样点:   {result['sampled']}")
    print(f"保留点:   {result['kept']}  (剔除 {result['dropped']})")
    print(f"矩阵:     {result['matrix_shape'][0]}×{result['matrix_shape'][1]}")
    print(f"耗时:     {result['elapsed_sec']:.2f}s")
    print(f"输出:     {result['output_path']}")
    return 0
```

在 `_build_parser` 中，`# --- migrate ---` 块之前插入：

```python
    # --- similarity-heatmap ---
    p_sim = sub.add_parser("similarity-heatmap",
                           help="双集合 embedding 相似度热力图对比")
    p_sim.add_argument("--n", type=_parse_positive_n, default=200,
                       help="采样点数（1..600，默认: 200）")
    p_sim.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    p_sim.add_argument("--image-id", type=str, default=None,
                       help="指定影像（图片模式）；缺省为数据库全库模式")
    p_sim.add_argument("--output", type=str, default="similarity_heatmap.png",
                       help="输出 PNG 路径 (默认: similarity_heatmap.png)")
    p_sim.add_argument("--google-collection", default=DEFAULT_COLLECTION,
                       help=f"google 侧 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
    p_sim.add_argument("--xian-collection", default="xian_aef_embedding",
                       help="xian 侧 Qdrant Collection 名称 (默认: xian_aef_embedding)")
    p_sim.add_argument("--qdrant-url", default=QDRANT_URL,
                       help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")
```

在 `main()` 中，`elif args.command == "migrate":` 分支之后插入：

```python
    elif args.command == "similarity-heatmap":
        return cmd_similarity_heatmap(args)
```

同时更新文件顶部 docstring 用法说明，追加一行：

```python
    python -m KNN_evaluation.cli similarity-heatmap [--n N] [--seed N] [--image-id ID] [--output PATH] [--google-collection NAME] [--xian-collection NAME] [--qdrant-url URL]
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_cli.py::TestSimilarityHeatmap -v`
Expected: PASS（8 个 CLI 测试通过）

Run: `uv run pytest KNN_evaluation/tests/test_cli.py -v`
Expected: PASS（既有 CLI 测试不回归）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/cli.py KNN_evaluation/tests/test_cli.py
git commit -m "feat: add similarity-heatmap CLI subcommand"
```

---

### Task 6: WebUI — 「相似度热力图对比」面板

**Files:**
- Modify: `KNN_evaluation/webui.py`（在评估面板 expansion 之后、`# ===== 检索结果对话框 =====` 之前插入面板代码）
- Modify: `KNN_evaluation/tests/test_webui.py`（`_FakeUI` 增加 `radios` / `images` 与 `radio` / `image` 方法，追加测试类）

**Interfaces:**
- Consumes: Task 4 `similarity_compare.compare_similarity_heatmaps`；`webui.PRESET_COLLECTIONS`（固定预置对，与 `_current_collection` 无关）；`webui._imported_image_ids()`（图片模式下拉选项）；`webui._CLI_QDRANT_URL`；`webui.QdrantManager`。
- Produces: WebUI 面板 expansion「相似度热力图对比」：`ui.number("采样数 N", min=1, max=600, value=200)`、`ui.number("Seed", value=42)`、`ui.radio(["数据库全库", "单张图片"], value="数据库全库", on_change=...)`、`ui.select("影像（单张图片模式）", ...)`（单张图片模式可见）、按钮「生成热力图对比」；异步 `asyncio.to_thread` 执行，结果用 base64 data URI + `ui.image` 内嵌展示，失败 `ui.notify(type="negative")`。

- [x] **Step 1: 写失败测试**

先修改 `KNN_evaluation/tests/test_webui.py` 的 `_FakeUI`：

在 `__init__` 中（`self.dialogs: list = []` 之后）加两行：

```python
        self.radios: list = []  # radio 控件列表（记录 on_change / options）
        self.images: list = []  # ui.image 渲染的 src 列表
```

在 `input` 方法之后新增两个方法：

```python
    def radio(self, options=None, value=None, on_change=None, **kw):
        m = self._make_element(value)
        m.options = options
        m.on_change = on_change
        self.radios.append(m)
        return m

    def image(self, src=None, **kw):
        m = self._make_element(src)
        self.images.append(src)
        return m
```

在文件末尾追加测试类：

```python
class _SimilarityPanelFixture:
    """「相似度热力图对比」面板的公共 mock 数据与 helper."""

    @staticmethod
    def manager(total_points=1000):
        mgr = MagicMock()
        mgr.health_check.return_value = True
        mgr.collection_exists.return_value = True
        mgr.collection_info.return_value = {"total_points": total_points}
        mgr.collection_name = "test_collection"
        return mgr

    @staticmethod
    def patch_core(monkeypatch, result=None):
        """在源模块打补丁 compare_similarity_heatmaps（do_sim_compare 内局部 import）."""
        result = result or {
            "sampled": 200, "kept": 200, "dropped": 0,
            "matrix_shape": [200, 200], "elapsed_sec": 1.5, "output_path": "",
        }
        me = MagicMock(return_value=result)
        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps", me,
        )
        return me


class TestSimilarityHeatmapPanel:
    """Task 4: WebUI 相似度热力图对比面板 — 控件、模式切换、异步执行、失败提示."""

    def test_similarity_panel_controls_present(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        assert fake.numbers["采样数 N"].value == 200
        assert fake.numbers["采样数 N"].min == 1
        assert fake.numbers["采样数 N"].max == 600
        assert fake.numbers["Seed"].value == 42
        assert fake.buttons["生成热力图对比"] is not None
        assert fake.radios[0].options == ["数据库全库", "单张图片"]
        assert fake.selects["影像（单张图片模式）"] is not None

    def test_image_select_hidden_in_db_mode_by_default(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        img = fake.selects["影像（单张图片模式）"]
        assert img.set_visibility.call_args.args == (False,)

    def test_mode_switch_toggles_image_select_visibility(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        radio = fake.radios[0]
        img = fake.selects["影像（单张图片模式）"]
        radio.value = "单张图片"
        monkeypatch.setattr(
            webui, "_get_manifest_cached",
            lambda: {"images": {"A": 16384}, "collection": "c", "updated_at": ""},
        )

        async def drive():
            radio.on_change()
            await asyncio.sleep(0.05)

        asyncio.run(drive())
        assert (True,) in [c.args for c in img.set_visibility.call_args_list]

    def test_do_sim_compare_renders_image(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        webui.state["manager"] = _SimilarityPanelFixture.manager()

        def fake_mgr(url=None, collection_name=None, timeout=5):
            return MagicMock()

        monkeypatch.setattr(webui, "QdrantManager", fake_mgr)
        calls: dict = {}

        def fake_compare(gm, xm, n=200, seed=42, image_id=None,
                         output=None, collection_names=None):
            calls["output"] = output
            calls["collection_names"] = collection_names
            output.write(b"\x89PNG")
            return {
                "sampled": n, "kept": n, "dropped": 0,
                "matrix_shape": [n, n], "elapsed_sec": 0.5, "output_path": "",
            }

        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps",
            fake_compare,
        )

        async def drive():
            await fake.buttons["生成热力图对比"].on_click()

        asyncio.run(drive())
        # 固定预置对：与 _current_collection 无关
        assert calls["collection_names"] == tuple(webui.PRESET_COLLECTIONS)
        assert isinstance(calls["output"], io.BytesIO)
        assert fake.images and fake.images[0].startswith("data:image/png;base64,")
        status = fake.find_label(classes=("text-sm", "text-grey", "mt-2"))
        assert "采样 200" in status.text

    def test_image_mode_without_selection_notifies(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        webui.state["manager"] = _SimilarityPanelFixture.manager()
        fake.radios[0].value = "单张图片"
        fake.selects["影像（单张图片模式）"].value = None

        async def drive():
            await fake.buttons["生成热力图对比"].on_click()

        asyncio.run(drive())
        assert ("请先选择影像", "negative") in fake.notify_calls

    def test_failure_notifies_negative(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        webui.state["manager"] = _SimilarityPanelFixture.manager()

        def boom(gm, xm, **kw):
            raise ValueError("采样失败")

        monkeypatch.setattr(
            "KNN_evaluation.similarity_compare.compare_similarity_heatmaps", boom,
        )

        async def drive():
            await fake.buttons["生成热力图对比"].on_click()

        asyncio.run(drive())
        assert any("对比失败" in msg for msg, _ in fake.notify_calls)
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py::TestSimilarityHeatmapPanel -v`
Expected: FAIL — `AssertionError: 未找到 label(text='', classes='text-sm text-grey mt-2')` 或 `KeyError: '采样数 N'`

- [x] **Step 3: 写最小实现**

在 `KNN_evaluation/webui.py` 中，评估面板 expansion 结束之后（`# ===== 检索结果对话框 =====` 之前，约 line 1527）插入：

```python
    # ===== 相似度热力图对比 =====
    with ui.expansion("相似度热力图对比", value=False).classes("w-full mt-2"):
        with ui.row().classes("items-center gap-4 mt-2"):
            sim_n_input = ui.number(label="采样数 N", value=200, min=1, max=600).classes("w-24")
            sim_seed_input = ui.number(label="Seed", value=42).classes("w-24")

        sim_mode = ui.radio(
            ["数据库全库", "单张图片"], value="数据库全库",
            on_change=lambda: asyncio.create_task(_apply_sim_mode()),
        ).props("inline")
        sim_image_select = ui.select(
            label="影像（单张图片模式）", options={}, value=None,
        ).classes("w-64")
        sim_image_select.set_visibility(False)

        async def _fill_sim_image_ids():
            """单张图片模式：从 manifest 缓存读已导入 image_id 填充下拉."""
            ids = _imported_image_ids()
            if not ids:
                return
            sim_image_select.set_options(
                {iid: iid for iid in sorted(ids)}, value=None,
            )

        async def _apply_sim_mode():
            if sim_mode.value == "单张图片":
                await _fill_sim_image_ids()
                sim_image_select.set_visibility(True)
            else:
                sim_image_select.set_visibility(False)

        sim_status_label = ui.label("").classes("text-sm text-grey mt-2")
        sim_status_label.set_visibility(False)
        sim_result_container = ui.column().classes("w-full mt-4")
        sim_result_container.set_visibility(False)

        async def do_sim_compare():
            if state["manager"] is None or not state["manager"].health_check():
                ui.notify("Qdrant 不可达", type="negative")
                return
            n = int(sim_n_input.value)
            seed = int(sim_seed_input.value)
            image_id = None
            if sim_mode.value == "单张图片":
                image_id = sim_image_select.value
                if not image_id:
                    ui.notify("请先选择影像", type="negative")
                    return
            from KNN_evaluation.similarity_compare import compare_similarity_heatmaps
            # 固定对比预置对（D7）：与 _current_collection 选择器无关
            g_manager = QdrantManager(
                url=_CLI_QDRANT_URL, collection_name=PRESET_COLLECTIONS[0],
            )
            x_manager = QdrantManager(
                url=_CLI_QDRANT_URL, collection_name=PRESET_COLLECTIONS[1],
            )
            buf = io.BytesIO()
            sim_status_label.set_visibility(True)
            sim_status_label.set_text("正在采样并提取双集合 embedding...")
            try:
                result = await asyncio.to_thread(
                    compare_similarity_heatmaps,
                    g_manager, x_manager,
                    n=n, seed=seed, image_id=image_id,
                    output=buf,
                    collection_names=(PRESET_COLLECTIONS[0], PRESET_COLLECTIONS[1]),
                )
            except (ValueError, RuntimeError, ConnectionError) as e:
                ui.notify(f"对比失败: {e}", type="negative")
                sim_status_label.set_text(f"失败: {e}")
                return
            buf.seek(0)
            data_uri = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
            sim_status_label.set_text(
                f"采样 {result['sampled']} | 保留 {result['kept']} "
                f"(剔除 {result['dropped']}) | 矩阵 {result['matrix_shape'][0]}×"
                f"{result['matrix_shape'][1]} | 耗时 {result['elapsed_sec']:.2f}s"
            )
            sim_result_container.clear()
            sim_result_container.set_visibility(True)
            with sim_result_container:
                ui.image(data_uri)

        with ui.row().classes("items-center gap-4 mt-2"):
            ui.button("生成热力图对比", on_click=do_sim_compare).props("flat")
```

> 注：`webui.py` 顶部已 import `base64` / `io` / `asyncio` / `PRESET_COLLECTIONS` / `QdrantManager` / `_CLI_QDRANT_URL` / `_imported_image_ids`，无需新增 import。`io.BytesIO` + base64 不落盘（D7 决策）；`plot_similarity_heatmap_pair` 原生支持类文件对象。

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py::TestSimilarityHeatmapPanel -v`
Expected: PASS（6 个面板测试通过）

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: PASS（既有 webui 测试不回归）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/webui.py KNN_evaluation/tests/test_webui.py
git commit -m "feat: add similarity heatmap compare panel to webui"
```

---

### Task 7: 文档更新

**Files:**
- Modify: `KNN_evaluation/README.md`
- Modify: `README.md`（项目根）

**Interfaces:**
- Consumes: Task 5 CLI 子命令参数、Task 6 WebUI 面板名称。

- [x] **Step 1: 更新 `KNN_evaluation/README.md`**

在 `### WebUI 评估面板` 小节（约 line 159）之后追加：

```markdown
### 双集合相似度热力图对比

系统内置双集合 embedding 相似度热力图对比功能：从两个集合（默认
`google_aef_embedding` × `xian_aef_embedding`，point_id 确定性一致，天然按位置
对齐）随机采样 N 个点，按同一批 point_id 分别取 embedding，各自计算 N×N 余弦
相似度矩阵，并排（1×2）统一色阶输出 PNG。

CLI 方式：

```bash
# 数据库全库模式
uv run python -m KNN_evaluation.cli similarity-heatmap --n 200 --seed 42 --output similarity_heatmap.png

# 单张图片模式（指定 image_id）
uv run python -m KNN_evaluation.cli similarity-heatmap --n 200 --seed 42 --image-id E121.4794_N25.1378 --output heatmap.png

# 自定义双集合
uv run python -m KNN_evaluation.cli similarity-heatmap --google-collection google_aef_embedding --xian-collection xian_aef_embedding
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n` | 200 | 采样点数，范围 1..600 |
| `--seed` | 42 | 随机种子，保证可复现 |
| `--image-id` | 无 | 指定影像则进入图片模式，否则数据库全库模式 |
| `--output` | similarity_heatmap.png | 输出 PNG 路径 |
| `--google-collection` / `--xian-collection` | google_aef_embedding / xian_aef_embedding | 对比双集合名称 |
| `--qdrant-url` | http://localhost:6333 | Qdrant 服务地址 |

WebUI 方式：展开「**相似度热力图对比**」expansion（位于"评估面板"下方），设置
采样数 N、seed，选择模式（数据库全库 / 单张图片），点击「生成热力图对比」即可
在面板内嵌展示并排热力图。WebUI 固定对比预置对 `google_aef_embedding` ×
`xian_aef_embedding`，与当前 collection 选择器无关。
```

- [x] **Step 2: 更新根 `README.md`**

在 `# 评估 embedding 质量 (F1 / Purity / Recall@K)` 代码块之后追加 CLI 用法：

```markdown
# 双集合相似度热力图对比（google_aef_embedding × xian_aef_embedding）
uv run python -m KNN_evaluation.cli similarity-heatmap [--n 200] [--seed 42] [--image-id <影像ID>] [--output similarity_heatmap.png]
```

在「启动 WebUI」段落的页面能力描述（`embedding 质量评估面板` 之后）追加一句：

```markdown
支持双集合 embedding 相似度热力图对比面板（固定预置对 `google_aef_embedding` × `xian_aef_embedding`，位于"评估面板"下方）。
```

在 `## 目录结构` 代码块中 `metrics.py        # KNN 质量评估（F1 / Purity / Recall@K）` 之后加一行：

```markdown
  similarity_compare.py # 双集合相似度热力图对比（采样/提取/矩阵/编排）
```

- [x] **Step 3: 全量回归 + 验证**

Run: `uv run pytest KNN_evaluation/tests/ -v`
Expected: PASS（全部既有 + 新增测试通过）

验证 CLI 帮助可用：

Run: `uv run python -m KNN_evaluation.cli similarity-heatmap --help`
Expected: 输出包含 `--n`、`--seed`、`--image-id`、`--output`、`--google-collection`、`--xian-collection`、`--qdrant-url`

- [x] **Step 4: Commit**

```bash
git add KNN_evaluation/README.md README.md
git commit -m "docs: add similarity-heatmap CLI and WebUI panel usage"
```

---

## Self-Review 记录

- **Spec 覆盖**：采样（db/图片两种模式、N 校验、seed 复现、样本不足、空集合）→ Task 1；提取（同 ids retrieve、单侧缺失剔除、行序对齐、N'=0 报错、维度防御）→ Task 2；矩阵（对称/对角 1.0/值域/零向量）→ Task 2；并排热力图（统一色阶、viridis、colorbar、BytesIO 支持）→ Task 3；编排与 D8 返回契约 → Task 4；CLI 子命令全部参数与错误码 → Task 5；WebUI 面板（固定预置对、模式单选、图片下拉、`asyncio.to_thread`、`ui.image` base64、失败 notify）→ Task 6；README 双文件 → Task 7。Non-Goals（无新依赖、不改缓存格式、不动既有流程）全部在 Global Constraints 落实。
- **占位符扫描**：所有步骤均含完整代码与可执行命令，无 TBD/TODO/"类似 Task N"。
- **类型一致性**：`sample_random_points(manager, n, seed, image_id=None)` / `extract_embeddings(points, google_manager, xian_manager) -> (mat_g, mat_x, dropped)` / `cosine_similarity_matrix(vecs)` / `plot_similarity_heatmap_pair(mat_g, mat_x, save_path, collection_names=...)` / `compare_similarity_heatmaps(gm, xm, n=200, seed=42, image_id=None, output, collection_names=...) -> {sampled, kept, dropped, matrix_shape, elapsed_sec, output_path}` 在各任务中签名一致；WebUI 传 `output=BytesIO` 返回 `output_path=""`，CLI 传路径返回路径，与 D8 契约一致。

---

### Task 8: 导出数据功能（npy 矩阵 + JSON 采样信息）

**Files:**
- Modify: `KNN_evaluation/similarity_compare.py`
- Modify: `KNN_evaluation/cli.py`
- Modify: `KNN_evaluation/webui.py`
- Modify: `KNN_evaluation/tests/test_similarity_compare.py`
- Modify: `KNN_evaluation/tests/test_cli.py`
- Modify: `KNN_evaluation/tests/test_webui.py`
- Modify: `KNN_evaluation/README.md` + 根 `README.md`

**Interfaces:**
- `extract_embeddings(points, google_manager, xian_manager)` 返回值扩展为 4 元 `(mat_g, mat_x, dropped, kept_records)`
- 新增 `export_similarity_outputs(sim_g, sim_x, meta, pixels, export_dir, collection_names)`
- `compare_similarity_heatmaps(..., export_dir=None)` 新增可选参数；返回元数据新增 `exported_files`
- CLI `similarity-heatmap` 新增 `--export-dir DIR`
- WebUI 面板新增「导出目录」输入框

- [x] **Step 1: 写失败测试（extract_embeddings 4 元返回 + kept_records 行序）**

更新 `KNN_evaluation/tests/test_similarity_compare.py` 中 `TestExtractEmbeddings`：解包改为 4 元，新增断言 `kept_records` 与 `kept_ids` 行序一致、含 utm 字段。

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_similarity_compare.py -v -k Extract`，Expected: 解包/断言相关失败（RED）。

- [x] **Step 3: 修改 extract_embeddings 返回 4 元**

`similarity_compare.py`：构建 `kept_records`（从 google 侧 payload 提取 point_id/image_id/pixel_row/pixel_col/utm_easting/utm_northing/utm_zone，按 kept_ids 顺序）。

- [x] **Step 4: 运行测试确认通过（GREEN）**

- [x] **Step 5: 写失败测试（export_similarity_outputs）**

新增 `TestExportSimilarityOutputs`：两 npy 存在且维度正确、json 含 params+pixels 且与行序一致、目录自动创建。

- [x] **Step 6: 实现 export_similarity_outputs**

`np.save(export_dir / f"{collection_names[i]}_similarity.npy", sim)` + `json.dump({"params": meta, "pixels": pixels}, ...)`，`mkdir(parents=True, exist_ok=True)`。

- [x] **Step 7: 写失败测试（compare_similarity_heatmaps export_dir 分支）**

新增断言：默认 export_dir=None 不导出（无文件）；指定 export_dir 导出且 `exported_files` 含 3 个文件路径。

- [x] **Step 8: 实现 compare_similarity_heatmaps export_dir 参数**

非 None 时调用 `export_similarity_outputs`，返回元数据加 `exported_files`；None 时行为不变。

- [x] **Step 9: CLI `--export-dir`**

`cli.py` parser 新增 `--export-dir`；`cmd_similarity_heatmap` 传给编排函数，导出时打印文件路径。测试：`test_cli.py` 新增参数解析与导出断言。

- [x] **Step 10: WebUI 导出目录输入框**

`webui.py` 面板新增「导出目录」`ui.input`，非空时传 `export_dir`，完成后展示导出文件路径。测试：`test_webui.py` 新增面板导出测试。

- [x] **Step 11: README 更新**

两份 README 新增 `--export-dir` 与导出文件说明。

- [x] **Step 12: 全量回归 + Commit**

Run: `uv run pytest KNN_evaluation/tests/ -v`，Expected: 全部通过（含新导出测试）。Commit message: `feat(similarity-heatmap-compare): 导出相似度矩阵与采样信息（npy + json）`。

---

### Task 9: WebUI 导入目录同步 + 分页默认目录 + 默认导出 outputs/（Bug 修复）

**Files:**
- Modify: `KNN_evaluation/webui.py`
- Modify: `KNN_evaluation/config.py`
- Modify: `KNN_evaluation/cli.py`
- Modify: `KNN_evaluation/similarity_compare.py`
- Modify: `KNN_evaluation/tests/test_webui.py`、`test_cli.py`、`test_similarity_compare.py`
- Modify: `KNN_evaluation/README.md` + 根 `README.md`

**Interfaces:**
- `config.py` 新增 `COLLECTION_DATA_DIRS = {"google_aef_embedding": "data_google", "xian_aef_embedding": "data_xian"}`
- `do_import` 从 `dir_input.value` 取值同步 `state["data_dir"]`；目录不存在报错
- `_apply_collection` 按映射更新数据目录输入框与 `state["data_dir"]`
- `compare_similarity_heatmaps` 默认 `export_dir="outputs"`；CLI `--export-dir` 默认 `outputs`；WebUI 导出输入框默认 `outputs`

- [x] **Step 1: 写失败测试（do_import 读输入框 + 分页联动）**

在 `test_webui.py` 新增：`do_import` 使用输入框当前值（改输入框不点浏览也生效）；目录不存在报错不导入；分页切换联动输入框（google→data_google / xian→data_xian）；映射正确性。

- [x] **Step 2: 运行测试确认失败（RED）**

- [x] **Step 3: 实现 config.py 映射 + do_import 目录同步 + 分页联动**

`COLLECTION_DATA_DIRS` 放 config.py；`do_import` 从 `dir_input.value` 解析并同步 state；`_apply_collection` 切分页时按映射更新输入框与 state（经闭包 hook）。

- [x] **Step 4: 运行测试确认通过（GREEN）**

- [x] **Step 5: 写失败测试（默认导出 outputs/）**

`test_similarity_compare.py`：`compare_similarity_heatmaps` 不传 export_dir 时导出到 `outputs/`；显式 None 禁用。`test_cli.py`：`--export-dir` 默认值 `outputs`。`test_webui.py`：导出输入框默认值 `outputs`。

- [x] **Step 6: 实现默认导出 outputs/**

`similarity_compare.py` 默认 `export_dir="outputs"`（空串/None 禁用）；`cli.py` parser 默认 `outputs`（`--export-dir ""` 禁用）；`webui.py` 输入框默认 `outputs`（留空禁用）。

- [x] **Step 7: 运行测试确认通过（GREEN）+ README 更新**

README 更新：默认导出 outputs/、data_google/data_xian 分页默认目录。

- [x] **Step 8: 全量回归 + Commit**

Run: `uv run pytest KNN_evaluation/tests/ -v`。Commit message: `fix(similarity-heatmap-compare): WebUI 导入目录同步 + 分页默认目录 + 默认导出 outputs/`。

---

### Task 10: 坐标段数值匹配配对（导入修复）

**Files:**
- Modify: `KNN_evaluation/data_loader.py`
- Modify: `KNN_evaluation/tests/test_data_loader.py`
- Modify: `KNN_evaluation/README.md`（如适用）

**Interfaces:**
- `scan_directory` 配对 key：SE/DW/TIF 各文件坐标段字符串 → `parse_location_coord` 数值 `(lon, lat)` 元组
- `ImagePair.image_id` 仍取 SE 侧 `extract_location_key` 原始字符串（不改 point_id 语义）

- [x] **Step 1: 写失败测试（数值匹配配对）**

`test_data_loader.py` 新增 `TestScanDirectoryNumericMatch`：SE 4 位小数（`E121.4033_N25.1370`）/ DW 3 位小数（`E121.4033_N25.137`）→ 配对成功且 `image_id == "E121.4033_N25.1370"`；数值相同字符串不同多文件配对；孤儿文件跳过。

- [x] **Step 2: 运行测试确认失败（RED）**

Run: `uv run pytest KNN_evaluation/tests/test_data_loader.py -v`，Expected: 数值配对用例失败（当前字符串精确匹配）。

- [x] **Step 3: 实现 scan_directory 数值配对**

`data_loader.py`：`scan_directory` 中 SE/DW/TIF 各 dict 的 key 从 `extract_location_key(fname)` 改为 `parse_location_coord(extract_location_key(fname))`（`(lon, lat)` 元组）；`ImagePair.image_id` 用 SE 文件原始 key（`se_raw_keys[num_key]`）。孤儿跳过语义不变。

- [x] **Step 4: 运行测试确认通过（GREEN）**

- [x] **Step 5: 全量回归 + Commit**

Run: `uv run pytest KNN_evaluation/tests/ -v`。Commit message: `fix(similarity-heatmap-compare): 坐标段数值匹配配对（SE/DW 精度不一致可配对）`。

---

### Task 11: image_id 全链路归一化

**Files:**
- Modify: `KNN_evaluation/data_loader.py`
- Modify: `KNN_evaluation/tests/test_data_loader.py`
- Modify: `KNN_evaluation/README.md` + 根 `README.md`

**Interfaces:**
- 新增 `PixelDataLoader.normalize_location_key(raw_key) -> str`：round 4 位小数 + 去尾随零（`E121.4033_N25.1370`→`E121.4033_N25.137`）
- `scan_directory` 的 `ImagePair.image_id` 改用 `normalize_location_key(se_raw)`（替代 SE 原始串）；配对 key 仍数值 `(lon, lat)`

- [x] **Step 1: 写失败测试（normalize_location_key + scan_directory image_id 归一化）**

`test_data_loader.py`：新增 `TestNormalizeLocationKey`（去尾随零、数值同字符串异→同串、round 4 位）；更新 `TestScanDirectoryNumericMatch` 断言 image_id 为归一化串。

- [x] **Step 2: 运行测试确认失败（RED）**

- [x] **Step 3: 实现 normalize_location_key + scan_directory image_id 归一化**

`data_loader.py`：新增静态方法 `normalize_location_key`；`scan_directory` 的 `se_raw[key]` 改为 `normalize_location_key(se_raw[key])` 作为 `image_id`。

- [x] **Step 4: 运行测试确认通过（GREEN）**

- [x] **Step 5: README 更新 + 全量回归 + Commit**

README 更新（image_id 归一化 + 升级需清空重导两 collection）；`uv run pytest KNN_evaluation/tests/ -v` 通过。Commit message: `feat(similarity-heatmap-compare): image_id 全链路归一化（point_id 双集合一致，需重导）`。

---

### Task 12: 可视化探索按检索 collection 定位影像文件

**Files:**
- Modify: `KNN_evaluation/webui.py`
- Modify: `KNN_evaluation/tests/test_webui.py`

**Interfaces:**
- `do_search` 记录 `state["search_collection"]`
- `_show_visualization` 按 collection 解析数据目录（`COLLECTION_DATA_DIRS` 回退 `_CLI_DATA_DIR`）→ `scan_directory` 构建局部 `viz_se_map`
- `_refresh_viz` / `on_mouse` 用 `viz_se_map`

- [x] **Step 1: 写失败测试（google 检索可视化用 google 数据）**

`test_webui.py`：`_show_visualization` 在 `search_collection=google_aef_embedding` 时扫描 `data_google`（`scan_directory` monkeypatch 断言调用路径）；`_refresh_viz` 用该映射查找。

- [x] **Step 2: 运行测试确认失败（RED）**

- [x] **Step 3: 实现 do_search 记录 collection + _show_visualization 按 collection 扫描**

`webui.py`：`do_search` 加 `state["search_collection"] = state["manager"].collection_name`；`_show_visualization` 解析数据目录（`COLLECTION_DATA_DIRS.get(col) or _CLI_DATA_DIR`）→ `scan_directory` 构建 `viz_se_map`；`_refresh_viz`/`on_mouse` 改用 `viz_se_map`。

- [x] **Step 4: 运行测试确认通过（GREEN）**

- [x] **Step 5: 全量回归 + Commit**

`uv run pytest KNN_evaluation/tests/ -v`。Commit message: `fix(similarity-heatmap-compare): 可视化探索按检索 collection 定位影像文件（修复背景图串集）`。

---

### Task 13: payload 索引自动补齐

**Files:**
- Modify: `KNN_evaluation/qdrant_client.py`
- Modify: `KNN_evaluation/webui.py`
- Modify: `KNN_evaluation/tests/test_qdrant_client.py`、`test_webui.py`

**Interfaces:**
- 新增 `QdrantManager.ensure_payload_indices()`：读 `payload_schema`，对缺失的 5 字段（label/label_name/utm_easting/utm_northing/image_id）逐个 `create_payload_index`，已有跳过（幂等）
- `webui.py` `_apply_collection` 与页面加载对当前 collection 调用 `ensure_payload_indices()`

- [x] **Step 1: 写失败测试（ensure_payload_indices 幂等）**

`test_qdrant_client.py`：mock `get_collection` 返回缺失部分字段的 schema → 断言只对缺失字段 `create_payload_index`；schema 齐全 → 不调用。

- [x] **Step 2: 运行测试确认失败（RED）**

- [x] **Step 3: 实现 ensure_payload_indices + webui 调用**

`qdrant_client.py`：读 `payload_schema` 比对缺失字段逐个建索引；`webui.py` `_apply_collection` 与初始化路径调用。

- [x] **Step 4: 运行测试确认通过（GREEN）**

- [x] **Step 5: 全量回归 + Commit**

`uv run pytest KNN_evaluation/tests/ -v`。Commit message: `feat(similarity-heatmap-compare): payload 索引自动补齐（防 UTM 过滤全量扫描超时）`。
