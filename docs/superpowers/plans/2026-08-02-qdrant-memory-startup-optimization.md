---
change: qdrant-memory-startup-optimization
design-doc: docs/superpowers/specs/2026-08-02-qdrant-memory-startup-optimization-design.md
base-ref: dc9dc0a43a2c4ef8d0607011c6f09437ea797ddb
archived-with: 2026-08-02-qdrant-memory-startup-optimization
---

# Qdrant 内存占用与 Web 启动优化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Web 启动从"全库 scroll 阻塞数分钟"降为"秒级显示 ✅ + 毫秒级影像列表（本地 manifest）"；将 Qdrant 常驻内存从 ≈6-10GB 降到 ≈2-3GB（磁盘化 + Docker volume 持久化 + 幂等启动）；将评估批量路径改为显式 opt-in 并加内存守卫，默认走服务端逐条 exact（内存安全）。

**Architecture:** 新增 `KNN_evaluation/manifest.py` 纯 IO 模块（原子写：tmp + `os.replace`），作为 collection 级导入清单的可重建缓存。`init_page` 拆为快速路径（health_check → 置 ✅ → `sleep(0)` 让出事件循环）+ 慢速路径（`asyncio.create_task(_background_init)` 顺序执行 load_manifest → scan_directory → facet 对账 → 渲染）；`_refresh_image_list`/`_render_preview` 改读进程级缓存 `_manifest_cache`，不再逐条 `check_image_count`。`create_collection(storage="disk")` 默认 `on_disk=True` + `on_disk_payload=True` + `quantization_config=None`；新增 `migrate` CLI 子命令（备份 → 删除重建 → 断点续传重导 → 重建 manifest），`_start_qdrant()` 改幂等三态。`metrics.py` 中 `use_batch` 默认改 `False`，新增 `estimate_batch_memory`/`guard_batch_memory`；CLI 新增 `--batch` 与 `--max-eval-ram`。

**Tech Stack:** Python 3.12+, numpy, qdrant-client 1.18.0（`facet` 无 `distinct`）, NiceGUI, Docker CLI, pytest（`uv run pytest`）

## Global Constraints

- 不新增第三方依赖（仅复用现有 qdrant-client、numpy、nicegui、tqdm、pytest；Docker CLI 用于容器管理）
- `reconcile_manifest()` 对账 API = `client.facet(key="image_id", limit=1000, exact=True)`（qdrant-client 1.18.0 **无** `distinct()`）；facet 只返回去重集合，**不含像素数**，对差异项用 `PixelDataLoader.check_image_count` 补精确像素数
- manifest 是可重建缓存，非唯一真相：并发写"最后一次写胜"（`save_manifest` 原子写 tmp + `os.replace`），无需文件锁；下次启动对账自动纠正
- `create_collection()` 默认 `storage="disk"`（向量 `on_disk=True` + `on_disk_payload=True` + `quantization_config=None`）；`storage="ram"` 保持现状（`on_disk=False`）；与 CLI `--storage` 参数（默认 disk）一致
- `compute_knn_accuracy` / `compute_purity_recall_curve` 的 `use_batch` 默认值从 `True` 改 `False`（默认服务端逐条 exact）；`--batch` 显式 opt-in 后才走批量 numpy 路径并触发 `guard_batch_memory`
- 内存守卫公式（仅客户端进程峰值，N=total_points，Q=查询数，K=max k）：`all_vecs = N×64×8B`，`a_norm = N×64×8B`，`topk idxs = Q×K×8B`，`labels/ids = N×8B`；`@10M 点 ≈ 10.4GB`；`--max-eval-ram` 默认 6.0，超过抛 `MemoryError` 含预估
- `migrate` 开头自动调用幂等 `_start_qdrant()`（`docker ps` → 运行中复用 / `docker start` / `docker run -v qdrant_data:/qdrant/storage`），失败返回 False，不阻塞既有 WebUI
- 不修改 Collection schema、payload 结构、`searcher.py` 语义；不改变 `import_directory` 断点续传/重试逻辑（仅复用）
- manifest 文件固定 `qdrant_import_manifest.json`（项目根），加入 `.gitignore`（生成文件不入库）
- 中文注释与错误信息；所有测试命令用 `uv run pytest`；使用 mock 验证磁盘化参数（避免依赖真实 Qdrant 服务）

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `KNN_evaluation/manifest.py` | CREATE | `MANIFEST_PATH` / `load_manifest` / `save_manifest`（原子写）/ `update_manifest` |
| `KNN_evaluation/tests/test_manifest.py` | CREATE | manifest 读取/增量更新/缺失返回空/原子写/对账一致与不一致/重建 |
| `KNN_evaluation/qdrant_client.py` | MODIFY | `get_imported_image_ids()` 改读 manifest；新增 `reconcile_manifest()` / `_facet_image_ids()`；`create_collection(storage=...)` |
| `KNN_evaluation/tests/test_qdrant_client.py` | MODIFY | manifest 版 `get_imported_image_ids` / `reconcile_manifest` / 磁盘化参数断言 |
| `KNN_evaluation/webui.py` | MODIFY | `init_page` 快速+慢速路径、`_background_init`、`_manifest_cache`、`_refresh_image_list`/`_render_preview` 读 manifest、评估面板批量 checkbox + 预估 label |
| `KNN_evaluation/tests/test_webui.py` | CREATE | mock manager，`init_page` 快速路径时限与事件循环不被全库 scroll 堵死 |
| `KNN_evaluation/importer.py` | MODIFY | `import_directory` 循环内无条件 `update_manifest`（导入成功记 16384 / 跳过记 count 值） |
| `KNN_evaluation/cli.py` | MODIFY | `evaluate` 新增 `--batch`/`--max-eval-ram` + 内存守卫；新增 `migrate` 子命令与 `cmd_migrate`；import 支持 `--storage` |
| `KNN_evaluation/metrics.py` | MODIFY | `use_batch` 默认改 False；新增 `estimate_batch_memory` / `guard_batch_memory` |
| `KNN_evaluation/tests/test_metrics.py` | MODIFY | 内存守卫单测（超阈值拒绝、默认不触发、批量/逐条一致） |
| `.gitignore` | MODIFY | 新增 `qdrant_import_manifest.json` |
| `README.md` | MODIFY | Docker volume 启动命令、数据持久化说明、`migrate` 用法 |

---

### Task 1.1: manifest.py 基础设施（读取 / 保存 / 增量更新）

**Files:**
- Create: `KNN_evaluation/manifest.py`
- Create: `KNN_evaluation/tests/test_manifest.py`
- Modify: `.gitignore`（追加 `qdrant_import_manifest.json`）

**Interfaces:**
- Produces:
  - `MANIFEST_PATH: Path = Path("qdrant_import_manifest.json")`
  - `load_manifest(path: Path = MANIFEST_PATH) -> dict`：返回 `{"collection": str, "images": {image_id: int}, "updated_at": str}`；文件缺失/损坏返回空结构 `{"collection": "", "images": {}, "updated_at": ""}`（不抛错，由对账路径重建）
  - `save_manifest(data: dict, path: Path = MANIFEST_PATH) -> None`：原子写（`tempfile.mkstemp(dir=path.parent, suffix=".tmp")` → `json.dump(ensure_ascii=False, indent=2)` → `os.replace`），父目录不存在时创建
  - `update_manifest(image_id: str, imported_pixels: int, collection: str, path: Path = MANIFEST_PATH) -> dict`：读当前 manifest（缺失按空结构），更新 `images[image_id] = imported_pixels`、`collection`、`updated_at`（`datetime.now().isoformat()`），原子写，返回更新后的 dict
- Consumed by: Task 1.2（qdrant_client）、Task 2.1/2.3/2.4（webui）、Task 2.5（importer）、Task 3.3（migrate）

**Dependencies:** 无（最先实现）。所有后续任务都依赖 `load_manifest`/`save_manifest` 的签名与原子写语义。

- [x] **Step 1: 写失败测试**

创建 `KNN_evaluation/tests/test_manifest.py`（用 `tmp_path` 隔离，不触碰项目根）：

```python
"""Tests for KNN_evaluation.manifest."""
from pathlib import Path

from KNN_evaluation.manifest import load_manifest, save_manifest, update_manifest


def _sample_data():
    return {"collection": "c", "images": {"E121.4_N25.1": 16384}, "updated_at": "2026-08-02T00:00:00"}


class TestLoadManifest:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        data = load_manifest(tmp_path / "nope.json")
        assert data == {"collection": "", "images": {}, "updated_at": ""}

    def test_corrupt_file_returns_empty(self, tmp_path: Path):
        p = tmp_path / "m.json"
        p.write_text("{ not json !!", encoding="utf-8")
        assert load_manifest(p) == {"collection": "", "images": {}, "updated_at": ""}

    def test_roundtrip(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_manifest(_sample_data(), p)
        assert load_manifest(p) == _sample_data()


class TestUpdateManifest:
    def test_adds_and_updates(self, tmp_path: Path):
        p = tmp_path / "m.json"
        out = update_manifest("E121.4_N25.1", 16384, "c", p)
        assert out["images"]["E121.4_N25.1"] == 16384
        out = update_manifest("E121.4_N25.1", 500, "c", p)
        assert out["images"]["E121.4_N25.1"] == 500
        assert out["collection"] == "c"
        assert out["updated_at"]
        assert load_manifest(p)["images"]["E121.4_N25.1"] == 500

    def test_missing_file_creates(self, tmp_path: Path):
        p = tmp_path / "m.json"
        out = update_manifest("A", 10, "col", p)
        assert out["images"] == {"A": 10}

    def test_keeps_unrelated_keys(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_manifest(_sample_data(), p)
        out = update_manifest("E121.4_N25.1", 0, "c", p)
        assert set(out["images"]) == {"E121.4_N25.1"}


class TestAtomicWrite:
    def test_no_tmp_leftover(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_manifest(_sample_data(), p)
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_non_ascii_roundtrip(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_manifest({"collection": "中文名", "images": {}, "updated_at": "x"}, p)
        assert load_manifest(p)["collection"] == "中文名"
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_manifest.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'KNN_evaluation.manifest'`）

- [x] **Step 3: 最小实现**

创建 `KNN_evaluation/manifest.py`：

```python
"""导入 manifest：collection 级导入清单的读取/保存/增量更新（原子写）.

manifest 是可重建缓存，不是唯一真相；文件缺失/损坏时返回空结构，
由对账路径（reconcile_manifest）重建。并发写时最后一次写胜
（原子写 tmp + os.replace），下次启动对账自动纠正。
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

MANIFEST_PATH = Path("qdrant_import_manifest.json")


def _empty() -> dict:
    return {"collection": "", "images": {}, "updated_at": ""}


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """读取 manifest；文件缺失/损坏返回空结构（不报错）."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty()
        return {
            "collection": str(data.get("collection", "")),
            "images": dict(data.get("images") or {}),
            "updated_at": str(data.get("updated_at", "")),
        }
    except (OSError, ValueError):
        return _empty()


def save_manifest(data: dict, path: Path = MANIFEST_PATH) -> None:
    """原子写：tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_manifest(
    image_id: str,
    imported_pixels: int,
    collection: str,
    path: Path = MANIFEST_PATH,
) -> dict:
    """增量更新单张影像的已导入像素数（原子写），返回更新后的 manifest."""
    data = load_manifest(path)
    images = dict(data.get("images") or {})
    images[image_id] = int(imported_pixels)
    data = {
        "collection": collection,
        "images": images,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_manifest(data, path)
    return data
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_manifest.py -v`
Expected: PASS（15 个用例全过）

- [x] **Step 5: 更新 .gitignore**

在 `.gitignore` 末尾追加一行 `qdrant_import_manifest.json`（生成文件不入库）。

- [x] **Step 6: 提交**

```bash
git add KNN_evaluation/manifest.py KNN_evaluation/tests/test_manifest.py .gitignore
git commit -m "feat(manifest): 新增导入清单 manifest.py（原子写 + 缺失容错）"
```

**验证方式:** `uv run pytest KNN_evaluation/tests/test_manifest.py -v` 全绿；`.gitignore` 含 `qdrant_import_manifest.json`。

---

### Task 1.2: qdrant_client.py — get_imported_image_ids 读 manifest + reconcile_manifest(facet)

**Files:**
- Modify: `KNN_evaluation/qdrant_client.py`（`get_imported_image_ids` 现值在 :147；新增 `reconcile_manifest` / `_facet_image_ids`）
- Modify: `KNN_evaluation/tests/test_qdrant_client.py`

**Interfaces:**
- Consumes: `manifest.load_manifest` / `manifest.save_manifest`（Task 1.1）、`PixelDataLoader.check_image_count(image_id, manager) -> int`（`KNN_evaluation/data_loader.py:155`，既有）
- Produces:
  - `QdrantManager.get_imported_image_ids() -> set[str]`（改为从 manifest 读，无 manifest 时回退 `_facet_image_ids` 重建）
  - `QdrantManager.reconcile_manifest() -> dict`：`db_ids = _facet_image_ids()`；`manifest_ids` 为本地 manifest 集合；`db_ids == manifest_ids` 时原样返回；不一致时以 db 集合重建 `images`，**对差异项用 `check_image_count` 补精确像素数**（`db_ids` 减去 `manifest_ids` 的新增项 → count；`manifest_ids` 减去 `db_ids` 的消失项 → 移除），`collection` 用 `self.collection_name`，`updated_at` 用当前时间，`save_manifest` 后返回 `load_manifest()`
  - `QdrantManager._facet_image_ids(limit: int = 1000) -> set[str]`：`self.client.facet(collection_name=..., key="image_id", limit=limit, exact=True)` → `{v.value for v in resp.hits}`
- Consumed by: Task 2.1/2.3（webui）、Task 3.3（migrate）、Task 5.3

**Dependencies:** Task 1.1。这是 manifest 与 Qdrant 的桥接层；Web 任务 2.x 与 migrate 任务 3.3 都依赖它。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_qdrant_client.py` 追加（mock manifest IO 与 facet，不连真实 Qdrant）：

```python
"""新增：manifest 版 get_imported_image_ids + reconcile_manifest（facet 对账）."""
from KNN_evaluation import qdrant_client as qc
from KNN_evaluation.qdrant_client import QdrantManager


def _manager() -> QdrantManager:
    return QdrantManager(url="http://localhost:1", collection_name="c")


def _seed_manifest(monkeypatch, images: dict, collection="c"):
    """替换 qdrant_client 命名空间内的 load_manifest/save_manifest，避免真实文件."""
    state = {"collection": collection, "images": dict(images), "updated_at": "x"}
    monkeypatch.setattr(qc, "load_manifest", lambda path=None: dict(state))
    def _save(data, path=None):
        state.clear(); state.update(data)
    monkeypatch.setattr(qc, "save_manifest", _save)
    return state


class TestGetImportedImageIdsFromManifest:
    def test_reads_manifest(self, monkeypatch):
        _seed_manifest(monkeypatch, {"A": 16384, "B": 16384})
        m = _manager()
        assert m.get_imported_image_ids() == {"A", "B"}

    def test_missing_manifest_falls_back_to_facet(self, monkeypatch):
        _seed_manifest(monkeypatch, {})          # 空 manifest → 回退 facet
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"X", "Y"}
        assert m.get_imported_image_ids() == {"X", "Y"}


class TestReconcileManifest:
    def test_consistent_no_op(self, monkeypatch):
        _seed_manifest(monkeypatch, {"A": 16384})
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A"}
        out = m.reconcile_manifest()
        assert set(out["images"]) == {"A"}

    def test_adds_missing_manually_counted(self, monkeypatch):
        from KNN_evaluation.data_loader import PixelDataLoader
        _seed_manifest(monkeypatch, {})
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A", "B"}
        # 对差异项精确 count：A 完整导入补 16384；B 无像素（count=0）不写入
        monkeypatch.setattr(
            PixelDataLoader, "check_image_count",
            lambda iid, mgr: 16384 if iid == "A" else 0,
        )
        out = m.reconcile_manifest()
        assert out["images"].get("A") == 16384
        assert "B" not in out["images"]

    def test_removes_stale(self, monkeypatch):
        _seed_manifest(monkeypatch, {"A": 16384, "GHOST": 16384})
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A"}
        out = m.reconcile_manifest()
        assert set(out["images"]) == {"A"}

    def test_rebuild_from_empty(self, monkeypatch):
        from KNN_evaluation.data_loader import PixelDataLoader
        _seed_manifest(monkeypatch, {})
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A"}
        monkeypatch.setattr(PixelDataLoader, "check_image_count", lambda iid, mgr: 16384)
        out = m.reconcile_manifest()
        assert set(out["images"]) == {"A"}
```

> **实现前提**：`qdrant_client.py` 顶部用 `from KNN_evaluation.manifest import load_manifest, save_manifest`（模块局部符号），函数体内直接调用 `load_manifest()`/`save_manifest(...)`，测试才可 `monkeypatch.setattr(qc, "load_manifest", ...)`。**不要**在 `get_imported_image_ids`/`reconcile_manifest` 内用 `from ... import`（会把名字绑定到函数局部，patch 失效）。`PixelDataLoader.check_image_count` 在 `reconcile_manifest` 内函数级 import（避免模块顶部循环依赖），测试 patch `PixelDataLoader.check_image_count` 有效。`.gitignore` 保证 `qdrant_import_manifest.json` 不入库，测试不触碰项目根真实 manifest。

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_qdrant_client.py -v`
Expected: FAIL（新增用例：facet / manifest 路径未实现；既有用例不受影响）

- [x] **Step 3: 最小实现**

修改 `KNN_evaluation/qdrant_client.py`：

```python
# 顶部 import
from KNN_evaluation.manifest import MANIFEST_PATH, load_manifest, save_manifest

def get_imported_image_ids(self) -> set[str]:
    """从本地 manifest 读取已导入 image_id（毫秒级，Qdrant 离线可用）.

    无 manifest（空清单）时回退 facet 重建。
    """
    data = load_manifest()
    ids = set((data.get("images") or {}).keys())
    if not ids:
        ids = self._facet_image_ids()
    return ids

def _facet_image_ids(self, limit: int = 1000) -> set[str]:
    """用 facet 取 Collection 内去重 image_id 集合（单次请求，keyword 索引毫秒级）."""
    resp = self.client.facet(
        collection_name=self.collection_name,
        key="image_id",
        limit=limit,
        exact=True,
    )
    return {v.value for v in resp.hits}

def reconcile_manifest(self) -> dict:
    """用 facet 对账 manifest：一致不改、不一致刷新、缺失重建.

    对差异项用 check_image_count 补精确像素数（facet 只返回去重集合，不含像素数）。
    Returns: 对账后的 manifest dict.
    """
    from KNN_evaluation.data_loader import PixelDataLoader  # 函数内 import 防循环
    db_ids = self._facet_image_ids()
    current = load_manifest()
    manifest_ids = set((current.get("images") or {}).keys())

    if db_ids == manifest_ids:
        return current

    images: dict[str, int] = {}
    for iid in db_ids:
        if iid in manifest_ids:
            images[iid] = int((current.get("images") or {}).get(iid, 0))
        else:
            # 新增差异项：精确 count（复用 check_image_count）
            images[iid] = PixelDataLoader.check_image_count(iid, self)
    save_manifest({
        "collection": self.collection_name,
        "images": images,
        "updated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    })
    return load_manifest()
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_qdrant_client.py -v`
Expected: PASS（新增用例 + 既有用例全绿）

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/qdrant_client.py KNN_evaluation/tests/test_qdrant_client.py
git commit -m "feat(qdrant_client): get_imported_image_ids 读 manifest + facet 对账 reconcile_manifest"
```

**验证方式:** `uv run pytest KNN_evaluation/tests/test_qdrant_client.py -v` 全绿（mock 无真实 Qdrant 依赖）。

---

### Task 1.3: manifest 全量单元测试补齐（对账一致/不一致/重建）

**Files:**
- Modify: `KNN_evaluation/tests/test_manifest.py`
- Modify: `KNN_evaluation/tests/test_qdrant_client.py`（如 Step 1 中已有覆盖，则本任务只补遗漏）

**Interfaces:**
- Consumes: Task 1.1 的 `load_manifest`/`save_manifest`/`update_manifest`、Task 1.2 的 `reconcile_manifest`
- Produces: 无新接口，仅测试覆盖补全

**Dependencies:** Task 1.1、Task 1.2。

- [x] **Step 1: 补全测试用例**

在 `KNN_evaluation/tests/test_manifest.py` 追加直接对 `reconcile_manifest` 的纯 mock 覆盖（与 Task 1.2 的 mock 风格一致，确保"一致 / 不一致刷新 / 缺失重建"三条路径各有断言）；对 Task 1.2 已覆盖的场景只需补充：`save_manifest` 后目录自动创建、`update_manifest` 并发顺序写（两次调用后文件合法可读）。

- [x] **Step 2: 运行全部 manifest + qdrant_client 测试**

Run: `uv run pytest KNN_evaluation/tests/test_manifest.py KNN_evaluation/tests/test_qdrant_client.py -v`
Expected: PASS

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/tests/test_manifest.py KNN_evaluation/tests/test_qdrant_client.py
git commit -m "test(manifest): 补全对账一致/不一致/重建与原子写覆盖"
```

**验证方式:** 上述 pytest 命令全绿。这完成 tasks.md 的第 1 组（1.1-1.3）。

---

### Task 2.1: init_page 重构（快速路径 + 后台慢速路径）

**Files:**
- Modify: `KNN_evaluation/webui.py`（`init_page` 现值 :1153；`refresh_status` :199）

**Interfaces:**
- Consumes: `QdrantManager`（既有）、`manifest.load_manifest`（Task 1.1）、`PixelDataLoader.scan_directory`（既有）、`QdrantManager.reconcile_manifest`（Task 1.2）
- Produces:
  - `init_page() -> None`：快速路径（创建 manager → 置 state → `refresh_status()` → `await asyncio.sleep(0)` 让出事件循环 → `asyncio.create_task(_background_init())`），**不 await 慢速路径**
  - `_background_init() -> None`：`await asyncio.to_thread(_load_manifest_cached)` → 目录存在则 `await asyncio.to_thread(_scan_directory_only)` → `await asyncio.to_thread(_reconcile_background)` → `await _refresh_image_list()` → `await _render_preview()`；整体 try/except 吞异常（后台路径失败不阻塞页面）
- Consumed by: Task 2.4（缓存失效）、Task 2.5（webui 测试）

**Dependencies:** Task 1.1、Task 1.2。这是 P1 的核心改动，必须先有 manifest 与对账能力。

- [x] **Step 1: 写失败测试（先建 test_webui.py 骨架）**

创建 `KNN_evaluation/tests/test_webui.py`。**关键**：`webui.py` 只在 `__main__` 分支启动 NiceGUI，`import KNN_evaluation.webui` 不会触发 `ui.run`，可安全导入；但 `@ui.page` 装饰器会注册页面，需 mock `ui` 以隔离。测试只验证 `init_page` 快速路径时序（不跑真实 event loop 之外的完整 `_background_init`）：

```python
"""Tests for webui init_page fast path."""
import asyncio

from KNN_evaluation import webui


class TestInitPageFastPath:
    def test_fast_path_yields_before_slow_tasks(self, monkeypatch):
        # init_page 通过 asyncio.create_task 启动慢速路径，而非 await
        created: list[str] = []

        def fake_create_task(coro):
            created.append("task")
            return coro
        monkeypatch.setattr(asyncio, "create_task", fake_create_task)
        monkeypatch.setattr(webui, "_CLI_QDRANT_URL", "http://localhost:1")
        monkeypatch.setattr(webui, "_CLI_DATA_DIR", "data_demo")
        monkeypatch.setattr(webui, "_init_hooks",
                            {"refresh_status": lambda: None})  # 模块级钩子注册表

        async def run():
            await webui.init_page()

        # 快速路径在一次 sleep(0) 让出后完成，create_task 被调用（慢速路径后台执行）
        asyncio.run(run())
        assert created, "慢速路径应通过 asyncio.create_task 启动（后台执行）"
```

> **提示**：`init_page` 是模块级 async 函数（见 Step 3 钩子注册表），不依赖闭包；`_CLI_QDRANT_URL` / `_CLI_DATA_DIR` 需在模块级给出默认值（`_CLI_QDRANT_URL = QDRANT_URL`、`_CLI_DATA_DIR = _DEFAULT_DATA_DIR`，`__main__` 再覆盖），否则测试与 import 期引用会 NameError。`refresh_status` 留在 `index()` 闭包内、经 `_init_hooks` 注入，测试只补 `_init_hooks` 字典即可。

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: FAIL（`webui.init_page` 不存在 / 未提升为模块级）

- [x] **Step 3: 最小实现**

修改 `KNN_evaluation/webui.py`，采用**模块级钩子注册表**（模块级函数可测、闭包内 UI 控件仍由 `index()` 提供）：

1. 模块级新增（`index()` 之外）：
```python
# 进程级 manifest 缓存（P1：同会话多 tab 复用）
_manifest_cache: dict | None = None

def _get_manifest_cached() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = load_manifest()
    return _manifest_cache

def _invalidate_manifest_cache() -> None:
    global _manifest_cache
    _manifest_cache = None

# 模块级状态与钩子：模块级函数可测；index() 闭包在页面构建时注入闭包函数
state: dict = {}
_init_hooks: dict = {}     # 由 index() 填充：refresh_status / _refresh_image_list / _render_preview / scan_directory

# CLI 覆盖参数需有模块级默认（__main__ 启动时再覆盖），否则模块级函数与测试引用 NameError
_CLI_QDRANT_URL: str = QDRANT_URL
_CLI_DATA_DIR: str = _DEFAULT_DATA_DIR
```

2. 在 `import` 区补：`from KNN_evaluation.manifest import load_manifest`。

3. 模块级 `init_page` / `_background_init` / 后台辅助（状态与闭包函数经 `state`/`_init_hooks` 间接访问）：
```python
async def init_page() -> None:
    manager = QdrantManager(url=_CLI_QDRANT_URL)
    state["manager"] = manager
    state["data_dir"] = Path(_CLI_DATA_DIR)
    _init_hooks.get("refresh_status", lambda: None)()   # 健康检查 + 置 ✅（快速路径）
    await asyncio.sleep(0)               # 让出事件循环，状态 flush 到浏览器
    asyncio.create_task(_background_init())   # 慢速路径：不 await


async def _background_init() -> None:
    """后台慢速初始化：load_manifest → scan_directory → facet 对账 → 渲染."""
    async def _noop() -> None:
        return None

    try:
        await asyncio.to_thread(_load_manifest_cached)
        if state["data_dir"].exists():
            await asyncio.to_thread(_scan_directory_only)
        await asyncio.to_thread(_reconcile_background)
        await _init_hooks.get("refresh_image_list", _noop)()
        await _init_hooks.get("render_preview", _noop)()
    except Exception:
        pass


def _load_manifest_cached() -> None:
    _get_manifest_cached()


def _scan_directory_only() -> None:
    pairs = PixelDataLoader.scan_directory(state["data_dir"])
    state["file_pairs"] = pairs
    state["se_paths_map"] = {p.image_id: p.se_path for p in pairs}
    state["preview_page"] = 0


def _reconcile_background() -> None:
    global _manifest_cache
    mgr = state.get("manager")
    if mgr is not None and mgr.collection_exists():
        try:
            _manifest_cache = mgr.reconcile_manifest()
        except Exception:
            pass
```

4. `index()` 内部：页面构建时注入钩子并绑定模块级 `init_page`：
```python
_init_hooks["refresh_status"] = refresh_status          # 闭包内既有 refresh_status
_init_hooks["refresh_image_list"] = _refresh_image_list # 闭包内既有 async 版本
_init_hooks["render_preview"] = _render_preview         # 闭包内既有版本
ui.timer(0.1, callback=init_page, once=True)            # 引用模块级 init_page
```
（`state` 保持 `index()` 闭包内的 dict，同时赋给模块级 `state` 引用同一对象：`state = <闭包 dict>` 在 `index()` 首行执行；或直接删掉闭包里的 `state` 定义、统一用模块级 `state`。）

> **为什么用钩子注册表**：NiceGUI 的 `refresh_status` / `_render_preview` / `_refresh_image_list` 都捕获了闭包 UI 控件（`status_label`、`file_column`、`spec_image_select` 等），无法提到模块级。让模块级函数经 `_init_hooks` 间接调用闭包函数，既保持可测（测试替换 `_init_hooks` 或直接测 `_load_manifest_cached`/`_scan_directory_only`/`_reconcile_background`/`_get_manifest_cached`），又不破坏 UI 闭包捕获。Task 2.3/2.4/2.5 的测试均针对这些模块级辅助函数编写。

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: PASS

- [x] **Step 5: 手工冒烟（可选，需要真实 Qdrant）**

Run: `uv run python KNN_evaluation/webui.py --dir data_demo --port 8003`
Expected: 打开 `http://localhost:8003`，页面**秒级**出现 "✅ Qdrant 连接正常"；影像列表随后毫秒级填充；期间浏览器可交互（事件循环未堵死）。

- [x] **Step 6: 提交**

```bash
git add KNN_evaluation/webui.py KNN_evaluation/tests/test_webui.py
git commit -m "feat(webui): init_page 快速路径 + 后台慢速路径（load_manifest/scan/reconcile/render）"
```

**验证方式:** `uv run pytest KNN_evaluation/tests/test_webui.py -v` 通过；手工冒烟确认状态标签先于慢速路径置 ✅。

---

### Task 2.2: 阻塞调用统一封装 asyncio.to_thread

**Files:**
- Modify: `KNN_evaluation/webui.py`（`do_import` :233、`do_random_query` :428、`do_spec_query` :464、`do_search` :536 中的 manager 阻塞调用）

**Interfaces:**
- Consumes: 既有 manager 方法（`scroll`/`count`/`collection_info`）
- Produces: 无新接口；确保所有直接触达 `state["manager"].client.*` 或 `check_image_count` 的调用都在 `await asyncio.to_thread(...)` 内
- Consumed by: 本任务自身（webui 交互路径），无后续任务依赖

**Dependencies:** Task 2.1（引入 `_background_init` 的同时为其他交互路径确立 `to_thread` 模式）。逻辑独立，可与 Task 2.3 并行。

- [x] **Step 1: 审查并列出所有阻塞点**

Run: `rg -n "manager\.client|check_image_count|collection_info|\.scroll\(" KNN_evaluation/webui.py`
Expected: 定位 `do_random_query`（`client.scroll`）、`do_spec_query`（`client.scroll`）、`do_search`（`searcher.search` 或 scroll）、`do_import`（`import_directory` 已 `to_thread`）、`_render_preview`（`check_image_count`，将在 Task 2.3 改读 manifest）、`do_evaluate`（`to_thread` 已覆盖）。

- [x] **Step 2: 逐个用 to_thread 包裹**

对 `do_random_query` / `do_spec_query` 中的 scroll，改为：
```python
scroll_result = await asyncio.to_thread(
    state["manager"].client.scroll,
    collection_name=COLLECTION_NAME, limit=1, with_vectors=True,
)
```
`do_search` 中 `searcher.search` 改为 `await asyncio.to_thread(searcher.search, vector, k, exact)`（保持既有参数与返回）。保持既有通知逻辑不变。

- [x] **Step 3: 运行既有测试确认无回归**

Run: `uv run pytest KNN_evaluation/tests/ -v`
Expected: 既有测试全绿（webui 无覆盖测试，主要验证不破坏 import/模块加载）

- [x] **Step 4: 提交**

```bash
git add KNN_evaluation/webui.py
git commit -m "feat(webui): 检索/查询路径阻塞调用封装 asyncio.to_thread"
```

**验证方式:** `uv run pytest KNN_evaluation/tests/ -v` 全绿 + 手工交互（随机查询、指定像素、搜索）可响应。

---

### Task 2.3: _render_preview 改读 manifest（翻页不卡事件循环）

**Files:**
- Modify: `KNN_evaluation/webui.py`（`_render_preview` 现值 :314）

**Interfaces:**
- Consumes: `_get_manifest_cached()`（Task 2.1）、`manifest.load_manifest`
- Produces: `_render_preview() -> None`（内部 `count` 从 manifest 字典读取，不再逐条 `check_image_count`）；提供模块级辅助 `_manifest_pixels(image_id: str) -> int` 供测试
- Consumed by: Task 2.5（webui 测试断言翻页不触 manager）

**Dependencies:** Task 2.1（缓存 + 提模块级）。逻辑可独立完成。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_webui.py` 追加：

```python
class TestRenderPreviewFromManifest:
    def test_reads_manifest_not_manager(self, monkeypatch):
        from KNN_evaluation import webui
        manifest = {"collection": "c", "images": {"A": 16384, "B": 1200, "C": 0}, "updated_at": "x"}
        monkeypatch.setattr(webui, "_get_manifest_cached", lambda: manifest)
        monkeypatch.setattr(webui, "_render_preview", lambda: None)  # 避免依赖 UI
        assert webui._manifest_pixels("A") == 16384
        assert webui._manifest_pixels("B") == 1200
        assert webui._manifest_pixels("C") == 0
        assert webui._manifest_pixels("MISSING") == 0
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: FAIL（`_manifest_pixels` 未定义）

- [x] **Step 3: 最小实现**

修改 `_render_preview` 中 per-pair 状态判定：

```python
def _manifest_pixels(image_id: str) -> int:
    """从进程级 manifest 缓存读取已导入像素数（无则 0）."""
    data = _get_manifest_cached()
    return int((data.get("images") or {}).get(image_id, 0))

def _render_preview():
    file_column.clear()
    pairs = state["file_pairs"]
    total = len(pairs)
    ...
    for pair in paginate_slice(pairs, page, PAGE_SIZE):
        count = _manifest_pixels(pair.image_id)   # 不再调用 check_image_count / 不触 manager
        status = (
            "✅ 已导入" if count >= 16384
            else f"⏳ {count}/16384" if count > 0
            else "📦 待导入"
        )
        ...
```

（`state["manager"].collection_exists()` 分支整体删除：预览状态列只反映 manifest。）

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/webui.py KNN_evaluation/tests/test_webui.py
git commit -m "feat(webui): _render_preview 改读 manifest 缓存，翻页不触 manager"
```

**验证方式:** 单测通过 + 手工翻页 624 张影像列表无卡顿、无 HTTP scroll。

---

### Task 2.4: 会话缓存 + 导入后失效刷新

**Files:**
- Modify: `KNN_evaluation/webui.py`（`_manifest_cache`、`do_import` 完成处 :281、`_refresh_image_list` :404）

**Interfaces:**
- Consumes: `_get_manifest_cached` / `_invalidate_manifest_cache`（Task 2.1）、`manifest.load_manifest`
- Produces:
  - `_refresh_image_list()` 改为 `async def`，内部用 `state["manager"].get_imported_image_ids()`（Task 1.2，读 manifest）填充 `spec_image_select`；若 manager 为空或不含 collection，仅用 `file_pairs`（加 " (未导入)" 后缀）
- Consumed by: Task 2.5（断言 do_import 后缓存失效 + `_refresh_image_list` 不触发全库 scroll）

**Dependencies:** Task 2.1、Task 2.3、Task 1.2。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_webui.py` 追加（对齐 Step 3 的模块级 `_imported_image_ids` 辅助，避免测试依赖闭包中的 UI 控件）：

```python
class TestCacheInvalidation:
    def test_invalidate_resets_cache(self, monkeypatch):
        from KNN_evaluation import webui
        monkeypatch.setattr(webui, "_manifest_cache", {"x": 1})
        webui._invalidate_manifest_cache()
        assert webui._manifest_cache is None

    def test_imported_image_ids_reads_manifest(self, monkeypatch):
        from KNN_evaluation import webui
        manifest = {"collection": "c", "images": {"A": 16384, "B": 16384}, "updated_at": "x"}
        monkeypatch.setattr(webui, "_get_manifest_cached", lambda: manifest)
        assert webui._imported_image_ids() == {"A", "B"}

    def test_imported_image_ids_empty_manifest(self, monkeypatch):
        from KNN_evaluation import webui
        monkeypatch.setattr(webui, "_get_manifest_cached", lambda: {"images": {}, "collection": "", "updated_at": ""})
        assert webui._imported_image_ids() == set()
```

> **实现约束**：`_refresh_image_list` 是 `index()` 内的 `async` 闭包（依赖 `spec_image_select`），测试不直接调用。把"读取已导入 image_id 集合"逻辑抽为模块级 `_imported_image_ids() -> set[str]`（见 Step 3），测试只验证该辅助读 manifest / 不触发全库 scroll。

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: FAIL

- [x] **Step 3: 最小实现**

```python
def _imported_image_ids() -> set[str]:
    """读 manifest 的 image_id 集合（无 manager 时返回空集）."""
    return set((_get_manifest_cached().get("images") or {}).keys())

# index() 内：
async def _refresh_image_list():
    image_options = {}
    if state["manager"] and state["manager"].collection_exists():
        try:
            imported = await asyncio.to_thread(state["manager"].get_imported_image_ids)
            for img_id in sorted(imported):
                image_options[img_id] = img_id
        except Exception:
            pass
    for pair in state.get("file_pairs", []):
        if pair.image_id not in image_options:
            image_options[pair.image_id] = pair.image_id + " (未导入)"
    spec_image_select.set_options(image_options or {"": "无可用影像"}, value=None)
```

`do_import` 完成处（`refresh_status()` 之后）追加：
```python
_invalidate_manifest_cache()
await browse_directory()
```

`_refresh_image_list()` 的所有既有调用点补 `await`（`browse_directory` 内、`init_page` 慢速路径等）。

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/webui.py KNN_evaluation/tests/test_webui.py
git commit -m "feat(webui): 会话缓存 + 导入后失效刷新，_refresh_image_list 异步化"
```

**验证方式:** 单测通过 + 手工：导入一张影像后影像下拉立即出现该影像（缓存已失效并刷新）。

---

### Task 2.5: test_webui.py 集成断言（init_page 快速路径 + 事件循环不被 scroll 堵死）

**Files:**
- Modify: `KNN_evaluation/tests/test_webui.py`

**Interfaces:**
- Consumes: Task 2.1-2.4 产出的模块级函数（`init_page`、`_background_init`、`_get_manifest_cached`、`_imported_image_ids`）
- Produces: 无新接口，完整集成断言

**Dependencies:** Task 2.1、2.3、2.4 全部完成后才可写完整断言。

- [x] **Step 1: 写集成测试**

在 `KNN_evaluation/tests/test_webui.py` 追加（mock 规避闭包 UI 依赖与真实 Qdrant）：

```python
class TestInitPageIntegration:
    def test_imported_ids_reads_manifest_not_scroll(self, monkeypatch):
        from KNN_evaluation import webui
        manifest = {"collection": "c", "images": {"A": 16384}, "updated_at": "x"}
        monkeypatch.setattr(webui, "_get_manifest_cached", lambda: manifest)
        assert webui._imported_image_ids() == {"A"}

    def test_init_page_schedules_slow_path_not_awaits(self, monkeypatch):
        from KNN_evaluation import webui
        started: list[str] = []
        monkeypatch.setattr(asyncio, "create_task", lambda c: (started.append("scheduled"), c)[1])
        monkeypatch.setattr(webui, "_init_hooks", {"refresh_status": lambda: None})
        monkeypatch.setattr(webui, "_CLI_QDRANT_URL", "http://localhost:1")
        monkeypatch.setattr(webui, "_CLI_DATA_DIR", "data_demo")

        async def run():
            await webui.init_page()

        asyncio.run(run())
        assert "scheduled" in started, "init_page 必须通过 create_task 启动慢速路径而非 await"

    def test_slow_path_does_not_call_get_imported_image_ids(self, monkeypatch):
        from KNN_evaluation import webui
        calls: list[str] = []
        monkeypatch.setattr(webui, "_load_manifest_cached", lambda: None)
        monkeypatch.setattr(webui, "_scan_directory_only", lambda: None)
        monkeypatch.setattr(webui, "_reconcile_background", lambda: None)
        monkeypatch.setattr(webui, "_init_hooks", {
            "refresh_status": lambda: None,
            "refresh_image_list": lambda: calls.append("refresh"),
            "render_preview": lambda: calls.append("render"),
        })
        asyncio.run(webui._background_init())
        assert calls == ["refresh", "render"]
```

> **说明**：`_background_init` / `init_page` 均为模块级 async 函数，经 `_init_hooks` 间接调用闭包内 UI 步骤（Task 2.1 钩子注册表）。测试替换 `_init_hooks` 字典即可验证慢速路径顺序，且不触碰真实 `state` / `spec_image_select`。若钩子缺失，`_background_init` 会 KeyError——实现须在 `_background_init` 内对缺失钩子容错（`_init_hooks.get("refresh_image_list", noop)`）。

- [x] **Step 2: 运行全部 webui 测试**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: PASS（含 Task 2.1/2.3/2.4 用例）

- [x] **Step 3: 运行全套回归**

Run: `uv run pytest KNN_evaluation/tests/ -v`
Expected: 全绿

- [x] **Step 4: 提交**

```bash
git add KNN_evaluation/tests/test_webui.py
git commit -m "test(webui): init_page 快速路径时限与不触发全库 scroll 断言"
```

**验证方式:** 全套测试通过；这完成 tasks.md 的第 2 组（2.1-2.5）。

---

### Task 3.1: create_collection 增加 storage 参数（默认 disk）

**Files:**
- Modify: `KNN_evaluation/qdrant_client.py`（`create_collection` 现值 :48）

**Interfaces:**
- Consumes: 既有 `models.VectorParams` / `models.HnswConfigDiff`
- Produces:
  - `QdrantManager.create_collection(vector_size: int = VECTOR_SIZE, m: int = HNSW_M, ef_construct: int = HNSW_EF_CONSTRUCT, storage: str = "disk") -> None`
  - `storage="disk"`：向量 `on_disk=True`、`on_disk_payload=True`、`quantization_config=None`（不引入量化）；`storage="ram"`：`on_disk=False`、`on_disk_payload=False`（保持现状）
  - 非法 storage 值抛 `ValueError`（`choices=("disk", "ram")`）
- Consumed by: Task 3.3（migrate）、Task 3.4（单测）、webui `_create_collection` 与 `do_import` 自动创建（用默认 disk）

**Dependencies:** 无强依赖（可并行于 Task 3.5）。P2 存储基础。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_qdrant_client.py` 追加（mock client，断言调用参数）：

```python
class TestCreateCollectionStorage:
    def _make_manager(self, monkeypatch):
        from KNN_evaluation.qdrant_client import QdrantManager
        m = QdrantManager(url="http://localhost:1", collection_name="c")
        monkeypatch.setattr(m, "collection_exists", lambda: False)
        calls: dict = {}

        def fake_create_collection(**kwargs):
            calls.update(kwargs)

        class _C:
            create_collection = staticmethod(fake_create_collection)

        monkeypatch.setattr(m, "client", _C())   # client 是属性；为实例设 client 需改实现为可赋值，或 patch 属性
        return m, calls
```

> **实现提示**：`QdrantManager.client` 是只读 `@property`（`KNN_evaluation/qdrant_client.py:23`），`monkeypatch.setattr(m, "client", _C())` 对实例属性不可行。改用类级 patch：`monkeypatch.setattr(QdrantManager, "client", property(lambda self: _C()))`；或将 `_C.create_collection` 挂到真实 client 类型上。**推荐**：直接 patch `m` 的属性为 mock 前，先在实现中给 `client` 增加可写性（改 `@property` 为 `@property + setter`），否则测试改 patch `QdrantManager.client` 类属性（见下）：

```python
    def test_default_is_disk(self, monkeypatch):
        m, calls = self._make_manager(monkeypatch)
        m.create_collection()
        vp = calls["vectors_config"]
        assert vp.on_disk is True
        assert calls.get("on_disk_payload") is True
        assert calls.get("quantization_config") is None
```

（`test_ram_preset` / `test_invalid_storage` 用例主体与上方相同，`storage="ram"` 断言 `on_disk is False` + `on_disk_payload is False`；`storage="ssd"` 断言抛 `ValueError`。若 `client` 只读属性无法注入 mock，`_make_manager` 改用 `monkeypatch.setattr(QdrantManager, "client", property(...))`。）

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_qdrant_client.py -v`
Expected: FAIL（无 `storage` 参数 / 断言不满足）

- [x] **Step 3: 最小实现**

```python
def create_collection(
    self,
    vector_size: int = VECTOR_SIZE,
    m: int = HNSW_M,
    ef_construct: int = HNSW_EF_CONSTRUCT,
    storage: str = "disk",
) -> None:
    """创建 Collection 并配置 HNSW 索引参数.

    若 Collection 已存在则跳过.
    storage: "disk" 时向量与 payload 落盘（on_disk=True/on_disk_payload=True，不量化），
             常驻内存 ≈2-3GB；"ram" 保持全内存（on_disk=False，现状）.
    """
    if storage not in ("disk", "ram"):
        raise ValueError(f"storage 必须是 'disk' 或 'ram'，实际: {storage!r}")
    if self.collection_exists():
        return

    disk = storage == "disk"
    self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
            on_disk=disk,
        ),
        hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
        on_disk_payload=disk,
        quantization_config=None,
    )
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_qdrant_client.py -v`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/qdrant_client.py KNN_evaluation/tests/test_qdrant_client.py
git commit -m "feat(qdrant_client): create_collection 增加 storage 参数（默认 disk 磁盘化）"
```

**验证方式:** 单测断言 `on_disk=True` / `on_disk_payload=True` / `quantization_config=None`（mock，无真实 Qdrant）。

---

### Task 3.2: 磁盘化参数单测 + 既有路径回归

**Files:**
- Modify: `KNN_evaluation/tests/test_qdrant_client.py`

**Interfaces:**
- Consumes: Task 3.1 的 `create_collection(storage=...)`
- Produces: 无新接口

**Dependencies:** Task 3.1。若 Task 3.1 的 Step 1 已含全部断言，本任务仅补 `HNSW` 参数不变与 `ram` 兼容性断言。

- [x] **Step 1: 补测 HNSW 与 ram 兼容**

追加断言：`storage="disk"` 时 `hnsw_config.m == HNSW_M`、`ef_construct == HNSW_EF_CONSTRUCT` 与现状一致；`storage="ram"` 生成的调用参数与改造前完全一致（`on_disk=False` + `on_disk_payload=False`）。

- [x] **Step 2: 运行测试**

Run: `uv run pytest KNN_evaluation/tests/test_qdrant_client.py KNN_evaluation/tests/conftest.py::qdrant_manager -v`
Expected: PASS（conftest 的 `create_collection()` 无参调用走默认 disk，fixture 已兼容）

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/tests/test_qdrant_client.py
git commit -m "test(qdrant_client): 磁盘化参数与 ram 兼容性回归断言"
```

**验证方式:** 测试全绿；确认 conftest session fixture 在 disk 默认下仍可创建测试 collection（如环境无 Docker 则标记 skip）。

---

### Task 3.3: migrate CLI 子命令（备份→删除重建→重导→重建 manifest）

**Files:**
- Modify: `KNN_evaluation/cli.py`（`main()` 现值 :485、新增 `cmd_migrate`）
- Modify: `KNN_evaluation/qdrant_client.py`（如 `cmd_migrate` 需 `collection_info`，已有 :170）

**Interfaces:**
- Consumes: `QdrantManager`（既有）、`_start_qdrant()`（Task 3.5）、`PixelImporter.import_directory(..., no_resume=..., reindex=True)`（既有）、`QdrantManager.reconcile_manifest()`（Task 1.2）、`QdrantManager.create_collection(storage=...)`（Task 3.1）
- Produces:
  - `sub` parser `migrate`：`--dir`（默认 `"data_demo"`）、`--storage`（`choices=["disk","ram"]`，默认 `"disk"`）、`--no-resume`（store_true）、`--qdrant-url`（默认 `QDRANT_URL`）
  - `cmd_migrate(args) -> int`：health_check 失败 → 调 `_start_qdrant()` 幂等后再查，仍失败返回 1；`collection_info()` 备份旧统计 → `delete_collection` → `create_collection(storage=args.storage)` → `create_payload_indices()` → `migrate_image_id_index()` → `import_directory(Path(args.dir), no_resume=args.no_resume, reindex=True)` → `reconcile_manifest()` → 打印 `old → new` 点数，返回 0
  - `main()` 新增 `elif args.command == "migrate": return cmd_migrate(args)`
- Consumed by: Task 3.4（单测）、Task 5.2（集成验证）

**Dependencies:** Task 1.2（reconcile_manifest）、Task 3.1（create_collection storage）、Task 3.5（_start_qdrant）。CLI 编排层，最后接。

- [x] **Step 1: 实现 cmd_migrate**

在 `KNN_evaluation/cli.py` 新增（参考 Design Doc §4.5，migrate 开头自动调用幂等 `_start_qdrant()`）：

```python
def cmd_migrate(args) -> int:
    """重建 Collection 为指定存储配置并重导数据（幂等可重试）."""
    manager = QdrantManager(url=args.qdrant_url)
    if not manager.health_check():
        _start_qdrant()                 # 幂等：确保容器挂 volume 就绪
        if not manager.health_check():
            print("Qdrant 不可达", file=sys.stderr)
            return 1
    old_info = manager.collection_info() if manager.collection_exists() else None
    if manager.collection_exists():
        manager.client.delete_collection(manager.collection_name)
    manager.create_collection(storage=args.storage)
    manager.create_payload_indices()
    manager.migrate_image_id_index()
    importer = PixelImporter(manager)
    stats = importer.import_directory(
        Path(args.dir), no_resume=args.no_resume, reindex=True,
    )
    manager.reconcile_manifest()        # 重建 manifest
    new_info = manager.collection_info()
    print(
        f"迁移完成: {old_info['total_points'] if old_info else 0:,} "
        f"→ {new_info['total_points']:,}"
    )
    return 0
```

`main()` 中注册子命令（放在 evaluate 之后）：
```python
p_migrate = sub.add_parser("migrate", help="重建 Collection 为指定存储配置并重导数据")
p_migrate.add_argument("--dir", default="data_demo", help="数据根目录")
p_migrate.add_argument("--storage", choices=["disk", "ram"], default="disk",
                       help="新 Collection 存储预设 (默认: disk)")
p_migrate.add_argument("--no-resume", action="store_true", help="强制重新导入")
p_migrate.add_argument("--qdrant-url", default=QDRANT_URL,
                       help=f"Qdrant 服务地址 (默认: {QDRANT_URL})")
```
并在 dispatch 加 `elif args.command == "migrate": return cmd_migrate(args)`。

- [x] **Step 2: 冒烟（真实 Qdrant 可选）**

Run: `uv run python -m KNN_evaluation.cli migrate --dir data_demo --help`
Expected: 帮助信息含 `--storage`/`--no-resume`（不连接 Qdrant）。

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/cli.py
git commit -m "feat(cli): 新增 migrate 子命令（备份→删除重建→重导→重建 manifest）"
```

**验证方式:** `--help` 冒烟 + 单元测试（Task 3.4）。

---

### Task 3.4: migrate 幂等单测

**Files:**
- Create: `KNN_evaluation/tests/test_migrate.py`

**Interfaces:**
- Consumes: `cmd_migrate`（Task 3.3）、`_start_qdrant`（Task 3.5）
- Produces: 无新接口

**Dependencies:** Task 3.3、Task 3.5（mock 版 `_start_qdrant`）。

- [x] **Step 1: 写测试（mock 所有 Qdrant/Docker 交互）**

创建 `KNN_evaluation/tests/test_migrate.py`：

```python
"""Tests for cli.cmd_migrate idempotency (all mocks, no real Qdrant)."""
from KNN_evaluation import cli


class _Args:
    def __init__(self, storage="disk", no_resume=False, dir="data_demo",
                 qdrant_url="http://localhost:1"):
        self.storage = storage
        self.no_resume = no_resume
        self.dir = dir
        self.qdrant_url = qdrant_url


class _FakeClient:
    def __init__(self):
        self.deleted = []

    def delete_collection(self, name):
        self.deleted.append(name)


class _FakeManager:
    collection_name = "c"

    def __init__(self, url=None, collection_name="c", timeout=10, existed=True):
        self.url = url
        self.collection_name = collection_name
        self.existed = existed
        self.client = _FakeClient()
        self.created_storage = None

    def health_check(self):
        return True

    def collection_exists(self):
        return self.existed

    def collection_info(self):
        return {"total_points": 1000}

    def create_collection(self, storage="disk"):
        self.created_storage = storage

    def create_payload_indices(self):
        pass

    def migrate_image_id_index(self):
        pass

    def reconcile_manifest(self):
        return {"collection": self.collection_name, "images": {}, "updated_at": "x"}


class _FakeImporter:
    def __init__(self, manager, batch_size=None):
        self.manager = manager

    def import_directory(self, data_dir, no_resume=False, reindex=False):
        return {"total_pixels": 1000, "imported_images": 1, "skipped_images": 0}


def _patch(monkeypatch, existed=True) -> _FakeManager:
    mgr = _FakeManager(existed=existed)
    monkeypatch.setattr(cli, "QdrantManager", lambda url=None, collection_name="c", timeout=10: mgr)
    monkeypatch.setattr(cli, "PixelImporter", _FakeImporter)
    monkeypatch.setattr(cli, "_start_qdrant", lambda: True)
    return mgr


def test_migrate_existing_collection_recreates(monkeypatch):
    mgr = _patch(monkeypatch, existed=True)
    assert cli.cmd_migrate(_Args(storage="disk")) == 0
    assert mgr.client.deleted == ["c"]          # 删除重建
    assert mgr.created_storage == "disk"        # 新存储预设


def test_migrate_missing_collection_skips_delete(monkeypatch):
    mgr = _patch(monkeypatch, existed=False)
    assert cli.cmd_migrate(_Args(storage="disk")) == 0
    assert mgr.client.deleted == []             # 不存在则无删除
    assert mgr.created_storage == "disk"


def test_migrate_no_resume_partial(monkeypatch):
    _patch(monkeypatch, existed=True)
    assert cli.cmd_migrate(_Args(no_resume=True)) == 0


def test_migrate_idempotent_rerun(monkeypatch):
    _patch(monkeypatch, existed=True)
    assert cli.cmd_migrate(_Args()) == 0
    assert cli.cmd_migrate(_Args()) == 0        # 二次执行不抛错（幂等可重试）
```

> **实现前提**：`cmd_migrate` 内 `QdrantManager(...)`、`PixelImporter(manager)` 必须解析到 `cli` 模块命名空间（顶部 `from KNN_evaluation.qdrant_client import QdrantManager` / `from KNN_evaluation.importer import PixelImporter`），测试才可 patch `cli.QdrantManager` / `cli.PixelImporter`；`_start_qdrant` 同理须在 `cli` 命名空间（Task 3.3 在 cli.py 顶部加 `from KNN_evaluation.webui import _start_qdrant`，无循环依赖——webui 不 import cli）。

- [x] **Step 2: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_migrate.py -v`
Expected: PASS（幂等：重复执行 `cmd_migrate` 不抛错；删除前记录旧统计；迁移后 `check_image_count`/断点续传正常——断点续传验证由 `import_directory` 既有测试承担，此处只验证编排顺序）

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/tests/test_migrate.py
git commit -m "test(cli): migrate 幂等单测（已存在/不存在/部分导入三态）"
```

**验证方式:** `uv run pytest KNN_evaluation/tests/test_migrate.py -v` 全绿（全 mock）。

---

### Task 3.5: _start_qdrant() 幂等三态 + Docker volume

**Files:**
- Modify: `KNN_evaluation/webui.py`（`_start_qdrant` 现值 :141）
- Modify: `KNN_evaluation/tests/test_qdrant_client.py` 或新建 `KNN_evaluation/tests/test_start_qdrant.py`

**Interfaces:**
- Consumes: `subprocess.run`（既有）
- Produces:
  - `_start_qdrant() -> bool`：幂等三态——`docker ps --format "{{.Names}}"` 含 `qdrant` → 复用返回 True；停止则 `docker start qdrant`；不存在则 `docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest`；任何失败返回 False（不抛异常）
- Consumed by: Task 3.3（migrate 开头）、Task 5.2

**Dependencies:** 独立（可与 Task 3.1-3.4 并行）。

- [x] **Step 1: 写失败测试**

创建 `KNN_evaluation/tests/test_start_qdrant.py`：

```python
"""Tests for _start_qdrant idempotent three-state behavior."""
import subprocess

from KNN_evaluation import webui


def _run(monkeypatch, docker_ps_out, expected_cmds):
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "ps":
            return type("R", (), {"stdout": docker_ps_out})()
        return type("R", (), {"stdout": ""})()
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = webui._start_qdrant()
    assert result is True
    assert calls == expected_cmds


def test_running_container_reused(monkeypatch):
    _run(monkeypatch, "qdrant\n", [["docker", "ps", "--format", "{{.Names}}"]])


def test_stopped_container_started(monkeypatch):
    _run(monkeypatch, "", [["docker", "ps", "--format", "{{.Names}}"],
                          ["docker", "start", "qdrant"]])


def test_missing_container_run_with_volume(monkeypatch):
    _run(monkeypatch, "", [["docker", "ps", "--format", "{{.Names}}"],
                           ["docker", "run", "-d", "--name", "qdrant",
                            "-p", "6333:6333", "-p", "6334:6334",
                            "-v", "qdrant_data:/qdrant/storage",
                            "qdrant/qdrant:latest"]])


def test_failure_returns_false(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.SubprocessError("boom")
    monkeypatch.setattr(subprocess, "run", boom)
    assert webui._start_qdrant() is False
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_start_qdrant.py -v`
Expected: FAIL（现有 `_start_qdrant` 无三态）

- [x] **Step 3: 最小实现**

```python
def _start_qdrant() -> bool:
    """幂等启动 Qdrant Docker 容器（挂 volume）.

    三态：docker ps 判断 → 运行中复用 / 停止则 docker start /
    不存在则 docker run（-v qdrant_data:/qdrant/storage）.
    失败返回 False，不影响既有 WebUI（仅状态区提示）.
    """
    try:
        ps = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=15,
        )
        names = ps.stdout.split()
        if "qdrant" in names:
            return True
        started = subprocess.run(
            ["docker", "start", "qdrant"], capture_output=True, timeout=30,
        )
        if started.returncode == 0:
            return True
        # 容器不存在（start 失败）→ 全新 run
        run_rc = subprocess.run(
            [
                "docker", "run", "-d", "--name", "qdrant",
                "-p", "6333:6333", "-p", "6334:6334",
                "-v", "qdrant_data:/qdrant/storage",
                "qdrant/qdrant:latest",
            ],
            capture_output=True, timeout=60,
        )
        return run_rc.returncode == 0
    except Exception:
        return False
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_start_qdrant.py -v`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/webui.py KNN_evaluation/tests/test_start_qdrant.py
git commit -m "feat(webui): _start_qdrant 幂等三态 + Docker volume 持久化"
```

**验证方式:** 三态 mock 单测通过 + 手工 `uv run python KNN_evaluation/webui.py`（状态区提示正常）。

---

### Task 3.6: Docker 幂等单测 + README 更新

**Files:**
- Modify: `KNN_evaluation/tests/test_start_qdrant.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 3.5 的 `_start_qdrant`
- Produces: 无新接口

**Dependencies:** Task 3.5。

- [x] **Step 1: 补容器三态行为与 volume 断言**

在 `KNN_evaluation/tests/test_start_qdrant.py` 追加：`docker ps` 输出含两个名称（`qdrant` + 其他）时复用；`docker start` 失败（returncode!=0）时回退 `docker run`；`docker run` 命令**必须**含 `-v qdrant_data:/qdrant/storage` 参数。

- [x] **Step 2: 更新 README.md**

找到启动命令区块，替换为（Design Doc §4.7）：
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
```
补充「数据持久化」小节：容器数据挂载到 `qdrant_data` 命名 volume；描述 `_start_qdrant()` 幂等启动；`migrate --storage disk` 用法示例与回滚说明（迁移失败保留旧数据，删除前记录 collection_info）。

- [x] **Step 3: 运行测试**

Run: `uv run pytest KNN_evaluation/tests/test_start_qdrant.py -v`
Expected: PASS

- [x] **Step 4: 提交**

```bash
git add KNN_evaluation/tests/test_start_qdrant.py README.md
git commit -m "docs(README): Docker volume 启动命令与 migrate/持久化说明"
```

**验证方式:** 单测断言 `-v qdrant_data:/qdrant/storage` 存在于 run 命令；README 命令可直接复制执行。这完成 tasks.md 的第 3 组（3.1-3.6）。

---

### Task 4.1: metrics use_batch 默认改 False（服务端逐条 exact）

**Files:**
- Modify: `KNN_evaluation/metrics.py`（`compute_knn_accuracy` :116、`compute_purity_recall_curve` :328 的 `use_batch: bool = True` → `False`）

**Interfaces:**
- Consumes: 既有函数体
- Produces:
  - `compute_knn_accuracy(..., use_batch: bool = False, ...) -> dict`
  - `compute_purity_recall_curve(..., use_batch: bool = False, ...) -> dict`
  - 默认走服务端逐条 exact（`PixelSearcher` 路径），批量路径仅显式 `use_batch=True` 时启用
- Consumed by: Task 4.3（CLI `--batch`）、Task 4.4（WebUI）、Task 4.5（单测）

**Dependencies:** 无强依赖（纯默认值变更，兼容性风险最低）。P3 基础。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_metrics.py` 追加：

```python
class TestUseBatchDefault:
    def test_default_is_false(self):
        import inspect
        from KNN_evaluation import metrics
        sig1 = inspect.signature(metrics.compute_knn_accuracy)
        sig2 = inspect.signature(metrics.compute_purity_recall_curve)
        assert sig1.parameters["use_batch"].default is False
        assert sig2.parameters["use_batch"].default is False
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: FAIL（当前默认 `True`）

- [x] **Step 3: 最小实现**

两处签名 `use_batch: bool = True` 改为 `use_batch: bool = False`，并更新 docstring（"True 使用客户端批量矩阵乘法（默认）" → "False 默认服务端逐条 exact；True 显式开启客户端批量矩阵乘法"）。

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS（既有批量用例显式传 `use_batch=True` 不受影响；新增默认值断言通过）

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/metrics.py KNN_evaluation/tests/test_metrics.py
git commit -m "feat(metrics): use_batch 默认改 False（默认服务端逐条 exact，内存安全）"
```

**验证方式:** 单测断言默认值；既有显式 `use_batch=True` 用例全绿。

---

### Task 4.2: 内存预估守卫函数 estimate_batch_memory / guard_batch_memory

**Files:**
- Modify: `KNN_evaluation/metrics.py`

**Interfaces:**
- Consumes: `QdrantManager.collection_info()`（既有）
- Produces:
  - `estimate_batch_memory(n_points: int, n_queries: int, k: int) -> dict`：返回 `{"all_vecs_gb", "a_norm_gb", "topk_gb", "labels_ids_gb", "total_gb"}`；公式 `all_vecs = N×64×8B`、`a_norm = N×64×8B`、`topk = Q×K×8B`、`labels_ids = N×8B`，total = 四者之和
  - `guard_batch_memory(manager: QdrantManager, n_queries: int, k: int, max_ram_gb: float = 6.0) -> dict`：`n = manager.collection_info()["total_points"]`；`est = estimate_batch_memory(n, n_queries, k)`；`est["total_gb"] > max_ram_gb` 时抛 `MemoryError`（含预估与提示"请降采样或使用默认服务端逐条路径"）；否则返回 `est`
- Consumed by: Task 4.3（CLI）、Task 4.4（WebUI）、Task 4.5（单测）

**Dependencies:** 独立于 Task 4.1（可与 4.1 并行，均只改 metrics.py）。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_metrics.py` 追加：

```python
class TestBatchMemoryGuard:
    def test_estimate_10m_points(self):
        from KNN_evaluation.metrics import estimate_batch_memory
        est = estimate_batch_memory(10_000_000, 100, 1000)
        # N×64×8B ×2 ≈ 10.24GB（10M 点时文档值 ≈10.4GB）
        assert est["total_gb"] > 10.0
        assert est["all_vecs_gb"] == round(10_000_000 * 64 * 8 / 1e9, 2)

    def test_guard_below_threshold_ok(self, monkeypatch):
        from KNN_evaluation.metrics import guard_batch_memory
        m = type("M", (), {"collection_info": lambda self: {"total_points": 100_000}})()
        est = guard_batch_memory(m, 100, 100, max_ram_gb=6.0)
        assert est["total_gb"] < 6.0

    def test_guard_above_threshold_raises(self, monkeypatch):
        from KNN_evaluation.metrics import guard_batch_memory
        m = type("M", (), {"collection_info": lambda self: {"total_points": 10_000_000}})()
        try:
            guard_batch_memory(m, 100, 1000, max_ram_gb=6.0)
            raise AssertionError("应抛 MemoryError")
        except MemoryError as e:
            assert "10" in str(e)  # 消息含预估 GB 值
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: FAIL（函数未定义）

- [x] **Step 3: 最小实现**

```python
def estimate_batch_memory(n_points: int, n_queries: int, k: int) -> dict:
    """估算批量路径客户端进程峰值（GB）.

    公式（Design Doc §3.2）：
      all_vecs  = N × 64 × 8B      _scroll_full_vectors 全量向量
      a_norm    = N × 64 × 8B      _batch_exact_knn 归一化副本
      topk      = Q × K × 8B       Top-K 索引数组
      labels_ids = N × 8B          标签与 point_id 数组
    @10M 点、Q×K 较小 → ≈10.4GB.
    """
    n = max(int(n_points), 0)
    q = max(int(n_queries), 0)
    kk = max(int(k), 1)
    all_vecs = n * 64 * 8
    a_norm = n * 64 * 8
    topk = q * kk * 8
    labels_ids = n * 8
    total = all_vecs + a_norm + topk + labels_ids
    return {
        "all_vecs_gb": round(all_vecs / 1e9, 2),
        "a_norm_gb": round(a_norm / 1e9, 2),
        "topk_gb": round(topk / 1e9, 2),
        "labels_ids_gb": round(labels_ids / 1e9, 2),
        "total_gb": round(total / 1e9, 2),
    }


def guard_batch_memory(
    manager: QdrantManager,
    n_queries: int,
    k: int,
    max_ram_gb: float = 6.0,
) -> dict:
    """批量路径内存守卫：预估超过阈值抛 MemoryError（含预估）."""
    n = manager.collection_info()["total_points"]
    est = estimate_batch_memory(n, n_queries, k)
    if est["total_gb"] > max_ram_gb:
        raise MemoryError(
            f"批量路径预估峰值 {est['total_gb']:.1f}GB 超阈值 {max_ram_gb}GB，"
            f"请降采样或使用默认服务端逐条路径"
        )
    return est
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add KNN_evaluation/metrics.py KNN_evaluation/tests/test_metrics.py
git commit -m "feat(metrics): 新增 estimate_batch_memory / guard_batch_memory 内存守卫"
```

**验证方式:** `@10M 点 ≈10.4GB`、超阈值抛 `MemoryError`、低于阈值返回预估。

---

### Task 4.3: cli.py evaluate 新增 --batch 与 --max-eval-ram

**Files:**
- Modify: `KNN_evaluation/cli.py`（`p_eval` 现值 :530、`cmd_evaluate` :295）

**Interfaces:**
- Consumes: `guard_batch_memory` / `estimate_batch_memory`（Task 4.2）、`compute_knn_accuracy` / `compute_purity_recall_curve`（Task 4.1，`use_batch` 参数）
- Produces:
  - `p_eval` 新增 `--batch`（store_true，help："显式开启 numpy 批量路径（默认服务端逐条 exact，内存安全）"）、`--max-eval-ram`（type=float，default=6.0，help："批量路径内存守卫阈值 GB (默认: 6)"）
  - `cmd_evaluate`：`use_batch = args.batch`；`if use_batch: guard_batch_memory(manager, num_queries, max(k_values + [args.k_f1]), args.max_eval_ram)`，抛 `MemoryError` 时打印并返回 1；`compute_knn_accuracy(..., use_batch=use_batch)`、`compute_purity_recall_curve(..., use_batch=use_batch)`；无论是否批量，打印预估内存提示行
- Consumed by: Task 4.5（单测）

**Dependencies:** Task 4.1、Task 4.2。

- [x] **Step 1: 实现 CLI 参数与守卫**

按上"Produces"修改 `cli.py`。在 `cmd_evaluate` 中采样后、F1 前插入：

```python
num_queries = sum(1 for q in queries if "point_id" in q)
print(f"   已采样 {num_queries} 个查询像素")

use_batch = args.batch
if use_batch:
    max_k = max(k_values + [args.k_f1])
    try:
        est = guard_batch_memory(manager, num_queries, max_k, args.max_eval_ram)
    except MemoryError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    print(f"   批量路径预估峰值: {est['total_gb']:.1f}GB "
          f"(阈值 {args.max_eval_ram}GB)")
else:
    print("   默认路径: 服务端逐条 exact（内存安全）")
```

F1/F2 调用处分别加 `use_batch=use_batch`（含 `--ann` 分支：ANN 批量也受同一 `use_batch` 控制）。

- [x] **Step 2: 冒烟**

Run: `uv run python -m KNN_evaluation.cli evaluate --help`
Expected: 帮助含 `--batch`、`--max-eval-ram`。

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/cli.py
git commit -m "feat(cli): evaluate 新增 --batch 与 --max-eval-ram + 内存守卫"
```

**验证方式:** `--help` 冒烟 + Task 4.5 单测（mock 模式）。

---

### Task 4.4: WebUI 评估面板 — 批量 opt-in + 预估 label

**Files:**
- Modify: `KNN_evaluation/webui.py`（`do_evaluate` :626 及评估面板 UI 区）

**Interfaces:**
- Consumes: `guard_batch_memory` / `estimate_batch_memory`（Task 4.2）、`compute_knn_accuracy` / `compute_purity_recall_curve`（Task 4.1）
- Produces:
  - 评估面板新增：`batch_checkbox = ui.checkbox("批量路径 (numpy matmul)", value=False)`；`eval_ram_label = ui.label("")` 显示"当前批量预估：X.XGB（阈值 6.0GB）"
  - `do_evaluate`：`use_batch = batch_checkbox.value`；采样后 `num_q` 已知，`if use_batch: est = guard_batch_memory(manager, num_q, max(k_values+[k_f1]), 6.0)`（抛 `MemoryError` → notify 并 return）；F1/F2 调用传 `use_batch=use_batch`；`_show_eval_results` 前置一行结果含批量预估
- Consumed by: Task 4.5（单测可只测 guard 交互逻辑）

**Dependencies:** Task 4.1、Task 4.2、Task 4.3（参数命名对齐）。

- [x] **Step 1: 找评估面板 UI 区**

Run: `rg -n "评估|spc_input|kf1_input|eval_progress|batch" KNN_evaluation/webui.py`
Expected: 定位评估 expansion（`spc_input` / `kf1_input` / `kvalues_input` / `seed_input` 定义处）。

- [x] **Step 2: 添加 checkbox 与预估 label**

在评估参数控件行旁添加：
```python
batch_checkbox = ui.checkbox("批量路径 (numpy matmul, 需大内存)", value=False)
eval_ram_label = ui.label("").classes("text-xs text-grey")
```
在 `do_evaluate` 内 `num_q` 计算后：
```python
use_batch = batch_checkbox.value
max_k = max(k_values + [k_f1])
if use_batch:
    try:
        est = guard_batch_memory(manager, num_q, max_k, 6.0)
    except MemoryError as e:
        ui.notify(str(e), type="negative")
        eval_progress_label.set_visibility(False)
        eval_progress_bar.set_visibility(False)
        return
    eval_ram_label.set_text(f"批量预估峰值: {est['total_gb']:.1f}GB（阈值 6.0GB）")
else:
    eval_ram_label.set_text("默认路径: 服务端逐条 exact（内存安全）")
```
F1/F2 的 `await asyncio.to_thread(compute_knn_accuracy, manager, queries, k_f1, True, make_cb(...))` 改为显式传 `use_batch=use_batch`（把 `True` 换成 `use_batch`，保持 exact=True 位置）：
```python
f1 = await asyncio.to_thread(
    compute_knn_accuracy, manager, queries, k_f1, True, use_batch, make_cb("F1 KNN Accuracy"),
)
```
F2 同理：`compute_purity_recall_curve, manager, queries, k_values, True, use_batch, make_cb(...)`。

- [x] **Step 3: 冒烟**

Run: `uv run python KNN_evaluation/webui.py --dir data_demo --port 8003`
Expected: 评估面板显示 checkbox 与预估 label；勾选批量且 N 大时提示超阈值并被拒绝；默认路径正常计算。

- [x] **Step 4: 提交**

```bash
git add KNN_evaluation/webui.py
git commit -m "feat(webui): 评估面板批量 opt-in checkbox + 内存预估 label"
```

**验证方式:** 手工冒烟 + Task 4.5 单测覆盖守卫拒绝路径。

---

### Task 4.5: 内存守卫单测（超阈值拒绝 / 默认不触发 / 批量与逐条一致）

**Files:**
- Modify: `KNN_evaluation/tests/test_metrics.py`

**Interfaces:**
- Consumes: Task 4.1、4.2 的产物
- Produces: 无新接口

**Dependencies:** Task 4.1、Task 4.2。

- [x] **Step 1: 写测试**

在 `KNN_evaluation/tests/test_metrics.py` 追加：

```python
class TestBatchVsSequentialConsistency:
    def test_batch_and_sequential_agree(self, qdrant_manager):
        """小数据下批量与逐条结果一致（真实 Qdrant，conftest fixture）."""
        from KNN_evaluation.metrics import (
            compute_knn_accuracy, compute_purity_recall_curve, sample_queries_by_label,
        )
        queries = sample_queries_by_label(qdrant_manager, samples_per_class=5, seed=7)
        seq = compute_knn_accuracy(qdrant_manager, queries, k=5, exact=True, use_batch=False)
        bat = compute_knn_accuracy(qdrant_manager, queries, k=5, exact=True, use_batch=True)
        assert seq["overall_accuracy"] == bat["overall_accuracy"]
        # 批量与逐条 Purity/Recall 一致
        seq2 = compute_purity_recall_curve(qdrant_manager, queries, [2, 5], exact=True, use_batch=False)
        bat2 = compute_purity_recall_curve(qdrant_manager, queries, [2, 5], exact=True, use_batch=True)
        assert seq2["global_purity"] == bat2["global_purity"]

    def test_guard_above_threshold_rejects_with_estimate(self):
        from KNN_evaluation.metrics import guard_batch_memory
        m = type("M", (), {"collection_info": lambda self: {"total_points": 10_000_000}})()
        try:
            guard_batch_memory(m, 100, 1000, max_ram_gb=6.0)
            raise AssertionError("应拒绝")
        except MemoryError as e:
            assert "超阈值" in str(e) and "GB" in str(e)

    def test_default_path_no_guard_trigger(self, qdrant_manager):
        """默认 use_batch=False 不进入批量路径，无需守卫."""
        from KNN_evaluation.metrics import compute_knn_accuracy
        queries = sample_queries_by_label(qdrant_manager, samples_per_class=3, seed=1)
        result = compute_knn_accuracy(qdrant_manager, queries, k=3, exact=True, use_batch=False)
        assert result["num_queries"] == sum(1 for q in queries if "point_id" in q)
```

> **提示**：`test_batch_and_sequential_agree` 依赖真实 Qdrant（conftest fixture）；无 Docker 时用 `pytest.mark.skipif(not _qdrant_is_running())` 跳过。CLI/WebUI 的守卫拒绝路径由 `test_guard_above_threshold_rejects_with_estimate`（纯 mock）覆盖。

- [x] **Step 2: 运行测试**

Run: `uv run pytest KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS（有 Qdrant 时批量/逐条一致；无 Qdrant 时仅纯 mock 用例通过，fixture 用例被 skip）

- [x] **Step 3: 提交**

```bash
git add KNN_evaluation/tests/test_metrics.py
git commit -m "test(metrics): 内存守卫拒绝 + 批量/逐条一致性 + 默认路径不触发"
```

**验证方式:** 上述测试全绿（或无 Qdrant 时对应 skip）。这完成 tasks.md 的第 4 组（4.1-4.5）。

---

### Task 5.1: 集成回归 — 既有测试全套通过 + data_demo 小规模流程

**Files:**
- 无源码修改（纯验证 + 必要时修回归）

**Interfaces:**
- Consumes: 前四组全部产物

**Dependencies:** Task 1.1-4.5 全部完成。

- [x] **Step 1: 全量测试**

Run: `uv run pytest KNN_evaluation/tests/ -v`
Expected: 既有 `test_importer` / `test_metrics` / `test_searcher` / `test_qdrant_client` / `test_progress_callback` / `test_import_retry` 等全部通过；新增 `test_manifest` / `test_webui` / `test_migrate` / `test_start_qdrant` 全绿。若发现回归，用 systematic-debugging 定位并修复后重新运行。

- [x] **Step 2: data_demo 小规模集成冒烟（真实 Qdrant）**

Run:
```bash
uv run python -m KNN_evaluation.cli migrate --dir data_demo --storage disk
uv run python -m KNN_evaluation.cli search --random --k 10
uv run python -m KNN_evaluation.cli evaluate --samples-per-class 10 --k-values 2,5
```
Expected: migrate 重建 disk collection 成功（`old → new` 点数）；search 返回 Top-K；evaluate 默认服务端逐条跑通（无 `--batch` 不触发守卫）；`qdrant_import_manifest.json` 生成且含 `data_demo` 的 image_id。

- [x] **Step 3: 提交（如有修复）**

```bash
git add -A
git commit -m "fix: 集成回归修复"
```

**验证方式:** 全量 pytest 绿 + data_demo 三条 CLI 命令成功。

---

### Task 5.2: 全量 data/ 重建 + 重导 + docker stats 内存验证

**Files:**
- 无源码修改（运行验证）

**Interfaces:**
- Consumes: Task 3.3/3.5（migrate + 幂等启动）、Task 3.1（disk 默认）

**Dependencies:** Task 5.1。

- [x] **Step 1: 全量迁移重建**

Run: `uv run python -m KNN_evaluation.cli migrate --dir data --storage disk --no-resume`
Expected: 删除重建 disk collection → 重导 624 张 / 1022 万点成功（断点续传可中断重试）；结束后 `reconcile_manifest()` 生成全量 manifest（624 个 image_id）；`qdrant_import_manifest.json` 的 `images` 键数 == 624。

- [x] **Step 2: 验证常驻内存**

Run: `docker stats --no-stream qdrant`
Expected: 磁盘化后 Qdrant 常驻内存 **≈2-3GB**（HNSW 留 RAM，向量/payload 落盘）。记录数据到 task 验证证据。

- [x] **Step 3: 记录验证证据（写入测试或手动备注）**

将 `docker stats` 输出与 `manager.collection_info()` 的 `total_points`（约 10,220,000）记录为验收证据；如与预期不符，用 systematic-debugging 定位（可能是 volume 未生效 / 旧内存容器复用）。

**验证方式:** `total_points` ≈10.2M；`docker stats` 内存 ≈2-3GB；manifest 含 624 个 image_id。

---

### Task 5.3: Web 启动秒级 + 多 tab + manifest 删除后回退重建

**Files:**
- 无源码修改（运行验证）

**Interfaces:**
- Consumes: Task 2.1-2.4（init_page 快速/慢速路径）、Task 1.2（manifest 回退 facet）

**Dependencies:** Task 5.1、5.2。

- [x] **Step 1: Web 启动秒级验证**

Run: `uv run python KNN_evaluation/webui.py --dir data --port 8003`
Expected: 打开页面**秒级**出现 "✅ Qdrant 连接正常" 且可交互（状态标签先于慢速路径）；影像列表毫秒级填充；随后后台 facet 对账完成。

- [x] **Step 2: 多 tab 验证**

打开 2-3 个 tab，观察浏览器 Network：**无重复全库 scroll**（`get_imported_image_ids` 走 manifest）；`_render_preview` 翻页不触发 `check_image_count` HTTP count。

- [x] **Step 3: manifest 删除后回退重建**

Run:
```bash
rm qdrant_import_manifest.json
# 刷新 WebUI
```
Expected: Web 启动时 `get_imported_image_ids` 回退 `_facet_image_ids`（facet 一次请求）重建；`_background_init` 中 `reconcile_manifest()` 重新生成 manifest 文件；页面仍秒级可交互。

- [x] **Step 4: 提交（如有修复）并总结**

```bash
git add -A
git commit -m "fix: 5.3 Web 集成验证修复"
```

**验证方式:** 秒级 ✅、多 tab 无 scroll、manifest 删除后 facet 回退重建且文件再生成。这完成 tasks.md 的第 5 组（5.1-5.3）与全部 23 个任务。

---

## 任务依赖图

```
1.1 manifest.py ──► 1.2 qdrant_client(manifest+facet) ──► 2.1 init_page 快速/慢速路径 ──► 2.4 缓存+失效
      │                    │                                  ├─► 2.2 to_thread（并行）
      │                    │                                  └─► 2.3 _render_preview 读 manifest ──► 2.5 集成断言
      │                    └─► 3.3 migrate ──► 3.4 幂等单测
      │                                 └──► 3.5 _start_qdrant 三态（独立）──► 3.6 README
      └─► 3.1 create_collection(storage) ──► 3.2 磁盘化单测
                                               3.5 与 3.1-3.4 可并行
4.1 use_batch 默认 False ──► 4.3 CLI --batch/--max-eval-ram
4.2 内存守卫（可与 4.1 并行）──► 4.4 WebUI 面板 ──► 4.5 单测
1-4 全部 ──► 5.1 集成回归 ──► 5.2 全量重建+docker stats ──► 5.3 Web 秒级+多 tab+回退
```

**执行顺序建议:** 1.1 → 1.2 → 1.3 →（2.1 → 2.2 ∥ 2.3 → 2.4 → 2.5）→（3.1 → 3.2，3.5 ∥ 3.5 → 3.6，3.3 → 3.4）→（4.1 ∥ 4.2 → 4.3 → 4.4 → 4.5）→ 5.1 → 5.2 → 5.3。
