---
change: collection-selector
design-doc: docs/superpowers/specs/2026-08-07-collection-selector-design.md
base-ref: 684934436b931d3c4fdeeaee015898e725b1890e
archived-with: 2026-08-07-collection-selector
---

# Collection Selector 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 CLI 与 WebUI 能在多个 Qdrant Collection（`google_aef_embedding` / `xian_aef_embedding` + 自定义）之间选择与切换，并将 manifest / 采样地图按 collection 隔离存储。

**Architecture:** 三层解耦：(1) `config.py` 定义 `DEFAULT_COLLECTION` / `PRESET_COLLECTIONS`，`COLLECTION_NAME` 作为别名保留；(2) `manifest.py` / `sampling_map.py` 新增 `safe_collection_token` + `manifest_path(collection)` / `sampling_map_path(collection)` 按 collection 派生文件名，`ensure_sampling_map` 缺省按 `manager.collection_name` 派生路径，使 `metrics.py` 无需改动；CLI 各子命令注入 `--collection`，WebUI 以模块级 `_current_collection`（与 `_CLI_QDRANT_URL` 同模式）承载当前选择，用 NiceGUI tabs 分页选择并配合 localStorage 记忆。

**Tech Stack:** Python 3.12+ / argparse / NiceGUI 3.15 / Qdrant client / pytest / uv

## Global Constraints

- 默认 collection 为 `google_aef_embedding`（D1）。
- `safe_collection_token` 保留字符集为 `[A-Za-z0-9_.-]`，其余替换为 `_`（D4，防路径穿越）。
- manifest 文件名 `qdrant_import_manifest_<token>.json`；采样地图文件名 `qdrant_sampling_map_<token>.json`（D4）。
- 不迁移、不删除旧缓存文件 `qdrant_import_manifest.json` / `qdrant_sampling_map.json`（Non-Goals）；首次访问新命名路径自动重建。
- 不修改 `LinearProbe_evaluation/`（Non-Goals）。
- `corpus_cache.py` 已按 sha256 隔离，**不改动**。
- `metrics.py` 的 `ensure_sampling_map(manager)` 调用点**不改动**（靠 path=None 缺省派生跟随 collection）。
- 全量测试命令：`uv run pytest KNN_evaluation/tests/ -v`（Windows 平台）。
- 所有注释与用户可见输出使用中文（与现有代码一致）。
- 每个任务结束必须 commit；base-ref 为 `684934436b931d3c4fdeeaee015898e725b1890e`。
- NiceGUI 3.15 的 `Tabs`/`TabPanels` 支持 `clear()` 与 `set_value()`，但不支持动态 `.add()`；动态增删分页用「clear + 重建 + set_value」实现（构建时如框架行为有出入，可调整 UI 表达，spec 只约束可观察行为）。

---

### Task 1: config.py 常量组织（D1）

**Files:**
- Modify: `KNN_evaluation/config.py`
- Create: `KNN_evaluation/tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `config.DEFAULT_COLLECTION = "google_aef_embedding"`、`config.PRESET_COLLECTIONS = ["google_aef_embedding", "xian_aef_embedding"]`、`config.COLLECTION_NAME == config.DEFAULT_COLLECTION`（别名）。后续 Task 5（CLI）、Task 6/7（WebUI）依赖这些常量。

- [x] **Step 1: 写失败测试**

创建 `KNN_evaluation/tests/test_config.py`：

```python
"""Tests for KNN_evaluation.config collection constants."""
from KNN_evaluation import config


class TestCollectionConstants:
    def test_default_collection(self):
        assert config.DEFAULT_COLLECTION == "google_aef_embedding"

    def test_preset_collections(self):
        assert config.PRESET_COLLECTIONS == ["google_aef_embedding", "xian_aef_embedding"]

    def test_collection_name_alias(self):
        assert config.COLLECTION_NAME == config.DEFAULT_COLLECTION
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_config.py -v`
Expected: FAIL — `AttributeError: module 'KNN_evaluation.config' has no attribute 'DEFAULT_COLLECTION'`

- [x] **Step 3: 实现 config.py 常量**

修改 `KNN_evaluation/config.py` 为：

```python
"""Qdrant KNN 评估系统配置."""

QDRANT_URL = "http://localhost:6333"

# 预置 Qdrant Collection 与默认值（D1）：
# DEFAULT_COLLECTION 供 CLI/WebUI 未指定时使用；PRESET_COLLECTIONS 为预置选择项；
# COLLECTION_NAME 作为 DEFAULT_COLLECTION 的别名保留，避免大范围替换引用、
# 保持 QdrantManager 默认参数兼容。
DEFAULT_COLLECTION = "google_aef_embedding"
PRESET_COLLECTIONS = ["google_aef_embedding", "xian_aef_embedding"]
COLLECTION_NAME = DEFAULT_COLLECTION

BATCH_SIZE = 10000
VECTOR_SIZE = 64
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
EF_SEARCH_DEFAULT = 64
QDRANT_TIMEOUT = 5
UTM_RESOLUTION_M = 10  # UTM 坐标推算分辨率（米/像素）
```

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_config.py -v`
Expected: PASS (3 passed)

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/config.py KNN_evaluation/tests/test_config.py
git commit -m "feat(config): 新增 DEFAULT_COLLECTION/PRESET_COLLECTIONS，COLLECTION_NAME 作别名保留"
```

---

### Task 2: manifest.py 按 collection 隔离 + safe_collection_token（D4）

**Files:**
- Modify: `KNN_evaluation/manifest.py`
- Modify: `KNN_evaluation/tests/test_manifest.py`

**Interfaces:**
- Consumes: 无（自包含）。
- Produces:
  - `manifest.safe_collection_token(collection: str) -> str`（保留 `[A-Za-z0-9_.-]`，其余替换 `_`）
  - `manifest.manifest_path(collection: str) -> Path` → `qdrant_import_manifest_<token>.json`
  - `manifest.load_manifest(path: Path | None = None) -> dict`（缺省回退 `MANIFEST_PATH` 遗留默认）
  - `manifest.save_manifest(data: dict, path: Path | None = None) -> None`
  - `manifest.update_manifest(image_id, imported_pixels, collection, path: Path | None = None) -> dict`（缺省按 `manifest_path(collection)` 派生）
  - 保留 `manifest.MANIFEST_PATH` 常量作为无 collection 上下文时的遗留默认。Task 4（qdrant_client）消费 `manifest_path`；Task 6（webui）消费 `manifest_path`；Task 7（webui 清理缓存）消费 `manifest_path`。`importer.py` 调用 `update_manifest` 时已传 collection，自动跟随派生路径，无需改动。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_manifest.py` 末尾追加：

```python
class TestManifestPathIsolation:
    """Task: manifest 按 collection 隔离的路径派生与安全清洗（D4）."""

    def test_safe_token_keeps_alnum_dot_underscore_dash(self):
        from KNN_evaluation.manifest import safe_collection_token
        assert safe_collection_token("google_aef_embedding") == "google_aef_embedding"
        assert safe_collection_token("my_col-2.v1") == "my_col-2.v1"

    def test_safe_token_replaces_special_chars(self):
        from KNN_evaluation.manifest import safe_collection_token
        assert safe_collection_token("a/b\\c:d*e?f\"g<h>i|j") == "a_b_c_d_e_f_g_h_i_j"

    def test_path_traversal_has_no_separator(self):
        from KNN_evaluation.manifest import safe_collection_token, manifest_path
        token = safe_collection_token("..\\..\\etc\\passwd")
        assert "/" not in token and "\\" not in token
        p = manifest_path("x/../../y")
        assert p.name == "qdrant_import_manifest_x_.._.._y.json"
        assert p.parent == Path(".")  # 无目录穿越

    def test_manifest_path_naming(self):
        from KNN_evaluation.manifest import manifest_path
        p = manifest_path("google_aef_embedding")
        assert p.name == "qdrant_import_manifest_google_aef_embedding.json"

    def test_update_manifest_default_path_derived_from_collection(self, tmp_path, monkeypatch):
        """update_manifest 未传 path 时按 collection 派生文件名，互不覆盖."""
        monkeypatch.chdir(tmp_path)
        update_manifest("A", 16384, "google_aef_embedding")
        p = tmp_path / "qdrant_import_manifest_google_aef_embedding.json"
        assert p.exists()
        assert load_manifest(p)["images"] == {"A": 16384}
        # 另一 collection 的文件应独立
        update_manifest("B", 1024, "xian_aef_embedding")
        assert load_manifest(tmp_path / "qdrant_import_manifest_xian_aef_embedding.json")["images"] == {"B": 1024}
        assert load_manifest(p)["images"] == {"A": 16384}
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_manifest.py -v`
Expected: FAIL — `ImportError: cannot import name 'safe_collection_token'`

- [x] **Step 3: 实现 manifest.py**

修改 `KNN_evaluation/manifest.py`：

顶部 import 区改为：

```python
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

# 遗留默认路径（无 collection 上下文时的回退；生产调用方一律走 manifest_path() 派生路径）
MANIFEST_PATH = Path("qdrant_import_manifest.json")


def safe_collection_token(collection: str) -> str:
    """将 collection 名称清洗为安全文件 token：保留 [A-Za-z0-9_.-]，其余替换为 '_'.

    防止 collection 名称含路径分隔符/特殊字符导致路径穿越或非法文件名。
    """
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(collection))


def manifest_path(collection: str) -> Path:
    """按 collection 派生的 manifest 路径：qdrant_import_manifest_<token>.json."""
    return Path(f"qdrant_import_manifest_{safe_collection_token(collection)}.json")
```

`load_manifest` 签名与开头改为：

```python
def load_manifest(path: Path | None = None) -> dict:
    """读取 manifest；文件缺失/损坏返回空结构（不报错）.

    path 缺省时回退 MANIFEST_PATH（遗留默认）；生产调用方应传入
    `manifest_path(collection)` 按 collection 隔离。
    """
    if path is None:
        path = MANIFEST_PATH
    try:
```

`save_manifest` 签名与开头改为：

```python
def save_manifest(data: dict, path: Path | None = None) -> None:
    """原子写：tmp + os.replace."""
    if path is None:
        path = MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
```

`update_manifest` 签名与开头改为：

```python
def update_manifest(
    image_id: str,
    imported_pixels: int,
    collection: str,
    path: Path | None = None,
) -> dict:
    """增量更新单张影像的已导入像素数（原子写），返回更新后的 manifest.

    path 缺省时按 collection 派生（manifest_path(collection)），
    保证不同 collection 的 manifest 互不覆盖。
    """
    if path is None:
        path = manifest_path(collection)
    data = load_manifest(path)
```

其余函数体不变。

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_manifest.py -v`
Expected: PASS（既有测试 + 新增 5 个全部通过）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/manifest.py KNN_evaluation/tests/test_manifest.py
git commit -m "feat(manifest): manifest 按 collection 隔离路径 + safe_collection_token 安全清洗"
```

---

### Task 3: sampling_map.py 按 collection 隔离 + conftest 更新（D4）

**Files:**
- Modify: `KNN_evaluation/sampling_map.py`
- Modify: `KNN_evaluation/tests/test_sampling_map.py`
- Modify: `KNN_evaluation/tests/conftest.py`

**Interfaces:**
- Consumes: `manifest.safe_collection_token`（Task 2）。
- Produces:
  - `sampling_map.sampling_map_path(collection: str) -> str` → `qdrant_sampling_map_<token>.json`
  - `sampling_map.ensure_sampling_map(manager, path: str | os.PathLike | None = None) -> dict`（path 缺省按 `manager.collection_name` 派生）
  - `load_sampling_map(path=None)` / `save_sampling_map(data, path=None)`（缺省回退 `SAMPLING_MAP_PATH` 遗留默认）
  - 保留 `sampling_map.SAMPLING_MAP_PATH` 常量作为遗留默认。
  - `metrics.py:105` 的 `ensure_sampling_map(manager)` 自动跟随，无需改动；Task 7（webui 刷新/清理）消费 `ensure_sampling_map` / `sampling_map_path`。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_sampling_map.py` 末尾追加：

```python
class TestSamplingMapPathIsolation:
    """Task: 采样地图按 collection 隔离路径 + ensure_sampling_map 缺省派生（D4）."""

    def _manager(self, total_points: int, collection: str):
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = collection
        manager.collection_info.return_value = {"total_points": total_points}
        manager.client.scroll.return_value = (
            [_mock_scroll_record(0, i) for i in range(total_points)], None,
        )
        return manager

    def test_sampling_map_path_naming(self):
        from KNN_evaluation.sampling_map import sampling_map_path
        assert sampling_map_path("google_aef_embedding") == "qdrant_sampling_map_google_aef_embedding.json"

    def test_sampling_map_path_sanitizes(self):
        from KNN_evaluation.sampling_map import sampling_map_path
        assert sampling_map_path("a/b\\c") == "qdrant_sampling_map_a_b_c.json"

    def test_ensure_sampling_map_derives_path_from_collection(self, tmp_path, monkeypatch):
        """path 缺省时按 manager.collection_name 派生路径，两个 collection 互不覆盖."""
        monkeypatch.chdir(tmp_path)
        m1 = self._manager(total_points=3, collection="google_aef_embedding")
        m2 = self._manager(total_points=2, collection="xian_aef_embedding")
        ensure_sampling_map(m1)
        ensure_sampling_map(m2)
        p1 = tmp_path / "qdrant_sampling_map_google_aef_embedding.json"
        p2 = tmp_path / "qdrant_sampling_map_xian_aef_embedding.json"
        assert p1.exists() and p2.exists()
        assert load_sampling_map(p1)["collection"] == "google_aef_embedding"
        assert load_sampling_map(p2)["collection"] == "xian_aef_embedding"
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_sampling_map.py -v`
Expected: FAIL — `ImportError: cannot import name 'sampling_map_path'`

- [x] **Step 3: 实现 sampling_map.py**

修改 `KNN_evaluation/sampling_map.py`：

import 区改为：

```python
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.manifest import safe_collection_token

# 遗留默认路径（无 collection 上下文回退；生产调用方走 sampling_map_path() 派生）
SAMPLING_MAP_PATH = "qdrant_sampling_map.json"


def sampling_map_path(collection: str) -> str:
    """按 collection 派生的采样地图路径：qdrant_sampling_map_<token>.json."""
    return f"qdrant_sampling_map_{safe_collection_token(collection)}.json"
```

`load_sampling_map` 签名与开头改为：

```python
def load_sampling_map(path: str | os.PathLike | None = None) -> dict:
    """读取采样地图；文件缺失/损坏返回空结构（不报错）.

    path 缺省时回退 SAMPLING_MAP_PATH（遗留默认）。
    """
    if path is None:
        path = SAMPLING_MAP_PATH
    try:
```

`save_sampling_map` 签名与开头改为：

```python
def save_sampling_map(data: dict, path: str | os.PathLike | None = None) -> None:
    """原子写：tmp + os.replace."""
    if path is None:
        path = SAMPLING_MAP_PATH
    path = os.fspath(path)
```

`ensure_sampling_map` 签名与 docstring 段改为：

```python
def ensure_sampling_map(
    manager: QdrantManager,
    path: str | os.PathLike | None = None,
) -> dict:
    """自动对账：地图指纹与当前 collection 不一致时重建并保存.

    比较 manifest 的 collection 名称 + total_points 与当前 collection 信息；
    不一致 / 文件缺失 / 损坏 → 自动调用 `build_sampling_map` 重建并保存；
    一致 → 直接返回缓存。

    Args:
        manager: QdrantManager 实例。
        path: 地图文件路径。缺省时按 `manager.collection_name` 派生
            （`sampling_map_path`），不同 collection 的地图写入独立文件互不覆盖。

    Returns:
        完整地图结构。
    """
    if path is None:
        path = sampling_map_path(manager.collection_name)
    cached = load_sampling_map(path)
```

其余函数体不变。

- [x] **Step 4: 更新 conftest.py 测试 collection 缓存清理路径**

修改 `KNN_evaluation/tests/conftest.py`：

第 10 行 import 改为：

```python
from KNN_evaluation.sampling_map import sampling_map_path, load_sampling_map
```

将 `qdrant_manager` fixture 中清理采样地图的代码段（原第 95-102 行）替换为：

```python
    # 清理测试 collection 对应的采样地图缓存：测试 collection 每次会话重建，
    # point_id（uuid4）全新生成，但 total_points 指纹可能相同，导致
    # ensure_sampling_map 误判缓存有效而返回过期 point_id → retrieve 取空。
    # 仅当缓存确实指向本测试 collection 时才清理，避免误删用户正式数据的地图。
    map_path = sampling_map_path(manager.collection_name)
    cached = load_sampling_map(map_path)
    if cached.get("collection") == manager.collection_name:
        try:
            os.remove(map_path)
        except OSError:
            pass
```

- [x] **Step 5: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_sampling_map.py KNN_evaluation/tests/test_metrics.py -v`
Expected: PASS（test_sampling_map 新增 3 个通过；test_metrics 全部通过——`metrics.py` 无需改动，`ensure_sampling_map(manager)` 缺省派生路径）

- [x] **Step 6: Commit**

```bash
git add KNN_evaluation/sampling_map.py KNN_evaluation/tests/test_sampling_map.py KNN_evaluation/tests/conftest.py
git commit -m "feat(sampling_map): 采样地图按 collection 隔离路径，ensure_sampling_map 缺省派生"
```

---

### Task 4: qdrant_client.py manifest 按 collection 派生（D4 调用点）

**Files:**
- Modify: `KNN_evaluation/qdrant_client.py`
- Modify: `KNN_evaluation/tests/test_qdrant_client.py`

**Interfaces:**
- Consumes: `manifest.manifest_path`（Task 2）。
- Produces: `QdrantManager` 的 manifest 读写（`get_imported_image_ids` / `reconcile_manifest`）均作用于 `manifest_path(self.collection_name)`。CLI `cmd_migrate`（Task 5）与 WebUI `_reconcile_background` 经 `manager.reconcile_manifest()` 自动跟随。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_qdrant_client.py` 末尾追加：

```python
class TestManifestPathPerCollection:
    """Task: QdrantManager 的 manifest 读写按当前 collection 派生路径（D4）."""

    def test_get_imported_image_ids_uses_collection_path(self, monkeypatch):
        calls: list = []
        state = {"collection": "c", "images": {"A": 16384}, "updated_at": "x"}
        monkeypatch.setattr(qc, "manifest_path", lambda collection: f"path/{collection}.json")
        monkeypatch.setattr(qc, "load_manifest",
                            lambda path=None: (calls.append(path), dict(state))[1])
        m = _manager()
        m.get_imported_image_ids()
        assert calls and calls[0] == "path/c.json"

    def test_reconcile_manifest_saves_to_collection_path(self, monkeypatch):
        state = {"collection": "c", "images": {"A": 16384, "GHOST": 16384}, "updated_at": "x"}
        load_calls: list = []
        save_calls: list = []
        monkeypatch.setattr(qc, "manifest_path", lambda collection: f"path/{collection}.json")
        monkeypatch.setattr(qc, "load_manifest",
                            lambda path=None: (load_calls.append(path), dict(state))[1])

        def _save(data, path=None):
            save_calls.append(path)
            state.clear()
            state.update(data)

        monkeypatch.setattr(qc, "save_manifest", _save)
        m = _manager()
        m._facet_image_ids = lambda limit=1000: {"A"}
        out = m.reconcile_manifest()
        assert save_calls and save_calls[0] == "path/c.json"
        assert set(out["images"]) == {"A"}
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_qdrant_client.py -v`
Expected: FAIL — 第 1 个测试 `assert calls and calls[0] == "path/c.json"` 失败（当前 `manifest_path` 不存在，monkeypatch 报 `AttributeError` 或路径为空）

- [x] **Step 3: 实现 qdrant_client.py**

修改 `KNN_evaluation/qdrant_client.py`：

第 7 行 import 改为：

```python
from KNN_evaluation.manifest import manifest_path, load_manifest, save_manifest
```

在 `QdrantManager` 类内新增私有 helper（放在 `get_imported_image_ids` 之前）：

```python
    def _manifest_file(self) -> Path:
        """当前 collection 的 manifest 文件路径（按 collection 隔离，D4）."""
        return manifest_path(self.collection_name)
```

`get_imported_image_ids` 方法开头改为：

```python
    def get_imported_image_ids(self) -> set[str]:
        """从本地 manifest 读取已导入 image_id（毫秒级，Qdrant 离线可用）.

        无 manifest（空清单）时回退 facet 重建。
        """
        data = load_manifest(self._manifest_file())
        ids = set((data.get("images") or {}).keys())
        if not ids:
            ids = self._facet_image_ids()
        return ids
```

`reconcile_manifest` 方法中的三处 manifest 读写改为按 collection 派生路径：

```python
        current = load_manifest(self._manifest_file())
```
```python
        save_manifest({
            "collection": self.collection_name,
            "images": images,
            "updated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }, self._manifest_file())
        return load_manifest(self._manifest_file())
```

需在文件顶部 `import` 区补充 `from pathlib import Path`（若文件已无该 import）。

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_qdrant_client.py KNN_evaluation/tests/test_manifest.py -v`
Expected: PASS（新增 2 个通过；既有 `_seed_manifest` 的 `load_manifest`/`save_manifest` monkeypatch 接受 path 参数，`test_manifest.py` 的 reconcile mock 测试不受影响）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/qdrant_client.py KNN_evaluation/tests/test_qdrant_client.py
git commit -m "feat(qdrant_client): manifest 读写按当前 collection 派生路径"
```

---

### Task 5: CLI `--collection` 参数注入（D2）

**Files:**
- Modify: `KNN_evaluation/cli.py`
- Modify: `KNN_evaluation/tests/test_cli.py`
- Modify: `KNN_evaluation/tests/test_migrate.py`

**Interfaces:**
- Consumes: `config.DEFAULT_COLLECTION`（Task 1）；`QdrantManager(url, collection_name)`。
- Produces: 五个子命令（`import`/`search`/`stats`/`evaluate`/`migrate`）均有 `args.collection`（default `DEFAULT_COLLECTION`），`cmd_*` 以 `QdrantManager(url=args.qdrant_url, collection_name=args.collection)` 构造。`cmd_migrate` 的 `manager.reconcile_manifest()` 经 Task 4 自动按 collection 派生路径。

- [x] **Step 1: 更新既有测试参数对象（先修编译性失败）**

修改 `KNN_evaluation/tests/test_cli.py`：

`TestCmdMigrateFailFast._args()` 改为：

```python
    def _args(self):
        return SimpleNamespace(
            qdrant_url="http://localhost:6333",
            collection="google_aef_embedding",
            storage="disk", dir="data_demo", no_resume=False,
        )
```

`_eval_args` base dict 增加 `collection`：

```python
def _eval_args(**overrides):
    """构造 evaluate 子命令的 SimpleNamespace 参数（默认 device=cpu 走 CPU 分块路径）."""
    base = dict(
        device="cpu", gpu_batch_q=None, max_gpu_mem=16, max_eval_ram=6.0,
        qdrant_url="http://localhost:6333", collection="test_collection",
        samples_per_class=10, seed=42,
        k_f1=5, k_values="5,10", ann=False, output=None, plot=False,
        plot_dir="./eval_plots",
    )
    base.update(overrides)
    return SimpleNamespace(**base)
```

修改 `KNN_evaluation/tests/test_migrate.py`：

`_Args.__init__` 增加 collection 属性：

```python
class _Args:
    def __init__(self, storage="disk", no_resume=False, dir="data_demo",
                 qdrant_url="http://localhost:1", collection="c"):
        self.storage = storage
        self.no_resume = no_resume
        self.dir = dir
        self.qdrant_url = qdrant_url
        self.collection = collection
```

- [x] **Step 2: 运行既有 CLI 测试确认仍失败（新功能未实现）**

Run: `uv run pytest KNN_evaluation/tests/test_cli.py KNN_evaluation/tests/test_migrate.py -v`
Expected: FAIL — `cmd_*` 尚未构造 `collection_name`（新增断言见 Step 4 前先确认既有用例因 `args.collection` 缺失或 QdrantManager 构造不匹配而失败）

- [x] **Step 3: 写失败测试（新行为）**

在 `KNN_evaluation/tests/test_cli.py` 末尾追加：

```python
class TestCollectionArg:
    """Task 2.1/2.2: --collection 参数解析与传入 manager（D2）."""

    def test_default_collection_for_all_subcommands(self):
        from KNN_evaluation.config import DEFAULT_COLLECTION
        parser = cli._build_parser()
        cases = {
            "import": ["import", "data_demo"],
            "search": ["search", "--random"],
            "stats": ["stats"],
            "evaluate": ["evaluate"],
            "migrate": ["migrate"],
        }
        for cmd, argv in cases.items():
            args = parser.parse_args(argv)
            assert args.collection == DEFAULT_COLLECTION, cmd

    def test_override_collection(self):
        parser = cli._build_parser()
        args = parser.parse_args(["stats", "--collection", "xian_aef_embedding"])
        assert args.collection == "xian_aef_embedding"


class TestCmdCollectionInjection:
    """cmd_* 以 args.collection 构造 QdrantManager."""

    def test_cmd_import_passes_collection_to_manager(self, monkeypatch):
        captured: dict = {}

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=5):
                captured["collection_name"] = collection_name

            def collection_exists(self):
                return True

            def collection_info(self):
                return {"total_points": 0}

        class FakeImporter:
            def __init__(self, manager, batch_size=None):
                pass

            def import_directory(self, data_dir, no_resume=False, reindex=False,
                                 progress_callback=None):
                return {"total_pixels": 0, "total_images": 0, "imported_images": 0,
                        "skipped_images": 0, "label_counts": {},
                        "elapsed_sec": 0.0, "rate_pps": 0.0}

        monkeypatch.setattr(cli, "QdrantManager", FakeManager)
        monkeypatch.setattr(cli, "PixelImporter", FakeImporter)
        args = SimpleNamespace(
            qdrant_url="http://localhost:1", collection="xian_aef_embedding",
            batch_size=10000, directory="data_demo", no_resume=False, reindex=False,
        )
        assert cli.cmd_import(args) == 0
        assert captured["collection_name"] == "xian_aef_embedding"

    def test_cmd_evaluate_passes_collection_to_manager(self, monkeypatch):
        manager = MagicMock()
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True
        manager.collection_info.return_value = {"total_points": 1000}
        manager.collection_name = "test_collection"
        args = _eval_args(collection="xian_aef_embedding")
        with patch("KNN_evaluation.cli.QdrantManager", return_value=manager) as mq, \
             patch("KNN_evaluation.metrics.sample_queries_by_label",
                   return_value=_eval_queries()), \
             patch("KNN_evaluation.metrics.evaluate_knn",
                   return_value={"f1": _eval_f1(), "f2": _eval_f2()}):
            code = cli.cmd_evaluate(args)
        assert code == 0
        assert mq.call_args.kwargs["collection_name"] == "xian_aef_embedding"

    def test_cmd_evaluate_missing_collection_returns_1_with_name(self, capsys, monkeypatch):
        mgr = MagicMock()
        mgr.health_check.return_value = True
        mgr.collection_exists.return_value = False
        mgr.collection_name = "xian_aef_embedding"
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        args = _eval_args(collection="xian_aef_embedding")
        assert cli.cmd_evaluate(args) == 1
        assert "xian_aef_embedding" in capsys.readouterr().err


class TestStatsCollection:
    """Task 2.3: stats 输出展示实际 collection 名称 + 不存在的 collection 报错."""

    def test_stats_missing_collection_returns_1_with_name(self, capsys, monkeypatch):
        mgr = MagicMock()
        mgr.collection_exists.return_value = False
        mgr.collection_name = "xian_aef_embedding"
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        args = SimpleNamespace(
            qdrant_url="http://localhost:1", collection="xian_aef_embedding", json=False,
        )
        code = cli.cmd_stats(args)
        assert code == 1
        assert "xian_aef_embedding" in capsys.readouterr().err

    def test_stats_output_shows_collection_name(self, capsys, monkeypatch):
        mgr = MagicMock()
        mgr.collection_exists.return_value = True
        mgr.collection_name = "xian_aef_embedding"
        mgr.collection_info.return_value = {
            "total_points": 100, "vectors_count": 100,
            "segments_count": 1, "status": "green",
        }
        monkeypatch.setattr(
            cli, "QdrantManager",
            lambda url=None, collection_name=None, timeout=5: mgr,
        )
        args = SimpleNamespace(
            qdrant_url="http://localhost:1", collection="xian_aef_embedding", json=False,
        )
        assert cli.cmd_stats(args) == 0
        assert "xian_aef_embedding" in capsys.readouterr().out
```

- [x] **Step 4: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_cli.py -v`
Expected: FAIL — `args.collection` 存在但 `cmd_*` 未把 collection 传入 `QdrantManager`（`mq.call_args.kwargs` 无 `collection_name`）；`--collection` 解析失败（argparse 拒绝未知参数）。

- [x] **Step 5: 实现 cli.py**

修改 `KNN_evaluation/cli.py`：

第 21 行 import 改为（`COLLECTION_NAME` 在 cli.py 中未被使用，替换为 `DEFAULT_COLLECTION`）：

```python
from KNN_evaluation.config import QDRANT_URL, DEFAULT_COLLECTION, BATCH_SIZE
```

`_build_parser` 中五个子命令各增加 `--collection` 参数：

`p_import` 区块（在 `--reindex` 之后、`--qdrant-url` 之前）插入：

```python
    p_import.add_argument("--collection", default=DEFAULT_COLLECTION,
                          help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
```

`p_search` 区块（在 `--output` 之后、`--qdrant-url` 之前）插入：

```python
    p_search.add_argument("--collection", default=DEFAULT_COLLECTION,
                          help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
```

`p_stats` 区块插入：

```python
    p_stats.add_argument("--collection", default=DEFAULT_COLLECTION,
                         help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
```

`p_eval` 区块（在 `--plot-dir` 之后、`--qdrant-url` 之前）插入：

```python
    p_eval.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
```

`p_migrate` 区块（在 `--no-resume` 之后、`--qdrant-url` 之前）插入：

```python
    p_migrate.add_argument("--collection", default=DEFAULT_COLLECTION,
                           help=f"目标 Qdrant Collection 名称 (默认: {DEFAULT_COLLECTION})")
```

五个 `cmd_*` 的 manager 构造改为（共 5 处，`cmd_import` 第 91 行、`cmd_search` 第 159 行、`cmd_stats` 第 267 行、`cmd_evaluate` 第 313 行、`cmd_migrate` 第 609 行）：

```python
    manager = QdrantManager(url=args.qdrant_url, collection_name=args.collection)
```

- [x] **Step 6: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_cli.py KNN_evaluation/tests/test_migrate.py -v`
Expected: PASS（新增 TestCollectionArg / TestCmdCollectionInjection / TestStatsCollection 全部通过；既有 evaluate / migrate 测试因 `_eval_args`/`_Args` 已补 `collection` 而通过）

- [x] **Step 7: Commit**

```bash
git add KNN_evaluation/cli.py KNN_evaluation/tests/test_cli.py KNN_evaluation/tests/test_migrate.py
git commit -m "feat(cli): 各子命令新增 --collection，manager 按 collection 注入"
```

---

### Task 6: WebUI 模块级 `_current_collection` + 替换硬编码引用（D3 第一部分）

**Files:**
- Modify: `KNN_evaluation/webui.py`
- Modify: `KNN_evaluation/tests/test_webui.py`

**Interfaces:**
- Consumes: `config.DEFAULT_COLLECTION` / `config.PRESET_COLLECTIONS`（Task 1）；`manifest.manifest_path`（Task 2）。
- Produces: 模块级 `webui._current_collection: str`（默认 `DEFAULT_COLLECTION`）、`webui._known_collections: list[str]`、`webui._LOCALSTORAGE_KEY`；`init_page` 以 `QdrantManager(url=_CLI_QDRANT_URL, collection_name=_current_collection)` 构造；`_get_manifest_cached` 读 `manifest_path(_current_collection)`；全部 `COLLECTION_NAME` 硬编码引用清除。Task 7 消费 `_current_collection` / `_known_collections`。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_webui.py` 修改 `_reset_module_state` fixture 并追加新测试类。

`_reset_module_state` 改为：

```python
@pytest.fixture(autouse=True)
def _reset_module_state():
    webui._manifest_cache = None
    webui.state = {}
    webui._current_collection = webui.DEFAULT_COLLECTION
    webui._known_collections = list(webui.PRESET_COLLECTIONS)
    yield
    webui._manifest_cache = None
    webui.state = {}
    webui._current_collection = webui.DEFAULT_COLLECTION
    webui._known_collections = list(webui.PRESET_COLLECTIONS)
```

`TestInitPageFastPath.test_init_page_sets_state_and_calls_refresh_status` 中增加断言（在 `assert webui.state["data_dir"] == Path("data_demo")` 之后）：

```python
        assert webui.state["manager"].collection_name == webui.DEFAULT_COLLECTION
```

追加新测试类：

```python
class TestModuleCurrentCollection:
    """Task 3.1/3.2: 模块级 _current_collection 承载当前选择，替换全部硬编码 COLLECTION_NAME."""

    def test_webui_source_does_not_reference_collection_name(self):
        import inspect
        src = inspect.getsource(webui)
        assert "COLLECTION_NAME" not in src, "硬编码 COLLECTION_NAME 应全部清除"

    def test_init_page_builds_manager_with_current_collection(self, monkeypatch):
        calls: dict = {}

        def fake_create_task(coro):
            coro.close()
            return coro

        class FakeManager:
            def __init__(self, url=None, collection_name=None, timeout=5):
                calls["collection_name"] = collection_name

        monkeypatch.setattr(asyncio, "create_task", fake_create_task)
        monkeypatch.setattr(webui, "QdrantManager", FakeManager)
        monkeypatch.setattr(webui, "_CLI_QDRANT_URL", "http://localhost:1")
        monkeypatch.setattr(webui, "_CLI_DATA_DIR", "data_demo")
        monkeypatch.setattr(webui, "_init_hooks", {"refresh_status": lambda: None})
        monkeypatch.setattr(webui, "_current_collection", "xian_aef_embedding")

        async def run():
            await webui.init_page()

        asyncio.run(run())
        assert calls["collection_name"] == "xian_aef_embedding"

    def test_get_manifest_cached_uses_current_collection_path(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(webui, "manifest_path",
                            lambda collection: f"path/{collection}.json")
        monkeypatch.setattr(webui, "load_manifest",
                            lambda path=None: (calls.append(path), {"images": {"A": 1}})[1])
        monkeypatch.setattr(webui, "_current_collection", "xian_aef_embedding")
        assert webui._get_manifest_cached() == {"images": {"A": 1}}
        assert calls and calls[0] == "path/xian_aef_embedding.json"
```

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: FAIL — `test_webui_source_does_not_reference_collection_name` 断言硬编码仍存在；`test_init_page_builds_manager_with_current_collection` 中 manager 构造缺 `collection_name`；`test_get_manifest_cached_uses_current_collection_path` 路径不是按 collection 派生。

- [x] **Step 3: 实现 webui.py（第一部分）**

修改 `KNN_evaluation/webui.py`：

模块级 import 区（第 37-40 行）改为：

```python
from KNN_evaluation.config import (
    QDRANT_URL, DEFAULT_COLLECTION, PRESET_COLLECTIONS, BATCH_SIZE, VECTOR_SIZE,
    EF_SEARCH_DEFAULT, UTM_RESOLUTION_M,
)
```

manifest import（第 44 行）改为：

```python
from KNN_evaluation.manifest import manifest_path, load_manifest
```

在 `_CLI_QDRANT_URL` / `_CLI_DATA_DIR` 定义之后（第 59 行附近）追加：

```python
# 当前 collection 状态：与 _CLI_QDRANT_URL 同模式存模块级。
# state 字典在 index() 每次请求重建，collection 选择必须存模块级避免切换丢失。
_current_collection: str = DEFAULT_COLLECTION
# 会话内已知 collection 列表（预置 + 自定义），用于分页渲染与 localStorage 校验
_known_collections: list[str] = list(PRESET_COLLECTIONS)
_LOCALSTORAGE_KEY = "comet.knn.current_collection"
```

`_get_manifest_cached` 改为：

```python
def _get_manifest_cached() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = load_manifest(manifest_path(_current_collection))
    return _manifest_cache
```

`init_page` 中 manager 构造（第 165 行）改为：

```python
    manager = QdrantManager(url=_CLI_QDRANT_URL, collection_name=_current_collection)
```

替换 7 处 `COLLECTION_NAME` 硬编码引用：

- 第 414 行：`f"Collection: {COLLECTION_NAME}  |  "` → `f"Collection: {_current_collection}  |  "`
- 第 420 行：`f"Collection '{COLLECTION_NAME}' 存在但获取信息失败"` → `f"Collection '{_current_collection}' 存在但获取信息失败"`
- 第 425 行：`info_label.set_text(f"Collection '{COLLECTION_NAME}' 不存在")` → `info_label.set_text(f"Collection '{_current_collection}' 不存在")`
- 第 651 行：`collection_name=COLLECTION_NAME,` → `collection_name=state["manager"].collection_name,`
- 第 662 行：`collection_name=COLLECTION_NAME,` → `collection_name=state["manager"].collection_name,`
- 第 724 行：`collection_name=COLLECTION_NAME,` → `collection_name=state["manager"].collection_name,`
- 第 930 行：`ui.notify(f"Collection '{COLLECTION_NAME}' 不存在", type="negative")` → `ui.notify(f"Collection '{state['manager'].collection_name}' 不存在", type="negative")`

（第 651/662/724/930 行位于 `state["manager"]` 已判非空的闭包内，安全。）

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: PASS（新增 3 个通过；既有 init_page / manifest cache / 评估面板 / 检索测试全部通过——它们 monkeypatch 的 `webui.load_manifest` 均接受 `path=None`）

- [x] **Step 5: Commit**

```bash
git add KNN_evaluation/webui.py KNN_evaluation/tests/test_webui.py
git commit -m "feat(webui): 模块级 _current_collection 承载当前 collection，清除硬编码 COLLECTION_NAME"
```

---

### Task 7: WebUI 分页选择器、切换、localStorage、刷新/清理缓存（D3 第二部分）

**Files:**
- Modify: `KNN_evaluation/webui.py`
- Modify: `KNN_evaluation/tests/test_webui.py`

**Interfaces:**
- Consumes: Task 6 的 `_current_collection` / `_known_collections` / `_LOCALSTORAGE_KEY`；Task 2 的 `manifest_path`；Task 3 的 `sampling_map.ensure_sampling_map` / `sampling_map.sampling_map_path`；`QdrantManager(url, collection_name)`。
- Produces: 模块级函数 `webui._resolve_stored_collection(stored) -> str`、`webui._persist_collection_choice(collection)`、`webui._apply_collection(new_col)`、`webui._add_custom_collection(name)`、`webui._do_refresh()`、`webui._do_clear_cache()`、`webui._restore_stored_collection()`；`index()` 内分页 tab 条 + 自定义添加 + 刷新/清理按钮 + localStorage 恢复定时器。

- [x] **Step 1: 写失败测试**

在 `KNN_evaluation/tests/test_webui.py` 末尾追加：

```python
class TestCollectionRestore:
    """Task 3.3/3.4: localStorage 记忆/恢复/失效回退 + 切换重建 manager."""

    def test_resolve_stored_valid(self, monkeypatch):
        monkeypatch.setattr(webui, "_known_collections", ["google_aef_embedding", "xian_aef_embedding"])
        assert webui._resolve_stored_collection("xian_aef_embedding") == "xian_aef_embedding"

    def test_resolve_stored_invalid_falls_back(self, monkeypatch):
        monkeypatch.setattr(webui, "_known_collections", ["google_aef_embedding", "xian_aef_embedding"])
        assert webui._resolve_stored_collection("deleted_collection") == webui.DEFAULT_COLLECTION
        assert webui._resolve_stored_collection("") == webui.DEFAULT_COLLECTION
        assert webui._resolve_stored_collection(None) == webui.DEFAULT_COLLECTION

    def test_restore_from_local_storage_switches(self, monkeypatch):
        async def fake_js(code, catch=True):
            return "xian_aef_embedding"

        monkeypatch.setattr(webui.ui, "run_javascript", fake_js)
        monkeypatch.setattr(webui, "_current_collection", "google_aef_embedding")
        monkeypatch.setattr(webui, "_known_collections",
                            ["google_aef_embedding", "xian_aef_embedding"])
        webui.state = {"manager": None}
        calls: list = []

        async def fake_hook(name):
            calls.append(name)

        monkeypatch.setattr(webui, "_call_hook", fake_hook)
        asyncio.run(webui._restore_stored_collection())
        assert webui._current_collection == "xian_aef_embedding"
        assert webui.state["manager"].collection_name == "xian_aef_embedding"

    def test_restore_invalid_falls_back_to_default(self, monkeypatch):
        async def fake_js(code, catch=True):
            return "deleted_collection"

        monkeypatch.setattr(webui.ui, "run_javascript", fake_js)
        monkeypatch.setattr(webui, "_current_collection", "google_aef_embedding")
        monkeypatch.setattr(webui, "_known_collections",
                            ["google_aef_embedding", "xian_aef_embedding"])
        webui.state = {"manager": None}

        async def fake_hook(name):
            pass

        monkeypatch.setattr(webui, "_call_hook", fake_hook)
        asyncio.run(webui._restore_stored_collection())
        assert webui._current_collection == webui.DEFAULT_COLLECTION
        assert webui.state["manager"].collection_name == webui.DEFAULT_COLLECTION


class TestApplyCollection:
    """切换后 state['manager'] 必须替换为新 collection 的 manager 实例."""

    def test_switches_and_rebuilds_manager(self, monkeypatch):
        monkeypatch.setattr(webui, "_current_collection", "google_aef_embedding")
        monkeypatch.setattr(webui, "_known_collections",
                            ["google_aef_embedding", "xian_aef_embedding"])
        calls: list = []

        async def fake_hook(name):
            calls.append(name)

        monkeypatch.setattr(webui, "_call_hook", fake_hook)
        persisted: list = []
        monkeypatch.setattr(webui, "_persist_collection_choice", lambda c: persisted.append(c))
        webui.state = {"manager": None}
        asyncio.run(webui._apply_collection("xian_aef_embedding"))
        assert webui._current_collection == "xian_aef_embedding"
        assert webui.state["manager"].collection_name == "xian_aef_embedding"
        assert webui._manifest_cache is None, "切换后 manifest 缓存应失效化"
        assert persisted == ["xian_aef_embedding"]
        assert calls == ["sync_tabs", "refresh_status", "refresh_image_list", "render_preview"]

    def test_same_collection_is_noop(self, monkeypatch):
        monkeypatch.setattr(webui, "_current_collection", "xian_aef_embedding")
        webui.state = {"manager": MagicMock()}
        manager = webui.state["manager"]
        asyncio.run(webui._apply_collection("xian_aef_embedding"))
        assert webui.state["manager"] is manager, "同 collection 不应重建 manager"


class TestAddCustomCollection:
    """自定义 collection：校验（非空/无路径分隔符）→ 加入已知列表 → 切换."""

    def test_valid_adds_and_switches(self, monkeypatch):
        async def fake_hook(name):
            pass

        monkeypatch.setattr(webui, "_call_hook", fake_hook)
        monkeypatch.setattr(webui, "_current_collection", "google_aef_embedding")
        monkeypatch.setattr(webui, "_known_collections", list(webui.PRESET_COLLECTIONS))
        webui.state = {"manager": None}
        asyncio.run(webui._add_custom_collection("my_embedding"))
        assert "my_embedding" in webui._known_collections
        assert webui._current_collection == "my_embedding"
        assert webui.state["manager"].collection_name == "my_embedding"

    def test_empty_name_rejected(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        asyncio.run(webui._add_custom_collection("   "))
        assert any("不能为空" in str(m) for m, _ in fake.notify_calls)

    def test_path_separator_rejected(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        monkeypatch.setattr(webui, "_known_collections", list(webui.PRESET_COLLECTIONS))
        asyncio.run(webui._add_custom_collection("a/b"))
        assert "a/b" not in webui._known_collections
        assert any("路径分隔符" in str(m) for m, _ in fake.notify_calls)


class TestCollectionCacheOps:
    """Task 3.4: 分页内「刷新」「清理缓存」，仅作用于当前 collection."""

    def test_refresh_reconciles_and_invalidates(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        mgr = MagicMock()
        monkeypatch.setattr(webui, "state", {"manager": mgr})
        monkeypatch.setattr(webui, "_current_collection", "google_aef_embedding")
        monkeypatch.setattr(
            "KNN_evaluation.sampling_map.ensure_sampling_map", lambda manager: None,
        )

        async def fake_hook(name):
            pass

        monkeypatch.setattr(webui, "_call_hook", fake_hook)
        asyncio.run(webui._do_refresh())
        assert mgr.reconcile_manifest.called
        assert webui._manifest_cache is None

    def test_clear_cache_deletes_only_current_collection(self, monkeypatch, tmp_path):
        fake = _build_harness(monkeypatch)
        from KNN_evaluation.sampling_map import sampling_map_path
        monkeypatch.setattr(webui, "_current_collection", "google_aef_embedding")
        monkeypatch.chdir(tmp_path)
        files = [
            manifest_path("google_aef_embedding"), manifest_path("xian_aef_embedding"),
            Path(sampling_map_path("google_aef_embedding")),
            Path(sampling_map_path("xian_aef_embedding")),
        ]
        for p in files:
            p.write_text("{}", encoding="utf-8")

        async def fake_hook(name):
            pass

        monkeypatch.setattr(webui, "_call_hook", fake_hook)
        asyncio.run(webui._do_clear_cache())
        assert not manifest_path("google_aef_embedding").exists()
        assert not Path(sampling_map_path("google_aef_embedding")).exists()
        assert manifest_path("xian_aef_embedding").exists()
        assert Path(sampling_map_path("xian_aef_embedding")).exists()
        assert webui._manifest_cache is None

    def test_page_renders_tabs_and_cache_buttons(self, monkeypatch):
        fake = _build_harness(monkeypatch)
        assert fake.buttons.get("刷新") is not None
        assert fake.buttons.get("清理缓存") is not None
        assert fake.inputs.get("自定义 collection 名称") is not None
        assert fake.buttons.get("添加") is not None
```

注：`TestClearCache` 需要 `manifest_path` / `Path` 已在 `KNN_evaluation/tests/test_webui.py` 作用域内可用。文件顶部已 `from pathlib import Path`；在测试类内用局部 import `from KNN_evaluation.manifest import manifest_path` 更稳妥——将上述测试中出现的 `manifest_path(...)` 改为文件级可用即可（在测试文件顶部追加 `from KNN_evaluation.manifest import manifest_path`）。

- [x] **Step 2: 运行测试确认失败**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: FAIL — `AttributeError: module 'KNN_evaluation.webui' has no attribute '_apply_collection'` 等（模块级函数尚未实现）

- [x] **Step 3: 实现 webui.py（第二部分）**

在 `KNN_evaluation/webui.py` 顶部 import 区补充 `import json`（放在 `import asyncio` 附近）：

```python
import json
```

在 `_LOCALSTORAGE_KEY` 定义之后（模块级）追加模块级函数：

```python
def _resolve_stored_collection(stored: str | None) -> str:
    """localStorage 记录值有效则恢复，否则回退默认（spec：记录失效回退）."""
    if stored and stored in _known_collections:
        return stored
    return DEFAULT_COLLECTION


def _persist_collection_choice(collection: str) -> None:
    """将当前选择写回 localStorage（fire-and-forget，不 await）."""
    ui.run_javascript(
        f"localStorage.setItem({json.dumps(_LOCALSTORAGE_KEY)}, {json.dumps(collection)})",
    )


async def _apply_collection(new_col: str) -> None:
    """切换当前 collection：更新模块级状态、重建 manager、失效缓存、持久化、刷新 UI.

    关键点（D3）：`state["manager"]` 必须替换为新的 manager 实例，否则旧 collection
    句柄残留导致数据串扰；`_current_collection` 存模块级，避免 index() 重建 state 丢失。
    """
    global _current_collection
    if new_col == _current_collection:
        return
    _current_collection = new_col
    _invalidate_manifest_cache()
    state["manager"] = QdrantManager(url=_CLI_QDRANT_URL, collection_name=_current_collection)
    _persist_collection_choice(_current_collection)
    await _call_hook("sync_tabs")
    await _call_hook("refresh_status")
    await _call_hook("refresh_image_list")
    await _call_hook("render_preview")


async def _add_custom_collection(name: str) -> None:
    """自定义 collection：校验名称并加入已知列表，随后切换.

    校验：非空、不含路径分隔符（/ 或 \\）。加入后重建分页并切换到新 collection.
    """
    name = (name or "").strip()
    if not name:
        ui.notify("collection 名称不能为空", type="warning")
        return
    if "/" in name or "\\" in name:
        ui.notify("collection 名称不能包含路径分隔符 / 或 \\", type="negative")
        return
    if name not in _known_collections:
        _known_collections.append(name)
        await _call_hook("rebuild_tabs")
    await _apply_collection(name)


async def _do_refresh() -> None:
    """分页内刷新：重新对账当前 collection 的 manifest 与采样地图，并刷新 UI."""
    mgr = state.get("manager")
    if mgr is None:
        return
    try:
        mgr.reconcile_manifest()
    except Exception:
        pass
    try:
        from KNN_evaluation.sampling_map import ensure_sampling_map
        ensure_sampling_map(mgr)
    except Exception:
        pass
    _invalidate_manifest_cache()
    await _call_hook("refresh_status")
    await _call_hook("refresh_image_list")
    await _call_hook("render_preview")
    ui.notify(f"已刷新 {_current_collection}", type="positive")


async def _do_clear_cache() -> None:
    """分页内清理缓存：删除当前 collection 的采样地图与导入 manifest 文件（不影响其他分页）."""
    from KNN_evaluation.sampling_map import sampling_map_path
    paths = [manifest_path(_current_collection), Path(sampling_map_path(_current_collection))]
    for p in paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    _invalidate_manifest_cache()
    await _call_hook("refresh_status")
    await _call_hook("refresh_image_list")
    ui.notify(f"已清理 {_current_collection} 的采样地图与 manifest 缓存", type="positive")


async def _restore_stored_collection() -> None:
    """页面加载：读取 localStorage 记录并恢复（记录失效回退默认）."""
    try:
        stored = await ui.run_javascript(
            f"localStorage.getItem({json.dumps(_LOCALSTORAGE_KEY)}) || ''",
            catch=True,
        )
    except Exception:
        stored = ""
    target = _resolve_stored_collection(stored)
    if target != _current_collection:
        await _apply_collection(target)
    else:
        await _call_hook("sync_tabs")
```

在 `index()` 中，header 之后（`with ui.header(...)` 块结束之后、"Qdrant 连接 & Collection 状态" expansion 之前）插入分页选择器：

```python
    # ===== Collection 分页选择器（D3） =====
    # tab 为空面板，页面内容在下方统一渲染，随 _current_collection 切换刷新（同可视化对话框模式）
    tabs = ui.tabs()
    panels = ui.tab_panels(tabs, value=_current_collection)
    with tabs:
        for col in _known_collections:
            ui.tab(col, label=col)
    with panels:
        for col in _known_collections:
            ui.tab_panel(col)

    def _sync_tabs():
        tabs.set_value(_current_collection)
        panels.set_value(_current_collection)

    def _rebuild_tabs():
        tabs.clear()
        with tabs:
            for col in _known_collections:
                ui.tab(col, label=col)
        panels.clear()
        with panels:
            for col in _known_collections:
                ui.tab_panel(col)
        _sync_tabs()

    async def _on_tab_change():
        await _apply_collection(tabs.value)

    tabs.on("update:model-value", lambda e: asyncio.create_task(_on_tab_change()))

    # 自定义 collection 添加 + 分页内缓存管理
    with ui.row().classes("w-full items-center gap-2"):
        custom_col_input = ui.input(
            label="自定义 collection 名称", placeholder="如 my_embedding",
        ).classes("w-64")
        ui.button(
            "添加",
            on_click=lambda: asyncio.create_task(_add_custom_collection(custom_col_input.value)),
        ).props("flat")
        ui.button("刷新", on_click=_do_refresh).props("flat")
        ui.button("清理缓存", on_click=_do_clear_cache).props("flat")
```

在 `index()` 末尾自动初始化区块中，`_init_hooks` 注入处（`_init_hooks["render_preview"] = _render_preview` 之后）追加：

```python
    _init_hooks["sync_tabs"] = _sync_tabs
    _init_hooks["rebuild_tabs"] = _rebuild_tabs
```

并在 `ui.timer(0.1, callback=init_page, once=True)` 之后追加 localStorage 恢复定时器：

```python
    ui.timer(0.2, callback=lambda: asyncio.create_task(_restore_stored_collection()), once=True)
```

注：NiceGUI 3.15 的 `Tabs`/`TabPanels` 无动态 `.add()`，故自定义分页用 `clear()` + 重建 + `set_value()` 实现；若构建时发现重建行为异常，保持可观察行为不变的前提下调整 UI 表达（见 Design Doc D3 的框架层说明）。

- [x] **Step 4: 运行测试确认通过**

Run: `uv run pytest KNN_evaluation/tests/test_webui.py -v`
Expected: PASS（新增 TestCollectionRestore / TestApplyCollection / TestAddCustomCollection / TestCollectionCacheOps 全部通过；既有测试不受影响）

- [x] **Step 5: 手动冒烟（可选，需 Qdrant 运行）**

Run: `uv run python KNN_evaluation/webui.py --port 8003 --dir data_demo`
Expected: 浏览器打开后顶部出现 `google_aef_embedding` / `xian_aef_embedding` 两个分页；点击切换后状态区 collection 名变化；输入自定义名称点「添加」生成新分页并切换；「刷新」「清理缓存」按钮可点；刷新页面后恢复上次选择。

- [x] **Step 6: Commit**

```bash
git add KNN_evaluation/webui.py KNN_evaluation/tests/test_webui.py
git commit -m "feat(webui): collection 分页选择器、切换重建 manager、localStorage 记忆、刷新/清理缓存"
```

---

### Task 8: README 文档 + 全量回归（tasks 6.1 + 5.4）

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: 全部已完成功能。

- [x] **Step 1: 更新 README.md**

修改 `README.md` 的「CLI 用法」代码块，为各命令补充 `--collection` 示例并新增说明：

```markdown
## CLI 用法

```bash
# 批量导入像素到 Qdrant（--collection 指定目标 collection，默认 google_aef_embedding）
uv run python -m KNN_evaluation.cli import <directory> [--no-resume] [--reindex] [--collection <名称>]

# 向量检索
uv run python -m KNN_evaluation.cli search --query-file <query.npy> [--k 10] [--exact] [--collection <名称>]

# 查看 Collection 统计（输出中展示实际 collection 名称）
uv run python -m KNN_evaluation.cli stats [--json] [--collection <名称>]

# 评估 embedding 质量 (F1 / Purity / Recall@K)
uv run python -m KNN_evaluation.cli evaluate [--samples-per-class 500] [--k-f1 10] [--output result.json] \
  [--device cuda|cpu|auto] [--gpu-batch-q N] [--max-gpu-mem GB] [--max-eval-ram GB] [--collection <名称>]

evaluate 评估用法说明：
- `--device cuda|cpu|auto`：评估执行设备（默认 auto——CUDA 可用则 GPU，否则 torch CPU 分块）
- `--gpu-batch-q`：查询分块大小（默认按预算推导）
- `--max-gpu-mem`：显存预算上限 GB（默认 16）
- `--max-eval-ram`：CPU 回退 RAM 预算 GB（默认 6）
- `--collection`：目标 Qdrant Collection 名称（默认 `google_aef_embedding`）
- 安装依赖：`uv sync`（含 torch）

# 重建 Collection（切换存储配置，默认 disk）
uv run python -m KNN_evaluation.cli migrate [--storage disk|ram] [--dir data_demo] [--collection <名称>]

collection 说明：
- 预置 collection：`google_aef_embedding`（默认）与 `xian_aef_embedding`。
- manifest 与采样地图按 collection 隔离存储：
  `qdrant_import_manifest_<collection>.json` / `qdrant_sampling_map_<collection>.json`。
```

修改「启动 WebUI」小节，追加 collection 选择器说明：

```markdown
打开浏览器访问 `http://localhost:8003`。支持数据导入、像素级向量检索、UTM/标签过滤、检索结果可视化与 embedding 质量评估面板。

**Collection 选择器**：页面顶部以分页（tab）形式展示 collection——预置 `google_aef_embedding` / `xian_aef_embedding` 两个分页，并支持输入自定义名称添加新分页。点击分页切换当前 collection，会话内所有操作（导入/检索/评估/清单缓存）跟随切换；切换选择会记忆到浏览器 localStorage，下次打开自动恢复（记录失效时回退默认 `google_aef_embedding`）。每个分页提供「刷新」（重新对账 manifest 与采样地图）与「清理缓存」（删除当前 collection 的采样地图与 manifest 文件，不影响其他分页）。
```

- [x] **Step 2: 全量回归测试**

Run: `uv run pytest KNN_evaluation/tests/ -v`
Expected: ALL PASS（若环境无 Qdrant Docker，依赖 `qdrant_manager` fixture 的集成测试可能跳过/失败——先确认本地 Qdrant 容器运行或 `docker ps` 可见 `qdrant`/`qdrant-knn-eval`；单元测试全部通过）

- [x] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README 补充 CLI --collection 用法与 WebUI collection 选择器说明"
```

---

### Task 9: 删除自定义 collection 分页（tasks 7.1-7.6，build 阶段用户新增需求）

**Files:**
- Modify: `KNN_evaluation/qdrant_client.py`（新增 `delete_collection()` 封装）
- Modify: `KNN_evaluation/webui.py`（删除入口 + 确认框 + 删除流程）
- Modify: `KNN_evaluation/tests/test_qdrant_client.py`、`KNN_evaluation/tests/test_webui.py`

**Interfaces:**
- Consumes: `QdrantManager`（collection_exists/collection_name）、`_known_collections`、`_apply_collection`、`manifest_path`/`sampling_map_path`、`_current_collection`。
- Produces: `QdrantManager.delete_collection()`；webui 删除按钮与确认流程。

**决策（用户已确认）**:
- 预置分页（`google_aef_embedding`/`xian_aef_embedding`）不显示删除入口，始终存在。
- 仅自定义添加的 collection 分页显示删除按钮。
- 删除流程：点击删除 → 确认提示框（提示「将同步删除 Qdrant Collection 及其全部数据，不可恢复」）→ 确认后：`delete_collection()` 删 Qdrant → 删除 `manifest_path`/`sampling_map_path` 缓存文件（OSError 容忍）→ 从 `_known_collections` 移除并重建分页 → 若删的是当前分页则 `_apply_collection(DEFAULT_COLLECTION)`。
- Qdrant 删除失败：保留分页与缓存，提示错误可重试。

- [x] **Step 1: 写失败测试**（test_qdrant_client.py 新增 `test_delete_collection_calls_client`；test_webui.py 新增删除流程/确认框/预置不可删/失败保留/删除后切换测试）
- [x] **Step 2: 运行测试确认失败**
- [x] **Step 3: 实现 qdrant_client.py `delete_collection()` + webui.py 删除功能**
- [x] **Step 4: 运行测试确认通过**（`uv run pytest KNN_evaluation/tests/test_qdrant_client.py KNN_evaluation/tests/test_webui.py -v`）
- [x] **Step 5: 全量回归**（`uv run pytest KNN_evaluation/tests/ -q`）
- [x] **Step 6: Commit**

---

## Self-Review 清单（实施前请逐项确认）

1. **spec 覆盖**：
   - tasks 1.1（config 常量）→ Task 1
   - tasks 2.1/2.2/2.3（CLI --collection / manager 注入 / 输出展示 collection 名）→ Task 5
   - tasks 3.1/3.2（模块级 _current_collection / 替换硬编码）→ Task 6
   - tasks 3.3/3.4（分页选择器 + 切换回调 + 缓存失效 + 刷新）→ Task 7
   - tasks 4.1/4.2/4.3/4.4（manifest / 采样地图隔离 + 安全清洗 + 调用点）→ Task 2/3/4（importer 经 `update_manifest` 缺省派生自动覆盖；metrics 经 `ensure_sampling_map` 缺省派生自动覆盖）
   - tasks 5.1/5.2/5.3/5.4（测试）→ 各任务内 TDD + Task 8 全量回归
   - tasks 6.1（README）→ Task 8
2. **占位符检查**：所有代码步骤均为完整可执行代码与精确命令。
3. **类型/命名一致性**：`safe_collection_token` / `manifest_path` / `sampling_map_path` / `_current_collection` / `_known_collections` / `_apply_collection` / `_add_custom_collection` / `_do_refresh` / `_do_clear_cache` / `_restore_stored_collection` / `_resolve_stored_collection` / `_persist_collection_choice` 在各任务间保持一致。
