---
comet_change: collection-selector
role: technical-design
canonical_spec: openspec
archived-with: 2026-08-07-collection-selector
status: final
---

# Collection Selector 深度技术设计

## Context

承接 open 阶段 design.md（`openspec/changes/collection-selector/design.md`）的高层决策，本文件细化实现。动机与需求见 proposal 与 delta spec。

关键现状（已核实）：
- `QdrantManager.__init__(url, collection_name, timeout)` 已支持注入 collection（`qdrant_client.py:13-22`），`collection_name` 属性贯穿 importer / searcher / metrics / corpus_cache / sampling_map。
- CLI 各子命令以 `QdrantManager(url=args.qdrant_url)` 构造 manager，未传 collection（`cli.py:91,159,267,313,609`）。
- WebUI 硬编码 `COLLECTION_NAME`（约 10 处：`webui.py:414,420,425,651,662,724,930`），manager 在 `init_page()` 构造并存入模块级 `state["manager"]`；`state` 字典在 `index()` 重建（`webui.py:125,363`）。
- `manifest.py` 使用常量 `MANIFEST_PATH = Path("qdrant_import_manifest.json")`，`load/save/update` 均以默认路径操作，内容记录 `collection` 字段但对账会跨 collection 覆盖。
- `sampling_map.py` 使用常量 `SAMPLING_MAP_PATH = "qdrant_sampling_map.json"`，`ensure_sampling_map` 以 collection 名称 + total_points 对账，但文件全局唯一，切换会互相覆盖。
- `corpus_cache.py` 已按 `sha256(collection)[:16]` 隔离文件名（`corpus_cache.py:42-46`），**无需改动**。
- `metrics.py:105` 调用 `ensure_sampling_map(manager)`；`cli.py` 的 `cmd_import` 通过 `PixelImporter` 写 manifest。

## Goals / Non-Goals

**Goals:**
- CLI 各子命令通过 `--collection` 选择 collection，默认 `google_aef_embedding`。
- WebUI 以分页（tab）展示 collection（预置两页 + 自定义添加），切换后会话内所有操作跟随；分页内提供「刷新」「清理缓存」；localStorage 记忆上次选择。
- 采样地图 / manifest 按 collection 隔离文件名，安全清洗 collection 名。
- 保持「可重建缓存」语义：文件缺失/损坏自动重建。

**Non-Goals:**
- 不迁移/重建任何现有 collection 数据；不删除旧缓存文件（仅不再读取）。
- 不修改 `LinearProbe_evaluation/`。
- 不做多用户、权限、collection 间数据复制。

## Decisions

### D1: config.py 常量组织
新增：
```python
DEFAULT_COLLECTION = "google_aef_embedding"
PRESET_COLLECTIONS = ["google_aef_embedding", "xian_aef_embedding"]
```
`COLLECTION_NAME` 保留为 `DEFAULT_COLLECTION` 的别名（`COLLECTION_NAME = DEFAULT_COLLECTION`），避免大范围替换引用、保持 `QdrantManager` 默认参数兼容。
- 备选：直接删除 `COLLECTION_NAME` 全量改名——引用点多、收益低。

### D2: CLI `--collection` 注入
各子命令 parser 增加 `--collection`（default=`DEFAULT_COLLECTION`），`cmd_*` 改为 `QdrantManager(url=args.qdrant_url, collection_name=args.collection)`。
- `cmd_evaluate` 内对 `ensure_sampling_map(manager, path)` 的调用需改为按 collection 派生路径（见 D4）。
- `cmd_import` / `cmd_migrate` 的 manifest 读写经 `PixelImporter` / `reconcile_manifest` 内部完成，需保证这些内部调用按 collection 派生路径。

### D3: WebUI 分页模式 + 全局 current collection
- 模块级 `_current_collection: str = DEFAULT_COLLECTION`（与 `_CLI_QDRANT_URL` 同模式）。
- `init_page()` 构造 `QdrantManager(url=_CLI_QDRANT_URL, collection_name=_current_collection)`。
- 所有 `COLLECTION_NAME` 引用改为读取 `_current_collection`（或 `state["manager"].collection_name`）。
- 分页组件：`ui.tabs()` 渲染预置 collection + 动态自定义 collection tab；`ui.tab_panels` 每 tab 一个面板，共用同一套 UI 元素（manager 唯一），切换 tab 时：
  1. 更新 `_current_collection`
  2. 失效化 manifest 缓存（`_invalidate_manifest_cache()`）与采样地图缓存
  3. 用新 collection 重建 manager（`QdrantManager(url=..., collection_name=_current_collection)`）
  4. 刷新 collection 信息与相关区域（复用 `refresh_status` / 数据列表重载）
- 自定义 collection：输入框 + 「添加」按钮 → 校验名称（非空、无路径分隔符）→ 追加 tab 并切换。
- 「刷新」按钮：重新对账当前 collection（`reconcile_manifest` + `ensure_sampling_map` 重建）+ 刷新 UI。
- 「清理缓存」按钮：删除当前 collection 的 `qdrant_sampling_map_<name>.json` 与 `qdrant_import_manifest_<name>.json`，不触碰其他分页。
- localStorage：页面加载时读 `localStorage.getItem("comet.knn.current_collection")`；若值在预置/已添加列表内则恢复，否则回退 `DEFAULT_COLLECTION`；切换时写回。用 `ui.run_javascript` / NiceGUI storage 能力实现（实现层细节，spec 只约束行为）。
- **关键点**：`state` 字典在 `index()` 中重建（`webui.py:363`），切换 collection 后 `state["manager"]` 必须替换为新的 manager 实例，否则旧 collection 句柄残留导致数据串扰。`_current_collection` 存模块级而非 state，避免 `index()` 重建丢失。

### D3a: 删除自定义 collection 分页（build 阶段增量，2026-08-07 用户新增需求）
- 预置分页（`google_aef_embedding` / `xian_aef_embedding`）不显示删除入口；仅自定义添加的 collection 分页显示删除按钮。
- 删除流程：点击删除 → 确认提示框（明确提示「将同步删除 Qdrant Collection 及其全部数据，不可恢复」）→ 确认后执行：
  1. `manager.client.delete_collection(collection)` 删除 Qdrant Collection；
  2. 删除该 collection 的本地缓存：`manifest_path(name)` 与 `sampling_map_path(name)` 对应文件（OSError 容忍）；
  3. 从 `_known_collections` 移除该名称并重建分页（`clear()` + `set_value()`）；
  4. 若删除的是当前分页 → `_apply_collection(DEFAULT_COLLECTION)` 切回默认页。
- 删除失败（Qdrant 不可达/权限）：保留分页与缓存，提示错误可重试；不误删分页。
- Qdrant 删除需在 `QdrantManager` 增加 `delete_collection()` 封装（现状 `cli.py:634` 直接调 `client.delete_collection`）。
- localStorage 记录若指向被删 collection，下次加载自动回退默认（既有失效回退逻辑覆盖）。

### D4: manifest / 采样地图按 collection 隔离
- 新增公共工具（放 `manifest.py` 或新 `cache_paths.py`）：
  ```python
  def safe_collection_token(collection: str) -> str:
      # 保留 [A-Za-z0-9_.-]，其余替换为 '_'
  ```
- `manifest.py`：
  - `MANIFEST_PATH` 常量改为函数 `manifest_path(collection) -> Path`，返回 `qdrant_import_manifest_<safe_token>.json`。
  - `load_manifest(path)` / `save_manifest(data, path)` / `update_manifest(..., path)` 保持接受显式 path；新增便捷入口或让调用方传入派生 path。
  - 调用方（`importer.py`、`qdrant_client.py`、`cli.py`、`webui.py`）需传入当前 collection 的派生路径。
- `sampling_map.py`：
  - `SAMPLING_MAP_PATH` 常量改为函数 `sampling_map_path(collection) -> str`，返回 `qdrant_sampling_map_<safe_token>.json`。
  - `ensure_sampling_map(manager, path=None)` 默认按 `manager.collection_name` 派生路径；`metrics.py:105` 无需改动即可跟随。
- 旧文件（`qdrant_import_manifest.json` / `qdrant_sampling_map.json`）不迁移、不删除；首次访问新命名路径自动重建。

### D5: 测试策略
- `test_cli.py`：`--collection` 参数解析与传入 manager（mock `QdrantManager` 断言 `collection_name`）；未指定用默认；指定不存在的 collection 时报错路径。
- `test_webui.py`：分页切换后 `state["manager"].collection_name` 变化；自定义 collection 添加；localStorage 记忆/恢复/失效回退（mock `ui.run_javascript` 或注入 localStorage 值）；刷新/清理缓存回调触发对应文件删除/重建。
- `test_manifest.py` / `test_sampling_map.py`：`manifest_path` / `sampling_map_path` 派生与安全清洗；不同 collection 读写互不覆盖。
- 全量 `uv run pytest KNN_evaluation/tests/ -v`。

## Risks / Trade-offs

- [WebUI `state` 在 `index()` 重建导致 collection 状态丢失] → `_current_collection` 存模块级，manager 在 `init_page` 重建时读取它。
- [localStorage 记录失效（collection 被删/重命名）] → 加载时校验值在已知列表内，否则回退默认。
- [collection 名称含特殊字符/路径分隔符] → `safe_collection_token` 清洗文件名；自定义输入校验拒绝空串与路径分隔符。
- [分页切换不清理缓存 → 旧 collection 缓存残留磁盘] → 设计如此（用户自主清理），磁盘占用可控（采样地图 480MB 级，按需手动清）。
- [`ensure_sampling_map` / manifest 调用点多，漏改某处导致跨 collection 覆盖] → 以「按 collection 派生路径」为唯一入口改造，测试覆盖两个 collection 并行场景。
- [NiceGUI 分页 + localStorage 的实现细节（tab 动态增删）] → 实现层风险，spec 只约束可观察行为；构建时如遇框架限制回到本设计调整 UI 表达。

## Migration Plan

1. 代码改动随 build 阶段分任务落地，先 CLI（低风险）后 WebUI（UI 复杂度高）。
2. 旧缓存文件保留，不迁移；新命名路径首次访问自动重建（采样地图重建约数分钟，manifest 对账毫秒级）。
3. 回滚：git revert；WebUI 回退后默认值变化（`pixel_embeddings` → `google_aef_embedding`）需注意——如目标环境仍只有旧 collection，可改 `DEFAULT_COLLECTION` 常量即可恢复。

## Open Questions

无（关键决策均已与用户确认；NiceGUI 分页动态 tab 的具体 API 属于实现层，构建时按框架能力落地）。
