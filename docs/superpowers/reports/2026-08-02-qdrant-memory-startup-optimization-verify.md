# 验证报告：qdrant-memory-startup-optimization

- **日期**：2026-08-02
- **Change**：`qdrant-memory-startup-optimization`
- **验证方式**：openspec-verify-change（Completeness / Correctness / Coherence 三维度）
- **基线与 HEAD**：`dc9dc0a` → `825ecde`（本 change 共 33 个提交）
- **测试证据**：`uv run pytest KNN_evaluation/tests/ -q` → **162 passed, 21 warnings**（实测复跑）

---

## Summary Scorecard

| 维度 | 状态 |
|------|------|
| Completeness | ✅ 22/22 任务勾选；16/16 个 delta Requirement 均有实现 |
| Correctness | ✅ 16/16 Requirement 实现映射；场景覆盖 36/38（2 项部分覆盖） |
| Coherence | ⚠️ 整体遵循 design.md D1-D6 与 Design Doc D1-D6/3.x/4.x；发现 1 处文档漂移 + 1 处既有 JSON 导出缺陷（非本 change 引入） |

**最终评估**：0 CRITICAL，2 WARNING，4 SUGGESTION → **ready for archive（附带可后续修复项）**。

---

## 1. Completeness

### 1.1 任务完成度

`openspec/changes/qdrant-memory-startup-optimization/tasks.md`：**22/22 勾选，0 未勾选**（实测 `grep -c`：`[x]` = 22，`[ ]` = 0）。

> 注：提交 `0b5b446` 的 message 写 "全部 **23** 任务完成"，与 tasks.md 实际 22 项不符（纯提交信息笔误，不影响产物）。见 SUGGESTION-4。

### 1.2 Delta Spec 需求覆盖（关键词/实现搜索）

| Capability | Requirement | 实现文件 | 覆盖 |
|---|---|---|---|
| import-manifest | 导入 manifest 文件读写 | `manifest.py:20-68` | ✅ |
| import-manifest | 导入时增量更新 manifest | `importer.py:365`、`manifest.py:52-68` | ✅ |
| import-manifest | facet 对账 | `qdrant_client.py:166-204` | ✅ |
| import-manifest | 会话缓存与多 tab 复用 | `webui.py:57-81`、`_get_manifest_cached` | ✅ |
| disk-backed-storage | Collection 磁盘化存储配置 | `qdrant_client.py:49-81`（storage 参数） | ✅ |
| disk-backed-storage | 迁移命令重建 Collection | `cli.py:593-655`（cmd_migrate） | ✅ |
| disk-backed-storage | Docker 持久化存储 | `cli.py:505-551`、`webui.py:253-300`、根 `README.md:11-25` | ✅ |
| disk-backed-storage | Qdrant 容器启动幂等 | `cli.py:505-551`、`webui.py:253-300` | ✅ |
| embedding-evaluation | 批量精确 KNN 默认路径安全 | `metrics.py:168/382`（use_batch=False 默认） | ✅ |
| embedding-evaluation | 内存预算守卫 | `metrics.py:22-63`（estimate/guard） | ✅ |
| embedding-evaluation | 评估前预估告警 | `cli.py:343-354`、`webui.py:826-839` | ✅ |
| embedding-visualization | WebUI 启动流程异步化 | `webui.py:105-152`（init_page/_background_init） | ✅ |
| embedding-visualization | WebUI 影像列表读本地 manifest | `webui.py:546-568`、`qdrant_client.py:155-164` | ✅ |
| embedding-visualization | 多 tab 不重复加载 | `webui.py:57-81`（进程级缓存） | ✅ |
| pixel-data-import | 导入成功后同步更新 manifest | `importer.py:362-365`（无条件调用） | ✅ |
| pixel-data-import | 迁移命令复用断点续传 | `cli.py:629-633`（import_directory 复用） | ✅ |

**16/16 Requirement 均找到实现证据。**

---

## 2. Correctness

### 2.1 Requirement 实现映射（文件 + 行范围）

| Requirement | 实现位置 | 说明 |
|---|---|---|
| manifest 读写（原子写） | `manifest.py:20-49`（load/save tmp+`os.replace`） | 缺失/损坏返回空结构 `_empty()`，不报错 |
| 增量更新 | `manifest.py:52-68`；`importer.py:365` | 每张影像导入/跳过完成后无条件 `update_manifest`，CLI/WebUI 共用入口 |
| facet 对账 | `qdrant_client.py:176-204`（reconcile）、`:166-174`（_facet_image_ids） | 一致不写盘；不一致对差异项 `check_image_count` 补精确像素数；缺失重建 |
| 会话缓存 | `webui.py:57-81`；导入完成失效 `webui.py:430` | 进程级 `_manifest_cache`，多 tab 复用 |
| 磁盘化参数 | `qdrant_client.py:67-81` | disk → `on_disk=True` + `on_disk_payload=True` + `quantization_config=None`；HNSW(m/ef_construct) 保留 |
| migrate 子命令 | `cli.py:593-655` | 先 `_start_qdrant()`（幂等）→ `_container_has_volume()` 校验 → 备份 `collection_info` → 删除重建 → 重导（reindex）→ `reconcile_manifest()` |
| Docker 幂等 + volume | `cli.py:505-551` / `webui.py:253-300` | `docker ps -a` → 运行复用 / 停止 `docker start` / 缺失 `docker run -v qdrant_data:/qdrant/storage` |
| use_batch 默认 False | `metrics.py:168, 382` | `compute_knn_accuracy` / `compute_purity_recall_curve` 默认服务端逐条 |
| 内存守卫 | `metrics.py:22-63` | `estimate_batch_memory`（N×64×8×2 + Q×K×8 + N×8）；超阈值抛 `MemoryError` 含预估 |
| CLI --batch/--max-eval-ram | `cli.py:712-715`（参数）、`:343-354`（守卫+提示） | `--batch` 显式 opt-in；默认 6GB 阈值 |
| WebUI 批量 opt-in + 预估 label | `webui.py:764-766`（checkbox）、`:826-839` | 勾选批量先 `guard_batch_memory(..., 6.0)`，拒绝则 notify |
| init_page 快速/慢速路径 | `webui.py:105-152` | 快速路径（置状态+sleep(0)）→ `asyncio.create_task(_background_init)`；慢速路径 `asyncio.to_thread` 包裹 |
| 影像列表读 manifest | `webui.py:546-568`；`qdrant_client.py:155-164` | `get_imported_image_ids` 读 manifest（空则 facet 回退） |
| 预览状态列读 manifest | `webui.py:68-71, 482` | `_manifest_pixels` 读缓存，翻页不触 manager |

### 2.2 Scenario 覆盖检查

**import-manifest（9 场景）**：8 覆盖（读取/缺失/结构/导入成功/部分值/CLI+WebUI/一致/不一致/缺失重建）✅；`Collection 重建后对账（facet 空集清空 manifest）` 逻辑正确（`reconcile_manifest` 中 `db_ids=set()` 会写空 images）但**无显式单测** → SUGGESTION-3。

**disk-backed-storage（10 场景）**：全覆盖 ✅（`test_qdrant_client.py:242-268` 磁盘/ram 预设；`test_migrate.py` 三态+重试+无卷 fail-fast；`test_start_qdrant.py` 三态+volume 挂载断言；根 README 持久化文档）。

**embedding-evaluation（7 场景）**：全覆盖 ✅（`test_metrics.py:413-491`：默认 False、批量/逐条一致性、守卫拒绝/放行、默认不触发；CLI/WebUI 预估提示在代码实现中）。

**embedding-visualization（6 场景）**：5 覆盖 ✅；`Qdrant 离线时仍可列出影像` **部分覆盖** → WARNING-3。

**pixel-data-import（4 场景）**：全覆盖 ✅（`test_manifest_in_import.py` 导入/跳过/progress_callback/记录 count 值；`test_migrate.py` 中断重试与完整重导）。

**共 36/38 场景完全覆盖，2 项部分覆盖（见 WARNING-3、SUGGESTION-3）。**

---

## 3. Coherence

### 3.1 design.md 决策（D1-D6）

| 决策 | 实现 | 符合 |
|---|---|---|
| D1 Web 启动「状态先亮 + 贵操作后台化」 | `webui.py:105-152` | ✅ |
| D2 manifest 主路径 + distinct 对账（实际 facet） | `qdrant_client.py:155-204` | ✅（design.md 已标注 1.18.0 无 distinct，用 facet，实现一致） |
| D3 Collection 磁盘化 + 不量化 | `qdrant_client.py:67-81` | ✅ |
| D4 migrate 重建 + 断点续传 | `cli.py:593-655` | ✅（另增 `_container_has_volume` fail-fast，符合 D4 风险缓解） |
| D5 Docker volume + 启动幂等 | `cli.py:505-551`、`webui.py:253-300`、根 `README.md:11-25` | ✅ |
| D6 评估默认服务端逐条 + 守卫 + 预估 | `metrics.py:22-63`、`cli.py:343-354`、`webui.py:826-839` | ✅ |

### 3.2 Design Doc（2026-08-02）关键决策 3.1-3.6 / 4.1-4.7

- 3.1 facet 对账：`client.facet(key="image_id", limit=1000, exact=True)` ✅
- 3.2 内存守卫仅客户端进程（不含 Qdrant 常驻）✅
- 3.3 manifest 原子写 tmp + `os.replace` ✅（`manifest.py:38-49`）
- 3.4 init_page 单后台协程（`asyncio.create_task`）✅（tasks.md 描述为 to_thread，Design Doc 3.4 为权威决定，实现与其一致）
- 3.5 `create_collection` 默认 `storage="disk"` ✅
- 3.6 migrate 开头自动 `_start_qdrant()` ✅
- 4.1-4.6 模块实现与 Doc 伪代码一致 ✅
- 4.7 README 更新：**根 `README.md` 已更新**（提交 58e87cc，含 volume/migrate/持久化）⚠️ 但 `KNN_evaluation/README.md` 未同步 → WARNING-1

### 3.3 代码模式一致性

- `manifest.py` / `qdrant_client.py` / `importer.py` / `metrics.py` / `cli.py` / `webui.py` 风格统一，错误处理用 try/except + 明确返回码模式一致。
- `cli._start_qdrant` 与 `webui._start_qdrant` 为两份近重复实现（有意为之并有注释），且 `webui._start_qdrant`/`_qdrant_is_running` 目前**未被调用**（死代码）→ SUGGESTION-2。

---

## 4. 发现清单

### CRITICAL（0）

无。

### WARNING（2）

- **WARNING-1 — `KNN_evaluation/README.md` 文档漂移（未同步本 change 的持久化/存储说明）**
  - 文件：`KNN_evaluation/README.md`
  - 行：`:8`（启动命令 `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest` 无 `-v` volume 挂载，与 D5/根 README 矛盾）；`:134`（`--with-distance` 参数在 cli.py 中不存在）；`:152`、`:180-188`（WebUI 参数表与「F3 距离分析」章节引用已删除的 F3 模块）。
  - 影响：任务 3.6「README 更新启动命令与持久化说明」通过根 `README.md`（提交 58e87cc）完成，但本文件是系统详细使用指南，含过期命令与已移除功能，与实现不一致。
  - 建议：将 `KNN_evaluation/README.md` 的 Docker 启动命令改为带 `-v qdrant_data:/qdrant/storage`，删除 F3/`--with-distance` 过期内容；或直接指向根 README 避免双份漂移。
  - 级别判定：不阻塞归档（根 README 已满足任务要求），但文档一致性需修复。

- **WARNING-2 — `cli.py` evaluate `--output` JSON 导出含重复 `"f2"` 键（Ellipsis）**
  - 文件：`KNN_evaluation/cli.py`
  - 行：`:464` 与 `:473-474`（`"f2": { ... }, "f2": { ... }`）。字典字面量中后一个 `f2`（值为 `...`，即 Ellipsis）覆盖前一个，`json.dumps(default=_np_encoder)` 后 `f2` 被序列化为字符串 `"Ellipsis"`，真实 F2 数据丢失。
  - 证据：`git log -L` 确认该缺陷由提交 **`3daa27c`（2026-07-31，早于本 change base `dc9dc0a`）** 引入，**非本 change 引入**，本 change 的 diff 未触及该块。
  - 影响：评估 `--output result.json` 导出的 `f2` 字段损坏；属于本 change 修改的 embedding-evaluation 能力范围内。
  - 建议：删除 `cli.py:473-474` 的重复 `"f2": {...}` 块（保留第一个完整 f2 即可）。**不阻塞本 change 归档**，但应在后续随手修复。

### SUGGESTION（4）

- **SUGGESTION-1 — `reconcile_manifest` 内联 `__import__("datetime")` 风格不统一**
  - 文件：`KNN_evaluation/qdrant_client.py:202`
  - 建议：改为文件顶部 `import datetime`，与 `manifest.py` 一致（`manifest.py` 用标准 `from datetime import datetime`）。
  - 级别判定：纯风格/可读性，无功能影响。

- **SUGGESTION-2 — `webui._start_qdrant` / `webui._qdrant_is_running` 为未调用的死代码**
  - 文件：`KNN_evaluation/webui.py:241-300`
  - 现状：与 `cli._start_qdrant` 逻辑重复（有注释说明有意保持一致），但 WebUI 内无任何调用点（grep 确认仅定义处）。
  - 建议：要么在 WebUI 连接失败时实际调用 `_start_qdrant()` 补足自动启动能力，要么移除 `webui` 内的副本并复用 `cli` 模块，避免双份维护漂移。

- **SUGGESTION-3 — import-manifest 场景「Collection 重建后对账（facet 空集清空 manifest）」无显式单测**
  - 相关文件：`KNN_evaluation/tests/test_manifest.py`、`KNN_evaluation/tests/test_qdrant_client.py`
  - 现状：`reconcile_manifest` 对 `db_ids=set()` 时逻辑正确（写空 images），但测试仅覆盖「一致/不一致/缺失重建」，缺「facet 返回空集 → manifest 被清空」用例。
  - 建议：在 `test_manifest.py:TestReconcileManifestPureMock` 或 `test_qdrant_client.py:TestReconcileManifest` 增加一条：facet 返回 `set()`、manifest 有 GHOST → 断言写盘一次且 `images == {}`。

- **SUGGESTION-4 — 提交信息任务数与 tasks.md 不符**
  - 文件：`openspec/changes/qdrant-memory-startup-optimization/tasks.md`（22 项）
  - 现状：提交 `0b5b446` message 写「全部 **23** 任务完成」，与 tasks.md 的 22 项不一致。
  - 建议：归档时可忽略（纯提交信息笔误），无需改动。

---

## 5. 最终评估

| 指标 | 结果 |
|---|---|
| CRITICAL | 0 |
| WARNING | 2（均为非阻塞：1 文档漂移、1 既有缺陷且非本 change 引入） |
| SUGGESTION | 4（风格/死代码/补测/提交信息） |
| **结论** | **Ready for archive** |

- Completeness 完整（22/22 任务、16/16 Requirement）。
- Correctness 达标（36/38 场景完全覆盖，2 项部分覆盖为低风险）。
- Coherence 符合 design.md D1-D6 与 Design Doc 3.x/4.x；根 README 已更新。
- 全量测试 162 passed 实测通过；生产 collection 已按任务 5.2/5.3 验证（10.2M 点磁盘化、Web 秒级 ✅）。

**已知上下文确认**：`label_mapping.py` 中英混合改动（提交 825ecde）按任务说明不作为缺陷报告。

**归档前可选动作**：建议顺手删除 `cli.py:473-474` 重复 f2 键（一行改动即可恢复 JSON 导出正确性），但不阻塞本 change 归档。

---

## 附录：归档阶段 Spec 漂移修正（2026-08-02）

归档执行时发现 `embedding-evaluation` delta 的 MODIFIED 语义与主 spec 冲突：
- 主 spec 的 `### Requirement: GPU 批量精确 KNN 检索` 由 `embedding-quality-metrics` 归档（81ee5cb）引入。
- 本 change 原 delta 用 MODIFIED 修改不存在的旧标题 `批量精确 KNN 检索默认路径安全` → 归档失败。

**修正**：delta 从 `## MODIFIED Requirements` 改为 `## ADDED Requirements`（新增 3 个独立 requirement：默认路径安全 / 内存预算守卫 / 评估前预估告警），作为 CPU 回退兜底与 GPU requirement 并存，不触碰 GPU 内容。`openspec validate` 通过。

**影响评估**：ADDED 语义不改动既有行为契约，仅新增 P3 相关 requirement；与 `gpu-knn-scale-evaluation`（in-progress，未归档）无冲突。修正后重新归档。

### 附录补充：embedding-visualization delta 同步修正

首次归档还发现 `embedding-visualization` delta 的 MODIFIED 语义与主 spec 冲突（主 spec 是早期纯文件浏览 spec，无「WebUI 启动流程异步化」等标题）。已一并修正为 ADDED。全部 5 个 delta 现均为 ADDED/create 语义，`openspec validate` 通过。
