# 验证报告：collection-selector

- Change: `collection-selector`
- 日期: 2026-08-07
- 验证模式: full（17 任务 / 2 delta capabilities / 20 变更文件，`comet state scale` 判定）
- 分支: `feature/20260807/collection-selector`
- base-ref: `684934436b931d3c4fdeeaee015898e725b1890e`

## Summary

| 维度 | 状态 |
|------|------|
| Completeness（完整性） | 17/17 任务完成；2 capability delta specs 全部实现 |
| Correctness（正确性） | 21/21 spec 场景有实现与测试覆盖；全量测试 324 passed, 1 skipped |
| Coherence（一致性） | 设计文档 D1-D5 全部落实；无 spec/design 漂移 |

## 1. 完整性核对

- tasks.md 未完成项：**0**（17/17 勾选）
- plan 未完成项：**0**（全部 Step 勾选）
- delta specs：`collection-selector`（16 场景）+ `import-manifest`（5 场景）均实现

## 2. 正确性核对

### 测试证据

- 全量：`uv run pytest KNN_evaluation/tests/ -q` → **324 passed, 1 skipped**（1 个 skipped 为依赖真实 Qdrant 数据的集成测试，与本次变更无关）
- build 证据：`comet state record-check` 已记录 exit 0

### Spec 场景 → 测试映射（21/21）

| Spec 场景 | 覆盖测试 |
|---|---|
| 未指定 collection 使用默认值 | `test_cli.py::TestCollectionArg::test_default_collection_for_all_subcommands` |
| 预置列表含两个 collection | `test_config.py`（PRESET_COLLECTIONS 断言） |
| WebUI 首次加载无记忆用默认 | `test_webui.py::TestCollectionRestore::test_resolve_stored_valid` |
| WebUI 恢复上次选择 | `test_webui.py::TestCollectionRestore::test_restore_from_local_storage_switches` |
| localStorage 失效回退默认 | `test_webui.py::TestCollectionRestore::test_restore_invalid_falls_back_to_default` |
| CLI 指定 collection 执行 | `test_cli.py::TestCollectionArg::test_override_collection` |
| CLI 未指定 collection | `test_cli.py::TestCollectionArg::test_default_collection_for_all_subcommands` |
| CLI 指定不存在 collection | `test_cli.py::TestCmdCollectionInjection::test_cmd_evaluate_missing_collection_returns_1_with_name`、`TestStatsCollection::test_stats_missing_collection_returns_1_with_name` |
| WebUI 选择预置分页 | `test_webui.py::TestCollectionCacheOps::test_page_renders_tabs_and_cache_buttons`（断言两预置分页渲染） |
| WebUI 自定义 collection | `test_webui.py::TestAddCustomCollection::test_valid_adds_and_switches`、`test_empty_name_rejected`、`test_path_separator_rejected` |
| 切换后导入/检索用新 collection | `test_webui.py::TestApplyCollection::test_switches_and_rebuilds_manager` |
| 分页内刷新 | `test_webui.py::TestCollectionCacheOps::test_refresh_reconciles_and_invalidates`、`test_refresh_notifies_success`、`test_refresh_reports_failure_when_reconcile_fails` |
| 分页内清理缓存 | `test_webui.py::TestCollectionCacheOps::test_clear_cache_deletes_only_current_collection`、`test_clear_cache_tolerates_locked_files` |
| 切换不自动清理缓存 | `test_webui.py::TestApplyCollection::test_same_collection_is_noop` |
| 不同 collection 独立地图 | `test_sampling_map.py`（隔离路径测试） |
| 切换后使用对应地图 | `test_sampling_map.py::test_ensure_sampling_map_derives_path_from_collection` |
| manifest 读取/缺失/结构 | `test_manifest.py` 既有测试 |
| manifest 按 collection 独立清单 | `test_manifest.py::test_update_manifest_default_path_derived_from_collection` |
| 切换后读取对应清单 | `test_webui.py::TestModuleCurrentCollection::test_get_manifest_cached_uses_current_collection_path` |

## 3. 一致性核对

### Design Doc D1-D5 落实

| 决策 | 实现证据 |
|---|---|
| D1 config 常量 | `config.py:9-11`（DEFAULT_COLLECTION / PRESET_COLLECTIONS / 别名） |
| D2 CLI 注入 | `cli.py` 5 处 `QdrantManager(url=..., collection_name=args.collection)` |
| D3 WebUI 模块级 + 分页 | `webui.py` `_current_collection` / `_apply_collection` / `ui.tabs` / `ui.tab_panels` / localStorage |
| D4 缓存隔离 | `manifest_path` / `sampling_map_path` / `safe_collection_token`（`[A-Za-z0-9_.-]` 保留） |
| D5 corpus_cache 不动 | `git diff 6849344...HEAD -- KNN_evaluation/corpus_cache.py` = 0 改动 |

### 设计符合性

- `metrics.py` 未改动（`ensure_sampling_map(manager)` 缺省派生自动跟随）✅
- `LinearProbe_evaluation/` 未触及 ✅
- 旧缓存文件未迁移/未删除（可重建语义）✅
- delta spec 与 Design Doc 无矛盾（spec patch 在 design 阶段已回写并重新生成 handoff）✅

## 4. 代码审查

- build 阶段：`review_mode: standard`——风险任务（Task 2/3/5/6/7）均派发每任务 reviewer 且 approved；最终轻量审查（opus）结论 **可 merge**（0 CRITICAL / 0 IMPORTANT）
- verify 阶段：full 模式按 openspec-verify-change 语义验证，未发现新问题

## 5. 已知限制 / 接受项

| 项 | 级别 | 处理 |
|---|---|---|
| 浏览器冒烟未执行（IAB webview 在本会话无法就绪，无 CDP 后端） | WARNING | WebUI 服务 HTTP 200 启动正常；核心交互（分页渲染/切换/localStorage/清理缓存）由 16 个单元测试全覆盖。接受，建议在真实浏览器环境手动复核 |
| stats `--json` / evaluate JSON 不含 collection 字段 | SUGGESTION | 非 JSON 模式已输出 collection 名，spec 2.3 由非 JSON 模式满足；记录接受 |
| 0.2s localStorage 恢复与后台对账竞态 | SUGGESTION | 最终收敛一致，无数据损坏；单用户本地工具接受 |
| token 清洗碰撞 / 超长名无截断 | SUGGESTION | Qdrant 合法 collection 名 `[A-Za-z0-9_.-]` 下恒等，不可触发 |

## 结论

**验证通过。** 完整性与正确性证据充分（17/17 任务、21/21 场景、324 passed），无 CRITICAL/IMPORTANT 问题，Design Doc 决策全部落实，无 spec 漂移。可进入归档。

## 增量验证（2026-08-07，删除自定义 collection 分页功能）

在归档前用户新增需求「删除自定义 collection 分页」，已按 build 阶段 Spec 增量更新流程处理：
- delta spec 新增「删除自定义 collection 分页」需求（5 场景）
- Design Doc 补充 D3a；plan 新增 Task 9；tasks.md 新增 7.1-7.6（已全部勾选）
- Task 9 实施（de4726b + 修复 1d529e6），审查通过（0 CRITICAL；1 IMPORTANT 已修复——非当前分页删除改用临时 manager；MINOR 修复 2 项，2 项接受）

### 增量场景覆盖（5/5）

| Spec 场景 | 覆盖测试 |
|---|---|
| 删除自定义 collection 分页 | test_delete_flow_removes_collection_and_cleans_cache |
| 删除前确认提示 | test_delete_shows_confirm_dialog_with_warning |
| 删除当前分页后切回默认 | test_delete_current_collection_switches_to_default |
| 预置分页不可删除 | test_preset_collections_have_no_delete_button、test_preset_delete_rejected_without_dialog |
| 删除失败保留分页 | test_delete_failure_keeps_tab_and_cache |
| （边界）非当前分页删除 | test_delete_other_tab_uses_temp_manager_and_keeps_current |

### 增量测试证据

- `uv run pytest KNN_evaluation/tests/ -q` → **335 passed, 1 skipped**（较上轮 +11）
- build 证据已更新记录

**结论：增量验证通过，可进入归档。**
