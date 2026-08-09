# 验证报告 — similarity-heatmap-compare

- **Change**: similarity-heatmap-compare
- **日期**: 2026-08-07
- **验证模式**: full（comet state scale 判定：16 tasks > 3、29 files > 8）
- **分支**: feature/20260807/similarity-heatmap-compare
- **语言**: zh-CN

## 验证证据

### 1. 全量回归测试（全新运行）

```bash
uv run pytest KNN_evaluation/tests/ -v
```

**结果: `369 passed, 22 warnings in 395.56s (0:06:35)`，exit 0（0 失败、0 错误）**

- 本轮为 verify 阶段**新鲜运行**（非复用 build 阶段记录）；证据已通过 `comet state record-check similarity-heatmap-compare verify --command "uv run pytest KNN_evaluation/tests/ -v" --exit-code 0` 记录。
- 22 warnings 为 importer 既有坐标推算回退告警（UserWarning），与本 change 无关。
- 新增测试文件 `test_similarity_compare.py`（21 个测试：采样 8 / 提取+矩阵 6 / 可视化 3 / 编排 4）与 `test_cli.py` TestSimilarityHeatmap（7 个）、`test_webui.py` 面板测试（6 个）全部通过。

### 2. 完整验证 7 项检查

| # | 检查项 | 结果 | 证据 |
|---|--------|------|------|
| 1 | tasks.md 全部任务已完成 | ✅ | 16/16 已勾选，0 未勾选 |
| 2 | 实现符合 design.md 高层决策 | ✅ | 独立模块 `similarity_compare.py` 单一事实源、采样地图复用、同 point_id 双集合提取、numpy 余弦矩阵、visualization 并排渲染、CLI+WebUI 双入口 |
| 3 | 实现符合 Design Doc | ✅ | D1-D8 逐项核对：D1 三函数+编排、D2 双模式采样、D3 同 ids retrieve+剔除、D4 归一化 `V@V.T`+对角 1.0、D5 统一色阶 viridis、D6 CLI 参数、D7 WebUI 固定预置对、D8 返回 6 键契约 |
| 4 | 能力规格场景全部通过 | ✅ | 6 Requirements / 16 Scenarios 全部有对应测试覆盖（见下映射）；369 passed 含全部新场景测试 |
| 5 | proposal.md 目标已满足 | ✅ | CLI 子命令 + WebUI 面板均实现，支持数据库/图片双模式采样，双集合 N×N 余弦矩阵并排热力图对比 |
| 6 | delta spec 与 design doc 无矛盾 | ✅ | 无 Spec 漂移；build 阶段仅 tasks.md 1.1 措辞修正（采样 dict 不携带 UTM，与 D2 对齐），未改 spec 行为 |
| 7 | Design Doc 可定位 | ✅ | `docs/superpowers/specs/2026-08-07-similarity-heatmap-compare-design.md` 存在，frontmatter 关联本 change |

### 3. 规格场景 → 测试覆盖映射

- **随机采样 N 个 UTM 坐标**（5 场景）→ `TestSampleRandomPoints`（test_db_mode_uses_map_and_seed_reproducible / test_n_out_of_range_raises / test_insufficient_candidates_sampled_actual / test_empty_collection_raises / test_image_mode_point_ids_deterministic / test_image_mode_unknown_image_raises 等 8 个）
- **双集合按位置提取 embedding**（2 场景）→ `TestExtractEmbeddings`（test_aligned_matrices_zero_dropped / test_single_side_missing_dropped_and_row_alignment / test_all_missing_raises / test_wrong_dimension_raises）
- **N×N 余弦相似度矩阵计算**（2 场景）→ `TestCosineSimilarityMatrix`（test_symmetric_diagonal_one_range / test_zero_vector_defense）
- **并排热力图生成**（1 场景）→ `TestPlotSimilarityHeatmapPair`（test_renders_png_file / test_renders_to_bytesio / test_unified_color_scale）
- **CLI 子命令**（3 场景）→ `TestSimilarityHeatmap`（test_parser_defaults / test_n_out_of_range_rejected / test_image_id_selects_image_mode / test_qdrant_unreachable_returns_1 / test_missing_collection_returns_1 / test_output_file_generated_and_printed）
- **WebUI 对比面板**（3 场景）→ test_similarity_panel_controls_present / test_mode_switch_toggles_image_select_visibility / test_do_sim_compare_renders_image / test_failure_notifies_negative 等 6 个

### 4. 代码审查

- **build 阶段**（review_mode: standard）：6 个风险任务派发每任务 reviewer（Task1-6），全部通过（Task 6 有 1 个 IMPORTANT 已修复并经复审批准）；最终全分支轻量审查结论 **Ready to merge**（0 CRITICAL / 0 IMPORTANT / 28 项历史 Minor 全部接受）。
- **verify 阶段**：build 之后无新增改动（本验证输入即 build 已审 diff），按去重规则不重复评审。

### 5. 安全检查

- 无硬编码密钥/凭证；无新增 unsafe 操作。
- 路径处理：CLI output 路径归一化 + 父目录自动创建；collection 名称注入走既有 `QdrantManager` 安全路径。
- 无数据库 schema / 数据迁移；不修改 manifest / 采样地图 / corpus 缓存格式。

## 无法验证项（不阻塞）

- 真实 Qdrant / NiceGUI 运行时往返（测试使用 fake manager / `_FakeUI`；CLI 端到端需真实 Qdrant 容器）。
- matplotlib 中文字体渲染视觉质量（依赖系统字体）。

## 结论

**验证通过** — 全部 7 项检查 PASS，无 CRITICAL / IMPORTANT 问题，无 Spec 漂移，无用户决策点。

---

## 增补：导出数据功能（Task 8，2026-08-07）

### 需求变更

用户新增导出功能（归档前确认）：将两个相似度矩阵导出为 `<collection>_similarity.npy`，将采样参数与像素信息导出为 JSON。经归档 reopen → verify-fail 回到 build，确认方案后实施（delta spec 新增 `### Requirement: 导出相似度矩阵与采样信息` 3 场景、Design Doc 追加 D9、tasks.md 新增组 7）。

### 增量实现

- `extract_embeddings` 返回 3 元扩展为 4 元 `(mat_g, mat_x, dropped, kept_records)`（kept_records 含 point_id/image_id/row/col/utm_*，行序与矩阵一致）
- 新增 `export_similarity_outputs`：`<collection>_similarity.npy` ×2 + `similarity_sampling.json`（params + pixels）
- `compare_similarity_heatmaps(..., export_dir=None)`：可选导出，None 时行为与既有版本完全一致（向后兼容）
- CLI `--export-dir`、WebUI「导出目录」输入框（非空才导出）
- OSError 修复（e719754）：导出目录不可写时 CLI/WebUI 显示明确错误而非裸 traceback

### 增量验证

- **全新全量回归：`379 passed, 1 skipped, 22 warnings in 367.23s`（exit 0，0 失败）**（含 OSError 修复后 2 个新测试；skip = CUDA 不可用）
- delta spec 7 需求 19 场景全部有测试覆盖；导出 3 场景 → TestExportSimilarityOutputs + CLI/WebUI 导出测试
- 每任务审查（Task 8）：IMPORTANT=1（全量回归证据缺失，协调者补跑取证 377 passed 闭环）→ 最终轻量审查 Ready to merge（0 CRIT/IMP；3 Minor + 4 观察级接受；OSError 建议已闭环修复）
- 向后兼容验证：`export_dir=None` 时无额外 I/O、既有测试保持 GREEN

---

## 增补：WebUI 导入目录修复 + 默认导出 outputs/（Task 9，2026-08-07）

### 需求变更

用户报告 bug（xian 页热力图与 google 完全一样）+ 两个需求变更（默认导出 outputs/、分页默认数据目录 data_google/data_xian）。经归档 reopen → verify-fail 回到 build，确认方案后实施。

### Bug 根因（已实证）

`xian_aef_embedding` collection 存了 google 数据（identical 向量）→ 热力图与 GOOGLE 完全一致。根因：`webui.py` 的 `do_import()` 用 `state["data_dir"]`（只在点「浏览」时更新），用户改输入框没点浏览 → 导入旧目录。证据链：data/SE(all_mean) vs data_xian/SE(xian_aef) 同坐标内容不同（maxdiff 0.62）；xian collection 向量 == data(all_mean)（maxdiff ~1e-8）；xian collection 全部 268 影像 ⊆ data(google)。

### 增量实现

- `config.py` 新增 `COLLECTION_DATA_DIRS = {"google_aef_embedding": "data_google", "xian_aef_embedding": "data_xian"}`
- `do_import` 从数据目录输入框当前值取值并同步 `state["data_dir"]`（与「浏览」同源）；空输入/目录不存在明确报错
- 分页切换 `_apply_collection` 按映射联动输入框与 state（`sync_data_dir` hook）；自定义 collection 保持默认
- **默认导出 outputs/**：`compare_similarity_heatmaps` 默认 `export_dir="outputs"`（None/空串禁用）；CLI `--export-dir` 默认 `outputs`（空白 strip 后禁用）；WebUI 导出输入框默认 `outputs`（留空禁用）
- 边界修复：`browse_directory` 空输入守卫（不扫 cwd）、CLI 空白导出目录（不建空格目录）

### 增量验证

- **全新全量回归：`396 passed, 1 skipped, 23 warnings in 428.53s`（exit 0，0 失败）**（Task 9 新增 17 个测试；skip = CUDA 不可用）
- delta spec 现 7 需求 24 场景（新增分页默认目录 Requirement 4 场景、导出 Requirement 改默认语义），全部有测试覆盖
- 每任务审查（Task 9）：spec ✅ / quality Approved（0 CRIT/IMP；4 Minor 中 Minor1/3 自动修复、Minor2/4 接受）；最终轻量审查 Ready to merge（新增 Minor5 browse 空输入已自动修复闭环）
- **说明（只修代码不碰数据）**：xian collection 现有错误数据未清空重导，需用户自行清空 `xian_aef_embedding` 后按分页默认目录（data_xian）重新导入，热力图才会正确区分。

---

## 增补：坐标段数值匹配配对（Task 10，2026-08-08）

### 需求变更

用户报告 XIAN 导入仅 499 张。根因（实证）：`data_xian` 的 SE 文件坐标段为 4 位小数（`E121.4033_N25.1370`）而 DW 为 3 位小数（`E121.4033_N25.137`），`scan_directory` 按字符串精确配对失败 125 对。用户确认方案：**匹配阶段把坐标段解析为浮点数做数值比较**，不改 image_id / point_id / 数据库内容（D11）。

### 增量实现

- `data_loader.py` 的 `scan_directory`：SE/DW/TIF 配对 key 由坐标段字符串改为 `parse_location_coord` 数值 `(lon, lat)` 元组；`ImagePair.image_id` 仍取 SE 侧 `extract_location_key` 原始字符串（point_id 语义不变）
- 孤儿文件（数值无对应）保持跳过语义

### 增量验证

- **全新全量回归：`399 passed, 1 skipped, 23 warnings in 500.29s`（exit 0，0 失败）**（Task 10 新增 3 测试）
- **真实数据验证**：`scan_directory(data_xian)` 624 对（修复前 499），`E121.4033_N25.1370` 等 125 对 SE/DW 全配对，image_id 取 SE 原始串
- delta spec 新增「坐标段数值匹配配对」Requirement（3 场景）；每任务审查 + 最终轻量审查 Ready to merge（0 CRIT/IMP，5 Minor 接受）
- 说明：修复后用户可在 XIAN 页重新导入补足 125 张（断点续传自动跳过已存在的 499 张）

---

## 增补：image_id 全链路归一化（Task 11，2026-08-08）

### 需求变更

Task 10 后 `image_id` 仍取 SE 原始字符串，两集合（google 混合精度如 `E121.403_N25.1601` vs xian 全 4 位如 `E121.4030_N25.1601`）image_id 字符串不一致 → point_id 不一致 → 双集合对比仍缺失 125 张。用户确认**全链路归一化（需重导）**：image_id 用数值归一化串（D11a），保证两集合 point_id 一致，代价是需清空重导两个 collection。

### 增量实现

- `data_loader.py` 新增 `normalize_location_key(raw_key)`：解析 `(lon, lat)` round 到 4 位小数、去尾随零（`E121.4033_N25.1370`→`E121.4033_N25.137`、`E121.4030_N25.1601`→`E121.403_N25.1601`；全零保留 `.0` 保证坐标段合法）
- `scan_directory` 的 `ImagePair.image_id` 改用归一化串（配对 key 仍数值 `(lon, lat)`）
- 下游影响：`point_id = uuid5(uuid5(DNS, image_id), "row_col")` 随 image_id 变化 → **需清空重导 `google_aef_embedding` / `xian_aef_embedding`**（manifest / 采样地图 / corpus 缓存按可重建语义自动重建）

### 增量验证

- **全新全量回归：`405 passed, 1 skipped, 23 warnings in 483.30s`（exit 0，0 失败）**（Task 11 新增 6 测试）
- **归一化格式验证**：真实数据 624/624 无冲突；`parse(normalize(x)) == parse(x)` 0 失配（UTM 推算不受影响）；幂等成立
- **point_id 双集合一致性冒烟**：归一化后两集合 point_id 完全一致（10,223,616 全交集、0 独有）→ 双集合对比 0 剔除，125 张缺失根治
- delta spec「坐标段数值匹配配对」Requirement 更新为 image_id 用归一化串；每任务审查 + 最终轻量审查 Ready to merge（0 CRIT/IMP，5 Minor 接受）
- **重导指引（升级必需）**：清空 `google_aef_embedding` 后用 `data_google`、清空 `xian_aef_embedding` 后用 `data_xian` 重新导入（WebUI 分页默认目录已配置）；断点续传判定基于 image_id 归一化后一致

---

## 增补：可视化探索按检索 collection 定位影像文件（Task 12，2026-08-08）

### 需求变更

用户报告「可视化探索的各通道图像有问题，GOOGLE 提取的向量可视化背景图与 XIAN 一样」。根因（实证）：Task 11 image_id 归一化后，两 collection 的 image_id 完全重叠（624/624），`state["se_paths_map"]`（全局单例，只在「浏览」按钮时更新）的 key 冲突——GOOGLE 检索可视化会取到最后浏览的 XIAN 目录的 SE 文件。

### 增量实现

- `do_search` 记录 `state["search_collection"]`（检索 collection）
- `_show_visualization` 按 collection 解析数据目录（`COLLECTION_DATA_DIRS.get` 回退 `_CLI_DATA_DIR`，相对路径 `_PROJECT_ROOT / mapped`）→ `scan_directory` 构建闭包局部 `viz_se_map`；`_refresh_viz`/`on_mouse` 改用 `viz_se_map`（替代全局 `se_paths_map`）
- I-1 修复（612c1f0）：`do_visualize` 捕获检索时 collection 参数化传入 `_show_visualization`，跨分页交错（GOOGLE 结果对话框存活期间切到 XIAN）不再串集

### 增量验证

- **全新全量回归：`412 passed, 1 skipped, 23 warnings in 519.84s`（exit 0，0 失败）**（Task 12 新增 7 测试，含 I-1 跨分页交错用例）
- 每任务审查（Task 12）：spec ✅ / quality Approved（0 CRIT/IMP，4 Minor 接受）；final review 发现 IMPORTANT I-1（跨分页交错串集）→ fix 612c1f0 → re-review Approved（6 Minor 维持接受）
- **验证**：google 检索可视化用 data_google 文件、xian 检索用 data_xian 文件（跨分页交错不串集）、自定义 collection 回退默认目录

---

## 增补：payload 索引自动补齐（Task 13，2026-08-08）

### 需求变更

用户报告「GOOGLE 检索后切 XIAN 页检索 timed out（K1000 + UTM 过滤）；重启后先 XIAN 检索直接报错」。根因（实证）：`xian_aef_embedding` collection 缺 `utm_easting`/`utm_northing` 索引（历史创建/重建时未建），UTM 过滤全量扫描 1023 万点 → 5.03s > `QDRANT_TIMEOUT=5s` → timed out；google 有索引 139ms 正常。补建索引后 xian 检索恢复 274ms。

### 增量实现

- `qdrant_client.py` 新增幂等 `ensure_payload_indices()`：读 `payload_schema`，对缺失的 5 字段逐个 create，已有跳过；`create_payload_indices` 零改动
- `webui.py` `_apply_collection` 与 `init_page` 对当前 collection 调用（`to_thread` + try/except 降级，不阻塞页面）
- 测试隔离修复（52802c1）：8 个触达 ensure 的用例全 patch QdrantManager，FakeManager 补方法

### 增量验证

- **全新全量回归：`419 passed, 1 skipped, 23 warnings in 580.53s`（exit 0，0 失败）**
- **真实检索复现验证**：补建索引后 xian 检索 5.03s（超时）→ **274ms**，与 google（235ms）一致
- 每任务审查 + 最终轻量审查 Ready to merge（0 CRIT/IMP；I-1 测试隔离已修复复审通过；6 Minor 接受）
- delta spec 新增「payload 索引自动补齐」Requirement（2 场景）
