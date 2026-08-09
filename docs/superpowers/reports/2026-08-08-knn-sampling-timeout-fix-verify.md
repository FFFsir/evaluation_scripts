# 验证报告：knn-sampling-timeout-fix

日期：2026-08-08
Change：`knn-sampling-timeout-fix`（tweak，verify_mode=full，delta spec MODIFIED 2 个 Requirement + 新增 1 个场景）
验证命令：`.venv/Scripts/python.exe -m pytest KNN_evaluation/tests/test_config.py KNN_evaluation/tests/test_metrics.py KNN_evaluation/tests/test_webui.py KNN_evaluation/tests/test_qdrant_client.py -k 'not integration'`
→ **127 passed, 2 deselected**（集成测试按惯例排除）

## 检查项

| # | 检查项 | 结果 |
|---|--------|------|
| 1 | tasks.md 全部任务已完成 | PASS（12/12 `[x]`） |
| 2 | 实现符合 design.md | PASS（`QDRANT_TIMEOUT=60`、`warn_callback` 可选参数、索引未就绪检查、状态栏索引进度） |
| 3 | 实现符合 Design Doc | N/A（tweak 不生成 Design Doc） |
| 4 | delta spec 场景全部通过 | PASS（见下方场景核验） |
| 5 | proposal.md 目标满足 | PASS（调大超时 + 索引未就绪预警 + 状态栏索引进度） |
| 6 | delta spec 与 design doc 无矛盾 | PASS（design.md 决策与 delta spec 描述一致，无漂移） |
| 7 | 关联设计文档可定位 | N/A（tweak 无关联 Design Doc，使用 change 内 design.md） |

## Delta spec 场景核验（MODIFIED embedding-evaluation）

**「分层随机采样」新增场景：索引未就绪预警**
- `metrics.py` `sample_queries_by_label`：`collection_info()` 后检查
  `indexed_vectors_count < total_points`，未就绪时 `warn_callback` 输出
  「向量索引构建中（已索引 X / 总点数 Y）」。
- 测试：`test_metrics.py::TestSampleQueriesIndexWarning` —
  `test_warns_when_index_incomplete`（indexed=100 < total=200 触发警告，含比例）、
  `test_no_warn_when_index_ready`（indexed==total 不触发）、
  `test_no_warn_when_total_zero`（total=0 先抛「Collection 为空」）。

**「WebUI 评估面板」新增场景：状态栏显示索引进度**
- `webui.py` `refresh_status`：`info_label` 追加 `已索引向量: X / Y`，
  `indexed < total` 时 `text-warning` 样式 + 「⚠️ 向量索引构建中」。
- 测试：`test_webui.py::TestStatusIndexProgress` —
  `test_status_shows_indexed_vectors`（就绪时显示进度、positive 样式）、
  `test_status_warns_when_index_incomplete`（未就绪时 warning 样式 + 比例）。

**「WebUI 评估面板」新增场景：一键构建向量索引**
- `qdrant_client.py` `QdrantManager.reindex_vectors()`：调
  `client.update_collection(optimizer_config=OptimizersConfigDiff(indexing_threshold=0))`
  触发全量 HNSW 重建（与 importer reindex 一致，仅重建向量索引）。
- `webui.py` 状态区「构建向量索引」按钮（collection 存在时可见），点击触发
  `mgr.reindex_vectors()`，提示「向量索引重建已触发」并刷新状态显示进度。
- 测试：`test_qdrant_client.py::TestReindexVectors`（update_collection 收到
  `indexing_threshold=0`）；`test_webui.py::TestReindexVectorsButton`
  （按钮触发 reindex_vectors / collection 不存在时提示）。

**保留场景回归**：正常采样（`TestSampleQueriesUsingMap`）、Collection 为空/不存在
（`TestSampleQueriesByLabel`）、WebUI 评估流程（`TestPageIsolation`、
`TestConfusionMatrixImage`）全部通过。

## 实现证据

- `config.py:26`：`QDRANT_TIMEOUT = 60`（原 5）。
- `metrics.py:66-152` `sample_queries_by_label`：
  - 新增可选参数 `warn_callback: Callable[[str], None] | None`（缺省 None 用 print）；
  - `collection_info()` 后检查 `indexed = vectors_count`、`total = total_points`，
    `total > 0 and indexed < total` 时经 `warn_callback` 提示索引构建中及比例。
- `webui.py` `refresh_status`：`info_label` 显示 `已索引向量: X / Y`，
  未就绪时 `replace="text-sm text-warning"` + 「⚠️ 向量索引构建中」。
- `webui.py` `do_evaluate`：采样调用传
  `warn_callback=lambda msg: eval_progress_label.set_text(msg)`。

## 测试证据

- `.venv/Scripts/python.exe -m pytest ... -k 'not integration'` → 127 passed, 2 deselected。
- 新增 11 项：metrics 索引警告 3 + config timeout 1 + webui 状态栏 2 + 回归修复 2 + reindex 3。
- OpenSpec `validate --changes knn-sampling-timeout-fix` → VALID。
- 安全：新增代码无硬编码凭据/密钥；仅调大客户端超时、增加提示与一键重建入口，
  不改变数据流。

## 已知限制

- `QDRANT_TIMEOUT=60` 是客户端兜底；若 Qdrant 服务端索引长时间不推进且读操作超过
  60s，仍会超时，但采样前会先提示索引构建中，用户有明确预期。
- 索引未就绪提示为 WARNING（不阻断采样）；`indexed_vectors_count` 计数可能有延迟，
  误报无实质影响。
- 「构建向量索引」触发的是后台异步重建，完成时间取决于数据量与 Qdrant 资源；
  用户通过「刷新状态」查看 `已索引向量 / 总点数` 进度。

## 结论

实现满足 proposal 目标与 delta spec 两个新增/更新场景，针对性测试全量通过（集成测试
按惯例排除），OpenSpec 校验通过，验证通过。
