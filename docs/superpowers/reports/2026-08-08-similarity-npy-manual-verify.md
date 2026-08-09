# 验证报告：similarity-npy-manual

日期：2026-08-08
Change：`similarity-npy-manual`（tweak，verify_mode=full，delta spec MODIFIED × 2）
验证命令：`pytest KNN_evaluation/tests LinearProbe_evaluation/tests -q -k 'not integration'`
→ **430 passed, 1 skipped, 3 deselected**

## 检查项

| # | 检查项 | 结果 |
|---|--------|------|
| 1 | tasks.md 全部任务已完成 | PASS（6/6 `[x]`） |
| 2 | 实现符合 design.md | PASS（export_npy 参数、sim_g/sim_x 返回、npy 手动分支、按钮） |
| 3 | 实现符合 Design Doc | N/A（tweak 不生成 Design Doc） |
| 4 | delta spec 场景全部通过 | PASS（见下方场景核验） |
| 5 | proposal.md 目标满足 | PASS |
| 6 | delta spec 与 design doc 无矛盾 | PASS |
| 7 | 关联设计文档可定位 | N/A |

## Delta spec 场景核验（similarity-heatmap-compare）

### 「WebUI 对比面板」

- **面板导出按钮（含 npy）**：`test_export_buttons_hidden_before_compare` /
  `test_export_buttons_visible_after_compare` 断言「输出 npy 文件」按钮默认隐藏、
  对比后显示。
- **npy 带时间戳、不覆盖旧文件**：`test_export_npy_writes_timestamped_files` 断言
  `full_col_{google,xian}_similarity_{ts}.npy` × 2、内容 (2,2) 矩阵、重复导出（sleep
  1.1s 错开时间戳）数量翻倍不覆盖。

### 「导出相似度矩阵与采样信息」

- **export_npy 参数（默认 True，CLI 不变）**：`test_default_export_dir_is_outputs` 断言
  默认 prefix 空且 export_npy 默认 True；`test_export_dir_triggers_export_and_metadata`
  透传校验。
- **export_npy=False 不写 npy**：`test_do_sim_compare_exports_to_similarity_dir` 断言
  WebUI 传 `export_npy=False`（npy 改手动导出）；sampling json 仍落盘（带前缀）。
- **WebUI 手动 npy 导出**：`test_export_npy_writes_timestamped_files` 覆盖。

## 测试证据

- `pytest KNN_evaluation/tests LinearProbe_evaluation/tests -q -k 'not integration'` → 430 passed。
- 安全：新增代码无硬编码凭据/密钥；文件名由内部常量/预置映射构成，无注入风险。

## 已知限制

- npy 时间戳精度为秒，同一秒内重复导出会覆盖同名文件（属预期——时间戳命名约定
  同 JSON/PNG）；跨秒导出不覆盖。
- CLI 路径未传 export_npy（默认 True）行为不变；WebUI 对比时不再自动落盘 npy，
  由「输出 npy 文件」按钮手动导出。

## 结论

实现满足 proposal 目标与 delta spec 全部场景，测试全量通过，验证通过。
