# 验证报告：linearprobe-collection-switch

日期：2026-08-08
Change：`linearprobe-collection-switch`（tweak，verify_mode=full，delta spec MODIFIED × 2）
验证命令：`pytest LinearProbe_evaluation/tests KNN_evaluation/tests -q -k 'not integration'`
→ **415 passed, 1 skipped, 3 deselected**（LinearProbe 79 + KNN 336）

## 检查项

| # | 检查项 | 结果 |
|---|--------|------|
| 1 | tasks.md 全部任务已完成 | PASS（4/4 `[x]`） |
| 2 | 实现符合 design.md | PASS（切换按钮 + 缩写映射 collection_short_name） |
| 3 | 实现符合 Design Doc | N/A（tweak 不生成 Design Doc） |
| 4 | delta spec 场景全部通过 | PASS（见下方场景核验） |
| 5 | proposal.md 目标满足 | PASS（切换按钮生效 + 导出缩写） |
| 6 | delta spec 与 design doc 无矛盾 | PASS |
| 7 | 关联设计文档可定位 | N/A |

## Delta spec 场景核验

### linear-probe-collection-selector「WebUI collection 选择与切换」

- **修改下拉未点击切换不生效**：`test_switch_collection_button_applies_selection` 断言
  修改下拉后 `_current_collection` 不变，点击「切换 collection」后才切换并重建 manager。
- **切换后采样与训练使用新 collection**：`do_train` 使用 `state["manager"]`（切换后重建），
  既有 `test_apply_collection_rebuilds_manager` 覆盖。
- **自定义 collection / localStorage 记忆**：既有场景未改动，测试继续通过。

### linear-probe-export「LinearProbe 训练结果导出」

- **导出 JSON 落盘（文件名缩写）**：`test_export_json_writes_to_lp_dir` 断言文件名含
  `xian` 且不含 `xian_aef_embedding`；`test_export_uses_google_short_after_switch` 断言
  切换 google 后含 `google`。
- **导出图片图表落盘（文件名缩写）**：`test_export_images_writes_pngs` 断言含 `xian`/`_mlp_`
  且不含全名。
- **未训练提示 / 按钮显示**：既有场景未改动，测试继续通过。
- **collection_short_name 单元断言**：`test_collection_short_name`（google/xian 缩写，
  自定义原名）。

## 测试证据

- `pytest LinearProbe_evaluation/tests KNN_evaluation/tests -q -k 'not integration'` → 415 passed。
- 安全：新增代码无硬编码凭据/密钥；缩写映射为静态字典，无外部输入注入风险。

## 已知限制

- 自定义 collection 导出文件名用原名（用户确认决策），若自定义名很长文件名仍较长；
  `_add_custom_collection` 已校验禁止 `/` 与 `\`，原名安全。
- 「切换 collection」按钮仅作用于当前页面会话；页面刷新后从 localStorage 恢复上次
  切换的 collection（既有机制）。

## 结论

实现满足 proposal 目标与两个 delta spec 全部场景，测试全量通过，验证通过。
