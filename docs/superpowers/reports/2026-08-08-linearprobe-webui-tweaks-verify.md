# 验证报告：linearprobe-webui-tweaks

日期：2026-08-08
Change：`linearprobe-webui-tweaks`（tweak，verify_mode=full，delta spec MODIFIED）
验证命令：`pytest LinearProbe_evaluation/tests KNN_evaluation/tests -q -k 'not integration'`
→ **411 passed, 1 skipped, 3 deselected**（LinearProbe 75 + KNN 336）

## 检查项

| # | 检查项 | 结果 |
|---|--------|------|
| 1 | tasks.md 全部任务已完成 | PASS（4/4 `[x]`） |
| 2 | 实现符合 design.md | PASS（列样式 align/宽度、文件名前缀 `{collection}_{variant}_{ts}`） |
| 3 | 实现符合 Design Doc | N/A（tweak 不生成 Design Doc） |
| 4 | delta spec 场景全部通过 | PASS（见下方场景核验） |
| 5 | proposal.md 目标满足 | PASS（表格靠左合理列宽 + 导出文件名含 collection/variant） |
| 6 | delta spec 与 design doc 无矛盾 | PASS |
| 7 | 关联设计文档可定位 | N/A |

## Delta spec 场景核验（MODIFIED linear-probe-export）

- **导出 JSON 落盘（文件名含 collection 与 variant）**：`test_export_json_writes_to_lp_dir`
  断言文件名含 `xian_aef_embedding` 与 `_mlp_`；`test_export_linear_variant_architecture`
  断言含 `_linear_`。
- **导出图片图表落盘（文件名含 collection 与 variant）**：`test_export_images_writes_pngs`
  断言 training_curves/confusion_matrix PNG 文件名均含 `xian_aef_embedding` 与 `_mlp_`。
- **训练完成后显示按钮 / 未训练提示**：既有场景，未改动（测试继续通过）。

## 表格样式核验（proposal 目标 1）

- `test_successful_train_shows_results` 新增断言：per-class 表所有列 `align: left`；
  类别列 `min-width: 200px` + `word-break: break-word`；数值列 `min-width: 110px`。

## 测试证据

- `pytest LinearProbe_evaluation/tests KNN_evaluation/tests -q -k 'not integration'` → 411 passed。
- 安全：新增代码无硬编码凭据/密钥；文件名仅含 collection 名与 variant（均为安全字符）。

## 已知限制

- 导出文件名中的 collection 为用户自定义名称时未做路径字符清洗（沿用现状：自定义
  collection 名称已由 `_add_custom_collection` 校验禁止 `/` 与 `\`）；预置名称均安全。

## 结论

实现满足 proposal 目标与 delta spec 全部场景，测试全量通过，验证通过。
