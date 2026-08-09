# 验证报告：linearprobe-export-images

日期：2026-08-08
Change：`linearprobe-export-images`（tweak，verify_mode=full，delta spec 新增 capability）
验证命令：`pytest LinearProbe_evaluation/tests KNN_evaluation/tests -q -k 'not integration'`
→ **406 passed, 1 skipped, 3 deselected**（LinearProbe 75 + KNN 331）

## 检查项

| # | 检查项 | 结果 |
|---|--------|------|
| 1 | tasks.md 全部任务已完成 | PASS（6/6 `[x]`） |
| 2 | 实现符合 design.md | PASS（LP_EXPORT_DIR、confusion_cm 保存、`_export_lp_results`、按钮训练后显示） |
| 3 | 实现符合 Design Doc | N/A（tweak 不生成 Design Doc） |
| 4 | delta spec 场景全部通过 | PASS（见下方场景核验） |
| 5 | proposal.md 目标满足 | PASS（导出图像图表 + JSON 落盘 outputs/evaluation/linearprobe） |
| 6 | delta spec 与 design doc 无矛盾 | PASS（tweak 无 design doc，delta spec 为新增 capability） |
| 7 | 关联设计文档可定位 | N/A（tweak 无关联设计文档） |

## Delta spec 场景核验（新增 capability linear-probe-export）

- **训练完成后显示导出按钮**：`test_export_buttons_visible_after_train`（set_visibility(True)）；
  未训练隐藏 `test_export_buttons_hidden_before_train`。
- **导出 JSON 落盘**：`test_export_json_writes_to_lp_dir`（`mlp_label_*.json` 含模型结构/history/验证报告，无浏览器下载）。
- **导出图片图表落盘**：`test_export_images_writes_pngs`（training_curves_*.png + confusion_matrix_*.png，PNG 魔数校验）。
- **未训练时点击导出**：`test_export_without_train_is_noop`（提示「请先完成训练」，不生成文件）。
- 结构变体：`test_export_linear_variant_architecture`（linear 单层结构 585 参数）。

## 测试证据

- `pytest LinearProbe_evaluation/tests KNN_evaluation/tests -q -k 'not integration'` → 406 passed
  （含 LinearProbe 新增/重写 6 条导出用例）。
- 安全：新增代码无硬编码凭据/密钥；导出内容仅训练结果与图表 PNG。

## 已知限制

- 图像导出中混淆矩阵 PNG 依赖训练时 `_compute_confusion_uri` 成功（`confusion_cm` 非 None）；
  若推理失败，仅导出训练曲线 PNG（JSON 不受影响）。
- 导出按钮状态在页面生命周期内保留；刷新页面后需重新训练。

## 结论

实现满足 proposal 目标与 delta spec 全部场景，测试全量通过，验证通过。
