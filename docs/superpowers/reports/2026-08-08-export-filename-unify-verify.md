# 验证报告：export-filename-unify

日期：2026-08-08
Change：`export-filename-unify`（tweak，verify_mode=full，delta spec MODIFIED × 3）
验证命令：`pytest KNN_evaluation/tests LinearProbe_evaluation/tests -q -k 'not integration'`
→ **416 passed, 1 skipped, 3 deselected**

## 检查项

| # | 检查项 | 结果 |
|---|--------|------|
| 1 | tasks.md 全部任务已完成 | PASS（6/6 `[x]`） |
| 2 | 实现符合 design.md | PASS（三处命名 + prefix 参数 + 路径统一） |
| 3 | 实现符合 Design Doc | N/A（tweak 不生成 Design Doc） |
| 4 | delta spec 场景全部通过 | PASS（见下方场景核验） |
| 5 | proposal.md 目标满足 | PASS |
| 6 | delta spec 与 design doc 无矛盾 | PASS |
| 7 | 关联设计文档可定位 | N/A |

## Delta spec 场景核验

### embedding-evaluation「WebUI 评估面板」

- **JSON 导出**：`test_export_json_writes_to_knn_eval_dir` 断言 `google_knn_result_*.json`
  写入 `outputs/evaluation/knn_eval`；`test_export_json_xian_uses_xian_knn_prefix` 断言
  `xian_knn_result_*.json`。
- **图片图表导出**：`test_export_images_writes_pngs` 断言 `google_knn_cm_*.png` /
  `google_knn_pr_*.png`。

### similarity-heatmap-compare「WebUI 对比面板」+「导出相似度矩阵与采样信息」

- **面板导出按钮（前缀）**：`test_do_sim_compare_exports_to_similarity_dir` 断言热力图
  `full_col_similarity_heatmap_*.png` 且 `prefix="full_col"` 传入 compare；
  `test_export_json_writes_sampling_json` / `test_export_images_writes_heatmap_png` 断言
  `full_col_` 前缀。
- **导出函数 prefix 参数**：`test_prefix_adds_filename_prefix`（`full_col_google_...npy`、
  `full_col_similarity_sampling.json`）；`test_default_export_dir_is_outputs` 断言默认
  prefix 为空（CLI 不变）。

### linear-probe-export「LinearProbe 训练结果导出」

- **导出 JSON**：`test_export_json_writes_to_lp_dir` 断言 `xian_mlp_result_*.json`；
  `test_export_uses_google_short_after_switch` 断言 `google_mlp_result_*.json`。
- **导出图片图表**：`test_export_images_writes_pngs` 断言 `xian_mlp_curves_*.png` 与
  `xian_mlp_cm_*.png`。

## 测试证据

- `pytest KNN_evaluation/tests LinearProbe_evaluation/tests -q -k 'not integration'` → 416 passed。
- 安全：新增代码无硬编码凭据/密钥；prefix/文件名由内部常量与预置映射构成，无注入风险。

## 已知限制

- SimilarityMatrix 导出 JSON 依赖 compare 落盘的 `{prefix}similarity_sampling.json`；
  若文件被删除，`pixels` 字段为空但仍生成。
- KNN GOOGLE/XIAN 导出路径由 `outputs/evaluation/{google_aef,xian_aef}` 迁移至
  `outputs/evaluation/knn_eval`，旧目录下历史文件不自动迁移。

## 结论

实现满足 proposal 目标与三个 delta spec 全部场景，测试全量通过，验证通过。
