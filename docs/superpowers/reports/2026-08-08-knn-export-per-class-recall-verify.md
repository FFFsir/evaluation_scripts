# 验证报告：knn-export-per-class-recall

日期：2026-08-08
Change：`knn-export-per-class-recall`（tweak，verify_mode=full，delta spec MODIFIED 2 个 Requirement）
验证命令：`.venv/Scripts/python.exe -m pytest KNN_evaluation/tests/test_visualization.py KNN_evaluation/tests/test_webui.py::TestExportPageResults`
→ **8 passed**（test_visualization 3 + TestExportPageResults 5）

## 检查项

| # | 检查项 | 结果 |
|---|--------|------|
| 1 | tasks.md 全部任务已完成 | PASS（6/6 `[x]`） |
| 2 | 实现符合 design.md | PASS（`f1_result` 可选参数、prbk 优先/ANN 回退、无 Global 线、标题/纵轴更新） |
| 3 | 实现符合 Design Doc | N/A（tweak 不生成 Design Doc） |
| 4 | delta spec 场景全部通过 | PASS（见下方场景核验） |
| 5 | proposal.md 目标满足 | PASS（导出右面板改为 Per-class Recall，与 Web 页面一致） |
| 6 | delta spec 与 design doc 无矛盾 | PASS（design.md 决策与 delta spec 描述一致，无漂移） |
| 7 | 关联设计文档可定位 | N/A（tweak 无关联 Design Doc，使用 change 内 design.md） |

## Delta spec 场景核验（MODIFIED embedding-evaluation）

- **CLI「图表生成」**：`cli.py:573` 调用
  `plot_purity_recall_curve(f2, purity_recall_curve.png, f1)`，`--plot` 生成混淆矩阵、
  Purity / Per-class Recall 曲线与距离分布直方图 PNG。
- **WebUI「图片图表导出」**：`webui.py:514` 调用
  `plot_purity_recall_curve(f2, pr_path, f1)`，生成
  `{集合缩写}_knn_cm_{时间戳}.png` 与 `{集合缩写}_knn_pr_{时间戳}.png`（右面板为
  Per-class Recall）。测试 `test_export_images_passes_f1_for_per_class_recall` 断言
  `plot_purity_recall_curve` 收到 `(f2, <路径>, f1)` 且 f1 含 `per_class_recall_by_k`。

## 实现证据

- `visualization.py:85-146` `plot_purity_recall_curve`：
  - 签名 `(purity_data, save_path, f1_result=None)`，向后兼容；
  - 右面板优先 `f1_result["per_class_recall_by_k"]`（K = `sorted(prbk)`，每类一条折线，
    与 Web 页面 `_show_eval_results` 逻辑一致），无该键回退 `purity_data["per_class_recall"]`；
  - 移除黑色 Global 线；标题 "Per-class Recall vs K"、y 轴 "Recall"；
    `fig.suptitle` "Embedding Quality: Purity & Per-class Recall Curves"；
  - 图例 `loc="lower left"`（用户反馈左上角遮挡数据，与左面板 Purity 一致）。
- `webui.py`：`_export_page_results` 图片分支传 `f1`；docstring 同步更新。
- `cli.py`：`--plot` 分支传 `f1`。
- 测试：`test_visualization.py`（分类器路径、ANN 回退、空 f1 回退均落盘 PNG）；
  `test_webui.py::TestExportPageResults` 新增 spy 断言。

## 测试证据

- `.venv/Scripts/python.exe -m pytest KNN_evaluation/tests/test_visualization.py KNN_evaluation/tests/test_webui.py::TestExportPageResults` → 8 passed。
- 全量 KNN 套件使用 venv 跑至 41% 无失败（约 250 项）；剩余慢速集成测试（GPU/torch/
  Qdrant）由用户人工验证。
- OpenSpec `validate --changes knn-export-per-class-recall` → VALID。
- 安全：新增代码无硬编码凭据/密钥；导出内容仅图表 PNG。

## 已知限制

- `per_class_recall_by_k` 仅在 exact（KNN 分类器多 K 混淆矩阵）路径存在；ANN 逐条模式
  回退 `f2.per_class_recall`（与 Web 页面行为一致）。
- 导出文件名与合并图结构不变（左 Purity + 右 Per-class Recall 单张 `_knn_pr_*.png`）。

## 结论

实现满足 proposal 目标与 delta spec 两个 MODIFIED 场景，针对性测试全量通过，OpenSpec
校验通过，验证通过。
