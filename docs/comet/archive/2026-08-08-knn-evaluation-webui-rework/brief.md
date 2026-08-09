# Outcome

KNN_evaluation WebUI 重构为固定 GOOGLE / XIAN 双页面 + 独立 SimilarityMatrix 页面，
五个需求全部落地：

1. 数据导入中途失败时自动从断点继续导入；
2. 默认固定 GOOGLE 与 XIAN 两个页面，去掉用户自定义添加/删除页面功能；
3. 原「相似度热力图对比」面板升级为独立页面 SimilarityMatrix；
4. GOOGLE 与 XIAN 两页面功能严格分离：Qdrant 连接 & Collection 状态、数据导入、
   向量检索、评估面板均在对应页面下展示，两页评估结果互不覆盖；
5. 评估面板混淆矩阵改为图片展示（同 LinearProbe_evaluation）；所有有「导出为 JSON」
   的模块同时提供「导出图片图表」：GOOGLE 页导出到 outputs/evaluation/google_aef、
   XIAN 页导出到 outputs/evaluation/xian_aef、SimilarityMatrix 页导出到
   outputs/evaluation/similarity，图片为 PNG 格式。

# Scope

- KNN_evaluation/webui.py：页面结构重构（GOOGLE/XIAN/SimilarityMatrix 三个页面、
  每页独立状态与内容）、移除自定义 collection 添加/删除 UI、导入自动续导、
  评估面板混淆矩阵图片化、新增「导出图片图表」。
- KNN_evaluation/importer.py：导入失败自动续导所需的故障处理（如逐影像失败隔离
  与重试策略，待澄清）。
- KNN_evaluation/visualization.py：新增混淆矩阵 base64 图（对齐 LinearProbe 模式）
  与图表 PNG 导出所需辅助函数。
- KNN_evaluation/tests/test_webui.py 及相关测试：随页面重构更新与新增用例。
- 页面/collection 映射沿用现有预置：GOOGLE ↔ google_aef_embedding（data_google），
  XIAN ↔ xian_aef_embedding（data_xian）。

# Non-goals

- 不修改 KNN_evaluation 的 CLI（cli.py）导入/评估/similarity-heatmap 行为。
- 不修改 LinearProbe_evaluation（其「导出 JSON」保持现状）。
- 不修改 Qdrant 数据结构、payload schema、导入点构造逻辑（build_points）。
- 不改变相似度热力图对比的计算语义（对比对象仍为 google × xian 预置对）。
- 不引入新的页面级数据库/持久化方案；页面状态保留在进程内（模块级）。

# Acceptance examples

- 用户在 GOOGLE 页评估面板完成评估并显示结果，切到 XIAN 页完成评估后，两页结果
  同时各自保留、互不覆盖；GOOGLE 页结果仍显示 GOOGLE 页内容。
- 页面只有 GOOGLE、XIAN、SimilarityMatrix 三个 tab，无「自定义 collection 名称」
  输入与「删除」按钮。
- 导入中途某张影像失败后，导入不会整个中止：能自动从断点（已导入影像跳过、
  失败影像重试）继续，最终汇总报告。
- 评估面板混淆矩阵以图片（PNG）展示，而非表格。
- 评估面板点击「导出图片图表」后在 outputs/evaluation/google_aef（GOOGLE 页）或
  outputs/evaluation/xian_aef（XIAN 页）生成 PNG 文件；SimilarityMatrix 页导出到
  outputs/evaluation/similarity。

# Constraints and invariants

- 页面状态（manager、评估结果、查询向量、导入进度等）按页面隔离，切页不丢失、
  不串扰（对应需求 4）。
- 评估结果存储必须保证 GOOGLE/XIAN 两页独立，满足「两页评估结果不能互相覆盖」。
- 「导出图片图表」写入的目录不存在时自动创建；文件名与格式为 PNG。
- 现有模块级可测试模式（_init_hooks、模块级状态字典）保持，便于 pytest 测试。
- 不写死 token/凭据/连接串之外的敏感信息；导出文件名不含页面切换引入的串扰。

# Decisions

- D1（Q1，用户已确认）: 导入失败自动续导采用「仅整体自动重试」：连接/服务端等
  持久性失败导致 import_directory 抛异常时，WebUI 自动重试整个导入流程（最多 3
  次、指数退避），利用现有按影像断点续传从失败处继续；重试耗尽后向用户报告失败。
  单张影像持久性失败不单独跳过，仍走整体中止→自动重试路径（需求 1）。

- D2（Q2，用户已确认）: 「导出图片图表」把 PNG 写入对应目录（自动创建）：
  GOOGLE 页 → outputs/evaluation/google_aef、XIAN 页 → outputs/evaluation/xian_aef、
  SimilarityMatrix 页 → outputs/evaluation/similarity；「导出为 JSON」同样改为写入
  同一页面目录（与 PNG 同目录落盘），不再走浏览器下载（需求 5）。

- D3（Q3，用户已确认）: SimilarityMatrix 页保留双采样模式（数据库全库 / 单张图片，
  含 N、Seed 参数与影像下拉），仅把导出目录固定为 outputs/evaluation/similarity
  并移除可编辑导出目录输入框（需求 3、5）。

# Open questions

（无未决阻塞项；共享理解已于 2026-08-08 经用户确认）

# Verification expectations

- pytest 覆盖：KNN_evaluation/tests 全量通过（含重写后的 test_webui.py）。
- 手动/自动化验证：页面结构（三个固定页面）、页面隔离（两页评估结果独立）、
  导入自动续导（失败注入后从断点继续）、混淆矩阵图片展示、三处导出目录的
  PNG 落盘。
