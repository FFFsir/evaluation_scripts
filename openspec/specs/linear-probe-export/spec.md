# linear-probe-export Specification

## Purpose
TBD - created by archiving change linearprobe-export-images. Update Purpose after archive.
## Requirements
### Requirement: LinearProbe 训练结果导出

系统 SHALL 在 LinearProbe_evaluation WebUI 训练面板提供结果导出能力：训练完成后
显示「导出 JSON」与「导出图片图表」两个按钮（未训练时隐藏）。「导出 JSON」把训练
结果（模型结构、历史曲线、验证报告等）写入 `outputs/evaluation/linearprobe` 目录
（自动创建，带时间戳 JSON 文件）；「导出图片图表」把训练曲线 PNG 与混淆矩阵 PNG
写入同一目录（带时间戳，PNG 格式）；两者共用同一时间戳前缀成组。导出文件名 SHALL
遵循统一格式 `集合缩写_模型结构_文件内容缩写_时间戳.xxx`（集合缩写：
`google_aef_embedding → google`、`xian_aef_embedding → xian`，自定义 collection 用
原名；模型结构缩写：`mlp` / `linear`；内容缩写：`result`=结果 JSON、
`curves`=训练曲线 PNG、`cm`=混淆矩阵 PNG。示例
`xian_mlp_result_20260808_120000.json`、
`google_linear_curves_20260808_120000.png`）。导出成功后向用户提示文件路径；
未完成训练时提示先完成训练。

#### Scenario: 训练完成后显示导出按钮
- **WHEN** 用户完成一次 MLP_label 训练并查看结果
- **THEN** 「导出 JSON」与「导出图片图表」按钮可见；未训练时两者隐藏

#### Scenario: 导出 JSON 落盘
- **WHEN** 用户点击「导出 JSON」且已有训练结果
- **THEN** 在 `outputs/evaluation/linearprobe` 生成文件名 `{集合缩写}_{模型结构}_result_{时间戳}.json`（含模型结构、history、验证报告），不再走浏览器下载

#### Scenario: 导出图片图表落盘
- **WHEN** 用户点击「导出图片图表」且已有训练结果
- **THEN** 在 `outputs/evaluation/linearprobe` 生成 `{集合缩写}_{模型结构}_curves_{时间戳}.png`（训练曲线）与 `{集合缩写}_{模型结构}_cm_{时间戳}.png`（混淆矩阵）

#### Scenario: 未训练时点击导出
- **WHEN** 用户尚未完成训练即点击导出按钮
- **THEN** 提示「请先完成训练」，不生成任何文件

