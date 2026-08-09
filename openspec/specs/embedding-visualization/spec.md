# embedding-visualization Specification

## Purpose
通过 NiceGUI Web 界面交互式浏览和预览 Satellite Embedding V1 嵌入数据文件，提供文件导航、数据摘要展示和单通道灰度渲染功能。
## Requirements
### Requirement: Web 界面文件浏览

系统 SHALL 提供一个基于 NiceGUI 的 Web 界面，支持指定浏览目录并列出其中所有 `.npy` / `.npz` 文件。

#### Scenario: 浏览包含数据文件的目录

- **WHEN** 用户通过 Web 界面打开一个包含 `.npy` 和 `.npz` 文件的目录
- **THEN** 页面显示该目录下所有匹配文件的列表，包含文件名、文件大小和修改时间

#### Scenario: 目录为空

- **WHEN** 用户浏览一个不包含任何 `.npy` / `.npz` 文件的目录
- **THEN** 页面显示"当前目录暂无数据文件"提示

### Requirement: 嵌入数据摘要预览

系统 SHALL 在 Web 界面中点击数据文件后展示该文件的 shape、dtype、全局 min、全局 max 和各通道的 min/max 统计信息。

#### Scenario: 查看 .npy 文件摘要

- **WHEN** 用户在 Web 界面点击一个 `.npy` 文件
- **THEN** 页面弹窗或展开区域显示数据的 shape `(64, 128, 128)`、dtype `float64`、全局 min/max 和各通道 min/max

#### Scenario: 查看 .npz 文件摘要

- **WHEN** 用户在 Web 界面点击一个 `.npz` 文件
- **THEN** 页面弹窗或展开区域显示 `embedding` 键对应数据的完整摘要信息

### Requirement: 单通道灰度渲染

系统 SHALL 支持选择 64 个通道中的任一通道，将该通道的灰度值（值域 `[-1, 1]`）线性映射为 `[0, 255]` 灰度图并渲染为 PNG 预览。

#### Scenario: 选择通道并查看灰度图

- **WHEN** 用户在 Web 界面选择一个数据文件并从 64 通道下拉列表中选择一个通道（如 A00）
- **THEN** 页面显示该通道数据的灰度预览图

### Requirement: WebUI 命令行启动

系统 SHALL 支持通过命令行启动 Web 服务器，并提供默认的浏览目录。

#### Scenario: 启动 Web 服务器

- **WHEN** 用户运行 WebUI 启动命令
- **THEN** NiceGUI 服务器在 `http://127.0.0.1:8002` 启动，默认浏览目录为 `../download_scripts/output/SE/`

### Requirement: WebUI 启动流程异步化
系统 SHALL 使 WebUI 启动流程为「健康检查 → 立即更新状态 → 让出事件循环 → 后台线程执行贵操作」，所有对 Qdrant manager 的阻塞调用（scroll/count/distinct）均封装为异步线程调用，不阻塞 NiceGUI 事件循环。

#### Scenario: 启动秒级显示连接状态
- **WHEN** 打开 WebUI 且 Qdrant 可达
- **THEN** 页面在秒级内显示 "✅ Qdrant 连接正常"，不卡在 "⏳ 正在连接..."

#### Scenario: 贵操作后台执行
- **WHEN** 启动过程中需要扫描目录、读取清单或对账
- **THEN** 这些操作在后台线程执行，事件循环保持响应，页面可交互

#### Scenario: 翻页不阻塞事件循环
- **WHEN** 用户翻页浏览影像预览列表
- **THEN** 预览状态列的像素数检查在后台线程执行，翻页不卡顿

### Requirement: WebUI 影像列表读本地 manifest
系统 SHALL 使 WebUI 启动时直接从本地 manifest 读取已导入影像列表（毫秒级、Qdrant 离线可用），替代全库逐页 scroll 推导型状态；预览状态列的每张影像像素数也读 manifest（不再逐条 count）。

#### Scenario: 影像列表读 manifest
- **WHEN** WebUI 启动且 manifest 文件存在
- **THEN** 影像下拉列表直接从 manifest 填充已导入 image_id，不触发全库 scroll

#### Scenario: Qdrant 离线时仍可列出影像
- **WHEN** Qdrant 服务不可达但 manifest 文件存在
- **THEN** 影像列表仍从 manifest 列出（Qdrant 离线可用）

#### Scenario: 预览状态列读 manifest
- **WHEN** 渲染影像预览列表
- **THEN** 每张影像的已导入像素数读 manifest（如 "✅ 已导入" / "⏳ N/16384"），不逐条 count

### Requirement: 多 tab 不重复加载
系统 SHALL 使同一会话内的多 tab/多页面刷新复用会话缓存的 manifest 读取结果，不重复读文件、不重复全库查询。

#### Scenario: 多 tab 复用缓存
- **WHEN** 同一会话内打开多个 tab 或刷新页面
- **THEN** 影像列表复用会话缓存，不重复读 manifest 文件或查询 Qdrant

