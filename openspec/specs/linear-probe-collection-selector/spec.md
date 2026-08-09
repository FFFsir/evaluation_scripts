# linear-probe-collection-selector Specification

## Purpose
允许 LinearProbe 评估系统在多个 Qdrant Collection（embedding 版本）之间选择与切换，并支持自定义 collection 名称，使 CLI 与 WebUI 所有入口都能针对指定 collection 工作，默认使用 `xian_aef_embedding`。
## Requirements
### Requirement: 预置 Collection 列表与默认值
系统 SHALL 预置两个可选的 Qdrant Collection 名称：`google_aef_embedding` 与 `xian_aef_embedding`，并 SHALL 在未显式指定 collection 时默认使用 `xian_aef_embedding`。

#### Scenario: 未指定 collection 时使用默认值
- **WHEN** 用户在 CLI 或 WebUI 未指定 collection
- **THEN** 系统使用默认 collection `xian_aef_embedding`

#### Scenario: 预置列表包含两个 collection
- **WHEN** 用户查看 collection 选择入口（CLI 帮助或 WebUI 选择器）
- **THEN** 系统提供 `google_aef_embedding` 与 `xian_aef_embedding` 两个预置选项

#### Scenario: WebUI 首次加载无记忆时使用默认值
- **WHEN** WebUI 首次加载且 localStorage 无上次选择记录
- **THEN** 默认选中 `xian_aef_embedding`

#### Scenario: WebUI 恢复上次选择
- **WHEN** 用户上次选择 `google_aef_embedding`，再次加载 WebUI
- **THEN** 系统从 localStorage 恢复上次选择的 collection `google_aef_embedding`

#### Scenario: localStorage 记录失效时回退默认
- **WHEN** localStorage 记录的 collection 名称已不存在（如被删除或记录损坏）
- **THEN** 系统回退到默认 collection `xian_aef_embedding`

### Requirement: CLI collection 选择
系统 SHALL 为 CLI 各子命令（`train` / `evaluate` / `stats`）提供 `--collection <名称>` 参数，用于覆盖默认 collection；未指定时使用默认值。

#### Scenario: CLI 指定 collection 执行
- **WHEN** 用户运行 `python -m LinearProbe_evaluation.cli stats --collection xian_aef_embedding`
- **THEN** 系统对 `xian_aef_embedding` Collection 执行统计并输出其 collection 名称

#### Scenario: CLI 未指定 collection
- **WHEN** 用户运行 CLI 子命令且未提供 `--collection`
- **THEN** 系统使用默认 collection `xian_aef_embedding`

#### Scenario: CLI 指定不存在的 collection
- **WHEN** 用户通过 CLI 指定一个不存在的 collection 执行需要该 collection 存在的命令（如 `stats`）
- **THEN** 系统输出明确错误信息（含 collection 名称）并以非零退出码结束

### Requirement: WebUI collection 选择与切换

系统 SHALL 在 WebUI 提供 collection 选择：预置 `google_aef_embedding` / `xian_aef_embedding`
两个选项，并提供自定义 collection 名称输入；用户修改选择后 MUST 点击「切换 collection」
按钮才生效（下拉仅更新选中值，不即时切换），生效后当前会话的 manager 与全部数据操作
（状态查询、分层采样、训练）SHALL 针对新选中的 collection 进行。

#### Scenario: WebUI 选择预置 collection
- **WHEN** 用户在 WebUI 选择 `xian_aef_embedding` 并点击「切换 collection」
- **THEN** 界面 Collection 名称与状态信息切换到 `xian_aef_embedding`，后续训练针对该 collection

#### Scenario: 修改下拉未点击切换不生效
- **WHEN** 用户修改 collection 下拉为 `google_aef_embedding` 但未点击「切换 collection」
- **THEN** 当前 collection 仍为原值，采样与训练继续针对原 collection

#### Scenario: WebUI 自定义 collection 名称
- **WHEN** 用户在 WebUI 自定义输入一个 collection 名称（如 `my_embedding`）并确认
- **THEN** 系统将当前 collection 切换为 `my_embedding`，后续操作针对该 collection

#### Scenario: 切换后采样与训练使用新 collection
- **WHEN** 用户切换到 `xian_aef_embedding` 后执行分层采样与训练
- **THEN** 采样与训练请求发往 `xian_aef_embedding`，而非其他 collection

#### Scenario: WebUI 记住上次选择
- **WHEN** 用户切换 collection 后重新加载 WebUI
- **THEN** 系统恢复用户上次选择的 collection（localStorage 记忆）

