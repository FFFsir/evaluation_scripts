# collection-selector Specification

## Purpose
允许 KNN 评估系统在多个 Qdrant Collection（embedding 版本）之间选择与切换，并支持自定义 collection 名称，使 CLI 与 WebUI 所有入口都能针对指定 collection 工作。
## Requirements
### Requirement: 预置 Collection 列表与默认值
系统 SHALL 预置两个可选的 Qdrant Collection 名称：`google_aef_embedding` 与 `xian_aef_embedding`，并 SHALL 在未显式指定 collection 时默认使用 `google_aef_embedding`。

#### Scenario: 未指定 collection 时使用默认值
- **WHEN** 用户在 CLI 或 WebUI 未指定 collection
- **THEN** 系统使用默认 collection `google_aef_embedding`

#### Scenario: 预置列表包含两个 collection
- **WHEN** 用户查看 collection 选择入口（CLI 帮助或 WebUI 分页）
- **THEN** 系统提供 `google_aef_embedding` 与 `xian_aef_embedding` 两个预置选项

#### Scenario: WebUI 首次加载无记忆时使用默认值
- **WHEN** WebUI 首次加载且 localStorage 无上次选择记录
- **THEN** 默认选中 `google_aef_embedding`

#### Scenario: WebUI 恢复上次选择
- **WHEN** 用户上次选择 `xian_aef_embedding`，再次加载 WebUI
- **THEN** 系统从 localStorage 恢复上次选择的 collection `xian_aef_embedding`

#### Scenario: localStorage 记录失效时回退默认
- **WHEN** localStorage 记录的 collection 名称已不存在（如被删除或记录损坏）
- **THEN** 系统回退到默认 collection `google_aef_embedding`

### Requirement: CLI collection 选择
系统 SHALL 为 CLI 各子命令（`import` / `search` / `stats` / `evaluate` / `migrate`）提供 `--collection <名称>` 参数，用于覆盖默认 collection；未指定时使用默认值。

#### Scenario: CLI 指定 collection 执行
- **WHEN** 用户运行 `python -m KNN_evaluation.cli stats --collection xian_aef_embedding`
- **THEN** 系统对 `xian_aef_embedding` Collection 执行统计并输出其 collection 名称

#### Scenario: CLI 未指定 collection
- **WHEN** 用户运行 CLI 子命令且未提供 `--collection`
- **THEN** 系统使用默认 collection `google_aef_embedding`

#### Scenario: CLI 指定不存在的 collection
- **WHEN** 用户通过 CLI 指定一个不存在的 collection 执行需要该 collection 存在的命令（如 `search` / `stats` / `evaluate`）
- **THEN** 系统输出明确错误信息（含 collection 名称）并以非零退出码结束

### Requirement: WebUI collection 分页选择与切换
系统 SHALL 在 WebUI 以分页（tab）形式展示 collection：预置 `google_aef_embedding` / `xian_aef_embedding` 两个分页，并提供自定义 collection 添加入口；用户点击分页切换当前 collection，切换后当前会话的 manager、清单缓存、采样地图与评估等所有操作 SHALL 针对新选中的 collection 进行。

#### Scenario: WebUI 选择预置 collection 分页
- **WHEN** 用户在 WebUI 点击 `xian_aef_embedding` 分页
- **THEN** 界面状态（Collection 名称、统计信息、数据导入/检索/评估区域）切换到 `xian_aef_embedding`

#### Scenario: WebUI 自定义 collection 名称
- **WHEN** 用户在 WebUI 自定义输入一个 collection 名称（如 `my_embedding`）并确认
- **THEN** 系统创建该 collection 的分页并将当前 collection 切换为 `my_embedding`，后续操作针对该 collection

#### Scenario: 切换后导入与检索使用新 collection
- **WHEN** 用户切换到 `xian_aef_embedding` 后执行数据导入或向量检索
- **THEN** 导入/检索请求发往 `xian_aef_embedding`，而非其他 collection

### Requirement: collection 分页缓存管理
系统 SHALL 在每个 collection 分页内提供「刷新」与「清理缓存」操作：「刷新」重新对账并加载该 collection 的最新状态与数据；「清理缓存」删除该 collection 的采样地图与导入 manifest 缓存文件，且只作用于当前分页的 collection，不影响其他分页的缓存。

#### Scenario: 分页内刷新
- **WHEN** 用户在当前 collection 分页点击「刷新」
- **THEN** 系统重新对账该 collection 的 manifest/采样地图并刷新界面数据

#### Scenario: 分页内清理缓存
- **WHEN** 用户在当前 collection 分页点击「清理缓存」
- **THEN** 系统删除该 collection 的采样地图与 manifest 缓存文件，其他 collection 分页的缓存保持不变

#### Scenario: 切换分页不自动清理缓存
- **WHEN** 用户从一个 collection 分页切换到另一个分页
- **THEN** 系统不自动清理任何分页的缓存（是否清理由用户在各分页内自主决定）

### Requirement: 采样地图按 collection 隔离
系统 SHALL 为不同 collection 维护独立的采样地图文件，文件名包含 collection 名称（如 `qdrant_sampling_map_<collection>.json`），使切换 collection 后采样地图不相互覆盖，并 SHALL 按当前 collection 对账与重建对应地图。

#### Scenario: 不同 collection 使用独立地图
- **WHEN** 系统分别对 `google_aef_embedding` 与 `xian_aef_embedding` 构建采样地图
- **THEN** 两个 collection 的地图写入各自独立的文件，互不覆盖

#### Scenario: 切换后使用对应地图
- **WHEN** 用户切换到 `xian_aef_embedding` 后执行评估采样
- **THEN** 系统读取/对账 `xian_aef_embedding` 对应的采样地图文件

### Requirement: 删除自定义 collection 分页
系统 SHALL 允许用户删除通过自定义添加创建的 collection 分页：分页提供删除入口，点击后弹出确认提示框明确提示「将同步删除该 Qdrant Collection 及其全部数据（不可恢复）」，用户确认后系统删除该 collection 对应的 Qdrant Collection、清理该 collection 的本地缓存文件（采样地图与导入 manifest）并移除该分页；若当前分页即被删除的分页，系统 SHALL 切换回默认 collection 分页。预置的两个 collection 分页（`google_aef_embedding` / `xian_aef_embedding`）SHALL 始终存在，不可删除。

#### Scenario: 删除自定义 collection 分页
- **WHEN** 用户对自定义添加的 collection 分页（如 `my_embedding`）点击删除并在确认提示框确认
- **THEN** 系统删除 Qdrant 中的 `my_embedding` Collection、删除其本地采样地图与 manifest 缓存文件，并从分页中移除 `my_embedding`

#### Scenario: 删除前确认提示
- **WHEN** 用户点击自定义 collection 分页的删除按钮
- **THEN** 系统弹出确认提示框，明确提示将同步删除该 collection 及其全部数据（不可恢复），用户确认前不执行任何删除

#### Scenario: 删除当前分页后切回默认
- **WHEN** 用户删除当前正在查看的自定义 collection 分页
- **THEN** 系统移除该分页并切换当前 collection 回默认 `google_aef_embedding`

#### Scenario: 预置分页不可删除
- **WHEN** 用户查看预置的 `google_aef_embedding` / `xian_aef_embedding` 分页
- **THEN** 这些分页不提供删除入口（始终存在）

#### Scenario: 删除失败保留分页
- **WHEN** Qdrant 删除该 collection 失败（如服务不可达、权限不足）
- **THEN** 系统不移除分页、不清理缓存，并向用户提示删除失败原因，用户可重试

