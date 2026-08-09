# import-manifest Specification

## Purpose
导入清单（manifest）将"已导入影像清单"从 Qdrant 推导型状态改为本地声明型状态：导入时增量更新、启动时毫秒级读取，Qdrant 离线可用，并通过对账防止与数据库漂移。
## Requirements
### Requirement: 导入 manifest 文件读写
系统 SHALL 为每个 collection 维护独立的导入清单文件（本地 JSON，文件名包含 collection 名称，如 `qdrant_import_manifest_<collection>.json`），记录已导入的 image_id 及每张影像的已导入像素数，支持读取、增量更新与持久化；对某 collection 的读写只作用于该 collection 对应的清单文件。

#### Scenario: 读取 manifest 得到影像列表
- **WHEN** 调用 manifest 读取接口（针对某 collection）且对应文件存在
- **THEN** 返回该 collection 已导入的 image_id 列表及每张的已导入像素数（如 `{"image_id": 16384}`）

#### Scenario: manifest 文件缺失
- **WHEN** 某 collection 的 manifest 文件不存在
- **THEN** 读取接口返回空清单（不报错），由对账/重建路径从数据库重建

#### Scenario: manifest 内容结构
- **WHEN** 写入 manifest
- **THEN** 内容包含 collection 名称、image_id→已导入像素数映射和更新时间戳

#### Scenario: 不同 collection 使用独立清单
- **WHEN** 系统分别对 `google_aef_embedding` 与 `xian_aef_embedding` 执行导入
- **THEN** 两个 collection 的清单写入各自独立的文件，互不覆盖

#### Scenario: 切换后读取对应清单
- **WHEN** 当前 collection 为 `xian_aef_embedding`
- **THEN** manifest 读取与更新只作用于 `xian_aef_embedding` 对应的清单文件

### Requirement: 导入时增量更新 manifest
系统 SHALL 在 `import_directory` 每张影像导入/跳过完成后同步更新 manifest，保证 manifest 与数据库同步咽喉点唯一、不遗漏任何导入入口。

#### Scenario: 导入成功后更新
- **WHEN** 某张影像完整导入 16384 像素
- **THEN** manifest 中该 image_id 的已导入像素数更新为 16384

#### Scenario: 部分导入记录实际值
- **WHEN** 某张影像导入中断，仅导入 N 像素（0 < N < 16384）
- **THEN** manifest 中该 image_id 记录实际值 N，用于预览状态列显示"⏳ N/16384"

#### Scenario: CLI 与 WebUI 导入均更新
- **WHEN** 通过 CLI `import` 或 WebUI"导入全部"任一路径导入
- **THEN** manifest 均被更新（两者共用同一导入入口）

### Requirement: facet 对账
系统 SHALL 在 Web 启动后台用 Qdrant `facet(key="image_id")` 对账一次：与 manifest 不一致时刷新 manifest；manifest 缺失或 Collection 重建时用 facet 重建。对账是单次去重查询，不阻塞启动。（注：qdrant-client 1.18.0 无 `distinct` 方法，使用 `facet` 获取字段去重值。）

#### Scenario: manifest 与数据库一致
- **WHEN** facet 返回的 image_id 集合与 manifest 一致
- **THEN** 不修改 manifest

#### Scenario: manifest 与数据库不一致
- **WHEN** facet 返回的 image_id 集合与 manifest 不一致（如外部改动 Collection）
- **THEN** 用 facet 结果刷新 manifest

#### Scenario: manifest 缺失时重建
- **WHEN** manifest 文件不存在但 Collection 存在且非空
- **THEN** 用 facet 结果重建 manifest，Web 仍能列出影像

#### Scenario: Collection 重建后对账
- **WHEN** Collection 被删除重建且为空
- **THEN** facet 返回空集，manifest 被清空或标记为空

### Requirement: 会话缓存与多 tab 复用
系统 SHALL 在同一会话内缓存 manifest 读取结果，同一会话的多 tab/多页面刷新不重复读文件。

#### Scenario: 同会话多 tab 复用
- **WHEN** 同一会话内打开多个 tab 或刷新页面
- **THEN** manifest 读取结果复用会话缓存，不重复读文件

