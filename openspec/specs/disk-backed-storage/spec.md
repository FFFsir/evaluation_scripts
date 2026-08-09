# disk-backed-storage Specification

## Purpose
将 Qdrant Collection 从"向量/payload 全量常驻 RAM"改为"磁盘化存储 + HNSW 图留 RAM"，降低 16GB 机器上的常驻内存（≈6-10GB → ≈2-3GB），并提供重建迁移命令与 Docker 持久化，保证容器重建不丢数据。
## Requirements
### Requirement: Collection 磁盘化存储配置
系统 SHALL 在创建 Collection 时配置磁盘化存储：向量 `on_disk=True`（mmap 存盘、按需换页）、collection 级 `on_disk_payload=True`；HNSW 图留 RAM；**不做向量量化**（不设置 quantization_config），保持 float32 全精度，避免量化带来的召回率/精度损失。`create_collection` 默认使用 `storage="disk"` 预设，`ram` 预设保留兼容旧行为。

#### Scenario: 创建磁盘化 Collection
- **WHEN** 创建 Collection 且未显式指定 storage（或指定 `storage=disk`）
- **THEN** 向量存储为 `on_disk=True`、payload 为 `on_disk_payload=True`、`quantization_config=None`，HNSW 索引参数（m/ef_construct）保留

#### Scenario: 保持全内存配置
- **WHEN** 创建 Collection 并指定 `storage=ram` 预设
- **THEN** 维持现状：向量 `on_disk=False`、payload 常驻内存

#### Scenario: 检索精度不受量化影响
- **WHEN** 使用磁盘化 Collection 进行检索
- **THEN** 无向量量化，检索结果与全内存配置一致（全精度）

### Requirement: 迁移命令重建 Collection
系统 SHALL 提供 `migrate` 迁移命令：备份旧 Collection 统计信息 → 删除重建（新存储配置）→ 重新导入全量数据 → 重建 manifest。全程幂等、可重试，迁移失败不破坏旧数据。

#### Scenario: 迁移为磁盘化存储
- **WHEN** 运行 `migrate --storage disk`
- **THEN** 旧 Collection 被删除并以磁盘化配置重建，全量影像重新导入，manifest 重建完成

#### Scenario: 迁移可重试
- **WHEN** 迁移过程中断后重新运行
- **THEN** 已完整导入的影像被跳过（断点续传），未完成的继续导入，不产生重复数据

#### Scenario: 迁移失败不破坏旧数据
- **WHEN** 迁移在删除旧 Collection 前失败
- **THEN** 旧 Collection 数据保持完好，可安全重试

### Requirement: Docker 持久化存储
系统 SHALL 在启动 Qdrant Docker 容器时挂载持久化 volume（如 `-v qdrant_data:/qdrant/storage`），容器被删除/重建后数据与索引不丢失。

#### Scenario: 容器重建后数据保留
- **WHEN** Qdrant 容器被删除并用相同 volume 重建
- **THEN** Collection 数据与索引完整保留，无需重新导入

#### Scenario: 启动命令挂载 volume
- **WHEN** 运行启动 Qdrant 的命令
- **THEN** 命令包含 volume 挂载参数

### Requirement: Qdrant 容器启动幂等
系统 SHALL 使 `_start_qdrant()` 幂等：容器已存在且运行中时直接复用；存在但停止时重启；不存在时才创建，并始终挂载 volume。

#### Scenario: 容器运行中重复调用
- **WHEN** Qdrant 容器已运行，再次调用 `_start_qdrant()`
- **THEN** 直接复用运行中容器，不创建新容器、不报错

#### Scenario: 容器停止时调用
- **WHEN** Qdrant 容器存在但已停止，调用 `_start_qdrant()`
- **THEN** 启动该容器（不重新创建）

#### Scenario: 容器不存在时调用
- **WHEN** 无 Qdrant 容器，调用 `_start_qdrant()`
- **THEN** 创建并启动新容器，挂载 volume

