# embedding-data-loading Specification

## Purpose
提供从 Google Satellite Embedding V1 数据集的 `.npy`/`.npz` 文件中加载嵌入数据并转换为标准多维浮点数组的功能。
## Requirements
### Requirement: 加载单个 .npy 文件

系统 SHALL 能够从指定路径读取 `.npy` 文件，将结构化 dtype（shape `(128,128)`，每元素 64 个命名字段）转换为形状为 `(64, 128, 128)` 的 `float64` numpy 数组返回。

#### Scenario: 成功加载 .npy 文件

- **WHEN** 调用加载函数并传入一个有效的 `.npy` 文件路径
- **THEN** 返回的 numpy 数组 shape 为 `(64, 128, 128)`，dtype 为 `float64`

#### Scenario: 文件不存在

- **WHEN** 调用加载函数并传入一个不存在的文件路径
- **THEN** 抛出 `FileNotFoundError` 异常

#### Scenario: 文件格式错误

- **WHEN** 调用加载函数并传入一个非 numpy 格式或非结构化 dtype 的文件
- **THEN** 抛出带有描述性错误信息的 `ValueError` 异常

### Requirement: 加载单个 .npz 文件

系统 SHALL 能够从指定路径读取 `.npz` 文件，从 `embedding` 键提取数据，并将结构化 dtype 转换为形状为 `(64, 128, 128)` 的 `float64` numpy 数组返回。

#### Scenario: 成功加载 .npz 文件

- **WHEN** 调用加载函数并传入一个有效的 `.npz` 文件路径
- **THEN** 返回的 numpy 数组 shape 为 `(64, 128, 128)`，dtype 为 `float64`

#### Scenario: .npz 中缺少 embedding 键

- **WHEN** 调用加载函数并传入一个不包含 `embedding` 键的 `.npz` 文件
- **THEN** 抛出带有描述性错误信息的 `KeyError` 异常

### Requirement: 幂等加载

系统 SHALL 在加载已为标准浮点数组格式的数据时直接返回原数组，不进行重复转换。

#### Scenario: 已转换过的数组再次加载

- **WHEN** 加载一个 dtype 为普通 `float64`（非结构化）的 `.npy` 文件，且 shape 为 `(64, 128, 128)`
- **THEN** 直接返回原数组，不报错，不重复转换

#### Scenario: 非结构化 dtype 但 shape 不兼容

- **WHEN** 加载一个 dtype 为普通 `float64` 但 shape 不是 `(64, 128, 128)` 的 `.npy` 文件
- **THEN** 记录 warning 并返回原数组

#### Scenario: 结构化 dtype 字段数不等于 64

- **WHEN** 加载一个结构化 dtype 的数组，但字段数 N ≠ 64
- **THEN** 返回 shape 为 `(N, 128, 128)` 的数组，并记录 warning 说明字段数变化

#### Scenario: 结构化 dtype shape 异常

- **WHEN** 加载一个结构化 dtype 但 shape 不是 `(128, 128)` 的数组
- **THEN** 抛出带有描述性错误信息的 `ValueError` 异常

### Requirement: 批量加载目录中的文件

系统 SHALL 支持扫描指定目录，加载其中所有匹配 `*.npy` 或 `*.npz` 模式的文件，返回一个以文件名为键、以对应 `(64, 128, 128)` 数组为值的字典。

#### Scenario: 批量加载混合格式文件

- **WHEN** 调用批量加载函数并传入包含 `.npy` 和 `.npz` 文件的目录路径
- **THEN** 返回的字典包含所有匹配文件的加载结果，键为文件名（不含扩展名）

#### Scenario: 目录中没有匹配文件

- **WHEN** 调用批量加载函数并传入一个不包含任何 `.npy` 或 `.npz` 文件的目录
- **THEN** 返回空字典

#### Scenario: 目录不存在

- **WHEN** 调用批量加载函数并传入一个不存在的目录路径
- **THEN** 抛出 `FileNotFoundError` 异常

#### Scenario: 同名文件优先加载 .npz

- **WHEN** 调用批量加载函数，目录中同时存在 `foo.npy` 和 `foo.npz` 两个同名文件
- **THEN** 默认优先加载 `.npz` 格式，跳过 `.npy`，结果中仅包含一条以 `foo` 为键的记录

### Requirement: .npy 与 .npz 一致性

系统 SHALL 确保同一数据的 `.npy` 和 `.npz` 格式加载后返回相同的 `(64, 128, 128)` 浮点数组值。

#### Scenario: 两种格式加载结果一致

- **WHEN** 分别加载同一数据源的 `.npy` 和 `.npz` 文件
- **THEN** 两个返回的数组在数值上完全相等（`np.allclose` 为 True）

### Requirement: 打印数据摘要信息

系统 SHALL 提供一个函数，接收 `(64, 128, 128)` 数组，打印其 shape、dtype、全局 min、全局 max、以及各通道的 min/max 统计信息。

#### Scenario: 打印摘要信息

- **WHEN** 调用摘要函数并传入一个有效的 `(64, 128, 128)` 数组
- **THEN** 打印输出包含 shape、dtype、全局 min/max 和各通道 min/max 的信息

