---
comet_change: satellite-embedding-loader
role: technical-design
canonical_spec: openspec
archived-with: 2026-07-29-satellite-embedding-loader
status: final
---

# Satellite Embedding 数据加载与可视化 — 技术设计

## 1. 概述

本 Design Doc 是对 OpenSpec `design.md` 高层架构决策的深度技术细化，覆盖 API 签面、边界条件处理、WebUI 交互流程、灰度渲染实现和测试策略。

## 2. 模块架构

```
src/
├── satellite_embedding_loader.py   # 核心：加载 + 格式转换 + 摘要
└── embedding_viewer.py             # WebUI：NiceGUI 文件浏览器 + 预览
```

加载模块不依赖 GUI 库，可独立于 WebUI 使用。两模块通过 `load_embedding` / `get_summary` 函数耦合。

## 3. loader 模块 API 签面

### 3.1 `load_embedding(filepath: str | Path) -> np.ndarray`

**参数**：文件路径（`.npy` 或 `.npz`）

**返回**：`(64, 128, 128)` shape、`float64` dtype 的 numpy 数组

**异常**：
| 条件 | 异常类型 | 错误信息（中文） |
|------|---------|-----------------|
| 文件不存在 | `FileNotFoundError` | `"文件不存在: {path}"` |
| 扩展名不是 `.npy`/`.npz` | `ValueError` | `"不支持的文件格式: {ext}，仅支持 .npy / .npz"` |
| 非结构化 dtype 且非 numpy 格式 | `ValueError` | `"数据不是有效的 numpy 结构化数组"` |
| shape 不是 `(128, 128)` (结构化) | `ValueError` | `"期望结构化 dtype shape (128, 128)，实际: {shape}"` |
| `.npz` 缺少 `embedding` 键 | `KeyError` | `".npz 文件中缺少 'embedding' 键，可用键: {keys}"` |

**转换逻辑**：
```
1. np.load(filepath) → raw: np.ndarray
2. 如果 raw.dtype 不是 structured（没有 .names）→ 检查 shape，直接返回（幂等）
3. 验证 raw.shape == (128, 128)，否则 ValueError
4. field_count = len(raw.dtype.names)
5. 使用 np.lib.recfunctions.structured_to_unstructured(raw) → (128, 128, N)
6. result = data.transpose(2, 0, 1) → (N, 128, 128)
7. result = result.astype(np.float64)  # 确保 dtype 一致
8. 如果 field_count != 64 → warnings.warn(f"字段数={field_count}，预期 64")
9. 返回 result
```

**幂等性**：如果 `raw.dtype.names is None`（已是普通数组），检查 shape 后直接返回，不做额外转换。

### 3.2 `load_embeddings_from_dir(directory: str | Path, pattern: str | None = None) -> dict[str, np.ndarray]`

**参数**：
- `directory`：目录路径
- `pattern`：文件匹配模式，`None` 时匹配 `*.npy` 和 `*.npz`

**返回**：`{文件名(不含扩展名): np.ndarray}`

**去重策略**：同名文件同时存在 `.npy` 和 `.npz` 时，默认优先加载 `.npz`（压缩格式），跳过 `.npy`。通过 `prefer_npz` 参数控制（默认 `True`）。

**异常**：
| 条件 | 异常类型 |
|------|---------|
| 目录不存在 | `FileNotFoundError` |
| 路径不是目录 | `NotADirectoryError` |

### 3.3 `get_summary(data: np.ndarray) -> dict`

**返回字典结构**：
```python
{
    "shape": (64, 128, 128),
    "dtype": "float64",
    "global_min": -0.874,
    "global_max": 0.923,
    "global_mean": 0.012,
    "channels": [
        {"name": "A00", "min": -0.123, "max": 0.456, "mean": 0.001},
        # ... 64 项
    ],
    "channel_names": ["A00", "A01", ..., "A63"],
}
```

### 3.4 `print_summary(data: np.ndarray) -> None`

CLI 文本输出函数，内部调用 `get_summary` 并格式化打印。不返回任何值。

## 4. WebUI 设计

### 4.1 技术选型

- **框架**：NiceGUI（参考 `download_scripts/SatelliteEmbedding/web.py` 的已验证模式）
- **灰度渲染**：PIL `Image.fromarray(gray, mode='L')` 内存生成，直接传给 `ui.image()`
- **启动端口**：`8002`（避免与参考项目的 8001 冲突）

### 4.2 页面布局

```
┌─────────────────────────────────────────────────┐
│  Satellite Embedding 数据浏览器                   │
├──────────────┬──────────────────────────────────┤
│  📂 浏览目录  │  文件列表                          │
│  [____] [浏览]│  每个文件一行：文件名 / 大小 / 时间   │
│              │  [查看数据] 按钮                    │
│              │                                  │
│              │  弹窗 (NiceGUI dialog):            │
│              │  ┌──────────────────────────────┐ │
│              │  │ 文件名 + shape/dtype/全局统计  │ │
│              │  │                              │ │
│              │  │ 通道: [A00 ▼]  缩放: [1x|2x|4x]│ │
│              │  │ ┌──────────────────────┐     │ │
│              │  │ │   灰度预览 (PIL Image) │     │ │
│              │  │ │                      │     │ │
│              │  │ └──────────────────────┘     │ │
│              │  │ [-1 ████████░░ +1] 灰度条     │ │
│              │  │                              │ │
│              │  │ 各通道 min/max 表（可滚动）     │ │
│              │  └──────────────────────────────┘ │
└──────────────┴──────────────────────────────────┘
```

### 4.3 灰度渲染实现

```python
def render_channel_grayscale(data: np.ndarray, channel_idx: int, zoom: int = 1) -> Image.Image:
    """将单个通道渲染为 PIL 灰度图。

    Args:
        data: shape (64, 128, 128) float64 数组
        channel_idx: 0-based 通道索引
        zoom: 放大倍率（1/2/4）

    Returns:
        PIL Image (mode='L')
    """
    channel = data[channel_idx]                          # (128, 128)
    gray = ((channel + 1.0) / 2.0 * 255.0)               # [-1,1] → [0,255]
    gray = gray.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode='L')
    if zoom > 1:
        img = img.resize((128 * zoom, 128 * zoom), Image.NEAREST)
    return img
```

**缩放选项**：
- 1x：原始 128×128（网页上约 130px 宽）
- 2x：256×256（最近邻插值，保持像素边界清晰）
- 4x：512×512

**为什么用 NEAREST 而非 LANCZOS**：嵌入向量每个"像素"是离散样本点，最近邻插值保持原始值不变；LANCZOS 会产生插值伪影，误导后续分析判断。

### 4.4 交互流程

1. 页面加载 → `refresh_file_list()` 扫描默认目录
2. 用户修改目录 → 点击"浏览" → 重新扫描
3. 点击"查看数据" → `load_embedding()` → 填充弹窗内容 → 渲染默认通道 A00 的灰度图
4. 切换通道下拉 → `render_channel_grayscale()` → 更新 `ui.image()`
5. 缩放选项切换 → 重新渲染对应倍率
6. 关闭弹窗 → 释放弹窗内引用

## 5. 测试策略

### 5.1 单元测试（pytest）

测试文件：`tests/test_loader.py`

```
test_load_embedding:
  ├── test_npy_success           # .npy → (64,128,128) float64
  ├── test_npz_success           # .npz → (64,128,128) float64
  ├── test_file_not_found        # FileNotFoundError + 中文信息
  ├── test_wrong_extension       # ValueError (.txt)
  ├── test_non_structured_dtype  # 普通数组 → 幂等返回
  ├── test_wrong_shape           # (256,256) → ValueError
  ├── test_npz_missing_key       # KeyError
  ├── test_idempotent            # 已是 (64,128,128) → 直接返回
  └── test_npy_npz_consistency   # 同数据两种格式返回相同值

test_load_from_dir:
  ├── test_mixed_formats         # .npy + .npz 混合
  ├── test_empty_dir             # 返回 {}
  ├── test_dir_not_found         # FileNotFoundError
  ├── test_not_a_directory       # NotADirectoryError
  ├── test_filename_keys         # 键不含扩展名
  └── test_prefer_npz            # 同名时优先 .npz

test_get_summary:
  ├── test_output_keys           # 包含所有必需字段
  └── test_channel_count         # channels 列表长度 = 64
```

### 5.2 集成测试

`tests/test_integration.py`：用 SE 目录真实样本验证端到端流程。

### 5.3 WebUI 手动验证清单

- [ ] 页面正常加载，默认目录显示文件列表
- [ ] 修改目录后浏览功能正常
- [ ] 空目录显示提示信息
- [ ] 点击 .npy 文件 → 弹窗正确显示摘要和灰度图
- [ ] 点击 .npz 文件 → 同上
- [ ] 通道下拉切换 → 灰度图实时更新
- [ ] 缩放 1x/2x/4x 切换正常
- [ ] 关闭弹窗后重新打开其他文件正常

## 6. 依赖

| 包 | 用途 | 安装方式 |
|----|------|---------|
| numpy | 数据加载和格式转换 | 已安装 |
| nicegui | Web 可视化框架 | `uv add nicegui` |
| pillow | 灰度图生成（PIL） | `uv add pillow` |
| pytest | 测试框架 | `uv add pytest --group dev` |
