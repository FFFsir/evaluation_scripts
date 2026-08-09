---
change: satellite-embedding-loader
design-doc: docs/superpowers/specs/2026-07-29-satellite-embedding-loader-design.md
base-ref: 7465e8af61e139f684b724c8da813ab8fa52695e
archived-with: 2026-07-29-satellite-embedding-loader
---

# Satellite Embedding 数据加载与可视化 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 satellite embedding 加载与可视化工具，将结构化 numpy 格式的卫星嵌入数据从 `.npy`/`.npz` 文件转换为标准 `(64, 128, 128)` 数组，并提供基于 NiceGUI 的 Web 浏览器界面来进行文件浏览、数据摘要预览和单通道灰度图渲染。

**Architecture:** 拆分为两个独立模块：`satellite_embedding_loader.py` 负责核心加载、格式转换和摘要统计；`embedding_viewer.py` 基于 NiceGUI 构建 Web 可视化界面，调用 loader 的 `load_embedding` 和 `get_summary` 函数。加载模块不依赖 GUI 库，可在命令行独立使用。WebUI 通过 PIL 生成灰度图并通过 NiceGUI 元素呈现。

**Tech Stack:** Python 3.12+, numpy 2.5.1+, NiceGUI, Pillow (PIL), pytest, uv 包管理

## 全局约束

- Python >= 3.12.12
- numpy >= 2.5.1（已安装）
- 加载模块不允许依赖 nicegui、pillow 或任何 GUI 相关库
- 所有面向用户的中文错误信息使用简体中文
- `load_embedding` 返回类型固定为 `np.ndarray`，shape `(64, 128, 128)`，dtype `float64`
- WebUI 默认端口 8002，默认数据目录 `../download_scripts/output/SE/`
- 项目根目录执行 `uv run` 运行脚本，`uv run pytest` 运行测试

---

## 文件结构

本计划涉及以下文件的创建：

| 文件 | 职责 |
|------|------|
| `src/__init__.py` | 空文件，标记 Python 包 |
| `src/satellite_embedding_loader.py` | 核心加载、格式转换、摘要统计、CLI 入口 |
| `src/embedding_viewer.py` | NiceGUI Web 界面：文件浏览、摘要预览、灰度渲染 |
| `tests/__init__.py` | 空文件，标记测试包 |
| `tests/test_loader.py` | 加载模块的单元测试（15 个测试用例） |
| `tests/conftest.py` | pytest fixtures：生成测试用 numpy 数据 |

无现有文件需修改。

---

### Task 1: 项目结构初始化

**Files:**
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `conftest.py` 中的 pytest fixtures，供 Task 2 所有测试使用

- [x] **Step 1: 创建 `src/__init__.py`**

```python
# src/__init__.py
```

- [x] **Step 2: 创建 `tests/__init__.py`**

```python
# tests/__init__.py
```

- [x] **Step 3: 创建 `tests/conftest.py`，编写测试 fixtures**

```python
"""pytest fixtures for satellite embedding loader tests."""
import numpy as np
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def sample_structured_array():
    """创建一个 (128, 128) 结构化数组，含 64 个 float64 命名字段 'A00'-'A63'."""
    dtype = [(f"A{i:02d}", np.float64) for i in range(64)]
    data = np.zeros((128, 128), dtype=dtype)
    rng = np.random.default_rng(42)
    for i in range(64):
        field_name = f"A{i:02d}"
        data[field_name] = rng.uniform(-1.0, 1.0, (128, 128))
    return data


@pytest.fixture
def sample_npy_path(sample_structured_array):
    """将结构化数组保存为临时 .npy 文件并返回路径."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, sample_structured_array)
        yield Path(f.name)
    import os
    os.unlink(f.name)


@pytest.fixture
def sample_npz_path(sample_structured_array):
    """将结构化数组保存为临时 .npz 文件（embedding 键）并返回路径."""
    converted = np.lib.recfunctions.structured_to_unstructured(sample_structured_array)
    converted = converted.transpose(2, 0, 1).astype(np.float64)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, embedding=converted)
        yield Path(f.name)
    import os
    os.unlink(f.name)


@pytest.fixture
def sample_plain_array():
    """创建一个普通的 (64, 128, 128) float64 数组（非结构化）。"""
    rng = np.random.default_rng(42)
    return rng.uniform(-1.0, 1.0, (64, 128, 128)).astype(np.float64)


@pytest.fixture
def sample_plain_npy_path(sample_plain_array):
    """将一个普通的 (64, 128, 128) 数组保存为临时 .npy 文件."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, sample_plain_array)
        yield Path(f.name)
    import os
    os.unlink(f.name)


@pytest.fixture
def sample_npz_missing_key_path():
    """创建一个 .npz 文件，不含 'embedding' 键."""
    arr = np.zeros((10, 10))
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, other_key=arr)
        yield Path(f.name)
    import os
    os.unlink(f.name)


@pytest.fixture
def temp_dir():
    """创建一个临时目录并返回路径."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
```

- [x] **Step 4: 验证 fixtures 可导入**

```bash
uv run python -c "import tests.conftest; print('Fixtures OK')"
```

- [x] **Step 5: 提交**

```bash
git add src/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: initialize project structure with test fixtures

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: 实现 `load_embedding` 核心加载函数

**Files:**
- Create: `src/satellite_embedding_loader.py`
- Create: `tests/test_loader.py`

**Interfaces:**
- Produces: `load_embedding(filepath: str | Path) -> np.ndarray` — 返回 shape `(64, 128, 128)` dtype `float64` 的数组
- Consumes: `tests/conftest.py` 中的所有 fixtures

- [x] **Step 1: 编写失败的测试（仅 import 测试）**

```python
"""Tests for satellite_embedding_loader module."""
import numpy as np
import pytest
from pathlib import Path
from src.satellite_embedding_loader import load_embedding


def test_import():
    """验证模块可导入."""
    assert load_embedding is not None
```

- [x] **Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_loader.py::test_import -v
```

- [x] **Step 3: 创建 `src/satellite_embedding_loader.py`，编写 import 和占位函数**

```python
"""Satellite Embedding 数据加载与格式转换模块.

将结构化 numpy dtype 存储的卫星嵌入数据从 .npy/.npz 文件转换为
标准 (64, 128, 128) float64 数组。
"""
import warnings
from pathlib import Path
import numpy as np


def load_embedding(filepath):
    """加载 satellite embedding 文件并转换为标准格式."""
    pass
```

- [x] **Step 4: 运行测试验证可通过**

```bash
uv run pytest tests/test_loader.py::test_import -v
```

- [x] **Step 5: 编写 `.npy` 加载的失败测试**

```python
def test_npy_success(sample_npy_path):
    """验证能从结构化 .npy 文件正确加载数据."""
    result = load_embedding(sample_npy_path)
    assert isinstance(result, np.ndarray)
    assert result.shape == (64, 128, 128)
    assert result.dtype == np.float64


def test_npz_success(sample_npz_path):
    """验证能从 .npz 文件正确加载数据."""
    result = load_embedding(sample_npz_path)
    assert isinstance(result, np.ndarray)
    assert result.shape == (64, 128, 128)
    assert result.dtype == np.float64


def test_file_not_found():
    """验证文件不存在时抛出 FileNotFoundError 并包含中文信息."""
    with pytest.raises(FileNotFoundError, match="文件不存在"):
        load_embedding(Path("/nonexistent/file.npy"))


def test_wrong_extension():
    """验证不支持的扩展名抛出 ValueError 并包含中文信息."""
    with pytest.raises(ValueError, match="不支持的文件格式"):
        load_embedding(Path("/tmp/test.txt"))


def test_non_structured_dtype(sample_plain_npy_path):
    """验证非结构化数组能幂等返回（已是 (64,128,128) 直接返回）."""
    result = load_embedding(sample_plain_npy_path)
    assert result.shape == (64, 128, 128)
    assert result.dtype == np.float64


def test_wrong_shape():
    """验证结构化数组 shape 不是 (128,128) 时抛出 ValueError."""
    pass  # 需要创建 shape 错误的测试文件，稍后实现


def test_npz_missing_key(sample_npz_missing_key_path):
    """验证 .npz 缺少 'embedding' 键时抛出 KeyError."""
    with pytest.raises(KeyError, match="缺少 'embedding' 键"):
        load_embedding(sample_npz_missing_key_path)


def test_idempotent(sample_plain_array):
    """验证已为标准格式的数组幂等返回."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, sample_plain_array)
        path = Path(f.name)
    try:
        result = load_embedding(path)
        assert result.shape == (64, 128, 128)
        assert result.dtype == np.float64
    finally:
        os.unlink(path)
```

- [x] **Step 6: 运行测试验证失败**

```bash
uv run pytest tests/test_loader.py -v -k "not test_wrong_shape"
```

预期：除 `test_import` 外全部 FAIL

- [x] **Step 7: 实现 `load_embedding` 完整逻辑**

```python
def load_embedding(filepath):
    """加载 satellite embedding 文件并转换为标准 (64, 128, 128) float64 格式.

    Args:
        filepath: .npy 或 .npz 文件路径.

    Returns:
        shape (64, 128, 128), dtype float64 的 numpy 数组.

    Raises:
        FileNotFoundError: 文件不存在.
        ValueError: 不支持的格式或 shape 不符合预期.
        KeyError: .npz 文件中缺少 'embedding' 键.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix not in (".npy", ".npz"):
        raise ValueError(
            f"不支持的文件格式: {suffix}，仅支持 .npy / .npz"
        )

    if suffix == ".npy":
        raw = np.load(filepath)
    elif suffix == ".npz":
        archive = np.load(filepath)
        if "embedding" not in archive:
            raise KeyError(
                f".npz 文件中缺少 'embedding' 键，可用键: {list(archive.keys())}"
            )
        raw = archive["embedding"]

    # 如果已是普通数组（非结构化），幂等返回
    if raw.dtype.names is None:
        expected_shape = (64, 128, 128)
        if raw.shape == expected_shape:
            return raw.astype(np.float64, copy=False)
        raise ValueError(
            f"期望 shape {expected_shape}，实际: {raw.shape}"
        )

    # 结构化 dtype → 验证 shape 再转换
    if raw.shape != (128, 128):
        raise ValueError(
            f"期望结构化 dtype shape (128, 128)，实际: {raw.shape}"
        )

    field_count = len(raw.dtype.names)
    if field_count != 64:
        warnings.warn(f"字段数={field_count}，预期 64")

    data = np.lib.recfunctions.structured_to_unstructured(raw)  # (128, 128, N)
    result = data.transpose(2, 0, 1)  # (N, 128, 128)
    result = result.astype(np.float64, copy=False)
    return result
```

- [x] **Step 8: 为 `test_wrong_shape` 创建测试 fixture 并完善测试**

需要在 `conftest.py` 末尾追加：

```python
@pytest.fixture
def wrong_shape_npy_path():
    """创建一个 shape (256, 256) 的结构化数组保存为临时 .npy."""
    dtype = [(f"A{i:02d}", np.float64) for i in range(64)]
    data = np.zeros((256, 256), dtype=dtype)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, data)
        yield Path(f.name)
    import os
    os.unlink(f.name)
```

更新 `test_wrong_shape`:

```python
def test_wrong_shape(wrong_shape_npy_path):
    """验证结构化数组 shape 不是 (128,128) 时抛出 ValueError."""
    with pytest.raises(ValueError, match="期望结构化 dtype shape"):
        load_embedding(wrong_shape_npy_path)
```

- [x] **Step 9: 运行全部测试验证通过**

```bash
uv run pytest tests/test_loader.py -v
```

预期：全部 PASS（约 8 个测试）

- [x] **Step 10: 提交**

```bash
git add src/satellite_embedding_loader.py tests/test_loader.py tests/conftest.py
git commit -m "feat: implement load_embedding with format detection and conversion

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 实现摘要统计 `get_summary` 和 `print_summary`

**Files:**
- Modify: `src/satellite_embedding_loader.py` — 追加两个函数
- Modify: `tests/test_loader.py` — 追加摘要相关测试

**Interfaces:**
- Consumes: `load_embedding` (Task 2)
- Produces:
  - `get_summary(data: np.ndarray) -> dict` — 返回包含 shape、dtype、全局统计和 64 通道统计的字典
  - `print_summary(data: np.ndarray) -> None` — 格式化打印摘要到 stdout

- [x] **Step 1: 编写失败测试**

在 `tests/test_loader.py` 末尾追加：

```python
from src.satellite_embedding_loader import get_summary, print_summary


def test_get_summary_output_keys(sample_plain_array):
    """验证 get_summary 返回所有必需字段."""
    summary = get_summary(sample_plain_array)
    assert "shape" in summary
    assert "dtype" in summary
    assert "global_min" in summary
    assert "global_max" in summary
    assert "global_mean" in summary
    assert "channels" in summary
    assert "channel_names" in summary


def test_get_summary_channel_count(sample_plain_array):
    """验证 channels 列表长度为 64."""
    summary = get_summary(sample_plain_array)
    assert len(summary["channels"]) == 64
    assert len(summary["channel_names"]) == 64


def test_get_summary_shape(sample_plain_array):
    """验证 shape 字段正确."""
    summary = get_summary(sample_plain_array)
    assert summary["shape"] == (64, 128, 128)
    assert summary["dtype"] == "float64"


def test_get_summary_channel_structure(sample_plain_array):
    """验证每个通道条目包含 name/min/max/mean."""
    summary = get_summary(sample_plain_array)
    for ch in summary["channels"]:
        assert "name" in ch
        assert "min" in ch
        assert "max" in ch
        assert "mean" in ch


def test_get_summary_global_stats_range(sample_plain_array):
    """验证全局统计在合理范围内."""
    summary = get_summary(sample_plain_array)
    assert -1.0 <= summary["global_min"] <= 1.0
    assert -1.0 <= summary["global_max"] <= 1.0
    assert -1.0 <= summary["global_mean"] <= 1.0
    assert summary["global_min"] <= summary["global_mean"] <= summary["global_max"]


def test_print_summary_runs(sample_plain_array, capsys):
    """验证 print_summary 不会崩溃并能输出内容."""
    print_summary(sample_plain_array)
    captured = capsys.readouterr()
    assert "shape" in captured.out or "Shape" in captured.out or "64" in captured.out
```

- [x] **Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_loader.py -v -k "summary or print_summary"
```

预期：全部 FAIL（函数未定义）

- [x] **Step 3: 在 `satellite_embedding_loader.py` 中实现 `get_summary` 和 `print_summary`**

在 `load_embedding` 函数之后追加：

```python
def get_summary(data):
    """生成 embedding 数据的摘要统计.

    Args:
        data: shape (64, 128, 128) 的 numpy 数组.

    Returns:
        dict: 包含 shape、dtype、全局统计和每个通道统计的字典.

        结构示例:
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
    """
    shape = data.shape
    dtype_name = str(data.dtype)
    global_min = float(data.min())
    global_max = float(data.max())
    global_mean = float(data.mean())

    channels = []
    channel_names = []
    for i in range(data.shape[0]):
        name = f"A{i:02d}"
        channel_data = data[i]
        channels.append({
            "name": name,
            "min": float(channel_data.min()),
            "max": float(channel_data.max()),
            "mean": float(channel_data.mean()),
        })
        channel_names.append(name)

    return {
        "shape": shape,
        "dtype": dtype_name,
        "global_min": global_min,
        "global_max": global_max,
        "global_mean": global_mean,
        "channels": channels,
        "channel_names": channel_names,
    }


def print_summary(data):
    """格式化打印 embedding 数据的摘要统计.

    Args:
        data: shape (64, 128, 128) 的 numpy 数组.
    """
    s = get_summary(data)
    print(f"Shape: {s['shape']}")
    print(f"Dtype: {s['dtype']}")
    print(f"Global  Min: {s['global_min']:.6f}  Max: {s['global_max']:.6f}  Mean: {s['global_mean']:.6f}")
    print(f"Channels ({len(s['channels'])}):")
    print(f"  {'Name':>6}  {'Min':>10}  {'Max':>10}  {'Mean':>10}")
    for ch in s["channels"]:
        print(f"  {ch['name']:>6}  {ch['min']:10.6f}  {ch['max']:10.6f}  {ch['mean']:10.6f}")
```

- [x] **Step 4: 运行测试验证通过**

```bash
uv run pytest tests/test_loader.py -v -k "summary or print_summary"
```

- [x] **Step 5: 运行全部测试确保无回归**

```bash
uv run pytest tests/test_loader.py -v
```

- [x] **Step 6: 提交**

```bash
git add src/satellite_embedding_loader.py tests/test_loader.py
git commit -m "feat: add get_summary and print_summary for embedding statistics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 实现 `load_embeddings_from_dir` 批量加载

**Files:**
- Modify: `src/satellite_embedding_loader.py` — 追加函数
- Modify: `tests/test_loader.py` — 追加批量加载测试

**Interfaces:**
- Consumes: `load_embedding` (Task 2)
- Produces: `load_embeddings_from_dir(directory: str | Path, pattern: str | None = None, prefer_npz: bool = True) -> dict[str, np.ndarray]` — 返回 `{文件名(不含扩展名): np.ndarray}`

- [x] **Step 1: 编写失败测试**

在 `tests/test_loader.py` 末尾追加：

```python
from src.satellite_embedding_loader import load_embeddings_from_dir


def test_load_from_dir_mixed_formats(sample_npy_path, sample_npz_path, temp_dir):
    """验证混合 .npy + .npz 文件的批量加载."""
    import shutil
    # 复制两个文件到临时目录
    npz_dest = temp_dir / "sample.npz"
    npy_dest = temp_dir / "other.npy"
    shutil.copy2(str(sample_npz_path), str(npz_dest))
    shutil.copy2(str(sample_npy_path), str(npy_dest))

    result = load_embeddings_from_dir(temp_dir)
    assert isinstance(result, dict)
    assert len(result) >= 2
    assert "sample" in result
    assert "other" in result

    for arr in result.values():
        assert arr.shape == (64, 128, 128)
        assert arr.dtype == np.float64


def test_filename_keys_no_extension(sample_npy_path, temp_dir):
    """验证返回的字典键不包含文件扩展名."""
    import shutil
    dest = temp_dir / "test_file.npy"
    shutil.copy2(str(sample_npy_path), str(dest))
    result = load_embeddings_from_dir(temp_dir)
    assert "test_file" in result
    assert "test_file.npy" not in result


def test_dir_not_found():
    """验证目录不存在时抛出 FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_embeddings_from_dir(Path("/nonexistent/directory"))


def test_not_a_directory(tmp_path):
    """验证路径不是目录时抛出 NotADirectoryError."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("not a dir")
    with pytest.raises(NotADirectoryError):
        load_embeddings_from_dir(file_path)
```

- [x] **Step 2: 运行测试验证失败**

```bash
uv run pytest tests/test_loader.py -v -k "from_dir"
```

- [x] **Step 3: 实现 `load_embeddings_from_dir`**

在 `satellite_embedding_loader.py` 中的 `load_embedding` 函数之后追加：

```python
def load_embeddings_from_dir(directory, pattern=None, prefer_npz=True):
    """批量加载目录中的 satellite embedding 文件.

    Args:
        directory: 目录路径.
        pattern: 文件匹配模式 (glob pattern)，None 时匹配 *.npy 和 *.npz.
        prefer_npz: 同名文件同时存在 .npy 和 .npz 时优先加载 .npz (默认 True).

    Returns:
        dict[str, np.ndarray]: 文件名(不含扩展名) → 数组的映射.

    Raises:
        FileNotFoundError: 目录不存在.
        NotADirectoryError: 路径不是目录.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"路径不是目录: {directory}")

    if pattern is None:
        npy_files = list(directory.glob("*.npy"))
        npz_files = list(directory.glob("*.npz"))
    else:
        npy_files = list(directory.glob(pattern))
        npz_files = []  # 自定义 pattern 时不区分 .npy/.npz

    # 收集已处理的 stem 以避免重复
    processed_stems = set()
    result = {}

    # 先处理 .npz（优先级更高）
    if prefer_npz:
        for fpath in npz_files:
            stem = fpath.stem
            processed_stems.add(stem)
            result[stem] = load_embedding(fpath)

        for fpath in npy_files:
            stem = fpath.stem
            if stem not in processed_stems:
                result[stem] = load_embedding(fpath)
    else:
        for fpath in npy_files:
            stem = fpath.stem
            processed_stems.add(stem)
            result[stem] = load_embedding(fpath)

        for fpath in npz_files:
            stem = fpath.stem
            if stem not in processed_stems:
                result[stem] = load_embedding(fpath)

    return result
```

- [x] **Step 4: 运行测试验证通过**

```bash
uv run pytest tests/test_loader.py -v -k "from_dir"
```

- [x] **Step 5: 运行全部测试确保无回归**

```bash
uv run pytest tests/test_loader.py -v
```

- [x] **Step 6: 提交**

```bash
git add src/satellite_embedding_loader.py tests/test_loader.py
git commit -m "feat: add load_embeddings_from_dir for batch directory loading

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 为 loader 模块添加 CLI 入口

**Files:**
- Modify: `src/satellite_embedding_loader.py` — 追加 `__main__` 块

**Interfaces:**
- Produces: `python -m src.satellite_embedding_loader` 可从命令行运行
- Consumes: `load_embedding`, `print_summary` (Task 2, 3)

- [x] **Step 1: 在 `satellite_embedding_loader.py` 末尾追加 `if __name__ == "__main__"` 块**

```python
if __name__ == "__main__":
    """CLI 入口：加载默认目录中的文件并打印摘要.

    用法:
        python -m src.satellite_embedding_loader

    默认扫描 ../download_scripts/output/SE/ 目录中的 .npy/.npz 文件，
    加载每个文件并打印摘要统计。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Satellite Embedding 数据加载与摘要工具"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="../download_scripts/output/SE/",
        help="文件或目录路径 (默认: ../download_scripts/output/SE/)",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="文件匹配模式 (glob pattern，默认: *.npy 和 *.npz)",
    )
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"错误: 路径不存在: {target}")
        exit(1)

    if target.is_file():
        print(f"加载文件: {target.name}")
        data = load_embedding(target)
        print_summary(data)
    elif target.is_dir():
        print(f"扫描目录: {target}")
        results = load_embeddings_from_dir(target, pattern=args.pattern)
        if not results:
            print("  未找到 .npy 或 .npz 文件.")
        for name, data in results.items():
            print(f"\n{'='*60}")
            print(f"文件: {name}")
            print(f"{'='*60}")
            print_summary(data)
```

- [x] **Step 2: 验证 CLI 模块可导入且无语法错误**

```bash
uv run python -c "import src.satellite_embedding_loader; print('CLI entry ready')"
```

- [x] **Step 3: 提交**

```bash
git add src/satellite_embedding_loader.py
git commit -m "feat: add CLI entry point for satellite_embedding_loader

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: 安装 NiceGUI 和 Pillow 依赖

- [x] **Step 1: 安装依赖**

```bash
uv add nicegui pillow
```

- [x] **Step 2: 验证安装**

```bash
uv run python -c "import nicegui; import PIL; print(f'NiceGUI {nicegui.__version__}, PIL OK')"
```

- [x] **Step 3: 提交**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add nicegui and pillow dependencies for WebUI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: 实现灰度渲染工具函数

**Files:**
- Create: `src/embedding_viewer.py`

**Interfaces:**
- Consumes: `np.ndarray` (shape `(64, 128, 128)` float64) 和 PIL `Image`
- Produces: `render_channel_grayscale(data: np.ndarray, channel_idx: int, zoom: int = 1) -> Image.Image`

注意：此任务创建文件但不包含 NiceGUI 界面逻辑，仅为渲染函数编写并验证。

- [x] **Step 1: 创建 `src/embedding_viewer.py`，编写灰度渲染函数**

```python
"""Satellite Embedding Web 可视化界面.

基于 NiceGUI 构建，提供文件浏览器、嵌入数据摘要预览和单通道灰度渲染功能。
"""
import numpy as np
from PIL import Image


def render_channel_grayscale(data, channel_idx, zoom=1):
    """将单个通道渲染为 PIL 灰度图.

    采用 [-1, 1] → [0, 255] 线性映射，缩放使用最近邻插值以保持像素边界清晰。

    Args:
        data: shape (64, 128, 128) float64 数组.
        channel_idx: 0-based 通道索引 (0-63).
        zoom: 放大倍率 (1/2/4)，默认 1.

    Returns:
        PIL Image (mode='L'): 灰度图。
    """
    channel = data[channel_idx]  # (128, 128)
    gray = ((channel + 1.0) / 2.0 * 255.0)  # [-1, 1] → [0, 255]
    gray = gray.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    if zoom > 1:
        img = img.resize((128 * zoom, 128 * zoom), Image.NEAREST)
    return img
```

- [x] **Step 2: 编写快速验证脚本**

```bash
uv run python -c "
from src.embedding_viewer import render_channel_grayscale
import numpy as np
data = np.random.default_rng(0).uniform(-1, 1, (64, 128, 128)).astype(np.float64)
img = render_channel_grayscale(data, 0, 1)
print(f'1x: size={img.size}, mode={img.mode}')
img2 = render_channel_grayscale(data, 0, 2)
print(f'2x: size={img2.size}, mode={img2.mode}')
img4 = render_channel_grayscale(data, 0, 4)
print(f'4x: size={img4.size}, mode={img4.mode}')
print('Render OK')
"
```

预期输出：
```
1x: size=(128, 128), mode=L
2x: size=(256, 256), mode=L
4x: size=(512, 512), mode=L
Render OK
```

- [x] **Step 3: 提交**

```bash
git add src/embedding_viewer.py
git commit -m "feat: add render_channel_grayscale for single-channel PIL rendering

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: 实现 NiceGUI Web 界面主框架

**Files:**
- Modify: `src/embedding_viewer.py` — 追加 NiceGUI 页面定义和 `__main__` 入口

**Interfaces:**
- Consumes: `render_channel_grayscale` (Task 7), `load_embedding` (Task 2), `get_summary` (Task 3)
- Produces: `python src/embedding_viewer.py` 启动 Web 服务器（端口 8002）

- [x] **Step 1: 编写完整的 NiceGUI Web 界面**

替换/追加 `src/embedding_viewer.py` 全部内容为：

```python
"""Satellite Embedding Web 可视化界面.

基于 NiceGUI 构建，提供文件浏览器、嵌入数据摘要预览和单通道灰度渲染功能。
"""
from pathlib import Path
import numpy as np
from PIL import Image

from nicegui import ui, app

from src.satellite_embedding_loader import load_embedding, get_summary

# ---------- 默认配置 ----------
DEFAULT_DIR = str(Path(__file__).resolve().parent.parent / "download_scripts" / "output" / "SE")
DEFAULT_PORT = 8002

# ---------- 灰度渲染 ----------
def render_channel_grayscale(data, channel_idx, zoom=1):
    """将单个通道渲染为 PIL 灰度图.

    Args:
        data: shape (64, 128, 128) float64 数组.
        channel_idx: 0-based 通道索引 (0-63).
        zoom: 放大倍率 (1/2/4)，默认 1.

    Returns:
        PIL Image (mode='L').
    """
    channel = data[channel_idx]
    gray = ((channel + 1.0) / 2.0 * 255.0)
    gray = gray.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    if zoom > 1:
        img = img.resize((128 * zoom, 128 * zoom), Image.NEAREST)
    return img


# ---------- 页面构建 ----------
@ui.page("/")
def index():
    """主页面：文件浏览器 + 数据预览."""
    current_data = {"data": None, "summary": None}

    ui.page_title("Satellite Embedding 数据浏览器")

    with ui.header(elevated=True).classes("bg-primary text-white"):
        ui.label("Satellite Embedding 数据浏览器").classes("text-h4")

    with ui.row().classes("w-full items-center gap-4"):
        dir_input = ui.input(
            label="数据目录",
            value=DEFAULT_DIR,
        ).classes("w-96")

        async def browse_directory():
            """扫描目录并刷新文件列表."""
            directory = Path(dir_input.value)
            if not directory.exists() or not directory.is_dir():
                ui.notify(f"目录不存在或无效: {dir_input.value}", type="negative")
                file_list.clear()
                return

            npy_files = sorted(directory.glob("*.npy"))
            npz_files = sorted(directory.glob("*.npz"))

            file_list.clear()
            if not npy_files and not npz_files:
                with file_list:
                    ui.label("未找到 .npy 或 .npz 文件").classes("text-grey")
                return

            processed = set()
            # .npz 优先
            for fpath in npz_files:
                stem = fpath.stem
                if stem in processed:
                    continue
                processed.add(stem)
                stat = fpath.stat()
                size_kb = stat.st_size / 1024
                with file_list:
                    with ui.row().classes("w-full items-center border-b py-2"):
                        ui.label(f"{fpath.name}").classes("font-mono text-sm")
                        ui.label(f"{size_kb:.1f} KB").classes("text-grey text-xs")
                        ui.button(
                            "查看数据",
                            on_click=lambda _, fp=fpath: show_preview(fp),
                        ).props("flat dense size=sm")

            for fpath in npy_files:
                stem = fpath.stem
                if stem in processed:
                    continue
                processed.add(stem)
                stat = fpath.stat()
                size_kb = stat.st_size / 1024
                with file_list:
                    with ui.row().classes("w-full items-center border-b py-2"):
                        ui.label(f"{fpath.name}").classes("font-mono text-sm")
                        ui.label(f"{size_kb:.1f} KB").classes("text-grey text-xs")
                        ui.button(
                            "查看数据",
                            on_click=lambda _, fp=fpath: show_preview(fp),
                        ).props("flat dense size=sm")

        ui.button("浏览", on_click=browse_directory).props("flat")

    file_list = ui.column().classes("w-full")

    # ---------- 预览弹窗 ----------
    async def show_preview(filepath):
        """加载文件并在弹窗中展示摘要和灰度图."""
        try:
            data = load_embedding(filepath)
            summary = get_summary(data)
            current_data["data"] = data
            current_data["summary"] = summary
        except Exception as e:
            ui.notify(f"加载失败: {e}", type="negative")
            return

        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
            dialog.open()

            ui.label(f"文件: {filepath.name}").classes("text-h5")
            ui.label(
                f"Shape: {summary['shape']}  |  Dtype: {summary['dtype']}  |  "
                f"Global Min: {summary['global_min']:.4f}  Max: {summary['global_max']:.4f}  "
                f"Mean: {summary['global_mean']:.4f}"
            ).classes("text-grey text-sm")

            with ui.row().classes("items-center gap-4 mt-4"):
                channel_names = summary["channel_names"]
                channel_select = ui.select(
                    label="通道",
                    options=channel_names,
                    value=channel_names[0],
                ).classes("w-32")

                zoom_options = {1: "1x", 2: "2x", 4: "4x"}
                zoom_select = ui.select(
                    label="缩放",
                    options=zoom_options,
                    value=1,
                ).props("dense")

            image_display = ui.image().classes("mt-4")

            # 灰度条
            gray_bar = ui.column().classes("w-full mt-2")
            with gray_bar:
                with ui.row().classes("w-full items-center"):
                    ui.label("-1").classes("text-xs")
                    # 颜色条：从黑到白的渐变
                    ui.html(
                        '<div style="width:256px;height:16px;'
                        'background:linear-gradient(to right,black,white);'
                        'border:1px solid #ccc;"></div>'
                    )
                    ui.label("+1").classes("text-xs")

            def update_image():
                """更新灰度图渲染."""
                ch_idx = channel_names.index(channel_select.value)
                z = zoom_select.value
                img = render_channel_grayscale(current_data["data"], ch_idx, z)
                # 将 PIL Image 转为 base64 data URI
                import io, base64
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                data_uri = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
                image_display.set_source(data_uri)

            channel_select.on("update:model-value", lambda _: update_image())
            zoom_select.on("update:model-value", lambda _: update_image())

            # 初始渲染
            update_image()

            # 通道统计表
            with ui.element("div").classes("max-h-64 overflow-y-auto mt-4 w-full"):
                columns = [
                    {"name": "name", "label": "通道", "field": "name", "align": "left"},
                    {"name": "min", "label": "Min", "field": "min", "align": "right"},
                    {"name": "max", "label": "Max", "field": "max", "align": "right"},
                    {"name": "mean", "label": "Mean", "field": "mean", "align": "right"},
                ]
                rows = [
                    {
                        "name": ch["name"],
                        "min": f"{ch['min']:.6f}",
                        "max": f"{ch['max']:.6f}",
                        "mean": f"{ch['mean']:.6f}",
                    }
                    for ch in summary["channels"]
                ]
                ui.table(columns=columns, rows=rows, row_key="name").classes("w-full")

            ui.button("关闭", on_click=lambda: dialog.close()).props("flat")

    # 页面加载后自动扫描默认目录
    ui.timer(0.1, callback=browse_directory, once=True)


# ---------- 启动入口 ----------
if __name__ in {"__main__", "__mp_main__"}:
    import argparse

    parser = argparse.ArgumentParser(
        description="Satellite Embedding Web 可视化界面"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"监听端口 (默认: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_DIR,
        help=f"默认数据目录 (默认: {DEFAULT_DIR})",
    )
    args, unknown = parser.parse_known_args()

    DEFAULT_DIR = args.dir
    ui.run(port=args.port, reload=False)
```

- [x] **Step 2: 验证模块可导入**

```bash
uv run python -c "from src.embedding_viewer import render_channel_grayscale; print('Module OK')"
```

- [x] **Step 3: 提交**

```bash
git add src/embedding_viewer.py
git commit -m "feat: implement NiceGUI WebUI with file browser, summary preview, and grayscale rendering

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: 运行完整验证清单

- [x] **Step 1: 运行所有单元测试**

```bash
uv run pytest tests/test_loader.py -v
```

预期：所有测试 PASS（约 17 个测试）

- [x] **Step 2: 验证 CLI 模块可运行（如果存在测试数据）**

```bash
uv run python -m src.satellite_embedding_loader --help
```

预期：打印 CLI 帮助信息

- [x] **Step 3: 验证 WebUI 模块可导入且无语法错误**

```bash
uv run python -c "import src.embedding_viewer; print('WebUI module ready')"
```

- [x] **Step 4: 验证 render_channel_grayscale 功能正确**

```bash
uv run python -c "
from src.embedding_viewer import render_channel_grayscale
import numpy as np

# 测试 [-1, 1] 映射
data = np.zeros((64, 128, 128), dtype=np.float64)
data[0, 0, 0] = -1.0
data[0, 127, 127] = 1.0
img = render_channel_grayscale(data, 0, 1)
pixels = list(img.getdata())
assert pixels[0] == 0, f'Expected 0 (black for -1), got {pixels[0]}'
assert pixels[-1] == 255, f'Expected 255 (white for 1), got {pixels[-1]}'

# 测试 zoom
img2x = render_channel_grayscale(data, 0, 2)
assert img2x.size == (256, 256), f'Expected (256,256), got {img2x.size}'
img4x = render_channel_grayscale(data, 0, 4)
assert img4x.size == (512, 512), f'Expected (512,512), got {img4x.size}'

print('All render checks passed')
"
```

- [x] **Step 5: 如有 SE 真实样本数据，启动 WebUI 并手动验证检查清单**

```bash
uv run python src/embedding_viewer.py
```

浏览器访问 `http://127.0.0.1:8002`，逐项验证：
- [x] 页面正常加载，默认目录显示文件列表
- [x] 修改目录后浏览功能正常
- [x] 空目录显示提示信息
- [x] 点击 .npy 文件 → 弹窗正确显示摘要和灰度图
- [x] 点击 .npz 文件 → 摘要和灰度图正确
- [x] 通道下拉切换 → 灰度图实时更新
- [x] 缩放 1x/2x/4x 切换正常
- [x] 关闭弹窗后重新打开其他文件正常

---

## 验收标准清单

| # | 标准 | 验证方式 |
|---|------|---------|
| 1 | `load_embedding` 接受 `.npy` 返回 `(64,128,128) float64` | `pytest tests/test_loader.py::test_npy_success` |
| 2 | `load_embedding` 接受 `.npz` 返回 `(64,128,128) float64` | `pytest tests/test_loader.py::test_npz_success` |
| 3 | 文件不存在抛出 `FileNotFoundError`（中文信息） | `pytest tests/test_loader.py::test_file_not_found` |
| 4 | 不支持的扩展名抛出 `ValueError` | `pytest tests/test_loader.py::test_wrong_extension` |
| 5 | 非结构化数组幂等返回 | `pytest tests/test_loader.py::test_idempotent` |
| 6 | `.npz` 缺少 `embedding` 键抛出 `KeyError` | `pytest tests/test_loader.py::test_npz_missing_key` |
| 7 | `get_summary` 返回完整统计字典 | `pytest tests/test_loader.py::test_get_summary_output_keys` |
| 8 | 通道统计列表长度 = 64 | `pytest tests/test_loader.py::test_get_summary_channel_count` |
| 9 | `print_summary` 可正常运行不崩溃 | `pytest tests/test_loader.py::test_print_summary_runs` |
| 10 | 批量加载处理混合格式并去重 | `pytest tests/test_loader.py::test_load_from_dir_mixed_formats` |
| 11 | 批量加载键不含扩展名 | `pytest tests/test_loader.py::test_filename_keys_no_extension` |
| 12 | 目录不存在抛出 `FileNotFoundError` | `pytest tests/test_loader.py::test_dir_not_found` |
| 13 | 路径不是目录抛出 `NotADirectoryError` | `pytest tests/test_loader.py::test_not_a_directory` |
| 14 | CLI 入口可通过 `--help` 运行 | `python -m src.satellite_embedding_loader --help` |
| 15 | WebUI 模块可导入无错误 | `import src.embedding_viewer` |
| 16 | 灰度渲染 `[-1,1]` 正确映射到 `[0,255]` | Step 4 验证脚本 |
| 17 | 缩放 2x/4x 使用 NEAREST 插值 | Step 4 验证脚本 |
