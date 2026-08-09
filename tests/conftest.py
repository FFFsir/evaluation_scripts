"""pytest fixtures for satellite embedding loader tests."""
import numpy as np
import tempfile
from pathlib import Path
import pytest
from numpy.lib.recfunctions import structured_to_unstructured


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
    converted = structured_to_unstructured(sample_structured_array)
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
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


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
