"""Tests for convert_clone_aef module."""
import numpy as np
import pytest
from pathlib import Path

from src.convert_clone_aef import (
    GRID,
    VECTOR_SIZE,
    build_all_mean_dtype,
    convert_dir,
    convert_file,
    convert_to_all_mean_format,
    output_path_for,
)
from src.satellite_embedding_loader import load_embedding


@pytest.fixture
def sample_clone_array():
    """创建一个 (128, 128, 64) float32 的普通数组（clone_aef 格式）."""
    rng = np.random.default_rng(42)
    return rng.uniform(-1.0, 1.0, (GRID, GRID, VECTOR_SIZE)).astype(np.float32)


@pytest.fixture
def sample_clone_npy_path(tmp_path, sample_clone_array):
    """将一个 (128, 128, 64) 数组保存为临时 clone_aef .npy 文件."""
    path = tmp_path / "clone_aef_E121.4025_N25.1947_2024.npy"
    np.save(path, sample_clone_array)
    return path


def test_build_all_mean_dtype():
    """验证结构化 dtype 含 64 个 float64 字段 'A00'-'A63'."""
    dtype = build_all_mean_dtype()
    assert len(dtype.names) == VECTOR_SIZE
    assert dtype.names[0] == "A00"
    assert dtype.names[-1] == "A63"
    for name in dtype.names:
        assert dtype[name] == np.float64


def test_convert_to_all_mean_format_shape(sample_clone_array):
    """验证转换结果为 (128, 128) 结构化数组."""
    result = convert_to_all_mean_format(sample_clone_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (GRID, GRID)
    assert result.dtype.names is not None
    assert len(result.dtype.names) == VECTOR_SIZE


def test_convert_values_preserved(sample_clone_array):
    """验证每个坐标点的 64 个值按顺序打包成元组且值保真."""
    result = convert_to_all_mean_format(sample_clone_array)
    expected = sample_clone_array.astype(np.float64)
    for i in range(VECTOR_SIZE):
        field_name = f"A{i:02d}"
        assert np.array_equal(result[field_name], expected[..., i]), field_name


def test_roundtrip_with_load_embedding(sample_clone_array):
    """验证转换结果可被 load_embedding() 读取且往返一致."""
    result = convert_to_all_mean_format(sample_clone_array)
    loaded = load_embedding_from_memory(result)
    expected = sample_clone_array.transpose(2, 0, 1).astype(np.float64)
    assert np.array_equal(loaded, expected)


def load_embedding_from_memory(structured):
    """内存版 load_embedding（转 .npy 临时文件后加载）."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, structured)
        name = f.name
    try:
        return load_embedding(name)
    finally:
        os.unlink(name)


def test_wrong_shape_raises():
    """验证非 (128, 128, 64) 的输入抛出 ValueError."""
    for bad_shape in [(64, 128, 128), (128, 128), (128, 64), (256, 256, 64)]:
        arr = np.zeros(bad_shape, dtype=np.float32)
        with pytest.raises(ValueError):
            convert_to_all_mean_format(arr)


def test_output_path_for():
    """验证输出路径：同目录、前缀替换、坐标不变、保留 .npy."""
    src = Path("D:/x/clone_aef_E121.4025_N25.1947_2024.npy")
    out = output_path_for(src)
    assert out.parent == src.parent
    assert out.name == "xian_aef_E121.4025_N25.1947_2024.npy"


def test_convert_file_ok(sample_clone_npy_path, sample_clone_array):
    """验证单个文件转换后写出 all_mean 格式文件."""
    result = convert_file(sample_clone_npy_path)
    assert result["status"] == "converted"
    out = result["output"]
    assert out.exists()
    assert out.name == "xian_aef_E121.4025_N25.1947_2024.npy"
    loaded = np.load(out, allow_pickle=False)
    assert loaded.shape == (GRID, GRID)
    assert loaded.dtype.names is not None
    assert np.array_equal(
        loaded["A00"], sample_clone_array[..., 0].astype(np.float64)
    )


def test_convert_file_skips_existing(sample_clone_npy_path):
    """验证输出已存在且未指定 overwrite 时跳过."""
    convert_file(sample_clone_npy_path)
    result = convert_file(sample_clone_npy_path)
    assert result["status"] == "skipped"


def test_convert_file_overwrite(sample_clone_npy_path):
    """验证指定 overwrite 时覆盖已存在的输出."""
    convert_file(sample_clone_npy_path)
    result = convert_file(sample_clone_npy_path, overwrite=True)
    assert result["status"] == "converted"


def test_convert_file_wrong_shape_fails(tmp_path):
    """验证 shape 不符的输入被标记为 failed 而非抛出."""
    bad = tmp_path / "clone_aef_bad.npy"
    np.save(bad, np.zeros((64, 128, 128), dtype=np.float32))
    result = convert_file(bad)
    assert result["status"] == "failed"
    assert "shape" in result["reason"]


def test_convert_dir_batch(tmp_path, sample_clone_array):
    """验证目录批量转换全部文件并正确统计."""
    for suffix in ["E121.1_N25.1_2024", "E121.2_N25.2_2024", "E121.3_N25.3_2024"]:
        np.save(tmp_path / f"clone_aef_{suffix}.npy", sample_clone_array)

    stats = convert_dir(tmp_path)
    assert stats["total"] == 3
    assert stats["converted"] == 3
    assert stats["skipped"] == 0
    assert stats["failed"] == 0
    outputs = sorted(tmp_path.glob("xian_aef_*.npy"))
    assert len(outputs) == 3


def test_convert_dir_skips_existing(tmp_path, sample_clone_array):
    """验证二次运行跳过已存在的输出."""
    np.save(tmp_path / "clone_aef_E121.1_N25.1_2024.npy", sample_clone_array)
    convert_dir(tmp_path)
    stats = convert_dir(tmp_path)
    assert stats["converted"] == 0
    assert stats["skipped"] == 1


def test_convert_dir_dry_run(tmp_path, sample_clone_array):
    """验证 dry-run 只计划不写盘."""
    np.save(tmp_path / "clone_aef_E121.1_N25.1_2024.npy", sample_clone_array)
    stats = convert_dir(tmp_path, dry_run=True)
    assert stats["converted"] == 1
    assert list(tmp_path.glob("xian_aef_*.npy")) == []
