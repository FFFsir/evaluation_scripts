"""Tests for satellite_embedding_loader module."""
import numpy as np
import pytest
from pathlib import Path
import tempfile
import os
from src.satellite_embedding_loader import load_embedding, get_summary, print_summary, load_embeddings_from_dir


def test_import():
    """验证模块可导入."""
    assert load_embedding is not None


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


def test_wrong_extension(tmp_path):
    """验证不支持的扩展名抛出 ValueError 并包含中文信息."""
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not a numpy file")
    with pytest.raises(ValueError, match="不支持的文件格式"):
        load_embedding(bad_file)


def test_non_structured_dtype(sample_plain_npy_path):
    """验证非结构化数组能幂等返回（已是 (64,128,128) 直接返回）."""
    result = load_embedding(sample_plain_npy_path)
    assert result.shape == (64, 128, 128)
    assert result.dtype == np.float64


def test_wrong_shape(wrong_shape_npy_path):
    """验证结构化数组 shape 不是 (128,128) 时抛出 ValueError."""
    with pytest.raises(ValueError, match="期望结构化 dtype shape"):
        load_embedding(wrong_shape_npy_path)


def test_npz_missing_key(sample_npz_missing_key_path):
    """验证 .npz 缺少 'embedding' 键时抛出 KeyError."""
    with pytest.raises(KeyError, match="缺少 'embedding' 键"):
        load_embedding(sample_npz_missing_key_path)


def test_idempotent(sample_plain_array):
    """验证已为标准格式的数组幂等返回."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, sample_plain_array)
        path = Path(f.name)
    try:
        result = load_embedding(path)
        assert result.shape == (64, 128, 128)
        assert result.dtype == np.float64
    finally:
        os.unlink(path)


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
    assert "Shape" in captured.out or "64" in captured.out or "Dtype" in captured.out


def test_load_from_dir_mixed_formats(sample_npy_path, sample_npz_path, temp_dir):
    """验证混合 .npy + .npz 文件的批量加载."""
    import shutil
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
