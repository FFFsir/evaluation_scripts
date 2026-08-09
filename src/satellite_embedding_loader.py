"""Satellite Embedding 数据加载与格式转换模块.

将结构化 numpy dtype 存储的卫星嵌入数据从 .npy/.npz 文件转换为
标准 (64, 128, 128) float64 数组。
"""
import warnings
from pathlib import Path
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured


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

    data = structured_to_unstructured(raw)  # (128, 128, N)
    result = data.transpose(2, 0, 1)  # (N, 128, 128)
    result = result.astype(np.float64, copy=False)
    return result


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
        npz_files = []

    processed_stems = set()
    result = {}

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
