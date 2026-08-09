"""Clone AEF 嵌入数据形状转换模块.

将 clone_aef_*.npy（普通 float32 数组，shape (128, 128, 64)）批量转换为
all_mean_*.npy 的格式：shape (128, 128) 的结构化数组，含 64 个 float64
命名字段 'A00'-'A63'（即把第三维的 64 个值按顺序打包成元组填到每个坐标点）。

转换结果与输入文件保存在同一目录，仅将文件名前缀 'clone_aef' 替换为
'xian_aef'（坐标部分不变），例如:
    clone_aef_E121.4025_N25.1947_2024.npy -> xian_aef_E121.4025_N25.1947_2024.npy

转换后的文件与 all_mean_*.npy 格式完全一致，可直接被
src.satellite_embedding_loader.load_embedding() 加载。
"""
import argparse
from pathlib import Path

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

GRID = 128
VECTOR_SIZE = 64
CLONE_PREFIX = "clone_aef"
OUT_PREFIX = "xian_aef"


def build_all_mean_dtype():
    """构造与 all_mean_*.npy 一致的结构化 dtype（'A00'-'A63' 共 64 个 float64 字段）.

    Returns:
        np.dtype: 结构化 dtype，字段名 'A00'-'A63'，每个字段为 float64.
    """
    return np.dtype([(f"A{i:02d}", np.float64) for i in range(VECTOR_SIZE)])


def convert_to_all_mean_format(arr):
    """将 (128, 128, 64) 普通数组转换为 (128, 128) 结构化数组（all_mean 格式）.

    Args:
        arr: shape (128, 128, 64) 的 numpy 数组，第三维为 64 维嵌入向量.

    Returns:
        shape (128, 128)、64 个 float64 字段 'A00'-'A63' 的结构化数组，
        每个坐标点的元组值等于原始数组该坐标处第三维的 64 个值按顺序打包.

    Raises:
        ValueError: shape 不是 (128, 128, 64).
    """
    if arr.shape != (GRID, GRID, VECTOR_SIZE):
        raise ValueError(
            f"期望 shape ({GRID}, {GRID}, {VECTOR_SIZE})，实际: {arr.shape}"
        )
    data = arr.astype(np.float64, copy=False)
    return unstructured_to_structured(data, dtype=build_all_mean_dtype())


def output_path_for(input_path):
    """计算转换输出路径：同目录、前缀 'clone_aef' → 'xian_aef'、保留 .npy.

    Args:
        input_path: 输入 .npy 文件路径.

    Returns:
        Path: 输出文件路径.
    """
    input_path = Path(input_path)
    return input_path.with_name(
        input_path.name.replace(CLONE_PREFIX, OUT_PREFIX, 1)
    )


def convert_file(input_path, overwrite=False):
    """转换单个 clone_aef .npy 文件为 all_mean 格式并保存.

    Args:
        input_path: 输入 .npy 文件路径.
        overwrite: 输出已存在时是否覆盖（默认 False，跳过）.

    Returns:
        dict: 包含 'output'、'status'（converted/skipped/failed）、'reason'.
    """
    input_path = Path(input_path)
    output_path = output_path_for(input_path)

    if output_path.exists() and not overwrite:
        return {"output": output_path, "status": "skipped", "reason": "输出已存在"}

    try:
        arr = np.load(input_path, allow_pickle=False)
        result = convert_to_all_mean_format(arr)
        np.save(output_path, result)
    except OSError as e:
        return {
            "output": output_path,
            "status": "failed",
            "reason": f"文件读写失败: {e}",
        }
    except (ValueError, TypeError) as e:
        return {"output": output_path, "status": "failed", "reason": str(e)}
    except Exception as e:  # noqa: BLE001 - 批量处理需捕获所有异常并继续
        return {
            "output": output_path,
            "status": "failed",
            "reason": f"{type(e).__name__}: {e}",
        }

    return {"output": output_path, "status": "converted", "reason": None}


def convert_dir(input_dir, overwrite=False, dry_run=False):
    """批量转换目录中的 clone_aef_*.npy 文件.

    Args:
        input_dir: 目录路径.
        overwrite: 输出已存在时是否覆盖（默认 False，跳过）.
        dry_run: 只列出将转换的文件，不写盘（默认 False）.

    Returns:
        dict: 包含 total/converted/skipped/failed 统计（dry_run 时 converted
        表示"将转换"的计划数）.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(f"{CLONE_PREFIX}_*.npy"))

    stats = {"total": len(files), "converted": 0, "skipped": 0, "failed": 0}

    for fpath in files:
        out = output_path_for(fpath)
        if dry_run:
            print(f"[计划] {fpath.name} -> {out.name}")
            stats["converted"] += 1
            continue
        result = convert_file(fpath, overwrite=overwrite)
        status = result["status"]
        stats[status] += 1
        if status == "converted":
            print(f"[转换] {fpath.name} -> {out.name}")
        elif status == "skipped":
            print(f"[跳过] {fpath.name}（输出已存在）")
        else:
            print(f"[失败] {fpath.name}: {result['reason']}")

    return stats


def main(argv=None):
    """CLI 入口：转换 clone_aef .npy 为 all_mean 格式.

    用法:
        python -m src.convert_clone_aef <file.npy>...
        python -m src.convert_clone_aef --input-dir <dir>
    """
    parser = argparse.ArgumentParser(
        description=(
            "clone_aef .npy 形状转换工具：(128, 128, 64) → all_mean 格式 "
            "(128, 128) 结构化数组，输出 xian_aef_*.npy（与输入同目录）"
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="具体 .npy 文件路径（可多个）",
    )
    parser.add_argument(
        "--input-dir",
        default="../download_scripts/output/embeddings(1)/",
        help="批量转换目录（默认: ../download_scripts/output/embeddings(1)/）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出已存在时覆盖（默认跳过）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将转换的文件，不写盘",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not args.files and not input_dir.exists():
        print(f"错误: 目录不存在: {input_dir}")
        return 1

    stats = {"total": 0, "converted": 0, "skipped": 0, "failed": 0}
    summary = []

    # 处理显式指定的文件
    for fpath_str in args.files:
        fpath = Path(fpath_str)
        if not fpath.exists():
            print(f"[失败] 文件不存在: {fpath}")
            stats["failed"] += 1
            continue
        stats["total"] += 1
        out = output_path_for(fpath)
        if args.dry_run:
            print(f"[计划] {fpath.name} -> {out.name}")
            stats["converted"] += 1
            continue
        result = convert_file(fpath, overwrite=args.overwrite)
        stats[result["status"]] += 1
        if result["status"] == "converted":
            print(f"[转换] {fpath.name} -> {out.name}")
        elif result["status"] == "skipped":
            print(f"[跳过] {fpath.name}（输出已存在）")
        else:
            print(f"[失败] {fpath.name}: {result['reason']}")

    # 批量处理目录
    if input_dir.is_dir():
        print(f"扫描目录: {input_dir}")
        dir_stats = convert_dir(
            input_dir, overwrite=args.overwrite, dry_run=args.dry_run
        )
        summary.append(f"目录 {input_dir}: {dir_stats['converted']} 转换/"
                       f"{dir_stats['skipped']} 跳过/{dir_stats['failed']} 失败"
                       f"（共 {dir_stats['total']}）")
        for key in stats:
            stats[key] += dir_stats[key]

    if summary:
        print("\n汇总:")
        for line in summary:
            print(f"  {line}")
    print(
        f"总计: {stats['total']} 个文件，"
        f"{stats['converted']} 转换，{stats['skipped']} 跳过，{stats['failed']} 失败"
    )

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
