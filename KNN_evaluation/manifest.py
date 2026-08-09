"""导入 manifest：collection 级导入清单的读取/保存/增量更新（原子写）.

manifest 是可重建缓存，不是唯一真相；文件缺失/损坏时返回空结构，
由对账路径（reconcile_manifest）重建。并发写时最后一次写胜
（原子写 tmp + os.replace），下次启动对账自动纠正。
"""
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

# 遗留默认路径（无 collection 上下文时的回退；生产调用方一律走 manifest_path() 派生路径）
MANIFEST_PATH = Path("qdrant_import_manifest.json")


def safe_collection_token(collection: str) -> str:
    """将 collection 名称清洗为安全文件 token：保留 [A-Za-z0-9_.-]，其余替换为 '_'.

    防止 collection 名称含路径分隔符/特殊字符导致路径穿越或非法文件名。
    """
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(collection))


def manifest_path(collection: str) -> Path:
    """按 collection 派生的 manifest 路径：qdrant_import_manifest_<token>.json."""
    return Path(f"qdrant_import_manifest_{safe_collection_token(collection)}.json")


def _empty() -> dict:
    return {"collection": "", "images": {}, "updated_at": ""}


def load_manifest(path: Path | None = None) -> dict:
    """读取 manifest；文件缺失/损坏返回空结构（不报错）.

    path 缺省时回退 MANIFEST_PATH（遗留默认）；生产调用方应传入
    `manifest_path(collection)` 按 collection 隔离。
    """
    if path is None:
        path = MANIFEST_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty()
        return {
            "collection": str(data.get("collection", "")),
            "images": dict(data.get("images") or {}),
            "updated_at": str(data.get("updated_at", "")),
        }
    except (OSError, ValueError):
        return _empty()


def save_manifest(data: dict, path: Path | None = None) -> None:
    """原子写：tmp + os.replace."""
    if path is None:
        path = MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_manifest(
    image_id: str,
    imported_pixels: int,
    collection: str,
    path: Path | None = None,
) -> dict:
    """增量更新单张影像的已导入像素数（原子写），返回更新后的 manifest.

    path 缺省时按 collection 派生（manifest_path(collection)），
    保证不同 collection 的 manifest 互不覆盖。
    """
    if path is None:
        path = manifest_path(collection)
    data = load_manifest(path)
    images = dict(data.get("images") or {})
    images[image_id] = int(imported_pixels)
    data = {
        "collection": collection,
        "images": images,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_manifest(data, path)
    return data
