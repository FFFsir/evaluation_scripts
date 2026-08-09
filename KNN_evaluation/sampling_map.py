"""采样地图 manifest：数据库外维护 point_id→label 地图（Section 10）.

背景：单次全量 scroll 采样需下载全量向量（10.2M×64×4B ≈ 2.6GB），即使每类只需
几千个样本。本模块在数据库外维护 point→label 地图，采样时本地随机选 point_id
再按 ID 精确取向量（`client.retrieve(ids)`），只下载样本量（MB 级）。

与 `manifest.py` 导入 manifest 同模式：可重建缓存，不是唯一真相；文件缺失/损坏
返回空结构，由 `ensure_sampling_map` 自动重建；原子写（tmp + os.replace）。
"""
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime

from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.manifest import safe_collection_token

# 遗留默认路径（无 collection 上下文回退；生产调用方走 sampling_map_path() 派生）
SAMPLING_MAP_PATH = "qdrant_sampling_map.json"


def sampling_map_path(collection: str) -> str:
    """按 collection 派生的采样地图路径：qdrant_sampling_map_<token>.json."""
    return f"qdrant_sampling_map_{safe_collection_token(collection)}.json"

_SCROLL_BATCH = 50000


def _empty() -> dict:
    """缺失/损坏/空文件对应的空地图结构."""
    return {"collection": "", "total_points": 0, "updated_at": "", "by_label": {}}


def _normalize(data: dict) -> dict:
    """校验并规范化地图结构（JSON key 是字符串，还原为 int label_id）."""
    by_label = {}
    for k, v in (data.get("by_label") or {}).items():
        try:
            lid = int(k)
        except (TypeError, ValueError):
            continue
        by_label[lid] = [str(x) for x in v] if isinstance(v, list) else []
    return {
        "collection": str(data.get("collection", "")),
        "total_points": int(data.get("total_points", 0) or 0),
        "updated_at": str(data.get("updated_at", "")),
        "by_label": by_label,
    }


def load_sampling_map(path: str | os.PathLike | None = None) -> dict:
    """读取采样地图；文件缺失/损坏返回空结构（不报错）.

    path 缺省时回退 SAMPLING_MAP_PATH（遗留默认）。
    """
    if path is None:
        path = SAMPLING_MAP_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty()
        return _normalize(data)
    except (OSError, ValueError):
        return _empty()


def save_sampling_map(data: dict, path: str | os.PathLike | None = None) -> None:
    """原子写：tmp + os.replace."""
    if path is None:
        path = SAMPLING_MAP_PATH
    path = os.fspath(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
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


def build_sampling_map(manager: QdrantManager) -> dict:
    """从 Qdrant 构建采样地图：分页 scroll 只取 point_id+label，不下载向量.

    Args:
        manager: QdrantManager 实例。

    Returns:
        完整地图结构：{"collection", "total_points", "updated_at", "by_label"}。
    """
    by_label: dict[int, list[str]] = defaultdict(list)
    total = 0
    offset = None
    while True:
        result = manager.client.scroll(
            collection_name=manager.collection_name,
            scroll_filter=None,
            limit=_SCROLL_BATCH,
            offset=offset,
            with_payload=["label"],
            with_vectors=False,
        )
        # 防御 mock 返回空值或意外类型（单元测试中 MagicMock 可能返回空）
        if not result or not isinstance(result, (tuple, list)) or len(result) < 2:
            break
        records, offset = result[:2]
        if not records:
            break
        for rec in records:
            payload = rec.payload or {}
            label = int(payload.get("label", -1))
            by_label[label].append(str(rec.id))
            total += 1
        if offset is None:
            break
    return {
        "collection": manager.collection_name,
        "total_points": total,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "by_label": dict(by_label),
    }


def ensure_sampling_map(
    manager: QdrantManager,
    path: str | os.PathLike | None = None,
) -> dict:
    """自动对账：地图指纹与当前 collection 不一致时重建并保存.

    比较 manifest 的 collection 名称 + total_points 与当前 collection 信息；
    不一致 / 文件缺失 / 损坏 → 自动调用 `build_sampling_map` 重建并保存；
    一致 → 直接返回缓存。

    Args:
        manager: QdrantManager 实例。
        path: 地图文件路径。缺省时按 `manager.collection_name` 派生
            （`sampling_map_path`），不同 collection 的地图写入独立文件互不覆盖。

    Returns:
        完整地图结构。
    """
    if path is None:
        path = sampling_map_path(manager.collection_name)
    cached = load_sampling_map(path)
    info = manager.collection_info()
    current_total = int(info.get("total_points", 0) or 0)
    if (
        cached.get("collection") == manager.collection_name
        and cached.get("total_points") == current_total
    ):
        return cached
    data = build_sampling_map(manager)
    save_sampling_map(data, path)
    return data
