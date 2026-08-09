"""Fixtures for KNN_evaluation integration tests."""
import os
import uuid
import numpy as np
import pytest
import subprocess
import time
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.label_mapping import LABEL_NAMES
from KNN_evaluation.sampling_map import sampling_map_path, load_sampling_map


def _qdrant_is_running() -> bool:
    """检查本地 Qdrant Docker 是否运行."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        return "qdrant" in result.stdout or "qdrant-knn-eval" in result.stdout
    except Exception:
        return False


def _start_qdrant():
    """启动 Qdrant Docker 容器."""
    subprocess.run(
        [
            "docker", "run", "-d", "--name", "qdrant-knn-eval",
            "-p", "6333:6333", "-p", "6334:6334",
            "qdrant/qdrant:latest",
        ],
        capture_output=True, timeout=30,
    )
    time.sleep(2)  # 等待启动


def _seed_test_collection(manager: QdrantManager, points_per_class: int = 6, seed: int = 42) -> None:
    """向测试 collection 写入确定性小数据集（全部标签类 × N 点）.

    `sample_queries_by_label` 需要 collection 非空才能执行分层采样；
    若仅创建空 collection，集成测试（batch/sequential 一致性等）会抛
    `Collection 为空`。此函数用固定随机种子写入少量点，保证测试可复现。
    """
    rng = np.random.default_rng(seed)
    points = []
    for label_id, label_name in LABEL_NAMES.items():
        for i in range(points_per_class):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": rng.normal(0, 1, 64).tolist(),
                "payload": {
                    "label": label_id,
                    "label_name": label_name,
                    "utm_easting": 339155.0,
                    "utm_northing": 2786000.0 + i,
                    "utm_zone": 50,
                    "image_id": "E121.4025_N25.1947",
                    "pixel_row": i,
                    "pixel_col": i,
                },
            })
    manager.client.upsert(
        collection_name=manager.collection_name,
        points=points,
    )


@pytest.fixture(scope="session")
def qdrant_manager():
    """提供连接到本地 Qdrant 的 QdrantManager.

    若 Qdrant 未运行则自动启动 Docker 容器.
    """
    if not _qdrant_is_running():
        _start_qdrant()

    manager = QdrantManager(
        url="http://localhost:6333",
        collection_name="test_pixel_embeddings",
        timeout=10,
    )

    # 确保 collection 存在
    if not manager.collection_exists():
        manager.create_collection()
        manager.create_payload_indices()

    # 集成测试需要非空 collection：无数据则写入确定性种子数据
    if manager.collection_info()["total_points"] == 0:
        _seed_test_collection(manager)

    # 清理测试 collection 对应的采样地图缓存：测试 collection 每次会话重建，
    # point_id（uuid4）全新生成，但 total_points 指纹可能相同，导致
    # ensure_sampling_map 误判缓存有效而返回过期 point_id → retrieve 取空。
    # 仅当缓存确实指向本测试 collection 时才清理，避免误删用户正式数据的地图。
    map_path = sampling_map_path(manager.collection_name)
    cached = load_sampling_map(map_path)
    if cached.get("collection") == manager.collection_name:
        try:
            os.remove(map_path)
        except OSError:
            pass

    # 同理清理全库向量磁盘缓存：total_points 指纹相同但 point_id（uuid4）已变，
    # 若缓存命中会返回过期 point_id → LOO 剔除自身失效、评估结果错误。
    # 仅当缓存文件内嵌元数据确实指向本测试 collection 时才删除，避免误删用户正式数据。
    from KNN_evaluation.corpus_cache import CORPUS_CACHE_DIR
    cache_path = CORPUS_CACHE_DIR / (
        __import__("hashlib").sha256(manager.collection_name.encode("utf-8")).hexdigest()[:16]
        + ".npz"
    )
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as d:
                cache_col = str(d["collection"].item())
        except Exception:
            cache_col = ""
        if cache_col == manager.collection_name:
            try:
                os.remove(cache_path)
            except OSError:
                pass

    yield manager

    # 清理测试 collection
    try:
        manager.client.delete_collection("test_pixel_embeddings")
    except Exception:
        pass
