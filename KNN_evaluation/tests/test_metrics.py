import collections
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from KNN_evaluation.metrics import (
    sample_queries_by_label,
    compute_knn_accuracy,
    compute_purity_recall_curve,
    evaluate_knn,
    _resolve_tie,
)
from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation.searcher import HitRecord
from KNN_evaluation.label_mapping import LABEL_NAMES


@pytest.fixture(autouse=True)
def _isolate_corpus_cache_dir(tmp_path, monkeypatch):
    """每个测试把全库向量缓存目录隔离到临时目录.

    Section 11 引入磁盘缓存后，`_scroll_full_vectors` 会读写 `qdrant_corpus_cache/`
    npz；若用默认项目目录，测试之间/测试与本地真实缓存会互相污染（如 MagicMock
    `collection_info` 未配置时 `int(MagicMock())=1` 与残留缓存指纹巧合匹配）。
    统一 monkeypatch 为 tmp_path，保证 `_scroll_full_vectors` 相关单测隔离且不落盘污染。
    """
    import KNN_evaluation.corpus_cache as CC
    monkeypatch.setattr(CC, "CORPUS_CACHE_DIR", tmp_path)


class TestSampleQueriesByLabel:
    def test_raises_connection_error_when_unreachable(self):
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = False
        manager.url = "http://localhost:6333"
        with pytest.raises(ConnectionError):
            sample_queries_by_label(manager, samples_per_class=10)

    def test_raises_value_error_when_collection_missing(self):
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = True
        manager.collection_exists.return_value = False
        manager.collection_name = "test_collection"
        with pytest.raises(ValueError, match="不存在"):
            sample_queries_by_label(manager, samples_per_class=10)

    def test_raises_value_error_when_collection_empty(self):
        manager = MagicMock(spec=QdrantManager)
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True
        manager.collection_info.return_value = {"total_points": 0}
        with pytest.raises(ValueError, match="为空"):
            sample_queries_by_label(manager, samples_per_class=10)


class TestSampleQueriesIndexWarning:
    """timeout 修复：索引未就绪（indexed_vectors_count < total_points）时 warn_callback 被调用."""

    def _manager(self, indexed, total):
        manager = MagicMock(spec=QdrantManager)
        manager.url = "http://localhost:6333"
        manager.collection_name = "test_collection"
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True
        manager.collection_info.return_value = {
            "total_points": total,
            "vectors_count": indexed,
        }
        return manager

    def test_warns_when_index_incomplete(self, monkeypatch):
        """indexed < total：warn_callback 收到「向量索引构建中」提示，采样继续."""
        import KNN_evaluation.metrics as M
        manager = self._manager(indexed=100, total=200)
        manager.client.retrieve.return_value = []
        monkeypatch.setattr(M, "ensure_sampling_map", lambda mgr, path=None: {
            "collection": "test_collection", "total_points": 200,
            "updated_at": "x", "by_label": {0: ["pt-0"]},
        })
        warned: list[str] = []
        sample_queries_by_label(
            manager, samples_per_class=1, seed=42, warn_callback=warned.append,
        )
        assert warned, "索引未就绪时必须触发警告"
        assert "向量索引构建中" in warned[0]
        assert "100" in warned[0] and "200" in warned[0]

    def test_no_warn_when_index_ready(self, monkeypatch):
        """indexed == total：不触发警告."""
        import KNN_evaluation.metrics as M
        manager = self._manager(indexed=200, total=200)
        manager.client.retrieve.return_value = []
        monkeypatch.setattr(M, "ensure_sampling_map", lambda mgr, path=None: {
            "collection": "test_collection", "total_points": 200,
            "updated_at": "x", "by_label": {0: ["pt-0"]},
        })
        warned: list[str] = []
        sample_queries_by_label(
            manager, samples_per_class=1, seed=42, warn_callback=warned.append,
        )
        assert warned == []

    def test_no_warn_when_total_zero(self, monkeypatch):
        """total_points=0 已先抛「Collection 为空」，不触发索引警告."""
        manager = self._manager(indexed=0, total=0)
        with pytest.raises(ValueError, match="为空"):
            sample_queries_by_label(manager, samples_per_class=10)


def _retrieve_record(point_id: str, label: int, image_id: str, row: int, col: int, idx: int = 0):
    """构造 retrieve 返回的 mock point：含 payload（image_id/pixel_row/pixel_col）与向量."""
    rec = MagicMock()
    rec.id = point_id
    rec.vector = [float(idx)] * 64
    rec.payload = {
        "label": label,
        "image_id": image_id,
        "pixel_row": row,
        "pixel_col": col,
    }
    return rec


def _map_sampling_manager(monkeypatch, by_label: dict, samples_per_class: int, seed: int,
                          total_points: int | None = None):
    """构造采样地图 + retrieve mock 的采样环境.

    在 metrics 命名空间 mock `ensure_sampling_map` 返回给定地图；`client.retrieve`
    记录收到的 ids 并按 ids 返回带 payload 的 mock point。返回 (manager, sampled_ids)。
    """
    import KNN_evaluation.metrics as M

    total_points = total_points if total_points is not None else sum(len(v) for v in by_label.values())
    manager = MagicMock(spec=QdrantManager)
    manager.url = "http://localhost:6333"
    manager.collection_name = "test_collection"
    manager.health_check.return_value = True
    manager.collection_exists.return_value = True
    manager.collection_info.return_value = {"total_points": total_points}

    pid_to_label = {pid: lid for lid, ids in by_label.items() for pid in ids}
    sampled_ids: list[str] = []

    def fake_retrieve(**kwargs):
        ids = list(kwargs.get("ids", []))
        sampled_ids.extend(ids)
        return [
            _retrieve_record(pid, pid_to_label[pid], f"img-{pid}", 0, 0)
            for pid in ids if pid in pid_to_label
        ]

    manager.client.retrieve.side_effect = fake_retrieve
    monkeypatch.setattr(M, "ensure_sampling_map", lambda mgr, path=None: {
        "collection": "test_collection",
        "total_points": total_points,
        "updated_at": "x",
        "by_label": dict(by_label),
    })
    return manager, sampled_ids


class TestSampleQueriesUsingMap:
    """Section 10.4: sample_queries_by_label 改用采样地图 + retrieve 精确取样本.

    读地图（ensure_sampling_map）→ 每类 `rng.sample` 选 point_id →
    `client.retrieve(ids)` 批量取向量。只下载样本量，不再全量 scroll 下载全向量；
    保留分层语义、seed 可复现、空类标记、返回结构兼容（payload 带真实影像坐标）。
    """

    def _build_by_label(self):
        """9 类中 label 0/1 各有 8 条，其余类无像素."""
        return {
            0: [f"pt-0-{i}" for i in range(8)],
            1: [f"pt-1-{i}" for i in range(8)],
        }

    def test_uses_retrieve_only_samples_no_full_download(self, monkeypatch):
        """采样通过 retrieve 只取样本量；不触发全量向量 scroll."""
        manager, sampled_ids = _map_sampling_manager(
            monkeypatch, self._build_by_label(), samples_per_class=3, seed=42)
        sample_queries_by_label(manager, samples_per_class=3, seed=42)
        # retrieve 只收到每类 3 个 id（共 6），不是全量 16
        assert len(sampled_ids) == 6
        # 不得触发带向量的全量 scroll（地图已缓存，无需重建）
        assert not manager.client.scroll.called, "不得再全量 scroll 下载向量"
        # retrieve 必须 with_vectors=True 取向量、with_payload 取影像坐标
        for call in manager.client.retrieve.call_args_list:
            assert call.kwargs.get("with_vectors") is True

    def test_retrieve_ids_match_selected_per_class(self, monkeypatch):
        """retrieve 收到的 ids 是每类 rng.sample 选中的样本量，且来自地图内."""
        manager, sampled_ids = _map_sampling_manager(
            monkeypatch, self._build_by_label(), samples_per_class=4, seed=7)
        queries = sample_queries_by_label(manager, samples_per_class=4, seed=7)
        # 地图中不存在的 id 不会出现在 retrieve 中
        assert len(sampled_ids) == 8
        assert all(pid in self._build_by_label()[0] + self._build_by_label()[1]
                   for pid in sampled_ids)
        # 返回的 point_id 与 retrieve 取回的一致
        assert {q["point_id"] for q in queries if "point_id" in q} == set(sampled_ids)

    def test_stratified_per_class_counts(self, monkeypatch):
        """分层语义保留：每类采样 samples_per_class 个，跨类去重且无空标记."""
        manager, _ = _map_sampling_manager(
            monkeypatch, self._build_by_label(), samples_per_class=3, seed=7)
        queries = sample_queries_by_label(manager, samples_per_class=3, seed=7)
        by_label = collections.defaultdict(list)
        for q in queries:
            if "point_id" not in q:
                continue
            by_label[q["label"]].append(q)
        assert len(by_label) == 2
        assert len(by_label[0]) == 3
        assert len(by_label[1]) == 3
        point_ids = [q["point_id"] for q in queries if "point_id" in q]
        assert len(set(point_ids)) == len(point_ids), "采样不得重复"

    def test_seed_reproducible(self, monkeypatch):
        """相同 seed 完全可复现；不同 seed 序列不同."""
        by_label = self._build_by_label()

        def real_ids(seed):
            manager, _ = _map_sampling_manager(
                monkeypatch, by_label, samples_per_class=3, seed=seed)
            return [q["point_id"] for q in sample_queries_by_label(
                manager, samples_per_class=3, seed=seed) if "point_id" in q]

        a = real_ids(42)
        b = real_ids(42)
        c = real_ids(1)
        assert a == b
        assert a != c

    def test_record_structure_fields_from_payload(self, monkeypatch):
        """返回结构兼容：vector float64 64 维 + label/label_name + payload 真实影像坐标 + point_id + actual_count.

        retrieve 返回的 point 含 payload（image_id/pixel_row/pixel_col）与 vector；
        构造返回结构时从 payload 读影像坐标，vector 转 float64。
        """
        manager, _ = _map_sampling_manager(
            monkeypatch, self._build_by_label(), samples_per_class=2, seed=3)
        queries = sample_queries_by_label(manager, samples_per_class=2, seed=3)
        for q in queries:
            if "point_id" not in q:
                continue  # 空类标记无 vector 字段
            assert q["vector"].dtype == np.float64
            assert q["vector"].shape == (64,)
            assert q["label"] in (0, 1)
            assert q["label_name"] == LABEL_NAMES[q["label"]]
            assert q["image_id"] == f"img-{q['point_id']}"  # payload 真实值
            assert isinstance(q["pixel_row"], int)
            assert isinstance(q["pixel_col"], int)
            assert q["point_id"].startswith("pt-")
            assert q["actual_count"] == 2

    def test_empty_class_marker_kept(self, monkeypatch):
        """某类在地图中无 ID：保留 {"label", "label_name", "actual_count": 0, "vectors": []} 空标记."""
        manager, _ = _map_sampling_manager(
            monkeypatch, self._build_by_label(), samples_per_class=3, seed=42)
        queries = sample_queries_by_label(manager, samples_per_class=3, seed=42)
        real = [q for q in queries if "point_id" in q]
        empty = [q for q in queries if "point_id" not in q]
        assert len(real) == 6
        assert len(empty) == 7
        for e in empty:
            assert e["actual_count"] == 0
            assert e["vectors"] == []
            assert e["label"] in LABEL_NAMES
            assert e["label_name"] == LABEL_NAMES[e["label"]]

    def test_capped_by_available_count(self, monkeypatch):
        """每类不足 samples_per_class 时 actual_count 取该类实际数量."""
        manager, sampled_ids = _map_sampling_manager(
            monkeypatch, {0: ["pt-0-0", "pt-0-1"]}, samples_per_class=5, seed=1)
        queries = sample_queries_by_label(manager, samples_per_class=5, seed=1)
        class0 = [q for q in queries if q["label"] == 0]
        assert len(class0) == 2
        assert all(q["actual_count"] == 2 for q in class0)
        assert len(sampled_ids) == 2

    def test_map_missing_build_failure_raises(self, monkeypatch):
        """地图缺失且构建失败（ensure 返回空 by_label 但 collection 非空）→ 抛明确错误."""
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.url = "http://localhost:6333"
        manager.collection_name = "test_collection"
        manager.health_check.return_value = True
        manager.collection_exists.return_value = True
        manager.collection_info.return_value = {"total_points": 10}
        # ensure 返回空地图（collection 匹配但 by_label 全空 → 构建失败）
        monkeypatch.setattr(M, "ensure_sampling_map", lambda mgr, path=None: {
            "collection": "test_collection", "total_points": 10,
            "updated_at": "x", "by_label": {},
        })
        with pytest.raises(RuntimeError, match="采样地图"):
            sample_queries_by_label(manager, samples_per_class=3, seed=1)

    def test_matches_old_scroll_selection_order(self, monkeypatch):
        """与旧采样结果一致：地图点序与旧 scroll 顺序相同时，选中点 ID 完全一致.

        旧路径（Section 9.3）按 scroll 返回顺序分组记录后 `rng.sample`；新路径按
        地图 `by_label`（同样按 scroll 顺序累积的 point_id）`rng.sample`。给定相同
        数据顺序与 seed，两路径选中同一批 point_id（分层/seed 可复现的一致性）。
        """
        import random
        records_order = [0] * 8 + [1] * 8  # label 0 的 8 条在前，label 1 的 8 条在后
        by_label = {0: [f"pt-0-{i}" for i in range(8)], 1: [f"pt-1-{i}" for i in range(8)]}
        # 旧路径：按 records_order 顺序构造记录 → 按 label 分组 → rng.sample
        old_selected: dict[int, list[str]] = {}
        rng = random.Random(42)
        grouped: dict[int, list[str]] = collections.defaultdict(list)
        for lid in records_order:
            grouped[lid].append(f"pt-{lid}-{len(grouped[lid])}")
        for lid in (0, 1):
            old_selected[lid] = rng.sample(grouped[lid], 3)
        # 新路径：地图 by_label 顺序相同 + 同一 rng 推进 → 选中点 ID 集合一致
        manager, sampled_ids = _map_sampling_manager(
            monkeypatch, by_label, samples_per_class=3, seed=42)
        sample_queries_by_label(manager, samples_per_class=3, seed=42)
        new_selected = collections.defaultdict(list)
        for pid in sampled_ids:
            new_selected[int(pid.split("-")[1])].append(pid)
        # 每类选中点 ID 集合与旧路径完全一致（分层/seed 可复现的一致性）
        assert set(new_selected[0]) == set(old_selected[0])
        assert set(new_selected[1]) == set(old_selected[1])
        # rng.sample 输出顺序一致（retrieve 按 ids 顺序返回，与旧路径逐项一致）
        assert sampled_ids == old_selected[0] + old_selected[1]


class TestComputeKnnAccuracy:
    def test_all_neighbors_same_label_gives_accuracy_one(self):
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "image_id": "img1", "pixel_row": 0, "pixel_col": 0,
             "point_id": "uuid-self", "actual_count": 10},
        ]

        # 构造 mock hits: 全部与查询同 label
        mock_hits = [
            HitRecord(id="uuid-1", score=0.95, label=0, label_name=LABEL_NAMES[0],
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img1", pixel_row=1, pixel_col=1),
        ] * 10

        mock_result = MagicMock()
        mock_result.hits = mock_hits

        with patch("KNN_evaluation.metrics.PixelSearcher") as mock_searcher_cls:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = mock_result
            mock_searcher_cls.return_value = mock_searcher

            result = compute_knn_accuracy(manager, queries, k=3)

        assert result["overall_accuracy"] == 1.0
        assert result["per_class_metrics"][LABEL_NAMES[0]]["f1"] == 1.0
        assert result["k"] == 3
        assert result["num_queries"] == 1

    def test_tie_break_decrement(self):
        """验证 _resolve_tie 递减 K 打破平票。"""
        votes = [0, 0, 1, 1, 2, 2, 0, 1, 2, 3]
        hits = [
            HitRecord(id=f"uuid-{i}", score=1.0 - i * 0.01,
                      label=votes[i], label_name="test",
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img", pixel_row=i, pixel_col=0)
            for i in range(10)
        ]
        winner = _resolve_tie(votes, hits)
        assert winner == 0  # K-1=9: 3 votes for 0, 可打破


class TestComputePurityRecallCurve:
    def test_loo_excludes_self(self):
        """验证 Leave-One-Out 剔除自身。"""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        # 模拟 count 返回每个 label 10000 个全局像素
        manager.client.count.return_value.count = 10000

        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "image_id": "img1", "pixel_row": 0, "pixel_col": 0,
             "point_id": "uuid-self", "actual_count": 10},
        ]

        # 构造 hits: 第一个就是自身
        mock_hits = [
            HitRecord(id="uuid-self", score=1.0, label=0, label_name=LABEL_NAMES[0],
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img1", pixel_row=0, pixel_col=0),
            HitRecord(id="uuid-1", score=0.99, label=1, label_name=LABEL_NAMES[1],
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img1", pixel_row=1, pixel_col=1),
        ] * 50  # 足够覆盖 max(k_values)

        mock_result = MagicMock()
        mock_result.hits = mock_hits

        with patch("KNN_evaluation.metrics.PixelSearcher") as mock_sc:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = mock_result
            mock_sc.return_value = mock_searcher

            result = compute_purity_recall_curve(
                manager, queries, k_values=[5, 10],
            )

        # 自身被剔除，所有 neighbor 都是 trees(label=1)
        # Purity@5: 0/5 = 0，非 1.0
        assert result["global_purity"][0] == 0.0

    def test_purity_decreases_with_k(self):
        """验证 Purity(K1) >= Purity(K2) when K1 < K2. (Purity 应随 K 增大单调递减)"""
        # 使用 mock 构造 mixed neighbors
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 10000

        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "image_id": "img1", "pixel_row": 0, "pixel_col": 0,
             "point_id": "uuid-self", "actual_count": 10},
        ]

        # 前 5 个 neighbor 与查询同 label，后面全是不同 label
        mock_hits = []
        for i in range(5):
            mock_hits.append(HitRecord(
                id=f"uuid-{i}", score=1.0 - i * 0.01, label=0, label_name=LABEL_NAMES[0],
                utm_easting=0, utm_northing=0, utm_zone=51,
                image_id="img1", pixel_row=i, pixel_col=0,
            ))
        for i in range(5, 100):
            mock_hits.append(HitRecord(
                id=f"uuid-{i}", score=1.0 - i * 0.01, label=1, label_name=LABEL_NAMES[1],
                utm_easting=0, utm_northing=0, utm_zone=51,
                image_id="img1", pixel_row=i, pixel_col=0,
            ))

        mock_result = MagicMock()
        mock_result.hits = mock_hits

        with patch("KNN_evaluation.metrics.PixelSearcher") as mock_sc:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = mock_result
            mock_sc.return_value = mock_searcher

            result = compute_purity_recall_curve(
                manager, queries, k_values=[5, 10],
            )

        # Purity@5: 5/5=1.0, Purity@10: 5/10=0.5
        assert result["global_purity"][0] >= result["global_purity"][1]

    def test_empty_k_values_raises(self):
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        with pytest.raises(ValueError, match="不能为空"):
            compute_purity_recall_curve(manager, [], k_values=[])


class TestPurityRecallGpuPath:
    """F2 GPU 分块路径（mock KnnEngine 返回固定 Top-K，验证 Purity/Recall 累加数学）."""

    def _fake_engine(self, chunk_results):
        """构造按块返回固定 Top-K 的 mock KnnEngine（每块 (scores, labels, ids)）."""
        from unittest.mock import MagicMock
        engine = MagicMock()
        engine.estimate_block_q.return_value = 2  # 每块 2 行，强制多块
        engine.knn_chunk.side_effect = chunk_results
        return engine

    def test_purity_recall_accumulation_with_loo(self, monkeypatch):
        """F2 分块路径：LOO 剔除自身 + Purity/Recall 累加数学 + 每类累加器 + 分母用 label_totals."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-6（max_k=5 → knn_chunk k=6）：首个为自身 → LOO 剔除
        # 自身(q0, label=0)不剔除时 k=2 的 [0,0] 会使 purity=1.0，与剔除后 0.5 区分开
        topk_labels = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int64)
        topk_ids = np.array([["q0", "n1", "n2", "n3", "n4", "n5"]])

        fake = self._fake_engine([
            (np.zeros((1, 6)), topk_labels, topk_ids),
        ])
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["q0"]),
        ))
        # 类别 0 全局同类总数 100（Recall@K 分母）
        monkeypatch.setattr(
            M, "_compute_per_class_label_totals",
            lambda mgr: {LABEL_NAMES[0]: 100},
        )

        result = M.compute_purity_recall_curve(
            manager, queries, k_values=[2, 5], exact=True, device="cpu",
        )

        # LOO 后 effective_labels = [0,1,0,1,1]（自身 q0 被剔除）
        # k=2: [0,1] → 同类 1 → purity 0.5, recall 1/100=0.01
        # k=5: [0,1,0,1,1] → 同类 2 → purity 0.4, recall 2/100=0.02
        assert result["k_values"] == [2, 5]
        assert result["global_purity"] == [0.5, 0.4]
        assert result["global_recall"] == [0.01, 0.02]
        assert result["num_queries"] == 1
        # 每类累加器
        assert result["per_class_purity"][LABEL_NAMES[0]] == [0.5, 0.4]
        assert result["per_class_recall"][LABEL_NAMES[0]] == [0.01, 0.02]
        # 其他类无样本 → 0
        assert result["per_class_purity"][LABEL_NAMES[1]] == [0.0, 0.0]
        assert result["per_class_recall"][LABEL_NAMES[1]] == [0.0, 0.0]

    def test_block_split_global_average_across_classes(self, monkeypatch):
        """F2 分块路径：block_q 强制多块，全局 Purity/Recall = 各查询均值，跨类累加器正确."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        # 3 个查询（3 类），block_q=2 → 两块 [q0,q1] + [q2]
        queries = [
            {"vector": np.ones(64), "label": i, "label_name": LABEL_NAMES[i],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(3)
        ]
        # max_k=2 → knn_chunk k=3；邻居不含自身（id 均非 q0/q1/q2）
        topk_labels = np.array([
            [0, 0, 0],   # q0 全 0 → k=2 purity 1.0
            [1, 1, 1],   # q1 全 1 → k=2 purity 1.0
            [2, 0, 0],   # q2 [2,0] → k=2 purity 0.5
        ], dtype=np.int64)
        topk_ids = np.array([
            ["n00", "n01", "n02"],
            ["n10", "n11", "n12"],
            ["n20", "n21", "n22"],
        ])

        fake = self._fake_engine([
            (np.zeros((2, 3)), topk_labels[:2], topk_ids[:2]),
            (np.zeros((1, 3)), topk_labels[2:], topk_ids[2:]),
        ])
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((3, 64), dtype=np.float32), np.array([0, 1, 2]),
            np.array(["q0", "q1", "q2"]),
        ))
        totals = {LABEL_NAMES[i]: 100 for i in range(3)}
        monkeypatch.setattr(M, "_compute_per_class_label_totals", lambda mgr: totals)

        result = M.compute_purity_recall_curve(
            manager, queries, k_values=[2], exact=True, device="cpu",
        )

        # global purity = (1.0 + 1.0 + 0.5)/3 = 0.8333
        assert result["k_values"] == [2]
        assert result["global_purity"] == [0.8333]
        # global recall = (2/100 + 2/100 + 1/100)/3 = 0.016667
        assert result["global_recall"] == [0.016667]
        assert result["num_queries"] == 3
        # 每类累加器
        assert result["per_class_purity"][LABEL_NAMES[0]] == [1.0]
        assert result["per_class_purity"][LABEL_NAMES[1]] == [1.0]
        assert result["per_class_purity"][LABEL_NAMES[2]] == [0.5]
        assert result["per_class_recall"][LABEL_NAMES[0]] == [0.02]
        assert result["per_class_recall"][LABEL_NAMES[2]] == [0.01]

    def test_recall_uses_global_totals_denominator(self, monkeypatch):
        """Recall@K 用全量同类总数作分母（非 min(K, N_same)）."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 10000  # 每类全局 10000

        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-3 全同类 label=0 → Purity@k=1.0；Recall 按 K 递增累积
        topk_labels = np.array([[0, 0, 0]], dtype=np.int64)
        topk_ids = np.array([["n1", "n2", "n3"]])

        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (np.zeros((1, 3)), topk_labels, topk_ids)

        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n1"]),
        ))

        result = M.compute_purity_recall_curve(
            manager, queries, k_values=[1, 2, 3], exact=True, device="cpu",
        )
        assert result["global_purity"] == [1.0, 1.0, 1.0]
        assert result["global_recall"] == [0.0001, 0.0002, 0.0003]  # 1/10000, 2/10000, 3/10000
        assert result["num_queries"] == 1

    def test_loo_excludes_self_f2(self, monkeypatch):
        """F2 GPU 路径同样剔除自身."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 10000

        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-3: [自身(label=0), 1, 1] → 剔除自身后 [1,1] → Purity@2=0
        topk_labels = np.array([[0, 1, 1]], dtype=np.int64)
        topk_ids = np.array([["q0", "n1", "n2"]])

        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (np.zeros((1, 3)), topk_labels, topk_ids)

        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["q0"]),
        ))

        result = M.compute_purity_recall_curve(
            manager, queries, k_values=[2], exact=True, device="cpu",
        )
        assert result["global_purity"] == [0.0]

    def test_parallel_accumulation_matches_serial(self, monkeypatch):
        """ThreadPoolExecutor 并行行累加与串行结果一致."""
        from unittest.mock import MagicMock
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 1000

        n_q = 8
        queries = [
            {"vector": np.ones(64), "label": i % 2, "label_name": LABEL_NAMES[i % 2],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(n_q)
        ]
        topk_labels = np.array([[i % 2] * 5 for i in range(n_q)], dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(n_q)])

        fake = MagicMock()
        fake.estimate_block_q.return_value = n_q
        fake.knn_chunk.return_value = (np.zeros((n_q, 5)), topk_labels, topk_ids)

        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((n_q, 64), dtype=np.float32),
            np.array([i % 2 for i in range(n_q)]),
            np.array([f"q{i}n0" for i in range(n_q)]),
        ))

        res = M.compute_purity_recall_curve(
            manager, queries, k_values=[3, 5], exact=True, device="cpu",
        )
        # 每行邻居全同类 → Purity@3 = Purity@5 = 1.0；Recall 按 K 累积 3/1000, 5/1000
        assert res["global_purity"] == [1.0, 1.0]
        assert res["global_recall"] == [0.003, 0.005]
        assert res["num_queries"] == n_q

    def test_parallel_uses_thread_pool_executor(self, monkeypatch):
        """分块多行时通过 ThreadPoolExecutor 并行提交行累加任务."""
        from unittest.mock import MagicMock
        from concurrent.futures import ThreadPoolExecutor
        import KNN_evaluation.metrics as M

        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 1000

        n_q = 8
        queries = [
            {"vector": np.ones(64), "label": i % 2, "label_name": LABEL_NAMES[i % 2],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(n_q)
        ]
        topk_labels = np.array([[i % 2] * 5 for i in range(n_q)], dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(n_q)])

        fake = MagicMock()
        fake.estimate_block_q.return_value = n_q
        fake.knn_chunk.return_value = (np.zeros((n_q, 5)), topk_labels, topk_ids)

        # 记录 ThreadPoolExecutor.submit 调用，确认并行路径真正被触发
        submit_calls: list = []
        real_submit = ThreadPoolExecutor.submit

        class RecordingExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                submit_calls.append(fn.__name__)
                return real_submit(self, fn, *args, **kwargs)

        monkeypatch.setattr(M, "ThreadPoolExecutor", RecordingExecutor)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((n_q, 64), dtype=np.float32),
            np.array([i % 2 for i in range(n_q)]),
            np.array([f"q{i}n0" for i in range(n_q)]),
        ))

        res = M.compute_purity_recall_curve(
            manager, queries, k_values=[3, 5], exact=True, device="cpu",
        )
        # 每行提交一个 _accumulate_purity_recall_row 任务
        assert len(submit_calls) == n_q
        assert all(name == "_accumulate_purity_recall_row" for name in submit_calls)
        assert res["num_queries"] == n_q

    def test_parallel_rows_use_independent_accumulators(self, monkeypatch):
        """并行每行使用独立累加器（避免共享 dict 并发 += 丢失更新）."""
        import threading
        from unittest.mock import MagicMock
        import KNN_evaluation.metrics as M

        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 1000

        n_q = 8
        queries = [
            {"vector": np.ones(64), "label": i % 2, "label_name": LABEL_NAMES[i % 2],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(n_q)
        ]
        topk_labels = np.array([[i % 2] * 5 for i in range(n_q)], dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(n_q)])

        fake = MagicMock()
        fake.estimate_block_q.return_value = n_q
        fake.knn_chunk.return_value = (np.zeros((n_q, 5)), topk_labels, topk_ids)

        # 记录每个任务收到的 purity_sums 对象 id：共享实现下全部相同
        seen: list = []
        lock = threading.Lock()
        real_helper = M._accumulate_purity_recall_row

        def recording_helper(*args, **kwargs):
            with lock:
                seen.append(id(args[6]))  # args[6] = purity_sums（第 7 个位置参数）
            return real_helper(*args, **kwargs)

        monkeypatch.setattr(M, "_accumulate_purity_recall_row", recording_helper)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((n_q, 64), dtype=np.float32),
            np.array([i % 2 for i in range(n_q)]),
            np.array([f"q{i}n0" for i in range(n_q)]),
        ))

        res = M.compute_purity_recall_curve(
            manager, queries, k_values=[3, 5], exact=True, device="cpu",
        )
        # 每行独立累加器 → n_q 个不同 id；若共享同一 dict 并发写会丢失更新
        assert len(seen) == n_q
        assert len(set(seen)) == n_q
        assert res["num_queries"] == n_q

    def test_accumulate_purity_recall_row_helper(self):
        """模块级助手单行累加：LOO 剔除自身 + 累加器写入 + 返回计入标志."""
        from KNN_evaluation.metrics import _accumulate_purity_recall_row
        sorted_k = [1, 2, 3]
        purity_sums = {k: 0.0 for k in sorted_k}
        recall_sums = {k: 0.0 for k in sorted_k}
        per_class_purity_sums = {ln: {k: 0.0 for k in sorted_k} for ln in LABEL_NAMES.values()}
        per_class_recall_sums = {ln: {k: 0.0 for k in sorted_k} for ln in LABEL_NAMES.values()}
        class_counts = {ln: 0 for ln in LABEL_NAMES.values()}
        topk_labels = np.array([0, 0, 1, 1], dtype=np.int64)
        topk_ids = np.array(["q0", "n1", "n2", "n3"])
        ok = _accumulate_purity_recall_row(
            topk_labels, topk_ids, "q0", LABEL_NAMES[0], sorted_k,
            {LABEL_NAMES[0]: 100, LABEL_NAMES[1]: 100},
            purity_sums, recall_sums, per_class_purity_sums, per_class_recall_sums,
            class_counts,
        )
        assert ok is True
        assert class_counts[LABEL_NAMES[0]] == 1
        # LOO 剔除 q0 → effective [0,1,1]
        # k=1: [0] → same=1 → purity 1.0, recall 1/100
        assert purity_sums[1] == 1.0
        assert recall_sums[1] == 0.01
        # k=2: [0,1] → same=1 → purity 0.5, recall 1/100
        assert purity_sums[2] == 0.5
        assert recall_sums[2] == 0.01
        # k=3: [0,1,1] → same=1 → purity 1/3, recall 1/100
        assert abs(purity_sums[3] - 1.0 / 3.0) < 1e-9
        assert recall_sums[3] == 0.01
        # 每类累加器也写入
        assert per_class_purity_sums[LABEL_NAMES[0]][2] == 0.5
        assert per_class_recall_sums[LABEL_NAMES[0]][2] == 0.01

    def test_accumulate_purity_recall_row_empty_returns_false(self):
        """无有效邻居（全部为自身）时不写入累加器并返回 False."""
        from KNN_evaluation.metrics import _accumulate_purity_recall_row
        sorted_k = [1, 2, 3]
        purity_sums = {k: 0.0 for k in sorted_k}
        recall_sums = {k: 0.0 for k in sorted_k}
        per_class_purity_sums = {ln: {k: 0.0 for k in sorted_k} for ln in LABEL_NAMES.values()}
        per_class_recall_sums = {ln: {k: 0.0 for k in sorted_k} for ln in LABEL_NAMES.values()}
        class_counts = {ln: 0 for ln in LABEL_NAMES.values()}
        topk_labels = np.array([0], dtype=np.int64)
        topk_ids = np.array(["q0"])
        ok = _accumulate_purity_recall_row(
            topk_labels, topk_ids, "q0", LABEL_NAMES[0], sorted_k,
            {LABEL_NAMES[0]: 100}, purity_sums, recall_sums,
            per_class_purity_sums, per_class_recall_sums, class_counts,
        )
        assert ok is False
        assert purity_sums == {k: 0.0 for k in sorted_k}
        assert class_counts[LABEL_NAMES[0]] == 0


class TestScrollFullVectors:
    """_scroll_full_vectors 单元测试 (RED: 实现前应失败)."""

    def test_scrolls_all_vectors(self):
        """单页 scroll 应返回所有向量、标签和 point_id。"""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"

        # 构造 mock records
        mock_records = []
        for i in range(5):
            rec = MagicMock()
            rec.vector = [float(i)] * 64
            rec.payload = {"label": i}
            rec.id = f"point-{i}"
            mock_records.append(rec)

        # scroll 第一次返回数据，第二次返回空
        manager.client.scroll.side_effect = [
            (mock_records, None),
            ([], None),
        ]

        from KNN_evaluation.metrics import _scroll_full_vectors

        vectors, labels, point_ids = _scroll_full_vectors(manager)

        assert vectors.shape == (5, 64)
        assert vectors.dtype == np.float32
        assert labels.shape == (5,)
        assert labels.dtype == np.int64
        assert point_ids.shape == (5,)
        assert list(labels) == [0, 1, 2, 3, 4]
        assert list(point_ids) == ["point-0", "point-1", "point-2", "point-3", "point-4"]

    def test_pagination_multiple_pages(self):
        """多页 scroll 应正确拼接所有页。"""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"

        page1 = []
        for i in range(3):
            rec = MagicMock()
            rec.vector = [float(i)] * 64
            rec.payload = {"label": i}
            rec.id = f"p1-{i}"
            page1.append(rec)

        page2 = []
        for i in range(2):
            rec = MagicMock()
            rec.vector = [float(i + 3)] * 64
            rec.payload = {"label": i + 3}
            rec.id = f"p2-{i}"
            page2.append(rec)

        manager.client.scroll.side_effect = [
            (page1, "offset-1"),
            (page2, None),
            ([], None),
        ]

        from KNN_evaluation.metrics import _scroll_full_vectors

        vectors, labels, point_ids = _scroll_full_vectors(manager)

        assert vectors.shape == (5, 64)
        assert list(labels) == [0, 1, 2, 3, 4]
        assert list(point_ids) == ["p1-0", "p1-1", "p1-2", "p2-0", "p2-1"]

    def test_empty_collection_returns_empty_arrays(self):
        """空 collection 应返回空数组。"""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.scroll.return_value = ([], None)

        from KNN_evaluation.metrics import _scroll_full_vectors

        vectors, labels, point_ids = _scroll_full_vectors(manager)

        assert vectors.shape == (0, 64)
        assert labels.shape == (0,)
        assert point_ids.shape == (0,)


class TestCorpusCacheLazyBuild:
    """Section 11: _scroll_full_vectors 惰性磁盘缓存.

    首次下载后写入缓存（mock scroll 计数 1）；二次调用读缓存（mock scroll 计数 0）；
    collection 指纹变化（total_points 不同）触发重建。缓存目录 monkeypatch 为
    tmp_path，避免污染项目工作区。
    """

    @staticmethod
    def _records(n):
        """构造 n 条带 label + 64 维向量的 mock scroll record."""
        out = []
        for i in range(n):
            rec = MagicMock()
            rec.vector = [float(i)] * 64
            rec.payload = {"label": i}
            rec.id = f"pt-{i}"
            out.append(rec)
        return out

    def test_first_call_builds_second_loads(self, tmp_path, monkeypatch):
        """首次下载后写入缓存（scroll 计数 1）；二次调用读缓存（scroll 计数 0）."""
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.collection_info.return_value = {"total_points": 3}

        records = self._records(3)
        manager.client.scroll.side_effect = [(records, None)]

        v1, l1, ids1 = M._scroll_full_vectors(manager)
        assert manager.client.scroll.call_count == 1, "首次应下载一次"
        assert v1.shape == (3, 64)

        # 二次调用：缓存命中，不再下载（scroll 计数仍为 1）
        manager.client.scroll.side_effect = [(records, None)]  # 防御：若误触不抛 StopIteration
        v2, l2, ids2 = M._scroll_full_vectors(manager)
        assert manager.client.scroll.call_count == 1, "二次调用应读缓存，scroll 计数为 0（新增）"
        assert np.array_equal(v1, v2)
        assert list(l1) == list(l2)
        assert list(ids1) == list(ids2)

    def test_fingerprint_change_triggers_rebuild(self, tmp_path, monkeypatch):
        """collection total_points 变化 → 缓存指纹不符 → 重新下载构建."""
        import KNN_evaluation.metrics as M
        from KNN_evaluation.corpus_cache import save_corpus_cache
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.collection_info.return_value = {"total_points": 5}

        # 预置 3 行的陈旧缓存（total_points 指纹 3，与当前 5 不符）
        stale_v = np.arange(3 * 64, dtype=np.float32).reshape(3, 64)
        stale_l = np.array([0, 1, 2], dtype=np.int64)
        stale_ids = np.array(["old-0", "old-1", "old-2"])
        save_corpus_cache("test_collection", stale_v, stale_l, stale_ids, tmp_path)

        records = self._records(5)
        manager.client.scroll.side_effect = [(records, None)]

        v, l, ids = M._scroll_full_vectors(manager)
        assert v.shape == (5, 64)
        assert list(ids) == ["pt-0", "pt-1", "pt-2", "pt-3", "pt-4"]
        assert manager.client.scroll.call_count == 1, "指纹变化应触发重建（重新下载）"


class TestKnnAccuracyGpuPath:
    """GPU 分块路径（device=cpu 跑 torch 分块，逻辑与 GPU 一致）。"""

    def _fake_engine(self, topk_labels, topk_ids):
        """构造返回固定 Top-K 的 mock KnnEngine（避免真实矩阵乘）。"""
        from unittest.mock import MagicMock
        engine = MagicMock()
        engine.estimate_block_q.return_value = 2  # 每块 2 行，强制多块
        engine.knn_chunk.side_effect = [
            (np.zeros((2, 5)), topk_labels[:2], topk_ids[:2]),
            (np.zeros((1, 5)), topk_labels[2:], topk_ids[2:]),
        ]
        return engine

    def test_accuracy_by_k_consistent_with_overall(self, monkeypatch):
        """accuracy_by_k 多 K 与单 K overall_accuracy 一致."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        # 3 个查询，每个 5 个邻居（已剔除自身后）；各查询真实标签 = 其类别 i
        queries = [
            {"vector": np.ones(64), "label": i, "label_name": LABEL_NAMES[i],
             "point_id": f"q{i}", "actual_count": 3}
            for i in range(3)
        ]
        # Top-5 邻居: q0 → 全 0; q1 → 全 1; q2 → 全 2（k=3 与 k=5 都应预测正确）
        topk_labels = np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
        ], dtype=np.int64)
        topk_ids = np.array([
            ["n00", "n01", "n02", "n03", "n04"],
            ["n10", "n11", "n12", "n13", "n14"],
            ["n20", "n21", "n22", "n23", "n24"],
        ])

        fake = self._fake_engine(topk_labels, topk_ids)
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((3, 64), dtype=np.float32), np.array([0, 1, 2]), np.array(["n00", "n10", "n20"]),
        ))

        result = M.compute_knn_accuracy(
            manager, queries, k=3, k_values=[3, 5], exact=True, device="cpu",
        )

        assert result["overall_accuracy"] == 1.0
        assert result["accuracy_by_k"] == {3: 1.0, 5: 1.0}
        assert result["num_queries"] == 3
        # 单 K overall_accuracy 与 accuracy_by_k[k] 一致
        assert result["overall_accuracy"] == result["accuracy_by_k"][3]

    def test_loo_excludes_self_in_gpu_path(self, monkeypatch):
        """GPU 路径 LOO 剔除自身."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-3: [自身(label=0), 0, 0] → 剔除自身后 [0, 0] → 预测 0 正确
        topk_labels = np.array([[0, 0, 0]], dtype=np.int64)
        topk_ids = np.array([["q0", "n1", "n2"]])

        fake = self._fake_engine(topk_labels, topk_ids)
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["q0"]),
        ))

        result = M.compute_knn_accuracy(manager, queries, k=2, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0

    def test_tie_break_decrement_in_gpu_path(self, monkeypatch):
        """GPU 路径平票用 _resolve_tie 递减 K 打破."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # Top-5: 0,0,1,1,2 → k=5 平票；k=4（取前 4）: 0,0,1,1 仍平票；
        # k=2（K-3）: 0,0 → 预测 0 正确
        topk_labels = np.array([[0, 0, 1, 1, 2]], dtype=np.int64)
        topk_ids = np.array([["n0", "n1", "n2", "n3", "n4"]])

        fake = self._fake_engine(topk_labels, topk_ids)
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n0"]),
        ))

        result = M.compute_knn_accuracy(manager, queries, k=5, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0


class TestBoundaryConditions:
    """Task 4 边界条件：K>N / 空 Collection 回退 / 某类无像素 / _scroll_full_vectors 64 维校验."""

    def test_k_greater_than_n_topk_capped(self, monkeypatch):
        """K > N 时 topk 取 min(k, N)=N，结果全部邻居."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # 邻居 label=0（非自身，id 不同），K=5 但 N 只有 1 → 1 个邻居 → 预测 0
        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (
            np.zeros((1, 1)), np.array([[0]], dtype=np.int64), np.array([["n1"]]),
        )
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n1"]),
        ))
        result = M.compute_knn_accuracy(manager, queries, k=5, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0

    def test_empty_collection_falls_back_sequential(self, monkeypatch):
        """空 Collection：_scroll_full_vectors 返回空 → 回退服务端逐条."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.empty((0, 64), dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=object),
        ))
        mock_hits = [
            HitRecord(id="n1", score=0.9, label=0, label_name=LABEL_NAMES[0],
                      utm_easting=0, utm_northing=0, utm_zone=51,
                      image_id="img", pixel_row=1, pixel_col=1),
        ]
        with patch("KNN_evaluation.metrics.PixelSearcher") as mock_sc:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = MagicMock(hits=mock_hits)
            mock_sc.return_value = mock_searcher
            result = M.compute_knn_accuracy(manager, queries, k=3, exact=True, device="cpu")
        assert result["overall_accuracy"] == 1.0
        assert result["num_queries"] == 1

    def test_class_with_no_pixels_returns_zero_metrics(self, monkeypatch):
        """某类无像素：queries 中该类为空标记（vectors 非 None），per-class 返回 0."""
        from unittest.mock import MagicMock
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        # label=0 无像素空标记；label=1 有 1 个真实查询
        queries = [
            {"label": 0, "label_name": LABEL_NAMES[0], "vectors": [], "actual_count": 0},
            {"vector": np.ones(64), "label": 1, "label_name": LABEL_NAMES[1],
             "point_id": "q1", "actual_count": 1},
        ]
        fake = MagicMock()
        fake.estimate_block_q.return_value = 1
        fake.knn_chunk.return_value = (
            np.zeros((1, 1)), np.array([[1]], dtype=np.int64), np.array([["n1"]]),
        )
        import KNN_evaluation.metrics as M
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([1]), np.array(["n1"]),
        ))
        result = M.compute_knn_accuracy(manager, queries, k=3, exact=True, device="cpu")
        assert result["num_queries"] == 1
        assert result["per_class_metrics"][LABEL_NAMES[0]]["support"] == 0
        assert result["per_class_metrics"][LABEL_NAMES[0]]["f1"] == 0.0
        assert result["per_class_metrics"][LABEL_NAMES[1]]["support"] == 1

    def test_scroll_returns_float32(self):
        """_scroll_full_vectors 返回 float32（供 GPU 常驻）."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        rec = MagicMock()
        rec.vector = [1.0] * 64
        rec.payload = {"label": 0}
        rec.id = "pt-0"
        manager.client.scroll.side_effect = [([rec], None), ([], None)]
        from KNN_evaluation.metrics import _scroll_full_vectors
        vectors, labels, point_ids = _scroll_full_vectors(manager)
        assert vectors.dtype == np.float32
        assert vectors.shape == (1, 64)

    def test_scroll_rejects_wrong_dimension(self):
        """_scroll_full_vectors 非 64 维向量应抛 ValueError（防静默 reshape 拼接）."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        rec = MagicMock()
        rec.vector = [1.0] * 128  # 维度错误（128 维，reshape(-1,64) 会静默拼成 2 行）
        rec.payload = {"label": 0}
        rec.id = "pt-bad"
        manager.client.scroll.side_effect = [([rec], None), ([], None)]
        from KNN_evaluation.metrics import _scroll_full_vectors
        with pytest.raises(ValueError, match="期望 64 维"):
            _scroll_full_vectors(manager)


@pytest.mark.integration
class TestIntegration:
    """需要运行中的 Qdrant + data_demo 数据。"""

    def test_evaluate_end_to_end(self):
        """使用 data_demo 数据运行完整评估流程。"""
        from KNN_evaluation.qdrant_client import QdrantManager
        from KNN_evaluation.metrics import (
            sample_queries_by_label,
            compute_knn_accuracy,
            compute_purity_recall_curve,
        )

        manager = QdrantManager()
        if not manager.health_check():
            pytest.skip("Qdrant 不可达")
        if not manager.collection_exists():
            pytest.skip("Collection 不存在")

        info = manager.collection_info()
        if info.get("total_points", 0) == 0:
            pytest.skip("Collection 为空")

        # 小规模采样评估
        queries = sample_queries_by_label(manager, samples_per_class=50, seed=42)
        num_q = sum(1 for q in queries if "point_id" in q)
        assert num_q > 0

        f1 = compute_knn_accuracy(manager, queries, k=10)
        assert 0.0 <= f1["overall_accuracy"] <= 1.0
        assert f1["confusion_matrix"].shape == (9, 9)
        assert len(f1["per_class_metrics"]) == 9

        f2 = compute_purity_recall_curve(manager, queries, k_values=[10, 30, 50])
        assert len(f2["k_values"]) == 3
        assert len(f2["global_purity"]) == 3
        assert len(f2["global_recall"]) == 3

        # Purity 单调递减验证
        for i in range(len(f2["k_values"]) - 1):
            assert f2["global_purity"][i] >= f2["global_purity"][i + 1] - 0.01, \
                f"Purity@{f2['k_values'][i]}: {f2['global_purity'][i]} < Purity@{f2['k_values'][i+1]}: {f2['global_purity'][i+1]}"


class TestKnnSignatures:
    def test_new_device_defaults(self):
        import inspect
        from KNN_evaluation import metrics
        sig1 = inspect.signature(metrics.compute_knn_accuracy)
        sig2 = inspect.signature(metrics.compute_purity_recall_curve)
        assert sig1.parameters["device"].default == "auto"
        assert sig1.parameters["gpu_batch_q"].default is None
        assert sig1.parameters["max_gpu_mem"].default == 16
        assert sig1.parameters["max_eval_ram"].default == 6.0
        assert sig1.parameters["k_values"].default is None  # 多 K（默认 [k]）
        assert "use_batch" not in sig1.parameters
        assert "use_batch" not in sig2.parameters
        assert not hasattr(metrics, "_batch_exact_knn")


def _qdrant_is_running() -> bool:
    """检查本地 Qdrant Docker 是否运行（无 Qdrant 时跳过集成用例）."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        return "qdrant" in result.stdout or "qdrant-knn-eval" in result.stdout
    except Exception:
        return False


class TestBatchVsSequentialConsistency:
    @pytest.mark.skipif(not _qdrant_is_running(), reason="Qdrant 未运行，跳过")
    def test_batch_and_sequential_agree(self, qdrant_manager):
        """小数据下 GPU 分块（KnnEngine）与逐条路径结果一致（真实 Qdrant，conftest fixture）.

        分块路径 = compute_*_curve(..., device="cpu")；逐条路径 = 内部
        _knn_accuracy_sequential / _purity_recall_sequential（Qdrant query_points）。
        两条不同计算路径必须在真实数据上结果一致，而非同义反复。
        """
        from KNN_evaluation.metrics import (
            compute_knn_accuracy, compute_purity_recall_curve, sample_queries_by_label,
            _knn_accuracy_sequential, _purity_recall_sequential,
            _compute_per_class_label_totals,
        )
        queries = sample_queries_by_label(qdrant_manager, samples_per_class=5, seed=7)
        # F1：分块路径 vs 逐条路径
        chunked = compute_knn_accuracy(qdrant_manager, queries, k=5, exact=True, device="cpu")
        sequential = _knn_accuracy_sequential(qdrant_manager, queries, 5, True, None)
        assert chunked["overall_accuracy"] == sequential["overall_accuracy"]
        assert chunked["confusion_matrix"].tolist() == sequential["confusion_matrix"].tolist()
        # F2：分块 Purity/Recall vs 逐条
        label_totals = _compute_per_class_label_totals(qdrant_manager)
        chunked2 = compute_purity_recall_curve(qdrant_manager, queries, [2, 5], exact=True, device="cpu")
        sequential2 = _purity_recall_sequential(
            qdrant_manager, queries, [2, 5], label_totals, True, None,
        )
        assert chunked2["global_purity"] == sequential2["global_purity"]
        assert chunked2["global_recall"] == sequential2["global_recall"]

    @pytest.mark.skipif(not _qdrant_is_running(), reason="Qdrant 未运行，跳过")
    def test_default_path_no_guard_trigger(self, qdrant_manager):
        """默认 device 参数走 GPU 分块路径，返回正确 num_queries."""
        from KNN_evaluation.metrics import compute_knn_accuracy
        queries = sample_queries_by_label(qdrant_manager, samples_per_class=3, seed=1)
        result = compute_knn_accuracy(qdrant_manager, queries, k=3, exact=True, device="cpu")
        assert result["num_queries"] == sum(1 for q in queries if "point_id" in q)

    @pytest.mark.skipif(not _qdrant_is_running(), reason="Qdrant 未运行，跳过")
    def test_evaluate_knn_matches_independent_calls(self, qdrant_manager):
        """集成：真实 Qdrant 上 evaluate_knn 与 compute_knn_accuracy + compute_purity_recall_curve 一致."""
        from KNN_evaluation.metrics import (
            evaluate_knn, compute_knn_accuracy, compute_purity_recall_curve,
            sample_queries_by_label,
        )
        queries = sample_queries_by_label(qdrant_manager, samples_per_class=4, seed=7)
        combined = evaluate_knn(
            qdrant_manager, queries, k_f1=3, k_values=[2, 3, 5],
            exact=True, device="cpu",
        )
        f1 = compute_knn_accuracy(qdrant_manager, queries, k=3, k_values=[2, 3, 5],
                                  exact=True, device="cpu")
        f2 = compute_purity_recall_curve(qdrant_manager, queries, k_values=[2, 3, 5],
                                         exact=True, device="cpu")
        assert combined["f1"]["overall_accuracy"] == f1["overall_accuracy"]
        assert combined["f1"]["accuracy_by_k"] == f1["accuracy_by_k"]
        assert combined["f1"]["confusion_matrix"].tolist() == f1["confusion_matrix"].tolist()
        assert combined["f2"]["global_purity"] == f2["global_purity"]
        assert combined["f2"]["global_recall"] == f2["global_recall"]

    @pytest.mark.integration
    @pytest.mark.skipif(not _qdrant_is_running(), reason="Qdrant 未运行，跳过")
    def test_cpu_and_auto_fallback_consistent(self, qdrant_manager):
        """集成断言：device=auto（CUDA 不可用回退 cpu）与 device=cpu 分块路径结果一致.

        真实 Qdrant + 确定性小数据集，验证 Task 1 auto 回退逻辑在完整调用链上
        与显式 cpu 产生完全一致结果（overall_accuracy / accuracy_by_k / 混淆矩阵）。
        """
        import warnings
        from KNN_evaluation.metrics import compute_knn_accuracy, sample_queries_by_label
        queries = sample_queries_by_label(qdrant_manager, samples_per_class=5, seed=7)
        cpu = compute_knn_accuracy(qdrant_manager, queries, k=5, exact=True, device="cpu")
        # auto 在本机（无 CUDA）回退 cpu，resolve_device 发出 RuntimeWarning，捕获避免污染测试输出
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            auto = compute_knn_accuracy(qdrant_manager, queries, k=5, exact=True, device="auto")
        assert cpu["overall_accuracy"] == auto["overall_accuracy"]
        assert cpu["accuracy_by_k"] == auto["accuracy_by_k"]
        assert cpu["confusion_matrix"].tolist() == auto["confusion_matrix"].tolist()
        assert cpu["num_queries"] == auto["num_queries"]
        assert cpu["num_queries"] == sum(1 for q in queries if "point_id" in q)


class TestDeviceFallback:
    """Task 6 CUDA 回退：auto → torch CPU 分块 + 警告；显式 cuda → 抛错."""

    def test_auto_falls_back_to_cpu_with_warning(self, monkeypatch):
        """auto + CUDA 不可用 → 回退 torch CPU 分块 + 警告（不抛错）."""
        import torch
        import KNN_evaluation.metrics as M
        # 守卫：metrics 模块不应直接引用 torch（delattr 容错，不存在时静默跳过）
        monkeypatch.delattr(M, "torch", raising=False)
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        # CUDA 不可用（mock，任何机器上都确定）：resolve_device('auto') 发出
        # RuntimeWarning 并回退 torch CPU 分块，验证 metrics 调用链的完整回退行为。
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n1"]),
        ))
        with pytest.warns(RuntimeWarning):
            result = M.compute_knn_accuracy(manager, queries, k=3, exact=True, device="auto")
        assert result["overall_accuracy"] == 1.0

    def test_explicit_cuda_unavailable_raises(self, monkeypatch):
        """device='cuda' 但 CUDA 不可用 → 抛 RuntimeError."""
        import KNN_evaluation.gpu_knn as GK
        monkeypatch.setattr(GK, "resolve_device",
                            lambda d: (_ for _ in ()).throw(
                                RuntimeError("device='cuda' 但当前环境 CUDA 不可用")))
        from KNN_evaluation.gpu_knn import KnnEngine
        with pytest.raises(RuntimeError, match="CUDA"):
            KnnEngine(device="cuda")

    def test_auto_fallback_budget_uses_max_eval_ram(self, monkeypatch):
        """[Important] device='auto' + CUDA 不可用回退 cpu 时，_device_budget 用 max_eval_ram(6GB) 推导 block_q.

        回归：原始 'auto' 会绕过 _device_budget 的 device=='cpu' 判断，误用
        max_gpu_mem(16GB) 推导 block_q（单块 sim 矩阵超 max_eval_ram 预算，
        16GB RAM 机器 OOM 风险）。修复后 compute_* 把已解析的 engine.device
        （'cpu'）传给 _device_budget，budget 应为 max_eval_ram。
        """
        import torch
        import KNN_evaluation.metrics as M
        from KNN_evaluation.gpu_knn import KnnEngine

        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        manager.client.count.return_value.count = 100
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]

        # CUDA 不可用 + 真实 resolve_device('auto') → 回退 cpu（任何机器上都确定）
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.warns(RuntimeWarning):
            real_engine = KnnEngine(device="auto")
        assert real_engine.device == "cpu"  # 前置条件：auto 已解析为 cpu

        # 记录 _device_budget 传给 estimate_block_q 的预算参数
        recorded_budgets: list[float] = []

        def recording_estimate(max_mem_gb):
            recorded_budgets.append(float(max_mem_gb))
            return 1  # 固定 block_q，快速执行

        real_engine.estimate_block_q = recording_estimate
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: real_engine)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["n1"]),
        ))
        monkeypatch.setattr(
            M, "_compute_per_class_label_totals", lambda mgr: {LABEL_NAMES[0]: 100},
        )

        M.compute_knn_accuracy(manager, queries, k=3, exact=True,
                               device="auto", max_gpu_mem=16, max_eval_ram=6.0)
        M.compute_purity_recall_curve(manager, queries, k_values=[2], exact=True,
                                      device="auto", max_gpu_mem=16, max_eval_ram=6.0)

        # 两条 compute_* 路径都应以 max_eval_ram(6.0) 而非 max_gpu_mem(16) 推导
        assert recorded_budgets == [6.0, 6.0]


class TestEvaluateKnn:
    """Section 8: evaluate_knn 联合评估 —— 单次 scroll + 单次 topk，F1/F2 共享结果."""

    def _fake_engine(self, topk_labels, topk_ids, block_q=2):
        """构造返回固定 Top-K 的 mock KnnEngine.

        knn_chunk 按查询块行数从前部切取结果（非 side_effect 一次性消费），
        使同一 fake 可在 evaluate_knn 与独立 compute_* 差分调用间复用。
        """
        engine = MagicMock()
        engine.estimate_block_q.return_value = block_q

        def chunk(qblock, k):
            s = int(np.asarray(qblock).shape[0])
            return (np.zeros((s, topk_labels.shape[1])),
                    topk_labels[:s], topk_ids[:s])

        engine.knn_chunk.side_effect = chunk
        return engine

    def _patch_env(self, monkeypatch, manager, queries, topk_labels, topk_ids,
                   scroll_count, block_q=2, totals=None):
        """打补丁 KnnEngine / _scroll_full_vectors / _compute_per_class_label_totals."""
        import KNN_evaluation.metrics as M
        fake = self._fake_engine(topk_labels, topk_ids, block_q=block_q)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)

        def fake_scroll(mgr):
            scroll_count["n"] += 1
            return (
                np.ones((len(queries), 64), dtype=np.float32),
                np.array([q["label"] for q in queries], dtype=np.int64),
                np.array([q["point_id"] for q in queries]),
            )

        monkeypatch.setattr(M, "_scroll_full_vectors", fake_scroll)
        if totals is None:
            totals = {LABEL_NAMES[i]: 100 for i in range(9)}
        monkeypatch.setattr(M, "_compute_per_class_label_totals", lambda mgr: totals)
        return fake

    def test_combined_matches_individual_calls(self, monkeypatch):
        """evaluate_knn 与 compute_knn_accuracy + compute_purity_recall_curve 结果一致."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(4)
        ]
        # Top-5（max_k=3 → k=4）：邻居全同类，ids 均非自身
        topk_labels = np.zeros((4, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(4)])

        scroll_count = {"n": 0}
        self._patch_env(monkeypatch, manager, queries, topk_labels, topk_ids,
                        scroll_count, block_q=2)

        combined = evaluate_knn(
            manager, queries, k_f1=3, k_values=[3, 5], exact=True, device="cpu",
        )
        assert scroll_count["n"] == 1, "evaluate_knn 必须单次 scroll"

        f1 = compute_knn_accuracy(manager, queries, k=3, k_values=[3, 5],
                                  exact=True, device="cpu")
        f2 = compute_purity_recall_curve(manager, queries, k_values=[3, 5],
                                         exact=True, device="cpu")
        assert combined["f1"]["overall_accuracy"] == f1["overall_accuracy"]
        assert combined["f1"]["accuracy_by_k"] == f1["accuracy_by_k"]
        assert combined["f1"]["confusion_matrix"].tolist() == f1["confusion_matrix"].tolist()
        assert combined["f2"]["global_purity"] == f2["global_purity"]
        assert combined["f2"]["global_recall"] == f2["global_recall"]

    def test_f1_includes_per_class_recall_by_k(self, monkeypatch):
        """evaluate_knn 输出 per_class_recall_by_k：每个 K 一张 {label: recall}."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(4)
        ]
        topk_labels = np.zeros((4, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(4)])
        self._patch_env(monkeypatch, manager, queries, topk_labels, topk_ids,
                        {"n": 0}, block_q=2)

        combined = evaluate_knn(
            manager, queries, k_f1=3, k_values=[3, 5], exact=True, device="cpu",
        )
        prbk = combined["f1"]["per_class_recall_by_k"]
        assert set(prbk.keys()) == {3, 5}
        for kv in (3, 5):
            assert set(prbk[kv].keys()) == set(LABEL_NAMES.values())
        # 查询全为 water 且全部正确分类 → water recall=1，其余类 tp=0 → recall=0
        assert prbk[3][LABEL_NAMES[0]] == pytest.approx(1.0)
        assert prbk[5][LABEL_NAMES[1]] == pytest.approx(0.0)

    def test_f2_output_uses_only_user_k_values_when_k_f1_not_in_k_values(self, monkeypatch):
        """k_f1 ∉ k_values 时 f2 输出只含用户 k_values，与独立 compute_purity_recall_curve 一致.

        回归 [Important]：入口 `k_values = sorted(set([k_f1] + k_values))` 曾把 k_f1
        并入 F2 累加器与输出，当 k_f1=7 而 k_values=[2,5] 时 f2 多出一个 K=7 点，
        违反 delta spec「联合结果与独立调用一致」。max_k 仍须含 k_f1（决定 topk 上限），
        但 F2 累加器/输出只用用户传入序列。
        """
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(4)
        ]
        # max_k = max([7, 2, 5]) = 7 → top-(8)；Top-K 固定 8 列供递增取 [2,5]
        topk_labels = np.zeros((4, 8), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(8)] for i in range(4)])

        scroll_count = {"n": 0}
        self._patch_env(monkeypatch, manager, queries, topk_labels, topk_ids,
                        scroll_count, block_q=2)

        combined = evaluate_knn(
            manager, queries, k_f1=7, k_values=[2, 5], exact=True, device="cpu",
        )
        f2 = compute_purity_recall_curve(manager, queries, k_values=[2, 5],
                                         exact=True, device="cpu")

        # f2 输出只含用户 k_values=[2,5]，不含 k_f1=7
        assert combined["f2"]["k_values"] == [2, 5]
        assert combined["f2"]["k_values"] == f2["k_values"]
        assert combined["f2"]["global_purity"] == f2["global_purity"]
        assert combined["f2"]["global_recall"] == f2["global_recall"]
        assert list(combined["f2"]["per_class_purity"].keys()) == list(LABEL_NAMES.values())
        assert combined["f2"]["per_class_purity"][LABEL_NAMES[0]] == f2["per_class_purity"][LABEL_NAMES[0]]
        # f1 侧仍含 k_f1=7 与用户 k_values（accuracy_by_k 与独立 compute_knn_accuracy 一致）
        assert combined["f1"]["accuracy_by_k"] == {7: 1.0, 2: 1.0, 5: 1.0}

    def test_f2_reuses_f1_topk_no_extra_scroll(self, monkeypatch):
        """F2 复用 F1 的 scroll + topk：单次 scroll、knn_chunk 只对每块调用一次."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(3)
        ]
        topk_labels = np.zeros((3, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(3)])

        scroll_count = {"n": 0}
        fake = self._patch_env(monkeypatch, manager, queries, topk_labels, topk_ids,
                               scroll_count, block_q=2)

        evaluate_knn(manager, queries, k_f1=3, k_values=[3, 5],
                     exact=True, device="cpu")
        assert scroll_count["n"] == 1, "单次 scroll"
        assert fake.knn_chunk.call_count == 2, "block_q=2, 3 行 → 2 块，每块一次 knn_chunk"

    def test_close_called_even_on_exception(self, monkeypatch):
        """engine.close() 在异常时也调用（try/finally 保证显存释放）."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        topk_labels = np.zeros((1, 5), dtype=np.int64)
        topk_ids = np.array([["q0n1", "q0n2", "q0n3", "q0n4", "q0n5"]])
        scroll_count = {"n": 0}
        fake = self._patch_env(monkeypatch, manager, queries, topk_labels, topk_ids,
                               scroll_count, block_q=2)

        # knn_chunk 抛异常 → 验证 close 仍被调用（try/finally）且异常向外传播
        def boom_chunk(*a, **k):
            raise RuntimeError("knn boom")
        fake.knn_chunk.side_effect = boom_chunk
        with pytest.raises(RuntimeError, match="knn boom"):
            evaluate_knn(manager, queries, k_f1=3, k_values=[3, 5],
                         exact=True, device="cpu")
        fake.close.assert_called_once()


class TestEvaluateCancellation:
    """Section 9.2: 取消令牌协作式中止评估.

    新增可选参数 `cancel_event: threading.Event | None = None`，分块循环每块
    检查 `is_set()` 命中即抛 `EvaluationCancelled`；`engine.close()` 由
    try/finally 保证在取消时仍释放显存。
    """

    def _fake_engine(self, topk_labels, topk_ids, block_q=2):
        """构造返回固定 Top-K 的 mock KnnEngine（复用 TestEvaluateKnn 的模式）."""
        engine = MagicMock()
        engine.estimate_block_q.return_value = block_q

        def chunk(qblock, k):
            s = int(np.asarray(qblock).shape[0])
            return (np.zeros((s, topk_labels.shape[1])),
                    topk_labels[:s], topk_ids[:s])

        engine.knn_chunk.side_effect = chunk
        return engine

    def _patch_env(self, monkeypatch, manager, queries, topk_labels, topk_ids,
                   scroll_count, block_q=2, totals=None):
        """打补丁 KnnEngine / _scroll_full_vectors / _compute_per_class_label_totals."""
        import KNN_evaluation.metrics as M
        fake = self._fake_engine(topk_labels, topk_ids, block_q=block_q)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)

        def fake_scroll(mgr):
            scroll_count["n"] += 1
            return (
                np.ones((len(queries), 64), dtype=np.float32),
                np.array([q["label"] for q in queries], dtype=np.int64),
                np.array([q["point_id"] for q in queries]),
            )

        monkeypatch.setattr(M, "_scroll_full_vectors", fake_scroll)
        if totals is None:
            totals = {LABEL_NAMES[i]: 100 for i in range(9)}
        monkeypatch.setattr(M, "_compute_per_class_label_totals", lambda mgr: totals)
        return fake

    def test_evaluate_knn_raises_when_event_set_before(self, monkeypatch):
        """cancel_event 已设置 → evaluate_knn 抛 EvaluationCancelled 且 close 被调用."""
        import threading
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(3)
        ]
        topk_labels = np.zeros((3, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(3)])

        fake = self._fake_engine(topk_labels, topk_ids, block_q=2)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((3, 64), dtype=np.float32), np.array([0, 0, 0]),
            np.array(["q0", "q1", "q2"]),
        ))
        totals = {LABEL_NAMES[i]: 100 for i in range(9)}
        monkeypatch.setattr(M, "_compute_per_class_label_totals", lambda mgr: totals)

        cancel = threading.Event()
        cancel.set()
        with pytest.raises(M.EvaluationCancelled):
            evaluate_knn(manager, queries, k_f1=3, k_values=[3, 5],
                         exact=True, device="cpu", cancel_event=cancel)
        fake.close.assert_called_once()

    def test_evaluate_knn_raises_when_event_set_mid_loop(self, monkeypatch):
        """第二块循环开始前事件被设置 → 抛 EvaluationCancelled 且 close 被调用."""
        import threading
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(4)
        ]
        topk_labels = np.zeros((4, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(4)])

        fake = self._fake_engine(topk_labels, topk_ids, block_q=2)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((4, 64), dtype=np.float32), np.array([0, 0, 0, 0]),
            np.array(["q0", "q1", "q2", "q3"]),
        ))
        totals = {LABEL_NAMES[i]: 100 for i in range(9)}
        monkeypatch.setattr(M, "_compute_per_class_label_totals", lambda mgr: totals)

        cancel = threading.Event()
        call_count = {"n": 0}
        original_chunk = fake.knn_chunk.side_effect

        def chunk(qblock, k):
            call_count["n"] += 1
            if call_count["n"] == 2:
                cancel.set()  # 第二块开始时设置事件
            return original_chunk(qblock, k)

        fake.knn_chunk.side_effect = chunk
        with pytest.raises(M.EvaluationCancelled):
            evaluate_knn(manager, queries, k_f1=3, k_values=[3, 5],
                         exact=True, device="cpu", cancel_event=cancel)
        assert call_count["n"] == 2, "第一块执行，第二块开始时取消命中"
        fake.close.assert_called_once()

    def test_no_cancel_event_completes_normally(self, monkeypatch):
        """缺省 cancel_event=None（向后兼容）→ 不检查取消，评估正常完成."""
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(3)
        ]
        topk_labels = np.zeros((3, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(3)])

        scroll_count = {"n": 0}
        fake = self._patch_env(monkeypatch, manager, queries, topk_labels, topk_ids,
                               scroll_count, block_q=2)
        result = evaluate_knn(manager, queries, k_f1=3, k_values=[3, 5],
                              exact=True, device="cpu")
        assert result["f1"]["overall_accuracy"] == 1.0
        fake.close.assert_called_once()

    def test_cancel_checked_after_scroll_full_vectors(self, monkeypatch):
        """scroll 完成后、进入分块循环前检查取消事件（引擎已建，close 仍被调用）."""
        import threading
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": "q0", "actual_count": 1},
        ]
        topk_labels = np.zeros((1, 5), dtype=np.int64)
        topk_ids = np.array([["q0n1", "q0n2", "q0n3", "q0n4", "q0n5"]])
        scroll_count = {"n": 0}
        fake = self._fake_engine(topk_labels, topk_ids, block_q=2)

        def counting_scroll(mgr):
            scroll_count["n"] += 1
            return (np.ones((1, 64), dtype=np.float32), np.array([0]), np.array(["q0"]))

        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", counting_scroll)
        totals = {LABEL_NAMES[i]: 100 for i in range(9)}
        monkeypatch.setattr(M, "_compute_per_class_label_totals", lambda mgr: totals)

        cancel = threading.Event()
        cancel.set()
        with pytest.raises(M.EvaluationCancelled):
            evaluate_knn(manager, queries, k_f1=3, k_values=[3, 5],
                         exact=True, device="cpu", cancel_event=cancel)
        assert scroll_count["n"] == 1, "scroll 已完成，取消检查在循环开始前命中"
        fake.close.assert_called_once()  # 引擎已建，try/finally 保证释放


class TestComputeFunctionsCancellation:
    """Section 9.2: compute_knn_accuracy / compute_purity_recall_curve 也接受 cancel_event.

    WebUI 只走 evaluate_knn，但两个独立函数与 evaluate_knn 同样共享分块循环，
    需保持一致的协作式取消能力（设计 D9.2「简化方案」）。
    """

    def _fake_engine(self, topk_labels, topk_ids, block_q=2):
        engine = MagicMock()
        engine.estimate_block_q.return_value = block_q

        def chunk(qblock, k):
            s = int(np.asarray(qblock).shape[0])
            return (np.zeros((s, topk_labels.shape[1])),
                    topk_labels[:s], topk_ids[:s])

        engine.knn_chunk.side_effect = chunk
        return engine

    def test_compute_knn_accuracy_cancel(self, monkeypatch):
        """cancel_event 已设置 → compute_knn_accuracy 抛 EvaluationCancelled 且 close 被调用."""
        import threading
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(3)
        ]
        topk_labels = np.zeros((3, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(3)])

        fake = self._fake_engine(topk_labels, topk_ids, block_q=2)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((3, 64), dtype=np.float32), np.array([0, 0, 0]),
            np.array(["q0", "q1", "q2"]),
        ))

        cancel = threading.Event()
        cancel.set()
        with pytest.raises(M.EvaluationCancelled):
            compute_knn_accuracy(manager, queries, k=3, exact=True, device="cpu",
                                 cancel_event=cancel)
        fake.close.assert_called_once()

    def test_compute_purity_recall_cancel(self, monkeypatch):
        """cancel_event 已设置 → compute_purity_recall_curve 抛 EvaluationCancelled 且 close 被调用."""
        import threading
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(3)
        ]
        topk_labels = np.zeros((3, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(3)])

        fake = self._fake_engine(topk_labels, topk_ids, block_q=2)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((3, 64), dtype=np.float32), np.array([0, 0, 0]),
            np.array(["q0", "q1", "q2"]),
        ))
        totals = {LABEL_NAMES[i]: 100 for i in range(9)}
        monkeypatch.setattr(M, "_compute_per_class_label_totals", lambda mgr: totals)

        cancel = threading.Event()
        cancel.set()
        with pytest.raises(M.EvaluationCancelled):
            compute_purity_recall_curve(manager, queries, k_values=[2, 3],
                                        exact=True, device="cpu", cancel_event=cancel)
        fake.close.assert_called_once()

    def test_compute_cancel_mid_loop_closes_engine(self, monkeypatch):
        """compute_purity_recall_curve 分块中途取消 → close 仍被调用（try/finally）."""
        import threading
        import KNN_evaluation.metrics as M
        manager = MagicMock(spec=QdrantManager)
        manager.collection_name = "test_collection"
        queries = [
            {"vector": np.ones(64), "label": 0, "label_name": LABEL_NAMES[0],
             "point_id": f"q{i}", "actual_count": 1}
            for i in range(4)
        ]
        topk_labels = np.zeros((4, 5), dtype=np.int64)
        topk_ids = np.array([[f"q{i}n{j}" for j in range(5)] for i in range(4)])

        fake = self._fake_engine(topk_labels, topk_ids, block_q=2)
        monkeypatch.setattr(M, "KnnEngine", lambda **kw: fake)
        monkeypatch.setattr(M, "_scroll_full_vectors", lambda mgr: (
            np.ones((4, 64), dtype=np.float32), np.array([0, 0, 0, 0]),
            np.array(["q0", "q1", "q2", "q3"]),
        ))
        totals = {LABEL_NAMES[i]: 100 for i in range(9)}
        monkeypatch.setattr(M, "_compute_per_class_label_totals", lambda mgr: totals)

        cancel = threading.Event()
        call_count = {"n": 0}
        original_chunk = fake.knn_chunk.side_effect

        def chunk(qblock, k):
            call_count["n"] += 1
            if call_count["n"] == 2:
                cancel.set()
            return original_chunk(qblock, k)

        fake.knn_chunk.side_effect = chunk
        with pytest.raises(M.EvaluationCancelled):
            compute_purity_recall_curve(manager, queries, k_values=[2, 3],
                                        exact=True, device="cpu", cancel_event=cancel)
        fake.close.assert_called_once()


class TestKnnEngineCloseFreesCuda:
    """Section 8.1: KnnEngine.close() 释放 PyTorch GPU 缓存."""

    def test_close_calls_empty_cache_on_cuda(self, monkeypatch):
        """cuda 设备 close() 调用 torch.cuda.empty_cache()."""
        import KNN_evaluation.gpu_knn as GK
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        # 真实 close 逻辑只在 self.device=='cuda' 时调用 empty_cache
        engine = MagicMock()
        engine.device = "cuda"
        engine.torch = fake_torch
        engine.corpus = object()
        engine._labels = None
        engine._ids = None
        real_close = GK.KnnEngine.close
        # 直接用绑定方法在 mock 实例上执行真实 close 逻辑
        monkeypatch.setattr(fake_torch.cuda, "empty_cache", MagicMock())
        real_close(engine)
        assert engine.corpus is None
        fake_torch.cuda.empty_cache.assert_called_once()

    def test_close_skips_empty_cache_on_cpu(self, monkeypatch):
        """cpu 设备 close() 不调用 torch.cuda.empty_cache()."""
        import KNN_evaluation.gpu_knn as GK
        fake_torch = MagicMock()
        engine = MagicMock()
        engine.device = "cpu"
        engine.torch = fake_torch
        engine.corpus = object()
        engine._labels = None
        engine._ids = None
        real_close = GK.KnnEngine.close
        real_close(engine)
        fake_torch.cuda.empty_cache.assert_not_called()
