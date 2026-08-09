"""Tests for KNN_evaluation.similarity_compare（双集合相似度热力图对比核心模块）."""
import io
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from KNN_evaluation.qdrant_client import QdrantManager
from KNN_evaluation import similarity_compare as SC


def _manager(collection="test_collection", total_points=0):
    manager = MagicMock(spec=QdrantManager)
    manager.collection_name = collection
    manager.url = "http://localhost:6333"
    manager.collection_exists.return_value = True
    manager.collection_info.return_value = {"total_points": total_points}
    return manager


def _by_label_map(by_label: dict, total_points: int) -> dict:
    return {
        "collection": "test_collection", "total_points": total_points,
        "updated_at": "", "by_label": dict(by_label),
    }


class TestSampleRandomPoints:
    def test_db_mode_uses_map_and_seed_reproducible(self, monkeypatch):
        by_label = {0: [f"pt-{i}" for i in range(50)], 1: [f"pt-{i + 50}" for i in range(50)]}
        mgr = _manager(total_points=100)
        monkeypatch.setattr(SC, "ensure_sampling_map",
                            lambda m, path=None: _by_label_map(by_label, 100))
        pts1 = SC.sample_random_points(mgr, 20, seed=42)
        pts2 = SC.sample_random_points(mgr, 20, seed=42)
        assert [p["point_id"] for p in pts1] == [p["point_id"] for p in pts2]
        assert len(pts1) == 20
        all_ids = {pid for ids in by_label.values() for pid in ids}
        assert all(p["point_id"] in all_ids for p in pts1)
        assert pts1 == [{"point_id": p["point_id"]} for p in pts1]

    def test_n_out_of_range_raises(self):
        mgr = _manager(total_points=100)
        for bad in (0, -1, 601):
            with pytest.raises(ValueError, match="n 必须在 1..600"):
                SC.sample_random_points(mgr, bad, seed=42)

    def test_insufficient_candidates_sampled_actual(self, monkeypatch):
        by_label = {0: [f"pt-{i}" for i in range(5)]}
        mgr = _manager(total_points=5)
        monkeypatch.setattr(SC, "ensure_sampling_map",
                            lambda m, path=None: _by_label_map(by_label, 5))
        pts = SC.sample_random_points(mgr, 200, seed=1)
        assert len(pts) == 5

    def test_empty_collection_raises(self):
        mgr = _manager(total_points=0)
        with pytest.raises(ValueError, match="为空"):
            SC.sample_random_points(mgr, 10, seed=42)

    def test_by_label_empty_collection_nonempty_raises(self, monkeypatch):
        mgr = _manager(total_points=100)
        monkeypatch.setattr(SC, "ensure_sampling_map",
                            lambda m, path=None: _by_label_map({}, 100))
        with pytest.raises(RuntimeError, match="采样地图为空"):
            SC.sample_random_points(mgr, 10, seed=42)

    def test_image_mode_point_ids_deterministic(self, monkeypatch):
        mgr = _manager(total_points=16384)
        monkeypatch.setattr(
            "KNN_evaluation.data_loader.PixelDataLoader.check_image_count",
            lambda iid, m: 16384,
        )
        pts1 = SC.sample_random_points(mgr, 5, seed=7, image_id="E121.4_N25.1")
        pts2 = SC.sample_random_points(mgr, 5, seed=7, image_id="E121.4_N25.1")
        assert pts1 == pts2
        assert len(pts1) == 5
        ns = uuid.uuid5(uuid.NAMESPACE_DNS, "E121.4_N25.1")
        for p in pts1:
            assert p["point_id"] == str(uuid.uuid5(ns, f"{p['pixel_row']}_{p['pixel_col']}"))
            assert p["image_id"] == "E121.4_N25.1"

    def test_image_mode_rows_cols_unique(self, monkeypatch):
        mgr = _manager(total_points=16384)
        monkeypatch.setattr(
            "KNN_evaluation.data_loader.PixelDataLoader.check_image_count",
            lambda iid, m: 16384,
        )
        pts = SC.sample_random_points(mgr, 100, seed=3, image_id="IMG")
        cells = {(p["pixel_row"], p["pixel_col"]) for p in pts}
        assert len(cells) == 100

    def test_image_mode_unknown_image_raises(self, monkeypatch):
        mgr = _manager(total_points=16384)
        monkeypatch.setattr(
            "KNN_evaluation.data_loader.PixelDataLoader.check_image_count",
            lambda iid, m: 0,
        )
        with pytest.raises(ValueError, match="不存在"):
            SC.sample_random_points(mgr, 5, seed=1, image_id="NO_SUCH")


class TestExtractEmbeddings:
    @staticmethod
    def _record(pid: str, value: float, payload: dict | None = None):
        rec = MagicMock()
        rec.id = pid
        rec.vector = [value] * 64
        rec.payload = payload if payload is not None else {}
        return rec

    def test_aligned_matrices_zero_dropped(self):
        g, x = _manager("g"), _manager("x")
        ids = ["a", "b", "c"]
        g.client.retrieve.return_value = [
            self._record(i, float(n), {
                "image_id": f"img{n}", "pixel_row": n, "pixel_col": n * 10,
                "utm_easting": 500000.0 + n, "utm_northing": 4000000.0 + n,
                "utm_zone": "50N",
            }) for n, i in enumerate(ids)
        ]
        x.client.retrieve.return_value = [
            self._record(i, float(n) * 10) for n, i in enumerate(ids)
        ]
        mat_g, mat_x, dropped, kept_records = SC.extract_embeddings(
            [{"point_id": i} for i in ids], g, x,
        )
        assert dropped == 0
        assert mat_g.shape == (3, 64) and mat_x.shape == (3, 64)
        np.testing.assert_allclose(mat_g[:, 0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(mat_x[:, 0], [0.0, 10.0, 20.0])
        # kept_records 与 kept_ids 行序严格一致，字段从 google 侧 payload 提取
        assert [r["point_id"] for r in kept_records] == ids
        assert kept_records[1]["image_id"] == "img1"
        assert kept_records[1]["pixel_row"] == 1
        assert kept_records[1]["pixel_col"] == 10
        assert kept_records[1]["utm_easting"] == 500001.0
        assert kept_records[1]["utm_northing"] == 4000001.0
        assert kept_records[1]["utm_zone"] == "50N"

    def test_retrieve_called_with_same_ids_and_vectors(self):
        g, x = _manager("g"), _manager("x")
        ids = ["a", "b"]
        g.client.retrieve.return_value = [self._record(i, 1.0) for i in ids]
        x.client.retrieve.return_value = [self._record(i, 2.0) for i in ids]
        SC.extract_embeddings([{"point_id": i} for i in ids], g, x)
        g.client.retrieve.assert_called_once_with(
            collection_name="g", ids=["a", "b"],
            with_payload=True, with_vectors=True,
        )
        x.client.retrieve.assert_called_once_with(
            collection_name="x", ids=["a", "b"],
            with_payload=True, with_vectors=True,
        )

    def test_single_side_missing_dropped_and_row_alignment(self):
        g, x = _manager("g"), _manager("x")
        g.client.retrieve.return_value = [
            self._record(i, float(n), {"image_id": f"img{n}"})
            for n, i in enumerate(("a", "b", "c"))
        ]
        x.client.retrieve.return_value = [
            self._record(i, float(n) * 10)
            for n, i in enumerate(("a", "b", "c")) if i != "b"
        ]
        mat_g, mat_x, dropped, kept_records = SC.extract_embeddings(
            [{"point_id": i} for i in ("a", "b", "c")], g, x,
        )
        assert dropped == 1
        assert mat_g.shape == (2, 64) and mat_x.shape == (2, 64)
        # 行序 = ids 原始顺序的保留子序列 (a, c)
        np.testing.assert_allclose(mat_g[:, 0], [0.0, 2.0])
        np.testing.assert_allclose(mat_x[:, 0], [0.0, 20.0])
        # kept_records 与 kept_ids 行序严格一致（剔除 b）
        assert [r["point_id"] for r in kept_records] == ["a", "c"]
        assert kept_records[0]["image_id"] == "img0"
        assert kept_records[1]["image_id"] == "img2"

    def test_kept_records_missing_payload_fields_use_none(self):
        g, x = _manager("g"), _manager("x")
        g.client.retrieve.return_value = [
            self._record("a", 1.0, {"image_id": "img"}),  # 缺 row/col/utm 字段
        ]
        x.client.retrieve.return_value = [self._record("a", 1.0)]
        _, _, _, kept = SC.extract_embeddings([{"point_id": "a"}], g, x)
        assert kept == [{
            "point_id": "a", "image_id": "img",
            "pixel_row": None, "pixel_col": None,
            "utm_easting": None, "utm_northing": None, "utm_zone": None,
        }]

    def test_all_missing_raises(self):
        g, x = _manager("g"), _manager("x")
        g.client.retrieve.return_value = [self._record("a", 1.0)]
        x.client.retrieve.return_value = []  # xian 侧全缺
        with pytest.raises(RuntimeError, match="无任何对齐点"):
            SC.extract_embeddings([{"point_id": "a"}], g, x)

    def test_wrong_dimension_raises(self):
        g, x = _manager("g"), _manager("x")
        rec = MagicMock()
        rec.id = "a"
        rec.vector = [1.0] * 32  # 非 64 维
        rec.payload = {}
        g.client.retrieve.return_value = [rec]
        x.client.retrieve.return_value = [rec]
        with pytest.raises(ValueError, match="维度应为 64"):
            SC.extract_embeddings([{"point_id": "a"}], g, x)


class TestCosineSimilarityMatrix:
    def test_symmetric_diagonal_one_range(self):
        rng = np.random.default_rng(0)
        v = rng.normal(0, 1, (10, 64))
        m = SC.cosine_similarity_matrix(v)
        assert m.shape == (10, 10)
        np.testing.assert_allclose(m, m.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-12)
        assert m.min() >= -1.0 - 1e-9 and m.max() <= 1.0 + 1e-9

    def test_zero_vector_defense(self):
        v = np.zeros((2, 64))
        m = SC.cosine_similarity_matrix(v)
        assert np.isfinite(m).all()
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-12)


class TestPlotSimilarityHeatmapPair:
    @staticmethod
    def _matrix(n=10):
        rng = np.random.default_rng(1)
        v = rng.normal(0, 1, (n, 64))
        return SC.cosine_similarity_matrix(v)

    def test_renders_png_file(self, tmp_path):
        from KNN_evaluation.visualization import plot_similarity_heatmap_pair
        out = tmp_path / "h.png"
        plot_similarity_heatmap_pair(self._matrix(), self._matrix(), out)
        assert out.exists() and out.stat().st_size > 0

    def test_renders_to_bytesio(self):
        from KNN_evaluation.visualization import plot_similarity_heatmap_pair
        buf = io.BytesIO()
        plot_similarity_heatmap_pair(self._matrix(), self._matrix(), buf)
        assert buf.getvalue().startswith(b"\x89PNG")

    def test_unified_color_scale(self, tmp_path, monkeypatch):
        import matplotlib.pyplot as plt
        from KNN_evaluation.visualization import plot_similarity_heatmap_pair
        captured = []
        orig = plt.Axes.imshow

        def spy(self, data, **kw):
            captured.append((kw.get("vmin"), kw.get("vmax")))
            return orig(self, data, **kw)

        monkeypatch.setattr(plt.Axes, "imshow", spy)
        plot_similarity_heatmap_pair(
            self._matrix(5), self._matrix(8), tmp_path / "h.png",
        )
        assert len(captured) == 2
        assert captured[0] == captured[1], "两子图必须统一 vmin/vmax 色阶"
        assert captured[0][0] is not None and captured[0][1] is not None


class TestExportSimilarityOutputs:
    """Task 8: export_similarity_outputs — npy 矩阵 + JSON 采样信息导出."""

    def _pixels(self, n=3):
        return [
            {
                "point_id": f"p{i}", "image_id": "img",
                "pixel_row": i, "pixel_col": i * 2,
                "utm_easting": 500000.0 + i, "utm_northing": 4000000.0 + i,
                "utm_zone": "50N",
            }
            for i in range(n)
        ]

    def test_writes_npy_and_json_auto_creates_dir(self, tmp_path):
        export_dir = tmp_path / "sub" / "dir"  # 目录不存在，应自动创建
        sim_g = np.array([[1.0, 0.5], [0.5, 1.0]])
        sim_x = np.array([[1.0, -0.2], [-0.2, 1.0]])
        meta = {"n": 2, "seed": 42, "image_id": "img", "collections": ["g", "x"],
                "sampled": 2, "kept": 2, "dropped": 0, "elapsed_sec": 0.1}
        pixels = self._pixels(2)
        files = SC.export_similarity_outputs(
            sim_g, sim_x, meta, pixels, export_dir, ("google", "xian"),
        )
        # 返回文件路径列表（npy × 2 + json × 1）
        assert len(files) == 3
        assert all(Path(f).exists() for f in files)
        # 两个 npy 维度正确
        np.testing.assert_allclose(np.load(export_dir / "google_similarity.npy"), sim_g)
        np.testing.assert_allclose(np.load(export_dir / "xian_similarity.npy"), sim_x)
        # json 含 params + pixels，且 pixels 与输入行序一致
        payload = json.loads((export_dir / "similarity_sampling.json").read_text(
            encoding="utf-8",
        ))
        assert payload["params"] == meta
        assert payload["pixels"] == pixels

    def test_json_uses_ensure_ascii_false(self, tmp_path):
        sim = np.eye(1)
        meta = {"n": 1, "seed": 1, "image_id": None, "collections": ["g", "x"],
                "sampled": 1, "kept": 1, "dropped": 0, "elapsed_sec": 0.0}
        SC.export_similarity_outputs(sim, sim, meta, self._pixels(1), tmp_path, ("g", "x"))
        raw = (tmp_path / "similarity_sampling.json").read_text(encoding="utf-8")
        assert "null" in raw
        assert "params" in raw and "pixels" in raw

    def test_prefix_adds_filename_prefix(self, tmp_path):
        """prefix 非空时 npy×2 与 sampling json 文件名带前缀（WebUI 全库/图片区分）."""
        sim = np.eye(1)
        meta = {"n": 1, "seed": 1, "image_id": None, "collections": ["g", "x"],
                "sampled": 1, "kept": 1, "dropped": 0, "elapsed_sec": 0.0}
        files = SC.export_similarity_outputs(
            sim, sim, meta, self._pixels(1), tmp_path, ("google", "xian"),
            prefix="full_col_",
        )
        names = {Path(f).name for f in files}
        assert "full_col_google_similarity.npy" in names
        assert "full_col_xian_similarity.npy" in names
        assert "full_col_similarity_sampling.json" in names
        assert (tmp_path / "full_col_similarity_sampling.json").exists()


class TestCompareSimilarityHeatmaps:
    def test_returns_metadata_and_writes_file(self, tmp_path, monkeypatch):
        out = tmp_path / "h.png"
        points = [{"point_id": f"p{i}"} for i in range(3)]
        monkeypatch.setattr(
            SC, "sample_random_points",
            lambda mgr, n, seed, image_id=None: points,
        )
        monkeypatch.setattr(
            SC, "extract_embeddings",
            lambda pts, gm, xm: (np.eye(3), np.eye(3), 0, []),
        )
        monkeypatch.setattr(
            SC, "plot_similarity_heatmap_pair",
            lambda a, b, out, collection_names=None: out.write_bytes(b"PNG"),
        )
        g, x = _manager("g"), _manager("x")
        result = SC.compare_similarity_heatmaps(g, x, n=3, seed=42, output=str(out))
        assert result["sampled"] == 3
        assert result["kept"] == 3
        assert result["dropped"] == 0
        assert result["matrix_shape"] == [3, 3]
        assert result["output_path"] == str(out)
        assert out.exists()

    def test_bytesio_output_path_empty(self, monkeypatch):
        buf = io.BytesIO()
        monkeypatch.setattr(
            SC, "sample_random_points",
            lambda mgr, n, seed, image_id=None: [{"point_id": "p0"}, {"point_id": "p1"}],
        )
        monkeypatch.setattr(
            SC, "extract_embeddings",
            lambda pts, gm, xm: (np.eye(2), np.eye(2), 0, []),
        )
        monkeypatch.setattr(
            SC, "plot_similarity_heatmap_pair",
            lambda a, b, out, collection_names=None: out.write(b"PNG"),
        )
        g, x = _manager("g"), _manager("x")
        result = SC.compare_similarity_heatmaps(g, x, n=2, seed=1, output=buf)
        assert result["output_path"] == ""
        assert buf.getvalue() == b"PNG"

    def test_default_export_dir_is_outputs(self, monkeypatch):
        """Task 9: 默认 export_dir='outputs'——不显式传参时导出到 outputs/ 并写入元数据."""
        monkeypatch.setattr(
            SC, "sample_random_points",
            lambda mgr, n, seed, image_id=None: [{"point_id": "p0"}],
        )
        monkeypatch.setattr(
            SC, "extract_embeddings",
            lambda pts, gm, xm: (np.eye(1), np.eye(1), 0, []),
        )
        monkeypatch.setattr(
            SC, "plot_similarity_heatmap_pair",
            lambda a, b, out, collection_names=None: None,
        )
        captured: dict = {}
        fake_files = ["a.npy", "b.npy", "s.json"]

        def fake_export(sim_g, sim_x, meta, pixels, export_dir, collection_names, prefix="", export_npy=True):
            captured["export_dir"] = export_dir
            captured["prefix"] = prefix
            captured["export_npy"] = export_npy
            return fake_files

        monkeypatch.setattr(SC, "export_similarity_outputs", fake_export)
        g, x = _manager("g"), _manager("x")
        result = SC.compare_similarity_heatmaps(g, x, n=1, seed=1, output=io.BytesIO())
        assert captured["export_dir"] == "outputs"
        assert captured["prefix"] == "", "默认无前缀"
        assert result["exported_files"] == fake_files

    def test_explicit_none_disables_export(self, monkeypatch):
        """显式 export_dir=None：禁用导出（向后兼容），不调用导出函数."""
        monkeypatch.setattr(
            SC, "sample_random_points",
            lambda mgr, n, seed, image_id=None: [{"point_id": "p0"}],
        )
        monkeypatch.setattr(
            SC, "extract_embeddings",
            lambda pts, gm, xm: (np.eye(1), np.eye(1), 0, []),
        )
        monkeypatch.setattr(
            SC, "plot_similarity_heatmap_pair",
            lambda a, b, out, collection_names=None: None,
        )
        calls: list = []

        def fake_export(*a, **kw):
            calls.append((a, kw))
            return []

        monkeypatch.setattr(SC, "export_similarity_outputs", fake_export)
        g, x = _manager("g"), _manager("x")
        result = SC.compare_similarity_heatmaps(
            g, x, n=1, seed=1, output=io.BytesIO(), export_dir=None,
        )
        assert calls == []
        assert "exported_files" not in result

    def test_empty_string_disables_export(self, monkeypatch):
        """空串 export_dir=''：禁用导出（避免 Path('')/mkdir('') 出错），不调用导出函数."""
        monkeypatch.setattr(
            SC, "sample_random_points",
            lambda mgr, n, seed, image_id=None: [{"point_id": "p0"}],
        )
        monkeypatch.setattr(
            SC, "extract_embeddings",
            lambda pts, gm, xm: (np.eye(1), np.eye(1), 0, []),
        )
        monkeypatch.setattr(
            SC, "plot_similarity_heatmap_pair",
            lambda a, b, out, collection_names=None: None,
        )
        calls: list = []

        def fake_export(*a, **kw):
            calls.append((a, kw))
            return []

        monkeypatch.setattr(SC, "export_similarity_outputs", fake_export)
        g, x = _manager("g"), _manager("x")
        result = SC.compare_similarity_heatmaps(
            g, x, n=1, seed=1, output=io.BytesIO(), export_dir="",
        )
        assert calls == []
        assert "exported_files" not in result

    def test_export_dir_triggers_export_and_metadata(self, tmp_path, monkeypatch):
        """指定 export_dir：调用导出函数，返回元数据含 3 个导出文件路径."""
        monkeypatch.setattr(
            SC, "sample_random_points",
            lambda mgr, n, seed, image_id=None: [{"point_id": "p0"}, {"point_id": "p1"}],
        )
        monkeypatch.setattr(
            SC, "extract_embeddings",
            lambda pts, gm, xm: (
                np.eye(2), np.eye(2), 1,
                [{"point_id": "p0", "image_id": "img", "pixel_row": 0,
                  "pixel_col": 0, "utm_easting": 1.0, "utm_northing": 2.0,
                  "utm_zone": "50N"}],
            ),
        )
        monkeypatch.setattr(
            SC, "plot_similarity_heatmap_pair",
            lambda a, b, out, collection_names=None: None,
        )
        captured: dict = {}
        fake_files = ["a.npy", "b.npy", "s.json"]

        def fake_export(sim_g, sim_x, meta, pixels, export_dir, collection_names, prefix="", export_npy=True):
            captured["sim_g"] = sim_g
            captured["sim_x"] = sim_x
            captured["meta"] = meta
            captured["pixels"] = pixels
            captured["export_dir"] = export_dir
            captured["collection_names"] = collection_names
            captured["prefix"] = prefix
            return fake_files

        monkeypatch.setattr(SC, "export_similarity_outputs", fake_export)
        g, x = _manager("g"), _manager("x")
        result = SC.compare_similarity_heatmaps(
            g, x, n=2, seed=7, output=io.BytesIO(), export_dir=tmp_path,
            collection_names=("g", "x"),
        )
        assert result["exported_files"] == fake_files
        # meta 契约：{n, seed, image_id, collections, sampled, kept, dropped, elapsed_sec}
        assert captured["meta"]["n"] == 2
        assert captured["meta"]["seed"] == 7
        assert captured["meta"]["image_id"] is None
        assert captured["meta"]["collections"] == ["g", "x"]
        assert captured["meta"]["sampled"] == 2
        assert captured["meta"]["kept"] == 2
        assert captured["meta"]["dropped"] == 1
        assert "elapsed_sec" in captured["meta"]
        # pixels 与 extract 返回的 kept_records 行序一致，原样透传
        assert captured["pixels"] == [{
            "point_id": "p0", "image_id": "img", "pixel_row": 0,
            "pixel_col": 0, "utm_easting": 1.0, "utm_northing": 2.0,
            "utm_zone": "50N",
        }]
        assert captured["export_dir"] == tmp_path
        assert captured["collection_names"] == ("g", "x")
        # 两个相似度矩阵维度正确
        assert captured["sim_g"].shape == (2, 2)
        assert captured["sim_x"].shape == (2, 2)
