"""Tests for KNN_evaluation.manifest."""
from pathlib import Path

from KNN_evaluation.manifest import load_manifest, save_manifest, update_manifest


def _sample_data():
    return {"collection": "c", "images": {"E121.4_N25.1": 16384}, "updated_at": "2026-08-02T00:00:00"}


class TestLoadManifest:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        data = load_manifest(tmp_path / "nope.json")
        assert data == {"collection": "", "images": {}, "updated_at": ""}

    def test_corrupt_file_returns_empty(self, tmp_path: Path):
        p = tmp_path / "m.json"
        p.write_text("{ not json !!", encoding="utf-8")
        assert load_manifest(p) == {"collection": "", "images": {}, "updated_at": ""}

    def test_roundtrip(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_manifest(_sample_data(), p)
        assert load_manifest(p) == _sample_data()


class TestUpdateManifest:
    def test_adds_and_updates(self, tmp_path: Path):
        p = tmp_path / "m.json"
        out = update_manifest("E121.4_N25.1", 16384, "c", p)
        assert out["images"]["E121.4_N25.1"] == 16384
        out = update_manifest("E121.4_N25.1", 500, "c", p)
        assert out["images"]["E121.4_N25.1"] == 500
        assert out["collection"] == "c"
        assert out["updated_at"]
        assert load_manifest(p)["images"]["E121.4_N25.1"] == 500

    def test_missing_file_creates(self, tmp_path: Path):
        p = tmp_path / "m.json"
        out = update_manifest("A", 10, "col", p)
        assert out["images"] == {"A": 10}

    def test_keeps_unrelated_keys(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_manifest(_sample_data(), p)
        out = update_manifest("E121.4_N25.1", 0, "c", p)
        assert set(out["images"]) == {"E121.4_N25.1"}


class TestAtomicWrite:
    def test_no_tmp_leftover(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_manifest(_sample_data(), p)
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_non_ascii_roundtrip(self, tmp_path: Path):
        p = tmp_path / "m.json"
        save_manifest({"collection": "中文名", "images": {}, "updated_at": "x"}, p)
        assert load_manifest(p)["collection"] == "中文名"


class TestSaveManifestCreatesDirectories:
    def test_parent_dir_auto_created(self, tmp_path: Path):
        """save_manifest 应在目标父目录不存在时自动创建."""
        p = tmp_path / "nested" / "sub" / "m.json"
        save_manifest(_sample_data(), p)
        assert p.exists()
        assert load_manifest(p) == _sample_data()


class TestUpdateManifestSequential:
    def test_two_updates_file_still_readable(self, tmp_path: Path):
        """顺序调用 update_manifest 两次后，文件应仍是合法 JSON 且为最后一次写值."""
        import json

        p = tmp_path / "m.json"
        update_manifest("E121.4_N25.1", 16384, "c", p)
        update_manifest("E121.4_N25.1", 500, "c", p)
        raw = json.loads(p.read_text(encoding="utf-8"))  # 未抛异常即合法
        assert raw["images"]["E121.4_N25.1"] == 500
        assert load_manifest(p)["images"]["E121.4_N25.1"] == 500


class TestReconcileManifestPureMock:
    """直接对 qdrant_client.reconcile_manifest 的纯 mock 覆盖.

    三条路径：一致（不写盘）、不一致刷新（写盘一次）、缺失重建（写盘一次）。
    所有 mock 均在 qdrant_client 命名空间内（load_manifest / save_manifest /
    _facet_image_ids）与 PixelDataLoader.check_image_count，不触碰真实文件/网络.
    """

    @staticmethod
    def _env(monkeypatch, images: dict, facet_ids: set, counts: dict):
        import KNN_evaluation.qdrant_client as qc
        from KNN_evaluation.data_loader import PixelDataLoader

        state = {"collection": "c", "images": dict(images), "updated_at": "x"}
        saved_calls: list = []

        monkeypatch.setattr(qc, "load_manifest", lambda path=None: dict(state))

        def _save(data, path=None):
            saved_calls.append(dict(data))
            state.clear()
            state.update(data)

        monkeypatch.setattr(qc, "save_manifest", _save)
        monkeypatch.setattr(
            qc.QdrantManager, "_facet_image_ids",
            lambda self, limit=1000: set(facet_ids),
        )
        monkeypatch.setattr(
            PixelDataLoader, "check_image_count",
            lambda iid, mgr: counts.get(iid, 0),
        )
        manager = qc.QdrantManager(url="http://localhost:1", collection_name="c")
        return manager, state, saved_calls

    def test_consistent_no_write(self, monkeypatch):
        """一致：DB 与 manifest 相同，直接返回且不写盘."""
        m, state, saved = self._env(monkeypatch, {"A": 16384}, {"A"}, {})
        out = m.reconcile_manifest()
        assert out["images"] == {"A": 16384}
        assert saved == []  # 一致 → 不触发 save_manifest
        assert state["images"] == {"A": 16384}

    def test_inconsistent_refresh(self, monkeypatch):
        """不一致刷新：清掉过期项 GHOST，补入新项 B（精确 count）."""
        m, state, saved = self._env(
            monkeypatch,
            {"A": 16384, "GHOST": 16384},  # GHOST 在 DB 中已不存在
            {"A", "B"},                    # B 是 DB 有而 manifest 缺的新项
            {"B": 32768},
        )
        out = m.reconcile_manifest()
        assert out["images"] == {"A": 16384, "B": 32768}
        assert len(saved) == 1  # 不一致 → 刷新写盘一次
        assert "GHOST" not in state["images"]

    def test_missing_rebuild(self, monkeypatch):
        """缺失重建：manifest 为空，全部按 facet + 精确 count 重建."""
        m, state, saved = self._env(
            monkeypatch,
            {},  # manifest 缺失/损坏/为空
            {"A", "B"},
            {"A": 16384, "B": 4096},
        )
        out = m.reconcile_manifest()
        assert out["images"] == {"A": 16384, "B": 4096}
        assert len(saved) == 1  # 重建 → 写盘一次
        assert state["images"] == {"A": 16384, "B": 4096}


class TestManifestPathIsolation:
    """Task: manifest 按 collection 隔离的路径派生与安全清洗（D4）."""

    def test_safe_token_keeps_alnum_dot_underscore_dash(self):
        from KNN_evaluation.manifest import safe_collection_token
        assert safe_collection_token("google_aef_embedding") == "google_aef_embedding"
        assert safe_collection_token("my_col-2.v1") == "my_col-2.v1"

    def test_safe_token_replaces_special_chars(self):
        from KNN_evaluation.manifest import safe_collection_token
        assert safe_collection_token("a/b\\c:d*e?f\"g<h>i|j") == "a_b_c_d_e_f_g_h_i_j"

    def test_path_traversal_has_no_separator(self):
        from KNN_evaluation.manifest import safe_collection_token, manifest_path
        token = safe_collection_token("..\\..\\etc\\passwd")
        assert "/" not in token and "\\" not in token
        p = manifest_path("x/../../y")
        assert p.name == "qdrant_import_manifest_x_.._.._y.json"
        assert p.parent == Path(".")  # 无目录穿越

    def test_manifest_path_naming(self):
        from KNN_evaluation.manifest import manifest_path
        p = manifest_path("google_aef_embedding")
        assert p.name == "qdrant_import_manifest_google_aef_embedding.json"

    def test_update_manifest_default_path_derived_from_collection(self, tmp_path, monkeypatch):
        """update_manifest 未传 path 时按 collection 派生文件名，互不覆盖."""
        monkeypatch.chdir(tmp_path)
        update_manifest("A", 16384, "google_aef_embedding")
        p = tmp_path / "qdrant_import_manifest_google_aef_embedding.json"
        assert p.exists()
        assert load_manifest(p)["images"] == {"A": 16384}
        # 另一 collection 的文件应独立
        update_manifest("B", 1024, "xian_aef_embedding")
        assert load_manifest(tmp_path / "qdrant_import_manifest_xian_aef_embedding.json")["images"] == {"B": 1024}
        assert load_manifest(p)["images"] == {"A": 16384}
