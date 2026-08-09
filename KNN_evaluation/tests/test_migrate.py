"""Tests for cli.cmd_migrate idempotency (all mocks, no real Qdrant).

覆盖 Task 3.3/3.4 的迁移编排三态：
- collection 已存在 → 备份旧统计后删除重建，返回 0
- collection 不存在 → 跳过删除，直接创建，返回 0
- no_resume 部分导入 → 编排顺序正常，返回 0
- 幂等可重试：连续执行两次不抛错

数据安全前提（Task 3.3 fix）：cmd_migrate 在删除前必须通过
`cli._container_has_volume()` 校验容器挂载 qdrant_data 卷；无卷时 fail-fast
返回 1，绝不执行删除。测试将 `_container_has_volume` mock 为 True 以覆盖正常
迁移路径，并单独断言无卷时中止且不删除。
"""
from pathlib import Path

from KNN_evaluation import cli


class _Args:
    def __init__(self, storage="disk", no_resume=False, dir="data_demo",
                 qdrant_url="http://localhost:1", collection="c"):
        self.storage = storage
        self.no_resume = no_resume
        self.dir = dir
        self.qdrant_url = qdrant_url
        self.collection = collection


class _FakeClient:
    def __init__(self):
        self.deleted = []

    def delete_collection(self, name):
        self.deleted.append(name)


class _FakeManager:
    collection_name = "c"

    def __init__(self, url=None, collection_name="c", timeout=10, existed=True):
        self.url = url
        self.collection_name = collection_name
        self.existed = existed
        self.client = _FakeClient()
        self.created_storage = None
        self.reconcile_calls = 0

    def health_check(self):
        return True

    def collection_exists(self):
        return self.existed

    def collection_info(self):
        return {"total_points": 1000}

    def create_collection(self, storage="disk"):
        self.created_storage = storage

    def create_payload_indices(self):
        pass

    def migrate_image_id_index(self):
        pass

    def reconcile_manifest(self):
        self.reconcile_calls += 1
        return {"collection": self.collection_name, "images": {}, "updated_at": "x"}


class _FakeImporter:
    def __init__(self, manager, batch_size=None):
        self.manager = manager
        self.imports = []

    def import_directory(self, data_dir, no_resume=False, reindex=False):
        self.imports.append((data_dir, no_resume, reindex))
        return {"total_pixels": 1000, "imported_images": 1, "skipped_images": 0}


def _patch(monkeypatch, existed=True) -> _FakeManager:
    mgr = _FakeManager(existed=existed)
    monkeypatch.setattr(cli, "QdrantManager", lambda url=None, collection_name="c", timeout=10: mgr)
    monkeypatch.setattr(cli, "PixelImporter", _FakeImporter)
    monkeypatch.setattr(cli, "_start_qdrant", lambda: True)
    # cmd_migrate 删除前必须确认容器挂载 qdrant_data 卷（Task 3.3 fix）
    monkeypatch.setattr(cli, "_container_has_volume", lambda name="qdrant": True)
    return mgr


def test_migrate_existing_collection_recreates(monkeypatch):
    mgr = _patch(monkeypatch, existed=True)
    assert cli.cmd_migrate(_Args(storage="disk")) == 0
    assert mgr.client.deleted == ["c"]          # 删除重建
    assert mgr.created_storage == "disk"        # 新存储预设
    assert mgr.reconcile_calls == 1             # 迁移后重建 manifest


def test_migrate_missing_collection_skips_delete(monkeypatch):
    mgr = _patch(monkeypatch, existed=False)
    assert cli.cmd_migrate(_Args(storage="disk")) == 0
    assert mgr.client.deleted == []             # 不存在则无删除
    assert mgr.created_storage == "disk"
    assert mgr.reconcile_calls == 1


def test_migrate_no_resume_partial(monkeypatch):
    mgr = _patch(monkeypatch, existed=True)
    importer = _FakeImporter(mgr)
    monkeypatch.setattr(cli, "PixelImporter", lambda manager: importer)
    assert cli.cmd_migrate(_Args(no_resume=True)) == 0
    # no_resume 透传给 import_directory 并强制 reindex
    assert importer.imports == [(Path("data_demo"), True, True)]


def test_migrate_idempotent_rerun(monkeypatch):
    mgr = _patch(monkeypatch, existed=True)
    assert cli.cmd_migrate(_Args()) == 0
    assert cli.cmd_migrate(_Args()) == 0        # 二次执行不抛错（幂等可重试）
    # 两次均删除重建，且无异常
    assert mgr.client.deleted == ["c", "c"]
    assert mgr.created_storage == "disk"
    assert mgr.reconcile_calls == 2


def test_migrate_aborts_without_deleting_when_volume_missing(monkeypatch):
    mgr = _patch(monkeypatch, existed=True)
    # 覆盖无 volume 保障的 fail-fast 路径：mock 返回 False
    monkeypatch.setattr(cli, "_container_has_volume", lambda name="qdrant": False)
    assert cli.cmd_migrate(_Args()) == 1
    assert mgr.client.deleted == []             # 无卷绝不删除
    assert mgr.created_storage is None          # 也不重建
    assert mgr.reconcile_calls == 0
