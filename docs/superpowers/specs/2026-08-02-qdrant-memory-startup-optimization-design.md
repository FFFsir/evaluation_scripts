---
comet_change: qdrant-memory-startup-optimization
role: technical-design
canonical_spec: openspec
archived-with: 2026-08-02-qdrant-memory-startup-optimization
status: final
---

# Qdrant 内存占用与 Web 启动优化 — 深度技术设计

## 1. 背景与问题

系统已全量导入 **1022 万点**（624 张 SE/DW 影像），运行于 16GB RAM。三个叠加问题（详见 `openspec/changes/qdrant-memory-startup-optimization/proposal.md`）：

- **P1 Web 启动阻塞**：`init_page` 同步执行 → `get_imported_image_ids()` 全库逐页 scroll（约 1023 次 HTTP）阻塞事件循环数分钟。
- **P2 Qdrant 全量常驻内存**：`on_disk=False` + 无量化 → 6-10GB；Docker 无 volume 持久化。
- **P3 评估批量路径重复装载全库**：`_scroll_full_vectors` 5.2GB + `_batch_exact_knn` 中间数组 → Web 进程峰值 11-16GB OOM。

## 2. 目标 / 非目标

**目标：**
- Web 启动秒级显示连接状态、影像列表毫秒级可读（本地 manifest）。
- Qdrant 常驻内存 ≈6-10GB → ≈2-3GB（磁盘化，HNSW 留 RAM）。
- 评估默认路径内存安全（服务端逐条），批量路径显式 opt-in + 内存守卫。
- Docker 数据持久化 + 启动幂等。

**非目标：**
- 不引入 GPU / faiss / annoy（属并行 change `gpu-knn-scale-evaluation`）。
- 不改 Collection schema、payload 结构、`searcher.py` 语义。
- 不改变 `import_directory` 断点续传/重试逻辑（仅复用）。

## 3. 已确认的关键决策

### 3.1 对账 API = `facet`（非 distinct）

**发现**：`qdrant-client 1.18.0` 无 `distinct()` 方法。`dir(QdrantClient)` 确认无 distinct/group_by，但有 `facet(key, limit, exact)`。

**决策**：`reconcile_manifest()` 用 `client.facet(key="image_id", limit=1000, exact=True)` 一次获取去重 image_id 集合，与 manifest 对比。单次请求、走 keyword 索引、毫秒级。

**备选**：`query_points_groups(group_by=...)`（可按字段分组，但语义偏检索）、scroll 全库（1023 次 HTTP，与 P1 目标冲突）。选 facet。

### 3.2 内存守卫范围 = 仅客户端进程

**决策**：`estimate_batch_memory(n_points, n_queries, k)` 估算 Web/CLI 进程内批量路径峰值：

| 项 | 公式 | @10M 点 |
|---|---|---|
| all_vecs | N×64×8B | 5.2GB |
| a_norm | N×64×8B | 5.2GB |
| topk idxs | Q×K×8B | Q×K 相关 |
| labels/ids | N×8B | 0.1GB |
| **合计** | | **≈10.4GB** |

`guard_batch_memory(manager, n_queries, k, max_ram_gb=6.0)`：超过阈值抛 `MemoryError` 含预估。不含 Qdrant 自身常驻（磁盘化后 2-3GB 单独计入）。

### 3.3 manifest 并发 = 原子写 + 后者胜

**决策**：`save_manifest` 写 `manifest.tmp` → `os.replace` 原子替换。对账线程与导入线程并发写时最后一次写胜出，下次启动对账自动纠正。无需文件锁（manifest 是可重建缓存，非唯一真相）。

### 3.4 init_page = 单后台协程

**决策**：`init_page` 快速路径（创建 manager → health_check → 置 ✅ → `sleep(0)` 让出事件循环）+ 慢速路径（`asyncio.create_task(_background_init)` 顺序执行 load_manifest / scan_directory / reconcile / render）。

### 3.5 create_collection 默认 `storage="disk"`

**决策**：`create_collection(storage="disk")` 默认磁盘化；`ram` 预设保留兼容。与 `--storage` CLI 参数（默认 disk）一致。

### 3.6 migrate 开头自动调用幂等 `_start_qdrant()`

**决策**：迁移前确保 Qdrant 容器挂 volume 就绪，防删除重建后容器重建丢数据。

## 4. 详细设计

### 4.1 manifest.py（新增）

```python
"""导入 manifest：collection 级导入清单的读取/保存/增量更新（原子写）."""
import json, os, tempfile
from pathlib import Path

MANIFEST_PATH = Path("qdrant_import_manifest.json")

def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """读取 manifest，返回 {collection, images: {image_id: pixels}, updated_at}。
    文件缺失/损坏返回空结构（不报错），由对账路径重建."""
    ...

def save_manifest(data: dict, path: Path = MANIFEST_PATH) -> None:
    """原子写：tmp + os.replace."""
    ...

def update_manifest(image_id: str, imported_pixels: int,
                    collection: str, path: Path = MANIFEST_PATH) -> dict:
    """增量更新单张影像的已导入像素数."""
    ...
```

**原子写实现**：
```python
fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
with os.fdopen(fd, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
os.replace(tmp, path)
```

### 4.2 qdrant_client.py（修改）

**`get_imported_image_ids()` 改读 manifest**：
```python
def get_imported_image_ids(self) -> set[str]:
    """从本地 manifest 读取已导入 image_id（毫秒级，Qdrant 离线可用）.
    无 manifest 时回退 facet 重建."""
    data = load_manifest()
    return set((data.get("images") or {}).keys())
```

**新增 `reconcile_manifest()`**：
```python
def reconcile_manifest(self) -> dict:
    """用 facet 对账 manifest：一致不改、不一致刷新、缺失重建.
    Returns: 对账后的 manifest dict."""
    db_ids = self._facet_image_ids()
    manifest_ids = set((load_manifest().get("images") or {}).keys())
    if db_ids != manifest_ids:
        images = {iid: 16384 for iid in db_ids}   # 对账用 facet 只能得集合，像素数需 count
        # 或：对差异项单独 count 精确像素数
        save_manifest({"collection": ..., "images": images, "updated_at": ...})
    return load_manifest()

def _facet_image_ids(self, limit: int = 1000) -> set[str]:
    resp = self.client.facet(
        collection_name=self.collection_name,
        key="image_id",
        limit=limit,
        exact=True,
    )
    return {v.value for v in resp.hits}
```

> **注意点**：`facet` 只返回去重值集合，**不含每张影像的像素数**。对账时若 `db_ids == manifest_ids` 无需动作；若不一致，需对差异项用 `check_image_count` 补精确像素数，或统一回填 16384（仅当差异影像都完整导入时）。**设计决策**：对差异项精确 count（复用 `check_image_count`），保证预览状态列准确。

### 4.3 webui.py（修改）

**init_page 重构**：
```python
async def init_page():
    manager = QdrantManager(url=_CLI_QDRANT_URL)
    state["manager"] = manager
    state["data_dir"] = Path(_CLI_DATA_DIR)
    refresh_status()                  # 健康检查 + 置 ✅
    await asyncio.sleep(0)            # 让出事件循环，状态 flush 到浏览器
    asyncio.create_task(_background_init())   # 慢速路径

async def _background_init():
    try:
        await asyncio.to_thread(_load_manifest_cached)
        if state["data_dir"].exists():
            await asyncio.to_thread(_scan_directory_only)
        await asyncio.to_thread(_reconcile_background)   # facet 对账
        await _refresh_image_list()
        await _render_preview()
    except Exception:
        pass
```

**进程级缓存**：
```python
_manifest_cache: dict | None = None

def _get_manifest_cached() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = load_manifest()
    return _manifest_cache

def _invalidate_manifest_cache():
    global _manifest_cache
    _manifest_cache = None
```

**`_refresh_image_list` / `_render_preview` 异步化**：改为 async，内部 `get_imported_image_ids`（读 manifest）与 `check_image_count`（读 manifest 字典）不再需要 HTTP，秒回。

**do_import 完成后**：`_invalidate_manifest_cache()` + `await browse_directory()`。

### 4.4 importer.py（修改）

**import_directory 循环内**，单张影像导入/跳过完成后：
```python
from KNN_evaluation.manifest import update_manifest
...
# 在 pair 循环末尾（已算 imported/skipped 与 label 统计后）
if progress_callback is None or True:   # 无条件更新
    update_manifest(pair.image_id, 16384 if imported else existing_count,
                    self.manager.collection_name)
```

> **注意**：部分导入（`existing_count > 0` 覆盖重传）时，`import_image_pair` 返回 `(total, existing_count)`，实际导入 `total` 像素。manifest 记录最终像素数（16384 或 count 值）。**设计决策**：导入成功（含覆盖重传）后统一记录 16384；跳过（`imported=0, skip=total`）时记录 count 值。

### 4.5 cli.py（修改）

**evaluate 子命令**：
```python
p_eval.add_argument("--batch", action="store_true",
                    help="显式开启 numpy 批量路径（默认服务端逐条 exact，内存安全）")
p_eval.add_argument("--max-eval-ram", type=float, default=6.0,
                    help="批量路径内存守卫阈值 GB (默认: 6)")

# cmd_evaluate 内
use_batch = args.batch
if use_batch:
    guard_batch_memory(manager, num_queries,
                       max(k_values + [args.k_f1]), args.max_eval_ram)
# 调用时传 use_batch=use_batch
```

**新增 migrate 子命令**：
```python
p_migrate = sub.add_parser("migrate", help="重建 Collection 为指定存储配置并重导数据")
p_migrate.add_argument("--dir", default="data_demo", help="数据根目录")
p_migrate.add_argument("--storage", choices=["disk", "ram"], default="disk",
                       help="新 Collection 存储预设 (默认: disk)")
p_migrate.add_argument("--no-resume", action="store_true", help="强制重新导入")

def cmd_migrate(args):
    manager = QdrantManager(url=args.qdrant_url)
    if not manager.health_check():
        _start_qdrant()                 # 幂等，确保容器挂 volume 就绪
        if not manager.health_check():
            print("Qdrant 不可达", file=sys.stderr); return 1
    old_info = manager.collection_info() if manager.collection_exists() else None
    # 备份旧统计（供对照）
    if manager.collection_exists():
        manager.client.delete_collection(manager.collection_name)
    manager.create_collection(storage=args.storage)
    manager.create_payload_indices()
    manager.migrate_image_id_index()
    importer = PixelImporter(manager)
    stats = importer.import_directory(Path(args.dir), no_resume=args.no_resume, reindex=True)
    manager.reconcile_manifest()        # 重建 manifest
    new_info = manager.collection_info()
    print(f"迁移完成: {old_info['total_points'] if old_info else 0:,} → {new_info['total_points']:,}")
    return 0
```

### 4.6 metrics.py（修改）

**use_batch 默认 False** + 内存守卫：
```python
def estimate_batch_memory(n_points, n_queries, k) -> dict:
    ...

def guard_batch_memory(manager, n_queries, k, max_ram_gb=6.0):
    n = manager.collection_info()["total_points"]
    est = estimate_batch_memory(n, n_queries, k)
    if est["total_gb"] > max_ram_gb:
        raise MemoryError(f"批量路径预估峰值 {est['total_gb']:.1f}GB 超阈值 {max_ram_gb}GB，"
                          f"请降采样或使用默认服务端逐条路径")
    return est
```

### 4.7 README.md（修改）

更新启动命令：
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
```
补充「数据持久化」说明与 `migrate` 用法。

## 5. 风险 / 取舍

| 风险 | 缓解 |
|---|---|
| 磁盘化后 ANN 冷启动慢数倍 | 首次查询可接受（随机访问几千 64 维向量仅几 MB），page cache 暖后稳定 |
| 磁盘化后批量 exact 全扫描冲刷 page cache | 默认路径移回服务端逐条；`--batch` 显式开启 + 内存守卫 |
| manifest 与数据库漂移 | 后台 facet 对账兜底；manifest 是可重建缓存 |
| facet 不返回像素数 | 对差异项用 `check_image_count` 补精确值 |
| 多 tab 进程级缓存陈旧 | 对账线程更新缓存；导入完成后失效 |
| `_start_qdrant()` docker CLI 依赖 | 失败返回 False，不影响既有 WebUI（仅状态区提示） |
| 迁移重导 624 张一次性成本高 | 断点续传可中断重试；失败不破坏旧数据（先挂 volume） |

## 6. 迁移计划

1. **部署**：先运行 `migrate --storage disk`（自动挂 volume 容器）完成重建 + 重导 + manifest。
2. **回滚**：迁移失败时旧数据保留（删除前记录 collection_info）；容器数据 volume 持久化。
3. **Docker**：更新启动命令挂 volume；`_start_qdrant()` 幂等。

## 7. 测试策略

见 `openspec/changes/qdrant-memory-startup-optimization/tasks.md` §5。新增 `test_manifest.py`、`test_webui.py`；扩展 `test_qdrant_client.py`、`test_metrics.py`。磁盘化参数断言（`on_disk=True`/`on_disk_payload=True`/`quantization_config=None`）用 mock 验证。

## 8. Open Questions

- `--max-eval-ram` 默认 6GB 是否合适 → 可配置，GPU change 落地后可调。
- manifest 路径固定项目根 → build 阶段按需评估 CLI 覆盖参数。
