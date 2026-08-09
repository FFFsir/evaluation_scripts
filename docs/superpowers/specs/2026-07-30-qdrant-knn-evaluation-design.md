---
comet_change: qdrant-knn-evaluation
role: technical-design
canonical_spec: openspec
archived-with: 2026-07-30-qdrant-knn-evaluation
status: final
---

# Qdrant KNN 评估系统 — 深度技术设计

## 1. Qdrant Collection 精确定义

### 1.1 Collection 创建参数

```python
from qdrant_client import QdrantClient, models

client.create_collection(
    collection_name="pixel_embeddings",
    vectors_config=models.VectorParams(
        size=64,
        distance=models.Distance.COSINE,
        on_disk=False,          # 全内存存储
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,                   # 每层最大连接数
        ef_construct=100,        # 构建时搜索宽度
    ),
    quantization_config=None,    # 不使用量化，全精度 float64
)
```

### 1.2 Payload Index 创建

```python
client.create_payload_index(
    collection_name="pixel_embeddings",
    field_name="label",
    field_schema=models.IntegerIndexParams(
        type=models.IntegerIndexType.INTEGER,
        lookup=True,
        range=True,
    ),
)
# 同样为 label_name(keyword)、utm_easting(float)、utm_northing(float)、
# image_id(keyword) 创建索引
# utm_zone、pixel_row、pixel_col 不建索引（不需要过滤/聚合）
```

### 1.3 Point 结构

```python
models.PointStruct(
    id=f"{image_id}_{row}_{col}",
    vector=vector_64d.tolist(),    # float64 → Python float
    payload={
        "label": int(label),
        "label_name": LABEL_NAMES[int(label)],
        "utm_easting": float(easting),
        "utm_northing": float(northing),
        "utm_zone": int(zone),
        "image_id": image_id,
        "pixel_row": int(row),
        "pixel_col": int(col),
    },
)
```

## 2. 导入管线设计

### 2.1 总体流程

```
PixelImporter.import_directory(data_dir)
│
├── 1. scan_directory() → List[ImagePair]
│    ├── 扫描 SE/、DW/ 子目录
│    ├── 提取文件名中 E{lon}_N{lat} 坐标段
│    └── 匹配 SE(.npy/.npz) + DW(.npy) + GeoTIFF(.tif)
│
├── 2. for each ImagePair:
│    ├── check_resume(image_id)
│    │    ├── count = client.count(filter: image_id match)
│    │    ├── count == 16384 → 跳过，mark "已导入"
│    │    ├── count < 16384  → 标记 "部分导入，覆盖重传"
│    │    └── count == 0     → 正常导入
│    │
│    ├── load_se() → (64, 128, 128) float64   [复用 satellite_embedding_loader]
│    ├── load_dw() → (128, 128) uint8
│    ├── compute_utm_grid() → (128, 128) × 3 arrays (easting, northing, zone)
│    │    └── 从 GeoTIFF Affine transform 计算；缺失则填 NaN + warn
│    │
│    ├── build_points_batch(row_range) → List[PointStruct]
│    │    └── 双重循环: for row in range(128), for col in range(128)
│    │
│    ├── batch 1: rows 0-78   (79×128=10,112 points) → client.upsert()
│    └── batch 2: rows 79-127 (49×128=6,272 points)  → client.upsert()
│
├── 3. if --reindex:
│    └── client.update_collection(optimizers_config=...)
│         触发 HNSW 全量索引重建
│
└── 4. print 导入统计
     ├── 总像素数 / 影像数
     ├── 各类标签像素计数与占比
     ├── 跳过数 / 覆盖数
     ├── 总耗时 / 平均速率
     └── Collection 当前总量
```

### 2.2 内存管理

- 逐文件加载：`load_se()` 和 `load_dw()` 返回完整的 `(64,128,128)` 和 `(128,128)` 数组
- 构造 Points 后立即 upsert，完成后释放数组引用
- 生成器模式：`_build_points()` 使用 Python generator，yield 逐批 Point 列表
- 最大内存占用：单张影像约 `64×128×128×8 + 128×128×1 + 10×16384×64×8 ≈ 400KB`（无 Points 缓存时）或 `~10MB`（缓存整张影像 Points 时）

### 2.3 DW Label 解析

```python
def load_dw_data(filepath: Path) -> np.ndarray:
    raw = np.load(filepath)  # dtype[('label', 'u1')], shape (128, 128)
    if raw.dtype.names is not None:
        label_array = raw['label'].astype(np.uint8)  # 提取字段
    else:
        label_array = raw.astype(np.uint8)
    assert label_array.shape == (128, 128)
    return label_array
```

## 3. 检索接口设计

### 3.1 PixelSearcher 类

```python
class PixelSearcher:
    def __init__(self, manager: QdrantManager):
        self.client = manager.client
        self.collection = manager.collection_name

    def search(
        self,
        query_vector: np.ndarray,          # (64,) float64
        k: int = 10,
        label_filter: list[int] | None = None,
        utm_range: dict | None = None,     # {min_e, max_e, min_n, max_n}
        exact: bool = False,
        ef_search: int = 64,
    ) -> SearchResult:
        ...
```

### 3.2 Filter 构建

```python
def _build_filter(label_filter, utm_range) -> models.Filter | None:
    conditions = []

    if label_filter:
        conditions.append(
            models.FieldCondition(
                key="label",
                match=models.MatchAny(any=label_filter),
            )
        )

    if utm_range:
        conditions.extend([
            models.FieldCondition(
                key="utm_easting",
                range=models.Range(gte=utm_range["min_e"], lte=utm_range["max_e"]),
            ),
            models.FieldCondition(
                key="utm_northing",
                range=models.Range(gte=utm_range["min_n"], lte=utm_range["max_n"]),
            ),
        ])

    if not conditions:
        return None
    return models.Filter(must=conditions)  # AND 语义
```

### 3.3 搜索执行

```python
hits = client.search(
    collection_name=self.collection,
    query_vector=query_vector.tolist(),
    query_filter=qdrant_filter,
    limit=k,
    search_params=models.SearchParams(
        exact=exact,
        hnsw_ef=None if exact else ef_search,
    ),
    with_payload=True,
)
```

### 3.4 SearchResult 结构

```python
@dataclass
class SearchResult:
    hits: list[HitRecord]          # 按 score 降序
    elapsed_ms: float
    label_distribution: dict[str, int]  # {label_name: count}
    search_mode: str               # "exact" | "ann"
    query_params: dict             # 记录 k、filter 等，可复现

@dataclass
class HitRecord:
    id: str
    score: float
    label: int
    label_name: str
    utm_easting: float
    utm_northing: float
    utm_zone: int
    image_id: str
    pixel_row: int
    pixel_col: int
```

## 4. Query Vector 获取策略

| 来源 | CLI 参数 | 实现方式 |
|---|---|---|
| A. 文件 | `--query-file` | `np.load()` 读取 (64,) 或 (1,64) |
| B. 随机 | `--random` | `client.scroll(limit=1)` 取一个 point 的 vector |
| C. 指定像素 | `--query-image --query-row --query-col` | `client.scroll(filter: image_id match + pixel_row match + pixel_col match, with_vector=True)` |

互斥性：A/B/C 三者互斥，同时指定时报错。

## 5. CLI 设计

### 5.1 子命令结构

```python
# cli.py
parser = argparse.ArgumentParser(prog="knn-eval")
sub = parser.add_subparsers(dest="command")

# import
p_import = sub.add_parser("import")
p_import.add_argument("directory")
p_import.add_argument("--batch-size", type=int, default=10000)
p_import.add_argument("--no-resume", action="store_true")
p_import.add_argument("--reindex", action="store_true")
p_import.add_argument("--qdrant-url", default="http://localhost:6333")

# search
p_search = sub.add_parser("search")
group = p_search.add_mutually_exclusive_group(required=True)
group.add_argument("--query-file")
group.add_argument("--random", action="store_true")
group.add_argument("--query-spec", nargs=3, metavar=("IMAGE_ID", "ROW", "COL"))
p_search.add_argument("--k", type=int, default=10)
p_search.add_argument("--label")
p_search.add_argument("--utm-range")
p_search.add_argument("--exact", action="store_true")
p_search.add_argument("--ef-search", type=int, default=64)
p_search.add_argument("--output", choices=["table", "json"], default="table")

# stats
p_stats = sub.add_parser("stats")
p_stats.add_argument("--json", action="store_true")
```

### 5.2 错误处理

| 场景 | 行为 |
|---|---|
| Qdrant 不可达 | 连接超时 5s，报错 exit 1 |
| Collection 不存在 | 提示先执行 `import` |
| query vector 维度≠64 | 报错说明期望与实际维度 |
| 过滤无匹配 | 返回空结果，exit 0 |
| GeoTIFF 缺失 | warn + UTM 填 NaN |
| --query-file/--random/--query-spec 互斥 | argparse 自动报错 |

## 6. 文件结构

```
KNN_evaluation/
├── __init__.py
├── config.py              # Qdrant URL, collection, batch_size 等
├── label_mapping.py       # 0-8 ↔ name 映射
├── qdrant_client.py       # QdrantManager: 连接、create_collection、reindex
├── data_loader.py         # PixelDataLoader: SE/DW 加载、文件配对
├── coordinate_utils.py    # GeoTIFF → UTM 坐标计算
├── importer.py            # PixelImporter: 导入管线编排
├── searcher.py            # PixelSearcher: 检索接口
├── cli.py                 # argparse CLI 入口
└── tests/
    ├── test_data_loader.py
    ├── test_importer.py
    ├── test_searcher.py
    └── conftest.py
```

## 7. 测试策略

### 单元测试
- `test_data_loader.py`: SE 加载维度正确性、DW label 值域、文件配对逻辑、孤立文件跳过
- `test_importer.py`: Point 构造字段完整性、断点续传 count 判定、批次边界
- `test_searcher.py`: Filter 构建（单标签、多标签、UTM 范围、组合）、exact/ANN 切换、空结果处理
- 使用 pytest fixtures 提供 mock Qdrant client 避免依赖外部服务

### 集成测试
- 在本地 Qdrant Docker 环境中，用 `data_demo/` 的 7 对文件进行端到端验证
- 导入 → 校验总 point 数 = 114,688 → 随机搜索 + 标签过滤 + UTM 过滤 → 精确搜索对比 ANN

### 边界测试
- 缺失 GeoTIFF 的影像导入
- 空结果集查询
- 向量维度不匹配错误
- Qdrant 不可达的优雅退出
- 导入中断后的幂等重跑
