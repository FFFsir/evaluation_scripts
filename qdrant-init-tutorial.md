# Qdrant 环境初始化教程

> 目标：从零（或彻底重置后）初始化 Qdrant，创建本项目两个默认集合
> `xian_aef_embedding`、`google_aef_embedding`（64 维 / COSINE / disk 存储），
> 补齐 5 个 payload 索引，并验证 `http://localhost:6333` 可达。

## 适用场景

- 全新机器 / 全新 Docker 环境，首次搭建
- 排查「qdrant 不可达」后重建可用环境

## 前置条件

- Docker 已安装且守护进程已启动（`docker info` 可验证）
- Python ≥ 3.12，且已安装 `qdrant-client`（本项目用 `uv` 管理，`uv sync` 后 `uv run python` 即可）

## 目录

1. [新建容器](#步骤-2新建-qdrant-容器)
2. [创建集合](#步骤-3创建两个默认集合)
3. [创建 payload 索引](#步骤-4创建-5-个-payload-索引)
4. [验证](#步骤-5验证)



## 步骤 1：新建 qdrant 容器

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  -e QDRANT__TELEMETRY_DISABLED=true \
  qdrant/qdrant
```

| 参数 | 作用 |
|------|------|
| `--name qdrant` | 容器名，便于后续管理 |
| `-p 6333:6333` | HTTP REST API 端口映射（web / CLI 连的就是它，**必须**） |
| `-p 6334:6334` | gRPC 端口映射 |
| `-v qdrant_data:/qdrant/storage` | 数据持久化到命名 volume，容器重建不丢数据（**必须**） |
| `-e QDRANT__TELEMETRY_DISABLED=true` | 关闭匿名遥测上报，避免日志刷 `Failed to report telemetry ... telemetry.qdrant.io` 噪音 |

检查容器状态：

```bash
docker ps --filter name=qdrant
# 期望：Status 为 Up，PORTS 显示 0.0.0.0:6333-6334->6333-6334/tcp
```

## 步骤 2：创建两个默认集合

集合参数与 `KNN_evaluation/config.py` / `qdrant_client.py` 保持一致：

| 参数 | 值 | 对应配置 |
|------|-----|---------|
| 向量维度 | `size=64` | `VECTOR_SIZE` |
| 距离度量 | `COSINE` | `models.Distance.COSINE` |
| 向量落盘 | `on_disk=True` | 磁盘化存储，降低常驻内存 |
| payload 落盘 | `on_disk_payload=True` | 同上 |
| HNSW `m` / `ef_construct` | `16` / `100` | `HNSW_M` / `HNSW_EF_CONSTRUCT` |
| 量化 | 关闭 | `quantization_config=None` |

```python
# setup_collections.py —— 在项目根目录执行：uv run python setup_collections.py
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333", timeout=10)

COLLECTIONS = ["xian_aef_embedding", "google_aef_embedding"]

for name in COLLECTIONS:
    if client.collection_exists(name):
        print(f"[skip] {name} 已存在")
        continue
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=64,
            distance=models.Distance.COSINE,
            on_disk=True,
        ),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        on_disk_payload=True,
        quantization_config=None,
    )
    print(f"[ok] {name} 创建成功")
```

## 步骤 3：创建 5 个 payload 索引

索引字段与类型（与 `QdrantManager.create_payload_indices` 一致）：

| 字段 | 索引类型 | 用途 |
|------|----------|------|
| `label` | integer（lookup + range） | 标签过滤 |
| `label_name` | text | 标签名过滤 |
| `utm_easting` | float | UTM 范围过滤 |
| `utm_northing` | float | UTM 范围过滤 |
| `image_id` | keyword | 精确匹配（检索 / 分页 / manifest 对账） |

> **为什么要建**：缺少索引时过滤 / 聚合会触发全量 payload 扫描。实测 xian 集合约 1000 万点，
> 无索引时 UTM 过滤耗时 5.03s，超过 `QDRANT_TIMEOUT=5s` 触发超时；补建索引后约 274ms。

```python
# setup_payload_indices.py —— 在项目根目录执行：uv run python setup_payload_indices.py
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333", timeout=10)

INDICES = [
    ("label", models.IntegerIndexParams(
        type=models.IntegerIndexType.INTEGER, lookup=True, range=True)),
    ("label_name", models.TextIndexParams(type=models.TextIndexType.TEXT)),
    ("utm_easting", models.FloatIndexParams(type=models.FloatIndexType.FLOAT)),
    ("utm_northing", models.FloatIndexParams(type=models.FloatIndexType.FLOAT)),
    ("image_id", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)),
]

for name in ["xian_aef_embedding", "google_aef_embedding"]:
    schema = (client.get_collection(name).payload_schema) or {}
    for field_name, field_schema in INDICES:
        if schema.get(field_name) is not None:
            continue  # 已存在，跳过（幂等）
        client.create_payload_index(
            collection_name=name, field_name=field_name, field_schema=field_schema,
        )
    keys = sorted(((client.get_collection(name).payload_schema) or {}).keys())
    print(f"[ok] {name} 索引: {keys}")
```

更省事的等价做法（直接复用项目代码路径，与 web/CLI 完全一致）：

```python
from KNN_evaluation.qdrant_client import QdrantManager

for name in ["xian_aef_embedding", "google_aef_embedding"]:
    QdrantManager(collection_name=name).create_payload_indices()
```

## 步骤 4：验证

```bash
# 5.1 服务可达
curl http://localhost:6333/
# 期望：{"title":"qdrant - vector search engine",...}

# 5.2 集合存在
curl http://localhost:6333/collections
# 期望：collections 包含 xian_aef_embedding、google_aef_embedding

# 5.3 索引就位（payload_schema 含上述 5 个字段）
curl -s http://localhost:6333/collections/xian_aef_embedding | python -m json.tool
```

WebUI 验证（可选）：

```bash
uv run python KNN_evaluation/webui.py --port 8003
```

打开 http://localhost:8003，页面应显示 Qdrant 连接正常，可切换 `google_aef_embedding` / `xian_aef_embedding` 分页。

## 常见问题

| 现象 | 原因与处理 |
|------|-----------|
| 日志刷 `Failed to report telemetry ... telemetry.qdrant.io` | Qdrant 匿名遥测上报失败（容器无法访问外网）。**不影响服务**，用 `-e QDRANT__TELEMETRY_DISABLED=true` 关闭 |
| web 报「qdrant 不可达」 | 容器未映射端口（`docker ps` 看 PORTS 为空）或容器未运行；重建时务必带 `-p 6333:6333` |
| `Bind for 0.0.0.0:6333 failed: port is already allocated` | 端口被占用，`docker ps -a` 找到占用容器 stop / rm 后重试 |
| 集合存在但数据为空 | 集合只建了骨架，需用 CLI 导入数据：`uv run python -m KNN_evaluation.cli import data_xian --collection xian_aef_embedding` |
| 删容器后数据丢失 | 创建容器时必须挂 `-v qdrant_data:/qdrant/storage`，数据在 volume 里，不随容器删除 |

## 备注

- 基础启动命令见根目录 `README.md`；本教程覆盖「重置 → 建容器 → 建集合 → 建索引 → 验证」的完整初始化流程。
- 集合名、维度、索引字段均与 `KNN_evaluation/config.py` 和 `KNN_evaluation/qdrant_client.py` 对齐，web / CLI 开箱可用。
