"""Qdrant KNN 评估系统配置."""

QDRANT_URL = "http://localhost:6333"

# 预置 Qdrant Collection 与默认值（D1）：
# DEFAULT_COLLECTION 供 CLI/WebUI 未指定时使用；PRESET_COLLECTIONS 为预置选择项；
# COLLECTION_NAME 作为 DEFAULT_COLLECTION 的别名保留，避免大范围替换引用、
# 保持 QdrantManager 默认参数兼容。
DEFAULT_COLLECTION = "google_aef_embedding"
PRESET_COLLECTIONS = ["google_aef_embedding", "xian_aef_embedding"]
COLLECTION_NAME = DEFAULT_COLLECTION

# 预置分页默认数据目录映射（相对项目根，Task 9）：
# WebUI 分页切换时，预置 collection 的数据目录输入框与 state["data_dir"]
# 自动切换为该映射目录；不在映射中的自定义 collection 保持 --dir 指定目录。
COLLECTION_DATA_DIRS = {
    "google_aef_embedding": "data_google",
    "xian_aef_embedding": "data_xian",
}

BATCH_SIZE = 10000
VECTOR_SIZE = 64
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
EF_SEARCH_DEFAULT = 64
QDRANT_TIMEOUT = 60
UTM_RESOLUTION_M = 10  # UTM 坐标推算分辨率（米/像素）
