"""Qdrant Linear Probe 评估系统配置."""
from pathlib import Path

# ---------- Qdrant ----------
QDRANT_URL = "http://localhost:6333"

# 预置 Qdrant Collection 与默认值：
# DEFAULT_COLLECTION 供 CLI/WebUI 未指定时使用；PRESET_COLLECTIONS 为预置选择项；
# COLLECTION_NAME 作为 DEFAULT_COLLECTION 的别名保留，避免大范围替换引用、
# 保持 QdrantManager 默认参数兼容。
DEFAULT_COLLECTION = "xian_aef_embedding"
PRESET_COLLECTIONS = ["google_aef_embedding", "xian_aef_embedding"]
COLLECTION_NAME = DEFAULT_COLLECTION

QDRANT_TIMEOUT = 10
SCROLL_BATCH_SIZE = 10000  # scroll 每批点数

# ---------- 数据模型 ----------
VECTOR_SIZE = 64       # 像素 embedding 维度（1 像素 = 1 个 64 维向量）
NUM_CLASSES = 9        # DW 硬分类标签数（0-8）
DEFAULT_LABEL_INDEX = 0  # payload 中标签字段名（整数 0-8）

# ---------- 训练默认超参 ----------
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 40960
DEFAULT_LR = 1e-2
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_OPTIMIZER = "adam"  # adam | sgd
DEFAULT_TRAIN_PER_CLASS = 10000  # 每类最多进入训练集的样本数（0 = 不限制，取该类全部）
DEFAULT_VAL_PER_CLASS = 1000     # 每类最多进入验证集的样本数（0 = 不限制，取该类全部）
DEFAULT_VAL_RATIO = 0.2         # 先按比例划分出验证集，再受 VAL_PER_CLASS 上限约束
DEFAULT_SEED = 42
DEFAULT_LOG_INTERVAL = 20       # 每 N 个 batch 回调一次训练进度

# ---------- 输出 ----------
DEFAULT_OUTPUT_DIR = Path("outputs/mlp_label")
