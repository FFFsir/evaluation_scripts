"""土地覆盖标签映射表（0-8 <-> 名称）.

直接复用 KNN_evaluation.label_mapping 的权威定义，保证整个评估
系统内标签语义单一事实源、不漂移。后续新增 MLP_prob（软分类概率
网络）时同样引用本映射表。
"""
from KNN_evaluation.label_mapping import LABEL_NAMES, LABEL_IDS

__all__ = ["LABEL_NAMES", "LABEL_IDS"]
