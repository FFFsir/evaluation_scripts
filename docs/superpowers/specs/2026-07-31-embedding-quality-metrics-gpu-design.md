---
comet_change: embedding-quality-metrics
role: technical-design
canonical_spec: openspec
archived-with: 2026-07-31-embedding-quality-metrics
status: final
---

# Embedding 质量评估模块（含 GPU 加速） — 技术设计

## 1. 架构概览



- **metrics.py**：三个公开函数 + GPU 加速内部函数 - **Qdrant**：数据源 — scroll 向量和标签，不做检索计算
- **PyTorch GPU**：计算层 — 批量矩阵乘法做精确 Cosine 距离计算
- **searcher.py**：不改动，保留为在线检索使用

## 2. GPU KNN 算法

### 2.1 核心流程

1. 从 Qdrant scroll 全量向量 + 标签到客户端
2. 转换为 torch.Tensor (N, 64), 上传 GPU
3. 查询向量 queries → (Q, 64) GPU
4. 分块计算 cosine similarity:  queries @ chunk.T → (Q, C)
5. 跨块保留 Top-K scores + labels + ids
6. GPU 上完成 LOO 过滤、多数投票、Purity/Recall 累加
7. 返回 CPU dict

### 2.2 分块策略

| GPU 显存预算 (GB) | 单块向量数 (float32) | 适用全量规模 |
|-------------------|--------------------|------------|
| 4 | ~16M | 小数据集 |
| 8 | ~32M | 千万级 |
| **16 (默认)** | **~64M** | **千万~亿级** |
| 24 | ~96M | 上亿级 |

\  # 预留一半给 queries

### 2.3 回退策略

- torch.cuda.is_available() == True  → GPU 路径 (默认)
- torch.cuda.is_available() == False → 警告 + 自动回退 numpy CPU 路径
- --no-gpu → 强制走 Qdrant exact 逐条检索 (用于校验)

### 2.4 精度分析

GPU 使用 float32 (Qdrant 存储 float64):
- float32 精度约 7 位有效数字
- Cosine 距离计算差异 < 1e-6
- Top-K 排序中极少出现跨精度 reorder (概率 < 0.01%)
- --no-gpu 保留 float64 Qdrant exact 路径用于精确校验

## 3. API 变更

### 3.1 metrics.py

新增参数 use_gpu=True, gpu_memory_gb=16 到 compute_knn_accuracy 和 compute_purity_recall_curve。
新增内部函数 _scroll_full_vectors() 和 _gpu_exact_knn()。

### 3.2 CLI

新增参数: --gpu-memory-gb (float, default=16), --no-gpu (flag)

### 3.3 WebUI

新增控件: GPU 显存预算 (number input, default=16, min=1, max=48)

## 4. 测试策略

- GPU 路径: mock scroll 数据, 用 PyTorch CPU tensor 模拟, 验证距离矩阵与 numpy 一致
- CPU fallback: use_gpu=False → 验证 numpy CPU 路径
- 分块正确性: 小 budget 分多块 → 验证结果与全量单块一致
- 回归: 现有单元测试继续通过 (mock PixelSearcher)
