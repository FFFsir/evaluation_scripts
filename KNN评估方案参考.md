# KNN Embedding 评估方案参考

## 项目背景

基于 Qdrant 的 KNN 像素评估系统已完成。系统将卫星影像（128×128 像素）的 64 维 embedding 向量存入 Qdrant（余弦距离），每个像素附有 9 类土地覆盖标签：

| ID | Label |
|----|-------|
| 0 | water |
| 1 | trees |
| 2 | grass |
| 3 | flooded_vegetation |
| 4 | crops |
| 5 | shrub_and_scrub |
| 6 | built |
| 7 | bare |
| 8 | snow_and_ice |

目标是设计一套统计指标，评估 **embedding 空间中 KNN 相似度与标签一致性的关系**。

---

## 一、两种原始方案评估

### 方案一：KNN Label Consistency（K近邻标签一致性）

> 对 label=built 的像素，查 Top-K 最近邻，统计其中 built 占比、非 built 占比及分布

**评估结论：非常有意义，是最标准的 embedding 质量评估方法之一。**

本质上是衡量 **embedding 空间的局部标签纯度（Local Label Purity）**，直接回答核心问题：*"embedding 相似的像素，是否真的有相似的地表类型？"*

**示例解读：**

- 若 K=10 时 built 像素的邻居中 90% 也是 built → embedding 对 built 类表达良好
- 若只有 30% → built 类 embedding 过于分散，与其他类别混合在一起

**细化建议：**

1. **不要只看一个 K 值** — 绘制 K = [1, 5, 10, 20, 50, 100] 的纯度曲线，观察衰减趋势
2. **按类别分别统计** — 不同类别的 embedding 质量差异可能很大（built 可能好但 grass/shrub 可能极度混淆）
3. **扩展为 KNN 分类准确率** — 用 K 近邻的多数票作为预测标签，与真实标签比较，直接得出分类指标

---

### 方案二：Similarity Threshold Analysis（相似度阈值分析）

> 对 label=built 的像素，找相似度 > 0.8 的所有邻居，统计标签一致性

**评估结论：同样很有意义，但需要注意几个技术细节。**

这个方案衡量的是 **"在 embedding 空间的某个半径球内，标签是否一致"**。与方案一不同，邻居数量不固定（由距离阈值决定），不同查询像素可能返回不同数量的邻居。

**需要注意的问题：**

1. **阈值选取是关键** — 0.8 在高维余弦空间中可能极其严格（高维向量往往集中在某个超球面附近）。建议先画**全局相似度分布直方图**，了解分数的实际分布后再定阈值
2. **稀疏性问题** — 很多像素可能相似度 >0.8 的邻居数为 0，需要同时统计 **"有多少查询像素在阈值内没有邻居"**
3. **阈值扫描比固定阈值更有洞察力** — 建议画 τ = [0.5, 0.6, 0.7, 0.8, 0.9] 的纯度随阈值变化曲线

---

## 二、业界常用 KNN Embedding 评估方案

### 1. KNN Classification Accuracy（最核心 ⭐）

对每个像素点，用 K 近邻的多数票做分类预测，与真实标签比较：

- **Overall Accuracy** — 整体正确率（KNN 投票结果与 ground truth 的一致性）
- **Per-class Precision / Recall / F1** — 揭示哪些类别容易混淆（如 grass vs crops、bare vs built）
- **Confusion Matrix of KNN Predictions** — 比原标签的混淆矩阵更有信息量，反映 embedding 空间中的混淆模式

```
对每个查询像素 q：
    预测标签 = K个邻居中数量最多的标签（多数投票）
    真实标签 = q 的真实标签
    统计 predicted vs actual 的各种指标
```

**优点：** 最标准、最易理解，直接回答"embedding 好用吗"
**缺点：** 需要采样（全量计算 O(N²)）

---

### 2. Neighborhood Purity Curve（邻居纯度曲线）

方案一的系统化版本。定义：

```
Purity(K) = Σᵢ (Kᵢ_same / K) / N

其中 Kᵢ_same = 查询像素 i 的 Top-K 邻居中与其同标签的数量
N = 查询像素总数
```

绘制 "K vs 平均邻居纯度" 曲线：

- 曲线下降越快 → embedding 的类间分离越差
- 不同类别画在同一张图上可直观对比各类质量差异
- 理想情况：K 较小时纯度接近 1.0，大 K 时仍保持较高水平

**优点：** 直观、实现简单
**缺点：** 未考虑不同类别样本量不均的问题

---

### 3. Recall@K（K 近邻召回率）

方案一的补充，消除样本不均衡影响：

```
Recall@K(i) = Kᵢ_same / min(K, N_same)

其中 N_same = 数据集中与查询像素 i 同标签的像素总数
```

- 对于小样本类别（如 snow_and_ice）：即使 K=10 邻居全为该类，Recall@K 也可能很低，说明该类在全局中占比极少
- 对于大样本类别（如 water）：Recall@K 接近 1 才说明 embedding 对同类像素聚集得好
- 同样可以绘制 Recall@K vs K 的曲线

**优点：** 考虑了类别不均衡
**缺点：** 需要全量标签统计

---

### 4. Intra-class vs Inter-class Distance Distribution

非常直观的诊断工具：

**定义：**
- **Intra-class distance** — 同一标签内部两两像素的余弦距离分布
- **Inter-class distance** — 不同标签间两两像素的余弦距离分布

**理想情况：** 两个分布明显分离（同类像素距离远小于异类像素距离）

**量化指标：**
- 两类分布之间的 **Wasserstein distance** 或 **KL divergence**
- 可分性比率：`mean(inter-class) / mean(intra-class)`，越大越好

**可视化：** 直方图叠加展示（inter-class 和 intra-class 在同一张图上不同颜色）

**优点：** 直观展示类间分离度，能诊断具体哪几类在混淆
**缺点：** 全量计算 O(N²)，建议采样

---

### 5. Silhouette Score（轮廓系数）

经典聚类质量指标，天然适用于评估 embedding 空间的类别分离度：

```
Silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))

其中 a(i) = 像素 i 到同标签所有其他像素的平均距离（紧凑度）
     b(i) = 像素 i 到最近的其他标签所有像素的平均距离（分离度）
```

**取值范围：** [-1, 1]，越接近 1 说明分离越好

- **全局 Silhouette Score** — 整体 embedding 质量的一个数字
- **Per-class 平均 Silhouette** — 看各类表现差异
- **Per-pixel Silhouette 分布直方图** — 识别异常像素

> ⚠️ 计算量 O(N²)，建议对每类随机采样 1000-5000 个点。Qdrant 中的 HNSW 索引可用于加速最近邻查找。

**优点：** 综合性指标，学术界广泛使用
**缺点：** 计算成本高

---

### 6. Mean Average Precision (mAP)

借用信息检索的指标。对每个查询像素，将其同标签像素视为"相关文档"：

```
AP(i) = Σₖ P(k) × rel(k) / |Relevant|

其中 P(k) = 前 k 个结果中同标签像素的占比（precision@k）
     rel(k) = 1 if 第k个邻居与查询像素同标签 else 0
     |Relevant| = 数据集中与查询像素同标签的像素总数

mAP = Σᵢ AP(i) / N
```

**优点：** 同时考量排序质量和召回完整性
**缺点：** 计算相对复杂

---

### 7. Trustworthiness & Continuity

来自流形学习/降维领域的经典指标：

- **Trustworthiness** — 在 embedding 空间中离我近的点，在标签空间中也离我近吗？
  - 惩罚 embedding 空间中的"假邻居"（离得近但标签不同）
- **Continuity** — 在标签空间中离我近的点，在 embedding 空间中也离我近吗？
  - 惩罚 embedding 空间中的"遗漏邻居"（标签相同但离得远）

这两个指标能揭示 embedding 是否存在扭曲或折叠。

**优点：** 从两个对称角度评估，互补性强
**缺点：** 理解和实现门槛稍高

---

### 8. Distance to Nearest Neighbor of Different Class（最近异类距离）

对每个像素，计算：

- `d_same` = 到最近同标签邻居的余弦距离
- `d_diff` = 到最近不同标签邻居的余弦距离

**衍生指标：**
- `margin = d_diff - d_same` — 正值越大说明边界越清晰，负值说明该像素的最近邻居是异类（可能是噪声或标注错误）
- `ratio = d_diff / d_same` — 比值越大越好
- 绘制 `margin` 分布直方图 per-class

**优点：** 实现简单，能发现标注错误或边界像素
**缺点：** 只考虑最近邻，信息量有限

---

## 三、实施优先级建议

| 优先级 | 指标 | 原因 |
|--------|------|------|
| 🥇 必做 | **KNN Accuracy + Per-class F1** | 最标准、最易理解，直接回答"embedding 好用吗" |
| 🥇 必做 | **Neighborhood Purity Curve** | 方案一的系统化版本，实现成本很低 |
| 🥈 推荐 | **Intra/Inter-class Distance Distribution** | 直观展示类间分离度，诊断具体哪几类在混淆 |
| 🥈 推荐 | **Recall@K 曲线** | 补充方案一，消除样本不均衡的影响 |
| 🥈 推荐 | **Margin Distribution (d_diff - d_same)** | 实现简单，能发现边界像素和潜在标注噪声 |
| 🥉 可选 | **Similarity Threshold Scan**（方案二的改进版） | 有价值但需先了解全局相似度分布再定阈值 |
| 🥉 可选 | **Silhouette Score (per-class)** | 综合性指标，但计算成本高，采样后可做 |
| 🥉 可选 | **KNN Confusion Matrix** | 比原始标签混淆矩阵更有诊断价值 |
| 🥉 可选 | **Trustworthiness & Continuity** | 流形质量指标，进阶分析时使用 |

---

## 四、实施建议

### 4.1 建议的模块结构

新增 `KNN_evaluation/metrics.py`，实现以下核心函数：

```python
# 核心指标
compute_knn_accuracy(searcher, queries, k) -> dict         # KNN 分类准确率
compute_purity_curve(searcher, queries, k_values) -> dict   # 邻居纯度曲线
compute_recall_at_k(searcher, queries, k_values) -> dict    # Recall@K 曲线

# 距离分析
compute_intra_inter_distance(searcher, samples) -> dict     # Intra/Inter class 距离分布
compute_margin_distribution(searcher, samples) -> dict      # d_diff - d_same 分布

# 阈值分析
compute_threshold_scan(searcher, queries, thresholds) -> dict  # 相似度阈值扫描

# 聚类质量
compute_silhouette(searcher, samples) -> dict               # 轮廓系数
```

### 4.2 关键设计原则

1. **所有指标按类别分别报告** — 不同地表类型的 embedding 质量大概率不同，合并报告会掩盖问题
2. **采样策略** — 全量计算 O(N²) 不现实，建议：
   - 每类随机采样 500-2000 个查询像素
   - 对于 Intra/Inter distance，每类采样 1000 对像素
3. **可视化优于纯数字** — Purity Curve、Distance Distribution Histogram、Confusion Matrix 都比表格直观
4. **与现有 searcher 模块集成** — 复用 `PixelSearcher.search()`，统一 `HitRecord` 和 `SearchResult` 数据结构

### 4.3 典型使用流程

```
1. 先跑 Neighborhood Purity Curve（快速，O(N·K)）
   → 判断哪些类别的 embedding 质量好/差

2. 对质量差的类别跑 Intra/Inter Distance Distribution
   → 诊断具体是和哪些类别混淆

3. 跑 KNN Accuracy + Confusion Matrix
   → 量化混淆程度

4. 按需跑 Silhouette Score（采样）
   → 获得整体质量的单一数字指标
```

---

## 五、参考资料

- **Neighborhood Purity**: Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- **Silhouette Score**: Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis". *Journal of Computational and Applied Mathematics*, 20, 53–65.
- **Trustworthiness & Continuity**: Venna, J., & Kaski, S. (2001). "Neighborhood preservation in nonlinear projection methods: An experimental study". *ICANN 2001*.
- **mAP for embedding evaluation**: Musgrave, K., Belongie, S., & Lim, S. N. (2020). "A Metric Learning Reality Check". *ECCV 2020*.
- **Qdrant 文档**: https://qdrant.tech/documentation/
