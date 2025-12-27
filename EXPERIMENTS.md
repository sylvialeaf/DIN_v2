# 实验设计详解 (EXPERIMENTS.md)

本文档详细介绍项目的实验设计思路、研究问题、实验配置和结果分析。

## 📋 目录

1. [实验设计总览](#实验设计总览)
2. [实验一：序列长度与模型对比](#实验一序列长度与模型对比)
3. [实验二：方法对比与混合精排](#实验二方法对比与混合精排)
4. [实验三：DIN改进消融实验](#实验三din改进消融实验)
5. [实验四：高级改进探索](#实验四高级改进探索)
6. [实验结论与启示](#实验结论与启示)

---

## 实验设计总览

### 研究问题矩阵

| 实验 | 核心问题 | 独立变量 | 因变量 | 控制变量 |
|------|----------|----------|--------|----------|
| 实验一 | 序列长度影响 | seq_len, 模型类型 | AUC, HR@K, NDCG@K | 数据集, epochs |
| 实验二 | 深度 vs 树模型 | 模型类型 | AUC, LogLoss | 特征集, 数据集 |
| 实验三 | 组件贡献 | 注意力类型 | AUC, 训练时间 | 模型架构 |
| 实验四 | 高级改进 | decay策略, 预训练 | AUC | 基础架构 |

### 实验逻辑链

```
实验一: 基础验证
├── 问题: DIN 在序列推荐中效果如何？
├── 发现: DIN 有效，但不是最优
└── 引出: 为什么？如何改进？

实验二: 方法对比
├── 问题: 深度模型 vs 传统方法？
├── 发现: 深度模型优于 LightGBM
└── 引出: 能否结合两者优势？

实验三: 组件分析
├── 问题: DIN 各组件贡献如何？
├── 发现: 时间衰减有效，多头需谨慎
└── 引出: 是否有更好的改进方向？

实验四: 创新探索
├── 问题: 自适应衰减？对比学习？
├── 发现: 均有潜力
└── 结论: 为后续研究指明方向
```

---

## 实验一：序列长度与模型对比

### 研究动机

在序列推荐中，**历史行为长度**是关键超参数：
- 太短：信息不足，无法捕捉用户偏好
- 太长：引入噪声，计算开销大

同时，不同模型对序列长度的敏感度不同，需要系统对比。

### 实验配置

```python
# experiment1.py 配置
SEQ_LENGTHS = [20, 50, 100, 150]
MODELS = ['DIN', 'GRU4Rec', 'SASRec', 'NARM', 'AvgPool']
DATASETS = ['ml-100k', 'ml-1m']
EPOCHS = 50
BATCH_SIZE = 256
```

### 对比模型简介

| 模型 | 核心机制 | 复杂度 | 特点 |
|------|----------|--------|------|
| **DIN** | 目标注意力 | O(L×D²) | 针对候选物品计算注意力 |
| **GRU4Rec** | GRU 循环 | O(L×D²) | 捕捉序列依赖 |
| **SASRec** | 自注意力 | O(L²×D) | Transformer架构 |
| **NARM** | GRU+注意力 | O(L×D²) | 结合两种机制 |
| **AvgPool** | 平均池化 | O(L×D) | 简单基线 |

### 实验结果 (ml-1m, seq_len=20)

| 模型 | Test AUC | HR@5 | HR@10 | NDCG@10 | QPS |
|------|----------|------|-------|---------|-----|
| SASRec | **0.9663** | 0.679 | **0.780** | **0.534** | 19.6K |
| GRU4Rec | 0.9608 | **0.679** | 0.780 | 0.534 | 8.2K |
| NARM | 0.9599 | 0.663 | 0.762 | 0.513 | 7.1K |
| DIN | 0.9584 | 0.629 | 0.758 | 0.509 | 6.8K |
| AvgPool | 0.9432 | 0.578 | 0.719 | 0.475 | 5.3K |

### 序列长度影响 (ml-100k, DIN)

| seq_len | Test AUC | HR@10 | 训练时间 |
|---------|----------|-------|----------|
| 20 | 0.9283 | 0.674 | 150s |
| 50 | **0.9359** | 0.663 | 245s |
| 100 | 0.9311 | **0.684** | 375s |
| 150 | 0.9199 | 0.642 | 498s |

### 关键发现

1. **最优序列长度**: 50-100，过长反而性能下降
2. **模型排名**: SASRec > GRU4Rec ≈ NARM > DIN > AvgPool
3. **DIN 优势**: 参数少（460K vs SASRec 543K），推理更快
4. **DIN 劣势**: 在长序列上不如 Transformer 架构

### 分析与反思

**为什么 DIN 不是最优？**

DIN 原论文场景是**广告点击预测**，特点：
- 候选集确定（给定广告）
- 短期兴趣为主
- 实时性要求高

而本实验是**序列推荐**场景：
- 需要建模长期偏好
- 序列内部依赖重要
- SASRec/GRU4Rec 更适合

**启示**: 模型选择需匹配场景，DIN 在 CTR 场景仍有优势。

---

## 实验二：方法对比与混合精排

### 研究动机

工业界常用**两阶段排序**：
1. 召回/粗排：深度模型
2. 精排：树模型（LightGBM）

问题：能否结合 DIN 和 LightGBM 的优势？

### 实验配置

```python
# experiment2.py 配置
METHODS = ['DIN', 'GRU4Rec', 'AvgPool', 'LightGBM', 'Hybrid']

# 混合精排架构
DIN embedding → ┐
手工特征     → ├→ LightGBM → 最终预测
交叉特征     → ┘
```

### 混合精排设计 (v2.0)

**关键改进**（修复了原设计缺陷）：

```python
# ❌ 原设计（有信息泄露）
features = [din_score, din_embedding, hand_features]  # din_score 是预测结果！

# ✅ 改进设计
features = [
    pca_embedding,      # DIN embedding 降维 (64→16)
    attention_stats,    # 注意力统计（熵、top1、方差）
    hand_features,      # 用户/物品/时间特征
    cross_features      # 交叉特征
]
```

### 实验结果

| 方法 | Test AUC | Test LogLoss | 说明 |
|------|----------|--------------|------|
| DIN | 0.9180 | 0.4387 | 深度模型 |
| GRU4Rec | 0.9135 | 0.4521 | 序列模型 |
| LightGBM | 0.8950 | 0.4823 | 纯特征工程 |
| AvgPool | 0.8876 | 0.5012 | 基线 |
| Hybrid | 0.9195 | 0.4352 | DIN+LightGBM |

### 关键发现

1. **深度模型优势明显**: DIN 比 LightGBM 高 2.3%
2. **混合精排有限提升**: 仅 +0.15%，低于预期
3. **原因分析**: 
   - DIN 已学到足够的特征表示
   - LightGBM 难以从 embedding 中提取额外信息

### 混合精排何时有效？

| 场景 | 混合精排效果 | 原因 |
|------|--------------|------|
| 特征丰富 | ✅ 有效 | 树模型擅长处理显式特征 |
| 冷启动 | ✅ 有效 | 深度模型对新用户/物品不友好 |
| 本实验 | ⚠️ 有限 | MovieLens 特征较简单 |

---

## 实验三：DIN改进消融实验

### 研究动机

DIN 原论文发表于 2018 年，后续有多种改进方向：
- 时间衰减：近期行为更重要
- 多头注意力：捕捉多维兴趣

问题：这些改进在实际中效果如何？

### 实验配置

```python
# experiment3.py 配置
ABLATION_CONFIGS = [
    {'name': 'DIN-Base', 'attention_type': 'base'},
    {'name': 'DIN-TimeDec', 'attention_type': 'time_decay'},
    {'name': 'DIN-MultiHead', 'attention_type': 'multi_head'},
    {'name': 'DIN-Full', 'attention_type': 'full'},  # TimeDec + MultiHead
]
```

### 改进机制详解

#### 1. 时间衰减注意力 (Time Decay)

```python
# 核心思想：近期行为权重更高
positions = torch.arange(seq_len)  # [0, 1, ..., L-1]
time_weights = torch.exp(decay_rate * (positions - seq_len + 1))
# positions - seq_len + 1 = [-L+1, -L+2, ..., 0]
# 最近的行为（位置0）权重为 exp(0)=1
# 最远的行为权重为 exp(-decay_rate*(L-1))

attention_scores = base_scores * time_weights
```

#### 2. 多头注意力 (Multi-Head)

```python
# 核心思想：捕捉多维兴趣
class MultiHeadRichAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        self.attention_heads = nn.ModuleList([
            build_attention_mlp() for _ in range(num_heads)
        ])
    
    def forward(self, query, keys):
        outputs = [head(query, keys) for head in self.attention_heads]
        return mean(outputs)  # 多头平均
```

### 实验结果 (ml-100k)

| 变体 | Test AUC | Test LogLoss | vs Base | QPS |
|------|----------|--------------|---------|-----|
| DIN-Base | 0.8976 | 0.5045 | — | 21.1K |
| **DIN-TimeDec** | **0.9120** | **0.4603** | **+1.44%** | 27.6K |
| DIN-MultiHead | 0.8872 | 0.6064 | -1.04% | 24.9K |
| DIN-Full | 0.8983 | 0.5321 | +0.07% | 23.4K |

### 关键发现

1. **时间衰减有效**: +1.44% AUC，符合直觉
2. **多头注意力反而下降**: -1.04%，可能过拟合
3. **两者结合效果不佳**: 相互抵消

### 为什么多头注意力效果差？

| 原因 | 分析 |
|------|------|
| 数据量小 | ml-100k 仅 10 万条，多头增加参数量 |
| 兴趣维度有限 | MovieLens 用户兴趣相对单一 |
| 原始 DIN 已足够 | 单头注意力已捕捉主要模式 |

**对比**: 在大规模数据集（如淘宝）上，多头注意力通常有效。

---

## 实验四：高级改进探索

### 研究动机

基于实验三的发现，探索更高级的改进方向：
1. **自适应衰减**: decay_rate 不应固定
2. **对比学习**: 改善序列表示

### Part 1: 自适应时间衰减

#### 设计思路

```python
# 固定衰减（原方案）
decay_rate = 0.1  # 超参数

# 可学习衰减（改进）
self.decay_rate = nn.Parameter(torch.tensor(0.1))

# 用户级个性化衰减（高级）
self.decay_rate = nn.Embedding(num_users, 1)
```

#### 实验配置

| 配置 | 说明 |
|------|------|
| Fixed-0.05 | 固定衰减 0.05 |
| Fixed-0.1 | 固定衰减 0.1 |
| Fixed-0.2 | 固定衰减 0.2 |
| Learnable | 全局可学习 |
| Per-User | 用户级个性化 |

#### 预期效果

- 可学习衰减能自动找到最优值
- 用户级衰减捕捉个体差异（活跃用户衰减更快）

### Part 2: 对比学习预训练

#### 设计思路

```
原始序列: [A, B, C, D, E]
           ↓ 数据增强
增强视图1: [A, B, C] (crop)
增强视图2: [A, _, C, D, E] (mask)
           ↓ 编码器
    z1, z2
           ↓ InfoNCE 损失
    最大化 sim(z1, z2)
    最小化 sim(z1, z_other)
```

#### 数据增强策略

| 策略 | 操作 | 参数 |
|------|------|------|
| Crop | 随机裁剪 | 保留 60% |
| Mask | 随机掩码 | 掩码 20% |
| Reorder | 随机重排 | 重排 20% |
| Substitute | 随机替换 | 替换 10% |

#### 训练流程

```python
# 两阶段训练
# Stage 1: 对比预训练
for epoch in range(pretrain_epochs):
    z1, z2 = encoder(augment(seq))
    loss = InfoNCE(z1, z2)
    
# Stage 2: CTR 微调
for epoch in range(finetune_epochs):
    logits = model(batch)
    loss = BCE(logits, labels)
```

#### 实验配置

| 配置 | 预训练 | 联合训练 |
|------|--------|----------|
| No-Pretrain | 0 epochs | ✗ |
| Pretrain-5ep | 5 epochs | ✗ |
| Pretrain-10ep | 10 epochs | ✗ |
| Joint-0.05 | ✗ | λ=0.05 |
| Joint-0.1 | ✗ | λ=0.1 |
| Pretrain+Joint | 5 epochs | λ=0.05 |

---

## 实验结论与启示

### 核心结论

| 结论 | 支撑实验 | 启示 |
|------|----------|------|
| DIN 在序列推荐中有效但非最优 | 实验一 | 模型选择需匹配场景 |
| 深度模型优于传统特征工程 | 实验二 | 深度学习的特征抽取能力 |
| 时间衰减是有效的改进方向 | 实验三 | 符合用户兴趣漂移规律 |
| 多头注意力需谨慎使用 | 实验三 | 数据量与模型复杂度匹配 |
| 混合精排收益有限 | 实验二 | DIN 已学到足够信息 |

### 对求职者的启示

#### 1. 论文复现要点

- **理解场景差异**: DIN 论文是广告 CTR，本项目是序列推荐
- **不要盲目套用**: 需要根据实际场景调整
- **关注细节**: padding 方向、mask 处理、损失函数

#### 2. 实验设计要点

- **对照组**: 每个实验都有 baseline
- **消融实验**: 验证各组件贡献
- **多数据集**: ml-100k/ml-1m 交叉验证
- **多指标**: AUC + TopK + 效率

#### 3. 工程实践要点

- **代码模块化**: 模型、数据、训练分离
- **可复现**: 固定随机种子
- **可视化**: TensorBoard 监控
- **文档完善**: README + 注释

### 未来改进方向

| 方向 | 难度 | 预期收益 | 说明 |
|------|------|----------|------|
| 自适应衰减 | 低 | +1-2% | 简单有效 |
| 对比学习 | 中 | +2-3% | 改善冷启动 |
| 跨字段注意力 | 中 | +1-2% | 需要丰富特征 |
| 知识蒸馏 | 高 | 部署加速 | 模型压缩 |

---

## 附录：运行命令参考

```bash
# 实验一：模型对比
python experiment1.py

# 实验三：消融实验
python experiment3.py

# 实验四：高级改进
python experiment4.py --dataset ml-100k --part all
python experiment4.py --dataset ml-1m --part adaptive
python experiment4.py --dataset ml-1m --part contrastive

# GPU 批量运行
python run_all_gpu.py --dataset ml-100k --experiments 1 3
python run_all_gpu.py --dataset ml-1m --experiments 1 --epochs 30
```
