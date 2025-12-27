# DIN 序列推荐研究项目

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-11.8+-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

一个**完全独立实现**的序列推荐模型研究项目，聚焦于 **Deep Interest Network (DIN)** 及其改进方案。

## 🎯 项目亮点

| 特点 | 说明 |
|------|------|
| 🔬 **系统性实验** | 4组实验，覆盖模型对比、消融分析、高级改进 |
| 📊 **真实数据验证** | MovieLens 100K/1M 数据集，AUC 最高达 **0.966** |
| 💡 **创新探索** | 自适应时间衰减、对比学习预训练、混合精排 |
| 🛠️ **工程完备** | TensorBoard 可视化、GPU 加速、模块化设计 |
| 📝 **面试友好** | 代码透明可解释，适合深入讲解 |

## 📈 核心实验结果

### 实验一：模型对比 (ml-1m, seq_len=20)

| 模型 | Test AUC | HR@10 | NDCG@10 | 参数量 |
|------|----------|-------|---------|--------|
| **SASRec** | **0.9663** | **0.780** | **0.534** | 543K |
| GRU4Rec | 0.9608 | 0.780 | 0.534 | 467K |
| NARM | 0.9599 | 0.762 | 0.513 | 536K |
| **DIN** | 0.9584 | 0.758 | 0.509 | 460K |
| AvgPool | 0.9432 | 0.719 | 0.475 | 390K |

### 实验三：消融实验 (ml-100k)

| 变体 | Test AUC | vs Base |
|------|----------|---------|
| DIN-Base | 0.8976 | baseline |
| **DIN-TimeDec** | **0.9120** | **+1.44%** |
| DIN-MultiHead | 0.8872 | -1.04% |
| DIN-Full | 0.8983 | +0.07% |

**关键发现**：时间衰减注意力带来显著提升，而多头注意力在小数据集上过拟合。

## 🏗️ 项目结构

```
DIN/
├── 📊 实验脚本
│   ├── experiment1.py          # 序列长度 + 模型对比
│   ├── experiment2.py          # 方法对比 + 混合精排
│   ├── experiment3.py          # 消融实验
│   └── experiment4.py          # 高级改进（自适应衰减+对比学习）
│
├── 🧠 模型定义
│   ├── models.py               # DIN, GRU4Rec, SASRec, NARM, AvgPool
│   ├── hybrid_ranker.py        # 混合精排模块
│   └── trainer.py              # 训练器 + TensorBoard
│
├── 📦 数据处理
│   ├── data_loader.py          # 数据加载 + 序列构建
│   └── feature_engineering.py  # 特征工程
│
├── 📄 文档
│   ├── README.md               # 项目总览（本文件）
│   ├── FEATURES.md             # 特征工程详解
│   └── EXPERIMENTS.md          # 实验设计详解
│
├── 📁 输出目录
│   ├── results/                # CPU 实验结果
│   └── results_gpu/            # GPU 实验结果
│
└── 🔧 工具脚本
    ├── run_experiments.py      # 主入口
    ├── run_all_gpu.py          # GPU 批量运行
    └── requirements.txt        # 依赖列表
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 进入项目目录
cd DIN

# 安装依赖
pip install -r requirements.txt

# 或手动安装
pip install torch numpy pandas matplotlib scikit-learn lightgbm tqdm tensorboard
```

### 2. 运行实验

```bash
# 运行全部实验（推荐）
python run_all_gpu.py --dataset ml-100k --experiments 1 2 3

# 单独运行
python experiment1.py   # 模型对比实验
python experiment3.py   # 消融实验
python experiment4.py   # 高级改进实验

# 指定数据集
python experiment4.py --dataset ml-1m --part adaptive
```

### 3. 查看结果

```bash
# TensorBoard 可视化
tensorboard --logdir runs/

# 结果文件位于 results_gpu/ 目录
```

## 📚 实验概览

| 实验 | 研究问题 | 关键发现 |
|------|----------|----------|
| **实验一** | 序列长度如何影响模型？ | 50-100 为最优区间，过长反而下降 |
| **实验二** | 深度模型 vs 树模型？ | DIN 优于 LightGBM 2-3%，混合精排有限提升 |
| **实验三** | DIN 各组件贡献？ | 时间衰减 +1.44%，多头注意力需谨慎 |
| **实验四** | 高级改进方向？ | 自适应衰减、对比学习有潜力 |

详见 [EXPERIMENTS.md](EXPERIMENTS.md)

## 🔬 技术亮点

### 1. DIN 核心注意力机制

```python
# 注意力公式: a(k, q) = softmax(MLP([k, q, k*q, k-q]))
attention_input = torch.cat([keys, query, keys * query, keys - query], dim=-1)
attention_scores = self.attention_mlp(attention_input)
```

### 2. 时间衰减注意力（改进）

```python
# 近期行为权重更高
positions = torch.arange(seq_len)
time_weights = torch.exp(decay_rate * (positions - seq_len + 1))
attention_scores = base_scores * time_weights
```

### 3. 自适应衰减（创新，实验四）

```python
# 衰减率作为可学习参数
self.decay_rate = nn.Parameter(torch.tensor(0.1))
```

### 4. 对比学习预训练（创新，实验四）

```python
# InfoNCE 损失 + 序列增强
z1 = encoder(augment(seq, 'crop'))
z2 = encoder(augment(seq, 'mask'))
loss = InfoNCE(z1, z2, temperature=0.1)
```

## 📊 特征工程

详见 [FEATURES.md](FEATURES.md)

| 特征类型 | 示例 | 维度 |
|----------|------|------|
| 用户画像 | age_bucket, gender, occupation | 3 |
| 物品属性 | genre, year_bucket, popularity | 3 |
| 序列特征 | history_genres, history_years | 2×L |
| 时间上下文 | hour_bucket, day_of_week, is_weekend | 3 |
| 统计特征 | user_activity, item_popularity | 2 |

## 🖥️ 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | GTX 1060 6GB | RTX 3080 Ti |
| RAM | 8GB | 16GB+ |
| 存储 | 1GB | 5GB |

**实测运行时间**（RTX 3080 Ti）：

| 实验 | ml-100k | ml-1m |
|------|---------|-------|
| 实验一（20组） | 2.5 小时 | 9.4 小时 |
| 实验三（5组） | 27 分钟 | 2 小时 |

## 📖 参考论文

1. **[DIN]** Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" (KDD 2018)
2. **[GRU4Rec]** Hidasi et al. "Session-based Recommendations with RNNs" (ICLR 2016)
3. **[SASRec]** Kang & McAuley. "Self-Attentive Sequential Recommendation" (ICDM 2018)
4. **[NARM]** Li et al. "Neural Attentive Session-based Recommendation" (CIKM 2017)
5. **[CL4SRec]** Xie et al. "Contrastive Learning for Sequential Recommendation" (ICDE 2022)

## 🤝 致谢

- MovieLens 数据集由 GroupLens Research 提供
- 项目灵感来源于阿里巴巴 DIN 论文及工业实践

## 📄 License

MIT License - 可自由使用于学习和研究目的

---

<p align="center">
  <b>如果这个项目对你有帮助，欢迎 ⭐ Star</b>
</p>
