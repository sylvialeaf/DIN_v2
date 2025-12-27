#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
云端 GPU 完整实验脚本

适合在 AutoDL / Colab / 阿里云等 GPU 环境运行。
支持 ml-100k 和 ml-1m 双数据集。
包含全部四个实验。

使用方法:
    python run_all_gpu.py                    # 运行所有实验（两个数据集）
    python run_all_gpu.py --dataset ml-100k  # 只运行 ml-100k
    python run_all_gpu.py --dataset ml-1m    # 只运行 ml-1m
    python run_all_gpu.py --quick            # 快速测试模式
    
    # 单独运行某个实验
    python run_all_gpu.py --exp 1            # 只运行实验1（模型对比）
    python run_all_gpu.py --exp 2            # 只运行实验2（方法对比）
    python run_all_gpu.py --exp 3            # 只运行实验3（消融实验）
    python run_all_gpu.py --exp 4            # 只运行实验4（高级改进）
    python run_all_gpu.py --exp 1,2,3        # 运行实验1-3
    python run_all_gpu.py --exp 1,3,4        # 运行实验1、3、4

预估时间 (单 GPU, 两个数据集):
    实验1（序列长度+模型对比）: 约 40-60 分钟
    实验2（方法对比+混合精排）: 约 30-40 分钟
    实验3（消融实验）:          约 20-30 分钟
    实验4（高级改进）:          约 60-90 分钟
    总计:                       约 2.5-4 小时
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
from tqdm import tqdm

from data_loader import get_rich_dataloaders, get_topk_eval_data, build_topk_batch_multi
from models import DINRichLite, SimpleAveragePoolingRich, GRU4Rec, SASRec, NARM, AttentionLayer
from trainer import RichTrainer, measure_inference_speed_rich
from feature_engineering import FeatureProcessor, InteractionFeatureExtractor, prepare_lightgbm_features

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM 未安装，混合精排将跳过")

# ========================================
# 配置
# ========================================

parser = argparse.ArgumentParser(description='云端 GPU 完整实验')
parser.add_argument('--dataset', type=str, default='both', 
                    choices=['ml-100k', 'ml-1m', 'both'],
                    help='数据集选择')
parser.add_argument('--quick', action='store_true', 
                    help='快速测试模式（减少 epochs 和序列长度）')
parser.add_argument('--epochs', type=int, default=50,
                    help='训练轮数（默认 50）')
parser.add_argument('--exp', type=str, default='all',
                    help='运行哪些实验: 1, 2, 3, 4, 1,2,3, all')
parser.add_argument('--no-topk', action='store_true',
                    help='禁用 Top-K 评估（加速训练）')
parser.add_argument('--topk-sample', type=str, default='auto',
                    help='Top-K 评估采样用户数（默认 auto：ml-100k全量，ml-1m采样2000；可指定数字或None）')
parser.add_argument('--exp4-part', type=str, default='all',
                    choices=['all', 'adaptive', 'contrastive'],
                    help='实验4子任务: all, adaptive, contrastive')
args = parser.parse_args()

# 解析要运行的实验
if args.exp == 'all':
    EXPERIMENTS_TO_RUN = [1, 2, 3, 4]
else:
    EXPERIMENTS_TO_RUN = [int(x.strip()) for x in args.exp.split(',')]

# Top-K 评估开关
ENABLE_TOPK = not args.no_topk

# Top-K 采样策略
# - 'auto': ml-100k全量，ml-1m采样2000（默认，推荐）
# - 数字: 指定采样数
# - None: 全量评估（慢）
TOPK_SAMPLE_CONFIG = args.topk_sample

def get_topk_sample_users(dataset_name, config):
    """
    智能决定 Top-K 评估的采样用户数
    
    Args:
        dataset_name: 数据集名称
        config: 'auto' / 数字 / None
    
    Returns:
        int or None: 采样用户数，None 表示全量
    
    统计学依据：
    - 1000 样本: 95%置信度，误差±3.1%
    - 2000 样本: 95%置信度，误差±1.8%
    - ml-100k 仅943用户，全量评估
    """
    if config == 'auto':
        # 智能模式：小数据集全量，大数据集采样
        if dataset_name == 'ml-100k':
            return None  # 全量（943用户）
        elif dataset_name == 'ml-1m':
            return 2000  # 采样（误差±1.8%）
        else:
            return 2000  # 其他数据集默认采样
    elif config is None or config == 'None':
        return None  # 全量
    else:
        try:
            return int(config)  # 指定采样数
        except:
            return None

# 设备检测
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count() if DEVICE == 'cuda' else 0
USE_MULTI_GPU = NUM_GPUS > 1

# 实验参数
if args.quick:
    EPOCHS = 10
    SEQ_LENGTHS = [20, 50]
    BATCH_SIZE = 1024
else:
    EPOCHS = args.epochs
    SEQ_LENGTHS = [20, 50, 100, 150]
    # 3x RTX 4090 (24GB each) - 大幅增加batch充分利用显存
    # 模型较小，可以用更大的batch提高GPU利用率
    BASE_BATCH_SIZE = 4096 if DEVICE == 'cuda' else 256
    # 多GPU时充分利用所有显卡
    if USE_MULTI_GPU:
        # 3x4090: 每卡4096，总batch = 12288
        BATCH_SIZE = BASE_BATCH_SIZE * NUM_GPUS
    else:
        BATCH_SIZE = BASE_BATCH_SIZE

EMBEDDING_DIM = 64

# SASRec 长序列时需要减小 batch size（因为 O(L²) 内存复杂度）
# 计算公式：显存 ∝ batch × seq² × heads
def get_adaptive_batch_size(model_name, seq_length, base_batch_size):
    """
    根据模型和序列长度自适应调整 batch size
    SASRec 的注意力矩阵是 [B, H, L, L]，显存占用与 L² 成正比
    3x RTX 4090 (72GB总显存) 可以处理更大的batch
    """
    if model_name == 'SASRec' and seq_length > 100:
        # seq=150 时，注意力矩阵是 seq=100 的 2.25 倍
        # 3x4090有充足显存，只需轻微缩减
        scale = (100 / seq_length) ** 1.5  # 比之前更激进
        return max(1024, int(base_batch_size * scale))
    return base_batch_size

# Top-K 评估参数
TOPK_VALUES = [5, 10, 20]  # 评估的 K 值
NUM_NEG_SAMPLES = 99  # 负采样数量（加上正样本共 100 个候选）

# 根据 CPU 核数设置 num_workers
# 多 GPU 时可以增加 workers
import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()
# 48 vCPU 可以使用更多 workers 加速数据加载
# 每个 GPU 分配 4-6 个 workers
NUM_WORKERS = min(18, CPU_COUNT - 2) if DEVICE == 'cuda' else 0

# 预取因子：每个 worker 预加载的 batch 数
# 3x4090 高吞吐量，增加预取减少等待
PREFETCH_FACTOR = 6  # 默认是 2，增加可以减少 GPU 等待

MODELS_TO_TEST = ['DIN', 'GRU4Rec', 'SASRec', 'NARM', 'AvgPool']

# TensorBoard 配置
ENABLE_TENSORBOARD = True
# AutoDL 默认的 TensorBoard 日志目录是 /root/tf-logs
# 本地测试时使用 ./runs
import platform
if platform.system() == 'Linux' and os.path.exists('/root'):
    TENSORBOARD_LOG_DIR = '/root/tf-logs'  # AutoDL 默认目录
else:
    TENSORBOARD_LOG_DIR = './runs'  # 本地 Windows/Mac

# 数据集
if args.dataset == 'both':
    DATASETS = ['ml-100k', 'ml-1m']
else:
    DATASETS = [args.dataset]

# 结果目录
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results_gpu')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("🚀 云端 GPU 完整实验")
print("=" * 80)
print(f"设备: {DEVICE}")
if DEVICE == 'cuda':
    for i in range(NUM_GPUS):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    if USE_MULTI_GPU:
        print(f"🔥 多 GPU 模式: {NUM_GPUS} 张卡，DataParallel 加速")
print(f"数据集: {DATASETS}")
print(f"实验: {EXPERIMENTS_TO_RUN}")
print(f"Epochs: {EPOCHS}")
print(f"序列长度: {SEQ_LENGTHS}")
print(f"Batch Size: {BATCH_SIZE}" + (f" ({BATCH_SIZE // NUM_GPUS} × {NUM_GPUS} GPUs)" if USE_MULTI_GPU else ""))
print(f"Num Workers: {NUM_WORKERS}")
print(f"模型: {MODELS_TO_TEST}")
print(f"Top-K 评估: {'启用' if ENABLE_TOPK else '禁用'} (K={TOPK_VALUES})")
print(f"Top-K 采样策略: {TOPK_SAMPLE_CONFIG} (auto=ml-100k全量/ml-1m采样2000)")
print(f"快速模式: {args.quick}")
print(f"TensorBoard: {'启用' if ENABLE_TENSORBOARD else '禁用'} (日志目录: {TENSORBOARD_LOG_DIR})")
print("=" * 80)


# ========================================
# 消融实验的注意力变体
# ========================================

class TimeDecayRichAttention(nn.Module):
    """时间衰减注意力"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], time_decay=0.1):
        super().__init__()
        self.time_decay = time_decay
        mlp_input = 4 * input_dim
        layers = []
        prev_dim = mlp_input
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.attention_mlp = nn.Sequential(*layers)
    
    def forward(self, query, keys, keys_mask=None):
        batch_size, seq_len, dim = keys.shape
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([
            keys, query_expanded,
            keys * query_expanded,
            keys - query_expanded
        ], dim=-1)
        attention_scores = self.attention_mlp(attention_input).squeeze(-1)
        
        positions = torch.arange(seq_len, device=keys.device).float()
        time_weights = torch.exp(self.time_decay * (positions - seq_len + 1))
        attention_scores = attention_scores * time_weights.unsqueeze(0)
        
        if keys_mask is not None:
            mask_bool = keys_mask.bool()
            attention_scores = attention_scores.masked_fill(~mask_bool, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        # 处理全零mask的情况（避免NaN）
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * keys, dim=1)
        return weighted_sum, attention_weights


class MultiHeadRichAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, input_dim, num_heads=4, hidden_dims=[64, 32]):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            self._build_attention_mlp(4 * input_dim, hidden_dims)
            for _ in range(num_heads)
        ])
        self.output_proj = nn.Linear(input_dim, input_dim)
    
    def _build_attention_mlp(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, query, keys, keys_mask=None):
        batch_size, seq_len, dim = keys.shape
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([
            keys, query_expanded,
            keys * query_expanded,
            keys - query_expanded
        ], dim=-1)
        
        head_outputs = []
        for head in self.attention_heads:
            scores = head(attention_input).squeeze(-1)
            if keys_mask is not None:
                mask_bool = keys_mask.bool()
                scores = scores.masked_fill(~mask_bool, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            # 处理全零mask的情况（避免NaN）
            weights = torch.where(
                torch.isnan(weights),
                torch.zeros_like(weights),
                weights
            )
            output = torch.sum(weights.unsqueeze(-1) * keys, dim=1)
            head_outputs.append(output)
        
        combined = torch.stack(head_outputs, dim=1).mean(dim=1)
        return self.output_proj(combined), None


class DINRichVariant(nn.Module):
    """DIN 消融变体 - 修复版"""
    
    def __init__(self, num_items, num_users, feature_dims, embedding_dim=64,
                 attention_type='base', enhanced_mlp=False):
        super().__init__()
        self.attention_type = attention_type
        self.enhanced_mlp = enhanced_mlp
        self.embedding_dim = embedding_dim
        
        # 基础嵌入
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        
        # 特征嵌入 (与 DINRichLite 一致)
        self.genre_embedding = nn.Embedding(
            feature_dims.get('primary_genre', 20) + 1, embedding_dim // 4, padding_idx=0
        )
        self.year_embedding = nn.Embedding(
            feature_dims.get('year_bucket', 10) + 1, embedding_dim // 4, padding_idx=0
        )
        self.age_embedding = nn.Embedding(
            feature_dims.get('age_bucket', 10) + 1, embedding_dim // 4
        )
        self.gender_embedding = nn.Embedding(3, embedding_dim // 4)
        self.occupation_embedding = nn.Embedding(
            feature_dims.get('occupation', 25) + 1, embedding_dim // 4
        )
        
        # 序列嵌入总维度: item + genre + year
        seq_embed_dim = embedding_dim + embedding_dim // 4 + embedding_dim // 4
        
        # 选择注意力类型
        if attention_type == 'time_decay':
            self.attention = TimeDecayRichAttention(seq_embed_dim)
        elif attention_type == 'multi_head':
            self.attention = MultiHeadRichAttention(seq_embed_dim, num_heads=4)
        else:
            self.attention = AttentionLayer(seq_embed_dim)
        
        # MLP 输入: interest + target + seq_mean + user_features
        mlp_input_dim = (
            seq_embed_dim +     # interest_emb
            seq_embed_dim +     # target_emb  
            seq_embed_dim +     # seq_mean
            embedding_dim +     # user_emb
            embedding_dim // 4 + # age
            embedding_dim // 4 + # gender
            embedding_dim // 4   # occupation
        )
        
        if enhanced_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 256),
                nn.BatchNorm1d(256),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.PReLU(),
                nn.Linear(64, 1)
            )
        else:
            # 基础 MLP 也需要 BatchNorm 和 Dropout 防止过拟合
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 256),
                nn.BatchNorm1d(256),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.PReLU(),
                nn.Linear(64, 1)
            )
    
    def forward(self, batch):
        # 序列
        item_seq = batch['item_seq']  # [B, L]
        seq_mask = batch['item_seq_mask']  # [B, L]
        
        # 序列嵌入
        item_emb = self.item_embedding(item_seq)  # [B, L, D]
        genre_emb = self.genre_embedding(batch['history_genres'])  # [B, L, D/4]
        year_emb = self.year_embedding(batch['history_years'])  # [B, L, D/4]
        seq_emb = torch.cat([item_emb, genre_emb, year_emb], dim=-1)  # [B, L, D+D/2]
        
        # 目标物品嵌入
        target_item_emb = self.item_embedding(batch['target_item'])  # [B, D]
        target_genre_emb = self.genre_embedding(batch['item_genre'])  # [B, D/4]
        target_year_emb = self.year_embedding(batch['item_year'])  # [B, D/4]
        target_emb = torch.cat([target_item_emb, target_genre_emb, target_year_emb], dim=-1)
        
        # 用户嵌入
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        occupation_emb = self.occupation_embedding(batch['user_occupation'])
        
        # 注意力
        interest_emb, _ = self.attention(target_emb, seq_emb, seq_mask)
        
        # 序列平均
        seq_mean = (seq_emb * seq_mask.unsqueeze(-1)).sum(dim=1) / (seq_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        # 拼接所有特征
        mlp_input = torch.cat([
            interest_emb, target_emb, seq_mean,
            user_emb, age_emb, gender_emb, occupation_emb
        ], dim=-1)
        
        logits = self.mlp(mlp_input).squeeze(-1)
        return logits


# ========================================
# 混合精排模块
# ========================================

class HybridRanker:
    """DIN + LightGBM 混合精排"""
    
    def __init__(self, din_model, device='cpu'):
        self.din_model = din_model
        self.device = device
        self.lgb_model = None
    
    @torch.no_grad()
    def extract_din_features(self, data_loader):
        """提取 DIN 嵌入作为特征"""
        self.din_model.eval()
        self.din_model.to(self.device)
        
        all_embeddings = []
        all_scores = []
        all_labels = []
        
        for batch in data_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # 获取嵌入
            item_seq = batch['item_seq']
            seq_emb = self.din_model.item_embedding(item_seq)
            target_emb = self.din_model.item_embedding(batch['target_item'])
            user_emb = self.din_model.user_embedding(batch['user_id'])
            
            seq_mask = (item_seq > 0).float()
            seq_mean = (seq_emb * seq_mask.unsqueeze(-1)).sum(dim=1) / (seq_mask.sum(dim=1, keepdim=True) + 1e-8)
            
            # 拼接特征
            features = torch.cat([target_emb, user_emb, seq_mean], dim=-1)
            all_embeddings.append(features.cpu().numpy())
            
            # DIN 分数
            score = torch.sigmoid(self.din_model(batch))
            all_scores.append(score.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # 拼接 DIN 分数作为特征
        features = np.column_stack([embeddings, scores])
        return features, labels
    
    def train_lgb(self, train_loader, valid_loader):
        """训练 LightGBM"""
        if not HAS_LIGHTGBM:
            return None
        
        X_train, y_train = self.extract_din_features(train_loader)
        X_valid, y_valid = self.extract_din_features(valid_loader)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1,
            'random_state': 2020
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        
        self.lgb_model = lgb.train(
            params, train_data,
            num_boost_round=300,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
        return self.lgb_model
    
    def evaluate(self, test_loader):
        """评估混合模型"""
        from sklearn.metrics import roc_auc_score, log_loss
        
        X_test, y_test = self.extract_din_features(test_loader)
        y_pred = self.lgb_model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)
        
        return {'auc': auc, 'logloss': logloss}


# ========================================
# 实验一：序列长度敏感性 + 模型对比
# ========================================

def run_experiment1(dataset_name):
    """实验一：不同序列长度下各模型的表现"""
    print("\n" + "=" * 80)
    print(f"📊 实验一：序列长度敏感性 + 模型对比 [{dataset_name}]")
    print("=" * 80)
    
    results = []
    
    for seq_length in SEQ_LENGTHS:
        print(f"\n🔬 序列长度: {seq_length}")
        
        # 获取 Top-K 评估数据（仅在启用时）
        eval_data, fp_eval, ie_eval = None, None, None
        if ENABLE_TOPK:
            eval_data, eval_info, fp_eval, ie_eval = get_topk_eval_data(
                data_dir='./data',
                dataset_name=dataset_name,
                max_seq_length=seq_length,
                num_neg_samples=NUM_NEG_SAMPLES
            )
            # 智能采样用户（根据数据集自动决定）
            topk_sample = get_topk_sample_users(dataset_name, TOPK_SAMPLE_CONFIG)
            if topk_sample and len(eval_data) > topk_sample:
                import random
                random.seed(2020)
                eval_data = random.sample(eval_data, topk_sample)
                print(f"  (Top-K 采样 {topk_sample}/{eval_info['num_users']} 用户，误差±{100*1.96/topk_sample**0.5:.1f}%)")
            else:
                print(f"  (Top-K 全量评估 {len(eval_data)} 用户)")
        
        for model_name in MODELS_TO_TEST:
            print(f"  🚀 {model_name}...", end=" ", flush=True)
            
            # 自适应 batch size（SASRec 长序列需要减小）
            adaptive_batch = get_adaptive_batch_size(model_name, seq_length, BATCH_SIZE)
            if adaptive_batch != BATCH_SIZE:
                print(f"(batch={adaptive_batch}) ", end="", flush=True)
            
            # 为每个模型重新加载数据（batch size 可能不同）
            train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
                data_dir='./data',
                dataset_name=dataset_name,
                max_seq_length=seq_length,
                batch_size=adaptive_batch,
                num_workers=NUM_WORKERS,
                prefetch_factor=PREFETCH_FACTOR
            )
            
            try:
                if model_name == 'DIN':
                    model = DINRichLite(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM
                    )
                elif model_name == 'GRU4Rec':
                    model = GRU4Rec(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM,
                        hidden_dim=EMBEDDING_DIM
                    )
                elif model_name == 'SASRec':
                    model = SASRec(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM,
                        num_heads=2,
                        num_layers=2,
                        max_seq_len=seq_length
                    )
                elif model_name == 'NARM':
                    model = NARM(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM,
                        hidden_dim=EMBEDDING_DIM
                    )
                elif model_name == 'AvgPool':
                    model = SimpleAveragePoolingRich(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM
                    )
                
                # 创建训练器 - 3x4090充分利用多GPU
                trainer = RichTrainer(
                    model=model, 
                    device=DEVICE, 
                    use_multi_gpu=USE_MULTI_GPU,
                    use_tensorboard=ENABLE_TENSORBOARD,
                    log_dir=TENSORBOARD_LOG_DIR,
                    experiment_name=f'exp1_{dataset_name}_{model_name}_seq{seq_length}'
                )
                t1 = time.time()
                train_result = trainer.fit(
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    epochs=EPOCHS,
                    early_stopping_patience=10,
                    show_progress=False
                )
                train_time = time.time() - t1
                
                # CTR 指标
                test_metrics = trainer.evaluate(test_loader)
                speed = measure_inference_speed_rich(trainer.raw_model, test_loader, DEVICE)
                
                result_entry = {
                    'experiment': 'exp1_seq_model',
                    'dataset': dataset_name,
                    'seq_length': seq_length,
                    'model': model_name,
                    'test_auc': test_metrics['auc'],
                    'test_logloss': test_metrics['logloss'],
                    'best_valid_auc': train_result['best_valid_auc'],
                    'train_time_sec': train_time,
                    'qps': speed['qps'],
                    'num_params': sum(p.numel() for p in trainer.raw_model.parameters()),
                    'status': 'success'
                }
                
                # Top-K 指标（仅在启用时）
                if ENABLE_TOPK and eval_data is not None:
                    topk_metrics = trainer.evaluate_topk(
                        eval_data=eval_data,
                        feature_processor=fp_eval,
                        interaction_extractor=ie_eval,
                        max_seq_length=seq_length,
                        ks=TOPK_VALUES,
                        show_progress=False
                    )
                    result_entry.update(topk_metrics)
                    print(f"AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}, Time={train_time:.1f}s")
                else:
                    print(f"AUC={test_metrics['auc']:.4f}, Time={train_time:.1f}s")
                
                results.append(result_entry)
                
            except Exception as e:
                print(f"❌ {str(e)[:50]}")
                results.append({
                    'experiment': 'exp1_seq_model',
                    'dataset': dataset_name,
                    'seq_length': seq_length,
                    'model': model_name,
                    'test_auc': None,
                    'status': f'error: {str(e)[:100]}'
                })
    
    return results


# ========================================
# 实验二：方法对比 + LightGBM + 混合精排
# ========================================

def run_experiment2(dataset_name):
    """实验二：DIN vs 传统方法 + 混合精排"""
    print("\n" + "=" * 80)
    print(f"📊 实验二：方法对比 + 混合精排 [{dataset_name}]")
    print("=" * 80)
    
    results = []
    seq_length = 50
    
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    # 获取 Top-K 评估数据（仅在启用时）
    eval_data, fp_eval, ie_eval = None, None, None
    if ENABLE_TOPK:
        eval_data, eval_info, fp_eval, ie_eval = get_topk_eval_data(
            data_dir='./data',
            dataset_name=dataset_name,
            max_seq_length=seq_length,
            num_neg_samples=NUM_NEG_SAMPLES
        )
        # 智能采样用户（根据数据集自动决定）
        topk_sample = get_topk_sample_users(dataset_name, TOPK_SAMPLE_CONFIG)
        if topk_sample and len(eval_data) > topk_sample:
            import random
            random.seed(2020)
            eval_data = random.sample(eval_data, topk_sample)
            print(f"  (Top-K 采样 {topk_sample}/{eval_info['num_users']} 用户)")
        else:
            print(f"  (Top-K 全量评估 {len(eval_data)} 用户)")
    
    din_model = None  # 保存用于混合精排
    din_train_time = 0  # 保存 DIN 训练时间，用于混合精排公平对比
    din_num_params = 0  # 保存 DIN 参数量
    
    # 测试各深度模型
    for model_name in MODELS_TO_TEST:
        print(f"  🚀 {model_name}...", end=" ", flush=True)
        
        try:
            if model_name == 'DIN':
                model = DINRichLite(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM
                )
            elif model_name == 'GRU4Rec':
                model = GRU4Rec(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM,
                    hidden_dim=EMBEDDING_DIM
                )
            elif model_name == 'SASRec':
                model = SASRec(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM,
                    num_heads=2,
                    num_layers=2,
                    max_seq_len=seq_length
                )
            elif model_name == 'NARM':
                model = NARM(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM,
                    hidden_dim=EMBEDDING_DIM
                )
            elif model_name == 'AvgPool':
                model = SimpleAveragePoolingRich(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM
                )
            
            # 3x4090充分利用多GPU
            trainer = RichTrainer(
                model=model, 
                device=DEVICE, 
                use_multi_gpu=USE_MULTI_GPU,
                use_tensorboard=ENABLE_TENSORBOARD,
                log_dir=TENSORBOARD_LOG_DIR,
                experiment_name=f'exp2_{dataset_name}_{model_name}'
            )
            t1 = time.time()
            train_result = trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=EPOCHS,
                early_stopping_patience=10,
                show_progress=False
            )
            train_time = time.time() - t1
            
            # 保存 DIN 模型和训练时间（用于混合精排）
            if model_name == 'DIN':
                din_model = trainer.raw_model
                din_train_time = train_time
                din_num_params = sum(p.numel() for p in trainer.raw_model.parameters())
            
            # CTR 指标
            test_metrics = trainer.evaluate(test_loader)
            speed = measure_inference_speed_rich(trainer.raw_model, test_loader, DEVICE)
            
            result_entry = {
                'experiment': 'exp2_method_compare',
                'dataset': dataset_name,
                'model': model_name,
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'train_time_sec': train_time,
                'qps': speed['qps'],
                'num_params': sum(p.numel() for p in trainer.raw_model.parameters()),
                'status': 'success'
            }
            
            # Top-K 指标（仅在启用时）
            if ENABLE_TOPK and eval_data is not None:
                topk_metrics = trainer.evaluate_topk(
                    eval_data=eval_data,
                    feature_processor=fp_eval,
                    interaction_extractor=ie_eval,
                    max_seq_length=seq_length,
                    ks=TOPK_VALUES,
                    show_progress=False
                )
                result_entry.update(topk_metrics)
                print(f"AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}")
            else:
                print(f"AUC={test_metrics['auc']:.4f}")
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            results.append({
                'experiment': 'exp2_method_compare',
                'dataset': dataset_name,
                'model': model_name,
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    # LightGBM 单独
    if HAS_LIGHTGBM:
        print("  🚀 LightGBM (pure)...", end=" ", flush=True)
        try:
            from sklearn.metrics import roc_auc_score, log_loss
            from sklearn.model_selection import train_test_split
            
            data_path = os.path.join('./data', dataset_name)
            if dataset_name == 'ml-100k':
                interactions = pd.read_csv(
                    os.path.join(data_path, 'u.data'),
                    sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp']
                )
            else:
                interactions = pd.read_csv(
                    os.path.join(data_path, 'ratings.dat'),
                    sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'],
                    engine='python'
                )
            
            interaction_extractor = InteractionFeatureExtractor(interactions)
            X, y, feature_names = prepare_lightgbm_features(
                interactions, fp, interaction_extractor, max_seq_length=seq_length
            )
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.125, random_state=2020)
            
            params = {
                'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
                'verbose': -1, 'random_state': 2020
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            
            t1 = time.time()
            lgb_model = lgb.train(
                params, train_data, num_boost_round=500,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            train_time = time.time() - t1
            
            y_pred = lgb_model.predict(X_test)
            test_auc = roc_auc_score(y_test, y_pred)
            
            # LightGBM 参数量估算（叶子数 × 树的数量 × 特征数）
            lgb_num_trees = lgb_model.num_trees()
            lgb_num_leaves = params['num_leaves']
            lgb_num_params = lgb_num_trees * lgb_num_leaves  # 近似参数量
            
            results.append({
                'experiment': 'exp2_method_compare',
                'dataset': dataset_name,
                'model': 'LightGBM',
                'test_auc': test_auc,
                'train_time_sec': train_time,
                'num_params': lgb_num_params,
                'status': 'success'
            })
            print(f"AUC={test_auc:.4f}, params={lgb_num_params}")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            results.append({
                'experiment': 'exp2_method_compare',
                'dataset': dataset_name,
                'model': 'LightGBM',
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    # 混合精排：DIN + LightGBM
    if HAS_LIGHTGBM and din_model is not None:
        print("  🚀 Hybrid (DIN + LightGBM)...", end=" ", flush=True)
        try:
            hybrid = HybridRanker(din_model, device=DEVICE)
            t1 = time.time()
            hybrid.train_lgb(train_loader, valid_loader)
            lgb_train_time = time.time() - t1

            # 公平对比：总训练时间 = DIN训练时间 + LightGBM训练时间
            total_train_time = din_train_time + lgb_train_time

            test_metrics = hybrid.evaluate(test_loader)

            # Top-K 指标（与其他模型对齐）
            topk_metrics = {}
            if ENABLE_TOPK and eval_data is not None:
                # 使用完整的 Hybrid 流程：DIN 提取特征 -> LightGBM 预测
                from tqdm import tqdm
                from data_loader import build_topk_batch_multi
                all_labels = []
                all_scores = []
                for entry in tqdm(eval_data, desc="Hybrid Top-K"):
                    # 构建单用户的候选 batch
                    batch = build_topk_batch_multi(
                        entry, fp_eval, ie_eval, seq_length, DEVICE
                    )
                    
                    # 用完整的 Hybrid 流程预测分数
                    with torch.no_grad():
                        # DIN 提取特征
                        features, _ = hybrid.extract_din_features([batch])
                        # LightGBM 预测
                        scores = hybrid.lgb_model.predict(features)
                    
                    all_scores.append(scores)
                    labels = [1 if c == entry['ground_truth'] else 0 for c in entry['candidates']]
                    all_labels.append(np.array(labels))
                # 拼接
                all_scores = np.stack(all_scores)
                all_labels = np.stack(all_labels)
                # 计算 Top-K 指标
                def calc_topk_metrics(scores, labels, ks):
                    metrics = {}
                    for k in ks:
                        # HR@K
                        hits = 0
                        ndcg = 0
                        mrr = 0
                        for s, l in zip(scores, labels):
                            idx = np.argsort(-s)[:k]
                            rel = l[idx]
                            hits += rel.max()
                            if rel.max() > 0:
                                rank = np.where(rel == 1)[0][0] + 1
                                ndcg += 1 / np.log2(rank + 1)
                                mrr += 1 / rank
                        n = len(scores)
                        hr_k = hits / n
                        metrics[f'HR@{k}'] = hr_k
                        metrics[f'Recall@{k}'] = hr_k  # 单 GT 等于 HR
                        metrics[f'NDCG@{k}'] = ndcg / n
                        metrics[f'MRR@{k}'] = mrr / n
                        metrics[f'Precision@{k}'] = hr_k / k
                    return metrics
                topk_metrics = calc_topk_metrics(all_scores, all_labels, TOPK_VALUES)
                print(f"AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}, total_time={total_train_time:.2f}s (DIN:{din_train_time:.2f}s + LGB:{lgb_train_time:.2f}s)")
            else:
                print(f"AUC={test_metrics['auc']:.4f}, total_time={total_train_time:.2f}s (DIN:{din_train_time:.2f}s + LGB:{lgb_train_time:.2f}s)")

            # Hybrid 参数量 = DIN参数量 + LightGBM参数量（估算）
            lgb_num_params_hybrid = hybrid.lgb_model.num_trees() * 31 if hybrid.lgb_model else 0
            total_num_params = din_num_params + lgb_num_params_hybrid

            result_entry = {
                'experiment': 'exp2_hybrid',
                'dataset': dataset_name,
                'model': 'DIN+LightGBM',
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'train_time_sec': total_train_time,
                'din_train_time': din_train_time,
                'lgb_train_time': lgb_train_time,
                'num_params': total_num_params,
                'din_num_params': din_num_params,
                'lgb_num_params': lgb_num_params_hybrid,
                'status': 'success'
            }
            result_entry.update(topk_metrics)
            results.append(result_entry)

        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            results.append({
                'experiment': 'exp2_hybrid',
                'dataset': dataset_name,
                'model': 'DIN+LightGBM',
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })

    return results


# ========================================
# 实验三：DIN 消融实验
# ========================================

def run_experiment3(dataset_name):
    """实验三：DIN 改进消融实验"""
    print("\n" + "=" * 80)
    print(f"📊 实验三：DIN 消融实验 [{dataset_name}]")
    print("=" * 80)
    
    results = []
    seq_length = 50
    
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    # 获取 Top-K 评估数据（仅在启用时）
    eval_data, fp_eval, ie_eval = None, None, None
    if ENABLE_TOPK:
        eval_data, eval_info, fp_eval, ie_eval = get_topk_eval_data(
            data_dir='./data',
            dataset_name=dataset_name,
            max_seq_length=seq_length,
            num_neg_samples=NUM_NEG_SAMPLES
        )
        # 智能采样用户（根据数据集自动决定）
        topk_sample = get_topk_sample_users(dataset_name, TOPK_SAMPLE_CONFIG)
        if topk_sample and len(eval_data) > topk_sample:
            import random
            random.seed(2020)
            eval_data = random.sample(eval_data, topk_sample)
            print(f"  (Top-K 采样 {topk_sample}/{eval_info['num_users']} 用户)")
        else:
            print(f"  (Top-K 全量评估 {len(eval_data)} 用户)")
    
    # 消融变体
    ablation_variants = [
        ('DIN-Base', 'base', False),
        ('DIN-TimeDec', 'time_decay', False),
        ('DIN-MultiHead', 'multi_head', False),
        ('DIN-Enhanced', 'base', True),
        ('DIN-Full', 'time_decay', True),
    ]
    
    for variant_name, attention_type, enhanced_mlp in ablation_variants:
        print(f"  🚀 {variant_name}...", end=" ", flush=True)
        
        try:
            model = DINRichVariant(
                num_items=dataset_info['num_items'],
                num_users=dataset_info['num_users'],
                feature_dims=dataset_info['feature_dims'],
                embedding_dim=EMBEDDING_DIM,
                attention_type=attention_type,
                enhanced_mlp=enhanced_mlp
            )
            
            # 3x4090充分利用多GPU
            trainer = RichTrainer(
                model=model, 
                device=DEVICE, 
                use_multi_gpu=USE_MULTI_GPU,
                use_tensorboard=ENABLE_TENSORBOARD,
                log_dir=TENSORBOARD_LOG_DIR,
                experiment_name=f'exp3_{dataset_name}_{variant_name}'
            )
            t1 = time.time()
            train_result = trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=EPOCHS,
                early_stopping_patience=10,
                show_progress=False
            )
            train_time = time.time() - t1
            
            # CTR 指标
            test_metrics = trainer.evaluate(test_loader)
            speed = measure_inference_speed_rich(trainer.raw_model, test_loader, DEVICE)
            
            result_entry = {
                'experiment': 'exp3_ablation',
                'dataset': dataset_name,
                'variant': variant_name,
                'attention_type': attention_type,
                'enhanced_mlp': enhanced_mlp,
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'best_valid_auc': train_result['best_valid_auc'],
                'train_time_sec': train_time,
                'qps': speed['qps'],
                'num_params': sum(p.numel() for p in trainer.raw_model.parameters()),
                'status': 'success'
            }
            
            # Top-K 指标（仅在启用时）
            if ENABLE_TOPK and eval_data is not None:
                topk_metrics = trainer.evaluate_topk(
                    eval_data=eval_data,
                    feature_processor=fp_eval,
                    interaction_extractor=ie_eval,
                    max_seq_length=seq_length,
                    ks=TOPK_VALUES,
                    show_progress=False
                )
                result_entry.update(topk_metrics)
                print(f"AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}")
            else:
                print(f"AUC={test_metrics['auc']:.4f}")
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            results.append({
                'experiment': 'exp3_ablation',
                'dataset': dataset_name,
                'variant': variant_name,
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    return results


# ========================================
# 实验四：高级改进实验
# ========================================

def run_experiment4(dataset_name, part='all'):
    """
    运行实验四：高级改进实验
    
    Args:
        dataset_name: 数据集名称
        part: 运行哪部分 ('all', 'adaptive', 'contrastive')
    """
    print(f"\n{'='*60}")
    print(f"🧪 实验四：高级改进实验 ({dataset_name})")
    print(f"{'='*60}")
    
    results = []
    
    try:
        # 动态导入 experiment4 模块
        import importlib.util
        exp4_path = os.path.join(os.path.dirname(__file__), 'experiment4.py')
        
        if not os.path.exists(exp4_path):
            print("❌ experiment4.py 不存在，跳过实验四")
            return results
            
        spec = importlib.util.spec_from_file_location("experiment4", exp4_path)
        exp4_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp4_module)
        
        # 运行自适应时间衰减实验
        if part in ['all', 'adaptive']:
            print("\n📊 Part 1: 自适应时间衰减实验")
            print("-" * 40)
            try:
                adaptive_results = exp4_module.run_adaptive_decay_experiment(
                    dataset_name=dataset_name,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    device=DEVICE
                )
                for r in adaptive_results:
                    r['experiment'] = 'exp4_adaptive_decay'
                    r['dataset'] = dataset_name
                results.extend(adaptive_results)
                print(f"✅ 自适应时间衰减实验完成，{len(adaptive_results)} 组结果")
            except Exception as e:
                print(f"❌ 自适应时间衰减实验失败: {e}")
                results.append({
                    'experiment': 'exp4_adaptive_decay',
                    'dataset': dataset_name,
                    'status': f'error: {str(e)[:100]}'
                })
        
        # 运行对比学习实验
        if part in ['all', 'contrastive']:
            print("\n📊 Part 2: 对比学习实验")
            print("-" * 40)
            try:
                contrastive_results = exp4_module.run_contrastive_experiment(
                    dataset_name=dataset_name,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    device=DEVICE
                )
                for r in contrastive_results:
                    r['experiment'] = 'exp4_contrastive'
                    r['dataset'] = dataset_name
                results.extend(contrastive_results)
                print(f"✅ 对比学习实验完成，{len(contrastive_results)} 组结果")
            except Exception as e:
                print(f"❌ 对比学习实验失败: {e}")
                results.append({
                    'experiment': 'exp4_contrastive',
                    'dataset': dataset_name,
                    'status': f'error: {str(e)[:100]}'
                })
                
    except Exception as e:
        print(f"❌ 实验四加载失败: {e}")
        results.append({
            'experiment': 'exp4',
            'dataset': dataset_name,
            'status': f'load_error: {str(e)[:100]}'
        })
    
    return results


# ========================================
# 主程序
# ========================================

if __name__ == '__main__':
    all_results = []
    experiment_start = datetime.now()
    
    print(f"\n⏰ 实验开始时间: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"📁 数据集: {dataset.upper()}")
        print(f"{'='*80}")
        
        if 1 in EXPERIMENTS_TO_RUN:
            results1 = run_experiment1(dataset)
            all_results.extend(results1)
        
        if 2 in EXPERIMENTS_TO_RUN:
            results2 = run_experiment2(dataset)
            all_results.extend(results2)
        
        if 3 in EXPERIMENTS_TO_RUN:
            results3 = run_experiment3(dataset)
            all_results.extend(results3)
        
        if 4 in EXPERIMENTS_TO_RUN:
            results4 = run_experiment4(dataset, part=args.exp4_part)
            all_results.extend(results4)
    
    # 保存结果
    experiment_end = datetime.now()
    total_time = (experiment_end - experiment_start).total_seconds()
    
    df_results = pd.DataFrame(all_results)
    timestamp = experiment_start.strftime('%Y%m%d_%H%M%S')
    
    # CSV
    csv_file = os.path.join(RESULTS_DIR, f'all_results_{timestamp}.csv')
    df_results.to_csv(csv_file, index=False)
    
    # JSON 报告
    report = {
        'timestamp': timestamp,
        'device': DEVICE,
        'gpu_name': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU',
        'datasets': DATASETS,
        'experiments': EXPERIMENTS_TO_RUN,
        'epochs': EPOCHS,
        'seq_lengths': SEQ_LENGTHS,
        'models': MODELS_TO_TEST,
        'topk_values': TOPK_VALUES,
        'total_time_minutes': total_time / 60,
        'num_results': len(all_results),
        'results': all_results
    }
    
    json_file = os.path.join(RESULTS_DIR, f'report_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("📋 实验完成！")
    print("=" * 80)
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"实验数量: {len(all_results)}")
    print(f"\n📂 结果文件:")
    print(f"   {csv_file}")
    print(f"   {json_file}")
    
    # 各数据集最佳结果
    df_success = df_results[df_results['status'] == 'success']
    
    print("\n📊 各实验最佳结果 (CTR 指标):")
    for exp_name in df_success['experiment'].unique():
        df_exp = df_success[df_success['experiment'] == exp_name]
        if len(df_exp) > 0 and 'test_auc' in df_exp.columns:
            best = df_exp.loc[df_exp['test_auc'].idxmax()]
            model_col = 'model' if 'model' in best else 'variant'
            print(f"  {exp_name}: {best.get(model_col, 'N/A')} - AUC={best['test_auc']:.4f}")
    
    # Top-K 指标摘要
    if 'HR@10' in df_success.columns:
        print("\n📊 各实验最佳结果 (Top-K 指标):")
        for exp_name in df_success['experiment'].unique():
            df_exp = df_success[df_success['experiment'] == exp_name]
            if len(df_exp) > 0 and 'NDCG@10' in df_exp.columns and df_exp['NDCG@10'].notna().any():
                best = df_exp.loc[df_exp['NDCG@10'].idxmax()]
                model_col = 'model' if 'model' in best else 'variant'
                print(f"  {exp_name}: {best.get(model_col, 'N/A')} - HR@10={best['HR@10']:.4f}, NDCG@10={best['NDCG@10']:.4f}")
    
    print("=" * 80)
    print("✅ 所有实验完成！")
