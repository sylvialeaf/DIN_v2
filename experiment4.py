#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验四：DIN 高级改进实验

研究方向：
1. 自适应时间衰减（Adaptive Time Decay）
   - 将固定的 decay_rate 改为可学习参数
   - 探索用户级别的个性化衰减

2. 对比学习预训练（Contrastive Learning Pre-training）
   - 使用 SimCLR/InfoNCE 风格的对比损失
   - 数据增强：序列裁剪、物品替换、掩码
   - 两阶段训练：预训练 → 微调

创新点：
- 首次在 DIN 框架下探索自适应衰减
- 对比学习解决冷启动问题
- 消融实验验证各组件贡献

输出:
- results/experiment4_adaptive_decay.csv
- results/experiment4_contrastive.csv
- results/experiment4_combined.csv
"""

import os
import sys
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
import copy

from data_loader import get_rich_dataloaders, get_topk_eval_data, build_topk_batch_multi
from trainer import RichTrainer, measure_inference_speed_rich


# ========================================
# Top-K 评估指标函数（与 run_ddp.py 一致）
# ========================================

def hit_at_k(ranked_items, ground_truth, k):
    """Hit Rate @ K"""
    return 1.0 if ground_truth in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items, ground_truth, k):
    """NDCG @ K"""
    for i, item in enumerate(ranked_items[:k]):
        if item == ground_truth:
            return 1.0 / np.log2(i + 2)
    return 0.0


def mrr_at_k(ranked_items, ground_truth, k):
    """MRR @ K"""
    for i, item in enumerate(ranked_items[:k]):
        if item == ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked_items, ground_truth, k):
    """Precision @ K"""
    hits = 1 if ground_truth in ranked_items[:k] else 0
    return hits / k


def evaluate_topk_metrics(model, eval_data, feature_processor, interaction_extractor, 
                          max_seq_length, device, ks=[5, 10, 20]):
    """
    统一的 Top-K 评估函数
    
    与 run_ddp.py 中的 SimpleDDPTrainer.evaluate_topk 保持一致
    """
    model.eval()
    
    all_hr = {k: [] for k in ks}
    all_ndcg = {k: [] for k in ks}
    all_mrr = {k: [] for k in ks}
    all_precision = {k: [] for k in ks}
    
    with torch.no_grad():
        for eval_item in eval_data:
            batch = build_topk_batch_multi(
                eval_item, feature_processor, interaction_extractor,
                max_seq_length, device
            )
            
            logits = model(batch)
            scores = torch.sigmoid(logits).cpu().numpy()
            
            candidates = eval_item['candidates']
            ground_truth = eval_item['ground_truth']
            sorted_indices = np.argsort(-scores)
            ranked_items = [candidates[i] for i in sorted_indices]
            
            for k in ks:
                all_hr[k].append(hit_at_k(ranked_items, ground_truth, k))
                all_ndcg[k].append(ndcg_at_k(ranked_items, ground_truth, k))
                all_mrr[k].append(mrr_at_k(ranked_items, ground_truth, k))
                all_precision[k].append(precision_at_k(ranked_items, ground_truth, k))
    
    results = {}
    for k in ks:
        results[f'HR@{k}'] = np.mean(all_hr[k])
        results[f'Recall@{k}'] = np.mean(all_hr[k])  # 单 GT 等于 HR
        results[f'NDCG@{k}'] = np.mean(all_ndcg[k])
        results[f'MRR@{k}'] = np.mean(all_mrr[k])
        results[f'Precision@{k}'] = np.mean(all_precision[k])
    
    return results


# ========================================
# Part 1: 自适应时间衰减注意力
# ========================================

class AdaptiveTimeDecayAttention(nn.Module):
    """
    自适应时间衰减注意力机制
    
    创新点：
    1. decay_rate 作为可学习参数
    2. 支持全局学习或用户级别个性化
    
    公式: score_i = base_score_i * exp(decay_rate * (pos_i - L + 1))
    """
    
    def __init__(
        self, 
        input_dim, 
        hidden_dims=[64, 32], 
        init_decay=0.1,
        learnable_decay=True,
        per_user_decay=False,
        num_users=None
    ):
        super(AdaptiveTimeDecayAttention, self).__init__()
        
        self.learnable_decay = learnable_decay
        self.per_user_decay = per_user_decay
        
        # 基础注意力 MLP
        mlp_input = 4 * input_dim
        layers = []
        prev_dim = mlp_input
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.attention_mlp = nn.Sequential(*layers)
        
        # 时间衰减参数
        if learnable_decay:
            if per_user_decay and num_users is not None:
                # 用户级别的个性化衰减（高级版本）
                self.decay_rate = nn.Embedding(num_users + 1, 1)
                nn.init.constant_(self.decay_rate.weight, init_decay)
            else:
                # 全局可学习衰减
                self.decay_rate = nn.Parameter(torch.tensor(init_decay))
        else:
            # 固定衰减（baseline）
            self.register_buffer('decay_rate', torch.tensor(init_decay))
    
    def forward(self, query, keys, keys_mask=None, user_ids=None):
        """
        Args:
            query: [B, D] 目标物品嵌入
            keys: [B, L, D] 历史序列嵌入
            keys_mask: [B, L] 有效位置掩码
            user_ids: [B] 用户ID（用于个性化衰减）
        """
        batch_size, seq_len, dim = keys.shape
        
        # 1. 计算基础注意力分数
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([
            keys, query_expanded,
            keys * query_expanded,
            keys - query_expanded
        ], dim=-1)
        base_scores = self.attention_mlp(attention_input).squeeze(-1)  # [B, L]
        
        # 2. 计算时间衰减权重
        positions = torch.arange(seq_len, device=keys.device).float()
        
        if self.learnable_decay:
            if self.per_user_decay and user_ids is not None:
                # 用户级别衰减
                user_decay = self.decay_rate(user_ids).squeeze(-1)  # [B]
                time_weights = torch.exp(
                    user_decay.unsqueeze(1) * (positions - seq_len + 1).unsqueeze(0)
                )
            else:
                # 全局衰减
                time_weights = torch.exp(self.decay_rate * (positions - seq_len + 1))
                time_weights = time_weights.unsqueeze(0)  # [1, L]
        else:
            time_weights = torch.exp(self.decay_rate * (positions - seq_len + 1))
            time_weights = time_weights.unsqueeze(0)
        
        # 3. 融合时间衰减
        attention_scores = base_scores * time_weights
        
        # 4. Mask 和 Softmax（使用 float('-inf') 与 models.py 保持一致）
        if keys_mask is not None:
            attention_scores = attention_scores.masked_fill(~keys_mask.bool(), float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 处理可能的 NaN（全 padding 情况）
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        
        # 5. 加权求和
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * keys, dim=1)
        
        return weighted_sum, attention_weights
    
    def get_decay_rate(self):
        """获取当前衰减率（用于监控和可视化）"""
        if self.learnable_decay and not self.per_user_decay:
            return self.decay_rate.item()
        return None


class DINAdaptiveDecay(nn.Module):
    """
    带自适应时间衰减的 DIN 模型
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        embedding_dim=64,
        feature_embedding_dim=16,
        mlp_hidden_dims=[256, 128, 64],
        dropout_rate=0.2,
        init_decay=0.1,
        learnable_decay=True,
        per_user_decay=False
    ):
        super(DINAdaptiveDecay, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.feature_embedding_dim = feature_embedding_dim
        
        # 嵌入层
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, feature_embedding_dim)
        self.genre_embedding = nn.Embedding(
            feature_dims.get('primary_genre', 20) + 1, 
            feature_embedding_dim, 
            padding_idx=0
        )
        self.year_embedding = nn.Embedding(
            feature_dims.get('year_bucket', 8) + 1, 
            feature_embedding_dim, 
            padding_idx=0
        )
        self.age_embedding = nn.Embedding(
            feature_dims.get('age_bucket', 10) + 1, 
            feature_embedding_dim
        )
        self.gender_embedding = nn.Embedding(3, feature_embedding_dim)
        self.occupation_embedding = nn.Embedding(
            feature_dims.get('occupation', 25) + 1, 
            feature_embedding_dim
        )
        
        # 序列特征维度
        self.seq_feature_dim = embedding_dim + 2 * feature_embedding_dim
        
        # 自适应时间衰减注意力
        self.attention = AdaptiveTimeDecayAttention(
            input_dim=self.seq_feature_dim,
            hidden_dims=[64, 32],
            init_decay=init_decay,
            learnable_decay=learnable_decay,
            per_user_decay=per_user_decay,
            num_users=num_users if per_user_decay else None
        )
        
        # MLP
        mlp_input_dim = (
            self.seq_feature_dim +  # 用户兴趣
            self.seq_feature_dim +  # 目标物品
            feature_embedding_dim +  # 用户嵌入
            feature_embedding_dim * 3  # 年龄 + 性别 + 职业
        )
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.PReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(self, batch):
        # 历史序列嵌入
        seq_item_emb = self.item_embedding(batch['item_seq'])
        seq_genre_emb = self.genre_embedding(batch['history_genres'])
        seq_year_emb = self.year_embedding(batch['history_years'])
        seq_combined = torch.cat([seq_item_emb, seq_genre_emb, seq_year_emb], dim=-1)
        
        # 目标物品嵌入
        target_item_emb = self.item_embedding(batch['target_item'])
        target_genre_emb = self.genre_embedding(batch['item_genre'])
        target_year_emb = self.year_embedding(batch['item_year'])
        target_combined = torch.cat([target_item_emb, target_genre_emb, target_year_emb], dim=-1)
        
        # 自适应时间衰减注意力
        user_interest, _ = self.attention(
            target_combined, 
            seq_combined, 
            batch['item_seq_mask'],
            batch.get('user_id', None)
        )
        
        # 用户特征
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        occupation_emb = self.occupation_embedding(batch['user_occupation'])
        
        # 拼接并预测
        features = torch.cat([
            user_interest, target_combined,
            user_emb, age_emb, gender_emb, occupation_emb
        ], dim=-1)
        
        return self.mlp(features).squeeze(-1)
    
    def get_decay_rate(self):
        """获取当前学习到的衰减率"""
        return self.attention.get_decay_rate()


# ========================================
# Part 2: 对比学习预训练
# ========================================

class SequenceAugmentation:
    """
    序列数据增强器
    
    支持多种增强策略：
    1. 随机裁剪（Crop）
    2. 随机掩码（Mask）
    3. 随机重排（Reorder）
    4. 随机替换（Substitute）
    """
    
    def __init__(
        self,
        crop_ratio=0.6,
        mask_ratio=0.2,
        reorder_ratio=0.2,
        substitute_ratio=0.1,
        num_items=None
    ):
        self.crop_ratio = crop_ratio
        self.mask_ratio = mask_ratio
        self.reorder_ratio = reorder_ratio
        self.substitute_ratio = substitute_ratio
        self.num_items = num_items
    
    def crop(self, seq, mask):
        """随机裁剪序列"""
        valid_len = mask.sum().int().item()
        if valid_len <= 2:
            return seq.clone(), mask.clone()
        
        crop_len = max(2, int(valid_len * self.crop_ratio))
        start = torch.randint(0, valid_len - crop_len + 1, (1,)).item()
        
        # 找到有效序列的起始位置
        valid_start = (mask == 0).sum().int().item()
        
        new_seq = seq.clone()
        new_mask = mask.clone()
        
        # 裁剪：将不在裁剪范围内的位置置为0
        crop_start = valid_start + start
        crop_end = crop_start + crop_len
        
        new_seq[:crop_start] = 0
        new_seq[crop_end:] = 0
        new_mask[:crop_start] = 0
        new_mask[crop_end:] = 0
        
        return new_seq, new_mask
    
    def mask(self, seq, mask):
        """随机掩码部分物品"""
        new_seq = seq.clone()
        valid_positions = mask.bool()
        num_valid = valid_positions.sum().item()
        
        if num_valid <= 1:
            return new_seq, mask
        
        num_mask = max(1, int(num_valid * self.mask_ratio))
        valid_indices = torch.where(valid_positions)[0]
        mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_mask]]
        new_seq[mask_indices] = 0  # 使用 padding_idx 作为 mask
        
        return new_seq, mask
    
    def reorder(self, seq, mask):
        """随机重排部分序列"""
        new_seq = seq.clone()
        valid_positions = mask.bool()
        valid_indices = torch.where(valid_positions)[0]
        num_valid = len(valid_indices)
        
        if num_valid <= 2:
            return new_seq, mask
        
        # 选择一段连续区间进行重排
        reorder_len = max(2, int(num_valid * self.reorder_ratio))
        start = torch.randint(0, num_valid - reorder_len + 1, (1,)).item()
        
        indices_to_reorder = valid_indices[start:start + reorder_len]
        reordered = indices_to_reorder[torch.randperm(reorder_len)]
        
        new_seq[indices_to_reorder] = seq[reordered]
        
        return new_seq, mask
    
    def substitute(self, seq, mask):
        """随机替换部分物品"""
        if self.num_items is None:
            return seq, mask
        
        new_seq = seq.clone()
        valid_positions = mask.bool()
        num_valid = valid_positions.sum().item()
        
        if num_valid <= 1:
            return new_seq, mask
        
        num_sub = max(1, int(num_valid * self.substitute_ratio))
        valid_indices = torch.where(valid_positions)[0]
        sub_indices = valid_indices[torch.randperm(len(valid_indices))[:num_sub]]
        
        # 随机替换为其他物品
        new_items = torch.randint(1, self.num_items + 1, (num_sub,), device=seq.device)
        new_seq[sub_indices] = new_items
        
        return new_seq, mask
    
    def augment(self, seq, mask, strategy='random'):
        """
        应用数据增强
        
        Args:
            seq: [L] 物品序列
            mask: [L] 有效位置掩码
            strategy: 增强策略，'crop', 'mask', 'reorder', 'substitute', 或 'random'
        """
        if strategy == 'crop':
            return self.crop(seq, mask)
        elif strategy == 'mask':
            return self.mask(seq, mask)
        elif strategy == 'reorder':
            return self.reorder(seq, mask)
        elif strategy == 'substitute':
            return self.substitute(seq, mask)
        elif strategy == 'random':
            # 随机选择一种策略
            strategies = ['crop', 'mask', 'reorder']
            if self.num_items is not None:
                strategies.append('substitute')
            choice = np.random.choice(strategies)
            return self.augment(seq, mask, choice)
        else:
            return seq, mask


class ContrastiveEncoder(nn.Module):
    """
    对比学习序列编码器
    
    将用户历史序列编码为向量表示，用于对比学习。
    """
    
    def __init__(
        self,
        num_items,
        embedding_dim=64,
        hidden_dim=128,
        output_dim=64
    ):
        super(ContrastiveEncoder, self).__init__()
        
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # 使用 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 投影头（对比学习关键组件）
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(self, item_seq, mask):
        """
        Args:
            item_seq: [B, L] 物品序列
            mask: [B, L] 有效位置掩码（1=有效，0=padding）
        
        Returns:
            z: [B, output_dim] 对比学习表示
        """
        # 嵌入
        seq_emb = self.item_embedding(item_seq)  # [B, L, D]
        
        # Transformer 编码（注意 mask 取反）
        src_key_padding_mask = ~mask.bool()  # True 表示忽略
        encoded = self.transformer(seq_emb, src_key_padding_mask=src_key_padding_mask)
        
        # 池化：取有效位置的平均
        mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
        pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # 投影
        z = self.projector(pooled)
        
        return z


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比损失
    
    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    
    其中 (z_i, z_j) 是正样本对，z_k 是批内负样本
    """
    
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Args:
            z1: [B, D] 增强视图1的表示
            z2: [B, D] 增强视图2的表示
        
        Returns:
            loss: 对比损失
        """
        batch_size = z1.shape[0]
        
        # L2 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 计算相似度矩阵
        # sim[i, j] = z1[i] · z2[j]
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature  # [B, B]
        
        # 正样本在对角线上
        labels = torch.arange(batch_size, device=z1.device)
        
        # 交叉熵损失（正样本是对角线元素）
        loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)
        
        return loss / 2


class DINWithContrastive(nn.Module):
    """
    带对比学习预训练的 DIN 模型
    
    训练流程：
    1. 预训练阶段：使用对比损失训练序列编码器
    2. 微调阶段：冻结/微调编码器，训练完整 DIN 模型
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        embedding_dim=64,
        feature_embedding_dim=16,
        mlp_hidden_dims=[256, 128, 64],
        dropout_rate=0.2,
        contrastive_dim=64,
        temperature=0.1
    ):
        super(DINWithContrastive, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.feature_embedding_dim = feature_embedding_dim
        
        # 对比学习编码器（用于预训练）
        self.contrastive_encoder = ContrastiveEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=128,
            output_dim=contrastive_dim
        )
        
        # 对比损失
        self.contrastive_loss = InfoNCELoss(temperature)
        
        # DIN 组件（复用编码器的嵌入层）
        self.item_embedding = self.contrastive_encoder.item_embedding  # 共享嵌入
        self.user_embedding = nn.Embedding(num_users + 1, feature_embedding_dim)
        self.genre_embedding = nn.Embedding(
            feature_dims.get('primary_genre', 20) + 1, 
            feature_embedding_dim, 
            padding_idx=0
        )
        self.year_embedding = nn.Embedding(
            feature_dims.get('year_bucket', 8) + 1, 
            feature_embedding_dim, 
            padding_idx=0
        )
        self.age_embedding = nn.Embedding(
            feature_dims.get('age_bucket', 10) + 1, 
            feature_embedding_dim
        )
        self.gender_embedding = nn.Embedding(3, feature_embedding_dim)
        self.occupation_embedding = nn.Embedding(
            feature_dims.get('occupation', 25) + 1, 
            feature_embedding_dim
        )
        
        # 注意力层
        self.seq_feature_dim = embedding_dim + 2 * feature_embedding_dim
        self.attention = self._build_attention(self.seq_feature_dim, [64, 32])
        
        # MLP
        mlp_input_dim = (
            self.seq_feature_dim +  # 用户兴趣
            self.seq_feature_dim +  # 目标物品
            feature_embedding_dim +  # 用户嵌入
            feature_embedding_dim * 3  # 年龄 + 性别 + 职业
        )
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.PReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 数据增强器
        self.augmenter = SequenceAugmentation(num_items=num_items)
        
        self._init_weights()
    
    def _build_attention(self, input_dim, hidden_dims):
        """构建注意力 MLP"""
        mlp_input = 4 * input_dim
        layers = []
        prev_dim = mlp_input
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for name, module in self.named_modules():
            # 跳过 contrastive_encoder（已经初始化过）
            if 'contrastive_encoder' in name:
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def attention_forward(self, query, keys, keys_mask):
        """注意力计算"""
        batch_size, seq_len, dim = keys.shape
        
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([
            keys, query_expanded,
            keys * query_expanded,
            keys - query_expanded
        ], dim=-1)
        
        attention_scores = self.attention(attention_input).squeeze(-1)
        
        if keys_mask is not None:
            attention_scores = attention_scores.masked_fill(~keys_mask.bool(), -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * keys, dim=1)
        
        return weighted_sum, attention_weights
    
    def contrastive_forward(self, item_seq, mask):
        """
        对比学习前向传播
        
        Returns:
            z1, z2: 两个增强视图的表示
        """
        batch_size = item_seq.shape[0]
        
        # 生成两个增强视图
        aug_seq1, aug_mask1 = [], []
        aug_seq2, aug_mask2 = [], []
        
        for i in range(batch_size):
            s1, m1 = self.augmenter.augment(item_seq[i], mask[i], 'random')
            s2, m2 = self.augmenter.augment(item_seq[i], mask[i], 'random')
            aug_seq1.append(s1)
            aug_mask1.append(m1)
            aug_seq2.append(s2)
            aug_mask2.append(m2)
        
        aug_seq1 = torch.stack(aug_seq1)
        aug_mask1 = torch.stack(aug_mask1)
        aug_seq2 = torch.stack(aug_seq2)
        aug_mask2 = torch.stack(aug_mask2)
        
        # 编码
        z1 = self.contrastive_encoder(aug_seq1, aug_mask1)
        z2 = self.contrastive_encoder(aug_seq2, aug_mask2)
        
        return z1, z2
    
    def forward(self, batch, return_contrastive_loss=False):
        """
        前向传播
        
        Args:
            batch: 数据批次
            return_contrastive_loss: 是否返回对比损失（用于联合训练）
        """
        # 历史序列嵌入
        seq_item_emb = self.item_embedding(batch['item_seq'])
        seq_genre_emb = self.genre_embedding(batch['history_genres'])
        seq_year_emb = self.year_embedding(batch['history_years'])
        seq_combined = torch.cat([seq_item_emb, seq_genre_emb, seq_year_emb], dim=-1)
        
        # 目标物品嵌入
        target_item_emb = self.item_embedding(batch['target_item'])
        target_genre_emb = self.genre_embedding(batch['item_genre'])
        target_year_emb = self.year_embedding(batch['item_year'])
        target_combined = torch.cat([target_item_emb, target_genre_emb, target_year_emb], dim=-1)
        
        # 注意力
        user_interest, _ = self.attention_forward(
            target_combined, seq_combined, batch['item_seq_mask']
        )
        
        # 用户特征
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        occupation_emb = self.occupation_embedding(batch['user_occupation'])
        
        # 拼接并预测
        features = torch.cat([
            user_interest, target_combined,
            user_emb, age_emb, gender_emb, occupation_emb
        ], dim=-1)
        
        logits = self.mlp(features).squeeze(-1)
        
        if return_contrastive_loss:
            z1, z2 = self.contrastive_forward(batch['item_seq'], batch['item_seq_mask'])
            cl_loss = self.contrastive_loss(z1, z2)
            return logits, cl_loss
        
        return logits


class ContrastiveTrainer(RichTrainer):
    """
    对比学习训练器
    
    支持两种训练模式：
    1. 预训练模式：只训练对比损失
    2. 联合训练模式：对比损失 + CTR 损失
    """
    
    def __init__(
        self,
        model,
        device='cpu',
        learning_rate=1e-3,
        weight_decay=1e-5,
        contrastive_weight=0.1,  # 对比损失权重
        use_tensorboard=True,
        log_dir='./runs',
        experiment_name=None
    ):
        super().__init__(
            model=model,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_tensorboard=use_tensorboard,
            log_dir=log_dir,
            experiment_name=experiment_name
        )
        self.contrastive_weight = contrastive_weight
    
    def pretrain_epoch(self, train_loader, show_progress=True):
        """预训练一个 epoch（只用对比损失）"""
        self.model.train()
        total_loss = 0
        
        from tqdm import tqdm
        iterator = tqdm(train_loader, desc='Pretraining') if show_progress else train_loader
        
        for batch in iterator:
            batch = self._move_batch_to_device(batch)
            
            self.optimizer.zero_grad()
            
            # 只计算对比损失
            z1, z2 = self.model.contrastive_forward(
                batch['item_seq'], batch['item_seq_mask']
            )
            loss = self.model.contrastive_loss(z1, z2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def train_epoch_joint(self, train_loader, show_progress=True):
        """联合训练一个 epoch（对比损失 + CTR 损失）"""
        self.model.train()
        total_loss = 0
        total_ctr_loss = 0
        total_cl_loss = 0
        
        from tqdm import tqdm
        iterator = tqdm(train_loader, desc='Joint Training') if show_progress else train_loader
        
        for batch in iterator:
            batch = self._move_batch_to_device(batch)
            
            self.optimizer.zero_grad()
            
            # CTR 损失 + 对比损失
            logits, cl_loss = self.model(batch, return_contrastive_loss=True)
            ctr_loss = self.criterion(logits, batch['label'])
            
            loss = ctr_loss + self.contrastive_weight * cl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_ctr_loss += ctr_loss.item()
            total_cl_loss += cl_loss.item()
        
        return {
            'total_loss': total_loss / len(train_loader),
            'ctr_loss': total_ctr_loss / len(train_loader),
            'cl_loss': total_cl_loss / len(train_loader)
        }
    
    def pretrain(self, train_loader, epochs=10, show_progress=True):
        """对比学习预训练"""
        print("=" * 60)
        print("对比学习预训练")
        print("=" * 60)
        
        for epoch in range(epochs):
            loss = self.pretrain_epoch(train_loader, show_progress)
            print(f"Pretrain Epoch {epoch+1}/{epochs} - CL Loss: {loss:.4f}")
            
            if self.use_tensorboard and self.writer is not None:
                self.writer.add_scalar('Pretrain/cl_loss', loss, epoch)
        
        return self
    
    def fit_joint(
        self,
        train_loader,
        valid_loader,
        epochs=20,
        early_stopping_patience=5,
        show_progress=True
    ):
        """联合训练（对比损失 + CTR 损失）"""
        best_valid_auc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            losses = self.train_epoch_joint(train_loader, show_progress)
            valid_metrics = self.evaluate(valid_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Total: {losses['total_loss']:.4f} - "
                  f"CTR: {losses['ctr_loss']:.4f} - "
                  f"CL: {losses['cl_loss']:.4f} - "
                  f"Valid AUC: {valid_metrics['auc']:.4f}")
            
            if self.use_tensorboard and self.writer is not None:
                self.writer.add_scalar('Loss/total', losses['total_loss'], epoch)
                self.writer.add_scalar('Loss/ctr', losses['ctr_loss'], epoch)
                self.writer.add_scalar('Loss/contrastive', losses['cl_loss'], epoch)
                self.writer.add_scalar('Metrics/valid_auc', valid_metrics['auc'], epoch)
            
            if valid_metrics['auc'] > best_valid_auc:
                best_valid_auc = valid_metrics['auc']
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
        
        return {
            'best_valid_auc': best_valid_auc,
            'final_epoch': epoch + 1
        }


# ========================================
# 实验主函数
# ========================================

def run_adaptive_decay_experiment(dataset_name='ml-100k', epochs=50, batch_size=256, device=None):
    """
    运行自适应衰减实验
    
    Args:
        dataset_name: 数据集名称
        epochs: 训练轮数
        batch_size: 批次大小
        device: 设备 (None 则自动检测)
    
    Returns:
        list: 实验结果列表
    """
    print("\n" + "=" * 80)
    print("🔬 实验 4.1: 自适应时间衰减")
    print("=" * 80)
    
    DEVICE = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {DEVICE}")
    
    # 加载数据
    print("\n📦 加载数据...")
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=50,
        batch_size=batch_size
    )
    
    # 加载 Top-K 评估数据
    print("📊 加载 Top-K 评估数据...")
    try:
        eval_data, _, fp_eval, ie_eval = get_topk_eval_data('./data', dataset_name, 50)
        print(f"   {len(eval_data)} 个测试用户")
        enable_topk = True
    except Exception as e:
        print(f"   ⚠️ Top-K 数据加载失败: {e}")
        eval_data, fp_eval, ie_eval = None, None, None
        enable_topk = False
    
    # 实验配置
    configs = [
        {'name': 'Fixed-Decay-0.05', 'learnable': False, 'init_decay': 0.05, 'per_user': False},
        {'name': 'Fixed-Decay-0.1', 'learnable': False, 'init_decay': 0.1, 'per_user': False},
        {'name': 'Fixed-Decay-0.2', 'learnable': False, 'init_decay': 0.2, 'per_user': False},
        {'name': 'Learnable-Decay', 'learnable': True, 'init_decay': 0.1, 'per_user': False},
        {'name': 'Per-User-Decay', 'learnable': True, 'init_decay': 0.1, 'per_user': True},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🚀 测试: {config['name']}")
        print("-" * 40)
        
        try:
            model = DINAdaptiveDecay(
                num_items=dataset_info['num_items'],
                num_users=dataset_info['num_users'],
                feature_dims=dataset_info['feature_dims'],
                embedding_dim=64,
                init_decay=config['init_decay'],
                learnable_decay=config['learnable'],
                per_user_decay=config['per_user']
            )
            
            trainer = RichTrainer(
                model=model, 
                device=DEVICE,
                use_tensorboard=True,
                experiment_name=f'exp4_adaptive_{config["name"]}'
            )
            
            t1 = time.time()
            train_result = trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=20,
                early_stopping_patience=5,
                show_progress=True
            )
            train_time = time.time() - t1
            
            test_metrics = trainer.evaluate(test_loader)
            
            # 获取学习到的衰减率
            learned_decay = model.get_decay_rate()
            
            result = {
                'variant': config['name'],
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'best_valid_auc': train_result['best_valid_auc'],
                'train_time_sec': train_time,
                'init_decay': config['init_decay'],
                'learned_decay': learned_decay,
                'status': 'success'
            }
            
            # Top-K 评估
            if enable_topk and eval_data is not None:
                topk_metrics = evaluate_topk_metrics(
                    model, eval_data, fp_eval, ie_eval, 50, DEVICE
                )
                result.update(topk_metrics)
                print(f"✅ AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}")
            else:
                print(f"✅ Test AUC: {test_metrics['auc']:.4f}")
            
            if learned_decay is not None:
                print(f"   学习到的衰减率: {learned_decay:.4f}")
            
            results.append(result)
                
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'variant': config['name'],
                'test_auc': None,
                'test_logloss': None,
                'best_valid_auc': None,
                'train_time_sec': None,
                'init_decay': config['init_decay'],
                'learned_decay': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    # 保存结果
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results_gpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, f'experiment4_adaptive_decay_{dataset_name}.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("📋 自适应衰减实验结果")
    print("=" * 60)
    print(df_results[['variant', 'test_auc', 'learned_decay']].to_string(index=False))
    
    # 返回 list 格式（兼容 run_all_gpu.py）
    return results


def run_contrastive_experiment(dataset_name='ml-100k', epochs=50, batch_size=256, device=None):
    """
    运行对比学习实验
    
    Args:
        dataset_name: 数据集名称
        epochs: 训练轮数
        batch_size: 批次大小
        device: 设备 (None 则自动检测)
    
    Returns:
        list: 实验结果列表
    """
    print("\n" + "=" * 80)
    print("🔬 实验 4.2: 对比学习预训练")
    print("=" * 80)
    
    DEVICE = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {DEVICE}")
    
    # 加载数据
    print("\n📦 加载数据...")
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=50,
        batch_size=batch_size
    )
    
    # 加载 Top-K 评估数据
    print("📊 加载 Top-K 评估数据...")
    try:
        eval_data, _, fp_eval, ie_eval = get_topk_eval_data('./data', dataset_name, 50)
        print(f"   {len(eval_data)} 个测试用户")
        enable_topk = True
    except Exception as e:
        print(f"   ⚠️ Top-K 数据加载失败: {e}")
        eval_data, fp_eval, ie_eval = None, None, None
        enable_topk = False
    
    # 实验配置
    configs = [
        {'name': 'No-Pretrain', 'pretrain_epochs': 0, 'joint': False, 'cl_weight': 0.0},
        {'name': 'Pretrain-5ep', 'pretrain_epochs': 5, 'joint': False, 'cl_weight': 0.0},
        {'name': 'Pretrain-10ep', 'pretrain_epochs': 10, 'joint': False, 'cl_weight': 0.0},
        {'name': 'Joint-0.05', 'pretrain_epochs': 0, 'joint': True, 'cl_weight': 0.05},
        {'name': 'Joint-0.1', 'pretrain_epochs': 0, 'joint': True, 'cl_weight': 0.1},
        {'name': 'Pretrain+Joint', 'pretrain_epochs': 5, 'joint': True, 'cl_weight': 0.05},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🚀 测试: {config['name']}")
        print("-" * 40)
        
        try:
            model = DINWithContrastive(
                num_items=dataset_info['num_items'],
                num_users=dataset_info['num_users'],
                feature_dims=dataset_info['feature_dims'],
                embedding_dim=64,
                contrastive_dim=64,
                temperature=0.1
            )
            
            trainer = ContrastiveTrainer(
                model=model, 
                device=DEVICE,
                contrastive_weight=config['cl_weight'],
                use_tensorboard=True,
                experiment_name=f'exp4_contrastive_{config["name"]}'
            )
            
            t1 = time.time()
            
            # 预训练阶段
            if config['pretrain_epochs'] > 0:
                print(f"预训练 {config['pretrain_epochs']} epochs...")
                trainer.pretrain(train_loader, epochs=config['pretrain_epochs'])
            
            # 微调/联合训练阶段
            if config['joint']:
                train_result = trainer.fit_joint(
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    epochs=20,
                    early_stopping_patience=5,
                    show_progress=True
                )
            else:
                train_result = trainer.fit(
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    epochs=20,
                    early_stopping_patience=5,
                    show_progress=True
                )
            
            train_time = time.time() - t1
            
            test_metrics = trainer.evaluate(test_loader)
            
            result = {
                'variant': config['name'],
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'best_valid_auc': train_result['best_valid_auc'],
                'train_time_sec': train_time,
                'pretrain_epochs': config['pretrain_epochs'],
                'cl_weight': config['cl_weight'],
                'status': 'success'
            }
            
            # Top-K 评估
            if enable_topk and eval_data is not None:
                topk_metrics = evaluate_topk_metrics(
                    model, eval_data, fp_eval, ie_eval, 50, DEVICE
                )
                result.update(topk_metrics)
                print(f"✅ AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}")
            else:
                print(f"✅ Test AUC: {test_metrics['auc']:.4f}")
            
            results.append(result)
                
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'variant': config['name'],
                'test_auc': None,
                'test_logloss': None,
                'best_valid_auc': None,
                'train_time_sec': None,
                'pretrain_epochs': config['pretrain_epochs'],
                'cl_weight': config['cl_weight'],
                'status': f'error: {str(e)[:100]}'
            })
    
    # 保存结果
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results_gpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RESULTS_DIR, f'experiment4_contrastive_{dataset_name}.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("📋 对比学习实验结果")
    print("=" * 60)
    print(df_results[['variant', 'test_auc', 'pretrain_epochs', 'cl_weight']].to_string(index=False))
    
    # 返回 list 格式（兼容 run_all_gpu.py）
    return results


def run_full_experiment(dataset_name='ml-100k'):
    """
    运行完整实验四
    """
    print("=" * 80)
    print("🧪 实验四：DIN 高级改进实验")
    print("=" * 80)
    print("包含：")
    print("  1. 自适应时间衰减（Adaptive Time Decay）")
    print("  2. 对比学习预训练（Contrastive Learning）")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Part 1: 自适应衰减
    df_adaptive = run_adaptive_decay_experiment(dataset_name)
    
    # Part 2: 对比学习
    df_contrastive = run_contrastive_experiment(dataset_name)
    
    # 综合报告
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
    
    report = {
        'experiment': 'Experiment 4: Advanced DIN Improvements',
        'dataset': dataset_name,
        'total_time_seconds': total_time,
        'adaptive_decay_results': df_adaptive.to_dict('records'),
        'contrastive_results': df_contrastive.to_dict('records'),
        'conclusions': {
            'adaptive_decay': '自适应衰减可学习最优衰减参数',
            'contrastive_learning': '对比学习预训练可改善序列表示'
        }
    }
    
    # 找出最佳结果
    df_adaptive_success = df_adaptive[df_adaptive['status'] == 'success']
    df_contrastive_success = df_contrastive[df_contrastive['status'] == 'success']
    
    if len(df_adaptive_success) > 0:
        best_adaptive = df_adaptive_success.loc[df_adaptive_success['test_auc'].idxmax()]
        report['best_adaptive_decay'] = {
            'variant': best_adaptive['variant'],
            'auc': float(best_adaptive['test_auc'])
        }
    
    if len(df_contrastive_success) > 0:
        best_contrastive = df_contrastive_success.loc[df_contrastive_success['test_auc'].idxmax()]
        report['best_contrastive'] = {
            'variant': best_contrastive['variant'],
            'auc': float(best_contrastive['test_auc'])
        }
    
    report_file = os.path.join(RESULTS_DIR, f'experiment4_{dataset_name}_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("🎉 实验四完成!")
    print("=" * 80)
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"报告已保存: {report_file}")
    
    if 'best_adaptive_decay' in report:
        print(f"\n🏆 自适应衰减最佳: {report['best_adaptive_decay']['variant']} "
              f"(AUC={report['best_adaptive_decay']['auc']:.4f})")
    
    if 'best_contrastive' in report:
        print(f"🏆 对比学习最佳: {report['best_contrastive']['variant']} "
              f"(AUC={report['best_contrastive']['auc']:.4f})")
    
    return df_adaptive, df_contrastive, report


# ========================================
# 入口点
# ========================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='实验四：DIN 高级改进')
    parser.add_argument('--dataset', type=str, default='ml-100k', 
                        choices=['ml-100k', 'ml-1m'],
                        help='数据集名称')
    parser.add_argument('--part', type=str, default='all',
                        choices=['all', 'adaptive', 'contrastive'],
                        help='运行哪部分实验')
    
    args = parser.parse_args()
    
    if args.part == 'adaptive':
        run_adaptive_decay_experiment(args.dataset)
    elif args.part == 'contrastive':
        run_contrastive_experiment(args.dataset)
    else:
        run_full_experiment(args.dataset)
