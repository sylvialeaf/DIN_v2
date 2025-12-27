#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版 DIN 模型

支持丰富的特征输入：
- 用户特征嵌入：年龄、性别、职业、活跃度
- 物品特征嵌入：类型、年份、热度
- 历史序列特征：多字段融合
- 时间特征嵌入：小时、星期、周末

参考工业实践：阿里 DIN、华为 DeepFM 等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiFieldEmbedding(nn.Module):
    """
    多字段嵌入层
    
    将多个类别特征映射到同一嵌入空间，然后拼接或求和。
    """
    
    def __init__(self, field_dims, embedding_dim, mode='concat'):
        """
        Args:
            field_dims: dict, 每个字段的类别数 {field_name: num_categories}
            embedding_dim: 每个字段的嵌入维度
            mode: 'concat' 或 'sum'
        """
        super(MultiFieldEmbedding, self).__init__()
        
        self.mode = mode
        self.field_names = list(field_dims.keys())
        self.num_fields = len(field_dims)
        self.embedding_dim = embedding_dim
        
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim + 1, embedding_dim, padding_idx=0)
            for name, dim in field_dims.items()
        })
        
        if mode == 'concat':
            self.output_dim = embedding_dim * self.num_fields
        else:
            self.output_dim = embedding_dim
        
        self._init_weights()
    
    def _init_weights(self):
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0, std=0.01)
    
    def forward(self, field_values):
        """
        Args:
            field_values: dict of {field_name: tensor}
            
        Returns:
            embedded: [batch_size, output_dim]
        """
        embeddings = []
        for name in self.field_names:
            if name in field_values:
                emb = self.embeddings[name](field_values[name])
                embeddings.append(emb)
        
        if self.mode == 'concat':
            return torch.cat(embeddings, dim=-1)
        else:
            return torch.stack(embeddings, dim=0).sum(dim=0)


class RichAttentionLayer(nn.Module):
    """
    增强版注意力层
    
    除了物品嵌入，还考虑物品的其他特征（类型、年份）。
    """
    
    def __init__(self, item_embedding_dim, feature_embedding_dim, hidden_dims=[64, 32]):
        super(RichAttentionLayer, self).__init__()
        
        # 输入：历史物品嵌入 + 特征嵌入，目标物品嵌入 + 特征嵌入
        total_dim = item_embedding_dim + feature_embedding_dim
        input_dim = 4 * total_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.attention_mlp = nn.Sequential(*layers)
    
    def forward(self, query, keys, keys_mask=None):
        """
        Args:
            query: [batch_size, total_dim] - 目标物品的拼接特征
            keys: [batch_size, seq_len, total_dim] - 历史物品的拼接特征
            keys_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, total_dim = keys.shape
        
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        attention_input = torch.cat([
            keys, query_expanded,
            keys * query_expanded,
            keys - query_expanded
        ], dim=-1)
        
        attention_scores = self.attention_mlp(attention_input).squeeze(-1)
        
        if keys_mask is not None:
            # 确保 mask 在同一设备上且类型正确（DataParallel 兼容）
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


class DINRich(nn.Module):
    """
    增强版 DIN 模型
    
    特点：
    1. 多字段用户特征嵌入
    2. 多字段物品特征嵌入
    3. 历史序列多字段融合
    4. 时间特征嵌入
    5. 交叉特征
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        item_embedding_dim=64,
        feature_embedding_dim=16,
        mlp_hidden_dims=[256, 128, 64],
        attention_hidden_dims=[64, 32],
        dropout_rate=0.2
    ):
        super(DINRich, self).__init__()
        
        self.item_embedding_dim = item_embedding_dim
        self.feature_embedding_dim = feature_embedding_dim
        
        # ========================================
        # 嵌入层
        # ========================================
        
        # 物品 ID 嵌入
        self.item_embedding = nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        
        # 用户 ID 嵌入（可选）
        self.user_embedding = nn.Embedding(
            num_users + 1, feature_embedding_dim
        )
        
        # 用户特征嵌入
        self.user_feature_fields = {
            'user_age': feature_dims.get('age_bucket', 10),
            'user_gender': feature_dims.get('gender', 3),
            'user_occupation': feature_dims.get('occupation', 25),
            'user_activity': feature_dims.get('user_activity', 6),
        }
        self.user_field_embedding = MultiFieldEmbedding(
            self.user_feature_fields, feature_embedding_dim, mode='concat'
        )
        
        # 物品特征嵌入
        self.item_feature_fields = {
            'item_genre': feature_dims.get('primary_genre', 20),
            'item_year': feature_dims.get('year_bucket', 8),
            'item_popularity': feature_dims.get('item_popularity', 6),
        }
        self.item_field_embedding = MultiFieldEmbedding(
            self.item_feature_fields, feature_embedding_dim, mode='concat'
        )
        
        # 历史序列特征嵌入
        self.history_genre_embedding = nn.Embedding(
            feature_dims.get('primary_genre', 20) + 1, 
            feature_embedding_dim, 
            padding_idx=0
        )
        self.history_year_embedding = nn.Embedding(
            feature_dims.get('year_bucket', 8) + 1, 
            feature_embedding_dim, 
            padding_idx=0
        )
        
        # 时间特征嵌入
        self.time_fields = {
            'time_hour': feature_dims.get('time_hour', 7),
            'time_dow': feature_dims.get('time_dow', 8),
            'time_weekend': feature_dims.get('time_weekend', 2),
        }
        self.time_field_embedding = MultiFieldEmbedding(
            self.time_fields, feature_embedding_dim, mode='concat'
        )
        
        # ========================================
        # 注意力层
        # ========================================
        
        # 历史序列的总维度：item_emb + genre_emb + year_emb
        self.seq_feature_dim = item_embedding_dim + 2 * feature_embedding_dim
        
        self.attention = RichAttentionLayer(
            item_embedding_dim=item_embedding_dim,
            feature_embedding_dim=2 * feature_embedding_dim,  # genre + year
            hidden_dims=attention_hidden_dims
        )
        
        # ========================================
        # MLP 层
        # ========================================
        
        # 计算 MLP 输入维度
        # 用户兴趣(seq_feature_dim) + 目标物品(seq_feature_dim) + 用户特征 + 物品特征 + 时间特征 + 用户嵌入
        mlp_input_dim = (
            self.seq_feature_dim +  # 用户兴趣
            self.seq_feature_dim +  # 目标物品
            self.user_field_embedding.output_dim +  # 用户特征
            self.item_field_embedding.output_dim +  # 物品特征
            self.time_field_embedding.output_dim +  # 时间特征
            feature_embedding_dim  # 用户 ID 嵌入
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
        """
        前向传播
        
        Args:
            batch: dict containing all features
        """
        # ========================================
        # 1. 嵌入提取
        # ========================================
        
        # 历史序列嵌入
        item_seq = batch['item_seq']
        seq_item_emb = self.item_embedding(item_seq)  # [B, L, item_dim]
        seq_genre_emb = self.history_genre_embedding(batch['history_genres'])  # [B, L, feat_dim]
        seq_year_emb = self.history_year_embedding(batch['history_years'])  # [B, L, feat_dim]
        
        # 拼接历史序列特征
        seq_combined = torch.cat([seq_item_emb, seq_genre_emb, seq_year_emb], dim=-1)  # [B, L, seq_feat_dim]
        
        # 目标物品嵌入
        target_item = batch['target_item']
        target_item_emb = self.item_embedding(target_item)  # [B, item_dim]
        target_genre_emb = self.history_genre_embedding(batch['item_genre'])  # [B, feat_dim]
        target_year_emb = self.history_year_embedding(batch['item_year'])  # [B, feat_dim]
        
        # 拼接目标物品特征
        target_combined = torch.cat([target_item_emb, target_genre_emb, target_year_emb], dim=-1)  # [B, seq_feat_dim]
        
        # ========================================
        # 2. 注意力计算
        # ========================================
        
        user_interest, _ = self.attention(
            target_combined, 
            seq_combined, 
            batch['item_seq_mask']
        )  # [B, seq_feat_dim]
        
        # ========================================
        # 3. 用户特征
        # ========================================
        
        user_id_emb = self.user_embedding(batch['user_id'])  # [B, feat_dim]
        
        user_field_values = {
            'user_age': batch['user_age'],
            'user_gender': batch['user_gender'],
            'user_occupation': batch['user_occupation'],
            'user_activity': batch['user_activity'],
        }
        user_feat_emb = self.user_field_embedding(user_field_values)  # [B, user_feat_dim]
        
        # ========================================
        # 4. 物品特征
        # ========================================
        
        item_field_values = {
            'item_genre': batch['item_genre'],
            'item_year': batch['item_year'],
            'item_popularity': batch['item_popularity'],
        }
        item_feat_emb = self.item_field_embedding(item_field_values)  # [B, item_feat_dim]
        
        # ========================================
        # 5. 时间特征
        # ========================================
        
        time_field_values = {
            'time_hour': batch['time_hour'],
            'time_dow': batch['time_dow'],
            'time_weekend': batch['time_weekend'],
        }
        time_feat_emb = self.time_field_embedding(time_field_values)  # [B, time_feat_dim]
        
        # ========================================
        # 6. 拼接所有特征
        # ========================================
        
        all_features = torch.cat([
            user_interest,      # 用户兴趣（注意力加权）
            target_combined,    # 目标物品
            user_id_emb,        # 用户 ID
            user_feat_emb,      # 用户特征
            item_feat_emb,      # 物品特征
            time_feat_emb,      # 时间特征
        ], dim=-1)
        
        # ========================================
        # 7. MLP 预测
        # ========================================
        
        logits = self.mlp(all_features).squeeze(-1)
        
        return logits
    
    def predict_proba(self, batch):
        """返回点击概率"""
        logits = self.forward(batch)
        return torch.sigmoid(logits)


class DINRichLite(nn.Module):
    """
    轻量版增强 DIN
    
    简化版本，适合快速实验。
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        embedding_dim=64,
        mlp_hidden_dims=[256, 128, 64],
        dropout_rate=0.2
    ):
        super(DINRichLite, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 嵌入层
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim // 4)
        self.genre_embedding = nn.Embedding(feature_dims.get('primary_genre', 20) + 1, embedding_dim // 4, padding_idx=0)
        self.age_embedding = nn.Embedding(feature_dims.get('age_bucket', 10) + 1, embedding_dim // 4)
        self.gender_embedding = nn.Embedding(3, embedding_dim // 4)
        
        # 注意力层（AttentionLayer已在本文件定义，直接使用）
        self.attention = AttentionLayer(embedding_dim, [64, 32])
        
        # MLP
        mlp_input_dim = (
            embedding_dim +     # 用户兴趣
            embedding_dim +     # 目标物品
            embedding_dim // 4 +  # 用户
            embedding_dim // 4 +  # 年龄
            embedding_dim // 4 +  # 性别
            embedding_dim // 4    # 类型
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
        # 序列嵌入
        seq_emb = self.item_embedding(batch['item_seq'])
        target_emb = self.item_embedding(batch['target_item'])
        
        # 注意力
        user_interest, _ = self.attention(target_emb, seq_emb, batch['item_seq_mask'])
        
        # 其他特征
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        genre_emb = self.genre_embedding(batch['item_genre'])
        
        # 拼接
        features = torch.cat([
            user_interest, target_emb,
            user_emb, age_emb, gender_emb, genre_emb
        ], dim=-1)
        
        return self.mlp(features).squeeze(-1)


class SimpleAveragePoolingRich(nn.Module):
    """
    增强版平均池化基线
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        embedding_dim=64,
        mlp_hidden_dims=[256, 128, 64],
        dropout_rate=0.2
    ):
        super(SimpleAveragePoolingRich, self).__init__()
        
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim // 4)
        self.genre_embedding = nn.Embedding(feature_dims.get('primary_genre', 20) + 1, embedding_dim // 4, padding_idx=0)
        self.age_embedding = nn.Embedding(feature_dims.get('age_bucket', 10) + 1, embedding_dim // 4)
        self.gender_embedding = nn.Embedding(3, embedding_dim // 4)
        
        mlp_input_dim = (
            embedding_dim +     # 用户兴趣（平均）
            embedding_dim +     # 目标物品
            embedding_dim // 4 +
            embedding_dim // 4 +
            embedding_dim // 4 +
            embedding_dim // 4
        )
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, batch):
        seq_emb = self.item_embedding(batch['item_seq'])
        target_emb = self.item_embedding(batch['target_item'])
        
        # 平均池化
        mask = batch['item_seq_mask'].unsqueeze(-1)
        user_interest = (seq_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        genre_emb = self.genre_embedding(batch['item_genre'])
        
        features = torch.cat([
            user_interest, target_emb,
            user_emb, age_emb, gender_emb, genre_emb
        ], dim=-1)
        
        return self.mlp(features).squeeze(-1)


# ==============================================================================
# 序列推荐模型: GRU4Rec, SASRec, NARM
# ==============================================================================

class AttentionLayer(nn.Module):
    """
    DIN 注意力层 (基础版，供其他模型引用)
    """
    
    def __init__(self, embedding_dim, hidden_dims=[64, 32]):
        super(AttentionLayer, self).__init__()
        
        input_dim = 4 * embedding_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.attention_mlp = nn.Sequential(*layers)
    
    def forward(self, query, keys, keys_mask=None):
        """
        Args:
            query: [batch_size, embedding_dim]
            keys: [batch_size, seq_len, embedding_dim]
            keys_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, embedding_dim = keys.shape
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        attention_input = torch.cat([
            keys, query_expanded,
            keys * query_expanded,
            keys - query_expanded
        ], dim=-1)
        
        attention_scores = self.attention_mlp(attention_input).squeeze(-1)
        
        if keys_mask is not None:
            # 确保 mask 在同一设备上且类型正确（DataParallel 兼容）
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


class GRU4Rec(nn.Module):
    """
    GRU4Rec: Session-based Recommendations with Recurrent Neural Networks
    
    使用 GRU 对用户历史序列建模，取最后一个隐藏状态作为用户表示。
    
    参考论文: Hidasi et al., 2016
    "Session-based Recommendations with Recurrent Neural Networks"
    
    特点:
    1. 使用 GRU 捕获序列中的时序依赖
    2. 最后隐藏状态表示用户当前兴趣
    3. 与目标物品计算点积得分
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        mlp_hidden_dims=[256, 128, 64],
        dropout_rate=0.2
    ):
        super(GRU4Rec, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 嵌入层
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim // 4)
        self.genre_embedding = nn.Embedding(feature_dims.get('primary_genre', 20) + 1, embedding_dim // 4, padding_idx=0)
        self.age_embedding = nn.Embedding(feature_dims.get('age_bucket', 10) + 1, embedding_dim // 4)
        self.gender_embedding = nn.Embedding(3, embedding_dim // 4)
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # MLP
        mlp_input_dim = (
            hidden_dim +        # GRU 输出
            embedding_dim +     # 目标物品
            embedding_dim // 4 +
            embedding_dim // 4 +
            embedding_dim // 4 +
            embedding_dim // 4
        )
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden))
            mlp_layers.append(nn.BatchNorm1d(hidden))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        item_seq = batch['item_seq']  # [B, L]
        seq_len = batch['seq_len']  # [B]
        batch_size = item_seq.shape[0]
        
        # 序列嵌入
        seq_emb = self.item_embedding(item_seq)  # [B, L, D]
        
        # 直接用 GRU 处理（不用 pack_padded_sequence，避免 DataParallel 问题）
        gru_output, _ = self.gru(seq_emb)  # [B, L, H]
        
        # 根据实际序列长度取最后有效位置的输出
        # seq_len: [B], last_idx = seq_len - 1
        last_idx = (seq_len - 1).clamp(min=0).long()  # [B]
        batch_idx = torch.arange(batch_size, device=item_seq.device)  # [B]
        user_interest = gru_output[batch_idx, last_idx]  # [B, H]
        
        # 目标物品
        target_emb = self.item_embedding(batch['target_item'])
        
        # 其他特征
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        genre_emb = self.genre_embedding(batch['item_genre'])
        
        # 拼接
        features = torch.cat([
            user_interest, target_emb,
            user_emb, age_emb, gender_emb, genre_emb
        ], dim=-1)
        
        return self.mlp(features).squeeze(-1)


class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation
    
    使用自定义的 Multi-Head Attention 实现（参考 RecBole）
    避免 PyTorch TransformerEncoderLayer 的 mask 兼容性问题。
    
    参考论文: Kang & McAuley, 2018
    "Self-Attentive Sequential Recommendation"
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        embedding_dim=64,
        num_heads=2,
        num_layers=2,
        max_seq_len=100,
        mlp_hidden_dims=[256, 128, 64],
        dropout_rate=0.2
    ):
        super(SASRec, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 嵌入层
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, embedding_dim)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim // 4)
        self.genre_embedding = nn.Embedding(feature_dims.get('primary_genre', 20) + 1, embedding_dim // 4, padding_idx=0)
        self.age_embedding = nn.Embedding(feature_dims.get('age_bucket', 10) + 1, embedding_dim // 4)
        self.gender_embedding = nn.Embedding(3, embedding_dim // 4)
        
        # 自定义 Transformer 层（参考 RecBole）
        self.attention_layers = nn.ModuleList([
            SASRecAttentionLayer(embedding_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # MLP
        mlp_input_dim = (
            embedding_dim +     # Transformer 输出
            embedding_dim +     # 目标物品
            embedding_dim // 4 +
            embedding_dim // 4 +
            embedding_dim // 4 +
            embedding_dim // 4
        )
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden))
            mlp_layers.append(nn.BatchNorm1d(hidden))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_attention_mask(self, item_seq):
        """生成因果注意力掩码（参考 RecBole）"""
        # item_seq: [B, L]
        attention_mask = (item_seq != 0)  # [B, L] padding mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        
        # 因果掩码：只能看到之前的位置
        seq_len = item_seq.size(1)
        tril_mask = torch.tril(torch.ones((seq_len, seq_len), device=item_seq.device))  # [L, L]
        extended_attention_mask = extended_attention_mask * tril_mask.unsqueeze(0).unsqueeze(0)  # [B, 1, L, L]
        
        # 转换为加法掩码：True -> 0, False -> -inf（使用 masked_fill 避免重复创建 tensor）
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = extended_attention_mask.masked_fill(
            extended_attention_mask == 0, float('-inf')
        ).masked_fill(extended_attention_mask == 1, 0.0)
        
        return extended_attention_mask  # [B, 1, L, L]
    
    def forward(self, batch):
        item_seq = batch['item_seq']  # [B, L]
        batch_size, seq_len = item_seq.shape
        
        # 位置嵌入
        positions = torch.arange(1, seq_len + 1, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        positions = positions.clamp(max=self.max_seq_len)
        
        # 序列嵌入
        seq_emb = self.item_embedding(item_seq)  # [B, L, D]
        pos_emb = self.position_embedding(positions)  # [B, L, D]
        
        hidden = self.layer_norm(seq_emb + pos_emb)
        hidden = self.emb_dropout(hidden)
        
        # 注意力掩码
        attention_mask = self.get_attention_mask(item_seq)  # [B, 1, L, L]
        
        # Transformer 编码
        for layer in self.attention_layers:
            hidden = layer(hidden, attention_mask)
        
        # 取最后一个有效位置的输出
        seq_len_tensor = batch['seq_len']  # [B]
        last_idx = (seq_len_tensor - 1).clamp(min=0).long()  # [B]
        batch_idx = torch.arange(batch_size, device=item_seq.device)
        user_interest = hidden[batch_idx, last_idx]  # [B, D]
        
        # 目标物品
        target_emb = self.item_embedding(batch['target_item'])
        
        # 其他特征
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        genre_emb = self.genre_embedding(batch['item_genre'])
        
        # 拼接
        features = torch.cat([
            user_interest, target_emb,
            user_emb, age_emb, gender_emb, genre_emb
        ], dim=-1)
        
        return self.mlp(features).squeeze(-1)


class SASRecAttentionLayer(nn.Module):
    """SASRec 的自注意力层（参考 RecBole 实现）"""
    
    def __init__(self, hidden_size, num_heads, dropout_rate=0.2):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** 0.5
        
        # Q, K, V 投影
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
    
    def transpose_for_scores(self, x):
        # x: [B, L, all_head_size] -> [B, num_heads, L, head_dim]
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask):
        # hidden_states: [B, L, D]
        # attention_mask: [B, 1, L, L] 加法掩码
        
        # Self-Attention
        query_layer = self.transpose_for_scores(self.query(hidden_states))  # [B, H, L, d]
        key_layer = self.transpose_for_scores(self.key(hidden_states))  # [B, H, L, d]
        value_layer = self.transpose_for_scores(self.value(hidden_states))  # [B, H, L, d]
        
        # 注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B, H, L, L]
        attention_scores = attention_scores / self.scale
        
        # 应用掩码（加法）
        attention_scores = attention_scores + attention_mask
        
        # Softmax + Dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # 加权求和
        context_layer = torch.matmul(attention_probs, value_layer)  # [B, H, L, d]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B, L, H, d]
        context_layer = context_layer.view(hidden_states.size(0), -1, self.all_head_size)  # [B, L, D]
        
        # 输出投影 + 残差连接
        hidden_states = self.layer_norm1(hidden_states + self.out_dropout(self.dense(context_layer)))
        
        # Feed-Forward + 残差连接
        hidden_states = self.layer_norm2(hidden_states + self.ffn(hidden_states))
        
        return hidden_states


class NARM(nn.Module):
    """
    NARM: Neural Attentive Recommendation Machine
    
    结合 GRU 和注意力机制，学习序列中的全局和局部偏好。
    
    参考论文: Li et al., 2017
    "Neural Attentive Session-based Recommendation"
    
    特点:
    1. 全局编码器: GRU 捕获序列的整体表示
    2. 局部编码器: 注意力机制捕获与当前兴趣相关的历史物品
    3. 混合表示: 结合全局和局部信息
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        mlp_hidden_dims=[256, 128, 64],
        dropout_rate=0.2
    ):
        super(NARM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 嵌入层
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim // 4)
        self.genre_embedding = nn.Embedding(feature_dims.get('primary_genre', 20) + 1, embedding_dim // 4, padding_idx=0)
        self.age_embedding = nn.Embedding(feature_dims.get('age_bucket', 10) + 1, embedding_dim // 4)
        self.gender_embedding = nn.Embedding(3, embedding_dim // 4)
        
        # 全局编码器 (GRU)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 局部注意力
        self.attention_W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_v = nn.Linear(hidden_dim, 1, bias=False)
        
        # 混合层
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # MLP
        mlp_input_dim = (
            hidden_dim +        # 混合表示
            embedding_dim +     # 目标物品
            embedding_dim // 4 +
            embedding_dim // 4 +
            embedding_dim // 4 +
            embedding_dim // 4
        )
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden))
            mlp_layers.append(nn.BatchNorm1d(hidden))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        item_seq = batch['item_seq']  # [B, L]
        seq_len = batch['seq_len']  # [B]
        seq_mask = batch['item_seq_mask']  # [B, L]
        batch_size = item_seq.shape[0]
        
        # 序列嵌入
        seq_emb = self.item_embedding(item_seq)  # [B, L, D]
        
        # 直接用 GRU 处理（不用 pack_padded_sequence，避免 DataParallel 问题）
        gru_output, _ = self.gru(seq_emb)  # [B, L, H]
        
        # 全局表示: 取最后有效位置的输出
        last_idx = (seq_len - 1).clamp(min=0).long()  # [B]
        batch_idx = torch.arange(batch_size, device=item_seq.device)  # [B]
        h_global = gru_output[batch_idx, last_idx]  # [B, H]
        
        # 局部注意力
        # attention = softmax(v^T * σ(W1 * h_t + W2 * h_n))
        h_t = self.attention_W1(gru_output)  # [B, L, H]
        h_n = self.attention_W2(h_global).unsqueeze(1)  # [B, 1, H]
        
        attention_scores = self.attention_v(torch.sigmoid(h_t + h_n)).squeeze(-1)  # [B, L]
        # 使用 float('-inf') 避免 DataParallel 数值问题
        attention_scores = attention_scores.masked_fill(~seq_mask.bool(), float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, L]
        
        # 处理可能的 NaN（全 padding 情况）
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        
        # 局部表示
        h_local = torch.sum(attention_weights.unsqueeze(-1) * gru_output, dim=1)  # [B, H]
        
        # 混合表示
        user_interest = self.combine(torch.cat([h_global, h_local], dim=-1))  # [B, H]
        
        # 目标物品
        target_emb = self.item_embedding(batch['target_item'])
        
        # 其他特征
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        genre_emb = self.genre_embedding(batch['item_genre'])
        
        # 拼接
        features = torch.cat([
            user_interest, target_emb,
            user_emb, age_emb, gender_emb, genre_emb
        ], dim=-1)
        
        return self.mlp(features).squeeze(-1)


if __name__ == "__main__":
    print("测试序列推荐模型...")
    
    batch_size = 32
    seq_len = 50
    
    # 模拟数据
    batch = {
        'user_id': torch.randint(1, 100, (batch_size,)),
        'item_seq': torch.randint(1, 1000, (batch_size, seq_len)),
        'item_seq_mask': torch.ones(batch_size, seq_len),
        'seq_len': torch.randint(10, seq_len, (batch_size,)),
        'target_item': torch.randint(1, 1000, (batch_size,)),
        'user_age': torch.randint(1, 8, (batch_size,)),
        'user_gender': torch.randint(1, 3, (batch_size,)),
        'user_occupation': torch.randint(1, 22, (batch_size,)),
        'user_activity': torch.randint(1, 6, (batch_size,)),
        'item_genre': torch.randint(1, 20, (batch_size,)),
        'item_year': torch.randint(1, 7, (batch_size,)),
        'item_popularity': torch.randint(1, 6, (batch_size,)),
        'history_genres': torch.randint(0, 20, (batch_size, seq_len)),
        'history_years': torch.randint(0, 7, (batch_size, seq_len)),
        'time_hour': torch.randint(1, 7, (batch_size,)),
        'time_dow': torch.randint(1, 8, (batch_size,)),
        'time_weekend': torch.randint(0, 2, (batch_size,)),
    }
    
    feature_dims = {
        'age_bucket': 8,
        'gender': 3,
        'occupation': 22,
        'user_activity': 6,
        'primary_genre': 20,
        'year_bucket': 7,
        'item_popularity': 6,
        'time_hour': 7,
        'time_dow': 8,
        'time_weekend': 2,
    }
    
    models_to_test = [
        ('DINRich', DINRich(
            num_items=1000, num_users=100, feature_dims=feature_dims,
            item_embedding_dim=64, feature_embedding_dim=16
        )),
        ('GRU4Rec', GRU4Rec(
            num_items=1000, num_users=100, feature_dims=feature_dims,
            embedding_dim=64, hidden_dim=64
        )),
        ('SASRec', SASRec(
            num_items=1000, num_users=100, feature_dims=feature_dims,
            embedding_dim=64, num_heads=2, num_layers=2
        )),
        ('NARM', NARM(
            num_items=1000, num_users=100, feature_dims=feature_dims,
            embedding_dim=64, hidden_dim=64
        )),
        ('AvgPool', SimpleAveragePoolingRich(
            num_items=1000, num_users=100, feature_dims=feature_dims,
            embedding_dim=64
        )),
    ]
    
    print("\n" + "=" * 60)
    print("模型测试结果:")
    print("=" * 60)
    
    for name, model in models_to_test:
        try:
            logits = model(batch)
            params = sum(p.numel() for p in model.parameters())
            print(f"✅ {name:15} | 输出: {logits.shape} | 参数量: {params:,}")
        except Exception as e:
            print(f"❌ {name:15} | 错误: {e}")
    
    print("=" * 60)
