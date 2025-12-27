#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合精排模块（改进版）

创新点: DIN 深度特征 + LightGBM 精排

架构设计理念:
1. DIN 模型提取用户兴趣向量（注意力加权后的历史表示）
2. 将 DIN embedding（中间层）作为特征，而非最终预测分数
3. LightGBM 处理 embedding + 手工特征 + 交叉特征

关键改进（v2.0）:
- ❌ 不再使用 DIN 的预测分数（避免信息泄露，这是之前的设计缺陷）
- ✅ 使用 PCA 对 DIN embedding 降维，保留关键信息
- ✅ 提取注意力统计特征（熵、top-k权重等）
- ✅ 构建交叉特征增强 LightGBM 表现

适用场景:
- 两阶段排序：DIN粗排 → LightGBM精排
- 需要模型可解释性的场景
- 线上部署：embedding可预计算
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("警告: LightGBM 未安装，混合精排功能不可用")


class DINEmbeddingExtractor(nn.Module):
    """
    DIN 嵌入提取器（改进版）
    
    从训练好的 DIN 模型中提取:
    1. 用户兴趣向量（注意力加权后）
    2. 注意力权重分布（用于计算统计特征）
    
    关键改进: 不再提取 DIN 的预测分数
    """
    
    def __init__(self, din_model, device='cpu'):
        super(DINEmbeddingExtractor, self).__init__()
        self.din_model = din_model
        self.device = device
        self.din_model.to(device)
        self.din_model.eval()
    
    @torch.no_grad()
    def extract_embeddings(self, data_loader):
        """
        从数据加载器中提取所有样本的 DIN 嵌入
        
        Returns:
            embeddings: [N, embedding_dim] 用户兴趣向量
            attention_stats: [N, 4] 注意力统计特征（熵、top1、top5和、方差）
            labels: [N] 真实标签
            extra_features: [N, extra_dim] 额外特征（用户/物品特征）
        """
        all_embeddings = []
        all_attention_stats = []
        all_labels = []
        all_extra_features = []
        
        for batch in tqdm(data_loader, desc="提取 DIN 嵌入"):
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # 提取嵌入、注意力权重和额外特征
            emb, attn_weights, extra = self._extract_single_batch(batch)
            all_embeddings.append(emb.cpu().numpy())
            all_extra_features.append(extra)
            all_labels.append(batch['label'].cpu().numpy())
            
            # 计算注意力统计特征
            if attn_weights is not None:
                attn_stats = self._compute_attention_stats(attn_weights)
                all_attention_stats.append(attn_stats)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        extra_features = np.concatenate(all_extra_features, axis=0)
        
        if all_attention_stats:
            attention_stats = np.concatenate(all_attention_stats, axis=0)
        else:
            attention_stats = None
        
        return embeddings, attention_stats, labels, extra_features
    
    def _compute_attention_stats(self, attn_weights):
        """
        计算注意力权重的统计特征
        
        Args:
            attn_weights: [B, seq_len] 注意力权重
        
        Returns:
            stats: [B, 4] 包含熵、top1、top5和、方差
        """
        attn_np = attn_weights.cpu().numpy()
        batch_size = attn_np.shape[0]
        
        # 1. 注意力熵（衡量集中度，越低越集中）
        entropy = -np.sum(attn_np * np.log(attn_np + 1e-8), axis=1, keepdims=True)
        
        # 2. Top-1 权重（最相关历史的重要性）
        top1 = np.max(attn_np, axis=1, keepdims=True)
        
        # 3. Top-5 权重和（前5个最相关历史的总重要性）
        if attn_np.shape[1] >= 5:
            top5_sum = np.sort(attn_np, axis=1)[:, -5:].sum(axis=1, keepdims=True)
        else:
            top5_sum = np.sum(attn_np, axis=1, keepdims=True)
        
        # 4. 方差（衡量注意力分布的多样性）
        variance = np.var(attn_np, axis=1, keepdims=True)
        
        return np.hstack([entropy, top1, top5_sum, variance])
    
    def _extract_single_batch(self, batch):
        """
        从单个 batch 提取嵌入和注意力权重
        """
        # 获取序列嵌入
        item_seq = batch['item_seq']
        seq_emb = self.din_model.item_embedding(item_seq)  # [B, L, D]
        target_emb = self.din_model.item_embedding(batch['target_item'])  # [B, D]
        
        attention_weights = None
        
        # 如果模型有注意力层，使用注意力加权
        if hasattr(self.din_model, 'attention'):
            user_interest, attention_weights = self.din_model.attention(
                target_emb, seq_emb, batch['item_seq_mask']
            )
        else:
            # 简单平均
            mask = batch['item_seq_mask'].unsqueeze(-1)
            user_interest = (seq_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # 提取额外特征
        extra_features = self._extract_extra_features(batch)
        
        return user_interest, attention_weights, extra_features
    
    def _extract_extra_features(self, batch):
        """提取额外的特征用于 LightGBM"""
        batch_size = batch['user_id'].shape[0]
        
        # 收集所有特征
        features = []
        
        # 用户特征
        for key in ['user_id', 'user_age', 'user_gender', 'user_occupation', 'user_activity']:
            if key in batch:
                features.append(batch[key].cpu().numpy().reshape(-1, 1))
        
        # 物品特征
        for key in ['target_item', 'item_genre', 'item_year', 'item_popularity']:
            if key in batch:
                val = batch.get(key, batch.get('target_item'))
                features.append(val.cpu().numpy().reshape(-1, 1))
        
        # 时间特征
        for key in ['time_hour', 'time_dow', 'time_weekend']:
            if key in batch:
                features.append(batch[key].cpu().numpy().reshape(-1, 1))
        
        # 序列统计特征
        if 'seq_len' in batch:
            features.append(batch['seq_len'].cpu().numpy().reshape(-1, 1))
        
        return np.hstack(features) if features else np.zeros((batch_size, 1))


class HybridRanker:
    """
    混合精排器（改进版 v2.0）
    
    结合 DIN 深度特征和 LightGBM 进行精排。
    
    关键改进:
    1. 不使用 DIN 预测分数（避免信息泄露）
    2. 使用 PCA 对 embedding 降维
    3. 加入注意力统计特征
    4. 构建交叉特征
    """
    
    def __init__(
        self,
        din_model,
        device='cpu',
        lgb_params=None,
        embedding_dim_reduced=16  # PCA 降维后的维度
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
        
        self.din_extractor = DINEmbeddingExtractor(din_model, device)
        self.device = device
        self.lgb_model = None
        self.pca = None
        self.embedding_dim_reduced = embedding_dim_reduced
        
        # LightGBM 改进参数（防止过拟合）
        self.lgb_params = lgb_params or {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 8,
            'learning_rate': 0.02,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': 2020
        }
    
    def fit(
        self,
        train_loader,
        valid_loader,
        num_boost_round=1000,
        early_stopping_rounds=100
    ):
        """
        训练混合精排模型
        """
        print("=" * 70)
        print("训练混合精排模型（改进版 v2.0）")
        print("=" * 70)
        print("✓ 不使用 DIN 预测分数（避免信息泄露）")
        print("✓ 使用 PCA 降维 + 注意力统计特征 + 交叉特征")
        
        # 1. 提取训练集的 DIN 嵌入
        print("\n[1/5] 提取训练集 DIN 嵌入...")
        train_emb, train_attn_stats, train_labels, train_extra = \
            self.din_extractor.extract_embeddings(train_loader)
        
        # 2. 提取验证集的 DIN 嵌入
        print("[2/5] 提取验证集 DIN 嵌入...")
        valid_emb, valid_attn_stats, valid_labels, valid_extra = \
            self.din_extractor.extract_embeddings(valid_loader)
        
        # 3. PCA 降维
        print(f"[3/5] PCA 降维: {train_emb.shape[1]} → {self.embedding_dim_reduced}...")
        self.pca = PCA(n_components=self.embedding_dim_reduced, random_state=2020)
        train_emb_reduced = self.pca.fit_transform(train_emb)
        valid_emb_reduced = self.pca.transform(valid_emb)
        explained_var = sum(self.pca.explained_variance_ratio_)
        print(f"   解释方差比例: {explained_var:.2%}")
        
        # 4. 构建 LightGBM 特征
        print("[4/5] 构建 LightGBM 特征...")
        train_features = self._build_lgb_features(train_emb_reduced, train_attn_stats, train_extra)
        valid_features = self._build_lgb_features(valid_emb_reduced, valid_attn_stats, valid_extra)
        
        print(f"   训练集特征: {train_features.shape}")
        print(f"   验证集特征: {valid_features.shape}")
        
        # 5. 训练 LightGBM
        print("[5/5] 训练 LightGBM...")
        train_data = lgb.Dataset(train_features, label=train_labels)
        valid_data = lgb.Dataset(valid_features, label=valid_labels, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(100)
        ]
        
        self.lgb_model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        print(f"\n✓ 训练完成! 最佳迭代: {self.lgb_model.best_iteration}")
        
        # 保存验证集结果供后续分析
        self.valid_predictions = self.lgb_model.predict(valid_features)
        self.valid_labels = valid_labels
        
        return self
    
    def evaluate(self, data_loader):
        """
        评估混合精排模型
        """
        if self.lgb_model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        # 提取特征
        emb, attn_stats, labels, extra = self.din_extractor.extract_embeddings(data_loader)
        emb_reduced = self.pca.transform(emb)
        features = self._build_lgb_features(emb_reduced, attn_stats, extra)
        
        # 预测
        predictions = self.lgb_model.predict(features)
        
        # 计算指标
        auc = roc_auc_score(labels, predictions)
        logloss = log_loss(labels, predictions)
        
        return {
            'auc': auc,
            'logloss': logloss,
            'predictions': predictions,
            'labels': labels
        }
    
    def _build_lgb_features(self, embeddings, attention_stats, extra_features):
        """
        构建 LightGBM 输入特征（改进版）
        
        特征组成:
        1. PCA 降维后的 DIN embedding
        2. 注意力统计特征（熵、top1、top5和、方差）
        3. 用户/物品/时间手工特征
        4. 交叉特征
        
        注意: 不再使用 DIN 预测分数！
        """
        features_list = [embeddings]
        
        # 注意力统计特征
        if attention_stats is not None:
            features_list.append(attention_stats)
        
        # 手工特征
        features_list.append(extra_features)
        
        # 交叉特征
        cross_features = self._build_cross_features(extra_features)
        if cross_features is not None:
            features_list.append(cross_features)
        
        return np.hstack(features_list)
    
    def _build_cross_features(self, extra_features):
        """
        构建交叉特征
        
        特征交叉能捕捉非线性关系，增强 LightGBM 表现
        """
        if extra_features.shape[1] < 5:
            return None
        
        cross_features = []
        
        # 假设 extra_features 顺序:
        # [user_id, user_age, user_gender, user_occupation, user_activity,
        #  item_id, item_genre, item_year, item_popularity, ...]
        
        try:
            # user_activity × item_popularity（活跃用户是否喜欢热门物品）
            if extra_features.shape[1] > 8:
                activity = extra_features[:, 4:5].astype(float)
                popularity = extra_features[:, 8:9].astype(float)
                cross_features.append(activity * popularity)
            
            # user_age × item_genre（年龄与类型的关系）
            if extra_features.shape[1] > 6:
                age = extra_features[:, 1:2].astype(float)
                genre = extra_features[:, 6:7].astype(float)
                cross_features.append(age * genre)
            
            # user_gender × item_genre（性别与类型的关系）
            if extra_features.shape[1] > 6:
                gender = extra_features[:, 2:3].astype(float)
                genre = extra_features[:, 6:7].astype(float)
                cross_features.append(gender * genre)
                
        except Exception:
            return None
        
        return np.hstack(cross_features) if cross_features else None
    
    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        if self.lgb_model is None:
            raise RuntimeError("模型未训练")
        
        importance = self.lgb_model.feature_importance(importance_type='gain')
        
        # 构建特征名称
        feature_names = []
        
        # PCA embedding 特征
        for i in range(self.embedding_dim_reduced):
            feature_names.append(f'din_emb_pca_{i}')
        
        # 注意力统计特征
        feature_names.extend(['attn_entropy', 'attn_top1', 'attn_top5_sum', 'attn_variance'])
        
        # 手工特征
        feature_names.extend([
            'user_id', 'user_age', 'user_gender', 'user_occupation', 'user_activity',
            'item_id', 'item_genre', 'item_year', 'item_popularity',
            'time_hour', 'time_dow', 'time_weekend', 'seq_len'
        ])
        
        # 交叉特征
        feature_names.extend([
            'cross_activity_popularity',
            'cross_age_genre',
            'cross_gender_genre'
        ])
        
        # 排序
        indices = np.argsort(importance)[::-1][:top_n]
        
        return [(feature_names[i] if i < len(feature_names) else f'feature_{i}', 
                 importance[i]) for i in indices]
    
    def get_validation_metrics(self):
        """
        获取验证集指标（用于与基线对比）
        """
        if self.valid_predictions is None:
            raise RuntimeError("请先调用 fit()")
        
        auc = roc_auc_score(self.valid_labels, self.valid_predictions)
        logloss = log_loss(self.valid_labels, self.valid_predictions)
        
        return {
            'auc': auc,
            'logloss': logloss
        }


def train_hybrid_ranker(
    din_model,
    train_loader,
    valid_loader,
    test_loader,
    device='cpu'
):
    """
    便捷函数：训练混合精排模型并评估
    """
    print("\n" + "=" * 80)
    print("混合精排（改进版 v2.0）: DIN Embedding + LightGBM")
    print("=" * 80)
    
    # 创建混合精排器
    ranker = HybridRanker(din_model, device=device)
    
    # 训练
    ranker.fit(train_loader, valid_loader)
    
    # 验证集指标
    valid_metrics = ranker.get_validation_metrics()
    print(f"\n验证集指标:")
    print(f"  Hybrid AUC: {valid_metrics['auc']:.4f}")
    print(f"  Hybrid LogLoss: {valid_metrics['logloss']:.4f}")
    
    # 测试集评估
    print("\n测试集评估:")
    test_results = ranker.evaluate(test_loader)
    print(f"  Hybrid AUC: {test_results['auc']:.4f}")
    print(f"  Hybrid LogLoss: {test_results['logloss']:.4f}")
    
    # 特征重要性
    print("\n特征重要性 Top 10:")
    for name, imp in ranker.get_feature_importance(10):
        print(f"  {name}: {imp:.2f}")
    
    return ranker, test_results


if __name__ == "__main__":
    print("混合精排模块测试（改进版 v2.0）")
    print("=" * 70)
    print("改进点:")
    print("  ✓ 不使用 DIN 预测分数（避免信息泄露）")
    print("  ✓ 使用 PCA 对 embedding 降维")
    print("  ✓ 提取注意力统计特征（熵、top1、top5和、方差）")
    print("  ✓ 构建交叉特征增强 LightGBM 表现")
    print("=" * 70)
    
    if not HAS_LIGHTGBM:
        print("\n请安装 LightGBM: pip install lightgbm")
    else:
        print("\nLightGBM 已安装，混合精排功能可用")
        print("\n使用示例:")
        print("  from hybrid_ranker import HybridRanker, train_hybrid_ranker")
        print("  ranker = HybridRanker(din_model, device='cuda')")
        print("  ranker.fit(train_loader, valid_loader)")
        print("  results = ranker.evaluate(test_loader)")
