#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验三（增强版）：DIN 改进消融实验

在丰富特征基础上，测试不同改进策略的效果。

消融变体：
1. DIN-Rich-Base: 基础丰富特征 DIN
2. DIN-Rich-TimeDec: + 时间衰减注意力
3. DIN-Rich-MultiHead: + 多头注意力
4. DIN-Rich-Full: 完整改进
5. DIN-Rich-Full-v2: 完整改进 + 增强 MLP

特征工程：
- 用户画像特征
- 物品属性特征
- 历史序列特征
- 时间上下文特征

输出:
- results/experiment3_rich_results.csv
- results/experiment3_rich_plot.png
- results/experiment3_rich_report.json
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

from data_loader import get_rich_dataloaders
from trainer import RichTrainer, measure_inference_speed_rich
from models import AttentionLayer


# ========================================
# 改进版注意力层（支持丰富特征）
# ========================================

class TimeDecayRichAttention(nn.Module):
    """
    时间衰减 + 丰富特征注意力
    
    近期行为权重更高，符合兴趣漂移规律。
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], time_decay=0.1):
        super(TimeDecayRichAttention, self).__init__()
        
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
        
        # 时间衰减：位置越靠后（越近），权重越大
        positions = torch.arange(seq_len, device=keys.device).float()
        time_weights = torch.exp(self.time_decay * (positions - seq_len + 1))
        attention_scores = attention_scores * time_weights.unsqueeze(0)
        
        if keys_mask is not None:
            attention_scores = attention_scores.masked_fill(~keys_mask.bool(), -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * keys, dim=1)
        
        return weighted_sum, attention_weights


class MultiHeadRichAttention(nn.Module):
    """
    多头注意力 + 丰富特征
    
    捕获用户的多维兴趣。
    """
    
    def __init__(self, input_dim, num_heads=4, hidden_dims=[64, 32]):
        super(MultiHeadRichAttention, self).__init__()
        
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
        all_weights = []  # 收集所有头的注意力权重
        for head in self.attention_heads:
            scores = head(attention_input).squeeze(-1)
            
            if keys_mask is not None:
                scores = scores.masked_fill(~keys_mask.bool(), -1e9)
            
            weights = F.softmax(scores, dim=-1)
            all_weights.append(weights)
            output = torch.sum(weights.unsqueeze(-1) * keys, dim=1)
            head_outputs.append(output)
        
        combined = torch.stack(head_outputs, dim=1).mean(dim=1)
        output = self.output_proj(combined)
        
        # 返回平均注意力权重（用于可视化和hybrid_ranker）
        avg_weights = torch.stack(all_weights, dim=1).mean(dim=1)  # [B, seq_len]
        return output, avg_weights


class TimeDecayMultiHeadRichAttention(nn.Module):
    """
    时间衰减 + 多头注意力（完整改进）
    """
    
    def __init__(self, input_dim, num_heads=4, hidden_dims=[64, 32], time_decay=0.1):
        super(TimeDecayMultiHeadRichAttention, self).__init__()
        
        self.num_heads = num_heads
        self.time_decay = time_decay
        
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
        
        # 时间衰减权重
        positions = torch.arange(seq_len, device=keys.device).float()
        time_weights = torch.exp(self.time_decay * (positions - seq_len + 1))
        
        head_outputs = []
        all_weights = []  # 收集所有头的注意力权重
        for head in self.attention_heads:
            scores = head(attention_input).squeeze(-1)
            scores = scores * time_weights.unsqueeze(0)
            
            if keys_mask is not None:
                scores = scores.masked_fill(~keys_mask.bool(), -1e9)
            
            weights = F.softmax(scores, dim=-1)
            all_weights.append(weights)
            output = torch.sum(weights.unsqueeze(-1) * keys, dim=1)
            head_outputs.append(output)
        
        combined = torch.stack(head_outputs, dim=1).mean(dim=1)
        output = self.output_proj(combined)
        
        # 返回平均注意力权重（用于可视化和hybrid_ranker）
        avg_weights = torch.stack(all_weights, dim=1).mean(dim=1)  # [B, seq_len]
        return output, avg_weights


# ========================================
# 改进版 DIN 模型
# ========================================

class DINRichImproved(nn.Module):
    """
    改进版丰富特征 DIN
    
    支持不同的注意力机制变体。
    """
    
    def __init__(
        self,
        num_items,
        num_users,
        feature_dims,
        embedding_dim=64,
        feature_embedding_dim=16,
        attention_type='base',  # 'base', 'time_decay', 'multi_head', 'full'
        mlp_hidden_dims=[256, 128, 64],
        dropout_rate=0.2,
        num_heads=4,
        time_decay=0.1
    ):
        super(DINRichImproved, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.feature_embedding_dim = feature_embedding_dim
        self.attention_type = attention_type
        
        # 嵌入层
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, feature_embedding_dim)
        self.genre_embedding = nn.Embedding(feature_dims.get('primary_genre', 20) + 1, feature_embedding_dim, padding_idx=0)
        self.year_embedding = nn.Embedding(feature_dims.get('year_bucket', 8) + 1, feature_embedding_dim, padding_idx=0)
        self.age_embedding = nn.Embedding(feature_dims.get('age_bucket', 10) + 1, feature_embedding_dim)
        self.gender_embedding = nn.Embedding(3, feature_embedding_dim)
        self.occupation_embedding = nn.Embedding(feature_dims.get('occupation', 25) + 1, feature_embedding_dim)
        
        # 序列特征维度
        self.seq_feature_dim = embedding_dim + 2 * feature_embedding_dim
        
        # 选择注意力机制
        if attention_type == 'base':
            from models import AttentionLayer
            self.attention = AttentionLayer(self.seq_feature_dim, [64, 32])
        elif attention_type == 'time_decay':
            self.attention = TimeDecayRichAttention(self.seq_feature_dim, [64, 32], time_decay)
        elif attention_type == 'multi_head':
            self.attention = MultiHeadRichAttention(self.seq_feature_dim, num_heads, [64, 32])
        elif attention_type == 'full':
            self.attention = TimeDecayMultiHeadRichAttention(self.seq_feature_dim, num_heads, [64, 32], time_decay)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # MLP
        mlp_input_dim = (
            self.seq_feature_dim +  # 用户兴趣
            self.seq_feature_dim +  # 目标物品
            feature_embedding_dim +  # 用户
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
        
        # 注意力
        user_interest, _ = self.attention(target_combined, seq_combined, batch['item_seq_mask'])
        
        # 用户特征
        user_emb = self.user_embedding(batch['user_id'])
        age_emb = self.age_embedding(batch['user_age'])
        gender_emb = self.gender_embedding(batch['user_gender'])
        occupation_emb = self.occupation_embedding(batch['user_occupation'])
        
        # 拼接
        features = torch.cat([
            user_interest, target_combined,
            user_emb, age_emb, gender_emb, occupation_emb
        ], dim=-1)
        
        return self.mlp(features).squeeze(-1)


# ========================================
# 主实验
# ========================================

def run_experiment(dataset_name='ml-100k'):
    """
    运行消融实验的主函数
    
    Args:
        dataset_name: 数据集名称，'ml-100k' 或 'ml-1m'
    """
    print("=" * 80)
    print("实验三（增强版）：DIN 改进消融实验")
    print("=" * 80)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {DEVICE}")
    
    # 实验参数
    MAX_SEQ_LENGTH = 50
    EPOCHS = 20
    BATCH_SIZE = 256
    EMBEDDING_DIM = 64
    
    # 消融配置
    ABLATION_CONFIGS = [
        {'name': 'DIN-Rich-Base', 'attention_type': 'base', 'description': '丰富特征 + 基础注意力'},
        {'name': 'DIN-Rich-TimeDec', 'attention_type': 'time_decay', 'description': '+ 时间衰减注意力'},
        {'name': 'DIN-Rich-MultiHead', 'attention_type': 'multi_head', 'description': '+ 多头注意力'},
        {'name': 'DIN-Rich-Full', 'attention_type': 'full', 'description': '时间衰减 + 多头注意力'},
    ]
    
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = []
    start_time = datetime.now()
    
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"消融配置数: {len(ABLATION_CONFIGS)}")
    print()
    
    # 加载数据
    print("📦 加载数据...")
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    for config in ABLATION_CONFIGS:
        print("\n" + "=" * 80)
        print(f"🚀 {config['name']}: {config['description']}")
        print("=" * 80)
        
        try:
            model = DINRichImproved(
                num_items=dataset_info['num_items'],
                num_users=dataset_info['num_users'],
                feature_dims=dataset_info['feature_dims'],
                embedding_dim=EMBEDDING_DIM,
                attention_type=config['attention_type'],
                mlp_hidden_dims=[256, 128, 64],
                dropout_rate=0.2
            )
            
            trainer = RichTrainer(model=model, device=DEVICE)
            
            t1 = time.time()
            train_result = trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=EPOCHS,
                early_stopping_patience=5,
                show_progress=True
            )
            train_time = time.time() - t1
            
            test_metrics = trainer.evaluate(test_loader)
            speed = measure_inference_speed_rich(model, test_loader, DEVICE)
            
            results.append({
                'variant': config['name'],
                'description': config['description'],
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'best_valid_auc': train_result['best_valid_auc'],
                'train_time_sec': train_time,
                'qps': speed['qps'],
                'status': 'success'
            })
            
            print(f"\n✅ {config['name']} 完成!")
            print(f"   Test AUC: {test_metrics['auc']:.4f}")
            print(f"   Test LogLoss: {test_metrics['logloss']:.4f}")
            print(f"   QPS: {speed['qps']:.0f}")
            
        except Exception as e:
            print(f"❌ {config['name']} 错误: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'variant': config['name'],
                'description': config['description'],
                'test_auc': None,
                'test_logloss': None,
                'best_valid_auc': None,
                'train_time_sec': None,
                'qps': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    # 完成
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # 保存结果
    df_results = pd.DataFrame(results)
    results_file = os.path.join(RESULTS_DIR, f'experiment3_{dataset_name}_results.csv')
    df_results.to_csv(results_file, index=False)
    
    print("\n" + "=" * 80)
    print("🎉 实验三完成!")
    print("=" * 80)
    
    # 可视化
    print("\n📊 生成可视化...")
    df_success = df_results[df_results['status'] == 'success'].copy()
    
    if len(df_success) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # AUC 对比
        bars = axes[0].bar(
            range(len(df_success)), 
            df_success['test_auc'],
            color=colors[:len(df_success)]
        )
        axes[0].set_xticks(range(len(df_success)))
        axes[0].set_xticklabels(df_success['variant'], rotation=20, ha='right')
        axes[0].set_ylabel('Test AUC', fontsize=12)
        axes[0].set_title('消融实验: AUC 对比', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, df_success['test_auc']):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 改进幅度
        base_auc = df_success[df_success['variant'] == 'DIN-Rich-Base']['test_auc'].values[0]
        improvements = [(auc - base_auc) / base_auc * 100 for auc in df_success['test_auc']]
        
        bars = axes[1].bar(
            range(len(df_success)), 
            improvements,
            color=colors[:len(df_success)]
        )
        axes[1].set_xticks(range(len(df_success)))
        axes[1].set_xticklabels(df_success['variant'], rotation=20, ha='right')
        axes[1].set_ylabel('相对基线提升 (%)', fontsize=12)
        axes[1].set_title('消融实验: 改进幅度', fontsize=14, fontweight='bold')
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, improvements):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.3,
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # QPS 对比
        bars = axes[2].bar(
            range(len(df_success)), 
            df_success['qps'],
            color=colors[:len(df_success)]
        )
        axes[2].set_xticks(range(len(df_success)))
        axes[2].set_xticklabels(df_success['variant'], rotation=20, ha='right')
        axes[2].set_ylabel('QPS', fontsize=12)
        axes[2].set_title('消融实验: 推理速度', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, df_success['qps']):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_file = os.path.join(RESULTS_DIR, f'experiment3_{dataset_name}_plot.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存: {plot_file}")
        plt.close()
    else:
        base_auc = 0
        improvements = []
    
    # 报告
    report = {
        'experiment': 'Experiment 3 (Rich Features): DIN Improvement Ablation Study',
        'dataset': dataset_name,
        'ablation_configs': [c['name'] for c in ABLATION_CONFIGS],
        'features_used': [
            'item_id', 'user_id',
            'history_genres', 'history_years',
            'item_genre', 'item_year',
            'user_age', 'user_gender', 'user_occupation'
        ],
        'total_time_seconds': total_time,
        'results': results
    }
    
    if len(df_success) > 0:
        best_idx = df_success['test_auc'].idxmax()
        report['best_variant'] = df_success.loc[best_idx, 'variant']
        report['best_auc'] = float(df_success.loc[best_idx, 'test_auc'])
        report['baseline_auc'] = float(base_auc)
        report['max_improvement'] = float(max(improvements)) if improvements else 0
    
    report_file = os.path.join(RESULTS_DIR, f'experiment3_{dataset_name}_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("📋 实验结果摘要")
    print("=" * 80)
    print(df_results[['variant', 'test_auc', 'test_logloss', 'qps']].to_string(index=False))
    
    if len(df_success) > 0:
        print("\n🔍 关键发现:")
        print(f"   基线 AUC: {base_auc:.4f}")
        print(f"   最佳变体: {report.get('best_variant', 'N/A')} (AUC={report.get('best_auc', 0):.4f})")
        print(f"   最大提升: {report.get('max_improvement', 0):.2f}%")
        
        for _, row in df_success.iterrows():
            improvement = (row['test_auc'] - base_auc) / base_auc * 100
            print(f"   {row['variant']}: AUC={row['test_auc']:.4f} ({improvement:+.2f}%)")
    
    print("=" * 80)
    
    return df_results, report


# ========================================
# 入口点
# ========================================

if __name__ == '__main__':
    run_experiment('ml-100k')
