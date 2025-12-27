#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验一：序列长度敏感性分析 + 模型对比

使用丰富特征测试不同序列推荐模型在不同历史长度下的效果。

特征：
- 用户特征：年龄、性别、职业、活跃度
- 物品特征：类型、年份、热度
- 历史序列特征：类型、年份
- 时间特征：小时、星期、周末

对比模型：
- DIN: Deep Interest Network（注意力机制）
- GRU4Rec: 基于 GRU 的序列推荐
- SASRec: 基于 Transformer 的自注意力推荐
- NARM: 神经注意力推荐机器（GRU + 注意力）
- AvgPool: 平均池化基线

输出:
- results/experiment1_results.csv
- results/experiment1_plot.png
- results/experiment1_report.json
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

from data_loader import get_rich_dataloaders
from models import DINRichLite, SimpleAveragePoolingRich, GRU4Rec, SASRec, NARM
from trainer import RichTrainer, measure_inference_speed_rich

# ========================================
# 配置
# ========================================

print("=" * 80)
print("实验一：序列长度敏感性分析 + 模型对比")
print("=" * 80)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {DEVICE}")

# 实验参数
SEQ_LENGTHS = [20, 50, 100, 150]
EPOCHS = 20
BATCH_SIZE = 256
EMBEDDING_DIM = 64

# 模型配置
MODELS_TO_TEST = ['DIN', 'GRU4Rec', 'SASRec', 'NARM', 'AvgPool']

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

results = []
start_time = datetime.now()

print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"序列长度: {SEQ_LENGTHS}")
print(f"模型: {MODELS_TO_TEST}")
print(f"Epochs: {EPOCHS}")
print()

# ========================================
# 主实验循环
# ========================================

for seq_length in SEQ_LENGTHS:
    print("\n" + "=" * 80)
    print(f"🔬 序列长度: {seq_length}")
    print("=" * 80)
    
    # 加载数据（丰富特征版本）
    print(f"\n📦 加载数据 (max_seq_length={seq_length})...")
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name='ml-100k',
        max_seq_length=seq_length,
        batch_size=BATCH_SIZE
    )
    
    for model_name in MODELS_TO_TEST:
        print(f"\n🚀 训练: {model_name}")
        
        try:
            # 创建模型
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
            
            # 统一使用 RichTrainer
            trainer = RichTrainer(model=model, device=DEVICE)
            
            # 训练
            t1 = time.time()
            train_result = trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=EPOCHS,
                early_stopping_patience=5,
                show_progress=False
            )
            train_time = time.time() - t1
            
            # 评估
            test_metrics = trainer.evaluate(test_loader)
            
            # 测量 QPS
            speed = measure_inference_speed_rich(model, test_loader, DEVICE)
            
            results.append({
                'seq_length': seq_length,
                'model': model_name,
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'best_valid_auc': train_result['best_valid_auc'],
                'train_time_sec': train_time,
                'qps': speed['qps'],
                'status': 'success'
            })
            
            print(f"   ✅ AUC: {test_metrics['auc']:.4f}, "
                  f"LogLoss: {test_metrics['logloss']:.4f}, "
                  f"QPS: {speed['qps']:.0f}")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'seq_length': seq_length,
                'model': model_name,
                'test_auc': None,
                'test_logloss': None,
                'best_valid_auc': None,
                'train_time_sec': None,
                'qps': None,
                'status': f'error: {str(e)[:100]}'
            })

# ========================================
# 保存结果
# ========================================

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()

df_results = pd.DataFrame(results)
results_file = os.path.join(RESULTS_DIR, 'experiment1_results.csv')
df_results.to_csv(results_file, index=False)

print("\n" + "=" * 80)
print("🎉 实验完成!")
print("=" * 80)

# ========================================
# 可视化
# ========================================

print("\n📊 生成可视化...")
df_success = df_results[df_results['status'] == 'success'].copy()

if len(df_success) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {
        'DIN': '#FF6B6B', 
        'GRU4Rec': '#4ECDC4', 
        'SASRec': '#45B7D1',
        'NARM': '#96CEB4',
        'AvgPool': '#FFEAA7'
    }
    markers = {'DIN': 'o', 'GRU4Rec': 's', 'SASRec': '^', 'NARM': 'D', 'AvgPool': 'v'}
    
    # AUC vs 序列长度
    for model_name in MODELS_TO_TEST:
        df_model = df_success[df_success['model'] == model_name]
        if len(df_model) > 0:
            axes[0].plot(
                df_model['seq_length'], 
                df_model['test_auc'],
                marker=markers.get(model_name, 'o'),
                color=colors.get(model_name, '#888888'),
                label=model_name,
                linewidth=2,
                markersize=8
            )
    
    axes[0].set_xlabel('序列长度', fontsize=12)
    axes[0].set_ylabel('Test AUC', fontsize=12)
    axes[0].set_title('AUC vs 序列长度', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # LogLoss vs 序列长度
    for model_name in MODELS_TO_TEST:
        df_model = df_success[df_success['model'] == model_name]
        if len(df_model) > 0:
            axes[1].plot(
                df_model['seq_length'], 
                df_model['test_logloss'],
                marker=markers.get(model_name, 'o'),
                color=colors.get(model_name, '#888888'),
                label=model_name,
                linewidth=2,
                markersize=8
            )
    
    axes[1].set_xlabel('序列长度', fontsize=12)
    axes[1].set_ylabel('Test LogLoss', fontsize=12)
    axes[1].set_title('LogLoss vs 序列长度', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 模型对比（序列长度=50）
    df_50 = df_success[df_success['seq_length'] == 50]
    if len(df_50) > 0:
        bar_colors = [colors.get(m, '#888888') for m in df_50['model']]
        bars = axes[2].bar(df_50['model'], df_50['test_auc'], color=bar_colors)
        axes[2].set_ylabel('Test AUC', fontsize=12)
        axes[2].set_title('模型对比 (seq_len=50)', fontsize=14, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, df_50['test_auc']):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_file = os.path.join(RESULTS_DIR, 'experiment1_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {plot_file}")
    plt.close()

# ========================================
# 报告
# ========================================

report = {
    'experiment': 'Experiment 1: Sequence Length Sensitivity + Model Comparison',
    'dataset': 'ml-100k',
    'seq_lengths': SEQ_LENGTHS,
    'models': MODELS_TO_TEST,
    'features_used': [
        'user_age', 'user_gender', 'user_occupation', 'user_activity',
        'item_genre', 'item_year', 'item_popularity',
        'history_genres', 'history_years',
        'time_hour', 'time_dow', 'time_weekend'
    ],
    'total_time_seconds': total_time,
    'results': results
}

# 关键发现
if len(df_success) > 0:
    # 最佳模型
    best_idx = df_success['test_auc'].idxmax()
    report['best_config'] = {
        'model': df_success.loc[best_idx, 'model'],
        'seq_length': int(df_success.loc[best_idx, 'seq_length']),
        'auc': float(df_success.loc[best_idx, 'test_auc'])
    }
    
    # 各模型最佳 AUC
    model_best = df_success.groupby('model')['test_auc'].max().to_dict()
    report['model_best_auc'] = model_best

report_file = os.path.join(RESULTS_DIR, 'experiment1_report.json')
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ========================================
# 打印结果
# ========================================

print("\n" + "=" * 80)
print("📋 实验结果摘要")
print("=" * 80)

# 按序列长度分组显示
for seq_len in SEQ_LENGTHS:
    print(f"\n序列长度 = {seq_len}:")
    df_seq = df_success[df_success['seq_length'] == seq_len]
    for _, row in df_seq.iterrows():
        print(f"  {row['model']}: AUC={row['test_auc']:.4f}, LogLoss={row['test_logloss']:.4f}")

print("\n🔍 关键发现:")
if 'best_config' in report:
    print(f"   最佳配置: {report['best_config']}")
if 'feature_improvement' in report:
    print(f"   丰富特征提升: {report['feature_improvement']}")

print(f"\n📁 结果文件:")
print(f"   - {results_file}")
print(f"   - {plot_file if os.path.exists(os.path.join(RESULTS_DIR, 'experiment1_rich_plot.png')) else 'N/A'}")
print(f"   - {report_file}")
print("=" * 80)
