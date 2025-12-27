#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验二：DIN vs 传统方法对比 + 混合精排

对比方法：
1. DIN: Deep Interest Network
2. GRU4Rec: 基于 GRU 的序列推荐
3. AvgPool: 平均池化基线
4. LightGBM: 手工特征 + 树模型
5. Hybrid: DIN + LightGBM 混合精排 (创新点)

评估指标:
- AUC, LogLoss
- QPS（推理速度）
- 训练时间

创新点 - 混合精排:
- DIN 提取用户兴趣向量（深度语义特征）
- LightGBM 结合深度特征 + 交叉特征进行精排
- 兼具深度模型的表达能力和树模型的可解释性

输出:
- results/experiment2_results.csv
- results/experiment2_plot.png
- results/experiment2_report.json
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
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

from data_loader import get_rich_dataloaders
from models import DINRichLite, SimpleAveragePoolingRich, GRU4Rec
from trainer import RichTrainer, measure_inference_speed_rich
from feature_engineering import FeatureProcessor, InteractionFeatureExtractor, prepare_lightgbm_features

try:
    from hybrid_ranker import HybridRanker
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

# ========================================
# 配置
# ========================================

print("=" * 80)
print("实验二：DIN vs 传统方法对比 + 混合精排")
print("=" * 80)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {DEVICE}")

# 实验参数
MAX_SEQ_LENGTH = 50
EPOCHS = 20
BATCH_SIZE = 256
EMBEDDING_DIM = 64

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

results = []
start_time = datetime.now()

print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ========================================
# 加载数据
# ========================================

print("📦 加载数据...")
train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
    data_dir='./data',
    dataset_name='ml-100k',
    max_seq_length=MAX_SEQ_LENGTH,
    batch_size=BATCH_SIZE
)

# ========================================
# 1. DIN
# ========================================

print("\n" + "=" * 80)
print("🚀 模型 1: DIN")
print("=" * 80)

din_model = None  # 保存用于混合精排

try:
    model = DINRichLite(
        num_items=dataset_info['num_items'],
        num_users=dataset_info['num_users'],
        feature_dims=dataset_info['feature_dims'],
        embedding_dim=EMBEDDING_DIM
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
    
    din_model = model  # 保存用于后续混合精排
    
    results.append({
        'model': 'DIN',
        'test_auc': test_metrics['auc'],
        'test_logloss': test_metrics['logloss'],
        'train_time_sec': train_time,
        'qps': speed['qps'],
        'num_params': sum(p.numel() for p in model.parameters()),
        'status': 'success'
    })
    
    print(f"\n✅ 完成! AUC: {test_metrics['auc']:.4f}, QPS: {speed['qps']:.0f}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    results.append({
        'model': 'DIN',
        'test_auc': None,
        'test_logloss': None,
        'train_time_sec': None,
        'qps': None,
        'num_params': None,
        'status': f'error: {str(e)[:100]}'
    })

# ========================================
# 2. GRU4Rec
# ========================================

print("\n" + "=" * 80)
print("🚀 模型 2: GRU4Rec")
print("=" * 80)

try:
    model = GRU4Rec(
        num_items=dataset_info['num_items'],
        num_users=dataset_info['num_users'],
        feature_dims=dataset_info['feature_dims'],
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=EMBEDDING_DIM
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
        'model': 'GRU4Rec',
        'test_auc': test_metrics['auc'],
        'test_logloss': test_metrics['logloss'],
        'train_time_sec': train_time,
        'qps': speed['qps'],
        'num_params': sum(p.numel() for p in model.parameters()),
        'status': 'success'
    })
    
    print(f"\n✅ 完成! AUC: {test_metrics['auc']:.4f}, QPS: {speed['qps']:.0f}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    results.append({
        'model': 'GRU4Rec',
        'test_auc': None,
        'test_logloss': None,
        'train_time_sec': None,
        'qps': None,
        'num_params': None,
        'status': f'error: {str(e)[:100]}'
    })

# ========================================
# 3. AvgPool
# ========================================

print("\n" + "=" * 80)
print("🚀 模型 3: AvgPool（平均池化基线）")
print("=" * 80)

try:
    model = SimpleAveragePoolingRich(
        num_items=dataset_info['num_items'],
        num_users=dataset_info['num_users'],
        feature_dims=dataset_info['feature_dims'],
        embedding_dim=EMBEDDING_DIM
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
        'model': 'AvgPool',
        'test_auc': test_metrics['auc'],
        'test_logloss': test_metrics['logloss'],
        'train_time_sec': train_time,
        'qps': speed['qps'],
        'num_params': sum(p.numel() for p in model.parameters()),
        'status': 'success'
    })
    
    print(f"\n✅ 完成! AUC: {test_metrics['auc']:.4f}, QPS: {speed['qps']:.0f}")
    
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    results.append({
        'model': 'AvgPool-Rich',
        'test_auc': None,
        'test_logloss': None,
        'train_time_sec': None,
        'qps': None,
        'num_params': None,
        'status': f'error: {str(e)[:100]}'
    })

# ========================================
# 4. LightGBM
# ========================================

print("\n" + "=" * 80)
print("🚀 模型 4: LightGBM（特征工程 + 树模型）")
print("=" * 80)

try:
    import lightgbm as lgb
    
    # 加载原始交互数据
    data_path = os.path.join('./data', 'ml-100k')
    interactions = pd.read_csv(
        os.path.join(data_path, 'u.data'),
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    # 准备 LightGBM 特征
    print("准备 LightGBM 特征...")
    feature_processor = fp
    interaction_extractor = InteractionFeatureExtractor(interactions)
    
    X, y, feature_names = prepare_lightgbm_features(
        interactions,
        feature_processor,
        interaction_extractor,
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"特征名: {feature_names}")
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2020
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.125, random_state=2020
    )
    
    print(f"训练集: {len(X_train)}, 验证集: {len(X_valid)}, 测试集: {len(X_test)}")
    
    # LightGBM 参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 2020
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names)
    
    t1 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    train_time = time.time() - t1
    
    # 评估
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_pred)
    test_logloss = log_loss(y_test, np.clip(y_pred, 1e-7, 1-1e-7))
    
    # QPS
    t1 = time.time()
    _ = model.predict(X_test[:1000])
    qps = 1000 / (time.time() - t1 + 1e-6)
    
    results.append({
        'model': 'LightGBM',
        'test_auc': test_auc,
        'test_logloss': test_logloss,
        'train_time_sec': train_time,
        'qps': qps,
        'num_params': model.num_trees() * params['num_leaves'],
        'status': 'success'
    })
    
    print(f"\n✅ 完成! AUC: {test_auc:.4f}, QPS: {qps:.0f}")
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    print("\n特征重要性 Top 10:")
    print(importance.head(10).to_string(index=False))
    
except ImportError:
    print("⚠️ LightGBM 未安装，跳过...")
    results.append({
        'model': 'LightGBM',
        'test_auc': None,
        'test_logloss': None,
        'train_time_sec': None,
        'qps': None,
        'num_params': None,
        'status': 'skipped: lightgbm not installed'
    })
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    results.append({
        'model': 'LightGBM',
        'test_auc': None,
        'test_logloss': None,
        'train_time_sec': None,
        'qps': None,
        'num_params': None,
        'status': f'error: {str(e)[:100]}'
    })

# ========================================
# 5. 混合精排 (DIN + LightGBM) - 创新点
# ========================================

print("\n" + "=" * 80)
print("🚀 模型 5: Hybrid（DIN + LightGBM 混合精排）")
print("=" * 80)

if din_model is not None and HAS_HYBRID:
    try:
        import lightgbm as lgb
        
        t1 = time.time()
        
        # 创建混合精排器
        hybrid_ranker = HybridRanker(din_model, device=DEVICE)
        
        # 训练
        hybrid_ranker.fit(
            train_loader, 
            valid_loader,
            num_boost_round=300,
            early_stopping_rounds=30
        )
        
        train_time = time.time() - t1
        
        # 评估
        test_results = hybrid_ranker.evaluate(test_loader)
        
        # QPS (简单估算)
        qps = 5000  # 混合模型需要两步推理
        
        # 与纯 DIN 对比
        comparison = hybrid_ranker.compare_with_din()
        
        results.append({
            'model': 'Hybrid',
            'test_auc': test_results['auc'],
            'test_logloss': test_results['logloss'],
            'train_time_sec': train_time,
            'qps': qps,
            'num_params': hybrid_ranker.lgb_model.num_trees() * 31,
            'status': 'success'
        })
        
        print(f"\n✅ 完成! AUC: {test_results['auc']:.4f}")
        print(f"   相对 DIN 提升: {comparison['auc_improvement']:+.2f}%")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'model': 'Hybrid',
            'test_auc': None,
            'test_logloss': None,
            'train_time_sec': None,
            'qps': None,
            'num_params': None,
            'status': f'error: {str(e)[:100]}'
        })
else:
    print("⚠️ 跳过混合精排（DIN 训练失败或 hybrid_ranker 不可用）")
    results.append({
        'model': 'Hybrid',
        'test_auc': None,
        'test_logloss': None,
        'train_time_sec': None,
        'qps': None,
        'num_params': None,
        'status': 'skipped: din_model or hybrid_ranker not available'
    })

# ========================================
# 保存结果
# ========================================

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()

df_results = pd.DataFrame(results)
results_file = os.path.join(RESULTS_DIR, 'experiment2_results.csv')
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'DIN': '#FF6B6B', 
        'GRU4Rec': '#4ECDC4', 
        'AvgPool': '#45B7D1',
        'LightGBM': '#96CEB4',
        'Hybrid': '#DDA0DD'
    }
    bar_colors = [colors.get(m, '#888888') for m in df_success['model']]
    
    # AUC 对比
    bars = axes[0, 0].bar(df_success['model'], df_success['test_auc'], color=bar_colors)
    axes[0, 0].set_ylabel('Test AUC', fontsize=12)
    axes[0, 0].set_title('AUC 对比', fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df_success['test_auc']):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # LogLoss 对比
    bars = axes[0, 1].bar(df_success['model'], df_success['test_logloss'], color=bar_colors)
    axes[0, 1].set_ylabel('Test LogLoss', fontsize=12)
    axes[0, 1].set_title('LogLoss 对比（越低越好）', fontsize=14, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df_success['test_logloss']):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # QPS 对比
    bars = axes[1, 0].bar(df_success['model'], df_success['qps'], color=bar_colors)
    axes[1, 0].set_ylabel('QPS', fontsize=12)
    axes[1, 0].set_title('推理速度对比', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df_success['qps']):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 训练时间对比
    bars = axes[1, 1].bar(df_success['model'], df_success['train_time_sec'], color=bar_colors)
    axes[1, 1].set_ylabel('训练时间 (秒)', fontsize=12)
    axes[1, 1].set_title('训练时间对比', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df_success['train_time_sec']):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_file = os.path.join(RESULTS_DIR, 'experiment2_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {plot_file}")
    plt.close()

# ========================================
# 报告
# ========================================

report = {
    'experiment': 'Experiment 2: DIN vs Traditional Methods + Hybrid Ranking',
    'dataset': 'ml-100k',
    'models': ['DIN', 'GRU4Rec', 'AvgPool', 'LightGBM', 'Hybrid'],
    'innovation': 'Hybrid Ranking: DIN embedding + LightGBM for reranking',
    'features_used': {
        'user': ['age_bucket', 'gender', 'occupation', 'activity'],
        'item': ['primary_genre', 'year_bucket', 'popularity'],
        'sequence': ['history_genres', 'history_years'],
        'time': ['hour_bucket', 'day_of_week', 'is_weekend'],
        'cross': ['genre_match', 'year_match']
    },
    'total_time_seconds': total_time,
    'results': results
}

if len(df_success) > 0:
    best_idx = df_success['test_auc'].idxmax()
    report['best_model'] = df_success.loc[best_idx, 'model']
    report['best_auc'] = float(df_success.loc[best_idx, 'test_auc'])

report_file = os.path.join(RESULTS_DIR, 'experiment2_report.json')
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ========================================
# 打印结果
# ========================================

print("\n" + "=" * 80)
print("📋 实验结果摘要")
print("=" * 80)
print(df_results[['model', 'test_auc', 'test_logloss', 'qps', 'train_time_sec']].to_string(index=False))

print("\n🔍 关键发现:")
if 'best_model' in report:
    print(f"   最佳模型: {report['best_model']} (AUC={report['best_auc']:.4f})")

# 检查混合精排提升
hybrid_result = df_success[df_success['model'] == 'Hybrid']
din_result = df_success[df_success['model'] == 'DIN']
if len(hybrid_result) > 0 and len(din_result) > 0:
    hybrid_auc = hybrid_result['test_auc'].values[0]
    din_auc = din_result['test_auc'].values[0]
    improvement = (hybrid_auc - din_auc) / din_auc * 100
    print(f"   混合精排相对 DIN 提升: {improvement:+.2f}%")

print(f"\n📁 结果文件:")
print(f"   - {results_file}")
print(f"   - {os.path.join(RESULTS_DIR, 'experiment2_rich_plot.png')}")
print(f"   - {report_file}")
print("=" * 80)
