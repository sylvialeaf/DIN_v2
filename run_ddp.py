#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDP (DistributedDataParallel) 云端 GPU 实验脚本

支持自动启动 DDP，无需手动使用 torchrun！

使用方法（直接运行，会自动启动 DDP）:
    python run_ddp.py                        # 运行所有实验（两个数据集）
    python run_ddp.py --dataset ml-100k      # 只运行 ml-100k
    python run_ddp.py --dataset ml-1m        # 只运行 ml-1m
    python run_ddp.py --quick                # 快速测试模式
    python run_ddp.py --exp1                 # 只运行实验1（快捷方式）
    python run_ddp.py --exp 1                # 只运行实验1（完整参数）
    python run_ddp.py --exp 1,2,3            # 运行实验1-3
    python run_ddp.py --exp2 --dataset ml-100k  # 实验2 + ml-100k
    
DDP vs DataParallel:
    - DDP 快 30-50%
    - GPU 利用率更高 (70-90% vs 15-30%)
    - 完全避免 pack_padded_sequence 问题

输出和原版 run_all_gpu.py 完全一致：
    - 结果保存到 results_gpu/
    - TensorBoard 日志保存到 /root/tf-logs（AutoDL）或 ./runs
"""

import os
import sys
import subprocess

# ========================================
# 自动启动 DDP 的入口点
# ========================================

def is_launched_by_torchrun():
    """检查是否已经通过 torchrun 启动"""
    return 'LOCAL_RANK' in os.environ


def auto_launch_ddp():
    """自动使用 torchrun 启动 DDP"""
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，将使用单 CPU 模式")
        return False
    
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        print(f"ℹ️ 检测到 {num_gpus} 个 GPU，将使用单 GPU 模式")
        return False
    
    print(f"🚀 检测到 {num_gpus} 个 GPU，自动启动 DDP...")
    print("=" * 60)
    
    # 构建 torchrun 命令
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        f'--nproc_per_node={num_gpus}',
        '--master_port=29500',
        script_path
    ] + sys.argv[1:]  # 传递原始参数
    
    print(f"执行: {' '.join(cmd)}")
    print("=" * 60)
    
    # 执行并等待完成
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# 如果不是 torchrun 启动，则自动启动
if not is_launched_by_torchrun():
    auto_launch_ddp()


# ========================================
# 以下是 DDP Worker 的主代码
# ========================================

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
from tqdm import tqdm
import platform
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import RichFeatureDataset, get_topk_eval_data, build_topk_batch_multi
from models import DINRichLite, SimpleAveragePoolingRich, GRU4Rec, SASRec, NARM, AttentionLayer
from trainer import RichTrainer, measure_inference_speed_rich
from feature_engineering import FeatureProcessor, InteractionFeatureExtractor

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ========================================
# Top-K 评估指标函数
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
    """Precision @ K (单 ground truth 场景，命中则为 1/k)"""
    hits = 1 if ground_truth in ranked_items[:k] else 0
    return hits / k


# ========================================
# DDP 工具函数
# ========================================

def setup_ddp():
    """初始化 DDP 环境"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    
    return local_rank, world_size, rank


def cleanup_ddp():
    """清理 DDP 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """判断是否是主进程（rank 0 负责输出和保存）"""
    return rank == 0


def print_main(msg, rank):
    """仅主进程打印"""
    if is_main_process(rank):
        print(msg)


def barrier():
    """同步所有进程"""
    if dist.is_initialized():
        dist.barrier()


# ========================================
# 配置解析
# ========================================

parser = argparse.ArgumentParser(description='DDP 云端 GPU 完整实验')
parser.add_argument('--dataset', type=str, default='both', 
                    choices=['ml-100k', 'ml-1m', 'both'])
parser.add_argument('--quick', action='store_true', help='快速测试模式')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp', type=str, default='all', help='实验编号: 1, 2, 3, 4, 1,2,3, all')
parser.add_argument('--no-topk', action='store_true', help='禁用 Top-K 评估')
parser.add_argument('--topk-sample', type=str, default='auto')
parser.add_argument('--exp4-part', type=str, default='all', 
                    choices=['all', 'adaptive', 'contrastive'])
# 快捷参数
parser.add_argument('--exp1', action='store_true', help='快捷方式: 仅运行实验1')
parser.add_argument('--exp2', action='store_true', help='快捷方式: 仅运行实验2')
parser.add_argument('--exp3', action='store_true', help='快捷方式: 仅运行实验3')
parser.add_argument('--exp4', action='store_true', help='快捷方式: 仅运行实验4')
args = parser.parse_args()

# 处理快捷参数
if args.exp1:
    args.exp = '1'
elif args.exp2:
    args.exp = '2'
elif args.exp3:
    args.exp = '3'
elif args.exp4:
    args.exp = '4'


# ========================================
# 主函数
# ========================================

def main():
    # 初始化 DDP
    local_rank, world_size, rank = setup_ddp()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    # 解析实验
    if args.exp == 'all':
        EXPERIMENTS_TO_RUN = [1, 2, 3, 4]
    else:
        EXPERIMENTS_TO_RUN = [int(x.strip()) for x in args.exp.split(',')]
    
    ENABLE_TOPK = not args.no_topk
    
    # 配置参数
    if args.quick:
        EPOCHS = 10
        SEQ_LENGTHS = [20, 50]
        BATCH_SIZE_PER_GPU = 1024
    else:
        EPOCHS = args.epochs
        SEQ_LENGTHS = [20, 50, 100, 150]
        BATCH_SIZE_PER_GPU = 2048  # DDP: 每个 GPU 的 batch size
    
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE_PER_GPU * world_size
    EMBEDDING_DIM = 64
    NUM_WORKERS = 4  # 每个进程的 workers
    PREFETCH_FACTOR = 4
    TOPK_VALUES = [5, 10, 20]
    NUM_NEG_SAMPLES = 99
    MODELS_TO_TEST = ['DIN', 'GRU4Rec', 'SASRec', 'NARM', 'AvgPool']
    
    # TensorBoard 目录
    if platform.system() == 'Linux' and os.path.exists('/root'):
        TENSORBOARD_LOG_DIR = '/root/tf-logs'
    else:
        TENSORBOARD_LOG_DIR = './runs'
    
    # 数据集
    if args.dataset == 'both':
        DATASETS = ['ml-100k', 'ml-1m']
    else:
        DATASETS = [args.dataset]
    
    # 结果目录
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_gpu')
    if is_main_process(rank):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 打印配置（仅主进程）
    print_main("=" * 80, rank)
    print_main("🚀 DDP 分布式训练", rank)
    print_main("=" * 80, rank)
    print_main(f"World Size: {world_size} GPUs", rank)
    if torch.cuda.is_available():
        for i in range(world_size):
            if rank == i:
                print(f"[Rank {rank}] GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
            barrier()
    print_main(f"数据集: {DATASETS}", rank)
    print_main(f"实验: {EXPERIMENTS_TO_RUN}", rank)
    print_main(f"Epochs: {EPOCHS}", rank)
    print_main(f"序列长度: {SEQ_LENGTHS}", rank)
    print_main(f"Batch Size: {BATCH_SIZE_PER_GPU} × {world_size} = {EFFECTIVE_BATCH_SIZE}", rank)
    print_main(f"Workers per GPU: {NUM_WORKERS}", rank)
    print_main(f"模型: {MODELS_TO_TEST}", rank)
    print_main(f"Top-K 评估: {'启用' if ENABLE_TOPK else '禁用'}", rank)
    print_main(f"TensorBoard: {TENSORBOARD_LOG_DIR}", rank)
    print_main("=" * 80, rank)
    
    # 开始实验
    experiment_start = datetime.now()
    all_results = []
    
    for dataset_name in DATASETS:
        print_main(f"\n{'='*80}", rank)
        print_main(f"📊 数据集: {dataset_name}", rank)
        print_main(f"{'='*80}", rank)
        
        # 实验1: 序列长度 + 模型对比
        if 1 in EXPERIMENTS_TO_RUN:
            results1 = run_experiment1(
                dataset_name, device, local_rank, world_size, rank,
                EPOCHS, SEQ_LENGTHS, BATCH_SIZE_PER_GPU, EMBEDDING_DIM,
                NUM_WORKERS, PREFETCH_FACTOR, MODELS_TO_TEST,
                ENABLE_TOPK, TOPK_VALUES, NUM_NEG_SAMPLES,
                TENSORBOARD_LOG_DIR
            )
            all_results.extend(results1)
        
        # 实验2: 方法对比（混合精排）
        if 2 in EXPERIMENTS_TO_RUN:
            results2 = run_experiment2(
                dataset_name, device, local_rank, world_size, rank,
                EPOCHS, BATCH_SIZE_PER_GPU, EMBEDDING_DIM,
                NUM_WORKERS, PREFETCH_FACTOR,
                TENSORBOARD_LOG_DIR,
                ENABLE_TOPK, TOPK_VALUES
            )
            all_results.extend(results2)
        
        # 实验3: 消融实验
        if 3 in EXPERIMENTS_TO_RUN:
            results3 = run_experiment3(
                dataset_name, device, local_rank, world_size, rank,
                EPOCHS, BATCH_SIZE_PER_GPU, EMBEDDING_DIM,
                NUM_WORKERS, PREFETCH_FACTOR,
                TENSORBOARD_LOG_DIR,
                ENABLE_TOPK, TOPK_VALUES
            )
            all_results.extend(results3)
        
        # 实验4: 高级改进
        if 4 in EXPERIMENTS_TO_RUN:
            results4 = run_experiment4(
                dataset_name, device, local_rank, world_size, rank,
                EPOCHS, BATCH_SIZE_PER_GPU, EMBEDDING_DIM,
                NUM_WORKERS, PREFETCH_FACTOR,
                TENSORBOARD_LOG_DIR, args.exp4_part
            )
            all_results.extend(results4)
    
    # 保存结果（仅主进程）
    if is_main_process(rank) and all_results:
        experiment_end = datetime.now()
        total_time = (experiment_end - experiment_start).total_seconds()
        
        df_results = pd.DataFrame(all_results)
        timestamp = experiment_start.strftime('%Y%m%d_%H%M%S')
        
        # CSV
        csv_file = os.path.join(RESULTS_DIR, f'ddp_results_{timestamp}.csv')
        df_results.to_csv(csv_file, index=False)
        
        # JSON 报告
        report = {
            'timestamp': timestamp,
            'mode': 'DDP',
            'world_size': world_size,
            'device': device,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'datasets': DATASETS,
            'experiments': EXPERIMENTS_TO_RUN,
            'epochs': EPOCHS,
            'seq_lengths': SEQ_LENGTHS,
            'batch_size_per_gpu': BATCH_SIZE_PER_GPU,
            'effective_batch_size': EFFECTIVE_BATCH_SIZE,
            'models': MODELS_TO_TEST,
            'topk_values': TOPK_VALUES,
            'total_time_minutes': total_time / 60,
            'num_results': len(all_results),
            'results': all_results
        }
        
        json_file = os.path.join(RESULTS_DIR, f'ddp_report_{timestamp}.json')
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
        
        # 最佳结果
        df_success = df_results[df_results['status'] == 'success']
        print("\n📊 各实验最佳结果:")
        for exp_name in df_success['experiment'].unique():
            df_exp = df_success[df_success['experiment'] == exp_name]
            if len(df_exp) > 0 and 'test_auc' in df_exp.columns:
                best = df_exp.loc[df_exp['test_auc'].idxmax()]
                model_col = 'model' if 'model' in best else 'variant'
                print(f"  {exp_name}: {best.get(model_col, 'N/A')} - AUC={best['test_auc']:.4f}")
        
        print("=" * 80)
        print("✅ DDP 训练完成！")
    
    # 清理
    cleanup_ddp()


# ========================================
# DDP 数据下载辅助函数
# ========================================

def _ensure_data_downloaded(dataset_name, rank, world_size):
    """
    DDP 安全的数据下载：只有 rank 0 下载，其他进程等待
    """
    import urllib.request
    import zipfile
    
    data_dir = './data'
    data_path = os.path.join(data_dir, dataset_name)
    
    # 确定关键文件
    if dataset_name == 'ml-100k':
        key_file = os.path.join(data_path, 'u.data')
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    elif dataset_name == 'ml-1m':
        key_file = os.path.join(data_path, 'ratings.dat')
        url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # Rank 0 检查并下载
    if is_main_process(rank):
        if os.path.exists(key_file):
            print(f"✅ 数据已存在: {data_path}")
        else:
            print(f"📥 Rank 0 开始下载数据集 {dataset_name}...")
            os.makedirs(data_dir, exist_ok=True)
            
            zip_path = os.path.join(data_dir, f'{dataset_name}.zip')
            urllib.request.urlretrieve(url, zip_path)
            print(f"📦 解压数据...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            os.remove(zip_path)
            print(f"✅ 数据准备完成: {data_path}")
    
    # 所有进程同步等待
    barrier()
    print_main(f"🔄 所有进程已同步，数据可用: {dataset_name}", rank)


# ========================================
# 实验1: 序列长度 + 模型对比
# ========================================

def run_experiment1(dataset_name, device, local_rank, world_size, rank,
                    EPOCHS, SEQ_LENGTHS, BATCH_SIZE_PER_GPU, EMBEDDING_DIM,
                    NUM_WORKERS, PREFETCH_FACTOR, MODELS_TO_TEST,
                    ENABLE_TOPK, TOPK_VALUES, NUM_NEG_SAMPLES,
                    TENSORBOARD_LOG_DIR):
    """实验1: 序列长度 + 模型对比"""
    
    print_main("\n" + "=" * 60, rank)
    print_main("📊 实验1: 序列长度 + 模型对比", rank)
    print_main("=" * 60, rank)
    
    # DDP: 只在 rank 0 预下载数据，其他进程等待
    _ensure_data_downloaded(dataset_name, rank, world_size)
    
    results = []
    
    for seq_length in SEQ_LENGTHS:
        print_main(f"\n🔬 序列长度: {seq_length}", rank)
        
        # 创建特征处理器（确保数据已准备好）
        barrier()  # 同步所有进程
        fp = FeatureProcessor('./data', dataset_name)
        barrier()  # 等待所有进程完成初始化
        
        # 加载 Top-K 评估数据（仅在启用时）
        eval_data, ie_eval = None, None
        if ENABLE_TOPK and is_main_process(rank):
            eval_data, _, fp_eval, ie_eval = get_topk_eval_data(
                data_dir='./data',
                dataset_name=dataset_name,
                max_seq_length=seq_length,
                num_neg_samples=NUM_NEG_SAMPLES
            )
        
        # 创建数据集
        train_dataset = RichFeatureDataset(
            data_dir='./data',
            dataset_name=dataset_name,
            max_seq_length=seq_length,
            split='train',
            feature_processor=fp
        )
        
        valid_dataset = RichFeatureDataset(
            data_dir='./data',
            dataset_name=dataset_name,
            max_seq_length=seq_length,
            split='valid',
            feature_processor=fp
        )
        
        test_dataset = RichFeatureDataset(
            data_dir='./data',
            dataset_name=dataset_name,
            max_seq_length=seq_length,
            split='test',
            feature_processor=fp
        )
        
        dataset_info = {
            'num_items': train_dataset.num_items,
            'num_users': train_dataset.num_users,
            'feature_dims': fp.get_feature_dims()
        }
        
        # DDP DataLoader
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler,
            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True if NUM_WORKERS > 0 else False
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=valid_sampler,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=test_sampler,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        # 测试各模型
        for model_name in MODELS_TO_TEST:
            print_main(f"  🚀 {model_name}...", rank)
            
            try:
                # 创建模型（所有进程都打印调试信息）
                print(f"    [Rank {rank}] 正在创建模型 {model_name}...")
                model = create_model(model_name, dataset_info, EMBEDDING_DIM, seq_length)
                print(f"    [Rank {rank}] 模型创建完成，移动到设备 {device}...")
                model = model.to(device)
                print(f"    [Rank {rank}] 模型已移动到设备")
                
                # 同步所有进程
                print(f"    [Rank {rank}] 等待barrier同步...")
                barrier()
                print(f"    [Rank {rank}] barrier同步完成")
                
                # DDP 包装
                if world_size > 1:
                    print(f"    [Rank {rank}] 正在包装 DDP...")
                    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                    print(f"    [Rank {rank}] DDP 包装完成")
                
                # 同步所有进程
                print(f"    [Rank {rank}] 等待barrier同步...")
                barrier()
                print(f"    [Rank {rank}] barrier同步完成，准备创建trainer")
                
                # 训练
                trainer = SimpleDDPTrainer(model, device, local_rank, rank, world_size,
                                          TENSORBOARD_LOG_DIR, f'exp1_{dataset_name}_{model_name}_seq{seq_length}',
                                          patience=7, grad_clip=1.0)
                
                t1 = time.time()
                early_stopped = False
                for epoch in range(EPOCHS):
                    train_sampler.set_epoch(epoch)  # DDP 重要！
                    train_loss = trainer.train_epoch(train_loader)
                    
                    if is_main_process(rank):
                        valid_metrics = trainer.evaluate(valid_loader)
                        # 学习率调度和早停检查
                        early_stopped = trainer.step_scheduler(valid_metrics['auc'])
                        
                        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                            lr = trainer.optimizer.param_groups[0]['lr']
                            print(f"    Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Valid AUC: {valid_metrics['auc']:.4f} - LR: {lr:.2e}")
                    
                    # 广播早停信号到所有进程
                    if world_size > 1:
                        early_stop_tensor = torch.tensor([1 if early_stopped else 0], device=device)
                        dist.broadcast(early_stop_tensor, src=0)
                        early_stopped = early_stop_tensor.item() == 1
                    
                    if early_stopped:
                        print_main(f"    ⏹️ 早停触发 @ epoch {epoch+1}", rank)
                        break
                
                train_time = time.time() - t1
                
                # 恢复最佳模型
                if is_main_process(rank):
                    trainer.restore_best_model()
                
                # 测试
                if is_main_process(rank):
                    test_metrics = trainer.evaluate(test_loader)
                    
                    result = {
                        'experiment': 'exp1_model_comparison',
                        'dataset': dataset_name,
                        'seq_length': seq_length,
                        'model': model_name,
                        'test_auc': test_metrics['auc'],
                        'test_logloss': test_metrics['logloss'],
                        'train_time_sec': train_time,
                        'world_size': world_size,
                        'status': 'success'
                    }
                    
                    # Top-K 评估
                    if ENABLE_TOPK and eval_data is not None:
                        topk_metrics = trainer.evaluate_topk(
                            eval_data=eval_data,
                            feature_processor=fp,
                            interaction_extractor=ie_eval,
                            max_seq_length=seq_length,
                            ks=TOPK_VALUES
                        )
                        result.update(topk_metrics)
                        print(f"    ✅ AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}, Time={train_time:.1f}s")
                    else:
                        print(f"    ✅ AUC={test_metrics['auc']:.4f}, LogLoss={test_metrics['logloss']:.4f}, Time={train_time:.1f}s")
                    
                    # 记录超参数
                    trainer.log_hparams(
                        {'model': model_name, 'seq_length': seq_length, 'epochs': EPOCHS},
                        {'hparam/test_auc': test_metrics['auc']}
                    )
                    trainer.close()
                    
                    results.append(result)
                
                barrier()
                
            except Exception as e:
                print_main(f"    ❌ {str(e)[:100]}", rank)
                if is_main_process(rank):
                    results.append({
                        'experiment': 'exp1_model_comparison',
                        'dataset': dataset_name,
                        'seq_length': seq_length,
                        'model': model_name,
                        'status': f'error: {str(e)[:100]}'
                    })
    
    return results


# ========================================
# 实验2: 方法对比
# ========================================

def run_experiment2(dataset_name, device, local_rank, world_size, rank,
                    EPOCHS, BATCH_SIZE_PER_GPU, EMBEDDING_DIM,
                    NUM_WORKERS, PREFETCH_FACTOR,
                    TENSORBOARD_LOG_DIR,
                    ENABLE_TOPK=True, TOPK_VALUES=[5, 10, 20]):
    """实验2: 方法对比（DIN vs 传统方法）"""
    
    print_main("\n" + "=" * 60, rank)
    print_main("📊 实验2: DIN vs 传统方法", rank)
    print_main("=" * 60, rank)
    
    # DDP: 预下载数据
    _ensure_data_downloaded(dataset_name, rank, world_size)
    
    results = []
    seq_length = 50  # 固定序列长度
    
    # 创建数据集（确保数据已准备好）
    barrier()  # 同步所有进程
    fp = FeatureProcessor('./data', dataset_name)
    barrier()  # 等待所有进程完成初始化
    
    # Top-K 评估数据
    eval_data = None
    ie_eval = None
    if ENABLE_TOPK and is_main_process(rank):
        try:
            eval_data, _, fp_eval, ie_eval = get_topk_eval_data('./data', dataset_name, seq_length)
            print_main(f"  📊 Top-K评估数据加载完成: {len(eval_data)} 条", rank)
        except Exception as e:
            print_main(f"  ⚠️ Top-K评估数据加载失败: {e}", rank)
            ENABLE_TOPK = False
    
    train_dataset = RichFeatureDataset(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        split='train',
        feature_processor=fp
    )
    valid_dataset = RichFeatureDataset(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        split='valid',
        feature_processor=fp
    )
    test_dataset = RichFeatureDataset(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        split='test',
        feature_processor=fp
    )
    
    dataset_info = {
        'num_items': train_dataset.num_items,
        'num_users': train_dataset.num_users,
        'feature_dims': fp.get_feature_dims()
    }
    
    # DDP DataLoader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_PER_GPU, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PER_GPU, num_workers=NUM_WORKERS, pin_memory=True)
    
    methods = ['DIN', 'AvgPool']
    
    for method in methods:
        print_main(f"  🚀 {method}...", rank)
        
        try:
            model = create_model(method, dataset_info, EMBEDDING_DIM, seq_length)
            model = model.to(device)
            
            if world_size > 1:
                model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            
            trainer = SimpleDDPTrainer(model, device, local_rank, rank, world_size,
                                      TENSORBOARD_LOG_DIR, f'exp2_{dataset_name}_{method}',
                                      patience=7, grad_clip=1.0)
            
            t1 = time.time()
            early_stopped = False
            for epoch in range(EPOCHS):
                train_sampler.set_epoch(epoch)
                train_loss = trainer.train_epoch(train_loader)
                
                if is_main_process(rank):
                    valid_metrics = trainer.evaluate(valid_loader)
                    early_stopped = trainer.step_scheduler(valid_metrics['auc'])
                    
                    if (epoch + 1) % 10 == 0:
                        lr = trainer.optimizer.param_groups[0]['lr']
                        print(f"    Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Valid AUC: {valid_metrics['auc']:.4f} - LR: {lr:.2e}")
                
                # 广播早停信号
                if world_size > 1:
                    early_stop_tensor = torch.tensor([1 if early_stopped else 0], device=device)
                    dist.broadcast(early_stop_tensor, src=0)
                    early_stopped = early_stop_tensor.item() == 1
                
                if early_stopped:
                    print_main(f"    ⏹️ 早停触发 @ epoch {epoch+1}", rank)
                    break
            
            train_time = time.time() - t1
            
            # 恢复最佳模型
            if is_main_process(rank):
                trainer.restore_best_model()
            
            if is_main_process(rank):
                test_metrics = trainer.evaluate(test_loader)
                
                result = {
                    'experiment': 'exp2_method_comparison',
                    'dataset': dataset_name,
                    'method': method,
                    'test_auc': test_metrics['auc'],
                    'test_logloss': test_metrics['logloss'],
                    'train_time_sec': train_time,
                    'status': 'success'
                }
                
                # Top-K 评估
                if ENABLE_TOPK and eval_data is not None:
                    topk_metrics = trainer.evaluate_topk(
                        eval_data=eval_data,
                        feature_processor=fp,
                        interaction_extractor=ie_eval,
                        max_seq_length=seq_length,
                        ks=TOPK_VALUES
                    )
                    result.update(topk_metrics)
                    print(f"    ✅ AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}")
                else:
                    print(f"    ✅ AUC={test_metrics['auc']:.4f}")
                
                # 记录超参数和关闭
                trainer.log_hparams(
                    {'method': method, 'epochs': EPOCHS},
                    {'hparam/test_auc': test_metrics['auc']}
                )
                trainer.close()
                
                results.append(result)
            
            barrier()
            
        except Exception as e:
            print_main(f"    ❌ {str(e)[:100]}", rank)
            if is_main_process(rank):
                results.append({
                    'experiment': 'exp2_method_comparison',
                    'dataset': dataset_name,
                    'method': method,
                    'status': f'error: {str(e)[:100]}'
                })
    
    return results


# ========================================
# 实验3: 消融实验
# ========================================

def run_experiment3(dataset_name, device, local_rank, world_size, rank,
                    EPOCHS, BATCH_SIZE_PER_GPU, EMBEDDING_DIM,
                    NUM_WORKERS, PREFETCH_FACTOR,
                    TENSORBOARD_LOG_DIR,
                    ENABLE_TOPK=True, TOPK_VALUES=[5, 10, 20]):
    """实验3: 消融实验"""
    
    print_main("\n" + "=" * 60, rank)
    print_main("📊 实验3: 消融实验", rank)
    print_main("=" * 60, rank)
    
    # DDP: 预下载数据
    _ensure_data_downloaded(dataset_name, rank, world_size)
    
    results = []
    seq_length = 50
    
    # 创建数据集（确保数据已准备好）
    barrier()  # 同步所有进程
    fp = FeatureProcessor('./data', dataset_name)
    barrier()  # 等待所有进程完成初始化
    
    # Top-K 评估数据
    eval_data = None
    ie_eval = None
    if ENABLE_TOPK and is_main_process(rank):
        try:
            eval_data, _, fp_eval, ie_eval = get_topk_eval_data('./data', dataset_name, seq_length)
            print_main(f"  📊 Top-K评估数据加载完成: {len(eval_data)} 条", rank)
        except Exception as e:
            print_main(f"  ⚠️ Top-K评估数据加载失败: {e}", rank)
            ENABLE_TOPK = False
    
    train_dataset = RichFeatureDataset(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        split='train',
        feature_processor=fp
    )
    valid_dataset = RichFeatureDataset(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        split='valid',
        feature_processor=fp
    )
    test_dataset = RichFeatureDataset(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        split='test',
        feature_processor=fp
    )
    
    dataset_info = {
        'num_items': train_dataset.num_items,
        'num_users': train_dataset.num_users,
        'feature_dims': fp.get_feature_dims()
    }
    
    # DDP DataLoader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_PER_GPU, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_PER_GPU, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 消融变体
    variants = ['full_din', 'no_attention', 'shallow_mlp']
    
    for variant in variants:
        print_main(f"  🚀 {variant}...", rank)
        
        try:
            model = create_ablation_model(variant, dataset_info, EMBEDDING_DIM)
            model = model.to(device)
            
            if world_size > 1:
                model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            
            trainer = SimpleDDPTrainer(model, device, local_rank, rank, world_size,
                                      TENSORBOARD_LOG_DIR, f'exp3_{dataset_name}_{variant}',
                                      patience=7, grad_clip=1.0)
            
            t1 = time.time()
            early_stopped = False
            for epoch in range(EPOCHS):
                train_sampler.set_epoch(epoch)
                train_loss = trainer.train_epoch(train_loader)
                
                if is_main_process(rank):
                    valid_metrics = trainer.evaluate(valid_loader)
                    early_stopped = trainer.step_scheduler(valid_metrics['auc'])
                    
                    if (epoch + 1) % 10 == 0:
                        lr = trainer.optimizer.param_groups[0]['lr']
                        print(f"    Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Valid AUC: {valid_metrics['auc']:.4f} - LR: {lr:.2e}")
                
                # 广播早停信号
                if world_size > 1:
                    early_stop_tensor = torch.tensor([1 if early_stopped else 0], device=device)
                    dist.broadcast(early_stop_tensor, src=0)
                    early_stopped = early_stop_tensor.item() == 1
                
                if early_stopped:
                    print_main(f"    ⏹️ 早停触发 @ epoch {epoch+1}", rank)
                    break
            
            train_time = time.time() - t1
            
            # 恢复最佳模型
            if is_main_process(rank):
                trainer.restore_best_model()
            
            if is_main_process(rank):
                test_metrics = trainer.evaluate(test_loader)
                
                result = {
                    'experiment': 'exp3_ablation',
                    'dataset': dataset_name,
                    'variant': variant,
                    'test_auc': test_metrics['auc'],
                    'test_logloss': test_metrics['logloss'],
                    'train_time_sec': train_time,
                    'status': 'success'
                }
                
                # Top-K 评估
                if ENABLE_TOPK and eval_data is not None:
                    topk_metrics = trainer.evaluate_topk(
                        eval_data=eval_data,
                        feature_processor=fp,
                        interaction_extractor=ie_eval,
                        max_seq_length=seq_length,
                        ks=TOPK_VALUES
                    )
                    result.update(topk_metrics)
                    print(f"    ✅ AUC={test_metrics['auc']:.4f}, HR@10={topk_metrics['HR@10']:.4f}, NDCG@10={topk_metrics['NDCG@10']:.4f}")
                else:
                    print(f"    ✅ AUC={test_metrics['auc']:.4f}")
                
                # 记录超参数和关闭
                trainer.log_hparams(
                    {'variant': variant, 'epochs': EPOCHS},
                    {'hparam/test_auc': test_metrics['auc']}
                )
                trainer.close()
                
                results.append(result)
            
            barrier()
            
        except Exception as e:
            print_main(f"    ❌ {str(e)[:100]}", rank)
            if is_main_process(rank):
                results.append({
                    'experiment': 'exp3_ablation',
                    'dataset': dataset_name,
                    'variant': variant,
                    'status': f'error: {str(e)[:100]}'
                })
    
    return results


def run_experiment4(dataset_name, device, local_rank, world_size, rank,
                    EPOCHS, BATCH_SIZE_PER_GPU, EMBEDDING_DIM,
                    NUM_WORKERS, PREFETCH_FACTOR,
                    TENSORBOARD_LOG_DIR, part='all'):
    """
    实验4: 高级改进实验（自适应时间衰减 + 对比学习）
    
    DDP 版本：通过动态导入 experiment4.py 实现
    仅在主进程上运行（因为 experiment4.py 内部不支持 DDP）
    """
    
    print_main("\n" + "=" * 60, rank)
    print_main("📊 实验4: 高级改进实验", rank)
    print_main("=" * 60, rank)
    
    results = []
    
    # 实验4 的原始实现不支持 DDP，仅在主进程运行
    # 非主进程直接跳过实验，最后统一 barrier
    if not is_main_process(rank):
        # 非主进程等待主进程完成后再同步
        barrier()
        return results
    
    # === 主进程执行实验 ===
    try:
        # 动态导入 experiment4 模块
        import importlib.util
        exp4_path = os.path.join(os.path.dirname(__file__), 'experiment4.py')
        
        if not os.path.exists(exp4_path):
            print("❌ experiment4.py 不存在，跳过实验四")
            # 主进程也要调用 barrier 以匹配非主进程
            barrier()
            return results
        
        spec = importlib.util.spec_from_file_location("experiment4", exp4_path)
        exp4_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp4_module)
        
        # 使用第一个 GPU 运行（主进程）
        exp4_device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        
        # Part 1: 自适应时间衰减实验
        if part in ['all', 'adaptive']:
            print("\n📊 Part 1: 自适应时间衰减实验")
            print("-" * 40)
            try:
                adaptive_results = exp4_module.run_adaptive_decay_experiment(
                    dataset_name=dataset_name,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE_PER_GPU * world_size,  # 使用完整 batch size
                    device=exp4_device
                )
                for r in adaptive_results:
                    r['experiment'] = 'exp4_adaptive_decay'
                    r['dataset'] = dataset_name
                results.extend(adaptive_results)
                print(f"✅ 自适应时间衰减实验完成，{len(adaptive_results)} 组结果")
            except Exception as e:
                print(f"❌ 自适应时间衰减实验失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'experiment': 'exp4_adaptive_decay',
                    'dataset': dataset_name,
                    'status': f'error: {str(e)[:100]}'
                })
        
        # Part 2: 对比学习实验
        if part in ['all', 'contrastive']:
            print("\n📊 Part 2: 对比学习实验")
            print("-" * 40)
            try:
                contrastive_results = exp4_module.run_contrastive_experiment(
                    dataset_name=dataset_name,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE_PER_GPU * world_size,
                    device=exp4_device
                )
                for r in contrastive_results:
                    r['experiment'] = 'exp4_contrastive'
                    r['dataset'] = dataset_name
                results.extend(contrastive_results)
                print(f"✅ 对比学习实验完成，{len(contrastive_results)} 组结果")
            except Exception as e:
                print(f"❌ 对比学习实验失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'experiment': 'exp4_contrastive',
                    'dataset': dataset_name,
                    'status': f'error: {str(e)[:100]}'
                })
                
    except Exception as e:
        print(f"❌ 实验四加载失败: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'experiment': 'exp4',
            'dataset': dataset_name,
            'status': f'load_error: {str(e)[:100]}'
        })
    
    barrier()  # 同步所有进程
    return results


# ========================================
# 工具类和函数
# ========================================

def create_model(model_name, dataset_info, embedding_dim, seq_length):
    """创建模型"""
    if model_name == 'DIN':
        return DINRichLite(
            num_items=dataset_info['num_items'],
            num_users=dataset_info['num_users'],
            feature_dims=dataset_info['feature_dims'],
            embedding_dim=embedding_dim
        )
    elif model_name == 'GRU4Rec':
        return GRU4Rec(
            num_items=dataset_info['num_items'],
            num_users=dataset_info['num_users'],
            feature_dims=dataset_info['feature_dims'],
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim
        )
    elif model_name == 'SASRec':
        return SASRec(
            num_items=dataset_info['num_items'],
            num_users=dataset_info['num_users'],
            feature_dims=dataset_info['feature_dims'],
            embedding_dim=embedding_dim,
            num_heads=2,
            num_layers=2,
            max_seq_len=seq_length
        )
    elif model_name == 'NARM':
        return NARM(
            num_items=dataset_info['num_items'],
            num_users=dataset_info['num_users'],
            feature_dims=dataset_info['feature_dims'],
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim
        )
    elif model_name == 'AvgPool':
        return SimpleAveragePoolingRich(
            num_items=dataset_info['num_items'],
            num_users=dataset_info['num_users'],
            feature_dims=dataset_info['feature_dims'],
            embedding_dim=embedding_dim
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_ablation_model(variant, dataset_info, embedding_dim):
    """创建消融实验模型"""
    if variant == 'full_din':
        return DINRichLite(
            num_items=dataset_info['num_items'],
            num_users=dataset_info['num_users'],
            feature_dims=dataset_info['feature_dims'],
            embedding_dim=embedding_dim
        )
    elif variant == 'no_attention':
        return SimpleAveragePoolingRich(
            num_items=dataset_info['num_items'],
            num_users=dataset_info['num_users'],
            feature_dims=dataset_info['feature_dims'],
            embedding_dim=embedding_dim
        )
    elif variant == 'shallow_mlp':
        return DINRichLite(
            num_items=dataset_info['num_items'],
            num_users=dataset_info['num_users'],
            feature_dims=dataset_info['feature_dims'],
            embedding_dim=embedding_dim,
            mlp_hidden_dims=[128, 64]  # 更浅的 MLP
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class SimpleDDPTrainer:
    """
    完整的 DDP 训练器
    
    包含：
    - 学习率调度器 (ReduceLROnPlateau)
    - 早停机制 (Early Stopping)
    - 梯度裁剪 (Gradient Clipping)
    - 混合精度训练 (AMP)
    - TensorBoard 日志
    """
    
    def __init__(self, model, device, local_rank, rank, world_size, log_dir, exp_name,
                 learning_rate=1e-3, weight_decay=1e-5, 
                 patience=5, grad_clip=1.0,
                 lr_scheduler_patience=3, lr_scheduler_factor=0.5):
        self.model = model
        self.device = device
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        
        # 训练配置
        self.patience = patience  # 早停耐心值
        self.grad_clip = grad_clip  # 梯度裁剪阈值
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 获取原始模型参数
        if hasattr(model, 'module'):
            params = model.module.parameters()
        else:
            params = model.parameters()
        
        self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        
        # 学习率调度器 (ReduceLROnPlateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience, verbose=False
        )
        
        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # 早停状态
        self.best_valid_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
        # TensorBoard（仅主进程）
        self.writer = None
        if self.is_main and HAS_TENSORBOARD:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_dir = os.path.join(log_dir, f"{exp_name}_{timestamp}")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)
        
        self.epoch = 0
        self.global_step = 0
    
    def _move_batch(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
    
    def _get_raw_model(self):
        """获取原始模型（DDP包装下）"""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model
    
    def train_epoch(self, train_loader):
        """训练一个 epoch（含梯度裁剪）"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch)
                    loss = self.criterion(logits, batch['label'])
                self.scaler.scale(loss).backward()
                # 梯度裁剪（AMP模式下需要先 unscale）
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._get_raw_model().parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(batch)
                loss = self.criterion(logits, batch['label'])
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self._get_raw_model().parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        
        if self.writer:
            self.writer.add_scalar('Loss/train', avg_loss, self.epoch)
            self.writer.add_scalar('LR/learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        
        self.epoch += 1
        return avg_loss
    
    def step_scheduler(self, valid_auc):
        """更新学习率调度器和早停检查"""
        # 更新学习率调度器
        self.scheduler.step(valid_auc)
        
        # 早停检查
        if valid_auc > self.best_valid_auc:
            self.best_valid_auc = valid_auc
            self.best_epoch = self.epoch
            self.epochs_without_improvement = 0
            # 保存最佳模型状态
            self.best_model_state = {k: v.cpu().clone() for k, v in self._get_raw_model().state_dict().items()}
            return False  # 不早停
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                return True  # 触发早停
            return False
    
    def restore_best_model(self):
        """恢复最佳模型"""
        if self.best_model_state is not None:
            self._get_raw_model().load_state_dict(self.best_model_state)
            if self.is_main:
                print(f"    📌 恢复到最佳模型 (epoch {self.best_epoch}, AUC={self.best_valid_auc:.4f})")
    
    def evaluate(self, data_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch(batch)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits = self.model(batch)
                else:
                    logits = self.model(batch)
                
                preds = torch.sigmoid(logits).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        from sklearn.metrics import roc_auc_score, log_loss
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        logloss = log_loss(all_labels, np.clip(all_preds, 1e-7, 1-1e-7))
        
        if self.writer:
            self.writer.add_scalar('Metrics/valid_auc', auc, self.epoch)
            self.writer.add_scalar('Metrics/valid_logloss', logloss, self.epoch)
        
        return {'auc': auc, 'logloss': logloss}
    
    def evaluate_topk(self, eval_data, feature_processor, interaction_extractor, 
                      max_seq_length, ks=[5, 10, 20]):
        """
        Top-K 推荐评估
        
        Args:
            eval_data: list of dict，来自 get_topk_eval_data
            feature_processor: 特征处理器
            interaction_extractor: 交互特征提取器
            max_seq_length: 最大序列长度
            ks: 评估的 K 值列表
        
        Returns:
            dict: 各指标在不同 K 下的值
        """
        self.model.eval()
        
        # 初始化指标累加器
        all_hr = {k: [] for k in ks}
        all_ndcg = {k: [] for k in ks}
        all_mrr = {k: [] for k in ks}
        all_precision = {k: [] for k in ks}
        
        with torch.no_grad():
            for eval_item in eval_data:
                # 构建单用户的候选 batch
                batch = build_topk_batch_multi(
                    eval_item, feature_processor, interaction_extractor,
                    max_seq_length, self.device
                )
                
                # 预测分数
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits = self.model(batch)
                else:
                    logits = self.model(batch)
                
                scores = torch.sigmoid(logits).cpu().numpy()
                
                # 排序
                candidates = eval_item['candidates']
                ground_truth = eval_item['ground_truth']
                sorted_indices = np.argsort(-scores)
                ranked_items = [candidates[i] for i in sorted_indices]
                
                # 计算指标
                for k in ks:
                    all_hr[k].append(hit_at_k(ranked_items, ground_truth, k))
                    all_ndcg[k].append(ndcg_at_k(ranked_items, ground_truth, k))
                    all_mrr[k].append(mrr_at_k(ranked_items, ground_truth, k))
                    all_precision[k].append(precision_at_k(ranked_items, ground_truth, k))
        
        # 计算平均值
        results = {}
        for k in ks:
            results[f'HR@{k}'] = np.mean(all_hr[k])
            results[f'Recall@{k}'] = np.mean(all_hr[k])  # 单 GT 等于 HR
            results[f'NDCG@{k}'] = np.mean(all_ndcg[k])
            results[f'MRR@{k}'] = np.mean(all_mrr[k])
            results[f'Precision@{k}'] = np.mean(all_precision[k])
        
        # 记录到 TensorBoard
        if self.writer:
            for k in ks:
                self.writer.add_scalar(f'TopK/HR@{k}', results[f'HR@{k}'], self.epoch)
                self.writer.add_scalar(f'TopK/NDCG@{k}', results[f'NDCG@{k}'], self.epoch)
                self.writer.add_scalar(f'TopK/MRR@{k}', results[f'MRR@{k}'], self.epoch)
                self.writer.add_scalar(f'TopK/Precision@{k}', results[f'Precision@{k}'], self.epoch)
        
        return results
    
    def log_hparams(self, hparams, metrics):
        """记录超参数和最终指标到 TensorBoard"""
        if self.writer:
            self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """关闭 TensorBoard writer"""
        if self.writer:
            self.writer.close()
            self.writer = None
    
    def __del__(self):
        self.close()


# ========================================
# 入口
# ========================================

if __name__ == '__main__':
    main()
