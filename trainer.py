#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版训练器

支持丰富特征的模型训练。
支持 CTR 指标（AUC, LogLoss）和 Top-K 推荐指标（Recall@K, NDCG@K, HR@K, MRR）。
支持 TensorBoard 可视化训练过程。
支持混合精度训练 (AMP) 加速。
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
from tqdm import tqdm
import time
import os
from datetime import datetime

# 混合精度训练支持 (PyTorch 1.6+)
try:
    from torch.cuda.amp import GradScaler, autocast
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

# TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("警告: TensorBoard 未安装，可视化功能不可用。安装: pip install tensorboard")


# ========================================
# Top-K 评估指标
# ========================================

def hit_at_k(ranked_items, ground_truth, k):
    """
    Hit Rate @ K
    如果 ground_truth 在 top-k 中，返回 1，否则返回 0
    """
    return 1.0 if ground_truth in ranked_items[:k] else 0.0


def recall_at_k(ranked_items, ground_truth, k):
    """
    Recall @ K
    对于单个 ground truth，等同于 Hit Rate
    """
    return hit_at_k(ranked_items, ground_truth, k)


def ndcg_at_k(ranked_items, ground_truth, k):
    """
    NDCG @ K (Normalized Discounted Cumulative Gain)
    """
    for i, item in enumerate(ranked_items[:k]):
        if item == ground_truth:
            # DCG = 1 / log2(rank + 1)，IDCG = 1 / log2(2) = 1
            return 1.0 / np.log2(i + 2)  # +2 因为 rank 从 1 开始
    return 0.0


def mrr_at_k(ranked_items, ground_truth, k):
    """
    MRR @ K (Mean Reciprocal Rank)
    """
    for i, item in enumerate(ranked_items[:k]):
        if item == ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked_items, ground_truth, k):
    """
    Precision @ K
    对于单个 ground truth: 命中则为 1/k，否则为 0
    """
    if ground_truth in ranked_items[:k]:
        return 1.0 / k
    return 0.0


class RichTrainer:
    """
    增强版训练器
    
    支持 batch 字典形式的输入。
    支持多 GPU DistributedDataParallel (DDP) 加速。
    支持 TensorBoard 可视化。
    """
    
    def __init__(
        self,
        model,
        device='cpu',
        learning_rate=1e-3,
        weight_decay=1e-5,
        use_multi_gpu=False,  # 是否使用多 GPU (DataParallel)
        use_ddp=False,  # 是否使用 DDP（更高效）
        local_rank=-1,  # DDP 的 local rank
        use_amp=True,  # 是否使用混合精度训练
        use_tensorboard=True,  # 是否使用 TensorBoard
        log_dir='./runs',  # TensorBoard 日志目录
        experiment_name=None  # 实验名称
    ):
        self.device = device
        self.use_ddp = use_ddp
        self.local_rank = local_rank
        self.is_main_process = (local_rank <= 0)  # rank 0 或非DDP模式
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1 and not use_ddp
        self.learning_rate = learning_rate
        
        # 混合精度训练 (AMP) - 仅在 GPU 上启用
        # 支持 'cuda', 'cuda:0', 'cuda:1' 等格式
        is_cuda_device = str(device).startswith('cuda') or device == 'cuda'
        self.use_amp = use_amp and HAS_AMP and is_cuda_device
        if self.use_amp and self.is_main_process:
            self.scaler = GradScaler()
            print("⚡ 混合精度训练 (AMP) 已启用")
        elif self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 将模型移到设备
        model = model.to(device)
        
        # DDP 支持（优先于 DataParallel）
        if self.use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            if self.is_main_process:
                print(f"🚀 使用 DistributedDataParallel (DDP): GPU {local_rank}")
        # DataParallel 支持（备选）
        elif self.use_multi_gpu:
            if self.is_main_process:
                print(f"🔥 使用 DataParallel: {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        self.model = model
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # TensorBoard 设置（仅主进程）
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD and self.is_main_process
        self.writer = None
        self.global_step = 0
        self._writer_closed = False  # 防止重复关闭
        
        if self.use_tensorboard:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = experiment_name or "default"
            self.log_dir = os.path.join(log_dir, f"{exp_name}_{timestamp}")
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
            print(f"📊 TensorBoard 已启用")
            print(f"   日志目录: {self.log_dir}")
            print(f"   启动命令: tensorboard --logdir {log_dir}")
    
    def close(self):
        """关闭 TensorBoard writer（显式调用）"""
        if self.writer is not None and not self._writer_closed:
            self.writer.close()
            self._writer_closed = True
    
    def __del__(self):
        """析构时确保 writer 被关闭"""
        self.close()
    
    @property
    def raw_model(self):
        """获取原始模型（用于访问模型属性或保存）"""
        if (self.use_multi_gpu or self.use_ddp) and hasattr(self.model, 'module'):
            return self.model.module
        return self.model
    
    def _move_batch_to_device(self, batch):
        """将 batch 移动到设备"""
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
    
    def train_epoch(self, train_loader, show_progress=True):
        """训练一个 epoch（支持混合精度）"""
        self.model.train()
        total_loss = 0
        
        # 仅主进程显示进度条
        show = show_progress and self.is_main_process
        iterator = tqdm(train_loader, desc='Training') if show else train_loader
        
        for batch in iterator:
            batch = self._move_batch_to_device(batch)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    logits = self.model(batch)
                    loss = self.criterion(logits, batch['label'])
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(batch)
                loss = self.criterion(logits, batch['label'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader, show_progress=False):
        """评估模型（支持混合精度）"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        iterator = tqdm(data_loader, desc='Evaluating') if show_progress else data_loader
        
        with torch.no_grad():
            for batch in iterator:
                batch = self._move_batch_to_device(batch)
                
                # 评估时也使用 AMP 加速
                if self.use_amp:
                    with autocast():
                        logits = self.model(batch)
                else:
                    logits = self.model(batch)
                
                preds = torch.sigmoid(logits).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        logloss = log_loss(all_labels, np.clip(all_preds, 1e-7, 1-1e-7))
        
        return {
            'auc': auc,
            'logloss': logloss
        }
    
    def fit(
        self,
        train_loader,
        valid_loader,
        epochs=20,
        early_stopping_patience=5,
        show_progress=True
    ):
        """训练模型（支持 TensorBoard 可视化）"""
        best_valid_auc = 0
        patience_counter = 0
        best_model_state = None
        
        # 记录超参数到 TensorBoard
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_text('Hyperparameters', 
                f'learning_rate={self.learning_rate}, epochs={epochs}, '
                f'early_stopping_patience={early_stopping_patience}')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, show_progress)
            valid_metrics = self.evaluate(valid_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - "
                  f"Valid AUC: {valid_metrics['auc']:.4f} - "
                  f"Valid LogLoss: {valid_metrics['logloss']:.4f}")
            
            # TensorBoard 记录
            if self.use_tensorboard and self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Metrics/valid_auc', valid_metrics['auc'], epoch)
                self.writer.add_scalar('Metrics/valid_logloss', valid_metrics['logloss'], epoch)
                self.writer.add_scalar('Learning_rate', 
                    self.optimizer.param_groups[0]['lr'], epoch)
            
            if valid_metrics['auc'] > best_valid_auc:
                best_valid_auc = valid_metrics['auc']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # 记录最佳指标
                if self.use_tensorboard and self.writer is not None:
                    self.writer.add_scalar('Metrics/best_valid_auc', best_valid_auc, epoch)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    if self.use_tensorboard and self.writer is not None:
                        self.writer.add_text('Training', f'Early stopped at epoch {epoch+1}')
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 关闭 TensorBoard writer（使用 close 方法避免重复关闭）
        if self.use_tensorboard and self.writer is not None and not self._writer_closed:
            self.writer.add_hparams(
                {'lr': self.learning_rate, 'epochs': epoch + 1},
                {'hparam/best_valid_auc': best_valid_auc}
            )
            self.close()
            print(f"✓ TensorBoard 日志已保存到: {self.log_dir}")
        
        return {
            'best_valid_auc': best_valid_auc,
            'final_epoch': epoch + 1
        }
    
    def evaluate_topk(
        self,
        eval_data,
        feature_processor,
        interaction_extractor,
        max_seq_length,
        ks=[5, 10, 20],
        show_progress=True,
        batch_size=256
    ):
        """
        Top-K 推荐评估（批量优化版）
        
        Args:
            eval_data: list of dict，来自 get_topk_eval_data
            feature_processor: 特征处理器
            interaction_extractor: 交互特征提取器
            max_seq_length: 最大序列长度
            ks: 评估的 K 值列表
            show_progress: 是否显示进度条
            batch_size: 批量评估的用户数
        
        Returns:
            dict: 各指标在不同 K 下的值
        """
        from data_loader import build_topk_batch_multi
        
        self.model.eval()
        
        # 初始化指标累加器
        all_hr = {k: [] for k in ks}
        all_ndcg = {k: [] for k in ks}
        all_mrr = {k: [] for k in ks}
        
        # 分批处理
        num_users = len(eval_data)
        num_batches = (num_users + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc='Top-K Eval')
        
        with torch.no_grad():
            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_users)
                batch_eval_data = eval_data[start_idx:end_idx]
                
                # 批量构建并评估
                for eval_item in batch_eval_data:
                    # 构建单用户的候选 batch
                    batch = build_topk_batch_multi(
                        eval_item, feature_processor, interaction_extractor,
                        max_seq_length, self.device
                    )
                    
                    # 预测分数
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
        
        # 计算平均值
        results = {}
        for k in ks:
            results[f'HR@{k}'] = np.mean(all_hr[k])
            results[f'Recall@{k}'] = np.mean(all_hr[k])  # 单 GT 等于 HR
            results[f'NDCG@{k}'] = np.mean(all_ndcg[k])
            results[f'MRR@{k}'] = np.mean(all_mrr[k])
            results[f'Precision@{k}'] = np.mean(all_hr[k]) / k
        
        return results


def measure_inference_speed_rich(model, data_loader, device='cpu', warmup_batches=5, measure_batches=20):
    """
    测量推理速度（QPS）
    
    适用于 batch 字典输入的模型。
    """
    model.eval()
    model = model.to(device)
    
    sample_batch = next(iter(data_loader))
    batch_size = sample_batch['user_id'].shape[0]
    
    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= warmup_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(batch)
    
    # 测量
    total_samples = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= measure_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(batch)
            total_samples += batch['user_id'].shape[0]
    
    elapsed = time.time() - start_time
    qps = total_samples / elapsed if elapsed > 0 else 0
    
    return {
        'qps': qps,
        'total_samples': total_samples,
        'elapsed_time': elapsed
    }


if __name__ == "__main__":
    print("测试增强版训练器...")
    
    from data_loader import get_rich_dataloaders
    from models import DINRichLite
    
    train_loader, valid_loader, test_loader, info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name='ml-100k',
        max_seq_length=50,
        batch_size=256
    )
    
    model = DINRichLite(
        num_items=info['num_items'],
        num_users=info['num_users'],
        feature_dims=info['feature_dims'],
        embedding_dim=64
    )
    
    trainer = RichTrainer(model=model, device='cpu')
    
    # 快速测试
    result = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=2,
        show_progress=True
    )
    
    print(f"\n训练结果: {result}")
    
    test_metrics = trainer.evaluate(test_loader)
    print(f"测试结果: {test_metrics}")
