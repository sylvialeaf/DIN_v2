#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版数据加载器

支持丰富的特征：
- 用户特征：年龄、性别、职业
- 物品特征：类型、年份
- 交互特征：时间、统计
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import urllib.request
import zipfile

from feature_engineering import FeatureProcessor, InteractionFeatureExtractor, GENRES


class RichFeatureDataset(Dataset):
    """
    增强版数据集
    
    包含完整的用户、物品、序列特征。
    """
    
    def __init__(
        self,
        data_dir,
        dataset_name='ml-100k',
        max_seq_length=50,
        min_seq_length=5,
        split='train',
        train_ratio=0.8,
        valid_ratio=0.1,
        feature_processor=None,
        interaction_extractor=None
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.split = split
        
        # 加载原始数据
        self._download_if_needed()
        self.raw_data = self._load_raw_data()
        
        # 特征处理器
        if feature_processor is None:
            self.feature_processor = FeatureProcessor(data_dir, dataset_name)
        else:
            self.feature_processor = feature_processor
        
        # 交互特征提取器
        if interaction_extractor is None:
            self.interaction_extractor = InteractionFeatureExtractor(self.raw_data)
        else:
            self.interaction_extractor = interaction_extractor
        
        # 构建用户历史序列
        self.user_sequences = self._build_sequences()
        
        # 预先缓存所有物品列表（用于负样本采样，避免重复创建）
        self._all_items_list = list(self.feature_processor.item_df['item_id'].values)
        
        # 划分数据集
        self._split_data(train_ratio, valid_ratio)
        
        # 构建样本
        self.samples = self._build_samples()
    
    def _download_if_needed(self):
        """如果数据不存在，自动下载（DDP安全）"""
        data_path = os.path.join(self.data_dir, self.dataset_name)
        
        # 检查关键文件是否存在
        if self.dataset_name == 'ml-100k':
            key_file = os.path.join(data_path, 'u.data')
        elif self.dataset_name == 'ml-1m':
            key_file = os.path.join(data_path, 'ratings.dat')
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        # 文件存在，直接返回
        if os.path.exists(key_file):
            return
        
        # DDP: 只在非分布式或 rank 0 下载
        if dist.is_initialized():
            if dist.get_rank() != 0:
                # 非 rank 0 进程等待 rank 0 下载完成
                dist.barrier()
                return
        
        # Rank 0 执行下载
        os.makedirs(self.data_dir, exist_ok=True)
        
        if self.dataset_name == 'ml-100k':
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        elif self.dataset_name == 'ml-1m':
            url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        print(f"[DataLoader Rank 0] 下载数据集 {self.dataset_name}...")
        zip_path = os.path.join(self.data_dir, f'{self.dataset_name}.zip')
        urllib.request.urlretrieve(url, zip_path)
        
        print("[DataLoader Rank 0] 解压数据...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        os.remove(zip_path)
        print("[DataLoader Rank 0] 数据准备完成!")
        
        # 等待所有进程同步
        if dist.is_initialized():
            dist.barrier()
    
    def _load_raw_data(self):
        """加载原始评分数据"""
        data_path = os.path.join(self.data_dir, self.dataset_name)
        
        if self.dataset_name == 'ml-100k':
            file_path = os.path.join(data_path, 'u.data')
            df = pd.read_csv(
                file_path, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
        elif self.dataset_name == 'ml-1m':
            file_path = os.path.join(data_path, 'ratings.dat')
            df = pd.read_csv(
                file_path, 
                sep='::', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        # 使用最大值而不是唯一数量，避免嵌入索引越界
        # 因为 item_id 可能不连续（比如 ml-1m 的 item_id 最大是 3952，但唯一数只有 3706）
        self.num_users = df['user_id'].max()
        self.num_items = df['item_id'].max()
        
        # 只在主进程或非DDP环境打印
        should_print = not dist.is_initialized() or dist.get_rank() == 0
        if should_print:
            print(f"数据集: {self.dataset_name}")
            print(f"  用户数: {self.num_users} (max_id)")
            print(f"  物品数: {self.num_items} (max_id)")
            print(f"  交互数: {len(df)}")
        
        return df
    
    def _build_sequences(self):
        """按用户构建时间排序的交互序列（优化版本）"""
        user_sequences = defaultdict(list)
        
        # 排序
        sorted_data = self.raw_data.sort_values(['user_id', 'timestamp'])
        
        # 使用 groupby 代替 iterrows，速度快 10-100 倍
        for user_id, group in sorted_data.groupby('user_id', sort=False):
            items = group['item_id'].tolist()
            ratings = group['rating'].tolist()
            timestamps = group['timestamp'].tolist()
            
            for i in range(len(items)):
                user_sequences[user_id].append({
                    'item_id': items[i],
                    'rating': ratings[i],
                    'timestamp': timestamps[i]
                })
        
        filtered_sequences = {
            u: seq for u, seq in user_sequences.items() 
            if len(seq) >= self.min_seq_length
        }
        
        # 只在主进程或非DDP环境打印
        should_print = not dist.is_initialized() or dist.get_rank() == 0
        if should_print:
            print(f"  有效用户数（序列长度 >= {self.min_seq_length}）: {len(filtered_sequences)}")
        
        return filtered_sequences
    
    def _split_data(self, train_ratio, valid_ratio):
        """按用户划分训练/验证/测试集（使用固定排序保证一致性）"""
        all_users = sorted(self.user_sequences.keys())  # 排序保证一致性
        # 使用独立的RandomState避免影响全局状态
        rng = np.random.RandomState(2020)
        rng.shuffle(all_users)
        
        n_total = len(all_users)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)
        # 确保 test 集包含所有剩余用户，避免边界问题
        # train: [0, n_train), valid: [n_train, n_train+n_valid), test: [n_train+n_valid, end]
        
        if self.split == 'train':
            self.active_users = all_users[:n_train]
        elif self.split == 'valid':
            self.active_users = all_users[n_train:n_train + n_valid]
        elif self.split == 'test':
            self.active_users = all_users[n_train + n_valid:]
        else:
            raise ValueError(f"无效的 split: {self.split}")
    
    def _build_samples(self):
        """构建训练样本，包含丰富特征"""
        samples = []
        
        for user_id in self.active_users:
            seq = self.user_sequences[user_id]
            
            if len(seq) < 2:
                continue
            
            items = [s['item_id'] for s in seq]
            timestamps = [s['timestamp'] for s in seq]
            
            # 获取用户特征
            user_feat = self.feature_processor.get_user_features(user_id)
            
            for i in range(1, len(items)):
                history = items[:i]
                history_ts = timestamps[:i]
                
                if len(history) > self.max_seq_length:
                    history = history[-self.max_seq_length:]
                    history_ts = history_ts[-self.max_seq_length:]
                
                positive_item = items[i]
                
                # 正样本
                samples.append({
                    'user_id': user_id,
                    'history': history,
                    'history_timestamps': history_ts,
                    'target_item': positive_item,
                    'timestamp': timestamps[i],
                    'label': 1,
                    **user_feat
                })
                
                # 负样本（从已存在的物品集合中采样，避免ml-1m中item_id不连续的问题）
                user_items = set(items)
                # 负样本采样（带最大尝试次数，防止极端情况下无限循环）
                max_tries = 100
                negative_item = np.random.choice(self._all_items_list)
                tries = 0
                while negative_item in user_items and tries < max_tries:
                    negative_item = np.random.choice(self._all_items_list)
                    tries += 1
                # 如果达到最大尝试次数仍未找到，使用随机 item_id（罕见情况）
                if tries >= max_tries:
                    negative_item = np.random.randint(1, self.num_items + 1)
                
                samples.append({
                    'user_id': user_id,
                    'history': history,
                    'history_timestamps': history_ts,
                    'target_item': negative_item,
                    'timestamp': timestamps[i],
                    'label': 0,
                    **user_feat
                })
        
        # 只在主进程或非DDP环境打印
        should_print = not dist.is_initialized() or dist.get_rank() == 0
        if should_print:
            print(f"  {self.split} 样本数: {len(samples)}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        history = sample['history']
        seq_len = len(history)
        
        # 获取历史物品特征
        history_genres = []
        history_years = []
        for item_id in history:
            item_feat = self.feature_processor.get_item_features(item_id)
            history_genres.append(item_feat['primary_genre'])
            history_years.append(item_feat['year_bucket'])
        
        # Padding
        if len(history) < self.max_seq_length:
            padding_len = self.max_seq_length - len(history)
            history = [0] * padding_len + history
            history_genres = [0] * padding_len + history_genres
            history_years = [0] * padding_len + history_years
            mask = [0] * padding_len + [1] * seq_len
        else:
            mask = [1] * self.max_seq_length
        
        # 目标物品特征
        target_item = sample['target_item']
        target_feat = self.feature_processor.get_item_features(target_item)
        
        # 时间特征
        time_feat = self.interaction_extractor.get_time_features(sample['timestamp'])
        
        # 用户统计特征
        user_activity = self.interaction_extractor.get_user_activity(sample['user_id'])
        item_popularity = self.interaction_extractor.get_item_popularity(target_item)
        
        return {
            # 基础 ID
            'user_id': torch.tensor(sample['user_id'], dtype=torch.long),
            'item_seq': torch.tensor(history, dtype=torch.long),
            'item_seq_mask': torch.tensor(mask, dtype=torch.float),
            'target_item': torch.tensor(target_item, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.float),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            
            # 用户特征
            'user_age': torch.tensor(sample['age_bucket'], dtype=torch.long),
            'user_gender': torch.tensor(sample['gender'], dtype=torch.long),
            'user_occupation': torch.tensor(sample['occupation'], dtype=torch.long),
            'user_activity': torch.tensor(user_activity, dtype=torch.long),
            
            # 物品特征
            'item_genre': torch.tensor(target_feat['primary_genre'], dtype=torch.long),
            'item_year': torch.tensor(target_feat['year_bucket'], dtype=torch.long),
            'item_popularity': torch.tensor(item_popularity, dtype=torch.long),
            
            # 历史序列特征
            'history_genres': torch.tensor(history_genres, dtype=torch.long),
            'history_years': torch.tensor(history_years, dtype=torch.long),
            
            # 时间特征
            'time_hour': torch.tensor(time_feat['hour_bucket'], dtype=torch.long),
            'time_dow': torch.tensor(time_feat['day_of_week'], dtype=torch.long),
            'time_weekend': torch.tensor(time_feat['is_weekend'], dtype=torch.long),
        }


def get_rich_dataloaders(
    data_dir='./data',
    dataset_name='ml-100k',
    max_seq_length=50,
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
):
    """
    获取带丰富特征的数据加载器
    
    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称
        max_seq_length: 最大序列长度
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用 pinned memory
        prefetch_factor: 每个 worker 预加载的 batch 数（增加可减少 GPU 等待）
    
    Returns:
        train_loader, valid_loader, test_loader, dataset_info, feature_processor
    """
    
    # 共享特征处理器
    feature_processor = FeatureProcessor(data_dir, dataset_name)
    
    # 加载交互数据用于统计
    data_path = os.path.join(data_dir, dataset_name)
    if dataset_name == 'ml-100k':
        interactions = pd.read_csv(
            os.path.join(data_path, 'u.data'),
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
    else:
        interactions = pd.read_csv(
            os.path.join(data_path, 'ratings.dat'),
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
    
    interaction_extractor = InteractionFeatureExtractor(interactions)
    
    # 创建数据集
    train_dataset = RichFeatureDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        split='train',
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor
    )
    
    valid_dataset = RichFeatureDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        split='valid',
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor
    )
    
    test_dataset = RichFeatureDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        split='test',
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    # 特征维度信息
    feature_dims = feature_processor.get_feature_dims()
    feature_dims['num_users'] = train_dataset.num_users
    feature_dims['num_items'] = train_dataset.num_items
    feature_dims['time_hour'] = 7  # 0-6
    feature_dims['time_dow'] = 8   # 0-7
    feature_dims['time_weekend'] = 2
    feature_dims['user_activity'] = 6
    feature_dims['item_popularity'] = 6
    
    dataset_info = {
        'num_users': train_dataset.num_users,
        'num_items': train_dataset.num_items,
        'max_seq_length': max_seq_length,
        'feature_dims': feature_dims
    }
    
    return train_loader, valid_loader, test_loader, dataset_info, feature_processor


def get_topk_eval_data(
    data_dir='./data',
    dataset_name='ml-100k',
    max_seq_length=50,
    num_neg_samples=99
):
    """
    获取 Top-K 评估数据
    
    对于每个测试用户，返回：
    - 用户历史序列
    - ground_truth（最后一个交互物品）
    - 候选集（1个正样本 + num_neg_samples 个负样本）
    
    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称
        max_seq_length: 最大序列长度
        num_neg_samples: 负采样数量（默认 99，加上正样本共 100 个候选）
    
    Returns:
        eval_data: list of dict，每个用户的评估数据
        dataset_info: 数据集信息
        feature_processor: 特征处理器
    """
    # 加载交互数据
    data_path = os.path.join(data_dir, dataset_name)
    if dataset_name == 'ml-100k':
        interactions = pd.read_csv(
            os.path.join(data_path, 'u.data'),
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
    else:
        interactions = pd.read_csv(
            os.path.join(data_path, 'ratings.dat'),
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
    
    num_users = interactions['user_id'].max()
    num_items = interactions['item_id'].max()
    
    # 特征处理器
    feature_processor = FeatureProcessor(data_dir, dataset_name)
    interaction_extractor = InteractionFeatureExtractor(interactions)
    
    # 按用户构建序列
    user_sequences = defaultdict(list)
    sorted_data = interactions.sort_values(['user_id', 'timestamp'])
    
    for user_id, group in sorted_data.groupby('user_id', sort=False):
        items = group['item_id'].tolist()
        timestamps = group['timestamp'].tolist()
        user_sequences[user_id] = list(zip(items, timestamps))
    
    # 划分用户
    all_users = [u for u, seq in user_sequences.items() if len(seq) >= 5]
    np.random.seed(2020)
    np.random.shuffle(all_users)
    
    n_train = int(len(all_users) * 0.8)
    n_valid = int(len(all_users) * 0.1)
    test_users = all_users[n_train + n_valid:]
    
    # 所有物品集合（用于负采样）
    all_items = set(range(1, num_items + 1))
    
    eval_data = []
    
    for user_id in test_users:
        seq = user_sequences[user_id]
        if len(seq) < 2:
            continue
        
        # 最后一个作为 ground truth
        history = [s[0] for s in seq[:-1]]
        history_ts = [s[1] for s in seq[:-1]]
        ground_truth = seq[-1][0]
        test_timestamp = seq[-1][1]
        
        # 截断历史
        if len(history) > max_seq_length:
            history = history[-max_seq_length:]
            history_ts = history_ts[-max_seq_length:]
        
        # 负采样
        user_items = set([s[0] for s in seq])
        neg_candidates = list(all_items - user_items)
        if len(neg_candidates) < num_neg_samples:
            neg_samples = neg_candidates
        else:
            neg_samples = np.random.choice(neg_candidates, num_neg_samples, replace=False).tolist()
        
        # 候选集 = [ground_truth] + 负样本
        candidates = [ground_truth] + neg_samples
        
        # 用户特征
        user_feat = feature_processor.get_user_features(user_id)
        
        eval_data.append({
            'user_id': user_id,
            'history': history,
            'history_timestamps': history_ts,
            'ground_truth': ground_truth,
            'candidates': candidates,
            'timestamp': test_timestamp,
            **user_feat
        })
    
    # 特征维度信息
    feature_dims = feature_processor.get_feature_dims()
    feature_dims['num_users'] = num_users
    feature_dims['num_items'] = num_items
    
    dataset_info = {
        'num_users': num_users,
        'num_items': num_items,
        'max_seq_length': max_seq_length,
        'feature_dims': feature_dims,
        'num_test_users': len(eval_data)
    }
    
    print(f"Top-K 评估数据: {len(eval_data)} 个测试用户")
    
    return eval_data, dataset_info, feature_processor, interaction_extractor


def build_topk_batch_multi(eval_item, feature_processor, interaction_extractor, max_seq_length, device='cpu'):
    """
    为单个用户构建 Top-K 评估的 batch（优化版，预计算特征）
    
    Args:
        eval_item: 单个用户的评估数据
        feature_processor: 特征处理器
        interaction_extractor: 交互特征提取器
        max_seq_length: 最大序列长度
        device: 设备
    
    Returns:
        batch: 模型输入 batch（候选物品数 × 特征）
    """
    user_id = eval_item['user_id']
    history = eval_item['history']
    candidates = eval_item['candidates']
    num_candidates = len(candidates)
    
    seq_len = len(history)
    
    # 获取历史物品特征
    history_genres = []
    history_years = []
    for item_id in history:
        item_feat = feature_processor.get_item_features(item_id)
        history_genres.append(item_feat['primary_genre'])
        history_years.append(item_feat['year_bucket'])
    
    # Padding
    if len(history) < max_seq_length:
        padding_len = max_seq_length - len(history)
        history_padded = [0] * padding_len + history
        history_genres = [0] * padding_len + history_genres
        history_years = [0] * padding_len + history_years
        mask = [0] * padding_len + [1] * seq_len
    else:
        history_padded = history
        mask = [1] * max_seq_length
    
    # 扩展到候选数量
    batch = {
        'user_id': torch.tensor([user_id] * num_candidates, dtype=torch.long, device=device),
        'item_seq': torch.tensor([history_padded] * num_candidates, dtype=torch.long, device=device),
        'item_seq_mask': torch.tensor([mask] * num_candidates, dtype=torch.float, device=device),
        'target_item': torch.tensor(candidates, dtype=torch.long, device=device),
        'seq_len': torch.tensor([seq_len] * num_candidates, dtype=torch.long, device=device),
        
        # 用户特征
        'user_age': torch.tensor([eval_item['age_bucket']] * num_candidates, dtype=torch.long, device=device),
        'user_gender': torch.tensor([eval_item['gender']] * num_candidates, dtype=torch.long, device=device),
        'user_occupation': torch.tensor([eval_item['occupation']] * num_candidates, dtype=torch.long, device=device),
        'user_activity': torch.tensor([interaction_extractor.get_user_activity(user_id)] * num_candidates, dtype=torch.long, device=device),
        
        # 历史序列特征
        'history_genres': torch.tensor([history_genres] * num_candidates, dtype=torch.long, device=device),
        'history_years': torch.tensor([history_years] * num_candidates, dtype=torch.long, device=device),
    }
    
    # 每个候选物品的特征
    item_genres = []
    item_years = []
    item_pops = []
    time_feats = []
    
    for item_id in candidates:
        item_feat = feature_processor.get_item_features(item_id)
        item_genres.append(item_feat['primary_genre'])
        item_years.append(item_feat['year_bucket'])
        item_pops.append(interaction_extractor.get_item_popularity(item_id))
        time_feats.append(interaction_extractor.get_time_features(eval_item['timestamp']))
    
    batch['item_genre'] = torch.tensor(item_genres, dtype=torch.long, device=device)
    batch['item_year'] = torch.tensor(item_years, dtype=torch.long, device=device)
    batch['item_popularity'] = torch.tensor(item_pops, dtype=torch.long, device=device)
    batch['time_hour'] = torch.tensor([t['hour_bucket'] for t in time_feats], dtype=torch.long, device=device)
    batch['time_dow'] = torch.tensor([t['day_of_week'] for t in time_feats], dtype=torch.long, device=device)
    batch['time_weekend'] = torch.tensor([t['is_weekend'] for t in time_feats], dtype=torch.long, device=device)
    
    return batch


# 保持向后兼容
build_topk_batch = build_topk_batch_multi


if __name__ == "__main__":
    print("测试增强版数据加载器...")
    
    train_loader, valid_loader, test_loader, info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name='ml-100k',
        max_seq_length=50,
        batch_size=32
    )
    
    print("\n数据集信息:")
    print(info)
    
    print("\n示例 batch:")
    batch = next(iter(train_loader))
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    print("\n测试 Top-K 评估数据...")
    eval_data, eval_info, fp2, ie = get_topk_eval_data(
        data_dir='./data',
        dataset_name='ml-100k',
        max_seq_length=50
    )
    print(f"测试用户数: {len(eval_data)}")
    print(f"示例: user_id={eval_data[0]['user_id']}, candidates={len(eval_data[0]['candidates'])}")
