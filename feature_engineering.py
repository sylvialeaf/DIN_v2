#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征工程模块

为 DIN 模型提供丰富的特征支持：
- 用户特征：年龄、性别、职业
- 物品特征：类目（19个genre）、发布年份
- 交互特征：时间衰减、周内/日内模式
- 统计特征：用户活跃度、物品热度

MovieLens 100k 数据文件：
- u.data: user_id | item_id | rating | timestamp
- u.user: user_id | age | gender | occupation | zip_code
- u.item: movie_id | title | release_date | ... | 19 genres
- u.genre: genre_id | genre_name
- u.occupation: occupation_name
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


# ========================================
# 常量定义
# ========================================

# MovieLens 100k 的 19 个电影类型
GENRES = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
    'Sci-Fi', 'Thriller', 'War', 'Western'
]

# MovieLens 100k 的 21 种职业
OCCUPATIONS = [
    'administrator', 'artist', 'doctor', 'educator', 'engineer',
    'entertainment', 'executive', 'healthcare', 'homemaker', 'lawyer',
    'librarian', 'marketing', 'none', 'other', 'programmer',
    'retired', 'salesman', 'scientist', 'student', 'technician', 'writer'
]

# 年龄分桶
AGE_BINS = [0, 18, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']


class FeatureProcessor:
    """
    特征处理器
    
    负责加载和处理 MovieLens 的所有特征。
    """
    
    def __init__(self, data_dir, dataset_name='ml-100k'):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.data_path = os.path.join(data_dir, dataset_name)
        
        # 特征信息
        self.feature_dims = {}
        self.feature_offsets = {}
        
        # 确保数据已下载
        self._download_if_needed()
        
        # 加载所有特征
        self._load_all_features()
    
    def _download_if_needed(self):
        """如果数据不存在，自动下载（DDP安全）"""
        import urllib.request
        import zipfile
        try:
            import torch.distributed as dist
        except ImportError:
            dist = None
        
        # 检查关键文件是否存在
        if self.dataset_name == 'ml-100k':
            key_file = os.path.join(self.data_path, 'u.data')
        elif self.dataset_name == 'ml-1m':
            key_file = os.path.join(self.data_path, 'ratings.dat')
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        # 如果文件存在，直接返回（无需打印）
        if os.path.exists(key_file):
            return
        
        # DDP: 只在非分布式或 rank 0 下载
        if dist and dist.is_initialized():
            if dist.get_rank() != 0:
                # 非 rank 0 进程等待
                dist.barrier()
                return
            else:
                # Rank 0 执行下载
                print(f"[FeatureProcessor Rank 0] 数据不存在，开始下载...")
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        if self.dataset_name == 'ml-100k':
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        elif self.dataset_name == 'ml-1m':
            url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        print(f"下载数据集 {self.dataset_name}...")
        zip_path = os.path.join(self.data_dir, f'{self.dataset_name}.zip')
        urllib.request.urlretrieve(url, zip_path)
        
        print("解压数据...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        os.remove(zip_path)
        print("数据准备完成!")
        
        # 同步
        if dist and dist.is_initialized():
            dist.barrier()
    
    def _load_all_features(self):
        """加载所有特征数据"""
        if self.dataset_name == 'ml-100k':
            self._load_ml100k_features()
        elif self.dataset_name == 'ml-1m':
            self._load_ml1m_features()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
    
    def _load_ml100k_features(self):
        """加载 ML-100K 的特征"""
        
        # ========================================
        # 1. 加载用户特征
        # ========================================
        user_file = os.path.join(self.data_path, 'u.user')
        self.user_df = pd.read_csv(
            user_file,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        
        # 处理用户特征
        # 年龄分桶
        self.user_df['age_bucket'] = pd.cut(
            self.user_df['age'], 
            bins=AGE_BINS, 
            labels=AGE_LABELS, 
            right=False
        ).cat.codes + 1  # 从1开始，0留给padding
        
        # 性别编码: M=1, F=2
        self.user_df['gender_encoded'] = self.user_df['gender'].map({'M': 1, 'F': 2})
        
        # 职业编码
        occupation_map = {occ: i+1 for i, occ in enumerate(OCCUPATIONS)}
        self.user_df['occupation_encoded'] = self.user_df['occupation'].map(occupation_map).fillna(0).astype(int)
        
        # ========================================
        # 2. 加载物品特征
        # ========================================
        item_file = os.path.join(self.data_path, 'u.item')
        
        # 列名：movie_id | title | release_date | video_release | IMDb URL | 19 genres
        item_columns = ['item_id', 'title', 'release_date', 'video_release', 'imdb_url'] + GENRES
        
        self.item_df = pd.read_csv(
            item_file,
            sep='|',
            names=item_columns,
            encoding='latin-1',
            engine='python'
        )
        
        # 处理发布年份（支持多种日期格式）
        def extract_year(date_str):
            if pd.isna(date_str):
                return 0
            try:
                # 尝试多种格式: "01-Jan-1995", "1995-01-01", "1995"
                import re
                # 先尝试匹配4位年份
                match = re.search(r'(19|20)\d{2}', str(date_str))
                if match:
                    return int(match.group())
                # 回退到原逻辑
                return int(date_str.split('-')[-1])
            except:
                return 0
        
        self.item_df['release_year'] = self.item_df['release_date'].apply(extract_year)
        
        # 年份分桶: 1920-2000, 每10年一个桶
        def year_bucket(year):
            if year == 0:
                return 0
            elif year < 1950:
                return 1
            elif year < 1960:
                return 2
            elif year < 1970:
                return 3
            elif year < 1980:
                return 4
            elif year < 1990:
                return 5
            else:
                return 6
        
        self.item_df['year_bucket'] = self.item_df['release_year'].apply(year_bucket)
        
        # 主类型（选取第一个为1的genre）
        def get_primary_genre(row):
            for i, genre in enumerate(GENRES):
                if row[genre] == 1:
                    return i + 1  # 从1开始
            return 0
        
        self.item_df['primary_genre'] = self.item_df.apply(get_primary_genre, axis=1)
        
        # 类型数量（一个电影可能有多个类型）
        self.item_df['genre_count'] = self.item_df[GENRES].sum(axis=1)
        
        # ========================================
        # 3. 构建特征查找表
        # ========================================
        self._build_feature_lookups()
        
        # 打印特征信息（只在主进程或非DDP环境）
        try:
            import torch.distributed as dist
            should_print = not dist.is_initialized() or dist.get_rank() == 0
        except:
            should_print = True
        
        if should_print:
            print(f"特征工程完成!")
            print(f"  用户数: {len(self.user_df)}")
            print(f"  物品数: {len(self.item_df)}")
            print(f"  年龄桶数: {len(AGE_LABELS) + 1}")
            print(f"  职业数: {len(OCCUPATIONS) + 1}")
            print(f"  类型数: {len(GENRES) + 1}")
    
    def _load_ml1m_features(self):
        """加载 ML-1M 的特征"""
        
        # 用户特征
        user_file = os.path.join(self.data_path, 'users.dat')
        self.user_df = pd.read_csv(
            user_file,
            sep='::',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            engine='python'
        )
        
        # ML-1M 的年龄已经是分桶的
        # 1: Under 18, 18: 18-24, 25: 25-34, 35: 35-44, 45: 45-49, 50: 50-55, 56: 56+
        age_map = {1: 1, 18: 2, 25: 3, 35: 4, 45: 5, 50: 6, 56: 7}
        self.user_df['age_bucket'] = self.user_df['age'].map(age_map).fillna(0).astype(int)
        
        # 性别
        self.user_df['gender_encoded'] = self.user_df['gender'].map({'M': 1, 'F': 2})
        
        # 职业 (0-20)
        self.user_df['occupation_encoded'] = self.user_df['occupation'] + 1
        
        # 物品特征
        movie_file = os.path.join(self.data_path, 'movies.dat')
        self.item_df = pd.read_csv(
            movie_file,
            sep='::',
            names=['item_id', 'title', 'genres'],
            encoding='latin-1',
            engine='python'
        )
        
        # 提取年份
        def extract_year_from_title(title):
            try:
                year = int(title[-5:-1])
                if 1900 <= year <= 2020:
                    return year
            except:
                pass
            return 0
        
        self.item_df['release_year'] = self.item_df['title'].apply(extract_year_from_title)
        
        # 年份分桶
        def year_bucket(year):
            if year == 0:
                return 0
            elif year < 1950:
                return 1
            elif year < 1960:
                return 2
            elif year < 1970:
                return 3
            elif year < 1980:
                return 4
            elif year < 1990:
                return 5
            else:
                return 6
        
        self.item_df['year_bucket'] = self.item_df['release_year'].apply(year_bucket)
        
        # 解析 genres
        all_genres = set()
        for genres in self.item_df['genres']:
            for g in genres.split('|'):
                all_genres.add(g)
        self.genre_list = sorted(list(all_genres))
        genre_map = {g: i+1 for i, g in enumerate(self.genre_list)}
        
        def get_primary_genre(genres_str):
            genres = genres_str.split('|')
            return genre_map.get(genres[0], 0) if genres else 0
        
        self.item_df['primary_genre'] = self.item_df['genres'].apply(get_primary_genre)
        self.item_df['genre_count'] = self.item_df['genres'].apply(lambda x: len(x.split('|')))
        
        self._build_feature_lookups()
        
        # 打印特征信息（只在主进程或非DDP环境）
        try:
            import torch.distributed as dist
            should_print = not dist.is_initialized() or dist.get_rank() == 0
        except:
            should_print = True
        
        if should_print:
            print(f"特征工程完成!")
            print(f"  用户数: {len(self.user_df)}")
            print(f"  物品数: {len(self.item_df)}")
    
    def _build_feature_lookups(self):
        """构建特征查找表"""
        
        # 用户特征查找表
        self.user_features = {}
        for _, row in self.user_df.iterrows():
            self.user_features[row['user_id']] = {
                'age_bucket': int(row['age_bucket']),
                'gender': int(row['gender_encoded']),
                'occupation': int(row['occupation_encoded'])
            }
        
        # 物品特征查找表
        self.item_features = {}
        for _, row in self.item_df.iterrows():
            item_feat = {
                'year_bucket': int(row['year_bucket']),
                'primary_genre': int(row['primary_genre']),
                'genre_count': int(row['genre_count'])
            }
            
            # 对于 ml-100k，添加多标签 genres
            if self.dataset_name == 'ml-100k':
                item_feat['genres'] = [int(row[g]) for g in GENRES]
            
            self.item_features[row['item_id']] = item_feat
        
        # 设置特征维度
        # ml-1m 使用实际解析出的 genre 数量，而非硬编码
        if self.dataset_name == 'ml-100k':
            genre_dim = len(GENRES) + 2
        else:
            # ml-1m: 使用 self.genre_list 的实际长度
            genre_dim = len(getattr(self, 'genre_list', [])) + 2
        
        self.feature_dims = {
            'user_id': len(self.user_df) + 1,
            'item_id': len(self.item_df) + 1,
            'age_bucket': len(AGE_LABELS) + 2,
            'gender': 3,  # 0=padding, 1=M, 2=F
            'occupation': len(OCCUPATIONS) + 2,
            'year_bucket': 8,  # 0-7
            'primary_genre': genre_dim,
        }
    
    def get_user_features(self, user_id):
        """获取用户特征"""
        if user_id in self.user_features:
            return self.user_features[user_id]
        return {'age_bucket': 0, 'gender': 0, 'occupation': 0}
    
    def get_item_features(self, item_id):
        """获取物品特征"""
        if item_id in self.item_features:
            return self.item_features[item_id]
        return {'year_bucket': 0, 'primary_genre': 0, 'genre_count': 0}
    
    def get_feature_dims(self):
        """获取所有特征的维度信息"""
        return self.feature_dims


class InteractionFeatureExtractor:
    """
    交互特征提取器
    
    从交互数据中提取：
    - 时间特征
    - 行为序列特征
    - 统计特征
    """
    
    def __init__(self, interactions_df):
        """
        Args:
            interactions_df: 包含 user_id, item_id, timestamp 的 DataFrame
        """
        self.interactions = interactions_df
        self._compute_statistics()
    
    def _compute_statistics(self):
        """计算统计特征"""
        
        # 用户活跃度（交互次数）
        user_counts = self.interactions.groupby('user_id').size()
        self.user_activity = {
            uid: self._activity_bucket(count) 
            for uid, count in user_counts.items()
        }
        
        # 物品热度（被交互次数）
        item_counts = self.interactions.groupby('item_id').size()
        self.item_popularity = {
            iid: self._popularity_bucket(count) 
            for iid, count in item_counts.items()
        }
        
        # 时间范围
        self.min_timestamp = self.interactions['timestamp'].min()
        self.max_timestamp = self.interactions['timestamp'].max()
    
    def _activity_bucket(self, count):
        """用户活跃度分桶"""
        if count < 20:
            return 1
        elif count < 50:
            return 2
        elif count < 100:
            return 3
        elif count < 200:
            return 4
        else:
            return 5
    
    def _popularity_bucket(self, count):
        """物品热度分桶"""
        if count < 5:
            return 1
        elif count < 20:
            return 2
        elif count < 50:
            return 3
        elif count < 100:
            return 4
        else:
            return 5
    
    def get_time_features(self, timestamp):
        """
        提取时间特征
        
        Returns:
            dict with:
                - hour_bucket: 0-5 (4小时一个桶)
                - day_of_week: 1-7
                - is_weekend: 0/1
                - time_position: 归一化的时间位置 (0-1)
        """
        try:
            dt = datetime.fromtimestamp(timestamp)
            hour_bucket = dt.hour // 4 + 1  # 1-6
            day_of_week = dt.weekday() + 1  # 1-7
            is_weekend = 1 if dt.weekday() >= 5 else 0
            
            # 归一化时间位置
            time_range = self.max_timestamp - self.min_timestamp
            if time_range > 0:
                time_position = (timestamp - self.min_timestamp) / time_range
            else:
                time_position = 0.5
            
            return {
                'hour_bucket': hour_bucket,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'time_position': time_position
            }
        except:
            return {
                'hour_bucket': 0,
                'day_of_week': 0,
                'is_weekend': 0,
                'time_position': 0.5
            }
    
    def get_user_activity(self, user_id):
        """获取用户活跃度"""
        return self.user_activity.get(user_id, 0)
    
    def get_item_popularity(self, item_id):
        """获取物品热度"""
        return self.item_popularity.get(item_id, 0)
    
    def compute_sequence_features(self, item_sequence, timestamps=None):
        """
        计算序列特征
        
        Args:
            item_sequence: 物品ID序列
            timestamps: 对应的时间戳序列
            
        Returns:
            dict with:
                - seq_length: 序列长度
                - unique_items: 唯一物品数
                - avg_popularity: 平均物品热度
                - recency_weight: 时间权重（越近越大）
        """
        seq_length = len(item_sequence)
        unique_items = len(set(item_sequence))
        
        # 平均热度
        popularities = [self.item_popularity.get(iid, 0) for iid in item_sequence]
        avg_popularity = np.mean(popularities) if popularities else 0
        
        # 时间权重（指数衰减）
        if timestamps and len(timestamps) > 1:
            time_weights = []
            max_ts = max(timestamps)
            for ts in timestamps:
                weight = np.exp(-0.1 * (max_ts - ts) / (24 * 3600))  # 天为单位的衰减
                time_weights.append(weight)
            recency_weight = np.mean(time_weights)
        else:
            recency_weight = 0.5
        
        return {
            'seq_length': seq_length,
            'unique_items': unique_items,
            'avg_popularity': avg_popularity,
            'recency_weight': recency_weight
        }


def prepare_lightgbm_features(
    interactions_df,
    feature_processor,
    interaction_extractor,
    max_seq_length=50
):
    """
    为 LightGBM 准备扁平化特征
    
    将用户、物品、交互特征组合成一个特征矩阵。
    
    Returns:
        X: 特征矩阵
        y: 标签
        feature_names: 特征名列表
    """
    
    features = []
    labels = []
    
    # 按用户分组
    user_groups = interactions_df.groupby('user_id')
    
    for user_id, group in user_groups:
        group = group.sort_values('timestamp')
        items = group['item_id'].tolist()
        timestamps = group['timestamp'].tolist()
        ratings = group['rating'].tolist()
        
        if len(items) < 5:
            continue
        
        # 获取用户特征
        user_feat = feature_processor.get_user_features(user_id)
        user_activity = interaction_extractor.get_user_activity(user_id)
        
        # 为每个交互构建样本
        for i in range(1, len(items)):
            history = items[:i][-max_seq_length:]
            history_ts = timestamps[:i][-max_seq_length:]
            
            target_item = items[i]
            target_rating = ratings[i]
            label = 1 if target_rating >= 4 else 0
            
            # 物品特征
            item_feat = feature_processor.get_item_features(target_item)
            item_pop = interaction_extractor.get_item_popularity(target_item)
            
            # 序列特征
            seq_feat = interaction_extractor.compute_sequence_features(history, history_ts)
            
            # 时间特征
            time_feat = interaction_extractor.get_time_features(timestamps[i])
            
            # 历史物品的统计特征
            hist_item_feats = [feature_processor.get_item_features(iid) for iid in history]
            hist_genres = [f['primary_genre'] for f in hist_item_feats]
            hist_years = [f['year_bucket'] for f in hist_item_feats]
            
            # 构建特征向量
            sample = [
                # 用户特征
                user_feat['age_bucket'],
                user_feat['gender'],
                user_feat['occupation'],
                user_activity,
                
                # 物品特征
                item_feat['year_bucket'],
                item_feat['primary_genre'],
                item_feat['genre_count'],
                item_pop,
                
                # 序列特征
                seq_feat['seq_length'],
                seq_feat['unique_items'],
                seq_feat['avg_popularity'],
                seq_feat['recency_weight'],
                
                # 时间特征
                time_feat['hour_bucket'],
                time_feat['day_of_week'],
                time_feat['is_weekend'],
                time_feat['time_position'],
                
                # 交叉特征
                1 if item_feat['primary_genre'] in hist_genres else 0,  # 类型匹配
                1 if item_feat['year_bucket'] in hist_years else 0,     # 年代匹配
                
                # 历史统计
                np.mean(hist_genres) if hist_genres else 0,
                np.std(hist_genres) if len(hist_genres) > 1 else 0,
            ]
            
            features.append(sample)
            labels.append(label)
            
            # 负样本
            neg_item = np.random.randint(1, len(feature_processor.item_features) + 1)
            while neg_item in set(items):
                neg_item = np.random.randint(1, len(feature_processor.item_features) + 1)
            
            neg_item_feat = feature_processor.get_item_features(neg_item)
            neg_item_pop = interaction_extractor.get_item_popularity(neg_item)
            
            neg_sample = [
                user_feat['age_bucket'],
                user_feat['gender'],
                user_feat['occupation'],
                user_activity,
                neg_item_feat['year_bucket'],
                neg_item_feat['primary_genre'],
                neg_item_feat['genre_count'],
                neg_item_pop,
                seq_feat['seq_length'],
                seq_feat['unique_items'],
                seq_feat['avg_popularity'],
                seq_feat['recency_weight'],
                time_feat['hour_bucket'],
                time_feat['day_of_week'],
                time_feat['is_weekend'],
                time_feat['time_position'],
                1 if neg_item_feat['primary_genre'] in hist_genres else 0,
                1 if neg_item_feat['year_bucket'] in hist_years else 0,
                np.mean(hist_genres) if hist_genres else 0,
                np.std(hist_genres) if len(hist_genres) > 1 else 0,
            ]
            
            features.append(neg_sample)
            labels.append(0)
    
    feature_names = [
        'user_age', 'user_gender', 'user_occupation', 'user_activity',
        'item_year', 'item_genre', 'item_genre_count', 'item_popularity',
        'seq_length', 'seq_unique_items', 'seq_avg_popularity', 'seq_recency',
        'time_hour', 'time_dow', 'time_weekend', 'time_position',
        'genre_match', 'year_match',
        'hist_genre_mean', 'hist_genre_std'
    ]
    
    return np.array(features), np.array(labels), feature_names


if __name__ == "__main__":
    # 测试
    print("测试特征工程模块...")
    
    data_dir = './data'
    fp = FeatureProcessor(data_dir, 'ml-100k')
    
    print("\n用户特征示例:")
    print(fp.get_user_features(1))
    
    print("\n物品特征示例:")
    print(fp.get_item_features(1))
    
    print("\n特征维度:")
    print(fp.get_feature_dims())
