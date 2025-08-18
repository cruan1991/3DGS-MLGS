#!/usr/bin/env python3
"""
Student Network训练数据集创建脚本
=================================

基于预计算的邻居关系，创建COLMAP→3DGS的训练数据集
- 输入特征：COLMAP点的几何和空间特征
- 输出标签：对应高斯球的聚合特征
- 支持多尺度邻域聚合
- 数据增强和质量控制
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import h5py
from torch.utils.data import Dataset, DataLoader
import argparse
from dataclasses import dataclass
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    neighbor_data_dir: str = "batches"
    output_dir: str = "training_data"
    
    # 数据选择
    use_radii: List[float] = None  # None表示使用所有半径
    min_neighbors: int = 3  # 最少邻居数
    max_neighbors: int = 512  # 最大邻居数
    
    # 特征配置
    use_spatial_features: bool = True
    use_density_features: bool = True
    use_scale_features: bool = True
    
    # 数据分割
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 其他
    random_seed: int = 42
    
    def __post_init__(self):
        if self.use_radii is None:
            self.use_radii = [0.012, 0.039, 0.107, 0.273]

class NeighborDataLoader:
    """邻居数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.neighbor_files = {}
        self.colmap_data = None
        self.gaussian_data = None
        
    def load_base_data(self):
        """加载基础数据"""
        logger.info("加载基础COLMAP和高斯数据...")
        
        # 加载COLMAP数据
        colmap_file = self.data_dir / "colmap_points.pt"
        if colmap_file.exists():
            self.colmap_data = torch.load(colmap_file, map_location='cpu')
            logger.info(f"COLMAP数据: {self.colmap_data.shape}")
        else:
            raise FileNotFoundError(f"COLMAP数据文件不存在: {colmap_file}")
        
        # 加载高斯数据
        gaussian_file = self.data_dir / "gaussian_data.pt"
        if gaussian_file.exists():
            gaussian_dict = torch.load(gaussian_file, map_location='cpu')
            self.gaussian_data = {
                'xyz': gaussian_dict['xyz'],
                'scale': gaussian_dict['scale']
            }
            logger.info(f"高斯数据: xyz={self.gaussian_data['xyz'].shape}, scale={self.gaussian_data['scale'].shape}")
        else:
            raise FileNotFoundError(f"高斯数据文件不存在: {gaussian_file}")
    
    def load_neighbor_data(self, radii: List[float]):
        """加载邻居数据"""
        logger.info(f"加载邻居数据，半径: {radii}")
        
        for radius in radii:
            # 尝试快速版本和完整版本
            fast_file = self.data_dir / f"neigh_r{radius:.6f}_fast.pt"
            full_file = self.data_dir / f"neigh_r{radius:.6f}.pt"
            
            if fast_file.exists():
                neighbor_file = fast_file
                logger.info(f"使用快速版本: {neighbor_file}")
            elif full_file.exists():
                neighbor_file = full_file
                logger.info(f"使用完整版本: {neighbor_file}")
            else:
                logger.warning(f"未找到半径 {radius} 的邻居数据，跳过")
                continue
            
            neighbor_data = torch.load(neighbor_file, map_location='cpu')
            self.neighbor_files[radius] = neighbor_data
            
            logger.info(f"  半径 {radius}: {len(neighbor_data['indices'])} 条邻接边")
        
        logger.info(f"成功加载 {len(self.neighbor_files)} 个半径的邻居数据")

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, colmap_data: torch.Tensor, gaussian_data: Dict[str, torch.Tensor]):
        self.colmap_xyz = colmap_data.float()
        self.gaussian_xyz = gaussian_data['xyz'].float()
        self.gaussian_scale = gaussian_data['scale'].float()
        
        # 预计算统计信息
        self._compute_global_stats()
    
    def _compute_global_stats(self):
        """计算全局统计信息用于特征归一化"""
        # COLMAP点的统计
        self.colmap_mean = self.colmap_xyz.mean(dim=0)
        self.colmap_std = self.colmap_xyz.std(dim=0)
        
        # 高斯球的统计  
        self.gaussian_mean = self.gaussian_xyz.mean(dim=0)
        self.gaussian_std = self.gaussian_xyz.std(dim=0)
        
        # 尺度统计
        scale_magnitudes = torch.norm(self.gaussian_scale, dim=1)
        self.scale_mean = scale_magnitudes.mean()
        self.scale_std = scale_magnitudes.std()
        
        logger.info(f"全局统计信息已计算")
    
    def extract_colmap_features(self, point_indices: torch.Tensor) -> torch.Tensor:
        """提取COLMAP点特征"""
        points = self.colmap_xyz[point_indices]  # [N, 3]
        
        features = []
        
        # 1. 归一化坐标
        normalized_coords = (points - self.colmap_mean) / (self.colmap_std + 1e-8)
        features.append(normalized_coords)
        
        # 2. 空间特征：距离原点
        distances = torch.norm(points, dim=1, keepdim=True)
        features.append(distances)
        
        # 3. 局部密度估计（基于最近邻）
        if len(point_indices) > 1:
            # 计算到最近邻的距离
            pairwise_dist = torch.cdist(points, points)
            # 排除自身（对角线）
            pairwise_dist.fill_diagonal_(float('inf'))
            nearest_dist = pairwise_dist.min(dim=1)[0].unsqueeze(1)
            density_features = 1.0 / (nearest_dist + 1e-6)  # 密度倒数
        else:
            density_features = torch.zeros(len(points), 1)
        
        features.append(density_features)
        
        return torch.cat(features, dim=1)  # [N, feature_dim]
    
    def aggregate_gaussian_features(self, neighbor_indices: torch.Tensor, 
                                  row_splits: torch.Tensor) -> torch.Tensor:
        """聚合高斯球特征作为标签"""
        N = len(row_splits) - 1
        features = []
        
        for i in range(N):
            start = row_splits[i].item()
            end = row_splits[i + 1].item()
            
            if start < end:
                # 获取邻居索引
                neighbor_idx = neighbor_indices[start:end]
                neighbor_xyz = self.gaussian_xyz[neighbor_idx]  # [K, 3]
                neighbor_scale = self.gaussian_scale[neighbor_idx]  # [K, 3]
                
                # 聚合特征
                point_features = self._aggregate_single_point(neighbor_xyz, neighbor_scale)
            else:
                # 没有邻居的点，使用默认值
                point_features = torch.zeros(self._get_output_feature_dim())
            
            features.append(point_features)
        
        return torch.stack(features, dim=0)  # [N, output_dim]
    
    def _aggregate_single_point(self, neighbor_xyz: torch.Tensor, 
                               neighbor_scale: torch.Tensor) -> torch.Tensor:
        """聚合单个点的邻居特征"""
        features = []
        
        # 1. 位置特征：质心
        centroid = neighbor_xyz.mean(dim=0)
        features.append(centroid)
        
        # 2. 位置分布：标准差
        position_std = neighbor_xyz.std(dim=0)
        features.append(position_std)
        
        # 3. 尺度特征：平均尺度
        mean_scale = neighbor_scale.mean(dim=0)
        features.append(mean_scale)
        
        # 4. 尺度分布：标准差
        scale_std = neighbor_scale.std(dim=0)
        features.append(scale_std)
        
        # 5. 邻居数量（对数）
        num_neighbors = torch.log(torch.tensor(len(neighbor_xyz)) + 1.0)
        features.append(num_neighbors.unsqueeze(0))
        
        # 6. 尺度大小统计
        scale_magnitudes = torch.norm(neighbor_scale, dim=1)
        scale_mean = scale_magnitudes.mean().unsqueeze(0)
        scale_max = scale_magnitudes.max().unsqueeze(0)
        scale_min = scale_magnitudes.min().unsqueeze(0)
        features.extend([scale_mean, scale_max, scale_min])
        
        return torch.cat(features, dim=0)
    
    def _get_output_feature_dim(self) -> int:
        """获取输出特征维度"""
        # centroid(3) + position_std(3) + mean_scale(3) + scale_std(3) + 
        # num_neighbors(1) + scale_mean(1) + scale_max(1) + scale_min(1)
        return 16

class TrainingDatasetCreator:
    """训练数据集创建器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_loader = NeighborDataLoader(config.neighbor_data_dir)
        self.feature_extractor = None
        
    def create_dataset(self):
        """创建完整的训练数据集"""
        logger.info("🚀 开始创建训练数据集...")
        
        # 1. 加载数据
        self.data_loader.load_base_data()
        self.data_loader.load_neighbor_data(self.config.use_radii)
        
        # 2. 初始化特征提取器
        self.feature_extractor = FeatureExtractor(
            self.data_loader.colmap_data,
            self.data_loader.gaussian_data
        )
        
        # 3. 处理每个半径
        all_samples = []
        
        for radius in self.config.use_radii:
            if radius not in self.data_loader.neighbor_files:
                logger.warning(f"跳过半径 {radius}，数据不存在")
                continue
            
            logger.info(f"处理半径 {radius}...")
            samples = self._process_radius_data(radius)
            all_samples.extend(samples)
            logger.info(f"  生成 {len(samples)} 个样本")
        
        logger.info(f"总共生成 {len(all_samples)} 个训练样本")
        
        # 4. 数据分割和保存
        self._split_and_save_dataset(all_samples)
        
        logger.info("✅ 训练数据集创建完成！")
    
    def _process_radius_data(self, radius: float) -> List[Dict]:
        """处理单个半径的数据"""
        neighbor_data = self.data_loader.neighbor_files[radius]
        
        indices = neighbor_data['indices']
        row_splits = neighbor_data['row_splits'] 
        count = neighbor_data['count']
        
        # 样本索引列表
        samples = []
        
        N = len(count)
        valid_indices = []
        
        # 筛选有效样本
        for i in range(N):
            neighbor_count = count[i].item()
            if self.config.min_neighbors <= neighbor_count <= self.config.max_neighbors:
                valid_indices.append(i)
        
        logger.info(f"  有效样本: {len(valid_indices)}/{N}")
        
        if len(valid_indices) == 0:
            return []
        
        # 批量提取特征
        valid_indices = torch.tensor(valid_indices)
        
        # 输入特征：COLMAP点特征
        input_features = self.feature_extractor.extract_colmap_features(valid_indices)
        
        # 输出标签：高斯球聚合特征
        output_features = self.feature_extractor.aggregate_gaussian_features(indices, row_splits)
        output_features = output_features[valid_indices]
        
        # 创建样本
        for i, idx in enumerate(valid_indices):
            sample = {
                'input_features': input_features[i].numpy(),
                'output_features': output_features[i].numpy(),
                'radius': radius,
                'colmap_index': idx.item(),
                'neighbor_count': count[idx].item(),
            }
            samples.append(sample)
        
        return samples
    
    def _split_and_save_dataset(self, all_samples: List[Dict]):
        """分割并保存数据集"""
        logger.info("分割并保存数据集...")
        
        # 设置随机种子
        np.random.seed(self.config.random_seed)
        
        # 打乱数据
        indices = np.random.permutation(len(all_samples))
        
        # 计算分割点
        n_total = len(all_samples)
        n_train = int(n_total * self.config.train_ratio)
        n_val = int(n_total * self.config.val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # 分割数据
        train_samples = [all_samples[i] for i in train_indices]
        val_samples = [all_samples[i] for i in val_indices]
        test_samples = [all_samples[i] for i in test_indices]
        
        logger.info(f"数据分割: 训练={len(train_samples)}, 验证={len(val_samples)}, 测试={len(test_samples)}")
        
        # 创建输出目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数据集
        self._save_split('train', train_samples, output_dir)
        self._save_split('val', val_samples, output_dir)
        self._save_split('test', test_samples, output_dir)
        
        # 保存配置和统计信息
        self._save_metadata(output_dir, len(train_samples), len(val_samples), len(test_samples))
    
    def _save_split(self, split_name: str, samples: List[Dict], output_dir: Path):
        """保存数据分割"""
        if len(samples) == 0:
            logger.warning(f"跳过空的 {split_name} 分割")
            return
        
        # 转换为numpy数组
        input_features = np.stack([s['input_features'] for s in samples])
        output_features = np.stack([s['output_features'] for s in samples])
        
        # 元数据
        metadata = {
            'radius': [s['radius'] for s in samples],
            'colmap_index': [s['colmap_index'] for s in samples],
            'neighbor_count': [s['neighbor_count'] for s in samples],
        }
        
        # 保存为HDF5格式
        h5_file = output_dir / f"{split_name}.h5"
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('input_features', data=input_features, compression='gzip')
            f.create_dataset('output_features', data=output_features, compression='gzip')
            
            # 保存元数据
            for key, values in metadata.items():
                f.create_dataset(f'meta_{key}', data=values)
        
        logger.info(f"保存 {split_name}: {h5_file} ({len(samples)} 样本)")
    
    def _save_metadata(self, output_dir: Path, n_train: int, n_val: int, n_test: int):
        """保存元数据"""
        metadata = {
            'config': self.config.__dict__,
            'dataset_stats': {
                'total_samples': n_train + n_val + n_test,
                'train_samples': n_train,
                'val_samples': n_val,
                'test_samples': n_test,
            },
            'feature_dims': {
                'input_dim': len(self.feature_extractor.extract_colmap_features(torch.tensor([0]))[0]),
                'output_dim': self.feature_extractor._get_output_feature_dim(),
            },
            'radii_used': list(self.data_loader.neighbor_files.keys()),
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"元数据已保存: {metadata_file}")

class StudentNetworkDataset(Dataset):
    """PyTorch数据集类"""
    
    def __init__(self, h5_file: str):
        self.h5_file = h5_file
        
        with h5py.File(h5_file, 'r') as f:
            self.n_samples = len(f['input_features'])
            self.input_dim = f['input_features'].shape[1]
            self.output_dim = f['output_features'].shape[1]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            input_features = torch.from_numpy(f['input_features'][idx]).float()
            output_features = torch.from_numpy(f['output_features'][idx]).float()
            
            return {
                'input': input_features,
                'target': output_features,
                'radius': f['meta_radius'][idx],
                'colmap_index': f['meta_colmap_index'][idx],
                'neighbor_count': f['meta_neighbor_count'][idx],
            }

def main():
    parser = argparse.ArgumentParser(description='创建Student Network训练数据集')
    parser.add_argument('--neighbor_data_dir', type=str, default='batches',
                       help='邻居数据目录')
    parser.add_argument('--output_dir', type=str, default='training_data',
                       help='输出目录')
    parser.add_argument('--min_neighbors', type=int, default=3,
                       help='最少邻居数')
    parser.add_argument('--max_neighbors', type=int, default=512,
                       help='最大邻居数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(
        neighbor_data_dir=args.neighbor_data_dir,
        output_dir=args.output_dir,
        min_neighbors=args.min_neighbors,
        max_neighbors=args.max_neighbors,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
    )
    
    # 创建数据集
    creator = TrainingDatasetCreator(config)
    creator.create_dataset()
    
    # 测试数据加载
    logger.info("测试数据加载...")
    train_file = Path(config.output_dir) / "train.h5"
    if train_file.exists():
        dataset = StudentNetworkDataset(str(train_file))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 测试一个批次
        for batch in dataloader:
            logger.info(f"批次测试成功:")
            logger.info(f"  输入形状: {batch['input'].shape}")
            logger.info(f"  输出形状: {batch['target'].shape}")
            logger.info(f"  半径范围: {batch['radius'].min():.6f} - {batch['radius'].max():.6f}")
            break
    
    logger.info("🎉 训练数据集创建并验证完成！")

if __name__ == "__main__":
    main() 