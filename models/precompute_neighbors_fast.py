#!/usr/bin/env python3
"""
快速邻居预计算脚本 - 采样版本
=====================================

专为大规模数据优化，通过智能采样大幅减少计算量
"""

import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import torch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_arrays(path: Union[str, Path]) -> torch.Tensor:
    """加载 .pt 或 .npz 格式的数组数据"""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    if path.suffix == '.pt':
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            if 'xyz' in data:
                return data['xyz']
            elif len(data) == 1:
                return list(data.values())[0]
            else:
                raise ValueError(f"字典格式的 .pt 文件需要包含 'xyz' 键: {list(data.keys())}")
        else:
            return data
    elif path.suffix == '.npz':
        data = np.load(path)
        if 'xyz' in data:
            return torch.from_numpy(data['xyz'])
        elif len(data.files) == 1:
            return torch.from_numpy(data[data.files[0]])
        else:
            raise ValueError(f"npz 文件需要包含 'xyz' 键: {list(data.keys())}")
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

def load_gaussian_data(path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载高斯数据，返回 (xyz, scale)"""
    path = Path(path)
    
    if path.suffix == '.pt':
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            xyz = data['xyz']
            scale = data['scale']
        else:
            raise ValueError("高斯数据需要字典格式，包含 'xyz' 和 'scale' 键")
    elif path.suffix == '.npz':
        data = np.load(path)
        xyz = torch.from_numpy(data['xyz'])
        scale = torch.from_numpy(data['scale'])
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    return xyz, scale

def smart_sample(data: torch.Tensor, target_size: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """智能采样，保持数据分布"""
    if len(data) <= target_size:
        return data, torch.arange(len(data))
    
    torch.manual_seed(seed)
    indices = torch.randperm(len(data))[:target_size]
    return data[indices], indices

def fast_ball_query_cpu(query_points: torch.Tensor, ref_points: torch.Tensor, 
                       radius: float, k: int) -> torch.Tensor:
    """CPU优化的球查询实现"""
    N = len(query_points)
    M = len(ref_points)
    
    # 结果存储
    neighbor_indices = torch.full((N, k), -1, dtype=torch.long)
    
    # 分批处理以节省内存
    batch_size = min(5000, N)
    
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_queries = query_points[start_idx:end_idx]  # [batch_size, 3]
        
        # 计算距离 [batch_size, M]
        distances = torch.cdist(batch_queries, ref_points, p=2)
        
        # 找到半径内的邻居
        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            dist_row = distances[local_i]
            valid_mask = dist_row <= radius
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                valid_distances = dist_row[valid_indices]
                sorted_indices = torch.argsort(valid_distances)
                num_neighbors = min(k, len(valid_indices))
                selected = valid_indices[sorted_indices[:num_neighbors]]
                neighbor_indices[global_i, :num_neighbors] = selected
    
    return neighbor_indices

def to_csr(neighbor_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """转换为CSR格式"""
    N, K = neighbor_indices.shape
    
    # 找到有效邻居
    valid_mask = neighbor_indices != -1
    neighbor_counts = valid_mask.sum(dim=1).int()
    
    # 生成row_splits
    row_splits = torch.zeros(N + 1, dtype=torch.int32)
    row_splits[1:] = torch.cumsum(neighbor_counts, dim=0)
    
    # 提取有效索引
    indices = neighbor_indices[valid_mask].int()
    
    return indices, row_splits, neighbor_counts

def compute_stats(indices: torch.Tensor, row_splits: torch.Tensor, 
                 gauss_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    """计算邻域统计"""
    N = len(row_splits) - 1
    
    # 计算尺度大小
    scale_magnitudes = torch.norm(gauss_scale, dim=1)
    scale_p995 = torch.quantile(scale_magnitudes, 0.995)
    scale_magnitudes = torch.clamp(scale_magnitudes, max=scale_p995)
    log_scale = torch.log(scale_magnitudes + 1e-8)
    
    mean_log_scale = torch.zeros(N, dtype=torch.float32)
    scale_sum = torch.zeros(N, dtype=torch.float32)
    
    for i in range(N):
        start = row_splits[i].item()
        end = row_splits[i + 1].item()
        
        if start < end:
            neighbor_idx = indices[start:end]
            neighbor_log_scales = log_scale[neighbor_idx]
            neighbor_scales = scale_magnitudes[neighbor_idx]
            
            mean_log_scale[i] = neighbor_log_scales.mean()
            scale_sum[i] = neighbor_scales.sum()
    
    return {
        'mean_log_scale': mean_log_scale,
        'scale_sum': scale_sum
    }

def main():
    parser = argparse.ArgumentParser(description='快速邻居预计算（采样版本）')
    parser.add_argument('--colmap', type=str, required=True,
                       help='COLMAP 点坐标文件')
    parser.add_argument('--gauss', type=str, required=True,
                       help='高斯数据文件')
    parser.add_argument('--radii', type=float, nargs='+',
                       default=[0.012, 0.039, 0.107, 0.273],
                       help='查询半径列表')
    parser.add_argument('--kmax', type=int, nargs='+',
                       default=[32, 128, 256, 512],
                       help='各半径的最大邻居数')
    parser.add_argument('--colmap_sample', type=int, default=50000,
                       help='COLMAP采样数量')
    parser.add_argument('--gauss_sample', type=int, default=500000,
                       help='高斯球采样数量')
    parser.add_argument('--out', type=str, default='batches',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if len(args.radii) != len(args.kmax):
        raise ValueError("radii 和 kmax 的长度必须相同")
    
    # 创建输出目录
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 开始快速邻居预计算...")
    start_time = time.time()
    
    # 加载数据
    logger.info("📂 加载数据...")
    colmap_xyz = load_arrays(args.colmap).float()
    gauss_xyz, gauss_scale = load_gaussian_data(args.gauss)
    gauss_xyz = gauss_xyz.float()
    gauss_scale = gauss_scale.float()
    
    logger.info(f"原始数据: COLMAP={len(colmap_xyz):,}, 高斯={len(gauss_xyz):,}")
    
    # 清理无效数据
    valid_colmap = torch.isfinite(colmap_xyz).all(dim=1)
    valid_gauss = torch.isfinite(gauss_xyz).all(dim=1) & torch.isfinite(gauss_scale).all(dim=1)
    
    colmap_xyz = colmap_xyz[valid_colmap]
    gauss_xyz = gauss_xyz[valid_gauss]
    gauss_scale = gauss_scale[valid_gauss]
    
    logger.info(f"清理后数据: COLMAP={len(colmap_xyz):,}, 高斯={len(gauss_xyz):,}")
    
    # 智能采样
    logger.info("🎯 智能采样...")
    colmap_sampled, colmap_indices = smart_sample(colmap_xyz, args.colmap_sample)
    gauss_sampled, gauss_indices = smart_sample(gauss_xyz, args.gauss_sample)
    gauss_scale_sampled = gauss_scale[gauss_indices]
    
    logger.info(f"采样后数据: COLMAP={len(colmap_sampled):,}, 高斯={len(gauss_sampled):,}")
    
    # 多半径计算
    results = {}
    
    for radius, kmax in zip(args.radii, args.kmax):
        logger.info(f"🔍 处理半径 {radius:.6f}, K={kmax}...")
        radius_start = time.time()
        
        # 球查询
        neighbor_idx = fast_ball_query_cpu(colmap_sampled, gauss_sampled, radius, kmax)
        
        # 转换为CSR
        indices, row_splits, count = to_csr(neighbor_idx)
        
        # 计算统计
        stats = compute_stats(indices, row_splits, gauss_scale_sampled)
        
        # 记录结果
        mean_neighbors = count.float().mean().item()
        median_neighbors = count.float().median().item()
        max_neighbors = count.max().item()
        
        results[f"{radius:.6f}"] = {
            'indices': indices,
            'row_splits': row_splits,
            'count': count,
            'mean_log_scale': stats['mean_log_scale'],
            'scale_sum': stats['scale_sum'],
            'radius': radius,
            'kmax': kmax,
            'colmap_sample_indices': colmap_indices,
            'gauss_sample_indices': gauss_indices,
            'stats': {
                'mean_neighbors': mean_neighbors,
                'median_neighbors': median_neighbors,
                'max_neighbors': max_neighbors,
                'total_edges': len(indices),
                'computation_time': time.time() - radius_start
            }
        }
        
        logger.info(f"  ✅ 完成: 平均邻居数={mean_neighbors:.2f}, 边数={len(indices):,}, 耗时={time.time()-radius_start:.1f}s")
    
    # 保存结果
    logger.info("💾 保存结果...")
    for radius_str, data in results.items():
        output_path = out_dir / f"neigh_r{radius_str}_fast.pt"
        torch.save(data, output_path)
        logger.info(f"  保存: {output_path}")
    
    total_time = time.time() - start_time
    logger.info(f"🎉 快速预计算完成！总耗时: {total_time:.1f}s")
    
    # 打印汇总
    logger.info("\n📊 结果汇总:")
    for radius_str, data in results.items():
        stats = data['stats']
        logger.info(f"  半径 {radius_str}: {stats['mean_neighbors']:.2f} 平均邻居, {stats['total_edges']:,} 边")

if __name__ == "__main__":
    main() 