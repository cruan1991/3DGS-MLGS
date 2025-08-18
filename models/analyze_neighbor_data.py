#!/usr/bin/env python3
"""
邻居数据分析脚本
===============

分析预计算的邻居关系数据，了解：
- 邻居分布统计
- 空间分布特征
- 数据质量评估
- 为训练集创建提供指导
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import logging

# 设置日志和绘图
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

def load_neighbor_data(file_path: str) -> Dict:
    """加载邻居数据文件"""
    logger.info(f"加载邻居数据: {file_path}")
    data = torch.load(file_path, map_location='cpu')
    
    # 提取基本信息
    info = {
        'radius': data['radius'],
        'kmax': data['kmax'],
        'indices': data['indices'],
        'row_splits': data['row_splits'],
        'count': data['count'],
        'mean_log_scale': data['mean_log_scale'],
        'scale_sum': data['scale_sum'],
    }
    
    # 如果是采样版本，还有采样索引
    if 'colmap_sample_indices' in data:
        info['colmap_sample_indices'] = data['colmap_sample_indices']
        info['gauss_sample_indices'] = data['gauss_sample_indices']
        info['is_sampled'] = True
    else:
        info['is_sampled'] = False
    
    return info

def analyze_neighbor_distribution(neighbor_data: Dict) -> Dict:
    """分析邻居数量分布"""
    counts = neighbor_data['count'].numpy()
    
    stats = {
        'total_points': len(counts),
        'total_edges': len(neighbor_data['indices']),
        'mean_neighbors': float(counts.mean()),
        'median_neighbors': float(np.median(counts)),
        'std_neighbors': float(counts.std()),
        'min_neighbors': int(counts.min()),
        'max_neighbors': int(counts.max()),
        'zero_neighbor_points': int(np.sum(counts == 0)),
        'zero_neighbor_ratio': float(np.sum(counts == 0) / len(counts)),
    }
    
    # 百分位数
    percentiles = [5, 10, 25, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'p{p}_neighbors'] = float(np.percentile(counts, p))
    
    return stats

def analyze_spatial_statistics(neighbor_data: Dict) -> Dict:
    """分析空间统计信息"""
    mean_log_scale = neighbor_data['mean_log_scale'].numpy()
    scale_sum = neighbor_data['scale_sum'].numpy()
    
    # 过滤掉没有邻居的点
    has_neighbors = neighbor_data['count'] > 0
    has_neighbors_np = has_neighbors.numpy()  # 转换为numpy
    valid_mean_log_scale = mean_log_scale[has_neighbors_np]
    valid_scale_sum = scale_sum[has_neighbors_np]
    
    spatial_stats = {
        'valid_points': int(np.sum(has_neighbors_np)),
        'mean_log_scale_mean': float(valid_mean_log_scale.mean()) if len(valid_mean_log_scale) > 0 else 0,
        'mean_log_scale_std': float(valid_mean_log_scale.std()) if len(valid_mean_log_scale) > 0 else 0,
        'scale_sum_mean': float(valid_scale_sum.mean()) if len(valid_scale_sum) > 0 else 0,
        'scale_sum_std': float(valid_scale_sum.std()) if len(valid_scale_sum) > 0 else 0,
    }
    
    return spatial_stats

def plot_neighbor_distribution(neighbor_data: Dict, save_path: str):
    """绘制邻居数量分布图"""
    counts = neighbor_data['count'].numpy()
    radius = neighbor_data['radius']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 直方图
    axes[0,0].hist(counts, bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('邻居数量')
    axes[0,0].set_ylabel('频次')
    axes[0,0].set_title(f'邻居数量分布 (半径={radius:.3f})')
    axes[0,0].axvline(counts.mean(), color='red', linestyle='--', label=f'均值={counts.mean():.1f}')
    axes[0,0].axvline(np.median(counts), color='green', linestyle='--', label=f'中位数={np.median(counts):.1f}')
    axes[0,0].legend()
    
    # 2. 累积分布
    sorted_counts = np.sort(counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[0,1].plot(sorted_counts, cumulative)
    axes[0,1].set_xlabel('邻居数量')
    axes[0,1].set_ylabel('累积概率')
    axes[0,1].set_title('邻居数量累积分布')
    axes[0,1].grid(True)
    
    # 3. 对数尺度直方图
    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) > 0:
        axes[1,0].hist(nonzero_counts, bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('邻居数量')
        axes[1,0].set_ylabel('频次')
        axes[1,0].set_title('非零邻居数量分布')
        axes[1,0].set_yscale('log')
    
    # 4. 箱线图
    axes[1,1].boxplot(counts, vert=True)
    axes[1,1].set_ylabel('邻居数量')
    axes[1,1].set_title('邻居数量箱线图')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"邻居分布图已保存: {save_path}")

def plot_spatial_analysis(neighbor_data: Dict, save_path: str):
    """绘制空间分析图"""
    counts = neighbor_data['count'].numpy()
    mean_log_scale = neighbor_data['mean_log_scale'].numpy()
    scale_sum = neighbor_data['scale_sum'].numpy()
    radius = neighbor_data['radius']
    
    # 过滤有效数据
    valid_mask = counts > 0
    valid_counts = counts[valid_mask]
    valid_mean_log_scale = mean_log_scale[valid_mask]
    valid_scale_sum = scale_sum[valid_mask]
    
    if len(valid_counts) == 0:
        logger.warning("没有有效的邻居数据用于空间分析")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 邻居数 vs 平均对数尺度
    axes[0,0].scatter(valid_counts, valid_mean_log_scale, alpha=0.5, s=1)
    axes[0,0].set_xlabel('邻居数量')
    axes[0,0].set_ylabel('平均对数尺度')
    axes[0,0].set_title(f'邻居数量 vs 平均对数尺度 (半径={radius:.3f})')
    
    # 2. 邻居数 vs 尺度总和
    axes[0,1].scatter(valid_counts, valid_scale_sum, alpha=0.5, s=1)
    axes[0,1].set_xlabel('邻居数量')
    axes[0,1].set_ylabel('尺度总和')
    axes[0,1].set_title('邻居数量 vs 尺度总和')
    
    # 3. 平均对数尺度分布
    axes[1,0].hist(valid_mean_log_scale, bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('平均对数尺度')
    axes[1,0].set_ylabel('频次')
    axes[1,0].set_title('平均对数尺度分布')
    
    # 4. 尺度总和分布
    axes[1,1].hist(valid_scale_sum, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('尺度总和')
    axes[1,1].set_ylabel('频次')
    axes[1,1].set_title('尺度总和分布')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"空间分析图已保存: {save_path}")

def analyze_multi_radius_comparison(data_files: List[str], save_dir: str):
    """比较不同半径的邻居分布"""
    all_data = []
    
    for file_path in data_files:
        neighbor_data = load_neighbor_data(file_path)
        dist_stats = analyze_neighbor_distribution(neighbor_data)
        spatial_stats = analyze_spatial_statistics(neighbor_data)
        
        combined_stats = {
            'radius': neighbor_data['radius'],
            'kmax': neighbor_data['kmax'],
            **dist_stats,
            **spatial_stats
        }
        all_data.append(combined_stats)
    
    # 转换为DataFrame用于分析
    df = pd.DataFrame(all_data)
    
    # 保存统计表
    csv_path = Path(save_dir) / "multi_radius_stats.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"多半径统计表已保存: {csv_path}")
    
    # 绘制比较图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 平均邻居数 vs 半径
    axes[0,0].plot(df['radius'], df['mean_neighbors'], 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('半径')
    axes[0,0].set_ylabel('平均邻居数')
    axes[0,0].set_title('平均邻居数 vs 半径')
    axes[0,0].set_xscale('log')
    axes[0,0].grid(True)
    
    # 2. 总边数 vs 半径
    axes[0,1].plot(df['radius'], df['total_edges'], 'ro-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('半径')
    axes[0,1].set_ylabel('总边数')
    axes[0,1].set_title('总边数 vs 半径')
    axes[0,1].set_xscale('log')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True)
    
    # 3. 零邻居比例 vs 半径
    axes[0,2].plot(df['radius'], df['zero_neighbor_ratio'] * 100, 'go-', linewidth=2, markersize=8)
    axes[0,2].set_xlabel('半径')
    axes[0,2].set_ylabel('零邻居比例 (%)')
    axes[0,2].set_title('零邻居比例 vs 半径')
    axes[0,2].set_xscale('log')
    axes[0,2].grid(True)
    
    # 4. 中位数邻居数 vs 半径
    axes[1,0].plot(df['radius'], df['median_neighbors'], 'mo-', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('半径')
    axes[1,0].set_ylabel('中位数邻居数')
    axes[1,0].set_title('中位数邻居数 vs 半径')
    axes[1,0].set_xscale('log')
    axes[1,0].grid(True)
    
    # 5. 最大邻居数 vs 半径
    axes[1,1].plot(df['radius'], df['max_neighbors'], 'co-', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('半径')
    axes[1,1].set_ylabel('最大邻居数')
    axes[1,1].set_title('最大邻居数 vs 半径')
    axes[1,1].set_xscale('log')
    axes[1,1].grid(True)
    
    # 6. 邻居数标准差 vs 半径
    axes[1,2].plot(df['radius'], df['std_neighbors'], 'yo-', linewidth=2, markersize=8)
    axes[1,2].set_xlabel('半径')
    axes[1,2].set_ylabel('邻居数标准差')
    axes[1,2].set_title('邻居数标准差 vs 半径')
    axes[1,2].set_xscale('log')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    comparison_path = Path(save_dir) / "multi_radius_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"多半径比较图已保存: {comparison_path}")
    
    return df

def main():
    # 配置路径
    data_dir = Path("batches")
    output_dir = Path("neighbor_analysis")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("🔍 开始邻居数据分析...")
    
    # 找到所有快速邻居数据文件
    fast_files = sorted(list(data_dir.glob("*_fast.pt")))
    if not fast_files:
        logger.error("未找到快速邻居数据文件")
        return
    
    logger.info(f"找到 {len(fast_files)} 个快速邻居数据文件")
    
    # 分析每个半径的数据
    all_stats = []
    
    for file_path in fast_files:
        logger.info(f"分析文件: {file_path}")
        
        # 加载数据
        neighbor_data = load_neighbor_data(str(file_path))
        radius = neighbor_data['radius']
        
        # 分析分布
        dist_stats = analyze_neighbor_distribution(neighbor_data)
        spatial_stats = analyze_spatial_statistics(neighbor_data)
        
        # 保存统计信息
        combined_stats = {
            'file': file_path.name,
            'radius': radius,
            **dist_stats,
            **spatial_stats
        }
        all_stats.append(combined_stats)
        
        # 生成单独的可视化
        file_prefix = file_path.stem
        
        # 邻居分布图
        dist_plot_path = output_dir / f"{file_prefix}_distribution.png"
        plot_neighbor_distribution(neighbor_data, str(dist_plot_path))
        
        # 空间分析图
        spatial_plot_path = output_dir / f"{file_prefix}_spatial.png"
        plot_spatial_analysis(neighbor_data, str(spatial_plot_path))
        
        # 打印关键统计
        logger.info(f"  半径 {radius:.6f}:")
        logger.info(f"    平均邻居数: {dist_stats['mean_neighbors']:.2f}")
        logger.info(f"    中位数邻居数: {dist_stats['median_neighbors']:.2f}")
        logger.info(f"    零邻居比例: {dist_stats['zero_neighbor_ratio']*100:.1f}%")
        logger.info(f"    总边数: {dist_stats['total_edges']:,}")
    
    # 多半径比较分析
    logger.info("进行多半径比较分析...")
    comparison_df = analyze_multi_radius_comparison([str(f) for f in fast_files], str(output_dir))
    
    # 保存完整统计
    full_stats_path = output_dir / "detailed_stats.csv"
    full_df = pd.DataFrame(all_stats)
    full_df.to_csv(full_stats_path, index=False)
    
    logger.info(f"✅ 分析完成！结果保存在: {output_dir}")
    logger.info(f"📊 详细统计: {full_stats_path}")
    logger.info(f"📈 可视化图表: {output_dir}/*.png")
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 邻居数据分析总结")
    print("="*60)
    
    for stats in all_stats:
        print(f"\n🔍 半径 {stats['radius']:.6f}:")
        print(f"  • 平均邻居数: {stats['mean_neighbors']:.2f}")
        print(f"  • 中位数邻居数: {stats['median_neighbors']:.2f}")
        print(f"  • 最大邻居数: {stats['max_neighbors']}")
        print(f"  • 零邻居比例: {stats['zero_neighbor_ratio']*100:.1f}%")
        print(f"  • 总边数: {stats['total_edges']:,}")
        print(f"  • 有效点数: {stats['valid_points']:,}")

if __name__ == "__main__":
    main() 