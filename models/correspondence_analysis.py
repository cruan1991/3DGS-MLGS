#!/usr/bin/env python3
"""
COLMAP点云 vs 3DGS高斯球对应关系分析工具

目标：
1. 分析原始COLMAP点与最终高斯球的空间分布关系
2. 发现densification的模式和规律
3. 提取可用于训练的关联特征

使用方法：
python correspondence_analysis.py --colmap-points points3D.ply --teacher-gaussians gaussian_ball.ply
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.cluster import DBSCAN
import argparse
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Add path for 3DGS imports
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene.gaussian_model import GaussianModel
from scene import dataset_readers
from plyfile import PlyData, PlyElement

class CorrespondenceAnalyzer:
    """分析COLMAP点云与3DGS高斯球的对应关系"""
    
    def __init__(self, colmap_points_path, teacher_gaussians_path):
        self.colmap_points_path = colmap_points_path
        self.teacher_gaussians_path = teacher_gaussians_path
        
        # 数据存储
        self.colmap_points = None
        self.gaussian_positions = None
        self.gaussian_scales = None
        self.gaussian_opacities = None
        
        # 分析结果
        self.correspondence_data = {}
        self.patterns = {}
        
    def load_data(self):
        """加载COLMAP点云和Teacher高斯球数据"""
        print("=== 数据加载 ===")
        
        # 1. 加载COLMAP点云
        print(f"加载COLMAP点云: {self.colmap_points_path}")
        try:
            if self.colmap_points_path.endswith('.ply'):
                pcd = dataset_readers.fetchPly(self.colmap_points_path)
                self.colmap_points = np.stack([pcd.points['x'], pcd.points['y'], pcd.points['z']], axis=1)
            else:
                # 尝试其他格式
                raise ValueError(f"Unsupported format: {self.colmap_points_path}")
                
            print(f"✅ COLMAP点云加载成功: {len(self.colmap_points)} 个点")
            
        except Exception as e:
            print(f"❌ COLMAP点云加载失败: {e}")
            return False
        
        # 2. 加载Teacher高斯球
        print(f"加载Teacher高斯球: {self.teacher_gaussians_path}")
        try:
            # 检测SH degree
            plydata = PlyData.read(self.teacher_gaussians_path)
            vertex = plydata['vertex']
            f_rest_props = [prop for prop in vertex.properties if prop.name.startswith('f_rest_')]
            if f_rest_props:
                max_f_rest = max([int(prop.name.split('_')[-1]) for prop in f_rest_props])
                sh_degree = int(np.sqrt((max_f_rest + 4) / 3)) - 1
            else:
                sh_degree = 0
            
            # 加载高斯球
            gaussians = GaussianModel(sh_degree)
            gaussians.load_ply(self.teacher_gaussians_path, use_train_test_exp=False)
            
            self.gaussian_positions = gaussians.get_xyz.detach().cpu().numpy()
            self.gaussian_scales = gaussians.get_scaling.detach().cpu().numpy()
            self.gaussian_opacities = gaussians.get_opacity.detach().cpu().numpy()
            
            # 过滤NaN值
            valid_mask = ~(np.isnan(self.gaussian_positions).any(axis=1) | 
                          np.isnan(self.gaussian_scales).any(axis=1))
            
            if not valid_mask.all():
                n_invalid = (~valid_mask).sum()
                print(f"⚠️  过滤掉 {n_invalid} 个包含NaN的高斯球")
                self.gaussian_positions = self.gaussian_positions[valid_mask]
                self.gaussian_scales = self.gaussian_scales[valid_mask]
                self.gaussian_opacities = self.gaussian_opacities[valid_mask]
            
            print(f"✅ Teacher高斯球加载成功: {len(self.gaussian_positions)} 个高斯球")
            
        except Exception as e:
            print(f"❌ Teacher高斯球加载失败: {e}")
            return False
        
        return True
    
    def analyze_spatial_distribution(self):
        """分析空间分布特征"""
        print("\n=== 空间分布分析 ===")
        
        # 1. 基础统计信息
        colmap_stats = {
            'count': len(self.colmap_points),
            'bounds': {
                'min': self.colmap_points.min(axis=0),
                'max': self.colmap_points.max(axis=0),
                'range': self.colmap_points.max(axis=0) - self.colmap_points.min(axis=0)
            },
            'center': self.colmap_points.mean(axis=0),
            'std': self.colmap_points.std(axis=0)
        }
        
        gaussian_stats = {
            'count': len(self.gaussian_positions),
            'bounds': {
                'min': self.gaussian_positions.min(axis=0),
                'max': self.gaussian_positions.max(axis=0),
                'range': self.gaussian_positions.max(axis=0) - self.gaussian_positions.min(axis=0)
            },
            'center': self.gaussian_positions.mean(axis=0),
            'std': self.gaussian_positions.std(axis=0)
        }
        
        # 2. 密度比较
        density_ratio = gaussian_stats['count'] / colmap_stats['count']
        
        print(f"📊 基础统计:")
        print(f"  COLMAP点数: {colmap_stats['count']:,}")
        print(f"  高斯球数: {gaussian_stats['count']:,}")
        print(f"  密度比例: {density_ratio:.2f}x (平均每个COLMAP点对应{density_ratio:.1f}个高斯球)")
        
        print(f"📊 空间范围:")
        print(f"  COLMAP: {colmap_stats['bounds']['range']}")
        print(f"  Gaussian: {gaussian_stats['bounds']['range']}")
        
        # 3. 计算局部密度
        colmap_density = self.compute_local_density(self.colmap_points, radius=0.1)
        gaussian_density = self.compute_local_density(self.gaussian_positions, radius=0.1)
        
        self.correspondence_data.update({
            'colmap_stats': colmap_stats,
            'gaussian_stats': gaussian_stats,
            'density_ratio': density_ratio,
            'colmap_local_density': colmap_density,
            'gaussian_local_density': gaussian_density
        })
        
        return colmap_stats, gaussian_stats
    
    def compute_local_density(self, points, radius=0.1, n_samples=5000):
        """计算局部密度"""
        if len(points) > n_samples:
            # 随机采样以提高计算效率
            indices = np.random.choice(len(points), n_samples, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
            indices = np.arange(len(points))
        
        # 使用KDTree计算局部密度
        tree = KDTree(points)
        neighbor_counts = tree.query_radius(sample_points, r=radius, count_only=True)
        
        # 密度 = 邻居数量 / 球体积
        sphere_volume = (4/3) * np.pi * (radius ** 3)
        densities = neighbor_counts / sphere_volume
        
        return densities, indices
    
    def analyze_correspondence_patterns(self):
        """分析对应关系模式"""
        print("\n=== 对应关系模式分析 ===")
        
        # 1. 为每个COLMAP点找到最近的高斯球
        print("计算最近邻对应关系...")
        tree = KDTree(self.gaussian_positions)
        
        # 对每个COLMAP点，找到最近的k个高斯球
        k_values = [1, 5, 10, 20, 30, 50]
        correspondence_results = {}
        
        for k in k_values:
            distances, indices = tree.query(self.colmap_points, k=k)
            correspondence_results[k] = {
                'distances': distances,
                'indices': indices,
                'mean_distance': distances.mean(),
                'std_distance': distances.std()
            }
            
        print(f"📊 最近邻距离统计 (k=1):")
        print(f"  平均距离: {correspondence_results[1]['mean_distance']:.4f}")
        print(f"  标准差: {correspondence_results[1]['std_distance']:.4f}")
        print(f"  最大距离: {correspondence_results[1]['distances'].max():.4f}")
        print(f"  最小距离: {correspondence_results[1]['distances'].min():.4f}")
        
        # 2. 分析densification模式
        densification_analysis = self.analyze_densification_patterns(correspondence_results)
        
        # 3. 分析空间梯度
        gradient_analysis = self.analyze_spatial_gradients()
        
        self.correspondence_data.update({
            'correspondence_results': correspondence_results,
            'densification_analysis': densification_analysis,
            'gradient_analysis': gradient_analysis
        })
        
        return correspondence_results
    
    def analyze_densification_patterns(self, correspondence_results):
        """分析densification模式"""
        print("分析densification模式...")
        
        # 1. 计算每个COLMAP点周围的高斯球密度
        radius_values = [0.05, 0.1, 0.2, 0.5]
        tree = KDTree(self.gaussian_positions)
        
        densification_patterns = {}
        
        for radius in radius_values:
            # 计算每个COLMAP点周围半径内的高斯球数量
            neighbor_counts = tree.query_radius(self.colmap_points, r=radius, count_only=True)
            
            densification_patterns[radius] = {
                'neighbor_counts': neighbor_counts,
                'mean_count': neighbor_counts.mean(),
                'std_count': neighbor_counts.std(),
                'max_count': neighbor_counts.max(),
                'min_count': neighbor_counts.min()
            }
            
            print(f"  半径 {radius}: 平均{neighbor_counts.mean():.1f}个高斯球 (±{neighbor_counts.std():.1f})")
        
        # 2. 识别高密度和低密度区域
        radius = 0.1  # 使用0.1作为分析半径
        neighbor_counts = densification_patterns[radius]['neighbor_counts']
        
        # 使用四分位数划分密度等级
        q25, q50, q75 = np.percentile(neighbor_counts, [25, 50, 75])
        
        density_categories = {
            'low': neighbor_counts <= q25,
            'medium': (neighbor_counts > q25) & (neighbor_counts <= q75),
            'high': neighbor_counts > q75
        }
        
        print(f"📊 密度分布 (半径{radius}):")
        for category, mask in density_categories.items():
            count = mask.sum()
            percentage = count / len(neighbor_counts) * 100
            print(f"  {category.capitalize()}: {count} 点 ({percentage:.1f}%)")
        
        # 3. 分析不同密度区域的特征
        density_analysis = {}
        for category, mask in density_categories.items():
            if mask.sum() > 0:
                category_points = self.colmap_points[mask]
                category_counts = neighbor_counts[mask]
                
                density_analysis[category] = {
                    'points': category_points,
                    'counts': category_counts,
                    'mean_count': category_counts.mean(),
                    'std_count': category_counts.std(),
                    'spatial_variance': np.var(category_points, axis=0).mean()
                }
        
        return {
            'patterns': densification_patterns,
            'density_categories': density_categories,
            'density_analysis': density_analysis
        }
    
    def analyze_spatial_gradients(self):
        """分析空间梯度特征"""
        print("分析空间梯度...")
        
        # 1. 计算COLMAP点的局部梯度
        colmap_gradients = self.compute_point_gradients(self.colmap_points)
        
        # 2. 计算高斯球的局部梯度
        gaussian_gradients = self.compute_point_gradients(self.gaussian_positions)
        
        # 3. 分析梯度与densification的关系
        tree = KDTree(self.gaussian_positions)
        neighbor_counts = tree.query_radius(self.colmap_points, r=0.1, count_only=True)
        
        # 计算梯度强度与密度的相关性
        gradient_magnitudes = np.linalg.norm(colmap_gradients, axis=1)
        correlation = np.corrcoef(gradient_magnitudes, neighbor_counts)[0, 1]
        
        print(f"📊 梯度-密度相关性: {correlation:.3f}")
        
        return {
            'colmap_gradients': colmap_gradients,
            'gaussian_gradients': gaussian_gradients,
            'gradient_density_correlation': correlation,
            'gradient_magnitudes': gradient_magnitudes
        }
    
    def compute_point_gradients(self, points, k=10):
        """计算点云的局部梯度"""
        tree = KDTree(points)
        gradients = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # 找到k个最近邻
            distances, indices = tree.query([point], k=k+1)  # +1 because it includes the point itself
            neighbors = points[indices[0][1:]]  # 排除自己
            
            # 计算到邻居的向量
            vectors = neighbors - point
            
            # 计算主方向 (第一主成分)
            if len(vectors) > 0:
                cov_matrix = np.cov(vectors.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                # 主方向是最大特征值对应的特征向量
                principal_direction = eigenvecs[:, -1]
                gradients[i] = principal_direction
        
        return gradients
    
    def visualize_correspondence_analysis(self, save_dir='correspondence_analysis'):
        """可视化对应关系分析结果"""
        print(f"\n=== 生成可视化结果 ===")
        Path(save_dir).mkdir(exist_ok=True)
        
        # 1. 3D散点图对比
        self.plot_3d_comparison(save_dir)
        
        # 2. 密度分布图
        self.plot_density_distribution(save_dir)
        
        # 3. 对应关系距离分布
        self.plot_correspondence_distances(save_dir)
        
        # 4. Densification模式可视化
        self.plot_densification_patterns(save_dir)
        
        # 5. 梯度分析图
        self.plot_gradient_analysis(save_dir)
        
        # 6. 交互式3D可视化
        self.create_interactive_3d_plot(save_dir)
        
        print(f"✅ 可视化结果已保存到 {save_dir}/")
    
    def plot_3d_comparison(self, save_dir):
        """3D空间分布对比图"""
        fig = plt.figure(figsize=(20, 8))
        
        # 子图1: COLMAP点云
        ax1 = fig.add_subplot(131, projection='3d')
        sample_colmap = self.colmap_points[::max(1, len(self.colmap_points)//5000)]
        ax1.scatter(sample_colmap[:, 0], sample_colmap[:, 1], sample_colmap[:, 2], 
                   c='blue', alpha=0.6, s=1)
        ax1.set_title('COLMAP Points')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 子图2: 高斯球位置
        ax2 = fig.add_subplot(132, projection='3d')
        sample_gaussian = self.gaussian_positions[::max(1, len(self.gaussian_positions)//5000)]
        ax2.scatter(sample_gaussian[:, 0], sample_gaussian[:, 1], sample_gaussian[:, 2], 
                   c='red', alpha=0.6, s=1)
        ax2.set_title('Gaussian Positions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 子图3: 叠加对比
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(sample_colmap[:, 0], sample_colmap[:, 1], sample_colmap[:, 2], 
                   c='blue', alpha=0.4, s=1, label='COLMAP')
        ax3.scatter(sample_gaussian[:, 0], sample_gaussian[:, 1], sample_gaussian[:, 2], 
                   c='red', alpha=0.4, s=1, label='Gaussians')
        ax3.set_title('Overlay Comparison')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/3d_spatial_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_density_distribution(self, save_dir):
        """密度分布分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 局部密度直方图
        colmap_densities = self.correspondence_data['colmap_local_density'][0]
        gaussian_densities = self.correspondence_data['gaussian_local_density'][0]
        
        axes[0, 0].hist(colmap_densities, bins=50, alpha=0.7, label='COLMAP', color='blue')
        axes[0, 0].hist(gaussian_densities, bins=50, alpha=0.7, label='Gaussians', color='red')
        axes[0, 0].set_xlabel('Local Density')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Local Density Distribution')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # 2. 密度比例分析
        densification_analysis = self.correspondence_data['densification_analysis']
        radius = 0.1
        neighbor_counts = densification_analysis['patterns'][radius]['neighbor_counts']
        
        axes[0, 1].hist(neighbor_counts, bins=50, alpha=0.7, color='green')
        axes[0, 1].axvline(neighbor_counts.mean(), color='red', linestyle='--', 
                          label=f'Mean: {neighbor_counts.mean():.1f}')
        axes[0, 1].set_xlabel('Gaussians per COLMAP Point')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Densification Distribution (r={radius})')
        axes[0, 1].legend()
        
        # 3. 密度-距离关系
        correspondence_results = self.correspondence_data['correspondence_results']
        distances = correspondence_results[1]['distances'].flatten()
        
        axes[1, 0].scatter(distances, neighbor_counts, alpha=0.5, s=1)
        axes[1, 0].set_xlabel('Distance to Nearest Gaussian')
        axes[1, 0].set_ylabel('Local Gaussian Count')
        axes[1, 0].set_title('Distance vs Local Density')
        
        # 计算相关性
        correlation = np.corrcoef(distances, neighbor_counts)[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 0].transAxes, verticalalignment='top')
        
        # 4. 密度分类饼图
        density_categories = densification_analysis['density_categories']
        category_counts = [mask.sum() for mask in density_categories.values()]
        category_labels = list(density_categories.keys())
        
        axes[1, 1].pie(category_counts, labels=category_labels, autopct='%1.1f%%')
        axes[1, 1].set_title('Density Category Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/density_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correspondence_distances(self, save_dir):
        """对应关系距离分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        correspondence_results = self.correspondence_data['correspondence_results']
        
        # 1. 不同k值的距离分布
        k_values = [1, 5, 10, 20]
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (k, color) in enumerate(zip(k_values, colors)):
            distances = correspondence_results[k]['distances']
            if k == 1:
                dist_to_plot = distances.flatten()
            else:
                dist_to_plot = distances.mean(axis=1)  # 平均距离
            
            axes[0, 0].hist(dist_to_plot, bins=50, alpha=0.7, 
                           label=f'k={k}', color=color)
        
        axes[0, 0].set_xlabel('Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distance Distribution for Different k')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # 2. 距离统计对比
        k_values_all = list(correspondence_results.keys())
        mean_distances = [correspondence_results[k]['mean_distance'] for k in k_values_all]
        std_distances = [correspondence_results[k]['std_distance'] for k in k_values_all]
        
        axes[0, 1].errorbar(k_values_all, mean_distances, yerr=std_distances, 
                           marker='o', capsize=5)
        axes[0, 1].set_xlabel('k (Number of Neighbors)')
        axes[0, 1].set_ylabel('Mean Distance')
        axes[0, 1].set_title('Distance Statistics vs k')
        axes[0, 1].grid(True)
        
        # 3. 最近邻距离的空间分布
        distances_k1 = correspondence_results[1]['distances'].flatten()
        
        # 创建距离的空间热图 (投影到2D)
        x_coords = self.colmap_points[:, 0]
        y_coords = self.colmap_points[:, 1]
        
        scatter = axes[1, 0].scatter(x_coords, y_coords, c=distances_k1, 
                                    cmap='viridis', s=1, alpha=0.6)
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Y Coordinate')
        axes[1, 0].set_title('Spatial Distribution of Nearest Distances')
        plt.colorbar(scatter, ax=axes[1, 0], label='Distance')
        
        # 4. 距离累积分布函数
        sorted_distances = np.sort(distances_k1)
        cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
        
        axes[1, 1].plot(sorted_distances, cumulative)
        axes[1, 1].set_xlabel('Distance')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution of Nearest Distances')
        axes[1, 1].grid(True)
        
        # 添加百分位数标记
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            dist_p = np.percentile(sorted_distances, p)
            axes[1, 1].axvline(dist_p, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].text(dist_p, p/100, f'{p}%', rotation=90, 
                           verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correspondence_distances.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_densification_patterns(self, save_dir):
        """Densification模式可视化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        densification_analysis = self.correspondence_data['densification_analysis']
        
        # 1. 不同半径的densification分布
        radius_values = [0.05, 0.1, 0.2]
        for i, radius in enumerate(radius_values):
            neighbor_counts = densification_analysis['patterns'][radius]['neighbor_counts']
            
            axes[0, i].hist(neighbor_counts, bins=50, alpha=0.7, color='green')
            axes[0, i].axvline(neighbor_counts.mean(), color='red', linestyle='--', 
                              label=f'Mean: {neighbor_counts.mean():.1f}')
            axes[0, i].set_xlabel('Gaussian Count')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].set_title(f'Densification at r={radius}')
            axes[0, i].legend()
        
        # 2. 密度分类的空间分布
        density_categories = densification_analysis['density_categories']
        colors = {'low': 'blue', 'medium': 'green', 'high': 'red'}
        
        for i, (category, mask) in enumerate(density_categories.items()):
            category_points = self.colmap_points[mask]
            if len(category_points) > 0:
                axes[1, i].scatter(category_points[:, 0], category_points[:, 1], 
                                  c=colors[category], s=1, alpha=0.6)
                axes[1, i].set_xlabel('X Coordinate')
                axes[1, i].set_ylabel('Y Coordinate')
                axes[1, i].set_title(f'{category.capitalize()} Density Regions')
                axes[1, i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/densification_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gradient_analysis(self, save_dir):
        """梯度分析可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        gradient_analysis = self.correspondence_data['gradient_analysis']
        gradient_magnitudes = gradient_analysis['gradient_magnitudes']
        
        # 1. 梯度强度分布
        axes[0, 0].hist(gradient_magnitudes, bins=50, alpha=0.7, color='purple')
        axes[0, 0].axvline(gradient_magnitudes.mean(), color='red', linestyle='--',
                          label=f'Mean: {gradient_magnitudes.mean():.3f}')
        axes[0, 0].set_xlabel('Gradient Magnitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Gradient Magnitude Distribution')
        axes[0, 0].legend()
        
        # 2. 梯度vs密度相关性
        densification_analysis = self.correspondence_data['densification_analysis']
        neighbor_counts = densification_analysis['patterns'][0.1]['neighbor_counts']
        
        axes[0, 1].scatter(gradient_magnitudes, neighbor_counts, alpha=0.5, s=1)
        axes[0, 1].set_xlabel('Gradient Magnitude')
        axes[0, 1].set_ylabel('Local Gaussian Count')
        axes[0, 1].set_title('Gradient vs Densification')
        
        correlation = gradient_analysis['gradient_density_correlation']
        axes[0, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=axes[0, 1].transAxes, verticalalignment='top')
        
        # 3. 梯度的空间分布
        x_coords = self.colmap_points[:, 0]
        y_coords = self.colmap_points[:, 1]
        
        scatter = axes[1, 0].scatter(x_coords, y_coords, c=gradient_magnitudes, 
                                    cmap='plasma', s=1, alpha=0.6)
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Y Coordinate')
        axes[1, 0].set_title('Spatial Distribution of Gradients')
        plt.colorbar(scatter, ax=axes[1, 0], label='Gradient Magnitude')
        
        # 4. 梯度方向分析（2D投影）
        colmap_gradients = gradient_analysis['colmap_gradients']
        gradient_angles = np.arctan2(colmap_gradients[:, 1], colmap_gradients[:, 0])
        
        axes[1, 1].hist(gradient_angles, bins=36, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Gradient Direction (radians)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Gradient Direction Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gradient_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_3d_plot(self, save_dir):
        """创建交互式3D可视化"""
        print("生成交互式3D可视化...")
        
        # 采样数据以提高性能
        n_sample = 5000
        colmap_sample_idx = np.random.choice(len(self.colmap_points), 
                                           min(n_sample, len(self.colmap_points)), 
                                           replace=False)
        gaussian_sample_idx = np.random.choice(len(self.gaussian_positions), 
                                             min(n_sample, len(self.gaussian_positions)), 
                                             replace=False)
        
        colmap_sample = self.colmap_points[colmap_sample_idx]
        gaussian_sample = self.gaussian_positions[gaussian_sample_idx]
        
        # 创建交互式plotly图
        fig = go.Figure()
        
        # 添加COLMAP点云
        fig.add_trace(go.Scatter3d(
            x=colmap_sample[:, 0],
            y=colmap_sample[:, 1], 
            z=colmap_sample[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='blue',
                opacity=0.6
            ),
            name='COLMAP Points',
            text=[f'COLMAP Point {i}' for i in colmap_sample_idx]
        ))
        
        # 添加高斯球位置
        fig.add_trace(go.Scatter3d(
            x=gaussian_sample[:, 0],
            y=gaussian_sample[:, 1],
            z=gaussian_sample[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.4
            ),
            name='Gaussian Positions',
            text=[f'Gaussian {i}' for i in gaussian_sample_idx]
        ))
        
        fig.update_layout(
            title='Interactive 3D Correspondence Analysis',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=1000,
            height=800
        )
        
        # 保存交互式图
        fig.write_html(f'{save_dir}/interactive_3d_comparison.html')
        print(f"✅ 交互式3D可视化已保存到 {save_dir}/interactive_3d_comparison.html")
    
    def generate_correspondence_report(self, save_dir='correspondence_analysis'):
        """生成对应关系分析报告"""
        print("\n=== 生成分析报告 ===")
        
        report_path = f'{save_dir}/correspondence_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("COLMAP点云 vs 3DGS高斯球对应关系分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 1. 基础统计
            f.write("1. 基础统计信息\n")
            f.write("-" * 30 + "\n")
            colmap_stats = self.correspondence_data['colmap_stats']
            gaussian_stats = self.correspondence_data['gaussian_stats']
            density_ratio = self.correspondence_data['density_ratio']
            
            f.write(f"COLMAP点数: {colmap_stats['count']:,}\n")
            f.write(f"高斯球数: {gaussian_stats['count']:,}\n")
            f.write(f"密度比例: {density_ratio:.2f}x\n")
            f.write(f"平均每个COLMAP点对应: {density_ratio:.1f} 个高斯球\n\n")
            
            # 2. 空间分布
            f.write("2. 空间分布特征\n")
            f.write("-" * 30 + "\n")
            f.write(f"COLMAP空间范围: {colmap_stats['bounds']['range']}\n")
            f.write(f"高斯球空间范围: {gaussian_stats['bounds']['range']}\n")
            f.write(f"COLMAP中心: {colmap_stats['center']}\n")
            f.write(f"高斯球中心: {gaussian_stats['center']}\n\n")
            
            # 3. 对应关系分析
            f.write("3. 对应关系分析\n")
            f.write("-" * 30 + "\n")
            correspondence_results = self.correspondence_data['correspondence_results']
            
            for k in [1, 5, 10, 20, 30]:
                if k in correspondence_results:
                    result = correspondence_results[k]
                    f.write(f"k={k}: 平均距离 {result['mean_distance']:.4f} ± {result['std_distance']:.4f}\n")
            
            f.write("\n")
            
            # 4. Densification分析
            f.write("4. Densification模式分析\n")
            f.write("-" * 30 + "\n")
            densification_analysis = self.correspondence_data['densification_analysis']
            
            for radius in [0.05, 0.1, 0.2]:
                if radius in densification_analysis['patterns']:
                    pattern = densification_analysis['patterns'][radius]
                    f.write(f"半径{radius}: 平均{pattern['mean_count']:.1f}个高斯球 (±{pattern['std_count']:.1f})\n")
            
            # 密度分类统计
            f.write("\n密度分类 (半径0.1):\n")
            density_categories = densification_analysis['density_categories']
            total_points = len(self.colmap_points)
            
            for category, mask in density_categories.items():
                count = mask.sum()
                percentage = count / total_points * 100
                f.write(f"  {category.capitalize()}: {count} 点 ({percentage:.1f}%)\n")
            
            f.write("\n")
            
            # 5. 梯度分析
            f.write("5. 梯度分析\n")
            f.write("-" * 30 + "\n")
            gradient_analysis = self.correspondence_data['gradient_analysis']
            
            correlation = gradient_analysis['gradient_density_correlation']
            f.write(f"梯度-密度相关性: {correlation:.3f}\n")
            
            gradient_magnitudes = gradient_analysis['gradient_magnitudes']
            f.write(f"梯度强度统计:\n")
            f.write(f"  平均值: {gradient_magnitudes.mean():.4f}\n")
            f.write(f"  标准差: {gradient_magnitudes.std():.4f}\n")
            f.write(f"  最大值: {gradient_magnitudes.max():.4f}\n")
            f.write(f"  最小值: {gradient_magnitudes.min():.4f}\n\n")
            
            # 6. 关键发现
            f.write("6. 关键发现与洞察\n")
            f.write("-" * 30 + "\n")
            
            # 分析结果并生成洞察
            insights = self.generate_insights()
            for insight in insights:
                f.write(f"• {insight}\n")
        
        print(f"✅ 分析报告已保存到 {report_path}")
        return report_path
    
    def generate_insights(self):
        """基于分析结果生成洞察"""
        insights = []
        
        # 1. 密度比例洞察
        density_ratio = self.correspondence_data['density_ratio']
        if density_ratio > 50:
            insights.append(f"密度比例极高({density_ratio:.1f}x)，说明3DGS进行了大量densification操作")
        elif density_ratio > 10:
            insights.append(f"密度比例较高({density_ratio:.1f}x)，存在显著的densification")
        else:
            insights.append(f"密度比例适中({density_ratio:.1f}x)，densification程度温和")
        
        # 2. 对应关系距离洞察
        correspondence_results = self.correspondence_data['correspondence_results']
        mean_distance = correspondence_results[1]['mean_distance']
        
        if mean_distance < 0.01:
            insights.append("COLMAP点与最近高斯球距离很小，空间对应关系较好")
        elif mean_distance < 0.1:
            insights.append("COLMAP点与最近高斯球距离适中，存在一定的空间偏移")
        else:
            insights.append("COLMAP点与最近高斯球距离较大，空间对应关系较弱")
        
        # 3. 梯度相关性洞察
        gradient_analysis = self.correspondence_data['gradient_analysis']
        correlation = gradient_analysis['gradient_density_correlation']
        
        if abs(correlation) > 0.3:
            insights.append(f"梯度与densification存在{'正' if correlation > 0 else '负'}相关性({correlation:.3f})，几何复杂度影响高斯球生成")
        else:
            insights.append(f"梯度与densification相关性较弱({correlation:.3f})，几何复杂度对高斯球生成影响有限")
        
        # 4. 密度分布洞察
        densification_analysis = self.correspondence_data['densification_analysis']
        density_categories = densification_analysis['density_categories']
        
        high_density_ratio = density_categories['high'].sum() / len(self.colmap_points)
        if high_density_ratio > 0.3:
            insights.append(f"高密度区域占比较大({high_density_ratio:.1%})，场景存在大量细节需要densification")
        else:
            insights.append(f"高密度区域占比适中({high_density_ratio:.1%})，densification分布相对均匀")
        
        # 5. 空间分布洞察
        colmap_stats = self.correspondence_data['colmap_stats']
        gaussian_stats = self.correspondence_data['gaussian_stats']
        
        range_ratio = gaussian_stats['bounds']['range'] / colmap_stats['bounds']['range']
        range_ratio_mean = range_ratio.mean()
        
        if range_ratio_mean > 1.2:
            insights.append("高斯球分布范围超出COLMAP点云，存在场景外推生成")
        elif range_ratio_mean < 0.8:
            insights.append("高斯球分布范围小于COLMAP点云，可能存在边界收缩")
        else:
            insights.append("高斯球与COLMAP点云空间范围基本一致")
        
        return insights
    
    def extract_training_features(self, save_dir='correspondence_analysis'):
        """提取可用于训练的特征"""
        print("\n=== 提取训练特征 ===")
        
        # 1. 为每个COLMAP点提取特征
        features = {}
        
        # 基础位置特征
        features['positions'] = self.colmap_points
        
        # 局部密度特征
        tree = KDTree(self.gaussian_positions)
        for radius in [0.05, 0.1, 0.2]:
            neighbor_counts = tree.query_radius(self.colmap_points, r=radius, count_only=True)
            features[f'density_r{radius}'] = neighbor_counts
        
        # 梯度特征
        gradient_analysis = self.correspondence_data['gradient_analysis']
        features['gradient_magnitude'] = gradient_analysis['gradient_magnitudes']
        features['gradient_vectors'] = gradient_analysis['colmap_gradients']
        
        # 最近邻距离特征
        correspondence_results = self.correspondence_data['correspondence_results']
        for k in [1, 5, 10]:
            if k in correspondence_results:
                distances = correspondence_results[k]['distances']
                if k == 1:
                    features[f'nearest_distance_k{k}'] = distances.flatten()
                else:
                    features[f'nearest_distance_k{k}'] = distances.mean(axis=1)
        
        # 局部几何复杂度
        complexity_scores = self.compute_local_complexity()
        features['geometric_complexity'] = complexity_scores
        
        # 保存特征
        feature_path = f'{save_dir}/training_features.npz'
        np.savez(feature_path, **features)
        
        print(f"✅ 训练特征已保存到 {feature_path}")
        print(f"📊 提取的特征:")
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        
        return features
    
    def compute_local_complexity(self, k=15):
        """计算局部几何复杂度"""
        tree = KDTree(self.colmap_points)
        complexity_scores = np.zeros(len(self.colmap_points))
        
        for i, point in enumerate(self.colmap_points):
            # 找到k个最近邻
            distances, indices = tree.query([point], k=k+1)
            neighbors = self.colmap_points[indices[0][1:]]  # 排除自己
            
            if len(neighbors) > 3:
                # 计算局部表面的主曲率
                centered_neighbors = neighbors - point
                
                # 计算协方差矩阵
                cov_matrix = np.cov(centered_neighbors.T)
                eigenvals = np.linalg.eigvals(cov_matrix)
                eigenvals = np.sort(eigenvals)[::-1]  # 降序排列
                
                # 使用特征值比例作为复杂度指标
                if eigenvals[0] > 1e-8:
                    # 线性度：λ1 >> λ2, λ3
                    linearity = (eigenvals[0] - eigenvals[1]) / eigenvals[0]
                    # 平面度：λ2 >> λ3
                    planarity = (eigenvals[1] - eigenvals[2]) / eigenvals[0] if eigenvals[0] > 1e-8 else 0
                    # 球形度：λ1 ≈ λ2 ≈ λ3
                    sphericity = eigenvals[2] / eigenvals[0] if eigenvals[0] > 1e-8 else 0
                    
                    # 复杂度 = 1 - 最大的形状特征（越不规则越复杂）
                    complexity = 1.0 - max(linearity, planarity, sphericity)
                else:
                    complexity = 1.0  # 如果点重合，认为是高复杂度
                
                complexity_scores[i] = complexity
        
        return complexity_scores
    
    def run_full_analysis(self, save_dir='correspondence_analysis'):
        """运行完整的对应关系分析"""
        print("🎯 开始COLMAP点云 vs 3DGS高斯球对应关系分析")
        print("=" * 60)
        
        # 1. 加载数据
        if not self.load_data():
            print("❌ 数据加载失败，分析终止")
            return False
        
        # 2. 空间分布分析
        self.analyze_spatial_distribution()
        
        # 3. 对应关系分析
        self.analyze_correspondence_patterns()
        
        # 4. 生成可视化
        self.visualize_correspondence_analysis(save_dir)
        
        # 5. 生成报告
        self.generate_correspondence_report(save_dir)
        
        # 6. 提取训练特征
        self.extract_training_features(save_dir)
        
        print("\n" + "=" * 60)
        print("✅ 对应关系分析完成！")
        print(f"📁 结果保存在: {save_dir}/")
        print("📋 主要输出文件:")
        print(f"  • correspondence_analysis_report.txt - 分析报告")
        print(f"  • training_features.npz - 训练特征")
        print(f"  • *.png - 可视化图表")
        print(f"  • interactive_3d_comparison.html - 交互式3D图")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='COLMAP点云与3DGS高斯球对应关系分析')
    parser.add_argument('--colmap-points', required=True, 
                       help='COLMAP点云文件路径 (points3D.ply)')
    parser.add_argument('--teacher-gaussians', required=True,
                       help='Teacher高斯球文件路径 (gaussian_ball.ply)')
    parser.add_argument('--output-dir', default='correspondence_analysis',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.colmap_points).exists():
        print(f"❌ COLMAP点云文件不存在: {args.colmap_points}")
        return
    
    if not Path(args.teacher_gaussians).exists():
        print(f"❌ Teacher高斯球文件不存在: {args.teacher_gaussians}")
        return
    
    # 创建分析器并运行分析
    analyzer = CorrespondenceAnalyzer(
        colmap_points_path=args.colmap_points,
        teacher_gaussians_path=args.teacher_gaussians
    )
    
    success = analyzer.run_full_analysis(save_dir=args.output_dir)
    
    if success:
        print(f"\n🎉 分析成功完成！查看 {args.output_dir}/ 获取详细结果。")
    else:
        print("\n❌ 分析失败，请检查输入文件和参数。")

if __name__ == "__main__":
    main()