import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json
from collections import defaultdict

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel

def load_gaussians(ply_path):
    """加载高斯球模型"""
    print(f"🎯 加载高斯球: {ply_path}")
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    return gaussians

def analyze_by_position_layers(gaussians, num_layers=10):
    """按位置分层分析"""
    print(f"\n📍 按位置分层分析 ({num_layers}层)")
    
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()  # 确保是1维
    scaling = gaussians.get_scaling.detach().cpu().numpy()
    
    # 过滤掉NaN值 - 修复：opacity是1维的
    valid_mask = ~np.isnan(xyz).any(axis=1) & ~np.isnan(opacity)
    xyz = xyz[valid_mask]
    opacity = opacity[valid_mask]
    scaling = scaling[valid_mask]
    
    print(f"  有效高斯球数量: {len(xyz)}")
    
    layers_data = []
    
    # 按Z轴分层（深度）
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    z_step = (z_max - z_min) / num_layers
    
    print(f"  Z轴范围: [{z_min:.3f}, {z_max:.3f}]")
    print(f"  每层厚度: {z_step:.3f}")
    
    for i in range(num_layers):
        z_start = z_min + i * z_step
        z_end = z_min + (i + 1) * z_step
        
        # 最后一层包含边界
        if i == num_layers - 1:
            layer_mask = (xyz[:, 2] >= z_start) & (xyz[:, 2] <= z_end)
        else:
            layer_mask = (xyz[:, 2] >= z_start) & (xyz[:, 2] < z_end)
        
        layer_xyz = xyz[layer_mask]
        layer_opacity = opacity[layer_mask]
        layer_scaling = scaling[layer_mask]
        
        if len(layer_xyz) == 0:
            continue
        
        # 统计信息
        layer_info = {
            'layer_id': i,
            'z_range': [z_start, z_end],
            'count': len(layer_xyz),
            'opacity_stats': {
                'mean': layer_opacity.mean(),
                'std': layer_opacity.std(),
                'min': layer_opacity.min(),
                'max': layer_opacity.max()
            },
            'scale_stats': {
                'mean': layer_scaling.mean(axis=0),
                'std': layer_scaling.std(axis=0),
                'volume_mean': np.prod(layer_scaling, axis=1).mean()
            },
            'position_stats': {
                'x_range': [layer_xyz[:, 0].min(), layer_xyz[:, 0].max()],
                'y_range': [layer_xyz[:, 1].min(), layer_xyz[:, 1].max()],
                'density': len(layer_xyz) / ((layer_xyz[:, 0].max() - layer_xyz[:, 0].min() + 1e-6) * 
                                           (layer_xyz[:, 1].max() - layer_xyz[:, 1].min() + 1e-6))
            }
        }
        
        layers_data.append(layer_info)
        
        print(f"  层 {i:2d} [{z_start:7.2f}, {z_end:7.2f}]: "
              f"{len(layer_xyz):6d}个高斯球, "
              f"平均透明度={layer_opacity.mean():.3f}, "
              f"平均体积={np.prod(layer_scaling, axis=1).mean():.6f}")
    
    return layers_data

def analyze_by_opacity_layers(gaussians, num_layers=5):
    """按透明度分层分析"""
    print(f"\n👻 按透明度分层分析 ({num_layers}层)")
    
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()  # 确保是1维
    scaling = gaussians.get_scaling.detach().cpu().numpy()
    
    # 过滤掉NaN值 - 修复：opacity是1维的
    valid_mask = ~np.isnan(xyz).any(axis=1) & ~np.isnan(opacity)
    xyz = xyz[valid_mask]
    opacity = opacity[valid_mask]
    scaling = scaling[valid_mask]
    
    # 按透明度分层
    opacity_thresholds = np.linspace(0, 1, num_layers + 1)
    layers_data = []
    
    print(f"  透明度范围: [{opacity.min():.3f}, {opacity.max():.3f}]")
    
    for i in range(num_layers):
        opacity_start = opacity_thresholds[i]
        opacity_end = opacity_thresholds[i + 1]
        
        if i == num_layers - 1:
            layer_mask = (opacity >= opacity_start) & (opacity <= opacity_end)
        else:
            layer_mask = (opacity >= opacity_start) & (opacity < opacity_end)
        
        layer_xyz = xyz[layer_mask]
        layer_opacity = opacity[layer_mask]
        layer_scaling = scaling[layer_mask]
        
        if len(layer_xyz) == 0:
            continue
        
        layer_info = {
            'layer_id': i,
            'opacity_range': [opacity_start, opacity_end],
            'count': len(layer_xyz),
            'position_spread': {
                'x_std': layer_xyz[:, 0].std(),
                'y_std': layer_xyz[:, 1].std(),
                'z_std': layer_xyz[:, 2].std(),
                'z_range': [layer_xyz[:, 2].min(), layer_xyz[:, 2].max()]
            },
            'scale_stats': {
                'mean': layer_scaling.mean(axis=0),
                'volume_mean': np.prod(layer_scaling, axis=1).mean()
            }
        }
        
        layers_data.append(layer_info)
        
        print(f"  透明度 [{opacity_start:.2f}, {opacity_end:.2f}]: "
              f"{len(layer_xyz):6d}个高斯球, "
              f"Z范围=[{layer_xyz[:, 2].min():6.2f}, {layer_xyz[:, 2].max():6.2f}], "
              f"平均体积={np.prod(layer_scaling, axis=1).mean():.6f}")
    
    return layers_data

def analyze_by_scale_layers(gaussians, num_layers=5):
    """按大小分层分析"""
    print(f"\n📏 按大小分层分析 ({num_layers}层)")
    
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()  # 确保是1维
    scaling = gaussians.get_scaling.detach().cpu().numpy()
    
    # 过滤掉NaN值 - 修复：opacity是1维的
    valid_mask = ~np.isnan(xyz).any(axis=1) & ~np.isnan(opacity)
    xyz = xyz[valid_mask]
    opacity = opacity[valid_mask]
    scaling = scaling[valid_mask]
    
    # 计算体积（近似）
    volumes = np.prod(scaling, axis=1)
    
    # 按体积分层（使用对数尺度，因为体积差异可能很大）
    log_volumes = np.log10(volumes + 1e-10)
    volume_thresholds = np.linspace(log_volumes.min(), log_volumes.max(), num_layers + 1)
    layers_data = []
    
    print(f"  体积范围: [{volumes.min():.2e}, {volumes.max():.2e}]")
    
    for i in range(num_layers):
        vol_start = 10 ** volume_thresholds[i]
        vol_end = 10 ** volume_thresholds[i + 1]
        
        if i == num_layers - 1:
            layer_mask = (volumes >= vol_start) & (volumes <= vol_end)
        else:
            layer_mask = (volumes >= vol_start) & (volumes < vol_end)
        
        layer_xyz = xyz[layer_mask]
        layer_opacity = opacity[layer_mask]
        layer_scaling = scaling[layer_mask]
        layer_volumes = volumes[layer_mask]
        
        if len(layer_xyz) == 0:
            continue
        
        layer_info = {
            'layer_id': i,
            'volume_range': [vol_start, vol_end],
            'count': len(layer_xyz),
            'opacity_stats': {
                'mean': layer_opacity.mean(),
                'std': layer_opacity.std()
            },
            'position_spread': {
                'z_range': [layer_xyz[:, 2].min(), layer_xyz[:, 2].max()],
                'z_std': layer_xyz[:, 2].std()
            },
            'scale_details': {
                'x_scale_mean': layer_scaling[:, 0].mean(),
                'y_scale_mean': layer_scaling[:, 1].mean(),
                'z_scale_mean': layer_scaling[:, 2].mean(),
                'aspect_ratio': (layer_scaling.max(axis=1) / layer_scaling.min(axis=1)).mean()
            }
        }
        
        layers_data.append(layer_info)
        
        print(f"  体积 [{vol_start:.2e}, {vol_end:.2e}]: "
              f"{len(layer_xyz):6d}个高斯球, "
              f"平均透明度={layer_opacity.mean():.3f}, "
              f"Z范围=[{layer_xyz[:, 2].min():6.2f}, {layer_xyz[:, 2].max():6.2f}]")
    
    return layers_data

def create_layer_visualizations(position_layers, opacity_layers, scale_layers, output_dir="layer_analysis"):
    """创建分层可视化"""
    print(f"\n📊 创建可视化图表...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 位置分层统计图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 每层高斯球数量
    layer_ids = [layer['layer_id'] for layer in position_layers]
    counts = [layer['count'] for layer in position_layers]
    ax1.bar(layer_ids, counts)
    ax1.set_title('每层高斯球数量 (按Z轴)')
    ax1.set_xlabel('层ID')
    ax1.set_ylabel('高斯球数量')
    
    # 每层平均透明度
    avg_opacities = [layer['opacity_stats']['mean'] for layer in position_layers]
    ax2.plot(layer_ids, avg_opacities, 'o-')
    ax2.set_title('每层平均透明度')
    ax2.set_xlabel('层ID')
    ax2.set_ylabel('平均透明度')
    
    # 每层平均体积
    avg_volumes = [layer['scale_stats']['volume_mean'] for layer in position_layers]
    ax3.semilogy(layer_ids, avg_volumes, 's-')
    ax3.set_title('每层平均体积 (对数尺度)')
    ax3.set_xlabel('层ID')
    ax3.set_ylabel('平均体积')
    
    # 每层密度
    densities = [layer['position_stats']['density'] for layer in position_layers]
    ax4.plot(layer_ids, densities, '^-')
    ax4.set_title('每层空间密度')
    ax4.set_xlabel('层ID')
    ax4.set_ylabel('密度 (个/单位面积)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_layers_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 透明度分层分析
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    opacity_ranges = [f"{layer['opacity_range'][0]:.2f}-{layer['opacity_range'][1]:.2f}" 
                     for layer in opacity_layers]
    opacity_counts = [layer['count'] for layer in opacity_layers]
    
    ax1.bar(range(len(opacity_layers)), opacity_counts)
    ax1.set_title('不同透明度层的高斯球数量')
    ax1.set_xticks(range(len(opacity_layers)))
    ax1.set_xticklabels(opacity_ranges, rotation=45)
    ax1.set_ylabel('高斯球数量')
    
    # Z轴分布
    z_spreads = [layer['position_spread']['z_std'] for layer in opacity_layers]
    ax2.plot(range(len(opacity_layers)), z_spreads, 'o-')
    ax2.set_title('不同透明度层的Z轴分散度')
    ax2.set_xticks(range(len(opacity_layers)))
    ax2.set_xticklabels(opacity_ranges, rotation=45)
    ax2.set_ylabel('Z轴标准差')
    
    # 平均体积
    opacity_volumes = [layer['scale_stats']['volume_mean'] for layer in opacity_layers]
    ax3.semilogy(range(len(opacity_layers)), opacity_volumes, 's-')
    ax3.set_title('不同透明度层的平均体积')
    ax3.set_xticks(range(len(opacity_layers)))
    ax3.set_xticklabels(opacity_ranges, rotation=45)
    ax3.set_ylabel('平均体积')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'opacity_layers_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 大小分层分析
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scale_ranges = [f"{layer['volume_range'][0]:.1e}-{layer['volume_range'][1]:.1e}" 
                   for layer in scale_layers]
    scale_counts = [layer['count'] for layer in scale_layers]
    
    ax1.bar(range(len(scale_layers)), scale_counts)
    ax1.set_title('不同大小层的高斯球数量')
    ax1.set_xticks(range(len(scale_layers)))
    ax1.set_xticklabels(scale_ranges, rotation=45)
    ax1.set_ylabel('高斯球数量')
    
    # 透明度分布
    scale_opacities = [layer['opacity_stats']['mean'] for layer in scale_layers]
    ax2.plot(range(len(scale_layers)), scale_opacities, 'o-')
    ax2.set_title('不同大小层的平均透明度')
    ax2.set_xticks(range(len(scale_layers)))
    ax2.set_xticklabels(scale_ranges, rotation=45)
    ax2.set_ylabel('平均透明度')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_layers_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化图表已保存到 {output_dir}/")

def save_layer_analysis(position_layers, opacity_layers, scale_layers, output_dir="layer_analysis"):
    """保存分层分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    analysis_data = {
        'position_layers': position_layers,
        'opacity_layers': opacity_layers,
        'scale_layers': scale_layers,
        'summary': {
            'total_position_layers': len(position_layers),
            'total_opacity_layers': len(opacity_layers),
            'total_scale_layers': len(scale_layers),
            'analysis_date': str(np.datetime64('now'))
        }
    }
    
    output_file = os.path.join(output_dir, 'layer_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"✅ 分析数据已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='高斯球分层分析工具')
    parser.add_argument('--ply-path', type=str, required=True, help='PLY文件路径')
    parser.add_argument('--position-layers', type=int, default=10, help='位置分层数量')
    parser.add_argument('--opacity-layers', type=int, default=5, help='透明度分层数量')
    parser.add_argument('--scale-layers', type=int, default=5, help='大小分层数量')
    parser.add_argument('--output-dir', type=str, default='layer_analysis', help='输出目录')
    
    args = parser.parse_args()
    
    print("🔍 高斯球分层分析工具")
    print("=" * 60)
    
    # 加载高斯球
    gaussians = load_gaussians(args.ply_path)
    
    # 按位置分层
    position_layers = analyze_by_position_layers(gaussians, args.position_layers)
    
    # 按透明度分层
    opacity_layers = analyze_by_opacity_layers(gaussians, args.opacity_layers)
    
    # 按大小分层
    scale_layers = analyze_by_scale_layers(gaussians, args.scale_layers)
    
    # 创建可视化
    create_layer_visualizations(position_layers, opacity_layers, scale_layers, args.output_dir)
    
    # 保存分析结果
    save_layer_analysis(position_layers, opacity_layers, scale_layers, args.output_dir)
    
    print("\n🎉 分层分析完成!")
    print(f"📊 结果保存在: {args.output_dir}/")

if __name__ == "__main__":
    main() 