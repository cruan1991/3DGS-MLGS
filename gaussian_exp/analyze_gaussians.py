#!/usr/bin/env python3
# 分析高斯球统计信息
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import numpy as np
from scene import GaussianModel

def analyze_gaussians(ply_path):
    print(f"🔍 分析高斯球: {ply_path}")
    
    # 加载高斯球
    gaussians = GaussianModel(3)  # 假设sh_degree=3
    gaussians.load_ply(ply_path)
    
    # 获取基本信息
    positions = gaussians.get_xyz.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    
    print(f"📊 基本统计:")
    print(f"  高斯球数量: {len(positions)}")
    print(f"  位置范围: X[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"             Y[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"             Z[{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    
    print(f"  不透明度: 平均={opacities.mean():.4f}, 标准差={opacities.std():.4f}")
    print(f"           最小={opacities.min():.4f}, 最大={opacities.max():.4f}")
    
    print(f"  缩放: 平均={scales.mean():.4f}, 标准差={scales.std():.4f}")
    print(f"        最小={scales.min():.4f}, 最大={scales.max():.4f}")
    
    # 检查异常值
    print(f"\n🚨 异常值检测:")
    
    # 检查极大的高斯球
    large_scales = scales.max(axis=1) > 1.0  # 缩放超过1.0的
    print(f"  缩放 > 1.0: {large_scales.sum()} 个 ({large_scales.sum()/len(scales)*100:.1f}%)")
    
    # 检查极小的不透明度
    low_opacity = opacities < 0.01
    print(f"  不透明度 < 0.01: {low_opacity.sum()} 个 ({low_opacity.sum()/len(opacities)*100:.1f}%)")
    
    # 检查极远的位置
    far_positions = np.abs(positions).max(axis=1) > 10.0
    print(f"  位置 > 10.0: {far_positions.sum()} 个 ({far_positions.sum()/len(positions)*100:.1f}%)")
    
    # 建议过滤
    suggested_filter = large_scales | low_opacity.flatten() | far_positions
    print(f"  建议过滤: {suggested_filter.sum()} 个 ({suggested_filter.sum()/len(positions)*100:.1f}%)")
    
    return suggested_filter

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply-path', required=True)
    args = parser.parse_args()
    
    analyze_gaussians(args.ply_path) 
# 分析高斯球统计信息
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import numpy as np
from scene import GaussianModel

def analyze_gaussians(ply_path):
    print(f"🔍 分析高斯球: {ply_path}")
    
    # 加载高斯球
    gaussians = GaussianModel(3)  # 假设sh_degree=3
    gaussians.load_ply(ply_path)
    
    # 获取基本信息
    positions = gaussians.get_xyz.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    
    print(f"📊 基本统计:")
    print(f"  高斯球数量: {len(positions)}")
    print(f"  位置范围: X[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"             Y[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"             Z[{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    
    print(f"  不透明度: 平均={opacities.mean():.4f}, 标准差={opacities.std():.4f}")
    print(f"           最小={opacities.min():.4f}, 最大={opacities.max():.4f}")
    
    print(f"  缩放: 平均={scales.mean():.4f}, 标准差={scales.std():.4f}")
    print(f"        最小={scales.min():.4f}, 最大={scales.max():.4f}")
    
    # 检查异常值
    print(f"\n🚨 异常值检测:")
    
    # 检查极大的高斯球
    large_scales = scales.max(axis=1) > 1.0  # 缩放超过1.0的
    print(f"  缩放 > 1.0: {large_scales.sum()} 个 ({large_scales.sum()/len(scales)*100:.1f}%)")
    
    # 检查极小的不透明度
    low_opacity = opacities < 0.01
    print(f"  不透明度 < 0.01: {low_opacity.sum()} 个 ({low_opacity.sum()/len(opacities)*100:.1f}%)")
    
    # 检查极远的位置
    far_positions = np.abs(positions).max(axis=1) > 10.0
    print(f"  位置 > 10.0: {far_positions.sum()} 个 ({far_positions.sum()/len(positions)*100:.1f}%)")
    
    # 建议过滤
    suggested_filter = large_scales | low_opacity.flatten() | far_positions
    print(f"  建议过滤: {suggested_filter.sum()} 个 ({suggested_filter.sum()/len(positions)*100:.1f}%)")
    
    return suggested_filter

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply-path', required=True)
    args = parser.parse_args()
    
    analyze_gaussians(args.ply_path) 