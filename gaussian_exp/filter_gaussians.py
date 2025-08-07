#!/usr/bin/env python3
# 过滤异常的高斯球
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import numpy as np
from scene import GaussianModel
from plyfile import PlyData, PlyElement

def filter_gaussians(input_ply, output_ply):
    print(f"🔍 加载高斯球: {input_ply}")
    
    # 加载高斯球
    gaussians = GaussianModel(3)
    gaussians.load_ply(input_ply)
    
    # 获取数据
    positions = gaussians.get_xyz.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    rotations = gaussians.get_rotation.detach().cpu().numpy()
    features = gaussians.get_features.detach().cpu().numpy()
    
    print(f"📊 原始高斯球数量: {len(positions)}")
    
    # 过滤条件
    valid_mask = np.ones(len(positions), dtype=bool)
    
    # 1. 过滤NaN值
    nan_mask = np.isnan(positions).any(axis=1)
    valid_mask &= ~nan_mask
    print(f"❌ 过滤NaN位置: {nan_mask.sum()} 个")
    
    # 2. 过滤极大缩放
    large_scale_mask = scales.max(axis=1) > 0.5  # 更严格的阈值
    valid_mask &= ~large_scale_mask
    print(f"❌ 过滤极大缩放 (>0.5): {large_scale_mask.sum()} 个")
    
    # 3. 过滤极小不透明度
    low_opacity_mask = opacities.flatten() < 0.05  # 提高阈值
    valid_mask &= ~low_opacity_mask
    print(f"❌ 过滤极小不透明度 (<0.05): {low_opacity_mask.sum()} 个")
    
    # 4. 过滤极远位置
    far_mask = np.abs(positions).max(axis=1) > 6.0  # truck场景应该在合理范围内
    valid_mask &= ~far_mask
    print(f"❌ 过滤极远位置 (>6.0): {far_mask.sum()} 个")
    
    # 5. 过滤极小缩放（基本不可见的）
    tiny_scale_mask = scales.max(axis=1) < 0.001
    valid_mask &= ~tiny_scale_mask
    print(f"❌ 过滤极小缩放 (<0.001): {tiny_scale_mask.sum()} 个")
    
    # 应用过滤
    filtered_positions = positions[valid_mask]
    filtered_opacities = opacities[valid_mask]
    filtered_scales = scales[valid_mask]
    filtered_rotations = rotations[valid_mask]
    filtered_features = features[valid_mask]
    
    print(f"✅ 保留高斯球数量: {len(filtered_positions)}")
    print(f"📉 过滤比例: {(1 - len(filtered_positions)/len(positions))*100:.1f}%")
    
    # 保存过滤后的PLY文件
    save_filtered_ply(output_ply, filtered_positions, filtered_opacities, 
                     filtered_scales, filtered_rotations, filtered_features)
    
    print(f"💾 已保存到: {output_ply}")

def save_filtered_ply(path, positions, opacities, scales, rotations, features):
    """保存过滤后的PLY文件"""
    
    # 准备数据
    xyz = positions
    normals = np.zeros_like(xyz)
    
    # SH特征
    sh_features = features.reshape(len(features), -1)
    
    # 构建数据数组
    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
    dtype_full += [(attribute, 'f4') for attribute in ['f_dc_0', 'f_dc_1', 'f_dc_2']]
    for i in range(sh_features.shape[1] - 3):
        dtype_full.append((f'f_rest_{i}', 'f4'))
    dtype_full += [(attribute, 'f4') for attribute in ['opacity']]
    dtype_full += [(attribute, 'f4') for attribute in ['scale_0', 'scale_1', 'scale_2']]
    dtype_full += [(attribute, 'f4') for attribute in ['rot_0', 'rot_1', 'rot_2', 'rot_3']]
    
    # 组装数据
    data = np.zeros(len(xyz), dtype=dtype_full)
    data['x'] = xyz[:, 0]
    data['y'] = xyz[:, 1] 
    data['z'] = xyz[:, 2]
    data['nx'] = normals[:, 0]
    data['ny'] = normals[:, 1]
    data['nz'] = normals[:, 2]
    
    # SH特征
    data['f_dc_0'] = sh_features[:, 0]
    data['f_dc_1'] = sh_features[:, 1] 
    data['f_dc_2'] = sh_features[:, 2]
    for i in range(3, sh_features.shape[1]):
        data[f'f_rest_{i-3}'] = sh_features[:, i]
    
    data['opacity'] = opacities.flatten()
    data['scale_0'] = scales[:, 0]
    data['scale_1'] = scales[:, 1]
    data['scale_2'] = scales[:, 2]
    data['rot_0'] = rotations[:, 0]
    data['rot_1'] = rotations[:, 1]
    data['rot_2'] = rotations[:, 2]
    data['rot_3'] = rotations[:, 3]
    
    # 保存PLY
    element = PlyElement.describe(data, 'vertex')
    PlyData([element]).write(path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    filter_gaussians(args.input, args.output) 