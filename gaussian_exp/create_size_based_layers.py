import os
import sys
import torch
import numpy as np
import json
from pathlib import Path

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel
from plyfile import PlyData, PlyElement

def create_size_based_layers(ply_path, analysis_results_path, output_dir='size_based_layers'):
    """根据尺寸分析结果创建分层PLY文件"""
    print("🔄 创建按尺寸分层的PLY文件...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载分析结果
    with open(analysis_results_path, 'r') as f:
        analysis_results = json.load(f)
    
    layer_suggestions = analysis_results['layer_suggestions']
    thresholds = analysis_results['thresholds']
    
    print(f"📊 加载分层方案: {len(layer_suggestions)} 层")
    for layer in layer_suggestions:
        print(f"  层{layer['layer_id']} ({layer['name']}): {layer['count']:,}球 ({layer['percentage']:.1f}%)")
    
    # 加载原始高斯球
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    # 获取所有参数
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    features_dc = gaussians._features_dc.detach().cpu().numpy()
    features_rest = gaussians._features_rest.detach().cpu().numpy()
    scaling = gaussians.get_scaling.detach().cpu().numpy()
    rotation = gaussians.get_rotation.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy()
    
    # 计算平均尺寸
    avg_scale = np.mean(scaling, axis=1)
    
    print(f"\n📏 原始模型参数:")
    print(f"  位置: {xyz.shape}")
    print(f"  DC特征: {features_dc.shape}")
    print(f"  Rest特征: {features_rest.shape}")
    print(f"  缩放: {scaling.shape}")
    print(f"  旋转: {rotation.shape}")
    print(f"  透明度: {opacity.shape}")
    
    # 为每层创建PLY文件
    layer_files = []
    
    for layer in layer_suggestions:
        layer_id = layer['layer_id']
        layer_name = layer['name']
        print(f"\n🎯 创建层{layer_id} ({layer_name})...")
        
        # 根据阈值创建mask
        if layer_id == 0:
            mask = avg_scale <= thresholds[0]
        elif layer_id == len(layer_suggestions) - 1:
            mask = avg_scale > thresholds[-1]
        else:
            mask = (avg_scale > thresholds[layer_id-1]) & (avg_scale <= thresholds[layer_id])
        
        layer_count = np.sum(mask)
        print(f"  筛选到 {layer_count:,} 个高斯球")
        
        if layer_count == 0:
            print(f"  ⚠️ 层{layer_id}为空，跳过")
            continue
        
        # 提取该层的参数
        layer_xyz = xyz[mask]
        layer_features_dc = features_dc[mask]
        layer_features_rest = features_rest[mask]
        layer_scaling = scaling[mask]
        layer_rotation = rotation[mask]
        layer_opacity = opacity[mask]
        
        # 构造PLY数据
        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # DC特征
            for i in range(layer_features_dc.shape[1] * layer_features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            # Rest特征  
            for i in range(layer_features_rest.shape[1] * layer_features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(layer_scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(layer_rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l
        
        # 准备数据
        normals = np.zeros_like(layer_xyz)  # 法向量设为0
        
        # DC特征reshape
        f_dc = layer_features_dc.reshape((layer_features_dc.shape[0], -1))
        f_rest = layer_features_rest.reshape((layer_features_rest.shape[0], -1))
        
        # 组合所有属性
        attributes = np.concatenate([
            layer_xyz, normals, f_dc, f_rest, 
            layer_opacity, layer_scaling, layer_rotation
        ], axis=1)
        
        # 构造PLY元素
        elements = np.empty(layer_count, dtype=[
            (attr, 'f4') for attr in construct_list_of_attributes()
        ])
        
        attr_names = construct_list_of_attributes()
        for i, attr_name in enumerate(attr_names):
            elements[attr_name] = attributes[:, i]
        
        # 保存PLY文件
        scale_range = layer['threshold_range'].replace('≤', 'le').replace('>', 'gt').replace('~', '_to_')
        avg_scale_in_layer = np.mean(avg_scale[mask])
        filename = f"size_layer_{layer_id}_{layer_name}_{scale_range}_{layer_count}balls_avg{avg_scale_in_layer:.6f}.ply"
        layer_file_path = os.path.join(output_dir, filename)
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(layer_file_path)
        
        layer_files.append(layer_file_path)
        print(f"  ✅ 保存: {filename}")
        print(f"     平均尺寸: {avg_scale_in_layer:.6f}")
        print(f"     尺寸范围: {np.min(avg_scale[mask]):.6f} ~ {np.max(avg_scale[mask]):.6f}")
    
    print(f"\n📁 单层文件创建完成: {len(layer_files)} 个")
    
    # 创建渐进式累积文件
    print(f"\n🔄 创建渐进式累积文件...")
    progressive_files = []
    
    for end_layer in range(len(layer_suggestions)):
        print(f"\n🎯 创建累积文件: 层0到层{end_layer}...")
        
        # 合并mask
        combined_mask = np.zeros(len(avg_scale), dtype=bool)
        total_count = 0
        
        for layer_id in range(end_layer + 1):
            layer = layer_suggestions[layer_id]
            
            # 重新计算该层的mask
            if layer_id == 0:
                mask = avg_scale <= thresholds[0]
            elif layer_id == len(layer_suggestions) - 1:
                mask = avg_scale > thresholds[-1]
            else:
                mask = (avg_scale > thresholds[layer_id-1]) & (avg_scale <= thresholds[layer_id])
            
            combined_mask |= mask
            total_count += np.sum(mask)
        
        print(f"  累积高斯球数: {np.sum(combined_mask):,}")
        
        if np.sum(combined_mask) == 0:
            print(f"  ⚠️ 累积层为空，跳过")
            continue
        
        # 提取累积参数
        prog_xyz = xyz[combined_mask]
        prog_features_dc = features_dc[combined_mask]
        prog_features_rest = features_rest[combined_mask]
        prog_scaling = scaling[combined_mask]
        prog_rotation = rotation[combined_mask]
        prog_opacity = opacity[combined_mask]
        
        # 构造PLY数据 (正确版本)
        def construct_progressive_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # DC特征
            for i in range(prog_features_dc.shape[1] * prog_features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            # Rest特征  
            for i in range(prog_features_rest.shape[1] * prog_features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(prog_scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(prog_rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l
        
        normals = np.zeros_like(prog_xyz)
        f_dc = prog_features_dc.reshape((prog_features_dc.shape[0], -1))
        f_rest = prog_features_rest.reshape((prog_features_rest.shape[0], -1))
        
        attributes = np.concatenate([
            prog_xyz, normals, f_dc, f_rest, 
            prog_opacity, prog_scaling, prog_rotation
        ], axis=1)
        
        elements = np.empty(np.sum(combined_mask), dtype=[
            (attr, 'f4') for attr in construct_progressive_attributes()
        ])
        
        attr_names = construct_progressive_attributes()
        for i, attr_name in enumerate(attr_names):
            elements[attr_name] = attributes[:, i]
        
        # 保存渐进式文件
        if end_layer == 0:
            filename = f"size_progressive_S0_{np.sum(combined_mask)}balls.ply"
        else:
            # 正确的累积命名：S0_S1_S2...S{end_layer}
            layer_names = '_'.join([f"S{i}" for i in range(end_layer + 1)])
            filename = f"size_progressive_{layer_names}_{np.sum(combined_mask)}balls.ply"
        
        prog_file_path = os.path.join(output_dir, filename)
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(prog_file_path)
        
        progressive_files.append(prog_file_path)
        print(f"  ✅ 保存: {filename}")
    
    print(f"\n📈 渐进式文件创建完成: {len(progressive_files)} 个")
    
    # 保存文件清单
    file_manifest = {
        'single_layers': [os.path.basename(f) for f in layer_files],
        'progressive_layers': [os.path.basename(f) for f in progressive_files],
        'layer_info': layer_suggestions,
        'thresholds': thresholds,
        'total_gaussians': len(avg_scale)
    }
    
    manifest_path = os.path.join(output_dir, 'size_layers_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(file_manifest, f, indent=2)
    
    print(f"✅ 文件清单保存: {manifest_path}")
    
    return layer_files, progressive_files

def main():
    print("🔄 按尺寸创建高斯球分层文件")
    print("=" * 50)
    
    # 文件路径
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    analysis_results_path = "./scale_analysis/scale_analysis_results.json"
    
    if not os.path.exists(ply_path):
        print(f"❌ PLY文件不存在: {ply_path}")
        return
    
    if not os.path.exists(analysis_results_path):
        print(f"❌ 分析结果文件不存在: {analysis_results_path}")
        print("请先运行 analyze_scale_distribution.py")
        return
    
    # 创建分层文件
    layer_files, progressive_files = create_size_based_layers(
        ply_path, analysis_results_path
    )
    
    print(f"\n🎉 尺寸分层文件创建完成!")
    print(f"📁 输出目录: size_based_layers/")
    print(f"📊 单层文件: {len(layer_files)} 个")
    print(f"📈 渐进文件: {len(progressive_files)} 个")
    print(f"📋 总计: {len(layer_files) + len(progressive_files)} 个PLY文件")

if __name__ == "__main__":
    main() 