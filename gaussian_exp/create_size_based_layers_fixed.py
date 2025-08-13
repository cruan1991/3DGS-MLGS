import os
import sys
import torch
import numpy as np
import json
from plyfile import PlyData, PlyElement

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel

def create_size_based_layers_fixed(ply_path, analysis_results_path, output_dir='size_based_layers_fixed'):
    """修复版的按尺寸分层生成，使用与原始save_ply完全相同的逻辑"""
    print("🔄 修复版按尺寸创建高斯球分层文件")
    print("=" * 50)
    print("🔧 使用与原始GaussianModel.save_ply完全相同的数据格式")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载分析结果
    if not os.path.exists(analysis_results_path):
        print(f"❌ 分析结果文件不存在: {analysis_results_path}")
        return
    
    with open(analysis_results_path, 'r') as f:
        analysis_results = json.load(f)
    
    layer_suggestions = analysis_results['layer_suggestions']
    thresholds = [layer['threshold'] for layer in layer_suggestions[:-1]]
    
    print(f"📊 加载分层方案: {len(layer_suggestions)} 层")
    for i, layer in enumerate(layer_suggestions):
        print(f"  层{i} ({layer['name']}): {layer['count']:,}球 ({layer['percentage']:.1f}%)")
    
    # 加载原始模型
    print(f"\n📏 加载原始模型...")
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    # 获取所有参数 (使用与save_ply相同的格式)
    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    features_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    features_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacity = gaussians._opacity.detach().cpu().numpy()
    scaling = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()
    
    print(f"📏 原始模型参数 (处理后格式):")
    print(f"  位置: {xyz.shape}")
    print(f"  法向: {normals.shape}")
    print(f"  DC特征: {features_dc.shape}")
    print(f"  Rest特征: {features_rest.shape}")
    print(f"  缩放: {scaling.shape}")
    print(f"  旋转: {rotation.shape}")
    print(f"  透明度: {opacity.shape}")
    
    # 计算平均缩放（用于分层）
    avg_scale = np.mean(scaling, axis=1)
    
    # 生成完整模型作为参考
    print(f"\n📸 生成完整模型参考文件...")
    def save_gaussians_like_original(xyz_data, normals_data, f_dc_data, f_rest_data, 
                                   opacity_data, scale_data, rotation_data, output_path):
        """使用与原始save_ply完全相同的逻辑保存"""
        
        # 构造属性列表 (与原始逻辑相同)
        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # DC特征
            for i in range(f_dc_data.shape[1]):
                l.append('f_dc_{}'.format(i))
            # Rest特征  
            for i in range(f_rest_data.shape[1]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(scale_data.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(rotation_data.shape[1]):
                l.append('rot_{}'.format(i))
            return l
        
        # 组合所有属性 (与原始完全相同)
        attributes = np.concatenate((xyz_data, normals_data, f_dc_data, f_rest_data, 
                                   opacity_data, scale_data, rotation_data), axis=1)
        
        # 构造PLY元素 (与原始完全相同)
        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
        elements = np.empty(xyz_data.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        
        # 保存文件
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)
    
    # 保存完整模型参考
    reference_path = os.path.join(output_dir, f"reference_full_model_{len(xyz)}balls.ply")
    save_gaussians_like_original(xyz, normals, features_dc, features_rest, 
                                opacity, scaling, rotation, reference_path)
    print(f"✅ 参考文件: {os.path.basename(reference_path)}")
    
    # 创建渐进式累积文件
    print(f"\n🔄 创建修复版渐进式累积文件...")
    progressive_files = []
    
    for end_layer in range(len(layer_suggestions)):
        print(f"\n🎯 创建累积文件: 层0到层{end_layer}...")
        
        # 合并mask
        combined_mask = np.zeros(len(avg_scale), dtype=bool)
        
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
        
        print(f"  累积高斯球数: {np.sum(combined_mask):,}")
        
        if np.sum(combined_mask) == 0:
            print(f"  ⚠️ 累积层为空，跳过")
            continue
        
        # 提取累积参数 (保持原始格式)
        prog_xyz = xyz[combined_mask]
        prog_normals = normals[combined_mask]
        prog_features_dc = features_dc[combined_mask]
        prog_features_rest = features_rest[combined_mask]
        prog_scaling = scaling[combined_mask]
        prog_rotation = rotation[combined_mask]
        prog_opacity = opacity[combined_mask]
        
        # 保存渐进式文件
        if end_layer == 0:
            filename = f"size_progressive_fixed_S0_{np.sum(combined_mask)}balls.ply"
        else:
            # 正确的累积命名：S0_S1_S2...S{end_layer}
            layer_names = '_'.join([f"S{i}" for i in range(end_layer + 1)])
            filename = f"size_progressive_fixed_{layer_names}_{np.sum(combined_mask)}balls.ply"
        
        prog_file_path = os.path.join(output_dir, filename)
        
        save_gaussians_like_original(prog_xyz, prog_normals, prog_features_dc, prog_features_rest,
                                    prog_opacity, prog_scaling, prog_rotation, prog_file_path)
        
        progressive_files.append(prog_file_path)
        print(f"  ✅ 保存: {filename}")
    
    print(f"\n📈 修复版渐进式文件创建完成: {len(progressive_files)} 个")
    
    # 保存文件清单
    file_manifest = {
        'progressive_layers': [os.path.basename(f) for f in progressive_files],
        'reference_file': os.path.basename(reference_path),
        'layer_info': layer_suggestions,
        'thresholds': thresholds,
        'total_gaussians': len(avg_scale),
        'fix_notes': [
            '使用与原始GaussianModel.save_ply完全相同的数据格式',
            '特征数据使用transpose(1,2).flatten(start_dim=1)处理',
            '属性concatenate顺序: xyz, normals, f_dc, f_rest, opacity, scale, rotation'
        ]
    }
    
    manifest_path = os.path.join(output_dir, 'fixed_layers_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(file_manifest, f, indent=2)
    
    print(f"✅ 文件清单保存: {manifest_path}")
    print(f"\n🎉 修复版尺寸分层文件创建完成!")
    print(f"📁 输出目录: {output_dir}/")
    print(f"📈 渐进文件: {len(progressive_files)} 个")
    print(f"📋 参考文件: 1 个")

def main():
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
    
    # 执行修复版分层
    create_size_based_layers_fixed(ply_path, analysis_results_path)

if __name__ == "__main__":
    main() 