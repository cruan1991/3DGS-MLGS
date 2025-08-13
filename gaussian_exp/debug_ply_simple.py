import os
import sys
import torch
import numpy as np

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel

def compare_gaussian_models():
    """对比原始模型和重新生成的模型"""
    print("🔍 高斯模型数据对比")
    print("=" * 60)
    
    # 文件路径
    original_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    regenerated_path = "./size_based_layers/size_progressive_S0_S1_S2_S3_S4_2046811balls.ply"
    
    models = {}
    
    for name, path in [("原始模型", original_path), ("重生模型", regenerated_path)]:
        print(f"\n📊 加载 {name}")
        print(f"文件: {path}")
        
        if not os.path.exists(path):
            print("❌ 文件不存在")
            continue
        
        try:
            gaussians = GaussianModel(3)
            gaussians.load_ply(path, use_train_test_exp=False)
            
            # 提取所有数据
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            opacity = gaussians.get_opacity.detach().cpu().numpy()
            scaling = gaussians.get_scaling.detach().cpu().numpy()
            rotation = gaussians.get_rotation.detach().cpu().numpy()
            features_dc = gaussians.get_features_dc.detach().cpu().numpy()
            features_rest = gaussians.get_features_rest.detach().cpu().numpy()
            
            print(f"✅ 加载成功")
            print(f"  高斯球数: {xyz.shape[0]:,}")
            print(f"  XYZ shape: {xyz.shape}")
            print(f"  透明度 shape: {opacity.shape}")
            print(f"  缩放 shape: {scaling.shape}")
            print(f"  旋转 shape: {rotation.shape}")
            print(f"  特征DC shape: {features_dc.shape}")
            print(f"  特征Rest shape: {features_rest.shape}")
            
            # 数据统计
            print(f"📈 数据统计:")
            print(f"  XYZ 范围: x[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}], y[{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}], z[{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
            print(f"  透明度 范围: [{opacity.min():.6f}, {opacity.max():.6f}], 均值: {opacity.mean():.6f}")
            print(f"  缩放 范围: [{scaling.min():.6f}, {scaling.max():.6f}], 均值: {scaling.mean():.6f}")
            print(f"  旋转 范围: [{rotation.min():.6f}, {rotation.max():.6f}], 均值: {rotation.mean():.6f}")
            
            # 检查异常值
            print(f"🔍 异常值检查:")
            print(f"  XYZ NaN数: {np.isnan(xyz).sum()}")
            print(f"  透明度 NaN数: {np.isnan(opacity).sum()}")
            print(f"  缩放 NaN数: {np.isnan(scaling).sum()}")
            print(f"  旋转 NaN数: {np.isnan(rotation).sum()}")
            print(f"  特征DC NaN数: {np.isnan(features_dc).sum()}")
            print(f"  特征Rest NaN数: {np.isnan(features_rest).sum()}")
            
            # 缩放统计（关键数据）
            avg_scale = np.mean(scaling, axis=1)
            print(f"📏 平均缩放统计:")
            print(f"  最小: {avg_scale.min():.6f}")
            print(f"  最大: {avg_scale.max():.6f}")
            print(f"  均值: {avg_scale.mean():.6f}")
            print(f"  中位数: {np.median(avg_scale):.6f}")
            print(f"  标准差: {avg_scale.std():.6f}")
            
            models[name] = {
                'xyz': xyz,
                'opacity': opacity,
                'scaling': scaling,
                'rotation': rotation,
                'features_dc': features_dc,
                'features_rest': features_rest,
                'avg_scale': avg_scale
            }
            
        except Exception as e:
            print(f"❌ 加载失败: {str(e)}")
    
    # 对比分析
    if len(models) == 2:
        print(f"\n🔍 详细对比分析:")
        print("=" * 40)
        
        orig = models["原始模型"]
        regen = models["重生模型"]
        
        # 检查数据是否完全相同
        xyz_diff = np.abs(orig['xyz'] - regen['xyz']).max()
        opacity_diff = np.abs(orig['opacity'] - regen['opacity']).max()
        scaling_diff = np.abs(orig['scaling'] - regen['scaling']).max()
        rotation_diff = np.abs(orig['rotation'] - regen['rotation']).max()
        features_dc_diff = np.abs(orig['features_dc'] - regen['features_dc']).max()
        features_rest_diff = np.abs(orig['features_rest'] - regen['features_rest']).max()
        
        print(f"📊 最大差异:")
        print(f"  XYZ: {xyz_diff:.10f}")
        print(f"  透明度: {opacity_diff:.10f}")
        print(f"  缩放: {scaling_diff:.10f}")
        print(f"  旋转: {rotation_diff:.10f}")
        print(f"  特征DC: {features_dc_diff:.10f}")
        print(f"  特征Rest: {features_rest_diff:.10f}")
        
        # 判断数据完整性
        max_diff = max(xyz_diff, opacity_diff, scaling_diff, rotation_diff, features_dc_diff, features_rest_diff)
        
        if max_diff < 1e-6:
            print(f"✅ 数据完全一致 (最大差异: {max_diff:.2e})")
        elif max_diff < 1e-3:
            print(f"⚠️ 数据基本一致 (最大差异: {max_diff:.2e})")
        else:
            print(f"❌ 数据存在显著差异 (最大差异: {max_diff:.2e})")
            print(f"   这解释了PSNR差异的原因！")
            
            # 找出最大差异的属性
            diffs = {
                'XYZ': xyz_diff,
                '透明度': opacity_diff,
                '缩放': scaling_diff,
                '旋转': rotation_diff,
                '特征DC': features_dc_diff,
                '特征Rest': features_rest_diff
            }
            max_attr = max(diffs, key=diffs.get)
            print(f"   最大差异属性: {max_attr} ({diffs[max_attr]:.2e})")

def main():
    compare_gaussian_models()

if __name__ == "__main__":
    main() 