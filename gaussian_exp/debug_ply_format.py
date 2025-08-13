import os
import sys
import numpy as np
from plyfile import PlyData

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel

def analyze_ply_format(ply_path, name):
    """分析PLY文件的格式和数据"""
    print(f"\n🔍 分析 {name}")
    print(f"文件: {ply_path}")
    print("-" * 50)
    
    if not os.path.exists(ply_path):
        print("❌ 文件不存在")
        return
    
    # 1. 用plyfile直接读取
    print("📋 PLY原始格式:")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    print(f"  顶点数: {len(vertex)}")
    print(f"  属性列表: {list(vertex.dtype.names)}")
    print(f"  前几个属性的数据类型:")
    for i, name in enumerate(list(vertex.dtype.names)[:10]):
        print(f"    {name}: {vertex.dtype.descr[i]}")
    
    # 2. 检查数据范围
    print(f"\n📊 数据范围检查:")
    xyz = np.array([vertex['x'], vertex['y'], vertex['z']]).T
    print(f"  XYZ 范围: x[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}], y[{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}], z[{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
    
    # 检查透明度
    if 'opacity' in vertex.dtype.names:
        opacity = vertex['opacity']
        print(f"  透明度范围: [{opacity.min():.6f}, {opacity.max():.6f}]")
        print(f"  透明度均值: {opacity.mean():.6f}")
    
    # 检查缩放
    scale_attrs = [name for name in vertex.dtype.names if name.startswith('scale_')]
    if scale_attrs:
        scales = np.array([vertex[attr] for attr in scale_attrs]).T
        print(f"  缩放范围: [{scales.min():.6f}, {scales.max():.6f}]")
        print(f"  缩放均值: {scales.mean():.6f}")
    
    # 检查旋转
    rot_attrs = [name for name in vertex.dtype.names if name.startswith('rot_')]
    if rot_attrs:
        rotations = np.array([vertex[attr] for attr in rot_attrs]).T
        print(f"  旋转范围: [{rotations.min():.6f}, {rotations.max():.6f}]")
        print(f"  旋转均值: {rotations.mean():.6f}")
    
    # 3. 用GaussianModel加载验证
    print(f"\n🧪 GaussianModel加载测试:")
    try:
        gaussians = GaussianModel(3)
        gaussians.load_ply(ply_path, use_train_test_exp=False)
        
        print(f"  ✅ 加载成功")
        print(f"  XYZ shape: {gaussians.get_xyz.shape}")
        print(f"  透明度 shape: {gaussians.get_opacity.shape}")
        print(f"  缩放 shape: {gaussians.get_scaling.shape}")
        print(f"  旋转 shape: {gaussians.get_rotation.shape}")
        print(f"  特征DC shape: {gaussians.get_features.shape}")
        
        # 检查数据有效性
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        opacity = gaussians.get_opacity.detach().cpu().numpy()
        scaling = gaussians.get_scaling.detach().cpu().numpy()
        
        print(f"  XYZ NaN数: {np.isnan(xyz).sum()}")
        print(f"  透明度 NaN数: {np.isnan(opacity).sum()}")
        print(f"  缩放 NaN数: {np.isnan(scaling).sum()}")
        
        print(f"  XYZ 实际范围: x[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
        print(f"  透明度 实际范围: [{opacity.min():.6f}, {opacity.max():.6f}]")
        print(f"  缩放 实际范围: [{scaling.min():.6f}, {scaling.max():.6f}]")
        
    except Exception as e:
        print(f"  ❌ 加载失败: {str(e)}")

def main():
    print("🔍 PLY格式对比分析")
    print("=" * 60)
    
    # 分析原始模型
    original_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    analyze_ply_format(original_path, "原始训练模型")
    
    # 分析重新生成的模型
    regenerated_path = "./size_based_layers/size_progressive_S0_S1_S2_S3_S4_2046811balls.ply"
    analyze_ply_format(regenerated_path, "重新生成的完整模型")
    
    print(f"\n🔍 对比总结:")
    print("=" * 30)
    print("如果发现数据范围、格式或NaN数量不一致，")
    print("说明PLY重新生成过程中存在数据转换错误")

if __name__ == "__main__":
    main() 