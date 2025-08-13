import os
import sys
import torch
import numpy as np

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel

def debug_nan_positions():
    """检查高斯球位置中的NaN值"""
    print("🔍 检查高斯球位置NaN值问题")
    print("=" * 40)
    
    # 加载原始模型
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    # 获取位置数据
    xyz = gaussians._xyz.detach().cpu().numpy()
    print(f"📊 高斯球总数: {len(xyz):,}")
    
    # 检查NaN值
    nan_mask = np.isnan(xyz)
    nan_count = np.sum(nan_mask)
    nan_positions = np.any(nan_mask, axis=1)
    nan_position_count = np.sum(nan_positions)
    
    print(f"🔍 NaN值检查:")
    print(f"  总NaN值数量: {nan_count}")
    print(f"  含NaN的位置数: {nan_position_count}")
    
    if nan_position_count > 0:
        print(f"  NaN位置比例: {nan_position_count/len(xyz)*100:.4f}%")
        
        # 显示有NaN的位置
        nan_indices = np.where(nan_positions)[0][:10]  # 显示前10个
        print(f"  前10个NaN位置索引: {nan_indices}")
        
        for i, idx in enumerate(nan_indices):
            print(f"    位置{idx}: [{xyz[idx, 0]}, {xyz[idx, 1]}, {xyz[idx, 2]}]")
        
        # 计算去除NaN后的场景中心
        valid_mask = ~nan_positions
        valid_xyz = xyz[valid_mask]
        scene_center = np.mean(valid_xyz, axis=0)
        
        print(f"📍 有效位置统计:")
        print(f"  有效位置数: {len(valid_xyz):,}")
        print(f"  场景中心: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
        print(f"  X范围: [{valid_xyz[:, 0].min():.3f}, {valid_xyz[:, 0].max():.3f}]")
        print(f"  Y范围: [{valid_xyz[:, 1].min():.3f}, {valid_xyz[:, 1].max():.3f}]")
        print(f"  Z范围: [{valid_xyz[:, 2].min():.3f}, {valid_xyz[:, 2].max():.3f}]")
        
        return valid_mask, scene_center
    else:
        scene_center = np.mean(xyz, axis=0)
        print(f"✅ 没有NaN值")
        print(f"📍 场景中心: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
        return None, scene_center

if __name__ == "__main__":
    debug_nan_positions() 