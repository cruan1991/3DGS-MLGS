#!/usr/bin/env python3
# 最小化测试曝光参数加载
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
from scene import GaussianModel

# Set CUDA device
torch.cuda.set_device(1)

def test_exposure_loading():
    print("🔍 Testing exposure parameter loading")
    
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230/point_cloud.ply"
    
    # 创建高斯模型
    gaussians = GaussianModel(3)
    
    print(f"📂 Loading PLY: {ply_path}")
    print(f"🎨 use_train_test_exp: True")
    
    # 直接测试load_ply
    gaussians.load_ply(ply_path, use_train_test_exp=True)
    
    print(f"✅ PLY loaded successfully")
    print(f"📊 Number of gaussians: {gaussians.get_xyz.shape[0]}")
    
    # 检查曝光参数
    print(f"\n🔍 Checking exposure parameters:")
    print(f"  - hasattr(gaussians, 'pretrained_exposures'): {hasattr(gaussians, 'pretrained_exposures')}")
    if hasattr(gaussians, 'pretrained_exposures'):
        print(f"  - gaussians.pretrained_exposures is None: {gaussians.pretrained_exposures is None}")
        if gaussians.pretrained_exposures is not None:
            print(f"  - Number of pretrained exposures: {len(gaussians.pretrained_exposures)}")
            # 显示前几个键
            keys = list(gaussians.pretrained_exposures.keys())[:5]
            print(f"  - Sample keys: {keys}")
    
    print(f"  - hasattr(gaussians, '_exposure'): {hasattr(gaussians, '_exposure')}")
    print(f"  - hasattr(gaussians, 'exposure_mapping'): {hasattr(gaussians, 'exposure_mapping')}")

if __name__ == "__main__":
    test_exposure_loading() 
# 最小化测试曝光参数加载
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
from scene import GaussianModel

# Set CUDA device
torch.cuda.set_device(1)

def test_exposure_loading():
    print("🔍 Testing exposure parameter loading")
    
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230/point_cloud.ply"
    
    # 创建高斯模型
    gaussians = GaussianModel(3)
    
    print(f"📂 Loading PLY: {ply_path}")
    print(f"🎨 use_train_test_exp: True")
    
    # 直接测试load_ply
    gaussians.load_ply(ply_path, use_train_test_exp=True)
    
    print(f"✅ PLY loaded successfully")
    print(f"📊 Number of gaussians: {gaussians.get_xyz.shape[0]}")
    
    # 检查曝光参数
    print(f"\n🔍 Checking exposure parameters:")
    print(f"  - hasattr(gaussians, 'pretrained_exposures'): {hasattr(gaussians, 'pretrained_exposures')}")
    if hasattr(gaussians, 'pretrained_exposures'):
        print(f"  - gaussians.pretrained_exposures is None: {gaussians.pretrained_exposures is None}")
        if gaussians.pretrained_exposures is not None:
            print(f"  - Number of pretrained exposures: {len(gaussians.pretrained_exposures)}")
            # 显示前几个键
            keys = list(gaussians.pretrained_exposures.keys())[:5]
            print(f"  - Sample keys: {keys}")
    
    print(f"  - hasattr(gaussians, '_exposure'): {hasattr(gaussians, '_exposure')}")
    print(f"  - hasattr(gaussians, 'exposure_mapping'): {hasattr(gaussians, 'exposure_mapping')}")

if __name__ == "__main__":
    test_exposure_loading() 