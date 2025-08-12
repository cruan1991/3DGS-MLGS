#!/usr/bin/env python3
# 调试相机参数，检查坐标系问题
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import json
import numpy as np
import torch

def debug_camera_params():
    cameras_json_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/cameras.json"
    
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)
    
    print("🔍 调试相机参数")
    print(f"总共有 {len(cameras_data)} 个相机")
    
    # 检查前几个相机的参数
    for i in range(3):
        cam = cameras_data[i]
        print(f"\n📷 相机 {i} ({cam['img_name']}):")
        print(f"  分辨率: {cam['width']} x {cam['height']}")
        print(f"  焦距: fx={cam['fx']}, fy={cam['fy']}")
        print(f"  位置: {cam['position']}")
        
        R = np.array(cam['rotation'])
        print(f"  旋转矩阵形状: {R.shape}")
        print(f"  旋转矩阵:\n{R}")
        
        # 检查旋转矩阵是否是正交矩阵
        should_be_identity = R @ R.T
        print(f"  R @ R.T (应该接近单位矩阵):\n{should_be_identity}")
        print(f"  det(R) = {np.linalg.det(R):.6f} (应该接近 ±1)")
        
        # 检查是否需要转置
        R_transposed = R.T
        print(f"  R.T @ R (检查是否需要转置):\n{R_transposed @ R}")

if __name__ == "__main__":
    debug_camera_params() 
# 调试相机参数，检查坐标系问题
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import json
import numpy as np
import torch

def debug_camera_params():
    cameras_json_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/cameras.json"
    
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)
    
    print("🔍 调试相机参数")
    print(f"总共有 {len(cameras_data)} 个相机")
    
    # 检查前几个相机的参数
    for i in range(3):
        cam = cameras_data[i]
        print(f"\n📷 相机 {i} ({cam['img_name']}):")
        print(f"  分辨率: {cam['width']} x {cam['height']}")
        print(f"  焦距: fx={cam['fx']}, fy={cam['fy']}")
        print(f"  位置: {cam['position']}")
        
        R = np.array(cam['rotation'])
        print(f"  旋转矩阵形状: {R.shape}")
        print(f"  旋转矩阵:\n{R}")
        
        # 检查旋转矩阵是否是正交矩阵
        should_be_identity = R @ R.T
        print(f"  R @ R.T (应该接近单位矩阵):\n{should_be_identity}")
        print(f"  det(R) = {np.linalg.det(R):.6f} (应该接近 ±1)")
        
        # 检查是否需要转置
        R_transposed = R.T
        print(f"  R.T @ R (检查是否需要转置):\n{R_transposed @ R}")

if __name__ == "__main__":
    debug_camera_params() 