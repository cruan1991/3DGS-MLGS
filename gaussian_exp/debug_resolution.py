#!/usr/bin/env python3
# 调试分辨率处理
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import os
import argparse
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams

# Set CUDA device
torch.cuda.set_device(1)

def debug_resolution():
    print("🔍 Debugging resolution handling")
    
    model_path = "./output/truck-150w"
    
    # 创建符号链接
    original_images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    model_images_path = os.path.join(model_path, "images")
    
    if not os.path.exists(model_images_path):
        os.symlink(original_images_path, model_images_path)
    
    # 设置参数
    parser = argparse.ArgumentParser()
    dataset_parser = ModelParams(parser)
    args = parser.parse_args([])
    
    args.source_path = model_path
    args.model_path = model_path
    args.images = "images"
    args.resolution = -1  # 与训练一致
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.train_test_exp = False  # 与训练一致
    
    dataset = dataset_parser.extract(args)
    
    print(f"📊 Dataset参数:")
    print(f"  - resolution: {args.resolution}")
    print(f"  - train_test_exp: {args.train_test_exp}")
    
    # 创建高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    
    # 🔍 调试：使用不同的resolution_scales来创建Scene
    print(f"\n🔍 测试不同的resolution_scales:")
    
    for scale in [1.0, 2.0, 0.5]:
        print(f"\n  📏 Testing resolution_scale = {scale}")
        try:
            scene = Scene(dataset, gaussians, load_iteration=994230, resolution_scales=[scale])
            
            train_cameras = scene.getTrainCameras()
            if len(train_cameras) > 0:
                first_cam = train_cameras[0]
                print(f"    - 第一个相机分辨率: {first_cam.image_width}x{first_cam.image_height}")
                print(f"    - 相机总数: {len(train_cameras)}")
            
        except Exception as e:
            print(f"    - Error: {e}")

if __name__ == "__main__":
    debug_resolution() 
# 调试分辨率处理
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import os
import argparse
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams

# Set CUDA device
torch.cuda.set_device(1)

def debug_resolution():
    print("🔍 Debugging resolution handling")
    
    model_path = "./output/truck-150w"
    
    # 创建符号链接
    original_images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    model_images_path = os.path.join(model_path, "images")
    
    if not os.path.exists(model_images_path):
        os.symlink(original_images_path, model_images_path)
    
    # 设置参数
    parser = argparse.ArgumentParser()
    dataset_parser = ModelParams(parser)
    args = parser.parse_args([])
    
    args.source_path = model_path
    args.model_path = model_path
    args.images = "images"
    args.resolution = -1  # 与训练一致
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.train_test_exp = False  # 与训练一致
    
    dataset = dataset_parser.extract(args)
    
    print(f"📊 Dataset参数:")
    print(f"  - resolution: {args.resolution}")
    print(f"  - train_test_exp: {args.train_test_exp}")
    
    # 创建高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    
    # 🔍 调试：使用不同的resolution_scales来创建Scene
    print(f"\n🔍 测试不同的resolution_scales:")
    
    for scale in [1.0, 2.0, 0.5]:
        print(f"\n  📏 Testing resolution_scale = {scale}")
        try:
            scene = Scene(dataset, gaussians, load_iteration=994230, resolution_scales=[scale])
            
            train_cameras = scene.getTrainCameras()
            if len(train_cameras) > 0:
                first_cam = train_cameras[0]
                print(f"    - 第一个相机分辨率: {first_cam.image_width}x{first_cam.image_height}")
                print(f"    - 相机总数: {len(train_cameras)}")
            
        except Exception as e:
            print(f"    - Error: {e}")

if __name__ == "__main__":
    debug_resolution() 