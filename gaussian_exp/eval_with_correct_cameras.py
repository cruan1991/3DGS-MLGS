#!/usr/bin/env python3
# 使用真实COLMAP相机参数的评估脚本
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import os
import argparse
from scene import GaussianModel
from arguments import PipelineParams
from gaussian_renderer import render
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from utils.graphics_utils import focal2fov
from utils.camera_utils import Camera
from PIL import Image
from utils.general_utils import PILtoTorch
import numpy as np

# Set CUDA device
torch.cuda.set_device(1)

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_cameras_from_colmap(colmap_path, images_path, resolution_scale=1.0):
    """从COLMAP数据直接加载相机，使用真实的相机参数"""
    cameras_bin = os.path.join(colmap_path, 'cameras.bin')
    images_bin = os.path.join(colmap_path, 'images.bin')
    
    cam_intrinsics = read_intrinsics_binary(cameras_bin)
    cam_extrinsics = read_extrinsics_binary(images_bin)
    
    cameras = []
    
    print(f"📷 从COLMAP加载 {len(cam_extrinsics)} 个相机")
    
    for idx, (img_id, img_info) in enumerate(cam_extrinsics.items()):
        # 获取内参
        cam_id = img_info.camera_id
        intrinsic = cam_intrinsics[cam_id]
        
        # 解析内参
        if intrinsic.model == 'PINHOLE':
            fx, fy, cx, cy = intrinsic.params
        else:
            raise ValueError(f"不支持的相机模型: {intrinsic.model}")
        
        # 应用分辨率缩放
        width = int(intrinsic.width / resolution_scale)
        height = int(intrinsic.height / resolution_scale)
        fx = fx / resolution_scale
        fy = fy / resolution_scale
        cx = cx / resolution_scale
        cy = cy / resolution_scale
        
        # 计算正确的FoV
        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)
        
        # 外参（参照dataset_readers.py的方式）
        R = np.transpose(qvec2rotmat(img_info.qvec))
        T = np.array(img_info.tvec)
        
        # 加载图像
        image_path = os.path.join(images_path, img_info.name)
        image = Image.open(image_path)
        
        # 调整图像尺寸
        if resolution_scale != 1.0:
            new_size = (width, height)
            image = image.resize(new_size, Image.LANCZOS)
        
        # 转换为tensor
        im_data = PILtoTorch(image, (width, height))
        
        # 创建相机（参照camera_utils.py中loadCam的方式）
        camera = Camera(
            resolution=(width, height),
            colmap_id=img_id,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            depth_params=None,
            image=image,
            invdepthmap=None,
            image_name=img_info.name,
            uid=idx,
            data_device="cuda",
            train_test_exp=False,
            is_test_dataset=False,
            is_test_view=False
        )
        
        cameras.append(camera)
        
        if idx < 3:  # 显示前几个相机的参数
            print(f"  相机 {idx} ({img_info.name}): {width}x{height}, FoVx={np.degrees(FoVx):.1f}°, FoVy={np.degrees(FoVy):.1f}°")
    
    return cameras

def eval_with_correct_cameras(model_path, ply_path):
    print("🚀 使用真实COLMAP相机参数进行评估")
    print(f"📁 模型路径: {model_path}")
    print(f"🎯 PLY文件: {ply_path}")
    
    # 检查SPARSE_ADAM_AVAILABLE
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except:
        SPARSE_ADAM_AVAILABLE = False
    
    print(f"🔧 SPARSE_ADAM_AVAILABLE: {SPARSE_ADAM_AVAILABLE}")
    
    # 加载高斯模型
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    print(f"✅ 加载了 {gaussians.get_xyz.shape[0]} 个高斯球")
    
    # 设置Pipeline参数
    parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(parser)
    args = parser.parse_args([])
    pipe = pipe_parser.extract(args)
    
    # 背景设置（与训练一致）
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 从COLMAP加载相机（使用适当的分辨率缩放）
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    
    # 回到最佳分辨率设置
    resolution_scale = 2.0  # 这个设置给出了最好的结果
    cameras = load_cameras_from_colmap(colmap_path, images_path, resolution_scale)
    
    print(f"✅ 加载了 {len(cameras)} 个相机")
    
    # 渲染参数
    renderArgs = (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, False)  # train_test_exp=False
    
    # 🔍 只按照train.py的方式选择5个相机进行快速评估
    train_camera_indices = [idx % len(cameras) for idx in range(5, 30, 5)]
    selected_cameras = [cameras[idx] for idx in train_camera_indices]
    
    print(f"🎯 快速评估：使用索引 {train_camera_indices} 的5个训练相机...")
    total_psnr = 0.0
    camera_count = len(selected_cameras)
    
    for i, camera in enumerate(selected_cameras):
        camera_idx = train_camera_indices[i]
        # 渲染
        rendered = torch.clamp(render(camera, gaussians, *renderArgs)["render"], 0.0, 1.0)
        gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
        
        # 计算PSNR
        psnr_val = psnr(rendered, gt_image).mean().item()
        total_psnr += psnr_val
    
    avg_psnr = total_psnr / camera_count
    print(f"🎉 训练风格评估PSNR: {avg_psnr:.2f} dB (评估了 {camera_count} 个训练相机)")
    print(f"📈 达成率: {(avg_psnr/33.83)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--ply-path', required=True)
    args = parser.parse_args()
    
    eval_with_correct_cameras(args.model_path, args.ply_path)

if __name__ == "__main__":
    main() 