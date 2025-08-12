#!/usr/bin/env python3
# 简化的评估脚本，直接从cameras.json加载相机参数，避免Scene的复杂性
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import os
import argparse
import json
from scene import GaussianModel
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.camera_utils import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov  # 🔥 添加focal2fov导入
from PIL import Image
import numpy as np

# Set CUDA device
torch.cuda.set_device(1)

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_cameras_from_json(cameras_json_path, images_path):
    """直接从cameras.json加载相机参数"""
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)
    
    cameras = []
    for cam_data in cameras_data:
        # 构造图片路径
        image_path = os.path.join(images_path, cam_data['img_name'])
        
        # 加载图片
        image = Image.open(image_path)
        
        # 🔥 关键修复：使用focal2fov将焦距转换为FoV（弧度）
        width = cam_data['width']
        height = cam_data['height']
        fx = cam_data['fx']
        fy = cam_data['fy']
        
        FoVx = focal2fov(fx, width)   # 转换为弧度
        FoVy = focal2fov(fy, height)  # 转换为弧度
        
        print(f"📷 {cam_data['img_name']}: fx={fx:.1f} -> FoVx={FoVx:.4f}rad, fy={fy:.1f} -> FoVy={FoVy:.4f}rad")
        
        # 创建Camera对象 - 使用正确的参数
        camera = Camera(
            resolution=(width, height),
            colmap_id=cam_data['id'],
            R=np.array(cam_data['rotation']),
            T=np.array(cam_data['position']),
            FoVx=FoVx,  # 现在是正确的弧度值
            FoVy=FoVy,  # 现在是正确的弧度值
            depth_params=None,
            image=image,
            invdepthmap=None,
            image_name=cam_data['img_name'],
            uid=cam_data['id'],
            data_device="cuda"
        )
        cameras.append(camera)
    
    return cameras

def simple_evaluation(model_path, ply_path):
    print("🚀 Simple evaluation")
    print(f"📁 Model path: {model_path}")
    print(f"🎯 PLY file: {ply_path}")
    
    # 🔥 检查SPARSE_ADAM_AVAILABLE，与train.py保持一致
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except:
        SPARSE_ADAM_AVAILABLE = False
    
    print(f"🔧 SPARSE_ADAM_AVAILABLE: {SPARSE_ADAM_AVAILABLE}")
    
    # 加载高斯球
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)
    print(f"✅ Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    # 🔥 修复：使用与train.py完全一致的Pipeline参数
    parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(parser)
    args = parser.parse_args([])
    pipe = pipe_parser.extract(args)
    
    # 背景
    bg_color = [0, 0, 0]  # 黑背景
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 加载相机
    cameras_json_path = os.path.join(model_path, "cameras.json")
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    cameras = load_cameras_from_json(cameras_json_path, images_path)
    print(f"✅ Loaded {len(cameras)} cameras")
    
    # 评估前5个相机
    total_psnr = 0.0
    for i, camera in enumerate(cameras[:5]):
        # 🔥 关键修复：使用与train.py完全相同的渲染参数
        # renderFunc(viewpoint, scene.gaussians, *renderArgs)
        # renderArgs = (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
        
        # 假设dataset.train_test_exp = False（需要从实际数据集加载）
        train_test_exp = False
        
        image = torch.clamp(render(camera, gaussians, pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, train_test_exp)["render"], 0.0, 1.0)
        gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
        
        # 如果train_test_exp为True，需要裁剪图像（但我们先假设为False）
        if train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
        
        # 计算PSNR
        psnr_val = psnr(image, gt_image).mean().item()
        total_psnr += psnr_val
        
        print(f"📷 Camera {i} ({camera.image_name}): PSNR = {psnr_val:.2f} dB")
        
        # 保存图像用于检查
        if i < 3:
            from torchvision.utils import save_image
            os.makedirs("simple_renders", exist_ok=True)
            save_image(image, f"simple_renders/{camera.image_name}_render.png")
            save_image(gt_image, f"simple_renders/{camera.image_name}_gt.png")
    
    avg_psnr = total_psnr / 5
    print(f"\n🎉 Average PSNR: {avg_psnr:.2f} dB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--ply-path', required=True)
    args = parser.parse_args()
    
    simple_evaluation(args.model_path, args.ply_path)

if __name__ == "__main__":
    main() 
# 简化的评估脚本，直接从cameras.json加载相机参数，避免Scene的复杂性
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import os
import argparse
import json
from scene import GaussianModel
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.camera_utils import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov  # 🔥 添加focal2fov导入
from PIL import Image
import numpy as np

# Set CUDA device
torch.cuda.set_device(1)

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_cameras_from_json(cameras_json_path, images_path):
    """直接从cameras.json加载相机参数"""
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)
    
    cameras = []
    for cam_data in cameras_data:
        # 构造图片路径
        image_path = os.path.join(images_path, cam_data['img_name'])
        
        # 加载图片
        image = Image.open(image_path)
        
        # 🔥 关键修复：使用focal2fov将焦距转换为FoV（弧度）
        width = cam_data['width']
        height = cam_data['height']
        fx = cam_data['fx']
        fy = cam_data['fy']
        
        FoVx = focal2fov(fx, width)   # 转换为弧度
        FoVy = focal2fov(fy, height)  # 转换为弧度
        
        print(f"📷 {cam_data['img_name']}: fx={fx:.1f} -> FoVx={FoVx:.4f}rad, fy={fy:.1f} -> FoVy={FoVy:.4f}rad")
        
        # 创建Camera对象 - 使用正确的参数
        camera = Camera(
            resolution=(width, height),
            colmap_id=cam_data['id'],
            R=np.array(cam_data['rotation']),
            T=np.array(cam_data['position']),
            FoVx=FoVx,  # 现在是正确的弧度值
            FoVy=FoVy,  # 现在是正确的弧度值
            depth_params=None,
            image=image,
            invdepthmap=None,
            image_name=cam_data['img_name'],
            uid=cam_data['id'],
            data_device="cuda"
        )
        cameras.append(camera)
    
    return cameras

def simple_evaluation(model_path, ply_path):
    print("🚀 Simple evaluation")
    print(f"📁 Model path: {model_path}")
    print(f"🎯 PLY file: {ply_path}")
    
    # 🔥 检查SPARSE_ADAM_AVAILABLE，与train.py保持一致
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except:
        SPARSE_ADAM_AVAILABLE = False
    
    print(f"🔧 SPARSE_ADAM_AVAILABLE: {SPARSE_ADAM_AVAILABLE}")
    
    # 加载高斯球
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path)
    print(f"✅ Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    # 🔥 修复：使用与train.py完全一致的Pipeline参数
    parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(parser)
    args = parser.parse_args([])
    pipe = pipe_parser.extract(args)
    
    # 背景
    bg_color = [0, 0, 0]  # 黑背景
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 加载相机
    cameras_json_path = os.path.join(model_path, "cameras.json")
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    cameras = load_cameras_from_json(cameras_json_path, images_path)
    print(f"✅ Loaded {len(cameras)} cameras")
    
    # 评估前5个相机
    total_psnr = 0.0
    for i, camera in enumerate(cameras[:5]):
        # 🔥 关键修复：使用与train.py完全相同的渲染参数
        # renderFunc(viewpoint, scene.gaussians, *renderArgs)
        # renderArgs = (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
        
        # 假设dataset.train_test_exp = False（需要从实际数据集加载）
        train_test_exp = False
        
        image = torch.clamp(render(camera, gaussians, pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, train_test_exp)["render"], 0.0, 1.0)
        gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
        
        # 如果train_test_exp为True，需要裁剪图像（但我们先假设为False）
        if train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
        
        # 计算PSNR
        psnr_val = psnr(image, gt_image).mean().item()
        total_psnr += psnr_val
        
        print(f"📷 Camera {i} ({camera.image_name}): PSNR = {psnr_val:.2f} dB")
        
        # 保存图像用于检查
        if i < 3:
            from torchvision.utils import save_image
            os.makedirs("simple_renders", exist_ok=True)
            save_image(image, f"simple_renders/{camera.image_name}_render.png")
            save_image(gt_image, f"simple_renders/{camera.image_name}_gt.png")
    
    avg_psnr = total_psnr / 5
    print(f"\n🎉 Average PSNR: {avg_psnr:.2f} dB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--ply-path', required=True)
    args = parser.parse_args()
    
    simple_evaluation(args.model_path, args.ply_path)

if __name__ == "__main__":
    main() 