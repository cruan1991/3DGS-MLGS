#!/usr/bin/env python3
"""
简化的高斯球模型评估脚本 - 支持自定义路径
基于之前成功的evaluate_filtered.py，去除Scene类依赖
"""

import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import argparse
import os
import torch
import csv
import json
from pathlib import Path
from tqdm import tqdm
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import Camera
from utils.image_utils import cal_psnr
from utils.general_utils import PILtoTorch, getWorld2View2, focal2fov
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

class DummyRenderPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.antialiasing = False
        self.debug = False

def load_camera_data(config_path, images_dir):
    """从JSON配置文件加载相机数据并创建Camera对象"""
    
    with open(config_path, 'r') as f:
        camera_data = json.load(f)
    
    cameras = []
    
    for idx, cam in enumerate(camera_data):
        try:
            # 加载图片
            img_path = os.path.join(images_dir, cam['img_name'])
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            pil_image = Image.open(img_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为tensor
            image = PILtoTorch(pil_image, None)
            
            # 相机参数
            width = cam['width']
            height = cam['height']
            
            # 计算FOV (假设焦距为width*0.8)
            focal_length = width * 0.8
            FoVx = focal2fov(focal_length, width)
            FoVy = focal2fov(focal_length, height)
            
            # 旋转和平移
            R = np.array(cam['rotation'])
            T = np.array(cam['position'])
            
            # 创建world2cam矩阵
            world_view_transform = getWorld2View2(R, T).transpose()
            projection_matrix = torch.eye(4)  # 简化的投影矩阵
            
            # 创建Camera对象
            camera = Camera(
                colmap_id=idx,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=image,
                gt_alpha_mask=None,
                image_name=cam['img_name'],
                uid=idx,
                data_device='cuda'
            )
            
            cameras.append(camera)
            
        except Exception as e:
            print(f"Error loading camera {idx}: {e}")
            continue
    
    print(f"Successfully loaded {len(cameras)} cameras")
    return cameras

def tensor_to_image(tensor):
    """将tensor转换为PIL图像"""
    if tensor.is_cuda:
        tensor = tensor.detach().cpu()
    else:
        tensor = tensor.detach()
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

def evaluate_gaussian_model(ply_path, config_path, gt_images_dir, output_dir, device='cuda'):
    """
    评估高斯球模型
    
    Args:
        ply_path: 高斯球PLY文件路径
        config_path: 相机配置JSON文件路径
        gt_images_dir: 真实图片目录路径
        output_dir: 输出目录路径
        device: 计算设备
    """
    
    # 验证输入文件
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"高斯球文件不存在: {ply_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"相机配置文件不存在: {config_path}")
    if not os.path.exists(gt_images_dir):
        raise FileNotFoundError(f"GT图片目录不存在: {gt_images_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 加载高斯球模型: {ply_path}")
    try:
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(ply_path)
    except Exception as e:
        raise RuntimeError(f"加载PLY文件失败: {e}")

    print(f"📸 加载相机数据: {config_path}")
    print(f"📂 GT图片目录: {gt_images_dir}")
    
    try:
        cameras = load_camera_data(config_path, gt_images_dir)
        if not cameras:
            raise ValueError("未找到有效的相机数据")
    except Exception as e:
        raise RuntimeError(f"加载相机数据失败: {e}")

    print(f"🎨 开始渲染 {len(cameras)} 个视角...")
    all_metrics = []
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    pipe = DummyRenderPipe()

    # 创建输出子目录
    renders_dir = os.path.join(output_dir, "rendered_images")
    os.makedirs(renders_dir, exist_ok=True)

    for idx, camera in enumerate(tqdm(cameras)):
        try:
            # 渲染
            render_pkg = render(camera, gaussians, pipe=pipe, bg_color=bg_color)
            image = torch.clamp(render_pkg['render'], 0.0, 1.0)
            gt = torch.clamp(camera.original_image.to(image.device), 0.0, 1.0)

            # 计算PSNR
            image_batch = image.unsqueeze(0)
            gt_batch = gt.unsqueeze(0)
            psnr_val = cal_psnr(image_batch, gt_batch).mean().item()
            all_metrics.append(psnr_val)

            # 保存渲染图片
            out_path = os.path.join(renders_dir, camera.image_name.replace('.jpg', '.png'))
            render_image = tensor_to_image(render_pkg['render'])
            render_image.save(out_path)
            
            print(f"视角 {idx:3d} ({camera.image_name}): PSNR = {psnr_val:.2f} dB")
            
        except Exception as e:
            print(f"⚠️  视角 {idx} 渲染失败: {e}")
            continue

    if not all_metrics:
        print("❌ 没有成功处理的视角！")
        return

    avg_psnr = sum(all_metrics) / len(all_metrics)
    print(f"\n✅ 评估完成!")
    print(f"📊 平均 PSNR: {avg_psnr:.2f} dB")
    print(f"📊 成功处理: {len(all_metrics)}/{len(cameras)} 个视角")

    # 保存指标
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["view_idx", "image_name", "psnr"])
        for i, (psnr, cam) in enumerate(zip(all_metrics, cameras)):
            writer.writerow([i, cam.image_name, f"{psnr:.4f}"])
        writer.writerow(["average", "", f"{avg_psnr:.4f}"])

    print(f"📄 指标保存到: {csv_path}")
    print(f"🖼️  渲染图片保存到: {renders_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='简化的高斯球模型评估脚本 - 支持自定义路径',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python evaluate_gaussians_simple.py \\
    --ply-path output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply \\
    --config-path output/truck-150w/cameras.json \\
    --gt-dir data/mipnerf360/360/tandt_db/tandt/truck/images \\
    --output-dir evaluation_results/gaussian_ball
        """
    )
    
    parser.add_argument('--ply-path', required=True, 
                        help='高斯球PLY文件路径')
    parser.add_argument('--config-path', required=True, 
                        help='相机配置JSON文件路径')
    parser.add_argument('--gt-dir', required=True, 
                        help='真实图片目录路径 (ground truth images)')
    parser.add_argument('--output-dir', default='evaluation_output', 
                        help='输出目录路径 (默认: evaluation_output)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='计算设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    print("🔍 高斯球模型评估 (简化版)")
    print("=" * 50)
    print(f"高斯球文件: {args.ply_path}")
    print(f"相机配置:   {args.config_path}")
    print(f"GT图片目录: {args.gt_dir}")
    print(f"输出目录:   {args.output_dir}")
    print("=" * 50)
    
    try:
        evaluate_gaussian_model(
            ply_path=args.ply_path,
            config_path=args.config_path,
            gt_images_dir=args.gt_dir,
            output_dir=args.output_dir,
            device=args.device
        )
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 