#!/usr/bin/env python3
"""
直接使用COLMAP数据的高斯球评估脚本
完全绕过Scene类，避免CUDA错误
"""

import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import argparse
import os
import torch
import csv
from pathlib import Path
from tqdm import tqdm
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import Camera
from utils.graphics_utils import focal2fov
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
from utils.general_utils import PILtoTorch
import numpy as np
from PIL import Image

class DummyRenderPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.antialiasing = False
        self.debug = False

def load_cameras_from_colmap(sparse_dir, images_dir, resolution_scale=1.0):
    """直接从COLMAP数据加载相机"""
    
    cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
    cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
    
    if not os.path.exists(cameras_intrinsic_file):
        raise FileNotFoundError(f"COLMAP相机内参文件不存在: {cameras_intrinsic_file}")
    if not os.path.exists(cameras_extrinsic_file):
        raise FileNotFoundError(f"COLMAP相机外参文件不存在: {cameras_extrinsic_file}")
    
    print(f"📂 加载COLMAP数据: {sparse_dir}")
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    print(f"✅ 成功读取 {len(cam_intrinsics)} 个内参, {len(cam_extrinsics)} 个外参")
    
    cameras = []
    
    for idx, (image_id, image) in enumerate(cam_extrinsics.items()):
        try:
            camera_id = image.camera_id
            intrinsics = cam_intrinsics[camera_id]
            
            # 图片路径
            image_path = os.path.join(images_dir, image.name)
            if not os.path.exists(image_path):
                print(f"⚠️  图片不存在: {image_path}")
                continue
                
            # 加载并处理图片
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
            # 获取尺寸
            width, height = pil_image.size
            
            # 应用分辨率缩放
            if resolution_scale != 1.0:
                new_width = int(width / resolution_scale)
                new_height = int(height / resolution_scale)
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                width, height = new_width, new_height
            
            # 转换为tensor
            import torchvision.transforms.functional as TF
            image_tensor = TF.to_tensor(pil_image)
            
            # 关键：正确计算FoV
            fx, fy = intrinsics.params[0], intrinsics.params[1]
            if resolution_scale != 1.0:
                fx = fx / resolution_scale
                fy = fy / resolution_scale
                
            FoVx = focal2fov(fx, width)
            FoVy = focal2fov(fy, height)
            
            # 相机外参
            R = image.qvec2rotmat()
            T = image.tvec
            
            # 创建Camera对象
            camera = Camera(
                colmap_id=image_id,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=image_tensor,
                gt_alpha_mask=None,
                image_name=image.name,
                uid=idx,
                data_device='cuda'
            )
            
            cameras.append(camera)
            
        except Exception as e:
            print(f"⚠️  加载相机 {image_id} 失败: {e}")
            continue
            
    print(f"✅ 成功加载 {len(cameras)} 个相机")
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

def evaluate_gaussian_direct(ply_path, colmap_sparse_dir, images_dir, output_dir, device='cuda'):
    """
    直接评估高斯球模型，绕过Scene类
    """
    
    # 验证输入文件
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"高斯球文件不存在: {ply_path}")
    if not os.path.exists(colmap_sparse_dir):
        raise FileNotFoundError(f"COLMAP sparse目录不存在: {colmap_sparse_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"GT图片目录不存在: {images_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 加载高斯球模型: {ply_path}")
    try:
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(ply_path)
    except Exception as e:
        raise RuntimeError(f"加载PLY文件失败: {e}")

    print(f"📸 加载COLMAP相机数据: {colmap_sparse_dir}")
    print(f"📂 GT图片目录: {images_dir}")
    
    try:
        cameras = load_cameras_from_colmap(colmap_sparse_dir, images_dir, resolution_scale=1.0)
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

            # 计算PSNR - 使用正确的公式
            mse = torch.mean((image - gt) ** 2)
            if mse > 0:
                psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
            else:
                psnr_val = 100.0  # 完美匹配
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
        description='直接使用COLMAP数据评估高斯球模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_direct_colmap.py \\
    --ply-path output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply \\
    --colmap-dir data/truck/sparse/0 \\
    --gt-dir data/truck/images \\
    --output-dir evaluation_results
        """
    )
    
    parser.add_argument('--ply-path', required=True, 
                        help='高斯球PLY文件路径')
    parser.add_argument('--colmap-dir', required=True, 
                        help='COLMAP sparse目录路径 (包含cameras.bin和images.bin)')
    parser.add_argument('--gt-dir', required=True, 
                        help='真实图片目录路径 (ground truth images)')
    parser.add_argument('--output-dir', default='evaluation_output', 
                        help='输出目录路径 (默认: evaluation_output)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='计算设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    print("🔍 高斯球模型评估 - 直接COLMAP版本")
    print("=" * 50)
    print(f"高斯球文件: {args.ply_path}")
    print(f"COLMAP目录: {args.colmap_dir}")
    print(f"GT图片目录: {args.gt_dir}")
    print(f"输出目录:   {args.output_dir}")
    print("=" * 50)
    
    try:
        evaluate_gaussian_direct(
            ply_path=args.ply_path,
            colmap_sparse_dir=args.colmap_dir,
            images_dir=args.gt_dir,
            output_dir=args.output_dir,
            device=args.device
        )
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 