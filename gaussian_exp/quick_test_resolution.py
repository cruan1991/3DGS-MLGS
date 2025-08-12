#!/usr/bin/env python3
"""
快速测试分辨率缩放的脚本
"""

import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import os
import torch
from tqdm import tqdm
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

class DummyRenderPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.antialiasing = False
        self.debug = False

def test_resolution_scale(ply_path, colmap_sparse_dir, images_dir, scale):
    print(f"\n🔍 测试分辨率缩放: {scale}")
    
    # 加载高斯球
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    
    # 读取COLMAP数据
    cameras_intrinsic_file = os.path.join(colmap_sparse_dir, "cameras.bin")
    cameras_extrinsic_file = os.path.join(colmap_sparse_dir, "images.bin")
    
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    
    # 只测试第一个相机
    image_id, image = list(cam_extrinsics.items())[0]
    camera_id = image.camera_id
    intrinsics = cam_intrinsics[camera_id]
    
    # 加载图片
    image_path = os.path.join(images_dir, image.name)
    pil_image = Image.open(image_path)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
        
    width, height = pil_image.size
    print(f"原始图片尺寸: {width}x{height}")
    
    # 应用分辨率缩放
    if scale != 1.0:
        new_width = int(width / scale)
        new_height = int(height / scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        width, height = new_width, new_height
        print(f"缩放后尺寸: {width}x{height}")
    
    # 转换为tensor
    image_tensor = TF.to_tensor(pil_image)
    
    # 计算FoV
    fx, fy = intrinsics.params[0], intrinsics.params[1]
    if scale != 1.0:
        fx = fx / scale
        fy = fy / scale
        
    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)
    print(f"FoV: {FoVx*180/np.pi:.1f}° x {FoVy*180/np.pi:.1f}°")
    
    # 相机外参
    R = image.qvec2rotmat()
    T = image.tvec
    
    # 创建Camera对象
    camera = Camera(
        resolution=(width, height),
        colmap_id=image_id,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=image_tensor,
        invdepthmap=None,
        image_name=image.name,
        uid=0,
        data_device='cuda'
    )
    
    # 渲染
    pipe = DummyRenderPipe()
    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')
    render_pkg = render(camera, gaussians, pipe=pipe, bg_color=bg_color)
    rendered = torch.clamp(render_pkg['render'], 0.0, 1.0)
    gt = torch.clamp(camera.original_image.to(rendered.device), 0.0, 1.0)
    
    print(f"渲染尺寸: {rendered.shape}")
    print(f"GT尺寸: {gt.shape}")
    
    # 计算PSNR
    mse = torch.mean((rendered - gt) ** 2)
    if mse > 0:
        psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    else:
        psnr_val = 100.0
    print(f"PSNR: {psnr_val:.2f} dB")
    
    # 保存样本图片
    def tensor_to_image(tensor):
        if tensor.is_cuda:
            tensor = tensor.detach().cpu()
        else:
            tensor = tensor.detach()
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(tensor)
    
    rendered_img = tensor_to_image(rendered)
    gt_img = tensor_to_image(gt)
    
    rendered_img.save(f"test_scale_{scale}_rendered.png")
    gt_img.save(f"test_scale_{scale}_gt.png")
    print(f"保存样本图片: test_scale_{scale}_*.png")

def main():
    ply_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    colmap_sparse_dir = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/sparse/0"
    images_dir = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/images"
    
    print("🚀 快速分辨率缩放测试")
    print("=" * 40)
    
    # 测试不同的分辨率缩放
    scales = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    for scale in scales:
        try:
            test_resolution_scale(ply_path, colmap_sparse_dir, images_dir, scale)
        except Exception as e:
            print(f"❌ 缩放 {scale} 失败: {e}")
    
    print("\n✅ 测试完成！检查生成的图片来选择最佳缩放比例")

if __name__ == '__main__':
    main() 