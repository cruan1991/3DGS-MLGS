#!/usr/bin/env python3
"""
准确模拟训练时参数的评估脚本
基于utils/camera_utils.py的逻辑
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
    """完全模拟train.py的PipelineParams"""
    def __init__(self):
        # 与train.py保持一致的渲染管道参数
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.antialiasing = False  # 除非特别指定
        self.debug = False
        
        # 确保与训练时的背景设置一致
        self.SPARSE_ADAM_AVAILABLE = False

class DummyArgs:
    """完全模拟train.py的参数"""
    def __init__(self):
        # 关键参数：与train.py保持完全一致
        self.resolution = -1  # 表示自动缩放
        self.train_test_exp = False  # cfg_args确认 
        self.eval = False  # 所以没有test set
        self.white_background = False
        self.data_device = "cuda"
        
        # 确保sparse adam设置一致
        self.SPARSE_ADAM_AVAILABLE = False

def create_camera_like_training(image_info, intrinsics, images_dir, resolution_scale=1.0):
    """完全按照训练时的逻辑创建Camera对象"""
    
    args = DummyArgs()
    
    # 加载图片
    image_path = os.path.join(images_dir, image_info.name)
    pil_image = Image.open(image_path)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # 模拟训练时的分辨率处理逻辑 (来自camera_utils.py)
    orig_w, orig_h = pil_image.size
    print(f"原始尺寸: {orig_w}x{orig_h}")
    
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # args.resolution == -1
        if orig_w > 1600:
            print("[ INFO ] 图片宽度>1600, 缩放到1600")
            global_down = orig_w / 1600
        else:
            global_down = 1
            print(f"[ INFO ] 图片宽度<1600, global_down = {global_down}")
        
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    print(f"计算的分辨率: {resolution}, global_down={global_down if 'global_down' in locals() else 'N/A'}, scale={scale if 'scale' in locals() else 'N/A'}")
    
    # 调整图片尺寸
    if resolution != (orig_w, orig_h):
        pil_image = pil_image.resize(resolution, Image.LANCZOS)
        print(f"调整后尺寸: {resolution}")
    
    # 转换为tensor
    image_tensor = TF.to_tensor(pil_image)
    
    # 计算相机参数
    width, height = resolution
    
    # 获取内参并根据分辨率调整
    fx, fy = intrinsics.params[0], intrinsics.params[1]
    cx, cy = intrinsics.params[2], intrinsics.params[3]
    
    # 调整内参以匹配新分辨率
    if resolution != (orig_w, orig_h):
        scale_x = width / orig_w
        scale_y = height / orig_h
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
    
    # 计算FoV
    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)
    
    print(f"调整后内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"FoV: {FoVx*180/np.pi:.1f}° x {FoVy*180/np.pi:.1f}°")
    
    # 相机外参 - 关键：与train.py保持一致！
    # 在dataset_readers.py第85行: R = np.transpose(qvec2rotmat(extr.qvec))
    R = np.transpose(image_info.qvec2rotmat())  # 🔥 这是关键修复！
    T = np.array(image_info.tvec)
    
    # 创建Camera对象 (完全按照训练时的方式)
    camera = Camera(
        resolution=resolution,
        colmap_id=image_info.id,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=image_tensor,
        invdepthmap=None,
        image_name=image_info.name,
        uid=image_info.id,
        data_device=args.data_device,
        train_test_exp=args.train_test_exp,
        is_test_dataset=True,
        is_test_view=True
    )
    
    return camera

def test_training_accurate(ply_path, colmap_sparse_dir, images_dir):
    print("🎯 使用训练精确参数测试")
    
    # 加载高斯球
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    
    # 读取COLMAP数据
    cameras_intrinsic_file = os.path.join(colmap_sparse_dir, "cameras.bin")
    cameras_extrinsic_file = os.path.join(colmap_sparse_dir, "images.bin")
    
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    
    # 测试多个不同的resolution_scale
    scales_to_test = [1.0, 2.0, 4.0]
    
    for resolution_scale in scales_to_test:
        print(f"\n{'='*50}")
        print(f"🔍 测试 resolution_scale = {resolution_scale}")
        print(f"{'='*50}")
        
        # 只测试第一个相机
        image_id, image_info = list(cam_extrinsics.items())[0]
        camera_id = image_info.camera_id
        intrinsics = cam_intrinsics[camera_id]
        
        try:
            camera = create_camera_like_training(image_info, intrinsics, images_dir, resolution_scale)
            
            # 渲染 - 使用与train.py一致的背景
            pipe = DummyRenderPipe()
            args = DummyArgs()
            # 与train.py第229行保持一致: bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            bg_color = torch.tensor([1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0], 
                                  dtype=torch.float32, device='cuda')
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
            
            rendered_img.save(f"train_accurate_scale_{resolution_scale}_rendered.png")
            gt_img.save(f"train_accurate_scale_{resolution_scale}_gt.png")
            print(f"保存样本: train_accurate_scale_{resolution_scale}_*.png")
            
        except Exception as e:
            print(f"❌ resolution_scale {resolution_scale} 失败: {e}")

def main():
    ply_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    colmap_sparse_dir = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/sparse/0"
    images_dir = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/images"
    
    test_training_accurate(ply_path, colmap_sparse_dir, images_dir)

if __name__ == '__main__':
    main() 