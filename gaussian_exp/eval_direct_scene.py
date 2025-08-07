#!/usr/bin/env python3
# 直接使用Scene加载最佳iteration进行评估
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import os
import argparse
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render

# Set CUDA device
torch.cuda.set_device(1)

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def eval_direct_scene(model_path):
    print("🚀 Direct Scene evaluation")
    print(f"📁 Model path: {model_path}")
    
    # 检查SPARSE_ADAM_AVAILABLE
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except:
        SPARSE_ADAM_AVAILABLE = False
    
    print(f"🔧 SPARSE_ADAM_AVAILABLE: {SPARSE_ADAM_AVAILABLE}")
    
    # 创建符号链接到原始图片
    original_images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    model_images_path = os.path.join(model_path, "images")
    
    if not os.path.exists(model_images_path):
        print(f"🔗 Creating symlink for images")
        os.symlink(original_images_path, model_images_path)
    
    # 设置参数
    parser = argparse.ArgumentParser()
    dataset_parser = ModelParams(parser)
    pipeline_parser = PipelineParams(parser)
    args = parser.parse_args([])
    
    args.source_path = model_path
    args.model_path = model_path
    args.images = "images"
    args.resolution = 2  # 🔍 尝试使用2倍下采样，可能更接近训练时的设置
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.train_test_exp = False  # 🔥 关键修复：训练时没有使用train_test_exp！
    
    dataset = dataset_parser.extract(args)
    pipe = pipeline_parser.extract(args)
    
    print(f"🎨 Dataset train_test_exp: {dataset.train_test_exp}")
    
    # 创建高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    
    # 🔥 关键：让Scene直接加载最佳PSNR的iteration
    # 我们需要找到实际的iteration号
    best_psnr_iteration = 994230  # 从之前的日志中知道
    
    scene = Scene(dataset, gaussians, load_iteration=best_psnr_iteration)
    
    # 检查曝光参数
    print(f"🔍 Checking exposure parameters...")
    print(f"  - hasattr(gaussians, 'pretrained_exposures'): {hasattr(gaussians, 'pretrained_exposures')}")
    if hasattr(gaussians, 'pretrained_exposures'):
        print(f"  - gaussians.pretrained_exposures is None: {gaussians.pretrained_exposures is None}")
        if gaussians.pretrained_exposures is not None:
            print(f"  - Number of pretrained exposures: {len(gaussians.pretrained_exposures)}")
    
    print(f"  - hasattr(gaussians, '_exposure'): {hasattr(gaussians, '_exposure')}")
    print(f"  - hasattr(gaussians, 'exposure_mapping'): {hasattr(gaussians, 'exposure_mapping')}")
    
    if hasattr(gaussians, 'pretrained_exposures') and gaussians.pretrained_exposures is not None:
        print(f"✅ Loaded exposure parameters for {len(gaussians.pretrained_exposures)} images")
    else:
        print("⚠️ No exposure parameters loaded")
    
    # 获取相机
    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    
    print(f"📷 Test cameras: {len(test_cameras)}")
    print(f"📷 Train cameras: {len(train_cameras)}")
    
    # 设置背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 选择评估相机
    if len(test_cameras) == 0:
        print("⚠️ No test cameras, using train camera subset")
        eval_cameras = [train_cameras[idx % len(train_cameras)] for idx in range(5, min(30, len(train_cameras)), 5)]
    else:
        eval_cameras = test_cameras
    
    # 渲染参数
    renderArgs = (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
    
    # 评估
    total_psnr = 0.0
    for i, viewpoint in enumerate(eval_cameras[:5]):
        # 按照train.py的渲染方式
        image = torch.clamp(render(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        
        # 处理train_test_exp
        if dataset.train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
        
        # 计算PSNR
        psnr_val = psnr(image, gt_image).mean().item()
        total_psnr += psnr_val
        
        print(f"📷 Camera {i} ({viewpoint.image_name}): PSNR = {psnr_val:.2f} dB")
        
        # 保存图像
        if i < 3:
            from torchvision.utils import save_image
            os.makedirs("direct_renders", exist_ok=True)
            save_image(image, f"direct_renders/{viewpoint.image_name}_render.png")
            save_image(gt_image, f"direct_renders/{viewpoint.image_name}_gt.png")
    
    avg_psnr = total_psnr / min(5, len(eval_cameras))
    print(f"\n🎉 Average PSNR: {avg_psnr:.2f} dB")
    print(f"📊 Expected from training: ~33.83 dB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    
    eval_direct_scene(args.model_path)

if __name__ == "__main__":
    main() 