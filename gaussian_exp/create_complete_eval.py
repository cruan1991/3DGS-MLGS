#!/usr/bin/env python3
# 完全按照train.py的逻辑进行评估，包括曝光参数
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import os
import argparse
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.graphics_utils import focal2fov

# Set CUDA device
torch.cuda.set_device(1)

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def complete_evaluation(model_path, ply_path):
    print("🚀 Complete evaluation with exposure parameters")
    print(f"📁 Model path: {model_path}")
    print(f"🎯 PLY file: {ply_path}")
    
    # 🔥 检查SPARSE_ADAM_AVAILABLE，与train.py保持一致
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except:
        SPARSE_ADAM_AVAILABLE = False
    
    print(f"🔧 SPARSE_ADAM_AVAILABLE: {SPARSE_ADAM_AVAILABLE}")
    
    # ===== 完全按照train.py的方式加载数据集 =====
    parser = argparse.ArgumentParser()
    dataset_parser = ModelParams(parser)
    pipeline_parser = PipelineParams(parser)
    args = parser.parse_args([])
    
    # 🔥 关键：设置正确的参数，特别是train_test_exp
    args.source_path = model_path  # 使用模型路径，这样会从训练时保存的相机参数加载
    args.model_path = model_path
    args.images = "images" 
    args.resolution = -1
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    
    # 🔥 非常重要：启用train_test_exp，这样会加载曝光参数
    # 从训练日志我们知道这个模型是用train_test_exp训练的
    args.train_test_exp = True
    
    # 提取参数
    dataset = dataset_parser.extract(args)
    pipe = pipeline_parser.extract(args)
    
    print(f"🎨 Dataset train_test_exp: {dataset.train_test_exp}")
    
    # ===== 按照train.py加载场景和高斯模型 =====
    gaussians = GaussianModel(dataset.sh_degree)
    
    # 🔥 关键：加载高斯球时启用train_test_exp，这样会加载曝光参数
    gaussians.load_ply(ply_path, use_train_test_exp=dataset.train_test_exp)
    print(f"✅ Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    # 检查是否加载了曝光参数
    if hasattr(gaussians, 'pretrained_exposures') and gaussians.pretrained_exposures is not None:
        print(f"✅ Loaded exposure parameters for {len(gaussians.pretrained_exposures)} images")
    else:
        print("⚠️ No exposure parameters loaded")
    
    # 创建Scene（这会创建符号链接到原始图片）
    original_images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    model_images_path = os.path.join(model_path, "images")
    
    if not os.path.exists(model_images_path) or len(os.listdir(model_images_path)) == 0:
        print(f"🔗 Creating symlink for images")
        if os.path.exists(model_images_path):
            os.rmdir(model_images_path)
        os.symlink(original_images_path, model_images_path)
    
    # 创建Scene
    # 🔥 重要：使用load_iteration=0来避免调用create_from_pcd
    # 这样Scene会尝试从iteration_0加载，但因为我们已经加载了高斯球，不会造成问题
    try:
        scene = Scene(dataset, gaussians, load_iteration=0)
    except:
        # 如果没有iteration_0，直接创建Scene但跳过初始化
        scene = Scene(dataset, gaussians, load_iteration=None, skip_train_test_split=True)
    
    # 获取相机
    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    
    print(f"📷 Test cameras: {len(test_cameras)}")
    print(f"📷 Train cameras: {len(train_cameras)}")
    
    # 设置背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 重新检查曝光参数（Scene可能会重新加载）
    if hasattr(gaussians, 'pretrained_exposures') and gaussians.pretrained_exposures is not None:
        print(f"✅ Exposure parameters available for {len(gaussians.pretrained_exposures)} images")
    else:
        print("⚠️ No exposure parameters available")
    
    # ===== 按照train.py的方式评估 =====
    # 使用train.py中完全相同的参数
    renderArgs = (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
    
    # 如果没有测试相机，使用训练相机的子集
    if len(test_cameras) == 0:
        print("⚠️ No test cameras, using train camera subset")
        eval_cameras = [train_cameras[idx % len(train_cameras)] for idx in range(5, min(30, len(train_cameras)), 5)]
        camera_type = "train_subset"
    else:
        eval_cameras = test_cameras
        camera_type = "test"
    
    # 评估
    total_psnr = 0.0
    for i, viewpoint in enumerate(eval_cameras[:5]):
        # 🔥 完全按照train.py的渲染方式
        image = torch.clamp(render(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        
        # 🔥 按照train.py处理train_test_exp
        if dataset.train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
        
        # 计算PSNR
        psnr_val = psnr(image, gt_image).mean().item()
        total_psnr += psnr_val
        
        print(f"📷 Camera {i} ({viewpoint.image_name}): PSNR = {psnr_val:.2f} dB")
        
        # 保存图像用于检查
        if i < 3:
            from torchvision.utils import save_image
            os.makedirs("complete_renders", exist_ok=True)
            save_image(image, f"complete_renders/{viewpoint.image_name}_render.png")
            save_image(gt_image, f"complete_renders/{viewpoint.image_name}_gt.png")
    
    avg_psnr = total_psnr / min(5, len(eval_cameras))
    print(f"\n🎉 Average PSNR: {avg_psnr:.2f} dB")
    print(f"📊 Expected from training: ~33.83 dB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--ply-path', required=True)
    args = parser.parse_args()
    
    complete_evaluation(args.model_path, args.ply_path)

if __name__ == "__main__":
    main() 