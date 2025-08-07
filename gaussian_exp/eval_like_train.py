# eval_like_train.py - 完全按照train.py的逻辑
import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

import torch
import os
import argparse
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.image_utils import cal_psnr
import csv
from utils.loss_utils import l1_loss

# Set CUDA device
torch.cuda.set_device(1)

# 添加与train.py完全一致的psnr函数
def psnr(img1, img2):
    """完全按照train.py中原始psnr函数的实现"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def evaluate_like_train(model_path, ply_file_path):
    print(f"🚀 Loading model from: {model_path}")
    print(f"🎯 Using gaussian ball: {ply_file_path}")
    
    # ===== 🔥 关键修复：确保model_path下有图片文件 =====
    original_images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    model_images_path = os.path.join(model_path, "images")
    
    # 如果model_path下没有图片，创建符号链接
    if not os.path.exists(model_images_path) or len(os.listdir(model_images_path)) == 0:
        print(f"🔗 Creating symlink for images from {original_images_path} to {model_images_path}")
        if os.path.exists(model_images_path):
            os.rmdir(model_images_path)  # 删除空目录
        os.symlink(original_images_path, model_images_path)
    
    # ===== 完全按照train.py的方式加载数据集 =====
    parser = argparse.ArgumentParser()
    dataset_parser = ModelParams(parser)
    pipeline_parser = PipelineParams(parser)  # 添加Pipeline参数
    args = parser.parse_args([])  # 空参数
    
    # 🔥 关键修复：使用训练时的模型路径作为source_path，这样会加载训练时保存的相机参数
    # 而不是重新从原始数据集加载，避免相机参数不匹配的问题
    args.source_path = model_path  # 使用model_path而不是原始数据集路径！
    args.model_path = model_path
    args.images = "images"
    args.resolution = -1
    args.white_background = False  # MipNeRF-360 通常用黑背景
    args.data_device = "cuda"
    args.eval = False
    
    # 提取参数
    dataset = dataset_parser.extract(args)
    pipe = pipeline_parser.extract(args)  # 使用真正的Pipeline参数
    
    # ===== 按照train.py加载高斯模型 =====
    gaussians = GaussianModel(dataset.sh_degree)
    
    # 直接加载指定的PLY文件
    print(f"Loading gaussians from: {ply_file_path}")
    gaussians.load_ply(ply_file_path)
    print(f"Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    # ===== 按照train.py加载场景（但不加载高斯球，因为我们已经手动加载了）=====
    # 🔥 关键：现在Scene会从model_path加载训练时保存的相机参数
    # 使用load_iteration=1来避免Scene调用create_from_pcd，但实际不会加载PLY因为我们已经加载了
    scene = Scene(dataset, gaussians, load_iteration=1)  # 使用虚拟iteration避免重复初始化
    
    print(f"Scene loaded successfully")
    
    # ===== 按照train.py的测试方式 =====
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 获取测试相机（和train.py完全一致）
    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    
    print(f"Found {len(test_cameras)} test cameras")
    print(f"Found {len(train_cameras)} train cameras")
    
    # ===== 评估测试集 =====
    print("\n🎨 Evaluating test cameras...")
    if len(test_cameras) == 0:
        print("⚠️ Warning: No test cameras found! This dataset might not have a separate test split.")
        print("🔄 Using a subset of training cameras for evaluation instead...")
        # 使用训练相机的一部分作为测试
        test_subset = [train_cameras[idx] for idx in range(0, min(len(train_cameras), 10), 2)]
        test_metrics = evaluate_camera_set(test_subset, gaussians, pipe, background, "test_from_train", dataset)
    else:
        test_metrics = evaluate_camera_set(test_cameras, gaussians, pipe, background, "test", dataset)
    
    # ===== 评估训练集（部分） =====
    print("\n🎨 Evaluating train cameras...")
    # 按照train.py的方式选择训练相机
    train_subset = [train_cameras[idx % len(train_cameras)] for idx in range(5, min(30, len(train_cameras)), 5)]
    train_metrics = evaluate_camera_set(train_subset, gaussians, pipe, background, "train", dataset)
    
    print(f"\n✅ Results:")
    print(f"Test PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"Train PSNR: {train_metrics['psnr']:.2f} dB")
    
    return test_metrics, train_metrics

def evaluate_camera_set(cameras, gaussians, pipe, background, name, dataset):
    """评估相机集合，完全按照train.py的方式"""
    # 检查相机集合是否为空
    if len(cameras) == 0:
        print(f"⚠️ Warning: No {name} cameras found, skipping evaluation")
        return {'psnr': 0.0, 'loss': 0.0}
    
    all_metrics = {'psnr': [], 'loss': []}
    
    for idx, viewpoint in enumerate(cameras):
        # ===== 完全按照train.py的渲染方式 =====
        # 使用与train.py完全相同的参数: (pipe, background, scaling_modifier, separate_sh, override_color, use_trained_exp)
        image = torch.clamp(render(viewpoint, gaussians, pipe, background, 1., False, None, dataset.train_test_exp)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        
        # ===== 按照train.py处理train_test_exp =====
        if dataset.train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
        
        # ===== 按照train.py计算指标 =====
        psnr_val = psnr(image, gt_image).mean().item()
        loss = l1_loss(image, gt_image).mean().item()
        
        all_metrics['psnr'].append(psnr_val)
        all_metrics['loss'].append(loss)
        
        print(f"{name} view {idx} ({viewpoint.image_name}): PSNR = {psnr_val:.2f} dB")
        
        # 保存前几张图像用于检查
        if idx < 3:
            from torchvision.utils import save_image
            os.makedirs(f"renders/{name}", exist_ok=True)
            save_image(image, f"renders/{name}/{viewpoint.image_name.replace('.jpg', '_render.png')}")
            save_image(gt_image, f"renders/{name}/{viewpoint.image_name.replace('.jpg', '_gt.png')}")
    
    # 计算平均指标
    avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
    return avg_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, help='Path to the model directory')
    parser.add_argument('--ply-path', required=True, help='Path to the gaussian ball PLY file')
    args = parser.parse_args()
    
    evaluate_like_train(args.model_path, args.ply_path)

if __name__ == '__main__':
    main()