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
    
    # 🚀 评估所有相机以找出33.83 dB的来源
    print(f"🎯 全面评估：评估所有 {len(cameras)} 个相机...")
    
    results = []
    total_psnr = 0.0
    
    for i, camera in enumerate(cameras):
        if i % 20 == 0:  # 每20个显示进度
            print(f"   进度: {i}/{len(cameras)} ({i/len(cameras)*100:.1f}%)")
        
        try:
            # 渲染
            rendered = torch.clamp(render(camera, gaussians, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
            
            # 计算PSNR
            psnr_val = psnr(rendered, gt_image).mean().item()
            total_psnr += psnr_val
            
            result = {
                'camera_idx': i,
                'camera_name': camera.image_name,
                'colmap_id': camera.colmap_id,
                'psnr': psnr_val,
                'resolution': f"{gt_image.shape[2]}x{gt_image.shape[1]}"
            }
            results.append(result)
            
            # 显示高PSNR的相机
            if psnr_val > 30.0:
                print(f"      🔥 Camera {i:3d} ({camera.image_name}): {psnr_val:.3f} dB")
                
        except Exception as e:
            print(f"      ❌ Error with camera {i}: {e}")
            continue
    
    print(f"   完成! 评估了 {len(results)}/{len(cameras)} 个相机")
    
    # 详细统计分析
    if results:
        avg_psnr = total_psnr / len(results)
        psnr_values = [r['psnr'] for r in results]
        
        import numpy as np
        
        print(f"\n📊 全面评估结果:")
        print(f"   总相机数: {len(results)}")
        print(f"   平均PSNR: {avg_psnr:.3f} dB")
        print(f"   最高PSNR: {max(psnr_values):.3f} dB")
        print(f"   最低PSNR: {min(psnr_values):.3f} dB")
        print(f"   中位数:   {np.median(psnr_values):.3f} dB")
        print(f"   标准差:   {np.std(psnr_values):.3f} dB")
        
        # 找出最佳相机
        best = max(results, key=lambda x: x['psnr'])
        worst = min(results, key=lambda x: x['psnr'])
        print(f"\n🏆 最佳相机: {best['camera_name']} (索引 {best['camera_idx']}) - {best['psnr']:.3f} dB")
        print(f"💩 最差相机: {worst['camera_name']} (索引 {worst['camera_idx']}) - {worst['psnr']:.3f} dB")
        
        # 分析高PSNR相机
        high_psnr_cameras = [r for r in results if r['psnr'] > 30.0]
        ultra_high_cameras = [r for r in results if r['psnr'] > 33.0]
        excellent_cameras = [r for r in results if r['psnr'] > 35.0]
        
        print(f"\n🎯 PSNR分档统计:")
        print(f"   > 30 dB: {len(high_psnr_cameras):3d} 个 ({len(high_psnr_cameras)/len(results)*100:.1f}%)")
        print(f"   > 33 dB: {len(ultra_high_cameras):3d} 个 ({len(ultra_high_cameras)/len(results)*100:.1f}%)")
        print(f"   > 35 dB: {len(excellent_cameras):3d} 个 ({len(excellent_cameras)/len(results)*100:.1f}%)")
        
        if ultra_high_cameras:
            print(f"\n🚀 超高PSNR相机 (>33 dB):")
            for r in sorted(ultra_high_cameras, key=lambda x: x['psnr'], reverse=True):
                print(f"     {r['camera_name']} (索引 {r['camera_idx']}): {r['psnr']:.3f} dB ⭐")
        else:
            print(f"\n❓ 未找到PSNR > 33 dB的相机")
            print(f"   训练时的33.83 dB可能来自:")
            print(f"   1. 不同的相机子集选择策略")
            print(f"   2. 不同的迭代/检查点")
            print(f"   3. 不同的评估参数设置")
            print(f"   4. 多次运行的平均值")
        
        # 显示Top 15
        print(f"\n🔟 前15名相机:")
        top_15 = sorted(results, key=lambda x: x['psnr'], reverse=True)[:15]
        for i, r in enumerate(top_15, 1):
            star = "⭐" if r['psnr'] > 33.0 else "🔥" if r['psnr'] > 30.0 else ""
            print(f"   {i:2d}. {r['camera_name']} (索引 {r['camera_idx']}): {r['psnr']:.3f} dB {star}")
        
        # 保存详细结果
        import json
        output_file = "complete_camera_evaluation.json"
        json_data = {
            'summary': {
                'total_cameras': len(results),
                'average_psnr': avg_psnr,
                'max_psnr': max(psnr_values),
                'min_psnr': min(psnr_values),
                'median_psnr': float(np.median(psnr_values)),
                'std_psnr': float(np.std(psnr_values)),
                'cameras_above_30': len(high_psnr_cameras),
                'cameras_above_33': len(ultra_high_cameras),
                'cameras_above_35': len(excellent_cameras)
            },
            'all_results': results,
            'top_15': top_15,
            'ultra_high_psnr': ultra_high_cameras
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\n💾 详细结果已保存至: {output_file}")
        
        print(f"\n📈 与目标33.83 dB的对比:")
        print(f"   当前最高: {max(psnr_values):.3f} dB")
        print(f"   差距:     {33.83 - max(psnr_values):.3f} dB")
        print(f"   当前平均: {avg_psnr:.3f} dB") 
        print(f"   平均差距: {33.83 - avg_psnr:.3f} dB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--ply-path', required=True)
    args = parser.parse_args()
    
    eval_with_correct_cameras(args.model_path, args.ply_path)

if __name__ == "__main__":
    main() 
        except Exception as e:
            print(f"      ❌ Error with camera {i}: {e}")
            continue
    
    print(f"   完成! 评估了 {len(results)}/{len(cameras)} 个相机")
    
    # 详细统计分析
    if results:
        avg_psnr = total_psnr / len(results)
        psnr_values = [r['psnr'] for r in results]
        
        import numpy as np
        
        print(f"\n📊 全面评估结果:")
        print(f"   总相机数: {len(results)}")
        print(f"   平均PSNR: {avg_psnr:.3f} dB")
        print(f"   最高PSNR: {max(psnr_values):.3f} dB")
        print(f"   最低PSNR: {min(psnr_values):.3f} dB")
        print(f"   中位数:   {np.median(psnr_values):.3f} dB")
        print(f"   标准差:   {np.std(psnr_values):.3f} dB")
        
        # 找出最佳相机
        best = max(results, key=lambda x: x['psnr'])
        worst = min(results, key=lambda x: x['psnr'])
        print(f"\n🏆 最佳相机: {best['camera_name']} (索引 {best['camera_idx']}) - {best['psnr']:.3f} dB")
        print(f"💩 最差相机: {worst['camera_name']} (索引 {worst['camera_idx']}) - {worst['psnr']:.3f} dB")
        
        # 分析高PSNR相机
        high_psnr_cameras = [r for r in results if r['psnr'] > 30.0]
        ultra_high_cameras = [r for r in results if r['psnr'] > 33.0]
        excellent_cameras = [r for r in results if r['psnr'] > 35.0]
        
        print(f"\n🎯 PSNR分档统计:")
        print(f"   > 30 dB: {len(high_psnr_cameras):3d} 个 ({len(high_psnr_cameras)/len(results)*100:.1f}%)")
        print(f"   > 33 dB: {len(ultra_high_cameras):3d} 个 ({len(ultra_high_cameras)/len(results)*100:.1f}%)")
        print(f"   > 35 dB: {len(excellent_cameras):3d} 个 ({len(excellent_cameras)/len(results)*100:.1f}%)")
        
        if ultra_high_cameras:
            print(f"\n🚀 超高PSNR相机 (>33 dB):")
            for r in sorted(ultra_high_cameras, key=lambda x: x['psnr'], reverse=True):
                print(f"     {r['camera_name']} (索引 {r['camera_idx']}): {r['psnr']:.3f} dB ⭐")
        else:
            print(f"\n❓ 未找到PSNR > 33 dB的相机")
            print(f"   训练时的33.83 dB可能来自:")
            print(f"   1. 不同的相机子集选择策略")
            print(f"   2. 不同的迭代/检查点")
            print(f"   3. 不同的评估参数设置")
            print(f"   4. 多次运行的平均值")
        
        # 显示Top 15
        print(f"\n🔟 前15名相机:")
        top_15 = sorted(results, key=lambda x: x['psnr'], reverse=True)[:15]
        for i, r in enumerate(top_15, 1):
            star = "⭐" if r['psnr'] > 33.0 else "🔥" if r['psnr'] > 30.0 else ""
            print(f"   {i:2d}. {r['camera_name']} (索引 {r['camera_idx']}): {r['psnr']:.3f} dB {star}")
        
        # 保存详细结果
        import json
        output_file = "complete_camera_evaluation.json"
        json_data = {
            'summary': {
                'total_cameras': len(results),
                'average_psnr': avg_psnr,
                'max_psnr': max(psnr_values),
                'min_psnr': min(psnr_values),
                'median_psnr': float(np.median(psnr_values)),
                'std_psnr': float(np.std(psnr_values)),
                'cameras_above_30': len(high_psnr_cameras),
                'cameras_above_33': len(ultra_high_cameras),
                'cameras_above_35': len(excellent_cameras)
            },
            'all_results': results,
            'top_15': top_15,
            'ultra_high_psnr': ultra_high_cameras
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\n💾 详细结果已保存至: {output_file}")
        
        print(f"\n📈 与目标33.83 dB的对比:")
        print(f"   当前最高: {max(psnr_values):.3f} dB")
        print(f"   差距:     {33.83 - max(psnr_values):.3f} dB")
        print(f"   当前平均: {avg_psnr:.3f} dB") 
        print(f"   平均差距: {33.83 - avg_psnr:.3f} dB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--ply-path', required=True)
    args = parser.parse_args()
    
    eval_with_correct_cameras(args.model_path, args.ply_path)

if __name__ == "__main__":
    main() 