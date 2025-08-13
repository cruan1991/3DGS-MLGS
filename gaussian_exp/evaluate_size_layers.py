import os
import sys
import torch
import numpy as np
import argparse
import json
import glob
from PIL import Image

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel
from scene.cameras import Camera
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_test_camera(colmap_path, images_path, camera_name="000001.jpg", resolution_scale=2.0):
    """加载测试相机"""
    cameras_bin = os.path.join(colmap_path, 'cameras.bin')
    images_bin = os.path.join(colmap_path, 'images.bin')
    
    cam_intrinsics = read_intrinsics_binary(cameras_bin)
    cam_extrinsics = read_extrinsics_binary(images_bin)
    
    # 找到指定相机
    target_img_id = None
    for img_id, img_info in cam_extrinsics.items():
        if img_info.name == camera_name:
            target_img_id = img_id
            break
    
    if target_img_id is None:
        print(f"❌ 未找到相机: {camera_name}")
        return None
    
    img_info = cam_extrinsics[target_img_id]
    intrinsic = cam_intrinsics[img_info.camera_id]
    
    # 解析参数
    fx, fy, cx, cy = intrinsic.params
    width = int(intrinsic.width / resolution_scale)
    height = int(intrinsic.height / resolution_scale)
    fx_scaled = fx / resolution_scale
    fy_scaled = fy / resolution_scale
    
    FoVx = focal2fov(fx_scaled, width)
    FoVy = focal2fov(fy_scaled, height)
    
    R = np.transpose(qvec2rotmat(img_info.qvec))
    T = np.array(img_info.tvec)
    
    # 加载图像
    image_path = os.path.join(images_path, img_info.name)
    image = Image.open(image_path)
    if resolution_scale != 1.0:
        image = image.resize((width, height), Image.LANCZOS)
    
    camera = Camera(
        resolution=(width, height),
        colmap_id=target_img_id,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=image,
        invdepthmap=None,
        image_name=img_info.name,
        uid=0,
        data_device="cuda",
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False
    )
    
    return camera

def evaluate_ply_file(ply_path, camera, pipe, background):
    """评估单个PLY文件的PSNR"""
    if not os.path.exists(ply_path):
        return None
    
    try:
        # 加载高斯球
        gaussians = GaussianModel(3)
        gaussians.load_ply(ply_path, use_train_test_exp=False)
        
        # 检查SPARSE_ADAM_AVAILABLE
        try:
            from diff_gaussian_rasterization import SparseGaussianAdam
            SPARSE_ADAM_AVAILABLE = True
        except:
            SPARSE_ADAM_AVAILABLE = False
        
        # 渲染
        render_result = render(camera, gaussians, pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, False)
        rendered_image = torch.clamp(render_result["render"], 0.0, 1.0)
        
        # GT图像
        gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
        
        # 计算指标
        psnr_val = psnr(rendered_image, gt_image).mean().item()
        l1_val = l1_loss(rendered_image, gt_image).mean().item()
        
        return {
            "psnr": psnr_val,
            "l1_loss": l1_val,
            "gaussian_count": gaussians.get_xyz.shape[0]
        }
        
    except Exception as e:
        print(f"  ❌ 评估失败: {str(e)}")
        return None

def evaluate_size_layers(layers_dir, output_file='size_layers_evaluation.json'):
    """评估尺寸分层的渐进式PSNR"""
    print("📊 评估尺寸分层的PSNR贡献")
    print("=" * 50)
    
    # 设置渲染环境
    pipeline_parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(pipeline_parser)
    pipe_args = pipeline_parser.parse_args([])
    pipe = pipe_parser.extract(pipe_args)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 加载相机
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    camera = load_test_camera(colmap_path, images_path, "000001.jpg", 2.0)
    
    if camera is None:
        return
    
    print(f"✅ 加载测试相机: 000001.jpg")
    
    # 查找所有PLY文件
    single_layer_files = sorted(glob.glob(os.path.join(layers_dir, "size_layer_*.ply")))
    progressive_files = sorted(glob.glob(os.path.join(layers_dir, "size_progressive_*.ply")))
    
    print(f"📁 找到单层文件: {len(single_layer_files)}个")
    print(f"📈 找到渐进文件: {len(progressive_files)}个")
    
    # 评估单层文件
    print(f"\n🎯 评估单层文件...")
    single_results = []
    
    for i, ply_file in enumerate(single_layer_files):
        filename = os.path.basename(ply_file)
        print(f"  评估 {filename}...")
        
        result = evaluate_ply_file(ply_file, camera, pipe, background)
        if result is not None:
            result['file'] = filename
            result['layer_id'] = i
            single_results.append(result)
            print(f"    ✅ PSNR: {result['psnr']:.2f}dB, L1: {result['l1_loss']:.6f}, 球数: {result['gaussian_count']:,}")
        else:
            print(f"    ❌ 评估失败")
    
    # 评估渐进文件
    print(f"\n📈 评估渐进式累积...")
    progressive_results = []
    
    for i, ply_file in enumerate(progressive_files):
        filename = os.path.basename(ply_file)
        print(f"  评估 {filename}...")
        
        result = evaluate_ply_file(ply_file, camera, pipe, background)
        if result is not None:
            result['file'] = filename
            result['cumulative_layers'] = i + 1
            progressive_results.append(result)
            print(f"    ✅ PSNR: {result['psnr']:.2f}dB, L1: {result['l1_loss']:.6f}, 球数: {result['gaussian_count']:,}")
        else:
            print(f"    ❌ 评估失败")
    
    # 分析贡献
    print(f"\n🔍 分析PSNR贡献...")
    contribution_analysis = []
    
    layer_names = ['超小球', '小球', '中球', '大球', '超大球']
    
    for i, result in enumerate(progressive_results):
        if i == 0:
            contribution = result['psnr']
            layers_desc = layer_names[0]
        else:
            contribution = result['psnr'] - progressive_results[i-1]['psnr']
            layers_desc = f"{layer_names[0]}-{layer_names[i]}"
        
        contribution_analysis.append({
            'stage': i,
            'layers_description': layers_desc,
            'cumulative_psnr': result['psnr'],
            'psnr_contribution': contribution,
            'gaussian_count': result['gaussian_count'],
            'efficiency': contribution / (result['gaussian_count'] / 1000000)  # PSNR/M balls
        })
        
        print(f"  阶段{i} ({layers_desc}): {result['psnr']:.2f}dB (+{contribution:.2f}), {result['gaussian_count']:,}球")
    
    # 保存结果
    evaluation_results = {
        'test_camera': '000001.jpg',
        'evaluation_timestamp': str(torch.cuda.Event()),
        'single_layer_results': single_results,
        'progressive_results': progressive_results,
        'contribution_analysis': contribution_analysis,
        'summary': {
            'total_layers': len(single_results),
            'final_psnr': progressive_results[-1]['psnr'] if progressive_results else 0,
            'total_gaussians': progressive_results[-1]['gaussian_count'] if progressive_results else 0,
            'best_efficiency_stage': max(contribution_analysis, key=lambda x: x['efficiency'])['stage'] if contribution_analysis else -1
        }
    }
    
    output_path = os.path.join(layers_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n✅ 评估结果保存: {output_path}")
    
    # 打印总结
    print(f"\n📊 评估总结:")
    print(f"  最终PSNR: {progressive_results[-1]['psnr']:.2f}dB")
    print(f"  总高斯球: {progressive_results[-1]['gaussian_count']:,}")
    print(f"  最高效阶段: 阶段{evaluation_results['summary']['best_efficiency_stage']}")
    
    # 找出贡献最大的层
    max_contrib = max(contribution_analysis, key=lambda x: x['psnr_contribution'])
    print(f"  最大贡献层: {max_contrib['layers_description']} (+{max_contrib['psnr_contribution']:.2f}dB)")
    
    return evaluation_results

def main():
    print("📊 尺寸分层PSNR评估")
    print("=" * 40)
    
    layers_dir = "size_based_layers"
    
    if not os.path.exists(layers_dir):
        print(f"❌ 分层目录不存在: {layers_dir}")
        print("请先运行 create_size_based_layers.py")
        return
    
    # 执行评估
    results = evaluate_size_layers(layers_dir)
    
    if results:
        print(f"\n🎉 尺寸分层评估完成!")
        print(f"📁 结果文件: {layers_dir}/size_layers_evaluation.json")

if __name__ == "__main__":
    main() 