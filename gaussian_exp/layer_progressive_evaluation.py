import os
import sys
import torch
import numpy as np
import argparse
import json
from pathlib import Path

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
from PIL import Image

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_gaussians(ply_path):
    """加载高斯球模型"""
    print(f"🎯 加载高斯球: {ply_path}")
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    return gaussians

def analyze_position_layers(gaussians, num_layers=5):
    """按位置分层分析并返回层级掩码"""
    print(f"📍 按位置分层分析 ({num_layers}层)")
    
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()
    
    # 过滤掉NaN值
    valid_mask = ~np.isnan(xyz).any(axis=1) & ~np.isnan(opacity)
    total_count = len(xyz)
    valid_indices = np.where(valid_mask)[0]
    
    xyz_valid = xyz[valid_mask]
    print(f"  有效高斯球数量: {len(xyz_valid)} / {total_count}")
    
    # 按Z轴分层（深度）
    z_min, z_max = xyz_valid[:, 2].min(), xyz_valid[:, 2].max()
    z_step = (z_max - z_min) / num_layers
    
    print(f"  Z轴范围: [{z_min:.3f}, {z_max:.3f}]")
    print(f"  每层厚度: {z_step:.3f}")
    
    layer_masks = []
    layer_info = []
    
    for i in range(num_layers):
        z_start = z_min + i * z_step
        z_end = z_min + (i + 1) * z_step
        
        # 最后一层包含边界
        if i == num_layers - 1:
            layer_valid_mask = (xyz_valid[:, 2] >= z_start) & (xyz_valid[:, 2] <= z_end)
        else:
            layer_valid_mask = (xyz_valid[:, 2] >= z_start) & (xyz_valid[:, 2] < z_end)
        
        # 将有效高斯球的掩码映射回全局掩码
        global_mask = np.zeros(total_count, dtype=bool)
        global_mask[valid_indices[layer_valid_mask]] = True
        
        layer_count = global_mask.sum()
        layer_masks.append(global_mask)
        
        info = {
            'layer_id': i,
            'z_range': [z_start, z_end],
            'count': layer_count,
            'percentage': layer_count / total_count * 100
        }
        layer_info.append(info)
        
        print(f"  层 {i:2d} [{z_start:7.2f}, {z_end:7.2f}]: {layer_count:7d}个高斯球 ({info['percentage']:5.1f}%)")
    
    return layer_masks, layer_info

def create_layer_ply_files(gaussians, layer_masks, layer_info, output_dir):
    """为每一层创建PLY文件"""
    print(f"\n💾 创建分层PLY文件...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有高斯球属性
    xyz = gaussians.get_xyz
    features_dc = gaussians._features_dc
    features_rest = gaussians._features_rest
    scaling = gaussians._scaling
    rotation = gaussians._rotation
    opacity = gaussians._opacity
    
    layer_files = []
    
    for i, (mask, info) in enumerate(zip(layer_masks, layer_info)):
        if mask.sum() == 0:
            print(f"  ⚠️ 层 {i} 为空，跳过")
            continue
            
        # 创建新的高斯模型
        layer_gaussians = GaussianModel(3)
        
        # 复制属性到新模型
        layer_gaussians._xyz = xyz[mask].clone()
        layer_gaussians._features_dc = features_dc[mask].clone()
        layer_gaussians._features_rest = features_rest[mask].clone()
        layer_gaussians._scaling = scaling[mask].clone()
        layer_gaussians._rotation = rotation[mask].clone()
        layer_gaussians._opacity = opacity[mask].clone()
        
        # 设置其他必要属性
        layer_gaussians.active_sh_degree = gaussians.active_sh_degree
        layer_gaussians.max_sh_degree = gaussians.max_sh_degree
        
        # 保存PLY文件
        layer_file = os.path.join(output_dir, f"layer_{i}_z{info['z_range'][0]:.1f}to{info['z_range'][1]:.1f}_{info['count']}balls.ply")
        
        # 手动保存PLY（因为save_ply可能需要优化器）
        save_layer_ply(layer_gaussians, layer_file)
        
        layer_files.append(layer_file)
        print(f"  ✅ 层 {i}: {layer_file} ({info['count']}个高斯球)")
    
    return layer_files

def save_layer_ply(gaussians, path):
    """手动保存PLY文件"""
    import plyfile
    
    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
    dtype_full += [(attribute, 'f4') for attribute in ['f_dc_0', 'f_dc_1', 'f_dc_2']]
    dtype_full += [(f'f_rest_{i}', 'f4') for i in range(f_rest.shape[1])]
    dtype_full += [('opacity', 'f4')]
    dtype_full += [(f'scale_{i}', 'f4') for i in range(scale.shape[1])]
    dtype_full += [(f'rot_{i}', 'f4') for i in range(rotation.shape[1])]

    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    for i, attr in enumerate(dtype_full):
        elements[attr[0]] = attributes[:, i]

    vertex_element = plyfile.PlyElement.describe(elements, 'vertex')
    plyfile.PlyData([vertex_element]).write(path)

def create_progressive_ply_files(layer_files, layer_info, output_dir):
    """创建渐进式累积的PLY文件 (1, 1+2, 1+2+3, ...)"""
    print(f"\n📈 创建渐进式累积PLY文件...")
    
    progressive_files = []
    
    for i in range(len(layer_files)):
        # 累积前i+1层
        combined_gaussians = None
        total_count = 0
        layer_names = []
        
        for j in range(i + 1):
            layer_gaussians = GaussianModel(3)
            layer_gaussians.load_ply(layer_files[j], use_train_test_exp=False)
            
            if combined_gaussians is None:
                combined_gaussians = layer_gaussians
            else:
                # 合并高斯球
                combined_gaussians = combine_gaussians(combined_gaussians, layer_gaussians)
            
            total_count += layer_info[j]['count']
            layer_names.append(f"L{j}")
        
        # 保存累积文件
        progressive_file = os.path.join(output_dir, f"progressive_{'_'.join(layer_names)}_{total_count}balls.ply")
        save_layer_ply(combined_gaussians, progressive_file)
        progressive_files.append(progressive_file)
        
        print(f"  ✅ 累积 {'->'.join(layer_names)}: {progressive_file} ({total_count}个高斯球)")
    
    return progressive_files

def combine_gaussians(gaussians1, gaussians2):
    """合并两个高斯模型"""
    combined = GaussianModel(3)
    
    # 合并所有属性
    combined._xyz = torch.cat([gaussians1._xyz, gaussians2._xyz], dim=0)
    combined._features_dc = torch.cat([gaussians1._features_dc, gaussians2._features_dc], dim=0)
    combined._features_rest = torch.cat([gaussians1._features_rest, gaussians2._features_rest], dim=0)
    combined._scaling = torch.cat([gaussians1._scaling, gaussians2._scaling], dim=0)
    combined._rotation = torch.cat([gaussians1._rotation, gaussians2._rotation], dim=0)
    combined._opacity = torch.cat([gaussians1._opacity, gaussians2._opacity], dim=0)
    
    # 设置其他属性
    combined.active_sh_degree = gaussians1.active_sh_degree
    combined.max_sh_degree = gaussians1.max_sh_degree
    
    return combined

def load_single_camera(colmap_path, images_path, resolution_scale=2.0):
    """加载单个相机用于快速测试"""
    cameras_bin = os.path.join(colmap_path, 'cameras.bin')
    images_bin = os.path.join(colmap_path, 'images.bin')
    
    cam_intrinsics = read_intrinsics_binary(cameras_bin)
    cam_extrinsics = read_extrinsics_binary(images_bin)
    
    # 取第一个相机
    first_img_id = list(cam_extrinsics.keys())[0]
    img_info = cam_extrinsics[first_img_id]
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
        colmap_id=first_img_id,
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
    """评估单个PLY文件"""
    if not os.path.exists(ply_path):
        return {"psnr": 0.0, "l1_loss": 0.0, "error": "File not found"}
    
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
            "gaussian_count": gaussians.get_xyz.shape[0],
            "error": None
        }
        
    except Exception as e:
        return {"psnr": 0.0, "l1_loss": 0.0, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='分层渐进式高斯球评估')
    parser.add_argument('--ply-path', type=str, required=True, help='原始PLY文件路径')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--num-layers', type=int, default=5, help='分层数量')
    parser.add_argument('--output-dir', type=str, default='layer_progressive_analysis', help='输出目录')
    parser.add_argument('--resolution-scale', type=float, default=2.0, help='分辨率缩放')
    
    args = parser.parse_args()
    
    print("🔍 分层渐进式高斯球评估")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载原始高斯球
    gaussians = load_gaussians(args.ply_path)
    
    # 2. 分层分析
    layer_masks, layer_info = analyze_position_layers(gaussians, args.num_layers)
    
    # 3. 创建分层PLY文件
    layer_files = create_layer_ply_files(gaussians, layer_masks, layer_info, args.output_dir)
    
    # 4. 创建渐进式PLY文件
    progressive_files = create_progressive_ply_files(layer_files, layer_info, args.output_dir)
    
    # 5. 设置评估环境
    print(f"\n🎨 设置评估环境...")
    
    # Pipeline参数
    pipeline_parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(pipeline_parser)
    pipe_args = pipeline_parser.parse_args([])
    pipe = pipe_parser.extract(pipe_args)
    
    # 背景
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 加载测试相机
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    camera = load_single_camera(colmap_path, images_path, args.resolution_scale)
    
    print(f"✅ 使用相机: {camera.image_name}")
    
    # 6. 评估所有文件
    print(f"\n📊 评估分层文件...")
    
    all_results = []
    
    # 评估单层文件
    for i, (layer_file, info) in enumerate(zip(layer_files, layer_info)):
        print(f"  评估层 {i}...")
        result = evaluate_ply_file(layer_file, camera, pipe, background)
        result.update({
            "type": "single_layer",
            "layer_id": i,
            "layer_range": info['z_range'],
            "file_path": layer_file
        })
        all_results.append(result)
        
        if result["error"]:
            print(f"    ❌ 错误: {result['error']}")
        else:
            print(f"    ✅ PSNR: {result['psnr']:.3f} dB, L1: {result['l1_loss']:.6f}, 高斯球: {result['gaussian_count']}")
    
    # 评估渐进式文件
    print(f"\n📈 评估渐进式累积文件...")
    for i, prog_file in enumerate(progressive_files):
        print(f"  评估累积 L0-L{i}...")
        result = evaluate_ply_file(prog_file, camera, pipe, background)
        result.update({
            "type": "progressive",
            "layers_included": list(range(i + 1)),
            "file_path": prog_file
        })
        all_results.append(result)
        
        if result["error"]:
            print(f"    ❌ 错误: {result['error']}")
        else:
            print(f"    ✅ PSNR: {result['psnr']:.3f} dB, L1: {result['l1_loss']:.6f}, 高斯球: {result['gaussian_count']}")
    
    # 7. 保存结果
    results_file = os.path.join(args.output_dir, 'layer_progressive_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'layer_info': layer_info,
            'evaluation_results': all_results,
            'camera_info': {
                'name': camera.image_name,
                'resolution': [camera.image_width, camera.image_height],
                'resolution_scale': args.resolution_scale
            },
            'original_file': args.ply_path,
            'total_gaussians': gaussians.get_xyz.shape[0]
        }, f, indent=2)
    
    print(f"\n🎉 分层评估完成!")
    print(f"📊 结果保存在: {args.output_dir}/")
    print(f"📁 分层文件: {len(layer_files)}个")
    print(f"📈 渐进文件: {len(progressive_files)}个")
    print(f"📋 评估结果: {results_file}")

if __name__ == "__main__":
    main() 