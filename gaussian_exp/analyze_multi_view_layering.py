import os
import sys
import torch
import numpy as np
import argparse
import json
import glob
from PIL import Image
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel
from scene.cameras import Camera
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_multi_cameras(colmap_path, images_path, resolution_scale=6.0):
    """加载多个测试相机"""
    cameras_bin = os.path.join(colmap_path, 'cameras.bin')
    images_bin = os.path.join(colmap_path, 'images.bin')
    
    cam_intrinsics = read_intrinsics_binary(cameras_bin)
    cam_extrinsics = read_extrinsics_binary(images_bin)
    
    # 选择几个不同角度的相机
    test_cameras = []
    camera_names = ["000001.jpg", "000030.jpg", "000060.jpg", "000090.jpg"]  # 不同角度
    
    for camera_name in camera_names:
        # 找到指定相机
        target_img_id = None
        for img_id, img_info in cam_extrinsics.items():
            if img_info.name == camera_name:
                target_img_id = img_id
                break
        
        if target_img_id is None:
            print(f"⚠️ 未找到相机: {camera_name}")
            continue
        
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
        if not os.path.exists(image_path):
            print(f"⚠️ 图像文件不存在: {camera_name}")
            continue
            
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
            uid=len(test_cameras),
            data_device="cuda",
            train_test_exp=False,
            is_test_dataset=False,
            is_test_view=False
        )
        
        test_cameras.append({
            'camera': camera,
            'name': camera_name,
            'position': T,
            'rotation': R
        })
        
        print(f"✅ 加载相机: {camera_name} 位置: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}]")
    
    return test_cameras

def save_gaussians_like_original(xyz_data, normals_data, f_dc_data, f_rest_data, 
                               opacity_data, scale_data, rotation_data, output_path):
    """使用与原始save_ply完全相同的逻辑保存"""
    
    # 构造属性列表 (与原始逻辑相同)
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # DC特征
        for i in range(f_dc_data.shape[1]):
            l.append('f_dc_{}'.format(i))
        # Rest特征  
        for i in range(f_rest_data.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale_data.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation_data.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    # 组合所有属性 (与原始完全相同)
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz_data.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz_data, normals_data, f_dc_data, f_rest_data, 
                               opacity_data, scale_data, rotation_data), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # 保存文件
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)

def create_alternative_layering_strategies():
    """创建多种分层策略"""
    print("🔄 创建多种分层策略")
    print("=" * 50)
    
    # 加载原始模型
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    # 使用与原始save_ply完全相同的逻辑提取数据
    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacity = gaussians._opacity.detach().cpu().numpy()
    scaling = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()
    
    # 检查并处理NaN值
    nan_mask = np.isnan(xyz)
    nan_positions = np.any(nan_mask, axis=1)
    nan_count = np.sum(nan_positions)
    
    if nan_count > 0:
        print(f"⚠️ 发现 {nan_count} 个NaN位置，将被排除")
        valid_mask = ~nan_positions
        xyz = xyz[valid_mask]
        normals = normals[valid_mask]
        f_dc = f_dc[valid_mask]
        f_rest = f_rest[valid_mask]
        opacity = opacity[valid_mask]
        scaling = scaling[valid_mask]
        rotation = rotation[valid_mask]
        print(f"📊 有效高斯球数: {len(xyz):,}")
    
    # 计算场景中心
    scene_center = np.mean(xyz, axis=0)
    print(f"📍 场景中心: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
    
    # 创建输出目录
    output_dir = "alternative_layering"
    os.makedirs(output_dir, exist_ok=True)
    
    layering_strategies = []
    
    # 策略1: 按距离场景中心的距离分层
    print(f"\n🎯 策略1: 按距离场景中心分层")
    distances_to_center = np.linalg.norm(xyz - scene_center, axis=1)
    distance_percentiles = np.percentile(distances_to_center, [20, 40, 60, 80])
    
    for i in range(5):
        if i == 0:
            mask = distances_to_center <= distance_percentiles[0]
            layer_name = "center_core"
            layer_desc = "核心区域"
        elif i == 4:
            mask = distances_to_center > distance_percentiles[3]
            layer_name = "center_outer"
            layer_desc = "外围区域"
        else:
            mask = (distances_to_center > distance_percentiles[i-1]) & (distances_to_center <= distance_percentiles[i])
            layer_name = f"center_ring{i}"
            layer_desc = f"环带{i}"
        
        count = np.sum(mask)
        print(f"  层{i} ({layer_desc}): {count:,}球 距离范围: {distances_to_center[mask].min():.3f}~{distances_to_center[mask].max():.3f}")
        
        # 保存单层文件
        filename = f"center_layer_{i}_{layer_name}_{count}balls.ply"
        layer_path = os.path.join(output_dir, filename)
        save_gaussians_like_original(
            xyz[mask], normals[mask], f_dc[mask], f_rest[mask],
            opacity[mask], scaling[mask], rotation[mask], layer_path
        )
    
    layering_strategies.append({
        'name': 'center_distance',
        'description': '按距离场景中心分层',
        'thresholds': distance_percentiles.tolist(),
        'metric': 'distance_to_center'
    })
    
    # 策略2: 按透明度分层
    print(f"\n🎯 策略2: 按透明度分层")
    opacity_flat = opacity.flatten()
    opacity_percentiles = np.percentile(opacity_flat, [20, 40, 60, 80])
    
    for i in range(5):
        if i == 0:
            mask = opacity_flat <= opacity_percentiles[0]
            layer_name = "opacity_low"
            layer_desc = "低透明度"
        elif i == 4:
            mask = opacity_flat > opacity_percentiles[3]
            layer_name = "opacity_high"
            layer_desc = "高透明度"
        else:
            mask = (opacity_flat > opacity_percentiles[i-1]) & (opacity_flat <= opacity_percentiles[i])
            layer_name = f"opacity_mid{i}"
            layer_desc = f"中等透明度{i}"
        
        count = np.sum(mask)
        print(f"  层{i} ({layer_desc}): {count:,}球 透明度范围: {opacity_flat[mask].min():.3f}~{opacity_flat[mask].max():.3f}")
        
        # 保存单层文件
        filename = f"opacity_layer_{i}_{layer_name}_{count}balls.ply"
        layer_path = os.path.join(output_dir, filename)
        save_gaussians_like_original(
            xyz[mask], normals[mask], f_dc[mask], f_rest[mask],
            opacity[mask], scaling[mask], rotation[mask], layer_path
        )
    
    layering_strategies.append({
        'name': 'opacity',
        'description': '按透明度分层',
        'thresholds': opacity_percentiles.tolist(),
        'metric': 'opacity'
    })
    
    # 策略3: 按某个固定视角的Z深度分层 (000001.jpg视角)
    print(f"\n🎯 策略3: 按视角000001.jpg的Z深度分层")
    # 需要加载相机参数来计算Z深度
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    cameras_bin = os.path.join(colmap_path, 'cameras.bin')
    images_bin = os.path.join(colmap_path, 'images.bin')
    
    cam_intrinsics = read_intrinsics_binary(cameras_bin)
    cam_extrinsics = read_extrinsics_binary(images_bin)
    
    # 找到000001.jpg的相机参数
    ref_camera_info = None
    for img_id, img_info in cam_extrinsics.items():
        if img_info.name == "000001.jpg":
            ref_camera_info = img_info
            break
    
    if ref_camera_info:
        ref_R = np.transpose(qvec2rotmat(ref_camera_info.qvec))
        ref_T = np.array(ref_camera_info.tvec)
        
        # 转换到相机坐标系计算Z深度
        xyz_cam = np.dot(xyz - ref_T, ref_R.T)  # 变换到相机坐标系
        z_depths = xyz_cam[:, 2]  # Z深度
        
        z_percentiles = np.percentile(z_depths, [20, 40, 60, 80])
        
        for i in range(5):
            if i == 0:
                mask = z_depths <= z_percentiles[0]
                layer_name = "zdepth_near"
                layer_desc = "近景"
            elif i == 4:
                mask = z_depths > z_percentiles[3]
                layer_name = "zdepth_far"
                layer_desc = "远景"
            else:
                mask = (z_depths > z_percentiles[i-1]) & (z_depths <= z_percentiles[i])
                layer_name = f"zdepth_mid{i}"
                layer_desc = f"中景{i}"
            
            count = np.sum(mask)
            print(f"  层{i} ({layer_desc}): {count:,}球 Z深度范围: {z_depths[mask].min():.3f}~{z_depths[mask].max():.3f}")
            
            # 保存单层文件
            filename = f"zdepth_layer_{i}_{layer_name}_{count}balls.ply"
            layer_path = os.path.join(output_dir, filename)
            save_gaussians_like_original(
                xyz[mask], normals[mask], f_dc[mask], f_rest[mask],
                opacity[mask], scaling[mask], rotation[mask], layer_path
            )
        
        layering_strategies.append({
            'name': 'zdepth_000001',
            'description': '按000001.jpg视角Z深度分层',
            'thresholds': z_percentiles.tolist(),
            'metric': 'z_depth_from_000001',
            'reference_camera': {
                'name': '000001.jpg',
                'position': ref_T.tolist(),
                'rotation': ref_R.tolist()
            }
        })
    
    # 保存策略信息
    strategies_info = {
        'strategies': layering_strategies,
        'scene_center': scene_center.tolist(),
        'total_gaussians': len(xyz),
        'created_files': len(layering_strategies) * 5
    }
    
    info_path = os.path.join(output_dir, 'layering_strategies_info.json')
    with open(info_path, 'w') as f:
        json.dump(strategies_info, f, indent=2)
    
    print(f"\n✅ 创建了 {len(layering_strategies)} 种分层策略，共 {len(layering_strategies) * 5} 个层文件")
    print(f"📁 保存到目录: {output_dir}")
    print(f"📋 策略信息保存: {info_path}")
    
    return layering_strategies

def evaluate_strategy_across_views(strategy_name, test_cameras):
    """评估某个分层策略在多个视角下的表现"""
    print(f"\n🔍 评估策略 '{strategy_name}' 在多视角下的表现")
    
    # 设置渲染环境
    pipeline_parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(pipeline_parser)
    pipe_args = pipeline_parser.parse_args([])
    pipe = pipe_parser.extract(pipe_args)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 查找该策略的层文件
    if strategy_name == "center_distance":
        layer_files = glob.glob(f"alternative_layering/center_layer_*.ply")
    elif strategy_name == "opacity":
        layer_files = glob.glob(f"alternative_layering/opacity_layer_*.ply")
    elif strategy_name == "zdepth_000001":
        layer_files = glob.glob(f"alternative_layering/zdepth_layer_*.ply")
    else:
        layer_files = glob.glob(f"alternative_layering/{strategy_name}_layer_*.ply")
    layer_files.sort()
    
    if len(layer_files) == 0:
        print(f"❌ 未找到策略 {strategy_name} 的层文件")
        return None
    
    print(f"📁 找到 {len(layer_files)} 个层文件")
    
    # 检查SPARSE_ADAM_AVAILABLE
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except:
        SPARSE_ADAM_AVAILABLE = False
    
    results = {}
    
    # 对每个视角测试每个层
    for camera_info in test_cameras:
        camera = camera_info['camera']
        camera_name = camera_info['name']
        
        print(f"\n📷 测试视角: {camera_name}")
        
        camera_results = []
        
        for i, layer_file in enumerate(layer_files):
            layer_name = os.path.basename(layer_file).replace('.ply', '')
            
            try:
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # 加载高斯球
                gaussians = GaussianModel(3)
                gaussians.load_ply(layer_file, use_train_test_exp=False)
                
                gaussian_count = gaussians.get_xyz.shape[0]
                
                # 渲染
                render_result = render(camera, gaussians, pipe, background, 1., 
                                     SPARSE_ADAM_AVAILABLE, None, False)
                rendered_image = torch.clamp(render_result["render"], 0.0, 1.0)
                
                # GT图像
                gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
                
                # 计算指标
                psnr_val = psnr(rendered_image, gt_image).mean().item()
                l1_val = l1_loss(rendered_image, gt_image).mean().item()
                
                camera_results.append({
                    'layer_id': i,
                    'layer_name': layer_name,
                    'gaussian_count': gaussian_count,
                    'psnr': psnr_val,
                    'l1_loss': l1_val
                })
                
                print(f"  层{i}: {gaussian_count:,}球 PSNR: {psnr_val:.3f}dB")
                
                # 清理内存
                del gaussians, render_result, rendered_image, gt_image
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  层{i}: ❌ 渲染失败 - {str(e)[:50]}...")
                camera_results.append({
                    'layer_id': i,
                    'layer_name': layer_name,
                    'gaussian_count': 0,
                    'psnr': 0.0,
                    'l1_loss': 1.0,
                    'error': str(e)
                })
        
        results[camera_name] = camera_results
    
    return results

def analyze_view_consistency():
    """分析不同分层策略在多视角下的一致性"""
    print("🎯 多视角分层一致性分析")
    print("=" * 60)
    
    # 1. 创建多种分层策略
    print("步骤1: 创建多种分层策略")
    strategies = create_alternative_layering_strategies()
    
    # 2. 加载多个测试相机
    print("\n步骤2: 加载多个测试相机")
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    test_cameras = load_multi_cameras(colmap_path, images_path, 6.0)
    
    if len(test_cameras) == 0:
        print("❌ 未能加载任何测试相机")
        return
    
    print(f"✅ 成功加载 {len(test_cameras)} 个测试相机")
    
    # 3. 评估每种策略在多视角下的表现
    print("\n步骤3: 评估多视角表现")
    
    all_results = {}
    
    for strategy in strategies:
        strategy_name = strategy['name']
        print(f"\n{'='*20} 评估策略: {strategy['description']} {'='*20}")
        
        strategy_results = evaluate_strategy_across_views(strategy_name, test_cameras)
        if strategy_results:
            all_results[strategy_name] = strategy_results
    
    # 4. 分析结果
    print(f"\n📊 多视角一致性分析结果")
    print("=" * 50)
    
    if len(all_results) == 0:
        print("❌ 没有成功的评估结果")
        return
    
    # 创建输出目录
    output_dir = "multi_view_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析每种策略的视角一致性
    consistency_analysis = {}
    
    for strategy_name, strategy_results in all_results.items():
        print(f"\n🔍 策略: {strategy_name}")
        
        # 计算每层在不同视角的PSNR方差（一致性指标）
        layer_consistency = []
        
        for layer_id in range(5):  # 假设每个策略都有5层
            layer_psnrs = []
            layer_counts = []
            
            for camera_name, camera_results in strategy_results.items():
                if layer_id < len(camera_results):
                    layer_result = camera_results[layer_id]
                    if 'error' not in layer_result:
                        layer_psnrs.append(layer_result['psnr'])
                        layer_counts.append(layer_result['gaussian_count'])
            
            if len(layer_psnrs) > 0:
                psnr_mean = np.mean(layer_psnrs)
                psnr_std = np.std(layer_psnrs)
                consistency_score = psnr_mean / (psnr_std + 1e-6)  # 避免除零
                
                layer_consistency.append({
                    'layer_id': layer_id,
                    'psnr_mean': psnr_mean,
                    'psnr_std': psnr_std,
                    'consistency_score': consistency_score,
                    'gaussian_count': layer_counts[0] if layer_counts else 0,
                    'view_count': len(layer_psnrs)
                })
                
                print(f"  层{layer_id}: PSNR={psnr_mean:.3f}±{psnr_std:.3f}dB 一致性={consistency_score:.2f} ({layer_counts[0] if layer_counts else 0:,}球)")
        
        consistency_analysis[strategy_name] = layer_consistency
        
        # 计算策略的整体一致性评分
        if layer_consistency:
            overall_consistency = np.mean([layer['consistency_score'] for layer in layer_consistency])
            print(f"  📈 整体一致性评分: {overall_consistency:.3f}")
    
    # 保存详细结果
    analysis_results = {
        'strategies_evaluated': list(all_results.keys()),
        'test_cameras': [cam['name'] for cam in test_cameras],
        'detailed_results': all_results,
        'consistency_analysis': consistency_analysis,
        'summary': {
            'best_strategy_by_consistency': max(consistency_analysis.keys(), 
                                              key=lambda k: np.mean([layer['consistency_score'] for layer in consistency_analysis[k]]) if consistency_analysis[k] else 0),
            'analysis_notes': [
                "consistency_score = psnr_mean / psnr_std: 越高表示在不同视角下表现越一致",
                "center_distance策略: 基于距离场景中心的距离，视角无关",
                "opacity策略: 基于透明度，视角无关",  
                "zdepth_000001策略: 基于特定视角的深度，视角相关"
            ]
        }
    }
    
    results_file = os.path.join(output_dir, 'multi_view_consistency_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n✅ 多视角一致性分析完成")
    print(f"📁 详细结果保存: {results_file}")
    
    # 推荐最佳策略
    if consistency_analysis:
        best_strategy = analysis_results['summary']['best_strategy_by_consistency']
        print(f"🏆 推荐策略: {best_strategy}")
        print(f"   原因: 在多个视角下表现最一致")
    
    return analysis_results

def main():
    analyze_view_consistency()

if __name__ == "__main__":
    main() 