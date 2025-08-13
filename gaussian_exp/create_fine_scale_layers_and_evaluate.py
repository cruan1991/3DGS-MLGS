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

def load_test_camera(colmap_path, images_path, camera_name="000001.jpg", resolution_scale=4.0):
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

def create_fine_scale_layers():
    """创建基于真实尺寸的精细分层"""
    print("🔄 创建基于真实尺寸的精细分层")
    print("=" * 60)
    
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
    scaling = gaussians._scaling.detach().cpu().numpy()  # log space
    rotation = gaussians._rotation.detach().cpu().numpy()
    
    # 处理NaN值
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
    
    # 转换到实际尺寸
    actual_scales = np.exp(scaling)  # 从log space转换到real space
    avg_actual_scale = np.mean(actual_scales, axis=1)
    max_actual_scale = np.max(actual_scales, axis=1)
    
    print(f"📊 数据统计:")
    print(f"  有效高斯球数: {len(avg_actual_scale):,}")
    print(f"  平均实际尺寸范围: {avg_actual_scale.min():.6f} ~ {avg_actual_scale.max():.6f}")
    print(f"  最大实际尺寸范围: {max_actual_scale.min():.6f} ~ {max_actual_scale.max():.6f}")
    
    # 创建输出目录
    output_dir = "fine_scale_layers"
    os.makedirs(output_dir, exist_ok=True)
    
    # 精细分层方案：15层 (至少3组细分)
    print(f"\n🎯 精细分层方案: 15层")
    
    # 使用平均实际尺寸进行分层
    layer_percentiles = np.linspace(6.67, 93.33, 14)  # 15层的14个分界点
    thresholds = np.percentile(avg_actual_scale, layer_percentiles)
    
    layer_info = []
    single_layer_files = []
    
    print(f"📋 单层文件生成:")
    for i in range(15):
        if i == 0:
            mask = avg_actual_scale <= thresholds[0]
            layer_name = f"nano_{i:02d}"
            layer_desc = "纳米级"
        elif i == 14:
            mask = avg_actual_scale > thresholds[13]
            layer_name = f"giant_{i:02d}"
            layer_desc = "巨型"
        else:
            mask = (avg_actual_scale > thresholds[i-1]) & (avg_actual_scale <= thresholds[i])
            if i < 5:
                layer_name = f"micro_{i:02d}"
                layer_desc = f"微型{i}"
            elif i < 10:
                layer_name = f"small_{i:02d}"
                layer_desc = f"小型{i-4}"
            else:
                layer_name = f"medium_{i:02d}"
                layer_desc = f"中型{i-9}"
        
        count = np.sum(mask)
        if count > 0:
            ratio = count / len(avg_actual_scale) * 100
            range_str = f"{avg_actual_scale[mask].min():.6f}~{avg_actual_scale[mask].max():.6f}"
            
            # 保存单层文件
            filename = f"layer_{i:02d}_{layer_name}_{count}balls.ply"
            layer_path = os.path.join(output_dir, filename)
            
            save_gaussians_like_original(
                xyz[mask], normals[mask], f_dc[mask], f_rest[mask],
                opacity[mask], scaling[mask], rotation[mask], layer_path
            )
            
            layer_info.append({
                'id': i,
                'name': layer_name,
                'description': layer_desc,
                'count': int(count),
                'ratio': float(ratio),
                'scale_range': [float(avg_actual_scale[mask].min()), float(avg_actual_scale[mask].max())],
                'file': filename
            })
            
            single_layer_files.append(layer_path)
            print(f"  层{i:2d} ({layer_desc}): {count:,}球 ({ratio:.1f}%) 范围: {range_str} -> {filename}")
    
    print(f"\n🔄 渐进式累积文件生成:")
    progressive_files = []
    
    for end_layer in range(len(layer_info)):
        # 合并mask
        combined_mask = np.zeros(len(avg_actual_scale), dtype=bool)
        
        for layer_id in range(end_layer + 1):
            layer = layer_info[layer_id]
            if layer_id == 0:
                mask = avg_actual_scale <= thresholds[0]
            elif layer_id == len(layer_info) - 1:
                mask = avg_actual_scale > thresholds[13]
            else:
                mask = (avg_actual_scale > thresholds[layer_id-1]) & (avg_actual_scale <= thresholds[layer_id])
            
            combined_mask |= mask
        
        cumulative_count = np.sum(combined_mask)
        print(f"  累积0-{end_layer}: {cumulative_count:,}球", end="")
        
        if cumulative_count == 0:
            print(f" (空层，跳过)")
            continue
        
        # 保存渐进式文件
        if end_layer < 5:
            group_name = "micro"
        elif end_layer < 10:
            group_name = "small"
        else:
            group_name = "medium_large"
        
        prog_filename = f"progressive_{group_name}_L0_L{end_layer:02d}_{cumulative_count}balls.ply"
        prog_file_path = os.path.join(output_dir, prog_filename)
        
        save_gaussians_like_original(
            xyz[combined_mask], normals[combined_mask], f_dc[combined_mask], f_rest[combined_mask],
            opacity[combined_mask], scaling[combined_mask], rotation[combined_mask], prog_file_path
        )
        
        progressive_files.append(prog_file_path)
        print(f" -> {prog_filename}")
    
    # 保存分层信息
    layering_manifest = {
        'method': 'fine_actual_scale_based',
        'total_layers': len(layer_info),
        'total_gaussians': len(avg_actual_scale),
        'thresholds': [float(t) for t in thresholds],
        'layer_info': layer_info,
        'single_layer_files': [os.path.basename(f) for f in single_layer_files],
        'progressive_files': [os.path.basename(f) for f in progressive_files],
        'groups': {
            'micro': {'layers': list(range(5)), 'description': '微型球 (层0-4)'},
            'small': {'layers': list(range(5, 10)), 'description': '小型球 (层5-9)'},
            'medium_large': {'layers': list(range(10, 15)), 'description': '中大型球 (层10-14)'}
        }
    }
    
    manifest_path = os.path.join(output_dir, 'fine_layers_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(layering_manifest, f, indent=2)
    
    print(f"\n✅ 精细分层完成:")
    print(f"  📁 输出目录: {output_dir}/")
    print(f"  📋 分层清单: {manifest_path}")
    print(f"  🎯 单层文件: {len(single_layer_files)} 个")
    print(f"  📈 渐进文件: {len(progressive_files)} 个")
    print(f"  📊 分组: 微型(0-4), 小型(5-9), 中大型(10-14)")
    
    return layering_manifest

def evaluate_fine_layers():
    """评估精细分层的渲染效果和PSNR"""
    print("\n📈 评估精细分层渲染效果和PSNR")
    print("=" * 60)
    
    # 设置渲染环境
    pipeline_parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(pipeline_parser)
    pipe_args = pipeline_parser.parse_args([])
    pipe = pipe_parser.extract(pipe_args)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 加载相机
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    camera = load_test_camera(colmap_path, images_path, "000001.jpg", 5.0)  # 5x缩放平衡质量和速度
    
    if camera is None:
        return
    
    print(f"✅ 加载测试相机: 000001.jpg (分辨率: {camera.image_width}x{camera.image_height})")
    
    # 加载分层清单
    manifest_path = "fine_scale_layers/fine_layers_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"❌ 分层清单不存在: {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # 创建输出目录
    output_dir = 'fine_layers_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查SPARSE_ADAM_AVAILABLE
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except:
        SPARSE_ADAM_AVAILABLE = False
    
    def render_ply_file(ply_path):
        """渲染PLY文件并计算PSNR"""
        if not os.path.exists(ply_path):
            return None, {"error": "File not found"}
        
        try:
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 加载高斯球
            gaussians = GaussianModel(3)
            gaussians.load_ply(ply_path, use_train_test_exp=False)
            
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
            
            # 转换为numpy用于可视化
            rendered_np = rendered_image.detach().cpu().numpy().transpose(1, 2, 0)
            gt_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
            
            # 清理内存
            del gaussians, render_result, rendered_image, gt_image
            torch.cuda.empty_cache()
            
            return (rendered_np, gt_np), {
                "psnr": psnr_val,
                "l1_loss": l1_val,
                "gaussian_count": gaussian_count
            }
            
        except Exception as e:
            print(f"    ❌ 渲染失败: {str(e)[:50]}...")
            torch.cuda.empty_cache()
            return None, {"error": str(e)}
    
    # 1. 评估渐进式文件
    print(f"\n🎯 评估渐进式累积效果:")
    
    progressive_files = [f"fine_scale_layers/{f}" for f in manifest['progressive_files']]
    progressive_results = []
    
    for i, prog_file in enumerate(progressive_files):
        filename = os.path.basename(prog_file)
        print(f"\n🎨 渲染渐进文件 {i+1}/{len(progressive_files)}: {filename}")
        
        file_size_mb = os.path.getsize(prog_file) / (1024 * 1024)
        print(f"   📐 文件大小: {file_size_mb:.1f}MB")
        
        images, metrics = render_ply_file(prog_file)
        
        if images is not None:
            print(f"   ✅ PSNR: {metrics['psnr']:.3f}dB, 高斯球数: {metrics['gaussian_count']:,}")
            
            progressive_results.append({
                'stage': i,
                'file': filename,
                'file_size_mb': file_size_mb,
                'psnr': metrics['psnr'],
                'l1_loss': metrics['l1_loss'],
                'gaussian_count': metrics['gaussian_count'],
                'images': images
            })
        else:
            print(f"   ❌ 渲染失败: {metrics.get('error', 'Unknown')}")
    
    # 2. 评估关键单层文件（每5层采样一次）
    print(f"\n🎯 评估关键单层效果:")
    
    single_layer_results = []
    key_layers = [0, 4, 9, 14]  # 每组的代表层
    
    for layer_id in key_layers:
        if layer_id < len(manifest['layer_info']):
            layer = manifest['layer_info'][layer_id]
            layer_file = f"fine_scale_layers/{layer['file']}"
            
            print(f"\n🎨 渲染单层 {layer_id}: {layer['description']}")
            
            images, metrics = render_ply_file(layer_file)
            
            if images is not None:
                print(f"   ✅ PSNR: {metrics['psnr']:.3f}dB, 高斯球数: {metrics['gaussian_count']:,}")
                
                single_layer_results.append({
                    'layer_id': layer_id,
                    'layer_name': layer['name'],
                    'layer_description': layer['description'],
                    'file': layer['file'],
                    'psnr': metrics['psnr'],
                    'l1_loss': metrics['l1_loss'],
                    'gaussian_count': metrics['gaussian_count'],
                    'images': images
                })
            else:
                print(f"   ❌ 渲染失败: {metrics.get('error', 'Unknown')}")
    
    # 3. 生成对比可视化
    print(f"\n🎨 生成对比可视化...")
    
    # 渐进式对比图
    if progressive_results:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Fine-Scale Progressive Layering Evaluation', fontsize=16, fontweight='bold')
        
        # 显示前7个渐进阶段
        for i in range(min(7, len(progressive_results))):
            row = i // 4
            col = i % 4
            
            if row < 2 and col < 4:
                ax = axes[row, col]
                result = progressive_results[i]
                
                ax.imshow(result['images'][0])  # 显示渲染图像
                
                title = f"L0-L{i:02d}\n{result['gaussian_count']:,} balls"
                title += f"\nPSNR: {result['psnr']:.2f}dB"
                
                ax.set_title(title, fontsize=10)
                ax.axis('off')
        
        # 最后一个子图显示PSNR曲线
        ax = axes[1, 3]
        stages = [r['stage'] for r in progressive_results]
        psnr_values = [r['psnr'] for r in progressive_results]
        
        ax.plot(stages, psnr_values, 'bo-', linewidth=2, markersize=6)
        ax.set_xlabel('Progressive Stage')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('PSNR Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注数值
        for stage, psnr_val in zip(stages[::2], psnr_values[::2]):  # 每隔一个标注
            ax.annotate(f'{psnr_val:.1f}', (stage, psnr_val),
                       textcoords="offset points", xytext=(0,5), 
                       ha='center', fontsize=8)
        
        plt.tight_layout()
        
        # 保存图像
        progressive_comparison_file = os.path.join(output_dir, 'fine_progressive_comparison.png')
        plt.savefig(progressive_comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 渐进对比图保存: {progressive_comparison_file}")
    
    # 单层对比图
    if single_layer_results:
        fig, axes = plt.subplots(1, len(single_layer_results), figsize=(5*len(single_layer_results), 5))
        if len(single_layer_results) == 1:
            axes = [axes]
        
        fig.suptitle('Key Single Layers Comparison', fontsize=16, fontweight='bold')
        
        for i, result in enumerate(single_layer_results):
            ax = axes[i]
            ax.imshow(result['images'][0])
            
            title = f"Layer {result['layer_id']}\n{result['layer_description']}"
            title += f"\n{result['gaussian_count']:,} balls"
            title += f"\nPSNR: {result['psnr']:.2f}dB"
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        single_comparison_file = os.path.join(output_dir, 'fine_single_layers_comparison.png')
        plt.savefig(single_comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 单层对比图保存: {single_comparison_file}")
    
    # 4. 保存详细结果
    evaluation_results = {
        'test_camera': '000001.jpg',
        'resolution_scale': 5.0,
        'total_layers': manifest['total_layers'],
        'total_gaussians': manifest['total_gaussians'],
        'progressive_evaluation': [
            {
                'stage': r['stage'],
                'file': r['file'],
                'psnr': r['psnr'],
                'l1_loss': r['l1_loss'],
                'gaussian_count': r['gaussian_count'],
                'file_size_mb': r['file_size_mb']
            }
            for r in progressive_results
        ],
        'single_layer_evaluation': [
            {
                'layer_id': r['layer_id'],
                'layer_name': r['layer_name'],
                'layer_description': r['layer_description'],
                'psnr': r['psnr'],
                'l1_loss': r['l1_loss'],
                'gaussian_count': r['gaussian_count']
            }
            for r in single_layer_results
        ],
        'analysis': {
            'psnr_range': [min(r['psnr'] for r in progressive_results), max(r['psnr'] for r in progressive_results)] if progressive_results else None,
            'best_single_layer': max(single_layer_results, key=lambda x: x['psnr']) if single_layer_results else None,
            'progressive_improvement': progressive_results[-1]['psnr'] - progressive_results[0]['psnr'] if len(progressive_results) > 1 else 0
        }
    }
    
    results_file = os.path.join(output_dir, 'fine_layers_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"✅ 详细结果保存: {results_file}")
    
    # 5. 打印总结
    print(f"\n📊 精细分层评估总结:")
    print(f"  总层数: {manifest['total_layers']}")
    print(f"  渐进式评估: {len(progressive_results)}/{len(progressive_files)} 成功")
    print(f"  单层评估: {len(single_layer_results)}/{len(key_layers)} 成功")
    
    if progressive_results:
        final_psnr = progressive_results[-1]['psnr']
        initial_psnr = progressive_results[0]['psnr']
        improvement = final_psnr - initial_psnr
        print(f"  最终PSNR: {final_psnr:.3f}dB")
        print(f"  渐进提升: {improvement:.3f}dB")
        
        if single_layer_results:
            best_single = max(single_layer_results, key=lambda x: x['psnr'])
            print(f"  最佳单层: Layer {best_single['layer_id']} ({best_single['layer_description']}) - {best_single['psnr']:.3f}dB")
    
    return evaluation_results

def main():
    print("🎉 精细尺寸分层完整流程")
    print("=" * 60)
    
    # 1. 创建精细分层文件
    print("步骤1: 创建精细分层文件 (15层)")
    manifest = create_fine_scale_layers()
    
    # 2. 评估分层效果
    print("\n步骤2: 评估分层渲染效果和PSNR")
    results = evaluate_fine_layers()
    
    if results:
        print(f"\n🎉 精细分层完整流程完成!")
        print(f"📁 分层文件目录: fine_scale_layers/")
        print(f"📁 评估结果目录: fine_layers_evaluation/")
        print(f"🎯 分组策略: 微型(0-4), 小型(5-9), 中大型(10-14)")

if __name__ == "__main__":
    main() 