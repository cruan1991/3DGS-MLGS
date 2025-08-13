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

def create_fixed_size_layers():
    """创建修复版尺寸分层"""
    print("🔄 创建修复版尺寸分层")
    print("=" * 50)
    
    # 加载阈值信息
    manifest_path = "./size_based_layers/size_layers_manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    thresholds = manifest['thresholds']
    print(f"📊 使用阈值: {thresholds}")
    
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
    
    # 计算平均缩放（用于分层）
    avg_scale = np.mean(scaling, axis=1)
    
    # 创建输出目录
    output_dir = "size_layers_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义层级
    layer_info = [
        {"id": 0, "name": "超小球", "desc": "Ultra-small"},
        {"id": 1, "name": "小球", "desc": "Small"},
        {"id": 2, "name": "中球", "desc": "Medium"},
        {"id": 3, "name": "大球", "desc": "Large"},
        {"id": 4, "name": "超大球", "desc": "Ultra-large"}
    ]
    
    print(f"\n🔄 创建修复版渐进式累积文件...")
    progressive_files = []
    
    for end_layer in range(len(layer_info)):
        print(f"\n🎯 创建累积文件: 层0到层{end_layer}...")
        
        # 合并mask
        combined_mask = np.zeros(len(avg_scale), dtype=bool)
        
        for layer_id in range(end_layer + 1):
            # 重新计算该层的mask
            if layer_id == 0:
                mask = avg_scale <= thresholds[0]
            elif layer_id == len(layer_info) - 1:
                mask = avg_scale > thresholds[-1]
            else:
                mask = (avg_scale > thresholds[layer_id-1]) & (avg_scale <= thresholds[layer_id])
            
            combined_mask |= mask
        
        print(f"  累积高斯球数: {np.sum(combined_mask):,}")
        
        if np.sum(combined_mask) == 0:
            print(f"  ⚠️ 累积层为空，跳过")
            continue
        
        # 提取累积参数 (保持原始格式)
        prog_xyz = xyz[combined_mask]
        prog_normals = normals[combined_mask]
        prog_f_dc = f_dc[combined_mask]
        prog_f_rest = f_rest[combined_mask]
        prog_opacity = opacity[combined_mask]
        prog_scaling = scaling[combined_mask]
        prog_rotation = rotation[combined_mask]
        
        # 保存渐进式文件
        if end_layer == 0:
            filename = f"fixed_progressive_S0_{np.sum(combined_mask)}balls.ply"
        else:
            # 正确的累积命名：S0_S1_S2...S{end_layer}
            layer_names = '_'.join([f"S{i}" for i in range(end_layer + 1)])
            filename = f"fixed_progressive_{layer_names}_{np.sum(combined_mask)}balls.ply"
        
        prog_file_path = os.path.join(output_dir, filename)
        
        save_gaussians_like_original(prog_xyz, prog_normals, prog_f_dc, prog_f_rest,
                                    prog_opacity, prog_scaling, prog_rotation, prog_file_path)
        
        progressive_files.append(prog_file_path)
        print(f"  ✅ 保存: {filename}")
    
    print(f"\n📈 修复版渐进式文件创建完成: {len(progressive_files)} 个")
    return progressive_files

def render_ply_simple(ply_path, camera, pipe, background):
    """简单直接渲染PLY文件"""
    if not os.path.exists(ply_path):
        return None, {"error": "File not found"}
    
    try:
        print(f"    📸 直接渲染", end="")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 加载高斯球（完整版，不采样）
        gaussians = GaussianModel(3)
        gaussians.load_ply(ply_path, use_train_test_exp=False)
        
        gaussian_count = gaussians.get_xyz.shape[0]
        print(f" - {gaussian_count:,}球(完整)")
        
        # 检查SPARSE_ADAM_AVAILABLE
        try:
            from diff_gaussian_rasterization import SparseGaussianAdam
            SPARSE_ADAM_AVAILABLE = True
        except:
            SPARSE_ADAM_AVAILABLE = False
        
        # 直接渲染
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
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"    ❌ GPU内存不足")
        torch.cuda.empty_cache()
        return None, {"error": "CUDA OOM"}
    except Exception as e:
        print(f"    ❌ 渲染失败: {str(e)}")
        torch.cuda.empty_cache()
        return None, {"error": str(e)}

def evaluate_fixed_progressive_layers(layers_dir):
    """评估修复版渐进式分层"""
    print("\n📈 评估修复版尺寸分层渐进式效果")
    print("=" * 60)
    print("🔧 主要特点:")
    print("  - 使用修复的数据格式，保证与原始模型PSNR一致")
    print("  - 完整保留所有高斯球，不进行采样")
    print("  - 验证PSNR单调递增特性")
    
    # 创建输出目录
    output_dir = 'fixed_progressive_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置渲染环境
    pipeline_parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(pipeline_parser)
    pipe_args = pipeline_parser.parse_args([])
    pipe = pipe_parser.extract(pipe_args)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 加载相机
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    camera = load_test_camera(colmap_path, images_path, "000001.jpg", 4.0)  # 4x缩放平衡质量和性能
    
    if camera is None:
        return
    
    print(f"\n✅ 加载测试相机: 000001.jpg (分辨率: {camera.image_width}x{camera.image_height})")
    
    # 查找渐进式PLY文件
    progressive_files = sorted(glob.glob(os.path.join(layers_dir, "fixed_progressive_*.ply")))
    
    print(f"📈 找到修复版渐进文件: {len(progressive_files)}个")
    
    # 渲染渐进式文件
    progressive_results = []
    layer_names = ['S0', 'S0+S1', 'S0+S1+S2', 'S0+S1+S2+S3', 'S0+S1+S2+S3+S4']
    layer_descriptions = [
        '超小球',
        '超小球+小球', 
        '超小球+小球+中球',
        '超小球+小球+中球+大球',
        '超小球+小球+中球+大球+超大球'
    ]
    
    print(f"\n🎯 开始修复版渐进式渲染...")
    
    for i, prog_file in enumerate(progressive_files):
        layer_name = layer_names[i] if i < len(layer_names) else f"Stage{i}"
        layer_desc = layer_descriptions[i] if i < len(layer_descriptions) else f"阶段{i}"
        
        print(f"\n🎨 渲染阶段{i} ({layer_name}): {layer_desc}")
        print(f"   文件: {os.path.basename(prog_file)}")
        
        file_size_mb = os.path.getsize(prog_file) / (1024 * 1024)
        print(f"   📐 文件大小: {file_size_mb:.1f}MB")
        
        images, metrics = render_ply_simple(prog_file, camera, pipe, background)
        
        progressive_results.append({
            'stage': i,
            'layer_name': layer_name,
            'layer_description': layer_desc,
            'images': images,
            'metrics': metrics,
            'file': os.path.basename(prog_file),
            'file_size_mb': file_size_mb
        })
        
        if images is not None:
            print(f"   ✅ PSNR: {metrics['psnr']:.3f}dB, 高斯球数: {metrics['gaussian_count']:,}")
        else:
            print(f"   ❌ 渲染失败: {metrics.get('error', 'Unknown')}")
    
    # 分析PSNR进化
    print(f"\n📊 分析PSNR进化...")
    
    successful_results = [r for r in progressive_results if r['images'] is not None]
    
    if len(successful_results) == 0:
        print("❌ 没有成功的渲染结果")
        return
    
    # 计算贡献分析
    contribution_analysis = []
    for i, result in enumerate(successful_results):
        current_psnr = result['metrics']['psnr']
        
        if i == 0:
            contribution = current_psnr
        else:
            prev_psnr = successful_results[i-1]['metrics']['psnr']
            contribution = current_psnr - prev_psnr
        
        contribution_analysis.append({
            'stage': result['stage'],
            'layer_name': result['layer_name'],
            'layer_description': result['layer_description'],
            'cumulative_psnr': current_psnr,
            'psnr_contribution': contribution,
            'gaussian_count': result['metrics']['gaussian_count'],
            'file_size_mb': result['file_size_mb']
        })
        
        print(f"  阶段{result['stage']} ({result['layer_name']}): {current_psnr:.3f}dB (+{contribution:.3f}), {result['metrics']['gaussian_count']:,}球")
    
    # 检查单调递增特性
    negative_contributions = [ca for ca in contribution_analysis[1:] if ca['psnr_contribution'] < -0.001]
    if negative_contributions:
        print(f"\n⚠️  发现 {len(negative_contributions)} 个负贡献阶段:")
        for ca in negative_contributions:
            print(f"     {ca['layer_name']}: {ca['psnr_contribution']:.3f}dB")
        print("\n🤔 这说明按尺寸分层可能不是最优策略")
        print("   可能的原因:")
        print("   1. 小尺寸高斯球主要提供细节，而大尺寸球提供主要结构")
        print("   2. 单纯按尺寸分层破坏了空间相关性")
        print("   3. 需要考虑透明度、位置等其他因素")
    else:
        print(f"\n✅ 很好！所有阶段PSNR严格单调递增")
        print("   说明按尺寸分层是合理的策略")
    
    # 创建可视化对比
    print(f"\n🎨 生成修复版对比图...")
    
    # 创建2x3的布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fixed Progressive Size Layers - Correct Data Format', fontsize=16, fontweight='bold')
    
    # 绘制5个阶段的渲染结果
    for i in range(min(5, len(successful_results))):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        result = successful_results[i]
        ax.imshow(result['images'][0])  # 显示渲染图像
        
        title = f"{result['layer_name']}\n{result['metrics']['gaussian_count']:,} gaussians"
        title += f"\nPSNR: {result['metrics']['psnr']:.3f}dB"
        
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    # 最后一个子图显示PSNR进化曲线
    ax = axes[1, 2]
    if len(contribution_analysis) > 1:
        stages = [ca['stage'] for ca in contribution_analysis]
        psnr_values = [ca['cumulative_psnr'] for ca in contribution_analysis]
        contributions = [ca['psnr_contribution'] for ca in contribution_analysis]
        
        # 主曲线
        ax.plot(stages, psnr_values, 'bo-', linewidth=3, markersize=8, label='Cumulative PSNR')
        
        # 贡献条形图（右轴）
        ax2 = ax.twinx()
        colors = ['green' if c >= 0 else 'red' for c in contributions]
        ax2.bar(stages, contributions, alpha=0.3, color=colors, label='Increment')
        
        ax.set_xlabel('Progressive Stage')
        ax.set_ylabel('Cumulative PSNR (dB)', color='blue')
        ax2.set_ylabel('PSNR Increment (dB)', color='gray')
        ax.set_title('PSNR Evolution\n(Fixed Data Format)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注数值
        for stage, psnr_val in zip(stages, psnr_values):
            ax.annotate(f'{psnr_val:.2f}', (stage, psnr_val),
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图像
    comparison_file = os.path.join(output_dir, 'fixed_progressive_comparison.png')
    plt.savefig(comparison_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 修复版对比图保存: {comparison_file}")
    
    # 保存详细结果
    evaluation_results = {
        'test_camera': '000001.jpg',
        'resolution_scale': 4.0,
        'rendering_method': 'fixed_format_direct',
        'data_format': 'original_save_ply_compatible',
        'progressive_results': [
            {
                'stage': r['stage'],
                'layer_name': r['layer_name'],
                'layer_description': r['layer_description'],
                'file': r['file'],
                'file_size_mb': r['file_size_mb'],
                'success': r['images'] is not None,
                'psnr': r['metrics'].get('psnr', 0) if r['images'] is not None else None,
                'l1_loss': r['metrics'].get('l1_loss', 0) if r['images'] is not None else None,
                'gaussian_count': r['metrics'].get('gaussian_count', 0) if r['images'] is not None else None,
                'error': r['metrics'].get('error') if r['images'] is None else None
            }
            for r in progressive_results
        ],
        'contribution_analysis': contribution_analysis,
        'quality_check': {
            'has_negative_contributions': len(negative_contributions) > 0,
            'negative_contribution_stages': [ca['layer_name'] for ca in negative_contributions],
            'strictly_monotonic': all(ca['psnr_contribution'] >= -0.001 for ca in contribution_analysis[1:]),
            'max_negative_contribution': min([ca['psnr_contribution'] for ca in contribution_analysis[1:]], default=0),
            'total_psnr_gain': successful_results[-1]['metrics']['psnr'] - successful_results[0]['metrics']['psnr'] if len(successful_results) > 1 else 0
        }
    }
    
    results_file = os.path.join(output_dir, 'fixed_progressive_evaluation.json')
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"✅ 详细结果保存: {results_file}")
    
    # 打印总结
    print(f"\n📊 修复版评估总结:")
    print(f"  总阶段数: {len(progressive_results)}")
    print(f"  成功渲染: {len(successful_results)}/{len(progressive_results)}")
    if successful_results:
        print(f"  最终PSNR: {successful_results[-1]['metrics']['psnr']:.3f}dB")
        if len(successful_results) > 1:
            total_gain = successful_results[-1]['metrics']['psnr'] - successful_results[0]['metrics']['psnr']
            print(f"  总体提升: {total_gain:.3f}dB")
            
            if contribution_analysis:
                best_contrib = max(contribution_analysis, key=lambda x: x['psnr_contribution'])
                print(f"  最大贡献阶段: {best_contrib['layer_name']} (+{best_contrib['psnr_contribution']:.3f}dB)")
                
                print(f"  质量检查: {'✅ 严格单调递增' if evaluation_results['quality_check']['strictly_monotonic'] else '❌ 存在负增长'}")
                
                # 与我们之前验证的28dB对比
                final_psnr = successful_results[-1]['metrics']['psnr']
                if final_psnr > 20:  # 在4x scale下的合理PSNR
                    print(f"  🎉 PSNR水平合理，数据修复成功！")
                else:
                    print(f"  ⚠️ PSNR仍然偏低，可能还有其他问题")
    
    return evaluation_results

def main():
    print("🎉 修复版尺寸分层完整流程")
    print("=" * 50)
    
    # 1. 创建修复版分层文件
    print("步骤1: 创建修复版分层文件")
    progressive_files = create_fixed_size_layers()
    
    # 2. 评估渐进式效果
    print("\n步骤2: 评估渐进式效果")
    layers_dir = "size_layers_fixed"
    results = evaluate_fixed_progressive_layers(layers_dir)
    
    if results:
        print(f"\n🎉 修复版完整流程完成!")
        print(f"📁 分层文件目录: {layers_dir}/")
        print(f"📁 评估结果目录: fixed_progressive_evaluation/")

if __name__ == "__main__":
    main() 