import os
import sys
import torch
import numpy as np
import argparse
import json
import glob
from PIL import Image
import matplotlib.pyplot as plt

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

def load_test_camera(colmap_path, images_path, camera_name="000001.jpg", resolution_scale=6.0):
    """加载测试相机，使用更大的缩放以节省内存"""
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

def intelligent_sampling(gaussians, target_count):
    """智能采样：优先保留重要的高斯球"""
    current_count = gaussians.get_xyz.shape[0]
    
    if current_count <= target_count:
        return gaussians, False
    
    print(f" -> 智能采样到 {target_count:,} 个")
    
    # 获取高斯球参数
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()
    scaling = gaussians.get_scaling.detach().cpu().numpy()
    
    # 计算重要性分数
    # 1. 透明度权重 (40%)
    opacity_score = opacity
    
    # 2. 距离相机权重 (30%) - 距离越近越重要
    camera_pos = np.array([0, 0, 0])  # 假设相机在原点
    distances = np.linalg.norm(xyz - camera_pos, axis=1)
    distance_score = 1.0 / (1.0 + distances)  # 距离越近分数越高
    
    # 3. 尺寸权重 (30%) - 适中的尺寸最重要
    avg_scale = np.mean(scaling, axis=1)
    # 使用钟形曲线，中等尺寸得分最高
    optimal_scale = 0.05  # 经验值，基于之前的分析
    size_score = np.exp(-((avg_scale - optimal_scale) / optimal_scale) ** 2)
    
    # 综合重要性分数
    importance_scores = (
        0.4 * opacity_score +
        0.3 * distance_score +
        0.3 * size_score
    )
    
    # 选择最重要的高斯球
    top_indices = np.argsort(importance_scores)[-target_count:]
    indices = torch.from_numpy(top_indices).long()
    
    # 更新高斯球参数
    gaussians._xyz = gaussians._xyz[indices]
    gaussians._features_dc = gaussians._features_dc[indices]
    gaussians._features_rest = gaussians._features_rest[indices]
    gaussians._scaling = gaussians._scaling[indices]
    gaussians._rotation = gaussians._rotation[indices]
    gaussians._opacity = gaussians._opacity[indices]
    
    return gaussians, True

def render_ply_with_smart_sampling(ply_path, camera, pipe, background, max_gaussians=500000):
    """使用智能采样的PLY渲染"""
    if not os.path.exists(ply_path):
        return None, {"error": "File not found"}
    
    try:
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 加载高斯球
        gaussians = GaussianModel(3)
        gaussians.load_ply(ply_path, use_train_test_exp=False)
        
        original_count = gaussians.get_xyz.shape[0]
        print(f"    加载了 {original_count:,} 个高斯球", end="")
        
        # 智能采样
        gaussians, was_sampled = intelligent_sampling(gaussians, max_gaussians)
        final_count = gaussians.get_xyz.shape[0]
        
        if not was_sampled:
            print("")
        
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
        
        # 转换为numpy用于可视化
        rendered_np = rendered_image.detach().cpu().numpy().transpose(1, 2, 0)
        gt_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
        
        # 清理内存
        del gaussians, render_result, rendered_image, gt_image
        torch.cuda.empty_cache()
        
        return (rendered_np, gt_np), {
            "psnr": psnr_val,
            "l1_loss": l1_val,
            "gaussian_count": final_count,
            "original_count": original_count,
            "was_sampled": was_sampled,
            "sampling_ratio": final_count / original_count if original_count > 0 else 1.0
        }
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"    ⚠️ GPU内存不足: {str(e)[:100]}...")
        torch.cuda.empty_cache()
        return None, {"error": "CUDA OOM"}
    except Exception as e:
        print(f"    ❌ 渲染失败: {str(e)}")
        torch.cuda.empty_cache()
        return None, {"error": str(e)}

def evaluate_size_progressive_fixed(layers_dir, output_dir='size_progressive_fixed'):
    """修复版渐进式评估"""
    print("📈 修复版尺寸分层渐进式评估")
    print("=" * 60)
    print("🔧 主要改进:")
    print("  - 使用智能采样替代随机采样")
    print("  - 基于透明度+距离+尺寸的重要性排序")
    print("  - 更大的分辨率缩放(6x)进一步节省内存")
    print("  - 降低最大高斯球数量(50万)")
    
    # 创建输出目录
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
    camera = load_test_camera(colmap_path, images_path, "000001.jpg", 6.0)
    
    if camera is None:
        return
    
    print(f"\n✅ 加载测试相机: 000001.jpg (分辨率: {camera.image_width}x{camera.image_height})")
    
    # 查找渐进式PLY文件
    progressive_files = sorted(glob.glob(os.path.join(layers_dir, "size_progressive_*.ply")))
    
    print(f"📈 找到渐进文件: {len(progressive_files)}个")
    
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
    
    print(f"\n🎯 开始智能采样渐进式渲染...")
    
    for i, prog_file in enumerate(progressive_files):
        layer_name = layer_names[i] if i < len(layer_names) else f"Stage{i}"
        layer_desc = layer_descriptions[i] if i < len(layer_descriptions) else f"阶段{i}"
        
        print(f"\n🎨 渲染阶段{i} ({layer_name}): {layer_desc}")
        print(f"   文件: {os.path.basename(prog_file)}")
        
        images, metrics = render_ply_with_smart_sampling(prog_file, camera, pipe, background)
        
        progressive_results.append({
            'stage': i,
            'layer_name': layer_name,
            'layer_description': layer_desc,
            'images': images,
            'metrics': metrics,
            'file': os.path.basename(prog_file)
        })
        
        if images is not None:
            sampled_note = f" (智能采样 {metrics['sampling_ratio']:.1%})" if metrics.get('was_sampled') else ""
            print(f"   ✅ PSNR: {metrics['psnr']:.2f}dB, 使用球数: {metrics['gaussian_count']:,}/{metrics['original_count']:,}{sampled_note}")
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
            'original_count': result['metrics']['original_count'],
            'sampling_ratio': result['metrics']['sampling_ratio']
        })
        
        print(f"  阶段{result['stage']} ({result['layer_name']}): {current_psnr:.2f}dB (+{contribution:.2f}), {result['metrics']['gaussian_count']:,}球")
    
    # 检查是否还有负增长
    negative_contributions = [ca for ca in contribution_analysis if ca['psnr_contribution'] < -0.1]
    if negative_contributions:
        print(f"\n⚠️  仍然发现 {len(negative_contributions)} 个负贡献阶段:")
        for ca in negative_contributions:
            print(f"     {ca['layer_name']}: {ca['psnr_contribution']:.2f}dB")
        print("   这可能是由于:")
        print("   1. 采样策略仍需要进一步优化")
        print("   2. 某些尺寸层之间存在干扰效应")
        print("   3. 分辨率过低影响了评估精度")
    else:
        print(f"\n✅ 修复成功！所有阶段PSNR均为正增长")
    
    # 创建可视化对比
    print(f"\n🎨 生成修复版对比图...")
    
    # 创建2x3的布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('修复版尺寸分层渐进式累积效果 - 智能采样', fontsize=16, fontweight='bold')
    
    # 绘制5个阶段的渲染结果
    for i in range(min(5, len(successful_results))):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        result = successful_results[i]
        ax.imshow(result['images'][0])  # 显示渲染图像
        
        title = f"{result['layer_name']}\n{result['metrics']['gaussian_count']:,}球"
        if result['metrics'].get('was_sampled'):
            title += f" ({result['metrics']['sampling_ratio']:.0%}采样)"
        title += f"\nPSNR: {result['metrics']['psnr']:.2f}dB"
        
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    # 最后一个子图显示PSNR进化曲线
    ax = axes[1, 2]
    if len(contribution_analysis) > 1:
        stages = [ca['stage'] for ca in contribution_analysis]
        psnr_values = [ca['cumulative_psnr'] for ca in contribution_analysis]
        contributions = [ca['psnr_contribution'] for ca in contribution_analysis]
        
        # 主曲线
        ax.plot(stages, psnr_values, 'bo-', linewidth=3, markersize=8, label='累积PSNR')
        
        # 贡献条形图（右轴）
        ax2 = ax.twinx()
        colors = ['green' if c >= 0 else 'red' for c in contributions]
        ax2.bar(stages, contributions, alpha=0.3, color=colors, label='增量')
        
        ax.set_xlabel('累积阶段')
        ax.set_ylabel('累积PSNR (dB)', color='blue')
        ax2.set_ylabel('PSNR增量 (dB)', color='gray')
        ax.set_title('PSNR进化曲线', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注数值
        for stage, psnr_val in zip(stages, psnr_values):
            ax.annotate(f'{psnr_val:.1f}', (stage, psnr_val),
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
        'resolution_scale': 6.0,
        'max_gaussians': 500000,
        'sampling_method': 'intelligent_sampling',
        'progressive_results': [
            {
                'stage': r['stage'],
                'layer_name': r['layer_name'],
                'layer_description': r['layer_description'],
                'file': r['file'],
                'success': r['images'] is not None,
                'psnr': r['metrics'].get('psnr', 0) if r['images'] is not None else None,
                'gaussian_count': r['metrics'].get('gaussian_count', 0) if r['images'] is not None else None,
                'original_count': r['metrics'].get('original_count', 0) if r['images'] is not None else None,
                'sampling_ratio': r['metrics'].get('sampling_ratio', 1.0) if r['images'] is not None else None,
                'was_sampled': r['metrics'].get('was_sampled', False) if r['images'] is not None else None,
                'error': r['metrics'].get('error') if r['images'] is None else None
            }
            for r in progressive_results
        ],
        'contribution_analysis': contribution_analysis,
        'quality_check': {
            'has_negative_contributions': len(negative_contributions) > 0,
            'negative_contribution_stages': [ca['layer_name'] for ca in negative_contributions],
            'monotonic_increase': all(ca['psnr_contribution'] >= -0.1 for ca in contribution_analysis[1:])
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
        print(f"  最终PSNR: {successful_results[-1]['metrics']['psnr']:.2f}dB")
        if len(successful_results) > 1:
            total_gain = successful_results[-1]['metrics']['psnr'] - successful_results[0]['metrics']['psnr']
            print(f"  总体提升: {total_gain:.2f}dB")
            
            if contribution_analysis:
                best_contrib = max(contribution_analysis, key=lambda x: x['psnr_contribution'])
                print(f"  最大贡献阶段: {best_contrib['layer_name']} (+{best_contrib['psnr_contribution']:.2f}dB)")
                
                print(f"  质量检查: {'✅ 单调递增' if evaluation_results['quality_check']['monotonic_increase'] else '❌ 存在负增长'}")
    
    return evaluation_results

def main():
    print("📈 修复版尺寸分层渐进式评估")
    print("=" * 50)
    
    layers_dir = "size_based_layers"
    
    if not os.path.exists(layers_dir):
        print(f"❌ 分层目录不存在: {layers_dir}")
        print("请先运行 create_size_based_layers.py")
        return
    
    # 执行修复版评估
    results = evaluate_size_progressive_fixed(layers_dir)
    
    if results:
        print(f"\n🎉 修复版评估完成!")
        print(f"📁 输出目录: size_progressive_fixed/")

if __name__ == "__main__":
    main() 