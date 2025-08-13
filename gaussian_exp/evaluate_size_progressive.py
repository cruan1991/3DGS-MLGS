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

def load_test_camera(colmap_path, images_path, camera_name="000001.jpg", resolution_scale=4.0):
    """加载测试相机，使用较大的缩放以节省内存"""
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

def render_ply_with_memory_management(ply_path, camera, pipe, background, max_gaussians=800000):
    """内存管理的PLY渲染"""
    if not os.path.exists(ply_path):
        return None, {"error": "File not found"}
    
    try:
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 加载高斯球
        gaussians = GaussianModel(3)
        gaussians.load_ply(ply_path, use_train_test_exp=False)
        
        gaussian_count = gaussians.get_xyz.shape[0]
        print(f"    加载了 {gaussian_count:,} 个高斯球", end="")
        
        # 如果高斯球太多，进行采样
        if gaussian_count > max_gaussians:
            print(f" -> 采样到 {max_gaussians:,} 个")
            
            # 随机采样
            indices = np.random.choice(gaussian_count, size=max_gaussians, replace=False)
            indices = torch.from_numpy(indices).long()
            
            # 更新高斯球参数
            gaussians._xyz = gaussians._xyz[indices]
            gaussians._features_dc = gaussians._features_dc[indices]
            gaussians._features_rest = gaussians._features_rest[indices]
            gaussians._scaling = gaussians._scaling[indices]
            gaussians._rotation = gaussians._rotation[indices]
            gaussians._opacity = gaussians._opacity[indices]
            
            gaussian_count = max_gaussians
        else:
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
            "gaussian_count": gaussian_count,
            "was_sampled": gaussian_count == max_gaussians
        }
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"    ⚠️ GPU内存不足: {str(e)[:100]}...")
        torch.cuda.empty_cache()
        return None, {"error": "CUDA OOM"}
    except Exception as e:
        print(f"    ❌ 渲染失败: {str(e)}")
        torch.cuda.empty_cache()
        return None, {"error": str(e)}

def evaluate_size_progressive_layers(layers_dir, output_dir='size_progressive_comparison'):
    """评估尺寸分层的渐进式累积效果"""
    print("📈 评估尺寸分层渐进式累积效果")
    print("=" * 60)
    
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
    camera = load_test_camera(colmap_path, images_path, "000001.jpg", 4.0)
    
    if camera is None:
        return
    
    print(f"✅ 加载测试相机: 000001.jpg (分辨率: {camera.image_width}x{camera.image_height})")
    
    # 查找渐进式PLY文件
    progressive_files = sorted(glob.glob(os.path.join(layers_dir, "size_progressive_*.ply")))
    
    print(f"📈 找到渐进文件: {len(progressive_files)}个")
    for i, f in enumerate(progressive_files):
        print(f"  {i}: {os.path.basename(f)}")
    
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
    
    print(f"\n🎯 开始渐进式渲染...")
    
    for i, prog_file in enumerate(progressive_files):
        layer_name = layer_names[i] if i < len(layer_names) else f"Stage{i}"
        layer_desc = layer_descriptions[i] if i < len(layer_descriptions) else f"阶段{i}"
        
        print(f"\n🎨 渲染阶段{i} ({layer_name}): {layer_desc}")
        print(f"   文件: {os.path.basename(prog_file)}")
        
        images, metrics = render_ply_with_memory_management(prog_file, camera, pipe, background)
        
        progressive_results.append({
            'stage': i,
            'layer_name': layer_name,
            'layer_description': layer_desc,
            'images': images,
            'metrics': metrics,
            'file': os.path.basename(prog_file)
        })
        
        if images is not None:
            sampled_note = " (采样)" if metrics.get('was_sampled') else ""
            print(f"   ✅ PSNR: {metrics['psnr']:.2f}dB, 球数: {metrics['gaussian_count']:,}{sampled_note}")
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
            prev_psnr = 0
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
            'efficiency': contribution / (result['metrics']['gaussian_count'] / 1000000) if result['metrics']['gaussian_count'] > 0 else 0
        })
        
        print(f"  阶段{result['stage']} ({result['layer_name']}): {current_psnr:.2f}dB (+{contribution:.2f}), {result['metrics']['gaussian_count']:,}球")
    
    # 创建可视化对比
    print(f"\n🎨 生成渐进式对比图...")
    
    num_stages = len(successful_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('尺寸分层渐进式累积效果 - Camera 000001.jpg', fontsize=16, fontweight='bold')
    
    # 绘制前5个阶段的渲染结果 (如果有的话)
    for i in range(min(5, num_stages)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        result = successful_results[i]
        if result['images'] is not None:
            ax.imshow(result['images'][0])  # 显示渲染图像
            title = f"{result['layer_name']}\n{result['metrics']['gaussian_count']:,}球\nPSNR: {result['metrics']['psnr']:.2f}dB"
        else:
            ax.text(0.5, 0.5, f"{result['layer_name']}\n渲染失败", 
                   ha='center', va='center', transform=ax.transAxes)
            title = f"{result['layer_name']}\n失败"
        
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # 最后一个子图显示PSNR进化曲线
    ax = axes[1, 2]
    if len(contribution_analysis) > 1:
        stages = [ca['stage'] for ca in contribution_analysis]
        psnr_values = [ca['cumulative_psnr'] for ca in contribution_analysis]
        
        ax.plot(stages, psnr_values, 'bo-', linewidth=3, markersize=8)
        ax.set_xlabel('累积阶段')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('PSNR进化曲线', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注数值
        for stage, psnr_val in zip(stages, psnr_values):
            ax.annotate(f'{psnr_val:.1f}', (stage, psnr_val),
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=10)
    
    # 隐藏多余的子图
    for i in range(min(5, num_stages), 6):
        row = i // 3
        col = i % 3
        if row < 2 and col < 3:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    comparison_file = os.path.join(output_dir, 'size_progressive_comparison.png')
    plt.savefig(comparison_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 渐进对比图保存: {comparison_file}")
    
    # 创建贡献分析图
    if len(contribution_analysis) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('尺寸分层渐进式贡献分析', fontsize=16, fontweight='bold')
        
        # 1. 每阶段PSNR贡献
        ax = axes[0]
        contributions = [ca['psnr_contribution'] for ca in contribution_analysis]
        stage_labels = [ca['layer_name'] for ca in contribution_analysis]
        
        bars = ax.bar(range(len(contributions)), contributions,
                     color=['red', 'orange', 'yellow', 'green', 'blue'][:len(contributions)])
        ax.set_xticks(range(len(contributions)))
        ax.set_xticklabels(stage_labels, rotation=45)
        ax.set_ylabel('PSNR贡献 (dB)')
        ax.set_title('各阶段PSNR贡献')
        
        # 标注数值
        for i, (bar, contrib) in enumerate(zip(bars, contributions)):
            height = bar.get_height()
            ax.annotate(f'{contrib:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # 2. 累积效率分析
        ax = axes[1]
        efficiencies = [ca['efficiency'] for ca in contribution_analysis]
        
        ax.plot(range(len(efficiencies)), efficiencies, 'go-', linewidth=2, markersize=6)
        ax.set_xticks(range(len(efficiencies)))
        ax.set_xticklabels(stage_labels, rotation=45)
        ax.set_ylabel('效率 (PSNR增量/M高斯球)')
        ax.set_title('各阶段效率变化')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        contribution_file = os.path.join(output_dir, 'size_progressive_contribution.png')
        plt.savefig(contribution_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 贡献分析图保存: {contribution_file}")
    
    # 保存详细结果
    evaluation_results = {
        'test_camera': '000001.jpg',
        'resolution_scale': 4.0,
        'progressive_results': [
            {
                'stage': r['stage'],
                'layer_name': r['layer_name'],
                'layer_description': r['layer_description'],
                'file': r['file'],
                'success': r['images'] is not None,
                'psnr': r['metrics'].get('psnr', 0) if r['images'] is not None else None,
                'l1_loss': r['metrics'].get('l1_loss', 0) if r['images'] is not None else None,
                'gaussian_count': r['metrics'].get('gaussian_count', 0) if r['images'] is not None else None,
                'was_sampled': r['metrics'].get('was_sampled', False) if r['images'] is not None else None,
                'error': r['metrics'].get('error') if r['images'] is None else None
            }
            for r in progressive_results
        ],
        'contribution_analysis': contribution_analysis,
        'summary': {
            'total_stages': len(progressive_results),
            'successful_stages': len(successful_results),
            'final_psnr': successful_results[-1]['metrics']['psnr'] if successful_results else 0,
            'total_psnr_gain': successful_results[-1]['metrics']['psnr'] - successful_results[0]['metrics']['psnr'] if len(successful_results) > 1 else 0,
            'best_contribution_stage': max(contribution_analysis, key=lambda x: x['psnr_contribution'])['stage'] if contribution_analysis else -1
        }
    }
    
    results_file = os.path.join(output_dir, 'size_progressive_evaluation.json')
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"✅ 详细结果保存: {results_file}")
    
    # 打印总结
    print(f"\n📊 渐进式评估总结:")
    print(f"  总阶段数: {len(progressive_results)}")
    print(f"  成功渲染: {len(successful_results)}/{len(progressive_results)}")
    if successful_results:
        print(f"  最终PSNR: {successful_results[-1]['metrics']['psnr']:.2f}dB")
        if len(successful_results) > 1:
            total_gain = successful_results[-1]['metrics']['psnr'] - successful_results[0]['metrics']['psnr']
            print(f"  总体提升: {total_gain:.2f}dB")
            
            best_contrib = max(contribution_analysis, key=lambda x: x['psnr_contribution'])
            print(f"  最大贡献阶段: {best_contrib['layer_name']} (+{best_contrib['psnr_contribution']:.2f}dB)")
    
    return evaluation_results

def main():
    print("📈 尺寸分层渐进式评估")
    print("=" * 50)
    
    layers_dir = "size_based_layers"
    
    if not os.path.exists(layers_dir):
        print(f"❌ 分层目录不存在: {layers_dir}")
        print("请先运行 create_size_based_layers.py")
        return
    
    # 执行渐进式评估
    results = evaluate_size_progressive_layers(layers_dir)
    
    if results:
        print(f"\n🎉 尺寸分层渐进式评估完成!")
        print(f"📁 输出目录: size_progressive_comparison/")

if __name__ == "__main__":
    main() 