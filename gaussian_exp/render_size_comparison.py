import os
import sys
import torch
import numpy as np
import argparse
import json
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def render_ply_file_safe(ply_path, camera, pipe, background):
    """安全渲染PLY文件，处理内存不足的情况"""
    if not os.path.exists(ply_path):
        return None, {"error": "File not found"}
    
    try:
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 加载高斯球
        gaussians = GaussianModel(3)
        gaussians.load_ply(ply_path, use_train_test_exp=False)
        
        gaussian_count = gaussians.get_xyz.shape[0]
        print(f"    加载了 {gaussian_count:,} 个高斯球")
        
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
            "gaussian_count": gaussian_count
        }
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"    ⚠️ GPU内存不足: {str(e)[:100]}...")
        torch.cuda.empty_cache()
        return None, {"error": "CUDA OOM"}
    except Exception as e:
        print(f"    ❌ 渲染失败: {str(e)}")
        torch.cuda.empty_cache()
        return None, {"error": str(e)}

def create_size_comparison_visualization(layers_dir, output_dir='size_visual_comparison'):
    """创建尺寸分层的可视化对比"""
    print("🎨 创建尺寸分层可视化对比")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置渲染环境
    pipeline_parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(pipeline_parser)
    pipe_args = pipeline_parser.parse_args([])
    pipe = pipe_parser.extract(pipe_args)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 加载相机 (使用4x缩放节省内存)
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    camera = load_test_camera(colmap_path, images_path, "000001.jpg", 4.0)
    
    if camera is None:
        return
    
    print(f"✅ 加载测试相机: 000001.jpg (分辨率: {camera.image_width}x{camera.image_height})")
    
    # 查找所有单层PLY文件
    single_layer_files = sorted(glob.glob(os.path.join(layers_dir, "size_layer_*.ply")))
    
    print(f"📁 找到单层文件: {len(single_layer_files)}个")
    
    # 渲染单层文件
    single_layer_results = []
    layer_names = ['超小球', '小球', '中球', '大球', '超大球']
    
    for i, layer_file in enumerate(single_layer_files):
        filename = os.path.basename(layer_file)
        layer_name = layer_names[i] if i < len(layer_names) else f"层{i}"
        print(f"\n🎯 渲染 {layer_name}: {filename}")
        
        images, metrics = render_ply_file_safe(layer_file, camera, pipe, background)
        single_layer_results.append((images, metrics, i, layer_name))
        
        if images is not None:
            print(f"    ✅ PSNR: {metrics['psnr']:.2f}dB, 球数: {metrics['gaussian_count']:,}")
        else:
            print(f"    ❌ 渲染失败: {metrics.get('error', 'Unknown')}")
    
    # 创建对比图
    print(f"\n🎨 生成可视化对比图...")
    
    # 计算布局
    num_layers = len(single_layer_results)
    cols = min(3, num_layers)  # 最多3列
    rows = (num_layers + cols - 1) // cols  # 计算需要的行数
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if num_layers == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    fig.suptitle('尺寸分层渲染对比 - Camera 000001.jpg', fontsize=16, fontweight='bold')
    
    # 绘制每层结果
    for i, (images, metrics, layer_id, layer_name) in enumerate(single_layer_results):
        ax = axes[i]
        
        if images is not None:
            ax.imshow(images[0])  # 显示渲染图像
            title = f"{layer_name}\n{metrics['gaussian_count']:,}球\nPSNR: {metrics['psnr']:.2f}dB"
        else:
            ax.text(0.5, 0.5, f"{layer_name}\n渲染失败\n{metrics.get('error', '')}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            title = f"{layer_name}\n渲染失败"
        
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_file = os.path.join(output_dir, 'size_layers_comparison.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化保存: {output_file}")
    
    # 创建统计分析图
    print(f"\n📊 生成统计分析图...")
    
    # 提取成功渲染的数据
    successful_results = [(images, metrics, layer_id, layer_name) 
                         for images, metrics, layer_id, layer_name in single_layer_results 
                         if images is not None]
    
    if len(successful_results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('尺寸分层统计分析', fontsize=16, fontweight='bold')
        
        # 1. PSNR对比
        ax = axes[0, 0]
        psnr_values = [metrics['psnr'] for _, metrics, _, _ in successful_results]
        layer_labels = [layer_name for _, _, _, layer_name in successful_results]
        
        bars = ax.bar(range(len(psnr_values)), psnr_values, 
                     color=['red', 'orange', 'yellow', 'green', 'blue'][:len(psnr_values)])
        ax.set_xticks(range(len(psnr_values)))
        ax.set_xticklabels(layer_labels, rotation=45)
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('各层PSNR对比')
        
        # 标注数值
        for i, (bar, psnr_val) in enumerate(zip(bars, psnr_values)):
            height = bar.get_height()
            ax.annotate(f'{psnr_val:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # 2. 高斯球数量分布
        ax = axes[0, 1]
        gaussian_counts = [metrics['gaussian_count'] for _, metrics, _, _ in successful_results]
        
        ax.pie(gaussian_counts, labels=layer_labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('高斯球数量分布')
        
        # 3. 效率分析 (PSNR/高斯球数)
        ax = axes[1, 0]
        efficiency = [psnr / (count / 100000) for psnr, count in zip(psnr_values, gaussian_counts)]
        
        bars = ax.bar(range(len(efficiency)), efficiency,
                     color=['red', 'orange', 'yellow', 'green', 'blue'][:len(efficiency)])
        ax.set_xticks(range(len(efficiency)))
        ax.set_xticklabels(layer_labels, rotation=45)
        ax.set_ylabel('PSNR per 100k Gaussians')
        ax.set_title('渲染效率分析')
        
        # 4. 尺寸范围说明
        ax = axes[1, 1]
        ax.axis('off')
        
        # 读取分层信息
        manifest_path = os.path.join(layers_dir, 'size_layers_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            layer_info = manifest.get('layer_info', [])
            
            info_text = "尺寸分层方案:\n\n"
            for layer in layer_info:
                if layer['layer_id'] < len(successful_results):
                    info_text += f"层{layer['layer_id']} ({layer['name']}):\n"
                    info_text += f"  尺寸范围: {layer['threshold_range']}\n"
                    info_text += f"  高斯球数: {layer['count']:,} ({layer['percentage']:.1f}%)\n\n"
            
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, '未找到分层信息', transform=ax.transAxes, ha='center', va='center')
        
        ax.set_title('分层方案详情')
        
        plt.tight_layout()
        
        # 保存统计图
        stats_file = os.path.join(output_dir, 'size_layers_statistics.png')
        plt.savefig(stats_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 统计图保存: {stats_file}")
    
    # 保存结果摘要
    summary = {
        'test_camera': '000001.jpg',
        'resolution_scale': 4.0,
        'total_layers': len(single_layer_results),
        'successful_renders': len(successful_results),
        'results': [
            {
                'layer_id': layer_id,
                'layer_name': layer_name,
                'success': images is not None,
                'psnr': metrics.get('psnr', 0) if images is not None else None,
                'gaussian_count': metrics.get('gaussian_count', 0) if images is not None else None,
                'error': metrics.get('error') if images is None else None
            }
            for images, metrics, layer_id, layer_name in single_layer_results
        ]
    }
    
    summary_file = os.path.join(output_dir, 'size_comparison_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ 结果摘要: {summary_file}")
    
    return summary

def main():
    print("🎨 尺寸分层可视化对比")
    print("=" * 40)
    
    layers_dir = "size_based_layers"
    
    if not os.path.exists(layers_dir):
        print(f"❌ 分层目录不存在: {layers_dir}")
        print("请先运行 create_size_based_layers.py")
        return
    
    # 执行可视化
    summary = create_size_comparison_visualization(layers_dir)
    
    if summary:
        print(f"\n🎉 尺寸分层可视化完成!")
        print(f"📁 输出目录: size_visual_comparison/")
        print(f"✅ 成功渲染: {summary['successful_renders']}/{summary['total_layers']} 层")
        
        # 打印成功的层
        for result in summary['results']:
            if result['success']:
                print(f"  {result['layer_name']}: {result['psnr']:.2f}dB ({result['gaussian_count']:,}球)")
            else:
                print(f"  {result['layer_name']}: 失败 ({result['error']})")

if __name__ == "__main__":
    main() 