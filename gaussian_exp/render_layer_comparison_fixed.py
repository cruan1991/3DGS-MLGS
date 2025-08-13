import os
import sys
import torch
import numpy as np
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob

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

def load_cameras_sample(colmap_path, images_path, resolution_scale=2.0, num_cameras=3):
    """加载几个采样相机用于对比"""
    cameras_bin = os.path.join(colmap_path, 'cameras.bin')
    images_bin = os.path.join(colmap_path, 'images.bin')
    
    cam_intrinsics = read_intrinsics_binary(cameras_bin)
    cam_extrinsics = read_extrinsics_binary(images_bin)
    
    # 选择几个有代表性的相机（均匀采样）
    img_ids = list(cam_extrinsics.keys())
    selected_ids = [img_ids[i] for i in np.linspace(0, len(img_ids)-1, num_cameras, dtype=int)]
    
    cameras = []
    
    for idx, img_id in enumerate(selected_ids):
        img_info = cam_extrinsics[img_id]
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
        print(f"✅ 相机 {idx}: {img_info.name}")
    
    return cameras

def render_ply_file(ply_path, camera, pipe, background):
    """渲染单个PLY文件"""
    if not os.path.exists(ply_path):
        return None, {"error": "File not found"}
    
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
        
        # 转换为numpy用于可视化
        rendered_np = rendered_image.detach().cpu().numpy().transpose(1, 2, 0)
        gt_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
        
        return (rendered_np, gt_np), {
            "psnr": psnr_val,
            "l1_loss": l1_val,
            "gaussian_count": gaussians.get_xyz.shape[0]
        }
        
    except Exception as e:
        return None, {"error": str(e)}

def create_comprehensive_comparison(camera_idx, camera_name, layer_files, progressive_files, camera, pipe, background, output_dir):
    """创建综合对比图"""
    print(f"🎨 渲染相机 {camera_idx}: {camera_name}")
    
    # 渲染所有文件
    single_layer_results = []
    progressive_results = []
    
    # 渲染单层文件
    for i, layer_file in enumerate(layer_files):
        print(f"  渲染层 {i}...")
        images, metrics = render_ply_file(layer_file, camera, pipe, background)
        single_layer_results.append((images, metrics, i))
    
    # 渲染渐进式文件
    for i, prog_file in enumerate(progressive_files):
        print(f"  渲染累积文件 {i+1}/5...")
        images, metrics = render_ply_file(prog_file, camera, pipe, background)
        progressive_results.append((images, metrics, i))
    
    # 创建大的对比图
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    fig.suptitle(f'高斯球分层渲染对比 - Camera {camera_idx} ({camera_name})', fontsize=20, fontweight='bold')
    
    # 第一行：单层渲染结果
    for i in range(5):
        ax = axes[0, i]
        if i < len(single_layer_results) and single_layer_results[i][0] is not None:
            images, metrics, layer_id = single_layer_results[i]
            ax.imshow(images[0])  # 显示渲染图像
            title = f"层{layer_id}\n{metrics['gaussian_count']:,}球\nPSNR: {metrics['psnr']:.2f}dB"
        else:
            ax.text(0.5, 0.5, f"层{i}\n无数据", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            title = f"层{i}"
        
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # 第二行：渐进式累积结果
    for i in range(5):
        ax = axes[1, i]
        if i < len(progressive_results) and progressive_results[i][0] is not None:
            images, metrics, prog_id = progressive_results[i]
            ax.imshow(images[0])  # 显示渲染图像
            if i == 0:
                layers_str = "L0"
            else:
                layers_str = f"L0-L{i}"
            title = f"{layers_str}\n{metrics['gaussian_count']:,}球\nPSNR: {metrics['psnr']:.2f}dB"
        else:
            ax.text(0.5, 0.5, f"累积{i+1}\n无数据", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            title = f"累积{i+1}"
        
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # 第三行：关键对比
    # GT
    ax = axes[2, 0]
    if progressive_results and progressive_results[0][0] is not None:
        ax.imshow(progressive_results[0][0][1])
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 层3核心
    ax = axes[2, 1]
    if len(progressive_results) > 3 and progressive_results[3][0] is not None:
        ax.imshow(progressive_results[3][0][0])
        ax.set_title(f'层3核心\n{progressive_results[3][1]["psnr"]:.2f}dB', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 完整模型
    ax = axes[2, 2]
    if progressive_results and progressive_results[-1][0] is not None:
        ax.imshow(progressive_results[-1][0][0])
        ax.set_title(f'完整模型\n{progressive_results[-1][1]["psnr"]:.2f}dB', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 差异图：完整-GT
    ax = axes[2, 3]
    if progressive_results and progressive_results[-1][0] is not None:
        rendered = progressive_results[-1][0][0]
        gt = progressive_results[-1][0][1]
        diff = np.abs(rendered - gt)
        ax.imshow(diff)
        ax.set_title('差异图\n(完整-GT)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # PSNR曲线
    ax = axes[2, 4]
    if progressive_results:
        psnr_values = [res[1]['psnr'] for res in progressive_results if res[0] is not None]
        counts = [res[1]['gaussian_count'] for res in progressive_results if res[0] is not None]
        
        x_labels = ['L0', 'L0-L1', 'L0-L2', 'L0-L3', 'L0-L4'][:len(psnr_values)]
        
        ax.plot(range(len(psnr_values)), psnr_values, 'bo-', linewidth=3, markersize=8)
        ax.set_xticks(range(len(psnr_values)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('PSNR Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注数值
        for i, (psnr_val, count) in enumerate(zip(psnr_values, counts)):
            ax.annotate(f'{psnr_val:.1f}', (i, psnr_val), 
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=10, fontweight='bold')
    
    # 第四行：贡献分析
    ax = axes[3, 0]
    if progressive_results:
        # 绘制每层的贡献
        contributions = []
        for i in range(len(progressive_results)):
            if i == 0:
                contrib = progressive_results[i][1]['psnr'] if progressive_results[i][0] else 0
            else:
                prev_psnr = progressive_results[i-1][1]['psnr'] if progressive_results[i-1][0] else 0
                curr_psnr = progressive_results[i][1]['psnr'] if progressive_results[i][0] else 0
                contrib = curr_psnr - prev_psnr
            contributions.append(contrib)
        
        bars = ax.bar(range(len(contributions)), contributions, 
                     color=['red', 'orange', 'yellow', 'green', 'blue'][:len(contributions)])
        ax.set_xticks(range(len(contributions)))
        ax.set_xticklabels([f'L{i}' for i in range(len(contributions))])
        ax.set_ylabel('PSNR Contribution (dB)')
        ax.set_title('Layer Contributions', fontsize=14, fontweight='bold')
        
        # 标注数值
        for i, (bar, contrib) in enumerate(zip(bars, contributions)):
            height = bar.get_height()
            ax.annotate(f'{contrib:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 高斯球数量分布
    ax = axes[3, 1]
    if single_layer_results:
        counts = [res[1]['gaussian_count'] for res in single_layer_results if res[0] is not None]
        labels = [f'L{i}' for i in range(len(counts))]
        
        ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Gaussian Distribution', fontsize=14, fontweight='bold')
    
    # 清空剩余子图
    for i in range(2, 5):
        axes[3, i].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_file = os.path.join(output_dir, f'comprehensive_comparison_camera_{camera_idx}_{camera_name.replace(".jpg", "")}.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存综合对比图: {output_file}")
    
    return output_file

def main():
    print("🎨 分层渲染综合可视化对比")
    print("=" * 60)
    
    # 参数
    layer_dir = 'layer_progressive_analysis'
    output_dir = 'layer_comprehensive_comparison'
    num_cameras = 3
    resolution_scale = 2.0
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 找到所有文件
    layer_files = sorted(glob.glob(os.path.join(layer_dir, "layer_*.ply")))
    progressive_files = sorted(glob.glob(os.path.join(layer_dir, "progressive_*.ply")))
    
    print(f"📁 找到分层文件: {len(layer_files)}个")
    for f in layer_files:
        print(f"  {os.path.basename(f)}")
    
    print(f"📈 找到渐进文件: {len(progressive_files)}个")
    for f in progressive_files:
        print(f"  {os.path.basename(f)}")
    
    if len(layer_files) == 0:
        print("❌ 未找到分层文件！")
        return
    
    # 设置渲染环境
    print(f"\n⚙️ 设置渲染环境...")
    
    # Pipeline参数
    pipeline_parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(pipeline_parser)
    pipe_args = pipeline_parser.parse_args([])
    pipe = pipe_parser.extract(pipe_args)
    
    # 背景
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 加载相机
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    cameras = load_cameras_sample(colmap_path, images_path, resolution_scale, num_cameras)
    
    print(f"✅ 加载了 {len(cameras)} 个相机")
    
    # 为每个相机创建综合对比图
    print(f"\n🎨 开始综合渲染对比...")
    
    comparison_files = []
    for i, camera in enumerate(cameras):
        comparison_file = create_comprehensive_comparison(
            i, camera.image_name, layer_files, progressive_files,
            camera, pipe, background, output_dir
        )
        comparison_files.append(comparison_file)
    
    print(f"\n🎉 综合对比完成!")
    print(f"📊 生成了 {len(comparison_files)} 个综合对比图")
    print(f"📁 保存在: {output_dir}/")
    
    for file in comparison_files:
        print(f"  📸 {file}")

if __name__ == "__main__":
    main() 