import os
import sys
import torch
import numpy as np
import argparse
import json
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

def load_cameras_sample(colmap_path, images_path, resolution_scale=2.0, num_cameras=5):
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

def create_layer_comparison_grid(camera_idx, camera_name, layer_files, progressive_files, layer_info, camera, pipe, background, output_dir):
    """为单个相机创建分层对比网格"""
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
        print(f"  渲染累积 L0-L{i}...")
        images, metrics = render_ply_file(prog_file, camera, pipe, background)
        progressive_results.append((images, metrics, i))
    
    # 创建对比图
    fig = plt.figure(figsize=(20, 16))
    
    # 使用GridSpec进行布局
    gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1])
    
    # 第一行：单层渲染结果
    fig.text(0.02, 0.85, f'单层渲染 - Camera {camera_idx} ({camera_name})', fontsize=16, fontweight='bold')
    for i, (images, metrics, layer_id) in enumerate(single_layer_results):
        ax = fig.add_subplot(gs[0, i])
        if images is not None:
            ax.imshow(images[0])  # 显示渲染图像
            title = f"层{layer_id}\n{metrics['gaussian_count']:,}球\nPSNR: {metrics['psnr']:.2f}dB"
        else:
            ax.text(0.5, 0.5, f"层{layer_id}\n渲染失败", ha='center', va='center', transform=ax.transAxes)
            title = f"层{layer_id}\n错误"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # 第二行：渐进式累积结果
    fig.text(0.02, 0.55, '渐进式累积渲染', fontsize=16, fontweight='bold')
    for i, (images, metrics, prog_id) in enumerate(progressive_results):
        ax = fig.add_subplot(gs[1, i])
        if images is not None:
            ax.imshow(images[0])  # 显示渲染图像
            layers_str = f"L0-L{prog_id}" if prog_id > 0 else "L0"
            title = f"{layers_str}\n{metrics['gaussian_count']:,}球\nPSNR: {metrics['psnr']:.2f}dB"
        else:
            layers_str = f"L0-L{prog_id}" if prog_id > 0 else "L0"
            ax.text(0.5, 0.5, f"{layers_str}\n渲染失败", ha='center', va='center', transform=ax.transAxes)
            title = f"{layers_str}\n错误"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # 第三行：关键对比（GT, 层3, 完整模型, 差异图）
    fig.text(0.02, 0.25, '关键对比分析', fontsize=16, fontweight='bold')
    
    # GT图像
    ax_gt = fig.add_subplot(gs[2, 0])
    if progressive_results[0][0] is not None:
        ax_gt.imshow(progressive_results[0][0][1])  # GT图像
    ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax_gt.axis('off')
    
    # 层3（核心层）
    ax_core = fig.add_subplot(gs[2, 1])
    if len(progressive_results) > 3 and progressive_results[3][0] is not None:
        ax_core.imshow(progressive_results[3][0][0])
        ax_core.set_title(f'层3 核心\n{progressive_results[3][1]["psnr"]:.2f}dB', fontsize=12, fontweight='bold')
    ax_core.axis('off')
    
    # 完整模型
    ax_full = fig.add_subplot(gs[2, 2])
    if progressive_results[-1][0] is not None:
        ax_full.imshow(progressive_results[-1][0][0])
        ax_full.set_title(f'完整模型\n{progressive_results[-1][1]["psnr"]:.2f}dB', fontsize=12, fontweight='bold')
    ax_full.axis('off')
    
    # 层4单独（前景细节）
    ax_layer4 = fig.add_subplot(gs[2, 3])
    if len(single_layer_results) > 4 and single_layer_results[4][0] is not None:
        ax_layer4.imshow(single_layer_results[4][0][0])
        ax_layer4.set_title(f'层4 细节\n{single_layer_results[4][1]["psnr"]:.2f}dB', fontsize=12, fontweight='bold')
    ax_layer4.axis('off')
    
    # PSNR进化曲线
    ax_curve = fig.add_subplot(gs[2, 4:])
    psnr_values = [res[1]['psnr'] for res in progressive_results if res[0] is not None]
    gaussian_counts = [res[1]['gaussian_count'] for res in progressive_results if res[0] is not None]
    
    ax_curve.plot(range(len(psnr_values)), psnr_values, 'bo-', linewidth=2, markersize=8)
    ax_curve.set_xlabel('累积层数')
    ax_curve.set_ylabel('PSNR (dB)')
    ax_curve.set_title('PSNR随层数累积的变化', fontsize=12, fontweight='bold')
    ax_curve.grid(True, alpha=0.3)
    
    # 添加数值标注
    for i, (psnr_val, count) in enumerate(zip(psnr_values, gaussian_counts)):
        ax_curve.annotate(f'{psnr_val:.1f}dB\n{count//1000}k球', 
                         (i, psnr_val), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图像
    output_file = os.path.join(output_dir, f'layer_comparison_camera_{camera_idx}_{camera_name.replace(".jpg", "")}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存对比图: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='分层渲染可视化对比')
    parser.add_argument('--layer-dir', type=str, default='layer_progressive_analysis', help='分层文件目录')
    parser.add_argument('--num-cameras', type=int, default=3, help='选择相机数量')
    parser.add_argument('--resolution-scale', type=float, default=2.0, help='分辨率缩放')
    parser.add_argument('--output-dir', type=str, default='layer_visual_comparison', help='输出目录')
    
    args = parser.parse_args()
    
    print("🎨 分层渲染可视化对比")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 找到分层文件
    layer_files = []
    progressive_files = []
    
    for i in range(5):  # 假设5层
        layer_file = os.path.join(args.layer_dir, f"layer_{i}_z*.ply")
        import glob
        matches = glob.glob(layer_file)
        if matches:
            layer_files.append(matches[0])
    
    for i in range(5):  # 5个渐进文件
        if i == 0:
            prog_file = os.path.join(args.layer_dir, "progressive_L0_*.ply")
        else:
            prog_file = os.path.join(args.layer_dir, f"progressive_L0_L{i}_*.ply")
        
        matches = glob.glob(prog_file)
        if matches:
            progressive_files.append(matches[0])
    
    print(f"📁 找到分层文件: {len(layer_files)}个")
    print(f"📈 找到渐进文件: {len(progressive_files)}个")
    
    if len(layer_files) == 0:
        print("❌ 未找到分层文件！请先运行layer_progressive_evaluation.py")
        return
    
    # 读取层级信息（如果有的话）
    layer_info = []
    results_file = os.path.join(args.layer_dir, 'layer_progressive_results.json')
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                layer_info = data.get('layer_info', [])
        except:
            pass
    
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
    cameras = load_cameras_sample(colmap_path, images_path, args.resolution_scale, args.num_cameras)
    
    print(f"✅ 加载了 {len(cameras)} 个相机")
    
    # 为每个相机创建对比图
    print(f"\n🎨 开始渲染对比...")
    
    comparison_files = []
    for i, camera in enumerate(cameras):
        comparison_file = create_layer_comparison_grid(
            i, camera.image_name, layer_files, progressive_files, 
            layer_info, camera, pipe, background, args.output_dir
        )
        comparison_files.append(comparison_file)
    
    print(f"\n🎉 渲染对比完成!")
    print(f"📊 生成了 {len(comparison_files)} 个对比图")
    print(f"📁 保存在: {args.output_dir}/")
    
    for file in comparison_files:
        print(f"  📸 {file}")

if __name__ == "__main__":
    main() 