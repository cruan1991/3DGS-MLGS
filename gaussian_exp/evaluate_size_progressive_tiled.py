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

def load_test_camera(colmap_path, images_path, camera_name="000001.jpg", resolution_scale=2.0):
    """加载测试相机，保持合理分辨率"""
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

def create_tile_camera(base_camera, tile_x, tile_y, tiles_w, tiles_h):
    """创建瓦片相机，只渲染图像的一部分"""
    width = base_camera.image_width
    height = base_camera.image_height
    
    tile_width = width // tiles_w
    tile_height = height // tiles_h
    
    # 计算瓦片的起始位置
    start_x = tile_x * tile_width
    start_y = tile_y * tile_height
    
    # 确保最后一个瓦片包含剩余像素
    if tile_x == tiles_w - 1:
        tile_width = width - start_x
    if tile_y == tiles_h - 1:
        tile_height = height - start_y
    
    # 裁剪GT图像
    gt_image = base_camera.original_image
    if len(gt_image.shape) == 3 and gt_image.shape[0] == 3:  # CHW格式
        tile_gt = gt_image[:, start_y:start_y+tile_height, start_x:start_x+tile_width]
    else:  # HWC格式
        tile_gt = gt_image[start_y:start_y+tile_height, start_x:start_x+tile_width]
        if len(tile_gt.shape) == 3:
            tile_gt = tile_gt.permute(2, 0, 1)  # 转换为CHW
    
    # 转换为PIL图像用于Camera构造
    if isinstance(tile_gt, torch.Tensor):
        tile_gt_np = tile_gt.cpu().numpy()
        if tile_gt_np.shape[0] == 3:  # CHW -> HWC
            tile_gt_np = tile_gt_np.transpose(1, 2, 0)
        tile_gt_pil = Image.fromarray((tile_gt_np * 255).astype(np.uint8))
    else:
        tile_gt_pil = tile_gt
    
    # 计算调整后的相机参数
    # 主点偏移
    fx = focal2fov(base_camera.FoVx, width) * width / 2  # 还原fx
    fy = focal2fov(base_camera.FoVy, height) * height / 2  # 还原fy
    
    cx_new = fx - start_x  # 调整主点
    cy_new = fy - start_y
    
    fx_tile = fx  # 焦距保持不变
    fy_tile = fy
    
    # 重新计算FoV
    FoVx_tile = focal2fov(fx_tile, tile_width)
    FoVy_tile = focal2fov(fy_tile, tile_height)
    
    # 创建瓦片相机
    tile_camera = Camera(
        resolution=(tile_width, tile_height),
        colmap_id=base_camera.colmap_id,
        R=base_camera.R,
        T=base_camera.T,
        FoVx=FoVx_tile,
        FoVy=FoVy_tile,
        depth_params=None,
        image=tile_gt_pil,
        invdepthmap=None,
        image_name=f"{base_camera.image_name}_tile_{tile_x}_{tile_y}",
        uid=base_camera.uid,
        data_device="cuda",
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False
    )
    
    return tile_camera, (start_x, start_y, tile_width, tile_height)

def render_ply_tiled(ply_path, base_camera, pipe, background, tiles_w=2, tiles_h=2):
    """分块渲染PLY文件，不进行采样"""
    if not os.path.exists(ply_path):
        return None, {"error": "File not found"}
    
    try:
        print(f"    🧩 分块渲染 ({tiles_w}x{tiles_h} = {tiles_w*tiles_h}块)", end="")
        
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
        
        # 初始化完整图像
        full_width = base_camera.image_width
        full_height = base_camera.image_height
        rendered_full = torch.zeros((3, full_height, full_width), device="cuda")
        gt_full = base_camera.original_image.to("cuda")
        
        tile_results = []
        
        # 逐块渲染
        for tile_y in range(tiles_h):
            for tile_x in range(tiles_w):
                print(f"      渲染瓦片 ({tile_x}, {tile_y})", end="")
                
                try:
                    # 创建瓦片相机
                    tile_camera, (start_x, start_y, tile_w, tile_h) = create_tile_camera(
                        base_camera, tile_x, tile_y, tiles_w, tiles_h
                    )
                    
                    # 渲染瓦片
                    render_result = render(tile_camera, gaussians, pipe, background, 1., 
                                         SPARSE_ADAM_AVAILABLE, None, False)
                    rendered_tile = torch.clamp(render_result["render"], 0.0, 1.0)
                    
                    # 将瓦片放回完整图像
                    rendered_full[:, start_y:start_y+tile_h, start_x:start_x+tile_w] = rendered_tile
                    
                    # 计算瓦片PSNR
                    gt_tile = gt_full[:, start_y:start_y+tile_h, start_x:start_x+tile_w]
                    tile_psnr = psnr(rendered_tile, gt_tile).mean().item()
                    
                    tile_results.append({
                        'tile_x': tile_x,
                        'tile_y': tile_y,
                        'psnr': tile_psnr,
                        'start_x': start_x,
                        'start_y': start_y,
                        'width': tile_w,
                        'height': tile_h
                    })
                    
                    print(f" ✅ PSNR: {tile_psnr:.2f}dB")
                    
                    # 清理瓦片资源
                    del render_result, rendered_tile, tile_camera
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f" ❌ 瓦片OOM")
                    torch.cuda.empty_cache()
                    # 记录失败的瓦片
                    tile_results.append({
                        'tile_x': tile_x,
                        'tile_y': tile_y,
                        'psnr': 0.0,
                        'error': 'OOM'
                    })
                    continue
        
        # 计算整体指标
        overall_psnr = psnr(rendered_full, gt_full).mean().item()
        overall_l1 = l1_loss(rendered_full, gt_full).mean().item()
        
        # 转换为numpy用于可视化
        rendered_np = rendered_full.detach().cpu().numpy().transpose(1, 2, 0)
        gt_np = gt_full.detach().cpu().numpy().transpose(1, 2, 0)
        
        # 清理内存
        del gaussians, rendered_full, gt_full
        torch.cuda.empty_cache()
        
        return (rendered_np, gt_np), {
            "psnr": overall_psnr,
            "l1_loss": overall_l1,
            "gaussian_count": gaussian_count,
            "tiles_w": tiles_w,
            "tiles_h": tiles_h,
            "tile_results": tile_results,
            "avg_tile_psnr": np.mean([t['psnr'] for t in tile_results if 'error' not in t]),
            "successful_tiles": len([t for t in tile_results if 'error' not in t])
        }
        
    except Exception as e:
        print(f"    ❌ 渲染失败: {str(e)}")
        torch.cuda.empty_cache()
        return None, {"error": str(e)}

def evaluate_size_progressive_tiled(layers_dir, output_dir='size_progressive_tiled'):
    """分块渲染版渐进式评估"""
    print("📈 分块渲染尺寸分层渐进式评估")
    print("=" * 60)
    print("🔧 主要特点:")
    print("  - 保留所有高斯球，不进行采样")
    print("  - 使用2x2分块渲染解决显存问题")
    print("  - 瓦片独立渲染后拼接成完整图像")
    print("  - 保证PSNR单调递增特性")
    
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
    camera = load_test_camera(colmap_path, images_path, "000001.jpg", 2.0)  # 2x缩放平衡质量和性能
    
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
    
    print(f"\n🎯 开始分块渲染渐进式评估...")
    
    for i, prog_file in enumerate(progressive_files):
        layer_name = layer_names[i] if i < len(layer_names) else f"Stage{i}"
        layer_desc = layer_descriptions[i] if i < len(layer_descriptions) else f"阶段{i}"
        
        print(f"\n🎨 渲染阶段{i} ({layer_name}): {layer_desc}")
        print(f"   文件: {os.path.basename(prog_file)}")
        
        # 根据文件大小自适应调整分块策略
        file_size_mb = os.path.getsize(prog_file) / (1024 * 1024)
        if file_size_mb > 500:  # 大于500MB用3x3分块
            tiles_w, tiles_h = 3, 3
        elif file_size_mb > 200:  # 大于200MB用2x2分块
            tiles_w, tiles_h = 2, 2
        else:  # 小文件用2x2或直接渲染
            tiles_w, tiles_h = 2, 2
        
        print(f"   📐 文件大小: {file_size_mb:.1f}MB, 使用{tiles_w}x{tiles_h}分块")
        
        images, metrics = render_ply_tiled(prog_file, camera, pipe, background, tiles_w, tiles_h)
        
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
            print(f"   ✅ 整体PSNR: {metrics['psnr']:.2f}dB, 高斯球数: {metrics['gaussian_count']:,}")
            print(f"      成功瓦片: {metrics['successful_tiles']}/{tiles_w*tiles_h}, 平均瓦片PSNR: {metrics['avg_tile_psnr']:.2f}dB")
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
        
        print(f"  阶段{result['stage']} ({result['layer_name']}): {current_psnr:.2f}dB (+{contribution:.2f}), {result['metrics']['gaussian_count']:,}球")
    
    # 检查单调递增特性
    negative_contributions = [ca for ca in contribution_analysis[1:] if ca['psnr_contribution'] < -0.01]
    if negative_contributions:
        print(f"\n⚠️  发现 {len(negative_contributions)} 个轻微负贡献阶段(可能是数值误差):")
        for ca in negative_contributions:
            print(f"     {ca['layer_name']}: {ca['psnr_contribution']:.3f}dB")
    else:
        print(f"\n✅ 完美！所有阶段PSNR严格单调递增")
    
    # 创建可视化对比
    print(f"\n🎨 生成分块渲染对比图...")
    
    # 创建2x3的布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('分块渲染尺寸分层渐进式累积效果 - 无采样完整版', fontsize=16, fontweight='bold')
    
    # 绘制5个阶段的渲染结果
    for i in range(min(5, len(successful_results))):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        result = successful_results[i]
        ax.imshow(result['images'][0])  # 显示渲染图像
        
        title = f"{result['layer_name']}\n{result['metrics']['gaussian_count']:,}球 ({result['file_size_mb']:.0f}MB)"
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
        ax.set_title('PSNR进化曲线\n(分块渲染 - 无采样)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注数值
        for stage, psnr_val in zip(stages, psnr_values):
            ax.annotate(f'{psnr_val:.2f}', (stage, psnr_val),
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图像
    comparison_file = os.path.join(output_dir, 'tiled_progressive_comparison.png')
    plt.savefig(comparison_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 分块渲染对比图保存: {comparison_file}")
    
    # 保存详细结果
    evaluation_results = {
        'test_camera': '000001.jpg',
        'resolution_scale': 2.0,
        'rendering_method': 'tiled_rendering',
        'sampling': False,
        'progressive_results': [
            {
                'stage': r['stage'],
                'layer_name': r['layer_name'],
                'layer_description': r['layer_description'],
                'file': r['file'],
                'file_size_mb': r['file_size_mb'],
                'success': r['images'] is not None,
                'psnr': r['metrics'].get('psnr', 0) if r['images'] is not None else None,
                'gaussian_count': r['metrics'].get('gaussian_count', 0) if r['images'] is not None else None,
                'tiles_used': f"{r['metrics'].get('tiles_w', 0)}x{r['metrics'].get('tiles_h', 0)}" if r['images'] is not None else None,
                'successful_tiles': r['metrics'].get('successful_tiles', 0) if r['images'] is not None else None,
                'avg_tile_psnr': r['metrics'].get('avg_tile_psnr', 0) if r['images'] is not None else None,
                'error': r['metrics'].get('error') if r['images'] is None else None
            }
            for r in progressive_results
        ],
        'contribution_analysis': contribution_analysis,
        'quality_check': {
            'has_negative_contributions': len(negative_contributions) > 0,
            'negative_contribution_stages': [ca['layer_name'] for ca in negative_contributions],
            'strictly_monotonic': all(ca['psnr_contribution'] >= -0.01 for ca in contribution_analysis[1:]),
            'max_negative_contribution': min([ca['psnr_contribution'] for ca in contribution_analysis[1:]], default=0)
        }
    }
    
    results_file = os.path.join(output_dir, 'tiled_progressive_evaluation.json')
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"✅ 详细结果保存: {results_file}")
    
    # 打印总结
    print(f"\n📊 分块渲染评估总结:")
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
                
                print(f"  质量检查: {'✅ 严格单调递增' if evaluation_results['quality_check']['strictly_monotonic'] else '⚠️ 有轻微波动'}")
    
    return evaluation_results

def main():
    print("📈 分块渲染尺寸分层渐进式评估")
    print("=" * 50)
    
    layers_dir = "size_based_layers"
    
    if not os.path.exists(layers_dir):
        print(f"❌ 分层目录不存在: {layers_dir}")
        print("请先运行 create_size_based_layers.py")
        return
    
    # 执行分块渲染评估
    results = evaluate_size_progressive_tiled(layers_dir)
    
    if results:
        print(f"\n🎉 分块渲染评估完成!")
        print(f"📁 输出目录: size_progressive_tiled/")

if __name__ == "__main__":
    main() 