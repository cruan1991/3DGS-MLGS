import os
import sys
import torch
import numpy as np
import argparse

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

def load_test_camera(colmap_path, images_path, camera_name="000001.jpg", resolution_scale=8.0):
    """加载测试相机"""
    from PIL import Image
    
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

def test_original_vs_regenerated():
    """测试原始模型vs重新生成的完整模型"""
    print("🔍 测试原始模型 vs 重新生成的完整模型")
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
    camera = load_test_camera(colmap_path, images_path, "000001.jpg", 8.0)
    
    if camera is None:
        return
    
    print(f"📷 测试相机: 000001.jpg (分辨率: {camera.image_width}x{camera.image_height})")
    
    # 检查SPARSE_ADAM_AVAILABLE
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
    except:
        SPARSE_ADAM_AVAILABLE = False
    
    # 测试文件列表
    test_files = [
        {
            "name": "原始训练模型",
            "path": "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply",
            "expected_psnr": "~28dB (scaled to 8x)"
        },
        {
            "name": "重新生成的完整模型",
            "path": "./size_based_layers/size_progressive_S0_S1_S2_S3_S4_2046811balls.ply",
            "expected_psnr": "应该和原始模型一致"
        }
    ]
    
    results = []
    
    for test_file in test_files:
        print(f"\n🧪 测试: {test_file['name']}")
        print(f"   文件: {test_file['path']}")
        print(f"   预期: {test_file['expected_psnr']}")
        
        if not os.path.exists(test_file["path"]):
            print(f"   ❌ 文件不存在")
            continue
        
        try:
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 加载高斯球
            gaussians = GaussianModel(3)
            gaussians.load_ply(test_file["path"], use_train_test_exp=False)
            
            gaussian_count = gaussians.get_xyz.shape[0]
            print(f"   📊 高斯球数: {gaussian_count:,}")
            
            # 渲染
            render_result = render(camera, gaussians, pipe, background, 1., 
                                 SPARSE_ADAM_AVAILABLE, None, False)
            rendered_image = torch.clamp(render_result["render"], 0.0, 1.0)
            
            # GT图像
            gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
            
            # 计算指标
            psnr_val = psnr(rendered_image, gt_image).mean().item()
            l1_val = l1_loss(rendered_image, gt_image).mean().item()
            
            print(f"   ✅ PSNR: {psnr_val:.3f}dB")
            print(f"   📏 L1 Loss: {l1_val:.6f}")
            
            results.append({
                "name": test_file["name"],
                "psnr": psnr_val,
                "l1_loss": l1_val,
                "gaussian_count": gaussian_count,
                "path": test_file["path"]
            })
            
            # 清理内存
            del gaussians, render_result, rendered_image, gt_image
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ 测试失败: {str(e)}")
            torch.cuda.empty_cache()
    
    # 对比分析
    print(f"\n📊 对比分析:")
    print("=" * 40)
    
    if len(results) >= 2:
        original_psnr = results[0]["psnr"]
        regenerated_psnr = results[1]["psnr"]
        psnr_diff = regenerated_psnr - original_psnr
        
        print(f"🔹 原始模型 PSNR: {original_psnr:.3f}dB ({results[0]['gaussian_count']:,}球)")
        print(f"🔹 重生模型 PSNR: {regenerated_psnr:.3f}dB ({results[1]['gaussian_count']:,}球)")
        print(f"🔹 PSNR差异: {psnr_diff:.3f}dB")
        
        if abs(psnr_diff) < 0.1:
            print(f"✅ 数据完整性: 良好 (差异 < 0.1dB)")
        elif abs(psnr_diff) < 1.0:
            print(f"⚠️ 数据完整性: 可接受 (差异 < 1.0dB)")
        else:
            print(f"❌ 数据完整性: 有问题 (差异 > 1.0dB)")
            print(f"   可能原因:")
            print(f"   1. PLY文件生成过程中数据丢失")
            print(f"   2. 高斯球参数格式不一致")
            print(f"   3. 球的数量不匹配")
        
        # 检查球数是否一致
        if results[0]["gaussian_count"] != results[1]["gaussian_count"]:
            count_diff = results[1]["gaussian_count"] - results[0]["gaussian_count"]
            print(f"⚠️ 高斯球数量不一致: 差异 {count_diff:,}球")
    
    else:
        print("❌ 无法进行对比，缺少测试文件")
    
    return results

if __name__ == "__main__":
    results = test_original_vs_regenerated() 