import os
import sys
import torch
import numpy as np
from PIL import Image
import argparse
import json

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import Scene, GaussianModel
from utils.general_utils import PILtoTorch
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import focal2fov
from scene.cameras import Camera
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render

def psnr(img1, img2):
    """按照train.py的PSNR计算"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def verify_colmap_loading():
    """验证COLMAP数据加载的正确性"""
    print("🔍 验证COLMAP数据加载...")
    
    colmap_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/sparse/0"
    cameras_bin = os.path.join(colmap_path, 'cameras.bin')
    images_bin = os.path.join(colmap_path, 'images.bin')
    
    # 加载COLMAP数据
    cam_intrinsics = read_intrinsics_binary(cameras_bin)
    cam_extrinsics = read_extrinsics_binary(images_bin)
    
    print(f"✅ 相机内参数量: {len(cam_intrinsics)}")
    print(f"✅ 相机外参数量: {len(cam_extrinsics)}")
    
    # 检查第一个相机的详细参数
    first_img_id = list(cam_extrinsics.keys())[0]
    first_img = cam_extrinsics[first_img_id]
    first_cam = cam_intrinsics[first_img.camera_id]
    
    print(f"\n📷 第一个相机详细信息:")
    print(f"  图片ID: {first_img_id}")
    print(f"  图片名: {first_img.name}")
    print(f"  相机模型: {first_cam.model}")
    print(f"  分辨率: {first_cam.width}x{first_cam.height}")
    print(f"  内参: {first_cam.params}")
    print(f"  四元数: {first_img.qvec}")
    print(f"  平移: {first_img.tvec}")
    
    return cam_intrinsics, cam_extrinsics

def verify_camera_construction(cam_intrinsics, cam_extrinsics, resolution_scale=2.0):
    """验证Camera对象构造的正确性"""
    print(f"\n🏗️ 验证Camera对象构造 (resolution_scale={resolution_scale})...")
    
    images_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck/images"
    
    # 取第一个相机进行详细验证
    first_img_id = list(cam_extrinsics.keys())[0]
    img_info = cam_extrinsics[first_img_id]
    intrinsic = cam_intrinsics[img_info.camera_id]
    
    # 解析内参
    fx, fy, cx, cy = intrinsic.params
    
    # 应用分辨率缩放
    width = int(intrinsic.width / resolution_scale)
    height = int(intrinsic.height / resolution_scale)
    fx_scaled = fx / resolution_scale
    fy_scaled = fy / resolution_scale
    cx_scaled = cx / resolution_scale
    cy_scaled = cy / resolution_scale
    
    print(f"📐 原始分辨率: {intrinsic.width}x{intrinsic.height}")
    print(f"📐 缩放后分辨率: {width}x{height}")
    print(f"📐 原始焦距: fx={fx:.2f}, fy={fy:.2f}")
    print(f"📐 缩放后焦距: fx={fx_scaled:.2f}, fy={fy_scaled:.2f}")
    
    # 计算FoV
    FoVx = focal2fov(fx_scaled, width)
    FoVy = focal2fov(fy_scaled, height)
    
    print(f"🔍 FoVx: {np.degrees(FoVx):.3f}°")
    print(f"🔍 FoVy: {np.degrees(FoVy):.3f}°")
    
    # 外参
    R = np.transpose(qvec2rotmat(img_info.qvec))
    T = np.array(img_info.tvec)
    
    print(f"🔄 旋转矩阵R形状: {R.shape}")
    print(f"🔄 平移向量T: {T}")
    
    # 加载图像
    image_path = os.path.join(images_path, img_info.name)
    print(f"📸 加载图像: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None
    
    image = Image.open(image_path)
    original_size = image.size
    print(f"📸 原始图像尺寸: {original_size}")
    
    # 调整图像尺寸
    if resolution_scale != 1.0:
        new_size = (width, height)
        image = image.resize(new_size, Image.LANCZOS)
        print(f"📸 调整后图像尺寸: {image.size}")
    
    # 转换为tensor
    im_data = PILtoTorch(image, (width, height))
    print(f"📸 Tensor形状: {im_data.shape}")
    print(f"📸 Tensor数值范围: [{im_data.min():.3f}, {im_data.max():.3f}]")
    
    # 创建Camera对象
    camera = Camera(
        resolution=(width, height),
        colmap_id=first_img_id,
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
    
    print(f"✅ Camera对象创建成功")
    print(f"   - image_width: {camera.image_width}")
    print(f"   - image_height: {camera.image_height}")
    print(f"   - FoVx: {np.degrees(camera.FoVx):.3f}°")
    print(f"   - FoVy: {np.degrees(camera.FoVy):.3f}°")
    print(f"   - original_image shape: {camera.original_image.shape}")
    
    return camera

def verify_gaussian_loading(ply_path):
    """验证高斯模型加载"""
    print(f"\n🎯 验证高斯模型加载...")
    print(f"PLY路径: {ply_path}")
    
    if not os.path.exists(ply_path):
        print(f"❌ PLY文件不存在: {ply_path}")
        return None
    
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    print(f"✅ 加载了 {gaussians.get_xyz.shape[0]} 个高斯球")
    print(f"   - 位置范围: x[{gaussians.get_xyz[:, 0].min():.3f}, {gaussians.get_xyz[:, 0].max():.3f}]")
    print(f"   - 位置范围: y[{gaussians.get_xyz[:, 1].min():.3f}, {gaussians.get_xyz[:, 1].max():.3f}]")
    print(f"   - 位置范围: z[{gaussians.get_xyz[:, 2].min():.3f}, {gaussians.get_xyz[:, 2].max():.3f}]")
    print(f"   - 透明度范围: [{gaussians.get_opacity.min():.3f}, {gaussians.get_opacity.max():.3f}]")
    
    return gaussians

def verify_rendering_parameters():
    """验证渲染参数设置"""
    print(f"\n⚙️ 验证渲染参数...")
    
    # 检查SPARSE_ADAM_AVAILABLE
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        SPARSE_ADAM_AVAILABLE = True
        print(f"✅ SparseGaussianAdam 可用")
    except:
        SPARSE_ADAM_AVAILABLE = False
        print(f"❌ SparseGaussianAdam 不可用")
    
    # Pipeline参数
    parser = argparse.ArgumentParser()
    pipe_parser = PipelineParams(parser)
    args = parser.parse_args([])
    pipe = pipe_parser.extract(args)
    
    print(f"✅ Pipeline参数:")
    print(f"   - convert_SHs_python: {pipe.convert_SHs_python}")
    print(f"   - compute_cov3D_python: {pipe.compute_cov3D_python}")
    print(f"   - debug: {pipe.debug}")
    
    # 背景设置
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    print(f"✅ 背景颜色: {background}")
    
    # 渲染参数
    renderArgs = (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, False)
    print(f"✅ 渲染参数: scaling_modifier=1.0, separate_sh={SPARSE_ADAM_AVAILABLE}, train_test_exp=False")
    
    return pipe, background, renderArgs

def verify_single_render(camera, gaussians, renderArgs):
    """验证单次渲染过程"""
    print(f"\n🎨 验证单次渲染...")
    
    # 执行渲染
    print(f"📸 渲染相机: {camera.image_name}")
    render_result = render(camera, gaussians, *renderArgs)
    rendered_image = torch.clamp(render_result["render"], 0.0, 1.0)
    
    print(f"✅ 渲染成功")
    print(f"   - 渲染图像形状: {rendered_image.shape}")
    print(f"   - 渲染图像数值范围: [{rendered_image.min():.3f}, {rendered_image.max():.3f}]")
    
    # GT图像
    gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
    print(f"   - GT图像形状: {gt_image.shape}")
    print(f"   - GT图像数值范围: [{gt_image.min():.3f}, {gt_image.max():.3f}]")
    
    # 检查尺寸匹配
    if rendered_image.shape != gt_image.shape:
        print(f"❌ 尺寸不匹配!")
        print(f"   渲染: {rendered_image.shape}")
        print(f"   GT: {gt_image.shape}")
        return None, None
    
    return rendered_image, gt_image

def verify_psnr_calculation(rendered_image, gt_image):
    """验证PSNR计算"""
    print(f"\n📊 验证PSNR计算...")
    
    # 方法1: 使用自定义psnr函数
    psnr1 = psnr(rendered_image, gt_image).mean().item()
    print(f"✅ 方法1 (自定义psnr): {psnr1:.3f} dB")
    
    # 方法2: 尝试使用utils.image_utils.psnr
    try:
        from utils.image_utils import psnr as train_psnr
        psnr2 = train_psnr(rendered_image, gt_image)
        print(f"✅ 方法2 (utils.image_utils.psnr): {psnr2:.3f} dB")
    except Exception as e:
        print(f"❌ 方法2失败: {e}")
        psnr2 = None
    
    # 方法3: 手动计算
    mse = torch.mean((rendered_image - gt_image) ** 2)
    psnr3 = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    print(f"✅ 方法3 (手动计算): {psnr3:.3f} dB")
    
    # L1 loss
    l1 = l1_loss(rendered_image, gt_image).mean().item()
    print(f"✅ L1 Loss: {l1:.6f}")
    
    return psnr1, l1

def save_verification_images(rendered_image, gt_image, camera_name):
    """保存验证图像"""
    print(f"\n💾 保存验证图像...")
    
    os.makedirs("verification_output", exist_ok=True)
    
    # 保存渲染图像
    from torchvision.utils import save_image
    save_image(rendered_image, f"verification_output/{camera_name}_render.png")
    save_image(gt_image, f"verification_output/{camera_name}_gt.png")
    
    # 保存差异图
    diff = torch.abs(rendered_image - gt_image)
    save_image(diff, f"verification_output/{camera_name}_diff.png")
    
    print(f"✅ 图像已保存到 verification_output/")

def main():
    print("🔍 3DGS渲染Pipeline完整验证\n" + "="*50)
    
    # 参数
    model_path = "./output/truck-150w"
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    
    # 1. 验证COLMAP加载
    cam_intrinsics, cam_extrinsics = verify_colmap_loading()
    
    # 2. 验证Camera构造
    camera = verify_camera_construction(cam_intrinsics, cam_extrinsics, resolution_scale=2.0)
    if camera is None:
        print("❌ Camera构造失败，终止验证")
        return
    
    # 3. 验证高斯模型加载
    gaussians = verify_gaussian_loading(ply_path)
    if gaussians is None:
        print("❌ 高斯模型加载失败，终止验证")
        return
    
    # 4. 验证渲染参数
    pipe, background, renderArgs = verify_rendering_parameters()
    
    # 5. 验证单次渲染
    rendered_image, gt_image = verify_single_render(camera, gaussians, renderArgs)
    if rendered_image is None:
        print("❌ 渲染失败，终止验证")
        return
    
    # 6. 验证PSNR计算
    psnr_value, l1_value = verify_psnr_calculation(rendered_image, gt_image)
    
    # 7. 保存验证图像
    save_verification_images(rendered_image, gt_image, camera.image_name.replace('.jpg', ''))
    
    print(f"\n🎉 验证完成!")
    print(f"📊 最终结果:")
    print(f"   - PSNR: {psnr_value:.3f} dB")
    print(f"   - L1 Loss: {l1_value:.6f}")
    print(f"   - 相机: {camera.image_name}")

if __name__ == "__main__":
    main() 