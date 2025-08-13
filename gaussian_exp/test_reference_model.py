import os
import sys
import torch
import numpy as np
import argparse
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

def create_reference_model():
    """重新生成参考模型，使用与原始save_ply完全相同的逻辑"""
    print("🔧 重新生成参考模型")
    print("=" * 40)
    
    # 加载原始模型
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    # 使用与原始save_ply完全相同的逻辑
    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()
    
    print(f"📊 提取数据完成:")
    print(f"  XYZ: {xyz.shape}")
    print(f"  Normals: {normals.shape}")
    print(f"  F_DC: {f_dc.shape}")
    print(f"  F_Rest: {f_rest.shape}")
    print(f"  Opacity: {opacities.shape}")
    print(f"  Scale: {scale.shape}")
    print(f"  Rotation: {rotation.shape}")
    
    # 使用原始的construct_list_of_attributes逻辑
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    # 使用原始的数据组合逻辑 (注意顺序!)
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # 保存参考模型
    output_path = "./reference_model_exact_copy.ply"
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    
    print(f"✅ 参考模型保存: {output_path}")
    return output_path

def test_reference_vs_original():
    """测试参考模型vs原始模型"""
    print("\n🔍 测试参考模型 vs 原始模型")
    print("=" * 40)
    
    # 创建参考模型
    reference_path = create_reference_model()
    
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
            "path": "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
        },
        {
            "name": "参考重生模型",
            "path": reference_path
        }
    ]
    
    results = []
    
    for test_file in test_files:
        print(f"\n🧪 测试: {test_file['name']}")
        print(f"   文件: {test_file['path']}")
        
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
                "gaussian_count": gaussian_count
            })
            
            # 清理内存
            del gaussians, render_result, rendered_image, gt_image
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ 测试失败: {str(e)}")
            torch.cuda.empty_cache()
    
    # 对比分析
    if len(results) >= 2:
        print(f"\n📊 对比分析:")
        print("=" * 30)
        
        original_psnr = results[0]["psnr"]
        reference_psnr = results[1]["psnr"]
        psnr_diff = reference_psnr - original_psnr
        
        print(f"🔹 原始模型 PSNR: {original_psnr:.3f}dB")
        print(f"🔹 参考模型 PSNR: {reference_psnr:.3f}dB")
        print(f"🔹 PSNR差异: {psnr_diff:.6f}dB")
        
        if abs(psnr_diff) < 0.001:
            print(f"✅ 数据完全一致！可以使用这个方法修复分层文件")
        else:
            print(f"❌ 还是有差异，需要进一步调查")
    
    return results

def main():
    test_reference_vs_original()

if __name__ == "__main__":
    main() 