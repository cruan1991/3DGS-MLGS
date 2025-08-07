import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")
from gaussian_renderer import render
import argparse
import os
import json
from scene.gaussian_model import GaussianModel
from utils.camera_utils import Camera
import torch
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import math
from utils.image_utils import cal_psnr

# Set CUDA device
torch.cuda.set_device(1)  # Use GPU 1 instead of GPU 0

class DummyArgs:
    def __init__(self):
        self.data_device = 'cuda'
        self.train_test_exp = None
        self.resolution = 1
        self.sh_degree = 3  # 添加这个参数
        self.optimizer_type = "default"  # 添加这个参数

class DummyRenderPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.antialiasing = False
        self.debug = False

# 添加一个简化的OptimizationParams类
class DummyOptimizationParams:
    def __init__(self):
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        self.optimizer_type = "default"

def load_camera_data(config_path, images_dir):
    print(f"Loading camera data from {config_path}")
    with open(config_path, 'r') as f:
        camera_data = json.load(f)
    print(f"Found {len(camera_data)} cameras")

    cameras = []
    for cam in camera_data:
        # Validate required fields
        required_fields = ['width', 'height', 'rotation', 'position', 'img_name']
        if not all(field in cam for field in required_fields):
            raise ValueError(f"Missing required fields in camera data. Required: {required_fields}")

        # Load and process the image using the img_name from camera data
        img_path = os.path.join(images_dir, cam['img_name'])
        print(f"Loading image: {img_path}")
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        image = Image.open(img_path).convert('RGB')
        if image.size != (cam['width'], cam['height']):
            print(f"Resizing image from {image.size} to {(cam['width'], cam['height'])}")
            image = image.resize((cam['width'], cam['height']))

        # ===== 关键修复: 严格按照train.py的图像处理方式 =====
        # Convert image to tensor - 确保和train.py完全一致
        image_tensor = TF.to_tensor(image)
        # 注意：先不要.cuda()，让Camera类自己处理设备
        
        # ===== 关键修复: 旋转矩阵处理 - 直接使用numpy =====
        # 不做任何转置，直接使用JSON中的旋转矩阵
        R = np.array(cam['rotation'], dtype=np.float32)
        T = np.array(cam['position'], dtype=np.float32)
        
        print(f"Camera {cam.get('id', 0)} R shape: {R.shape}, T shape: {T.shape}")
        
        # ===== 关键修复: 使用真实的focal length计算FOV =====
        fx = cam.get('fx', 1565.6)
        fy = cam.get('fy', 872.8)
        
        # 计算FOV - 使用标准公式
        FoVx = 2.0 * math.atan(cam['width'] / (2.0 * fx))
        FoVy = 2.0 * math.atan(cam['height'] / (2.0 * fy))
        
        print(f"Camera {cam.get('id', 0)}: fx={fx}, fy={fy}, FoVx={math.degrees(FoVx):.1f}°, FoVy={math.degrees(FoVy):.1f}°")
        
        # ===== 关键修复: 使用和train.py完全一致的Camera构造 =====
        camera = Camera(
            resolution=(cam['width'], cam['height']),
            colmap_id=cam.get('id', 0),
            R=R,  # 直接传numpy数组
            T=T,  # 直接传numpy数组
            FoVx=FoVx,
            FoVy=FoVy,
            depth_params={"scale": 1.0, "offset": 0.0, "med_scale": 1.0},
            image=image_tensor,
            invdepthmap=None,
            image_name=cam['img_name'],
            uid=cam.get('id', 0),
            data_device="cuda"  # 让Camera类处理GPU转换
        )
        cameras.append(camera)
        print(f"Added camera {len(cameras)} with image {cam['img_name']}")

    if not cameras:
        raise ValueError("No valid cameras loaded!")

    print(f"Successfully loaded {len(cameras)} cameras")
    return cameras

def load_camera_data_from_colmap(sparse_dir, images_dir):
    """直接从COLMAP数据加载相机（和训练时完全一致）"""
    print(f"Loading COLMAP data from {sparse_dir}")
    
    # 导入COLMAP加载器
    sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text
    from scene.colmap_loader import qvec2rotmat
    
    # 尝试读取二进制格式
    try:
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
        
        if os.path.exists(cameras_intrinsic_file):
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            print("Loaded binary COLMAP data")
        else:
            # 尝试文本格式
            cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.txt")
            cameras_extrinsic_file = os.path.join(sparse_dir, "images.txt")
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            print("Loaded text COLMAP data")
            
    except Exception as e:
        print(f"Failed to load COLMAP data: {e}")
        return None
    
    cameras = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write(f"Loading camera {idx+1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        # 获取图像路径
        image_path = os.path.join(images_dir, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            print(f"\nWarning: Image not found: {image_path}")
            continue
            
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = TF.to_tensor(image)
        
        # ===== 使用和train.py完全一致的坐标系处理 =====
        # 旋转：从四元数转换为旋转矩阵
        R = qvec2rotmat(-extr.qvec).astype(np.float32)  # 注意这里的负号！
        T = np.array(extr.tvec, dtype=np.float32)
        
        # 计算FOV（和train.py保持一致）
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
        elif intr.model in ["PINHOLE", "OPENCV"]:
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
        else:
            print(f"\nWarning: Unknown camera model {intr.model}")
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0] if len(intr.params) == 1 else intr.params[1]
        
        FovY = 2.0 * math.atan(intr.height / (2.0 * focal_length_y))
        FovX = 2.0 * math.atan(intr.width / (2.0 * focal_length_x))
        
        # 创建相机对象
        camera = Camera(
            resolution=(intr.width, intr.height),
            colmap_id=intr.id,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params={"scale": 1.0, "offset": 0.0, "med_scale": 1.0},
            image=image_tensor,
            invdepthmap=None,
            image_name=os.path.basename(extr.name),
            uid=idx,
            data_device="cuda"
        )
        cameras.append(camera)
    
    print(f"\nSuccessfully loaded {len(cameras)} cameras from COLMAP data")
    return cameras

def tensor_to_image(tensor):
    # Convert tensor to PIL Image
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert from CxHxW to HxWxC
    tensor = tensor.permute(1, 2, 0)
    
    # Scale values to [0, 255] and convert to uint8
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    
    return Image.fromarray(tensor.numpy())

def evaluate_ply(ply_path, config_path, images_dir, output_dir):
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Loading filtered .ply: {ply_path}")
    
    # ===== 尝试方案1: 不使用训练设置 =====
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    
    print(f"Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    # 如果上面的方法失败，则使用备用方案
    try:
        # 测试是否能正常访问高斯参数
        _ = gaussians.get_xyz
        _ = gaussians.get_features
        _ = gaussians.get_opacity
        _ = gaussians.get_scaling
        _ = gaussians.get_rotation
        print("✅ Gaussian parameters accessible without training setup")
    except Exception as e:
        print(f"⚠️ Need training setup due to: {e}")
        # 备用方案：使用最小的训练设置
        opt_args = DummyOptimizationParams()
        gaussians.training_setup(opt_args)
        print("✅ Applied minimal training setup")

    print(f"📸 Loading cameras from config: {config_path}")
    try:
        # ===== 智能选择相机数据加载方式 =====
        if config_path.endswith('.json'):
            cam_data = load_camera_data(config_path, images_dir)
        elif 'sparse' in config_path or config_path.endswith('.txt') or config_path.endswith('.bin'):
            # 如果是COLMAP数据目录或文件
            if os.path.isdir(config_path):
                # 检查是否有0子目录（COLMAP标准结构）
                sparse_dir = config_path
                if os.path.exists(os.path.join(config_path, "0")):
                    sparse_dir = os.path.join(config_path, "0")
            else:
                sparse_dir = os.path.dirname(config_path)
            cam_data = load_camera_data_from_colmap(sparse_dir, images_dir)
        else:
            # 默认尝试JSON格式
            cam_data = load_camera_data(config_path, images_dir)
            
        if cam_data is None:
            print("Failed to load camera data!")
            return
            
    except Exception as e:
        print(f"Error loading camera data: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"🎨 Rendering views...")
    all_metrics = []
    pipe = DummyRenderPipe()
    bg_color = torch.tensor([1.0, 1.0, 1.0], device='cuda')

    for idx, viewpoint in enumerate(cam_data):
        try:
            print(f"\nProcessing view {idx} ({viewpoint.image_name})")
            
            # ===== 调试相机信息 =====
            print(f"Camera position: {viewpoint.camera_center}")
            print(f"Image resolution: {viewpoint.image_width}x{viewpoint.image_height}")
            print(f"FOV: {math.degrees(viewpoint.FoVx):.1f}° x {math.degrees(viewpoint.FoVy):.1f}°")
            
            # ===== 完全按照train.py的渲染方式 =====
            try:
                # 先尝试最简单的渲染调用
                render_pkg = render(viewpoint, gaussians, pipe, bg_color)
                rendered_image = torch.clamp(render_pkg['render'], 0.0, 1.0)
                
                print(f"Render successful, output shape: {rendered_image.shape}")
                
            except Exception as render_error:
                print(f"First render attempt failed: {render_error}")
                
                # 尝试添加更多参数（和train.py保持一致）
                try:
                    render_pkg = render(
                        viewpoint, 
                        gaussians, 
                        pipe, 
                        bg_color,
                        use_trained_exp=False,  # 默认为False
                        separate_sh=False       # 默认为False
                    )
                    rendered_image = torch.clamp(render_pkg['render'], 0.0, 1.0)
                    print(f"Second render attempt successful, output shape: {rendered_image.shape}")
                    
                except Exception as render_error2:
                    print(f"Second render attempt failed: {render_error2}")
                    continue

            # Save rendered image
            rendered_pil = tensor_to_image(rendered_image)
            out_path = os.path.join(output_dir, viewpoint.image_name.replace('.jpg', '.png'))
            print(f"Saving rendered image to: {out_path}")
            rendered_pil.save(out_path)

            # ===== 调试GT图像处理 =====
            gt_image = torch.clamp(viewpoint.original_image.to(rendered_image.device), 0.0, 1.0)
            print(f"GT image shape: {gt_image.shape}, Rendered shape: {rendered_image.shape}")
            
            # 确保尺寸匹配
            if gt_image.shape != rendered_image.shape:
                print(f"WARNING: Shape mismatch! GT: {gt_image.shape}, Rendered: {rendered_image.shape}")
                # 如果尺寸不匹配，跳过这个视图
                continue
            
            # ===== PSNR计算（添加调试信息）=====
            rendered_batch = rendered_image.unsqueeze(0)
            gt_batch = gt_image.unsqueeze(0)
            
            psnr = cal_psnr(rendered_batch, gt_batch)
            psnr_value = psnr.mean().item()
            all_metrics.append(psnr_value)

            print(f"View {idx}: PSNR = {psnr_value:.2f} dB")
            
            # ===== 额外调试：保存前几个视图的GT图像用于对比 =====
            if idx < 3:
                gt_pil = tensor_to_image(gt_image)
                gt_out_path = os.path.join(output_dir, f"GT_{viewpoint.image_name.replace('.jpg', '.png')}")
                gt_pil.save(gt_out_path)
                print(f"Also saved GT image to: {gt_out_path}")

        except Exception as e:
            print(f"Error processing view {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_metrics:
        print("No views were successfully processed!")
        return

    avg_psnr = sum(all_metrics) / len(all_metrics)
    print(f"✅ Done. Avg PSNR: {avg_psnr:.2f} dB")

    # Save metrics
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['View', 'PSNR'])
        for idx, psnr in enumerate(all_metrics):
            writer.writerow([idx, psnr])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply-path', required=True, help='Path to the .ply file')
    parser.add_argument('--config-path', required=True, help='Path to the camera config JSON file')
    parser.add_argument('--images-dir', required=True, help='Path to the directory containing ground truth images')
    parser.add_argument('--output-dir', required=True, help='Directory to save rendered images and metrics')
    args = parser.parse_args()

    evaluate_ply(args.ply_path, args.config_path, args.images_dir, args.output_dir)

if __name__ == '__main__':
    main()