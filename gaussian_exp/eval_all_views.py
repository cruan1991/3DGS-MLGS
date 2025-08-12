#!/usr/bin/env python3
"""
Complete evaluation script for all camera views
找出33.83 dB PSNR的真正来源
"""

import torch
import os
import numpy as np
from argparse import ArgumentParser
from plyfile import PlyData
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from utils.general_utils import PILtoTorch
from PIL import Image
import json

def psnr(img1, img2):
    """PSNR calculation identical to train.py"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_cameras_from_colmap(sparse_dir, images_dir, resolution_scale=1.0):
    """从COLMAP加载所有相机"""
    try:
        cameras = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
        images = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    except:
        print("Failed to read binary files, the folder structure is probably wrong.")
        return []

    camera_list = []
    
    for idx, (img_id, img_info) in enumerate(images.items()):
        # Get camera intrinsics
        cam_id = img_info.camera_id
        if cam_id not in cameras:
            continue
            
        intrinsics = cameras[cam_id]
        
        # Image dimensions
        width = int(intrinsics.width / resolution_scale)
        height = int(intrinsics.height / resolution_scale)
        
        # Camera parameters
        if intrinsics.model == "PINHOLE":
            fx, fy, cx, cy = intrinsics.params
            fx /= resolution_scale
            fy /= resolution_scale
            cx /= resolution_scale
            cy /= resolution_scale
        else:
            print(f"Unsupported camera model: {intrinsics.model}")
            continue
            
        # Convert quaternion to rotation matrix
        qvec = img_info.qvec
        tvec = img_info.tvec
        R = qvec2rotmat(qvec).T  # Transpose for COLMAP convention
        T = np.array(tvec)
        
        # Calculate FoV
        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)
        
        # Load image
        image_path = os.path.join(images_dir, img_info.name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        try:
            image = Image.open(image_path)
            if resolution_scale != 1.0:
                image = image.resize((width, height), Image.LANCZOS)
            image = PILtoTorch(image, (width, height))
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue
            
        # Create camera
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
        
        camera_list.append(camera)
        
    print(f"Loaded {len(camera_list)} cameras from COLMAP")
    return camera_list

def evaluate_all_cameras(cameras, gaussians, pipe, background):
    """评估所有相机视角"""
    results = []
    total_psnr = 0.0
    
    print(f"\n🎨 Evaluating all {len(cameras)} camera views...")
    
    for idx, camera in enumerate(cameras):
        try:
            # Render
            render_pkg = render(camera, gaussians, pipe, background, 
                              sparse_adam=False, train_test_exp=False)
            image = render_pkg["render"]
            
            # Get ground truth
            gt_image = camera.original_image.cuda()
            
            # Ensure same shape
            if image.shape != gt_image.shape:
                print(f"Camera {idx}: Shape mismatch - render: {image.shape}, gt: {gt_image.shape}")
                continue
                
            # Calculate PSNR
            current_psnr = psnr(image, gt_image).mean().double()
            total_psnr += current_psnr.item()
            
            result = {
                'camera_idx': idx,
                'image_name': camera.image_name,
                'colmap_id': camera.colmap_id,
                'psnr': current_psnr.item(),
                'resolution': f"{camera.original_image.shape[2]}x{camera.original_image.shape[1]}"
            }
            results.append(result)
            
            if idx % 10 == 0 or current_psnr.item() > 30.0:
                print(f"Camera {idx:3d} ({camera.image_name:15s}): PSNR = {current_psnr.item():.3f} dB")
                
        except Exception as e:
            print(f"Error evaluating camera {idx} ({camera.image_name}): {e}")
            continue
    
    if results:
        avg_psnr = total_psnr / len(results)
        print(f"\n📊 Evaluation Results:")
        print(f"   Total cameras evaluated: {len(results)}")
        print(f"   Average PSNR: {avg_psnr:.3f} dB")
        
        # Find best and worst cameras
        best_result = max(results, key=lambda x: x['psnr'])
        worst_result = min(results, key=lambda x: x['psnr'])
        
        print(f"   Best camera: {best_result['image_name']} - {best_result['psnr']:.3f} dB")
        print(f"   Worst camera: {worst_result['image_name']} - {worst_result['psnr']:.3f} dB")
        
        # Show distribution
        psnr_values = [r['psnr'] for r in results]
        print(f"   PSNR distribution:")
        print(f"     Min: {min(psnr_values):.3f} dB")
        print(f"     25th percentile: {np.percentile(psnr_values, 25):.3f} dB")
        print(f"     Median: {np.percentile(psnr_values, 50):.3f} dB") 
        print(f"     75th percentile: {np.percentile(psnr_values, 75):.3f} dB")
        print(f"     Max: {max(psnr_values):.3f} dB")
        
        # Find cameras with PSNR > 30
        high_psnr_cameras = [r for r in results if r['psnr'] > 30.0]
        if high_psnr_cameras:
            print(f"\n🔥 High PSNR cameras (>30 dB): {len(high_psnr_cameras)} found")
            for r in sorted(high_psnr_cameras, key=lambda x: x['psnr'], reverse=True)[:10]:
                print(f"     {r['image_name']}: {r['psnr']:.3f} dB")
        
        # Find cameras with PSNR > 33
        ultra_high_psnr = [r for r in results if r['psnr'] > 33.0]
        if ultra_high_psnr:
            print(f"\n🚀 Ultra-high PSNR cameras (>33 dB): {len(ultra_high_psnr)} found")
            for r in sorted(ultra_high_psnr, key=lambda x: x['psnr'], reverse=True):
                print(f"     {r['image_name']}: {r['psnr']:.3f} dB ⭐")
        else:
            print(f"\n❓ No cameras found with PSNR > 33 dB")
            print(f"   The 33.83 dB might come from:")
            print(f"   1. Different subset of cameras during training")
            print(f"   2. Different rendering parameters")
            print(f"   3. Different evaluation timing/iteration")
    
    return results

def save_detailed_results(results, output_path):
    """保存详细结果到JSON文件"""
    if not results:
        return
        
    # Prepare data for JSON
    json_data = {
        'total_cameras': len(results),
        'average_psnr': sum(r['psnr'] for r in results) / len(results),
        'camera_results': results
    }
    
    # Add statistics
    psnr_values = [r['psnr'] for r in results]
    json_data['statistics'] = {
        'min_psnr': min(psnr_values),
        'max_psnr': max(psnr_values),
        'median_psnr': float(np.median(psnr_values)),
        'std_psnr': float(np.std(psnr_values)),
        'cameras_above_30': len([p for p in psnr_values if p > 30.0]),
        'cameras_above_33': len([p for p in psnr_values if p > 33.0])
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"💾 Detailed results saved to: {output_path}")

def main():
    parser = ArgumentParser(description="Evaluate all camera views")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--ply-path", type=str, help="Path to PLY file (default: model-path/gaussian_ball.ply)")
    parser.add_argument("--resolution-scale", type=float, default=2.0, help="Resolution scale factor")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    model_path = args.model_path
    ply_path = args.ply_path or os.path.join(model_path, "gaussian_ball.ply")
    
    if not os.path.exists(ply_path):
        # Try the best iteration path
        ply_path = os.path.join(model_path, "gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply")
    
    if not os.path.exists(ply_path):
        print(f"❌ PLY file not found: {ply_path}")
        return
    
    print(f"🚀 Starting complete evaluation")
    print(f"   Model path: {model_path}")
    print(f"   PLY path: {ply_path}")
    print(f"   Resolution scale: {args.resolution_scale}")
    
    # Load pipeline parameters using dummy parser
    from argparse import ArgumentParser
    dummy_parser = ArgumentParser()
    
    # Initialize ModelParams with parser
    dataset_params = ModelParams(dummy_parser)
    dataset = dataset_params.extract(type('Args', (), {
        'source_path': model_path,
        'model_path': model_path,
        'images': 'images',
        'resolution': -1,
        'white_background': False,
        'data_device': 'cuda',
        'train_test_exp': False,
        'sh_degree': 3,
        'depths': '',
        'eval': False
    })())
    
    # Initialize PipelineParams
    pipe_params = PipelineParams(dummy_parser)
    pipe = pipe_params.extract(type('Args', (), {
        'convert_SHs_python': False,
        'compute_cov3D_python': False,
        'debug': False
    })())
    
    # Load Gaussians
    print(f"📦 Loading Gaussians from {ply_path}")
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(ply_path, train_test_exp=False)
    print(f"   Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    # Load cameras
    sparse_dir = os.path.join(model_path, "sparse/0")
    images_dir = os.path.join(model_path, "images")
    
    print(f"📷 Loading cameras from COLMAP")
    print(f"   Sparse dir: {sparse_dir}")
    print(f"   Images dir: {images_dir}")
    
    cameras = load_cameras_from_colmap(sparse_dir, images_dir, args.resolution_scale)
    
    if not cameras:
        print("❌ No cameras loaded!")
        return
    
    # Setup rendering
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Evaluate all cameras
    results = evaluate_all_cameras(cameras, gaussians, pipe, background)
    
    # Save results if requested
    if args.output:
        save_detailed_results(results, args.output)
    elif results:
        # Auto-save with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"all_views_eval_{timestamp}.json"
        save_detailed_results(results, output_file)

if __name__ == "__main__":
    main() 
"""
Complete evaluation script for all camera views
找出33.83 dB PSNR的真正来源
"""

import torch
import os
import numpy as np
from argparse import ArgumentParser
from plyfile import PlyData
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from utils.general_utils import PILtoTorch
from PIL import Image
import json

def psnr(img1, img2):
    """PSNR calculation identical to train.py"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_cameras_from_colmap(sparse_dir, images_dir, resolution_scale=1.0):
    """从COLMAP加载所有相机"""
    try:
        cameras = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
        images = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    except:
        print("Failed to read binary files, the folder structure is probably wrong.")
        return []

    camera_list = []
    
    for idx, (img_id, img_info) in enumerate(images.items()):
        # Get camera intrinsics
        cam_id = img_info.camera_id
        if cam_id not in cameras:
            continue
            
        intrinsics = cameras[cam_id]
        
        # Image dimensions
        width = int(intrinsics.width / resolution_scale)
        height = int(intrinsics.height / resolution_scale)
        
        # Camera parameters
        if intrinsics.model == "PINHOLE":
            fx, fy, cx, cy = intrinsics.params
            fx /= resolution_scale
            fy /= resolution_scale
            cx /= resolution_scale
            cy /= resolution_scale
        else:
            print(f"Unsupported camera model: {intrinsics.model}")
            continue
            
        # Convert quaternion to rotation matrix
        qvec = img_info.qvec
        tvec = img_info.tvec
        R = qvec2rotmat(qvec).T  # Transpose for COLMAP convention
        T = np.array(tvec)
        
        # Calculate FoV
        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)
        
        # Load image
        image_path = os.path.join(images_dir, img_info.name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        try:
            image = Image.open(image_path)
            if resolution_scale != 1.0:
                image = image.resize((width, height), Image.LANCZOS)
            image = PILtoTorch(image, (width, height))
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue
            
        # Create camera
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
        
        camera_list.append(camera)
        
    print(f"Loaded {len(camera_list)} cameras from COLMAP")
    return camera_list

def evaluate_all_cameras(cameras, gaussians, pipe, background):
    """评估所有相机视角"""
    results = []
    total_psnr = 0.0
    
    print(f"\n🎨 Evaluating all {len(cameras)} camera views...")
    
    for idx, camera in enumerate(cameras):
        try:
            # Render
            render_pkg = render(camera, gaussians, pipe, background, 
                              sparse_adam=False, train_test_exp=False)
            image = render_pkg["render"]
            
            # Get ground truth
            gt_image = camera.original_image.cuda()
            
            # Ensure same shape
            if image.shape != gt_image.shape:
                print(f"Camera {idx}: Shape mismatch - render: {image.shape}, gt: {gt_image.shape}")
                continue
                
            # Calculate PSNR
            current_psnr = psnr(image, gt_image).mean().double()
            total_psnr += current_psnr.item()
            
            result = {
                'camera_idx': idx,
                'image_name': camera.image_name,
                'colmap_id': camera.colmap_id,
                'psnr': current_psnr.item(),
                'resolution': f"{camera.original_image.shape[2]}x{camera.original_image.shape[1]}"
            }
            results.append(result)
            
            if idx % 10 == 0 or current_psnr.item() > 30.0:
                print(f"Camera {idx:3d} ({camera.image_name:15s}): PSNR = {current_psnr.item():.3f} dB")
                
        except Exception as e:
            print(f"Error evaluating camera {idx} ({camera.image_name}): {e}")
            continue
    
    if results:
        avg_psnr = total_psnr / len(results)
        print(f"\n📊 Evaluation Results:")
        print(f"   Total cameras evaluated: {len(results)}")
        print(f"   Average PSNR: {avg_psnr:.3f} dB")
        
        # Find best and worst cameras
        best_result = max(results, key=lambda x: x['psnr'])
        worst_result = min(results, key=lambda x: x['psnr'])
        
        print(f"   Best camera: {best_result['image_name']} - {best_result['psnr']:.3f} dB")
        print(f"   Worst camera: {worst_result['image_name']} - {worst_result['psnr']:.3f} dB")
        
        # Show distribution
        psnr_values = [r['psnr'] for r in results]
        print(f"   PSNR distribution:")
        print(f"     Min: {min(psnr_values):.3f} dB")
        print(f"     25th percentile: {np.percentile(psnr_values, 25):.3f} dB")
        print(f"     Median: {np.percentile(psnr_values, 50):.3f} dB") 
        print(f"     75th percentile: {np.percentile(psnr_values, 75):.3f} dB")
        print(f"     Max: {max(psnr_values):.3f} dB")
        
        # Find cameras with PSNR > 30
        high_psnr_cameras = [r for r in results if r['psnr'] > 30.0]
        if high_psnr_cameras:
            print(f"\n🔥 High PSNR cameras (>30 dB): {len(high_psnr_cameras)} found")
            for r in sorted(high_psnr_cameras, key=lambda x: x['psnr'], reverse=True)[:10]:
                print(f"     {r['image_name']}: {r['psnr']:.3f} dB")
        
        # Find cameras with PSNR > 33
        ultra_high_psnr = [r for r in results if r['psnr'] > 33.0]
        if ultra_high_psnr:
            print(f"\n🚀 Ultra-high PSNR cameras (>33 dB): {len(ultra_high_psnr)} found")
            for r in sorted(ultra_high_psnr, key=lambda x: x['psnr'], reverse=True):
                print(f"     {r['image_name']}: {r['psnr']:.3f} dB ⭐")
        else:
            print(f"\n❓ No cameras found with PSNR > 33 dB")
            print(f"   The 33.83 dB might come from:")
            print(f"   1. Different subset of cameras during training")
            print(f"   2. Different rendering parameters")
            print(f"   3. Different evaluation timing/iteration")
    
    return results

def save_detailed_results(results, output_path):
    """保存详细结果到JSON文件"""
    if not results:
        return
        
    # Prepare data for JSON
    json_data = {
        'total_cameras': len(results),
        'average_psnr': sum(r['psnr'] for r in results) / len(results),
        'camera_results': results
    }
    
    # Add statistics
    psnr_values = [r['psnr'] for r in results]
    json_data['statistics'] = {
        'min_psnr': min(psnr_values),
        'max_psnr': max(psnr_values),
        'median_psnr': float(np.median(psnr_values)),
        'std_psnr': float(np.std(psnr_values)),
        'cameras_above_30': len([p for p in psnr_values if p > 30.0]),
        'cameras_above_33': len([p for p in psnr_values if p > 33.0])
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"💾 Detailed results saved to: {output_path}")

def main():
    parser = ArgumentParser(description="Evaluate all camera views")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--ply-path", type=str, help="Path to PLY file (default: model-path/gaussian_ball.ply)")
    parser.add_argument("--resolution-scale", type=float, default=2.0, help="Resolution scale factor")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    model_path = args.model_path
    ply_path = args.ply_path or os.path.join(model_path, "gaussian_ball.ply")
    
    if not os.path.exists(ply_path):
        # Try the best iteration path
        ply_path = os.path.join(model_path, "gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply")
    
    if not os.path.exists(ply_path):
        print(f"❌ PLY file not found: {ply_path}")
        return
    
    print(f"🚀 Starting complete evaluation")
    print(f"   Model path: {model_path}")
    print(f"   PLY path: {ply_path}")
    print(f"   Resolution scale: {args.resolution_scale}")
    
    # Load pipeline parameters using dummy parser
    from argparse import ArgumentParser
    dummy_parser = ArgumentParser()
    
    # Initialize ModelParams with parser
    dataset_params = ModelParams(dummy_parser)
    dataset = dataset_params.extract(type('Args', (), {
        'source_path': model_path,
        'model_path': model_path,
        'images': 'images',
        'resolution': -1,
        'white_background': False,
        'data_device': 'cuda',
        'train_test_exp': False,
        'sh_degree': 3,
        'depths': '',
        'eval': False
    })())
    
    # Initialize PipelineParams
    pipe_params = PipelineParams(dummy_parser)
    pipe = pipe_params.extract(type('Args', (), {
        'convert_SHs_python': False,
        'compute_cov3D_python': False,
        'debug': False
    })())
    
    # Load Gaussians
    print(f"📦 Loading Gaussians from {ply_path}")
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(ply_path, train_test_exp=False)
    print(f"   Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    # Load cameras
    sparse_dir = os.path.join(model_path, "sparse/0")
    images_dir = os.path.join(model_path, "images")
    
    print(f"📷 Loading cameras from COLMAP")
    print(f"   Sparse dir: {sparse_dir}")
    print(f"   Images dir: {images_dir}")
    
    cameras = load_cameras_from_colmap(sparse_dir, images_dir, args.resolution_scale)
    
    if not cameras:
        print("❌ No cameras loaded!")
        return
    
    # Setup rendering
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Evaluate all cameras
    results = evaluate_all_cameras(cameras, gaussians, pipe, background)
    
    # Save results if requested
    if args.output:
        save_detailed_results(results, args.output)
    elif results:
        # Auto-save with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"all_views_eval_{timestamp}.json"
        save_detailed_results(results, output_file)

if __name__ == "__main__":
    main() 