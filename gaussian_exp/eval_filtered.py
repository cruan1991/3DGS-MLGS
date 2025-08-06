# eval_filtered.py
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

class DummyRenderPipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.antialiasing = False
        self.debug = False

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

        # Validate matrix shapes
        if len(cam['rotation']) != 3 or len(cam['rotation'][0]) != 3:
            raise ValueError("Invalid rotation matrix shape")
        if len(cam['position']) != 3:
            raise ValueError("Invalid position vector shape")

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

        # Convert image to tensor
        image_tensor = TF.to_tensor(image).cuda()

        # Process rotation matrix and position
        R = torch.tensor(cam['rotation'], device='cuda')  # Load directly to GPU
        R = R.transpose(0, 1).contiguous()  # Transpose to match getWorld2View2's expected format
        R = R.cpu().numpy()  # Convert to numpy array for getWorld2View2
        T = torch.tensor(cam['position'], device='cuda')
        T = T.cpu().numpy()  # Convert to numpy array for getWorld2View2

        # Create camera object with reasonable default FoV
        camera = Camera(
            resolution=(cam['width'], cam['height']),
            colmap_id=cam.get('id', 0),
            R=R,
            T=T,
            FoVx=0.6911112070083618,  # ~40 degrees
            FoVy=0.6911112070083618,  # ~40 degrees
            depth_params={"scale": 1.0, "offset": 0.0, "med_scale": 1.0},
            image=image_tensor,
            invdepthmap=None,
            image_name=cam['img_name'],
            uid=cam.get('id', 0),
            data_device="cuda"
        )
        cameras.append(camera)
        print(f"Added camera {len(cameras)} with image {cam['img_name']}")

    if not cameras:
        raise ValueError("No valid cameras loaded!")

    print(f"Successfully loaded {len(cameras)} cameras")
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
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)

    print(f"📸 Loading cameras from config: {config_path}")
    try:
        cam_data = load_camera_data(config_path, images_dir)
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
            # Render the image
            render_pkg = render(viewpoint, gaussians, pipe, bg_color)
            rendered_image = torch.clamp(render_pkg['render'], 0.0, 1.0)

            # Save rendered image
            rendered_pil = tensor_to_image(rendered_image)
            out_path = os.path.join(output_dir, viewpoint.image_name.replace('.jpg', '.png'))
            print(f"Saving rendered image to: {out_path}")
            rendered_pil.save(out_path)

            # Calculate metrics
            rendered_image = rendered_image.unsqueeze(0)  # Add batch dimension
            gt_image = torch.clamp(viewpoint.original_image.to(rendered_image.device), 0.0, 1.0).unsqueeze(0)
            psnr = cal_psnr(rendered_image, gt_image)
            psnr_value = psnr.item()
            all_metrics.append(psnr_value)

            print(f"View {idx}: PSNR = {psnr_value:.2f} dB")

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
