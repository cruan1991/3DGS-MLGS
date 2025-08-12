#!/usr/bin/env python3
"""
直接调用train.py的evaluation逻辑
100%复制training_report函数，确保一致性
"""

import os
import sys
import torch

# 添加路径以导入3DGS模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.image_utils import psnr
from utils.loss_utils import l1_loss

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def test_with_train_eval_logic():
    """直接调用train.py的evaluation逻辑"""
    
    # 1. 设置参数 - 完全模拟train.py
    class DummyModelParams:
        def __init__(self):
            self.source_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/mipnerf360/360/tandt_db/tandt/truck"
            self.model_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w"
            self.images = "images"
            self.depths = ""
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = True  # 启用eval模式
            self.train_test_exp = False
            self.sh_degree = 3
    
    class DummyPipelineParams:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.antialiasing = False
            self.debug = False
    
    dataset = DummyModelParams()
    pipe = DummyPipelineParams()
    
    # 2. 加载模型和场景
    gaussians = GaussianModel(dataset.sh_degree)
    ply_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    gaussians.load_ply(ply_path)
    
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    
    # 3. 设置渲染参数 - 完全按照train.py第350行
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    renderArgs = (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
    
    # 4. 定义渲染函数 - 完全按照train.py第467行
    def renderFunc(viewpoint, gaussians, *args):
        return render(viewpoint, gaussians, *args)
    
    # 5. 执行评估 - 完全复制train.py第456-518行的逻辑
    print("🎬 开始直接复制train.py的评估逻辑")
    
    validation_configs = (
        {'name': 'test', 'cameras': scene.getTestCameras()}, 
        {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
    )
    
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            print(f"\n📊 评估 {config['name']} 集合: {len(config['cameras'])} 个相机")
            
            l1_test = 0.0
            psnr_test = 0.0
            
            for idx, viewpoint in enumerate(config['cameras']):
                # 完全按照train.py第467行
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                
                # 完全按照train.py第469-471行
                if dataset.train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                    gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                
                # 完全按照train.py第501-502行
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_value = psnr(image, gt_image).mean().double()
                psnr_test += psnr_value
                
                print(f"  视角 {idx:3d} ({viewpoint.image_name}): PSNR = {psnr_value:.2f} dB")
                
                if idx >= 9:  # 限制输出
                    break
            
            # 完全按照train.py第503-504行
            psnr_test /= len(config['cameras'][:10])
            l1_test /= len(config['cameras'][:10])
            
            print(f"\n✅ {config['name']} 集合结果:")
            print(f"  - 平均 PSNR: {psnr_test:.2f} dB")
            print(f"  - 平均 L1: {l1_test:.6f}")

if __name__ == "__main__":
    test_with_train_eval_logic() 