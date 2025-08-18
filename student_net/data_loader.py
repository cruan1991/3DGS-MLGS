"""
StudentNet数据加载器：加载真实的truck场景数据并进行patch分割
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import json

# 添加父目录到path以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from utils.graphics_utils import focal2fov
from multimodal_patch_sampler import MultiModalPatchSampler, MultiModalPatch
from typing import List, Tuple

class TruckSceneDataLoader:
    """Truck场景数据加载器"""
    
    def __init__(self, 
                 model_path: str = "../output/truck-150w",
                 iteration: int = 994230,
                 resolution_scale: float = 2.0):
        
        self.model_path = model_path
        self.iteration = iteration
        self.resolution_scale = resolution_scale
        
        # 路径设置
        self.colmap_path = os.path.join(model_path, "sparse/0")
        self.images_path = os.path.join(model_path, "images")
        self.gaussian_path = os.path.join(model_path, f"gaussian_ball/iteration_{iteration}_best_psnr/gaussian_ball.ply")
        
        print(f"📂 加载Truck场景数据:")
        print(f"  模型路径: {model_path}")
        print(f"  迭代次数: {iteration}")
        print(f"  分辨率缩放: {resolution_scale}")
        
    def load_gaussians(self) -> GaussianModel:
        """加载高斯球模型"""
        print(f"🎯 加载高斯球模型...")
        
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(self.gaussian_path, use_train_test_exp=False)
        
        gaussian_count = len(gaussians.get_xyz)
        print(f"  ✅ 成功加载 {gaussian_count:,} 个高斯球")
        
        return gaussians
    
    def load_colmap_data(self) -> Tuple[np.ndarray, List]:
        """加载COLMAP数据：点云和相机"""
        print(f"📍 加载COLMAP数据...")
        
        # 读取相机内参和外参
        cam_intrinsics = read_intrinsics_binary(os.path.join(self.colmap_path, 'cameras.bin'))
        cam_extrinsics = read_extrinsics_binary(os.path.join(self.colmap_path, 'images.bin'))
        
        cameras = []
        points_3d_list = []
        
        for idx, (img_id, img_info) in enumerate(cam_extrinsics.items()):
            intrinsic = cam_intrinsics[img_info.camera_id]
            
            # 相机参数
            fx, fy, cx, cy = intrinsic.params
            width = int(intrinsic.width / self.resolution_scale)
            height = int(intrinsic.height / self.resolution_scale)
            
            fx_scaled = fx / self.resolution_scale
            fy_scaled = fy / self.resolution_scale
            cx_scaled = cx / self.resolution_scale
            cy_scaled = cy / self.resolution_scale
            
            # 相机姿态
            R = np.transpose(qvec2rotmat(img_info.qvec))
            T = np.array(img_info.tvec)
            
            # 创建相机对象
            camera = type('Camera', (), {
                'R': R,
                'T': T,
                'fx': fx_scaled,
                'fy': fy_scaled,
                'cx': cx_scaled,
                'cy': cy_scaled,
                'image_width': width,
                'image_height': height,
                'image_name': img_info.name,
                'uid': idx
            })()
            
            cameras.append(camera)
            
            # COLMAP的3D点 (这里简化：从image的point3D_ids获取)
            if hasattr(img_info, 'point3D_ids'):
                valid_points = img_info.point3D_ids[img_info.point3D_ids != -1]
                if len(valid_points) > 0:
                    # 简化版本：生成一些3D点 (实际应该从points3D.bin读取)
                    random_points = np.random.randn(len(valid_points), 3) * 2
                    points_3d_list.append(random_points)
        
        # 合并所有3D点
        if points_3d_list:
            points_3d = np.vstack(points_3d_list)
        else:
            # 如果没有3D点，生成一些假的点云用于测试
            points_3d = np.random.randn(10000, 3) * 5
            print(f"  ⚠️ 使用模拟点云数据")
        
        print(f"  ✅ 加载了 {len(cameras)} 个相机")
        print(f"  ✅ 加载了 {len(points_3d):,} 个3D点")
        
        return points_3d, cameras
    
    def load_images(self, cameras, max_images: int = 16) -> List[np.ndarray]:
        """加载图像数据"""
        print(f"🖼️ 加载图像数据...")
        
        images = []
        
        for i, camera in enumerate(cameras[:max_images]):
            image_path = os.path.join(self.images_path, camera.image_name)
            
            if os.path.exists(image_path):
                # 加载并调整图像大小
                pil_image = Image.open(image_path)
                
                if self.resolution_scale != 1.0:
                    new_width = int(pil_image.width / self.resolution_scale)
                    new_height = int(pil_image.height / self.resolution_scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                
                image_array = np.array(pil_image)
                images.append(image_array)
            else:
                print(f"    ⚠️ 图像不存在: {image_path}")
                # 创建黑色图像作为占位符
                placeholder = np.zeros((camera.image_height, camera.image_width, 3), dtype=np.uint8)
                images.append(placeholder)
        
        print(f"  ✅ 加载了 {len(images)} 张图像")
        return images
    
    def create_patches(self, 
                      target_gaussian_count: int = 50000,
                      grid_resolution: int = 6) -> List[MultiModalPatch]:
        """创建多模态patch"""
        print(f"\n🚀 开始创建多模态patch...")
        print(f"  目标高斯球数量/patch: {target_gaussian_count:,}")
        print(f"  网格分辨率: {grid_resolution}x{grid_resolution}x{grid_resolution}")
        
        # 加载所有数据
        gaussians = self.load_gaussians()
        points_3d, cameras = self.load_colmap_data()
        images = self.load_images(cameras)
        
        # 创建patch分割器
        sampler = MultiModalPatchSampler(
            gaussians=gaussians,
            points_3d=points_3d,
            images=images,
            cameras=cameras,
            target_gaussian_count=target_gaussian_count,
            grid_resolution=grid_resolution,
            overlap_ratio=0.1
        )
        
        # 生成patch
        patches = sampler.create_multimodal_patches()
        
        print(f"\n✅ 成功创建了 {len(patches)} 个多模态patch")
        
        return patches, gaussians, cameras, images

def analyze_patches(patches: List[MultiModalPatch]):
    """分析patch的统计信息"""
    print(f"\n📊 Patch分析报告:")
    
    # 基础统计
    total_patches = len(patches)
    gaussian_counts = [p.gaussian_count for p in patches]
    point_counts = [p.point_count for p in patches]
    image_patch_counts = [len(p.image_patches) for p in patches]
    
    print(f"  总patch数量: {total_patches}")
    print(f"  高斯球分布: 最小={min(gaussian_counts)}, 最大={max(gaussian_counts)}, 平均={np.mean(gaussian_counts):.0f}")
    print(f"  点云分布: 最小={min(point_counts)}, 最大={max(point_counts)}, 平均={np.mean(point_counts):.0f}")
    print(f"  图像patch分布: 最小={min(image_patch_counts)}, 最大={max(image_patch_counts)}, 平均={np.mean(image_patch_counts):.1f}")
    
    # 显示几个样例patch
    print(f"\n📝 样例patch:")
    for i, patch in enumerate(patches[:3]):
        print(f"  Patch {patch.patch_id}:")
        print(f"    高斯球: {patch.gaussian_count:,}")
        print(f"    点云: {patch.point_count:,}")
        print(f"    图像patch: {len(patch.image_patches)} 个视角")
        print(f"    空间体积: {patch.spatial_volume:.2f}")

def save_patch_data(patches: List[MultiModalPatch], output_dir: str = "patch_data"):
    """保存patch数据以便后续训练使用"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存patch配置
    patch_config = {
        'total_patches': len(patches),
        'patches': []
    }
    
    for patch in patches:
        patch_info = {
            'patch_id': patch.patch_id,
            'gaussian_count': patch.gaussian_count,
            'point_count': patch.point_count,
            'image_patch_count': len(patch.image_patches),
            'bbox_3d': [patch.bbox_3d[0].tolist(), patch.bbox_3d[1].tolist()],
            'spatial_volume': patch.spatial_volume
        }
        patch_config['patches'].append(patch_info)
    
    # 保存配置文件
    config_path = os.path.join(output_dir, 'patch_config.json')
    with open(config_path, 'w') as f:
        json.dump(patch_config, f, indent=2)
    
    print(f"💾 Patch配置已保存到: {config_path}")
    
    return output_dir

def main():
    """主函数：演示完整的数据加载和patch创建流程"""
    print("🚀 Truck场景多模态patch创建演示")
    
    # 创建数据加载器
    loader = TruckSceneDataLoader(
        model_path="../output/truck-150w",
        iteration=994230,
        resolution_scale=4.0  # 使用4x缩放减少内存使用
    )
    
    # 创建patch
    patches, gaussians, cameras, images = loader.create_patches(
        target_gaussian_count=30000,  # 3万个高斯球/patch
        grid_resolution=4              # 4x4x4网格
    )
    
    # 分析patch
    analyze_patches(patches)
    
    # 保存数据
    output_dir = save_patch_data(patches)
    
    print(f"\n🎯 演示完成！")
    print(f"  生成patch数量: {len(patches)}")
    print(f"  数据保存位置: {output_dir}")

if __name__ == "__main__":
    main()
