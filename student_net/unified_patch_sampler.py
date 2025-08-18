"""
基于2D图像分割的统一Patch采样器
将图片裁成16块，然后找到对应的3D高斯球和点云
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
from PIL import Image

@dataclass 
class UnifiedPatch:
    """统一patch数据结构"""
    patch_id: int
    # 2D图像patch
    image_patch: np.ndarray  # 固定尺寸的图像块
    bbox_2d: Tuple[int, int, int, int]  # (x, y, w, h)
    
    # 对应的3D数据
    gaussian_indices: np.ndarray  # 该patch对应的高斯球索引
    point_indices: np.ndarray     # 该patch对应的点云索引
    
    # 统计信息
    gaussian_count: int
    point_count: int
    image_size: Tuple[int, int]

class ImageBasedUnifiedPatchSampler:
    """基于2D图像分割的统一patch采样器"""
    
    def __init__(self, 
                 gaussians,           # 完整高斯球模型
                 points_3d,           # 完整点云
                 images,              # 多视角图像列表
                 cameras,             # 相机参数列表
                 patch_grid: Tuple[int, int] = (4, 4),  # 4x4=16块
                 overlap_pixels: int = 32):             # 重叠像素数
        
        self.gaussians = gaussians
        self.points_3d = points_3d
        self.images = images
        self.cameras = cameras
        self.patch_grid = patch_grid
        self.overlap_pixels = overlap_pixels
        
        # 获取完整数据
        self.gaussian_positions = gaussians.get_xyz.cpu().numpy()
        self.total_gaussians = len(self.gaussian_positions)
        
        print(f"🎯 基于2D图像的统一Patch分割器")
        print(f"  高斯球总数: {self.total_gaussians:,}")
        print(f"  点云总数: {len(self.points_3d):,}")
        print(f"  图像数量: {len(self.images)}")
        print(f"  Patch网格: {patch_grid[0]}x{patch_grid[1]} = {patch_grid[0]*patch_grid[1]} 块")
        print(f"  重叠像素: {overlap_pixels}")
    
    def create_2d_image_patches(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """将单张图像分割成固定大小的patches"""
        h, w = image.shape[:2]
        grid_h, grid_w = self.patch_grid
        
        # 计算每个patch的尺寸
        patch_h = h // grid_h
        patch_w = w // grid_w
        
        print(f"  图像尺寸: {w}x{h}")
        print(f"  每个patch尺寸: {patch_w}x{patch_h}")
        
        patches = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                # 计算patch边界 (考虑overlap)
                start_y = max(0, i * patch_h - self.overlap_pixels)
                end_y = min(h, (i + 1) * patch_h + self.overlap_pixels)
                start_x = max(0, j * patch_w - self.overlap_pixels)
                end_x = min(w, (j + 1) * patch_w + self.overlap_pixels)
                
                # 提取patch
                patch_image = image[start_y:end_y, start_x:end_x]
                bbox = (start_x, start_y, end_x - start_x, end_y - start_y)
                
                patches.append((patch_image, bbox))
        
        return patches
    
    def project_3d_to_2d(self, positions_3d: np.ndarray, camera) -> np.ndarray:
        """将3D点投影到2D图像平面"""
        # 简化的投影函数 (实际应该用完整的相机模型)
        # 这里用简化的透视投影
        
        # 世界坐标 -> 相机坐标
        positions_cam = []
        for pos in positions_3d:
            # 简化的相机变换 (实际应该用R, T矩阵)
            cam_x = pos[0] - camera.T[0] if hasattr(camera, 'T') else pos[0]
            cam_y = pos[1] - camera.T[1] if hasattr(camera, 'T') else pos[1]
            cam_z = pos[2] - camera.T[2] if hasattr(camera, 'T') else pos[2]
            
            if cam_z > 0:  # 在相机前方
                # 投影到图像平面
                img_x = cam_x / cam_z * camera.fx + camera.cx
                img_y = cam_y / cam_z * camera.fy + camera.cy
                positions_cam.append([img_x, img_y, cam_z])
            else:
                positions_cam.append([-1, -1, -1])  # 无效投影
        
        return np.array(positions_cam)
    
    def find_3d_data_for_2d_patch(self, 
                                  bbox_2d: Tuple[int, int, int, int], 
                                  camera,
                                  gaussian_projected: np.ndarray,
                                  points_projected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """找到2D patch对应的3D数据"""
        x, y, w, h = bbox_2d
        
        # 找到投影在patch内的3D点
        gaussian_mask = (
            (gaussian_projected[:, 0] >= x) & 
            (gaussian_projected[:, 0] < x + w) &
            (gaussian_projected[:, 1] >= y) & 
            (gaussian_projected[:, 1] < y + h) &
            (gaussian_projected[:, 2] > 0)  # 有效的深度
        )
        
        points_mask = (
            (points_projected[:, 0] >= x) & 
            (points_projected[:, 0] < x + w) &
            (points_projected[:, 1] >= y) & 
            (points_projected[:, 1] < y + h) &
            (points_projected[:, 2] > 0)  # 有效的深度
        )
        
        # 返回对应的索引
        gaussian_indices = np.where(gaussian_mask)[0]
        point_indices = np.where(points_mask)[0]
        
        return gaussian_indices, point_indices
    
    def create_unified_patches(self) -> List[UnifiedPatch]:
        """创建基于2D图像的统一patches"""
        print(f"\n🚀 开始创建统一patches...")
        
        all_patches = []
        patch_id = 0
        
        # 对每个视角的图像创建patches
        for cam_id, (image, camera) in enumerate(zip(self.images, self.cameras)):
            print(f"📷 处理相机 {cam_id}: {image.shape}")
            
            # 1. 创建2D图像patches
            image_patches = self.create_2d_image_patches(image)
            
            # 2. 将3D数据投影到当前视角
            print(f"  🔄 投影3D数据到2D...")
            gaussian_projected = self.project_3d_to_2d(self.gaussian_positions, camera)
            points_projected = self.project_3d_to_2d(self.points_3d, camera)
            
            # 3. 为每个2D patch找到对应的3D数据
            for patch_idx, (patch_image, bbox_2d) in enumerate(image_patches):
                # 找到对应的3D数据
                gaussian_indices, point_indices = self.find_3d_data_for_2d_patch(
                    bbox_2d, camera, gaussian_projected, points_projected
                )
                
                # 创建统一patch
                unified_patch = UnifiedPatch(
                    patch_id=patch_id,
                    image_patch=patch_image,
                    bbox_2d=bbox_2d,
                    gaussian_indices=gaussian_indices,
                    point_indices=point_indices,
                    gaussian_count=len(gaussian_indices),
                    point_count=len(point_indices),
                    image_size=patch_image.shape[:2]
                )
                
                all_patches.append(unified_patch)
                patch_id += 1
                
                print(f"    Patch {patch_id-1}: {len(gaussian_indices):,} 高斯球, {len(point_indices):,} 点云, 图像 {patch_image.shape[:2]}")
        
        print(f"\n✅ 成功创建 {len(all_patches)} 个统一patches")
        
        # 统计信息
        total_gaussians = sum(p.gaussian_count for p in all_patches)
        total_points = sum(p.point_count for p in all_patches)
        
        print(f"📊 Patch统计:")
        print(f"  总patch数: {len(all_patches)}")
        print(f"  高斯球覆盖: {total_gaussians:,} / {self.total_gaussians:,} ({total_gaussians/self.total_gaussians*100:.1f}%)")
        print(f"  点云覆盖: {total_points:,} / {len(self.points_3d):,} ({total_points/len(self.points_3d)*100:.1f}%)")
        
        # 分析patch大小分布
        gaussian_counts = [p.gaussian_count for p in all_patches]
        point_counts = [p.point_count for p in all_patches]
        
        print(f"  高斯球分布: 最小={min(gaussian_counts):,}, 最大={max(gaussian_counts):,}, 平均={np.mean(gaussian_counts):.0f}")
        print(f"  点云分布: 最小={min(point_counts):,}, 最大={max(point_counts):,}, 平均={np.mean(point_counts):.0f}")
        
        return all_patches

def test_unified_patch_sampler():
    """测试统一patch分割器"""
    print("🧪 测试基于2D图像的统一Patch分割")
    print("=" * 60)
    
    # 创建模拟数据
    # 模拟完整的高斯球数据 (204万个)
    np.random.seed(42)
    mock_gaussians = type('MockGaussians', (), {
        'get_xyz': torch.randn(100000, 3) * 10  # 10万个用于测试
    })()
    
    # 模拟点云数据
    mock_points = np.random.randn(20000, 3) * 8
    
    # 模拟图像 (1280x960)
    mock_images = [np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8) for _ in range(4)]
    
    # 模拟相机
    mock_cameras = []
    for i in range(4):
        camera = type('MockCamera', (), {
            'fx': 1000, 'fy': 1000,
            'cx': 640, 'cy': 480,
            'T': np.random.randn(3) * 5
        })()
        mock_cameras.append(camera)
    
    # 创建统一patch分割器
    sampler = ImageBasedUnifiedPatchSampler(
        gaussians=mock_gaussians,
        points_3d=mock_points,
        images=mock_images,
        cameras=mock_cameras,
        patch_grid=(4, 4),      # 4x4=16块
        overlap_pixels=32       # 32像素重叠
    )
    
    # 生成patches
    patches = sampler.create_unified_patches()
    
    print(f"\n🎯 测试完成！生成了 {len(patches)} 个统一patches")
    
    # 显示几个样例patch
    print(f"\n📝 样例patches:")
    for i, patch in enumerate(patches[:3]):
        print(f"  Patch {patch.patch_id}:")
        print(f"    图像尺寸: {patch.image_size}")
        print(f"    高斯球: {patch.gaussian_count:,}")
        print(f"    点云: {patch.point_count:,}")
        print(f"    2D边界: {patch.bbox_2d}")
    
    return patches

if __name__ == "__main__":
    patches = test_unified_patch_sampler()
