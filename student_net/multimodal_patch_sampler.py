"""
多模态Patch分割器：同时处理图片、点云、高斯球
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass 
class MultiModalPatch:
    """多模态patch数据结构"""
    patch_id: int
    # 3D数据
    gaussian_indices: np.ndarray  # 该patch包含的高斯球索引
    point_indices: np.ndarray     # 该patch包含的点云索引
    bbox_3d: Tuple[np.ndarray, np.ndarray]  # 3D边界框 (min_point, max_point)
    
    # 2D数据 (每个相机一个)
    image_patches: Dict[int, Dict]  # {camera_id: {bbox_2d, patch_image}}
    
    # 统计信息
    gaussian_count: int
    point_count: int
    spatial_volume: float

class MultiModalPatchSampler:
    """多模态patch分割器"""
    
    def __init__(self, 
                 gaussians,           # GaussianModel对象
                 points_3d,           # COLMAP点云 (N, 3)
                 images,              # 多视角图像列表
                 cameras,             # 相机参数列表
                 target_gaussian_count: int = 50000,  # 每个patch目标高斯球数量
                 grid_resolution: int = 8,            # 网格分辨率
                 overlap_ratio: float = 0.1):         # 重叠比例
        
        self.gaussians = gaussians
        self.points_3d = points_3d
        self.images = images
        self.cameras = cameras
        self.target_gaussian_count = target_gaussian_count
        self.grid_resolution = grid_resolution
        self.overlap_ratio = overlap_ratio
        
        # 获取场景边界
        self.gaussian_positions = gaussians.get_xyz.cpu().numpy()
        self.scene_bounds = self._compute_scene_bounds()
        
        print(f"📊 场景统计:")
        print(f"  高斯球数量: {len(self.gaussian_positions):,}")
        print(f"  点云数量: {len(self.points_3d):,}")
        print(f"  图像数量: {len(self.images)}")
        print(f"  场景边界: {self.scene_bounds}")
    
    def _compute_scene_bounds(self):
        """计算场景3D边界"""
        # 结合高斯球和点云位置
        all_positions = np.vstack([
            self.gaussian_positions,
            self.points_3d
        ])
        
        min_bounds = all_positions.min(axis=0)
        max_bounds = all_positions.max(axis=0)
        
        # 稍微扩大边界
        margin = (max_bounds - min_bounds) * 0.05
        return min_bounds - margin, max_bounds + margin
    
    def create_spatial_grid(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """创建3D空间网格"""
        min_bounds, max_bounds = self.scene_bounds
        
        # 创建网格边界
        x_edges = np.linspace(min_bounds[0], max_bounds[0], self.grid_resolution + 1)
        y_edges = np.linspace(min_bounds[1], max_bounds[1], self.grid_resolution + 1) 
        z_edges = np.linspace(min_bounds[2], max_bounds[2], self.grid_resolution + 1)
        
        grid_cells = []
        
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                for k in range(self.grid_resolution):
                    cell_min = np.array([x_edges[i], y_edges[j], z_edges[k]])
                    cell_max = np.array([x_edges[i+1], y_edges[j+1], z_edges[k+1]])
                    
                    # 添加重叠
                    if self.overlap_ratio > 0:
                        size = cell_max - cell_min
                        overlap = size * self.overlap_ratio
                        cell_min -= overlap / 2
                        cell_max += overlap / 2
                    
                    grid_cells.append((cell_min, cell_max))
        
        print(f"✅ 创建了 {len(grid_cells)} 个3D网格单元")
        return grid_cells
    
    def assign_gaussians_to_cells(self, grid_cells) -> Dict[int, np.ndarray]:
        """将高斯球分配到网格单元"""
        gaussian_assignments = {}
        
        for cell_id, (cell_min, cell_max) in enumerate(grid_cells):
            # 找到该单元内的高斯球
            mask = np.all(
                (self.gaussian_positions >= cell_min) & 
                (self.gaussian_positions <= cell_max), 
                axis=1
            )
            
            gaussian_indices = np.where(mask)[0]
            
            if len(gaussian_indices) > 0:
                gaussian_assignments[cell_id] = gaussian_indices
        
        print(f"📦 {len(gaussian_assignments)} 个网格单元包含高斯球")
        return gaussian_assignments
    
    def assign_points_to_cells(self, grid_cells) -> Dict[int, np.ndarray]:
        """将点云分配到网格单元"""
        point_assignments = {}
        
        for cell_id, (cell_min, cell_max) in enumerate(grid_cells):
            # 找到该单元内的点云
            mask = np.all(
                (self.points_3d >= cell_min) & 
                (self.points_3d <= cell_max), 
                axis=1
            )
            
            point_indices = np.where(mask)[0]
            
            if len(point_indices) > 0:
                point_assignments[cell_id] = point_indices
        
        print(f"📍 {len(point_assignments)} 个网格单元包含点云")
        return point_assignments
    
    def project_3d_cell_to_2d(self, cell_min, cell_max, camera) -> Optional[Tuple[int, int, int, int]]:
        """将3D网格单元投影到2D图像"""
        # 生成3D边界框的8个顶点
        corners_3d = np.array([
            [cell_min[0], cell_min[1], cell_min[2]],
            [cell_min[0], cell_min[1], cell_max[2]],
            [cell_min[0], cell_max[1], cell_min[2]],
            [cell_min[0], cell_max[1], cell_max[2]],
            [cell_max[0], cell_min[1], cell_min[2]],
            [cell_max[0], cell_min[1], cell_max[2]],
            [cell_max[0], cell_max[1], cell_min[2]],
            [cell_max[0], cell_max[1], cell_max[2]]
        ])
        
        # 投影到2D
        projected_2d = []
        for corner in corners_3d:
            # 世界坐标 -> 相机坐标
            corner_cam = camera.R @ corner + camera.T
            
            # 检查是否在相机前方
            if corner_cam[2] <= 0:
                return None
            
            # 相机坐标 -> 图像坐标
            x = corner_cam[0] / corner_cam[2] * camera.fx + camera.cx
            y = corner_cam[1] / corner_cam[2] * camera.fy + camera.cy
            
            projected_2d.append([x, y])
        
        projected_2d = np.array(projected_2d)
        
        # 计算2D边界框
        min_x = max(0, int(projected_2d[:, 0].min()))
        max_x = min(camera.image_width, int(projected_2d[:, 0].max()))
        min_y = max(0, int(projected_2d[:, 1].min()))
        max_y = min(camera.image_height, int(projected_2d[:, 1].max()))
        
        # 检查边界框有效性
        if max_x <= min_x or max_y <= min_y:
            return None
            
        return min_x, min_y, max_x - min_x, max_y - min_y
    
    def extract_image_patches(self, grid_cells) -> Dict[int, Dict[int, Dict]]:
        """为每个网格单元提取对应的图像patch"""
        image_patch_assignments = {}
        
        for cell_id, (cell_min, cell_max) in enumerate(grid_cells):
            cell_image_patches = {}
            
            for cam_id, (image, camera) in enumerate(zip(self.images, self.cameras)):
                # 投影3D单元到2D
                bbox_2d = self.project_3d_cell_to_2d(cell_min, cell_max, camera)
                
                if bbox_2d is not None:
                    x, y, w, h = bbox_2d
                    
                    # 提取图像patch
                    if isinstance(image, np.ndarray):
                        patch_image = image[y:y+h, x:x+w]
                    else:  # PIL Image
                        patch_image = np.array(image.crop((x, y, x+w, y+h)))
                    
                    cell_image_patches[cam_id] = {
                        'bbox_2d': bbox_2d,
                        'patch_image': patch_image,
                        'camera_id': cam_id
                    }
            
            if len(cell_image_patches) > 0:
                image_patch_assignments[cell_id] = cell_image_patches
        
        print(f"🖼️ 为 {len(image_patch_assignments)} 个网格单元生成了图像patch")
        return image_patch_assignments
    
    def balance_patches(self, gaussian_assignments) -> Dict[int, np.ndarray]:
        """平衡patch大小，避免某些patch过大或过小"""
        balanced_assignments = {}
        
        # 统计每个cell的高斯球数量
        cell_sizes = [(cell_id, len(indices)) for cell_id, indices in gaussian_assignments.items()]
        cell_sizes.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 网格单元大小分布:")
        for i, (cell_id, size) in enumerate(cell_sizes[:5]):
            print(f"  Cell {cell_id}: {size:,} 高斯球")
        
        for cell_id, gaussian_indices in gaussian_assignments.items():
            if len(gaussian_indices) > self.target_gaussian_count:
                # 大cell：随机采样
                selected = np.random.choice(
                    gaussian_indices, 
                    self.target_gaussian_count, 
                    replace=False
                )
                balanced_assignments[cell_id] = selected
                print(f"  Cell {cell_id}: {len(gaussian_indices):,} -> {len(selected):,} (采样)")
                
            elif len(gaussian_indices) < self.target_gaussian_count // 4:
                # 小cell：与邻近cell合并 (简化版本：直接跳过)
                print(f"  Cell {cell_id}: {len(gaussian_indices):,} (跳过-太小)")
                continue
                
            else:
                # 合适大小：直接使用
                balanced_assignments[cell_id] = gaussian_indices
        
        return balanced_assignments
    
    def create_multimodal_patches(self) -> List[MultiModalPatch]:
        """创建完整的多模态patch"""
        print(f"\n🚀 开始创建多模态patch...")
        
        # 1. 创建空间网格
        grid_cells = self.create_spatial_grid()
        
        # 2. 分配高斯球
        gaussian_assignments = self.assign_gaussians_to_cells(grid_cells)
        balanced_gaussian_assignments = self.balance_patches(gaussian_assignments)
        
        # 3. 分配点云
        point_assignments = self.assign_points_to_cells(grid_cells)
        
        # 4. 提取图像patch
        image_patch_assignments = self.extract_image_patches(grid_cells)
        
        # 5. 组合成最终patch
        multimodal_patches = []
        
        for cell_id in balanced_gaussian_assignments.keys():
            cell_min, cell_max = grid_cells[cell_id]
            
            # 获取数据
            gaussian_indices = balanced_gaussian_assignments[cell_id]
            point_indices = point_assignments.get(cell_id, np.array([]))
            image_patches = image_patch_assignments.get(cell_id, {})
            
            # 计算统计信息
            spatial_volume = np.prod(cell_max - cell_min)
            
            patch = MultiModalPatch(
                patch_id=cell_id,
                gaussian_indices=gaussian_indices,
                point_indices=point_indices,
                bbox_3d=(cell_min, cell_max),
                image_patches=image_patches,
                gaussian_count=len(gaussian_indices),
                point_count=len(point_indices),
                spatial_volume=spatial_volume
            )
            
            multimodal_patches.append(patch)
        
        print(f"\n✅ 成功创建 {len(multimodal_patches)} 个多模态patch")
        
        # 统计信息
        total_gaussians = sum(p.gaussian_count for p in multimodal_patches)
        total_points = sum(p.point_count for p in multimodal_patches)
        
        print(f"📊 Patch统计:")
        print(f"  总高斯球覆盖: {total_gaussians:,} / {len(self.gaussian_positions):,} ({total_gaussians/len(self.gaussian_positions)*100:.1f}%)")
        print(f"  总点云覆盖: {total_points:,} / {len(self.points_3d):,} ({total_points/len(self.points_3d)*100:.1f}%)")
        print(f"  平均每patch高斯球: {total_gaussians/len(multimodal_patches):.0f}")
        print(f"  平均每patch点云: {total_points/len(multimodal_patches):.0f}")
        
        return multimodal_patches

def test_patch_sampler():
    """测试patch分割器"""
    print("🧪 测试多模态patch分割器...")
    
    # 创建模拟数据
    mock_gaussians = type('MockGaussians', (), {
        'get_xyz': torch.randn(100000, 3) * 5  # 10万个高斯球
    })()
    
    mock_points = np.random.randn(20000, 3) * 5  # 2万个点云点
    mock_images = [np.random.randint(0, 255, (1091, 1957, 3), dtype=np.uint8) for _ in range(8)]
    
    # 模拟相机参数
    mock_cameras = []
    for i in range(8):
        camera = type('MockCamera', (), {
            'R': np.eye(3),
            'T': np.random.randn(3),
            'fx': 1000, 'fy': 1000,
            'cx': 978, 'cy': 545,
            'image_width': 1957,
            'image_height': 1091
        })()
        mock_cameras.append(camera)
    
    # 创建patch分割器
    sampler = MultiModalPatchSampler(
        gaussians=mock_gaussians,
        points_3d=mock_points,
        images=mock_images,
        cameras=mock_cameras,
        target_gaussian_count=5000,
        grid_resolution=4  # 小一点用于测试
    )
    
    # 生成patch
    patches = sampler.create_multimodal_patches()
    
    print(f"\n🎯 测试完成！生成了 {len(patches)} 个patch")
    return patches

if __name__ == "__main__":
    test_patch_sampler()
