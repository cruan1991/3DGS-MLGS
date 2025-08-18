"""
真实Truck场景数据加载器：加载真实的点云、图片、高斯球数据并生成patch dataset
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import json
import pickle
import struct
from typing import List, Dict, Tuple, Optional
import shutil

# 添加父目录到path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_patch_sampler import MultiModalPatchSampler, MultiModalPatch

class PLYReader:
    """PLY文件读取器"""
    
    @staticmethod
    def read_ply_points(ply_path: str) -> np.ndarray:
        """读取PLY格式的点云数据"""
        print(f"📂 读取PLY点云: {ply_path}")
        
        with open(ply_path, 'rb') as f:
            # 读取header
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # 解析header信息
            vertex_count = 0
            properties = []
            
            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('property'):
                    properties.append(line.split()[1:])  # [type, name]
            
            print(f"  顶点数量: {vertex_count:,}")
            print(f"  属性数量: {len(properties)}")
            
            # 读取二进制数据
            points = []
            for i in range(vertex_count):
                point_data = []
                for prop_type, prop_name in properties:
                    if prop_type == 'float':
                        value = struct.unpack('f', f.read(4))[0]
                    elif prop_type == 'double': 
                        value = struct.unpack('d', f.read(8))[0]
                    elif prop_type == 'uchar':
                        value = struct.unpack('B', f.read(1))[0]
                    elif prop_type == 'int':
                        value = struct.unpack('i', f.read(4))[0]
                    else:
                        raise ValueError(f"未支持的数据类型: {prop_type}")
                    
                    point_data.append(value)
                
                # 只取前3个坐标 (x, y, z)
                points.append(point_data[:3])
        
        points = np.array(points)
        print(f"  ✅ 成功读取 {len(points):,} 个3D点")
        return points

    @staticmethod 
    def read_gaussian_positions(ply_path: str, max_points: Optional[int] = None) -> np.ndarray:
        """读取高斯球PLY文件中的位置信息"""
        print(f"📂 读取高斯球位置: {ply_path}")
        
        # 使用与points3D.ply相同的读取逻辑
        points = PLYReader.read_ply_points(ply_path)
        
        if max_points and len(points) > max_points:
            # 随机采样减少内存使用
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            print(f"  ⚠️ 随机采样到 {max_points:,} 个高斯球")
        
        return points

class RealTruckDataLoader:
    """真实Truck场景数据加载器"""
    
    def __init__(self):
        # 真实数据路径
        self.points_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/sparse/0/points3D.ply"
        self.images_dir = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/images"
        self.colmap_dir = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/sparse/0"
        self.gaussian_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
        
        print(f"🚛 真实Truck场景数据加载器")
        print(f"  点云路径: {self.points_path}")
        print(f"  图片目录: {self.images_dir}")
        print(f"  高斯球路径: {self.gaussian_path}")
        
        # 检查文件存在性
        self._check_files()
    
    def _check_files(self):
        """检查必要文件是否存在"""
        files_to_check = [
            self.points_path,
            self.images_dir,
            self.gaussian_path
        ]
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
        
        print(f"✅ 所有必要文件检查通过")
    
    def load_points_3d(self) -> np.ndarray:
        """加载真实的3D点云"""
        points = PLYReader.read_ply_points(self.points_path)
        return points
    
    def load_gaussian_positions(self, max_gaussians: Optional[int] = 100000) -> np.ndarray:
        """加载高斯球位置 (为了内存考虑，可以限制数量)"""
        positions = PLYReader.read_gaussian_positions(self.gaussian_path, max_gaussians)
        return positions
    
    def load_images(self, max_images: Optional[int] = 16, resolution_scale: float = 4.0) -> List[np.ndarray]:
        """加载真实图片"""
        print(f"🖼️ 加载图片数据...")
        print(f"  最大图片数: {max_images}")
        print(f"  分辨率缩放: {resolution_scale}x")
        
        # 获取所有图片文件
        image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])
        
        if max_images:
            # 均匀采样图片
            step = len(image_files) // max_images
            selected_files = image_files[::step][:max_images]
        else:
            selected_files = image_files
        
        images = []
        for img_file in selected_files:
            img_path = os.path.join(self.images_dir, img_file)
            
            # 加载并缩放图片
            pil_image = Image.open(img_path)
            
            if resolution_scale != 1.0:
                new_width = int(pil_image.width / resolution_scale)
                new_height = int(pil_image.height / resolution_scale)
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            image_array = np.array(pil_image)
            images.append(image_array)
        
        print(f"  ✅ 加载了 {len(images)} 张图片")
        print(f"  图片尺寸: {images[0].shape}")
        return images
    
    def create_mock_cameras(self, num_cameras: int) -> List:
        """创建模拟相机参数 (简化版本，实际应该从COLMAP读取)"""
        print(f"📷 创建模拟相机参数...")
        
        cameras = []
        
        # 围绕truck创建相机
        angles = np.linspace(0, 2*np.pi, num_cameras, endpoint=False)
        radius = 10
        height = 2
        
        for i, angle in enumerate(angles):
            cam_x = radius * np.cos(angle)
            cam_y = radius * np.sin(angle)
            cam_z = height
            
            # 简化的相机参数
            camera = type('Camera', (), {
                'R': np.eye(3),
                'T': np.array([cam_x, cam_y, cam_z]),
                'fx': 500,  # 缩放后的焦距
                'fy': 500,
                'cx': 250,  # 缩放后的主点
                'cy': 250,
                'image_width': 500,  # 缩放后的分辨率
                'image_height': 500,
                'uid': i
            })()
            
            cameras.append(camera)
        
        print(f"  ✅ 创建了 {len(cameras)} 个相机")
        return cameras

class PatchDatasetGenerator:
    """Patch数据集生成器"""
    
    def __init__(self, output_dir: str = "truck_patch_dataset"):
        self.output_dir = output_dir
        self.patches_dir = os.path.join(output_dir, "patches")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        # 创建输出目录
        os.makedirs(self.patches_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        print(f"📁 Patch数据集生成器")
        print(f"  输出目录: {output_dir}")
    
    def save_patch_data(self, patches: List[MultiModalPatch], 
                       gaussians_positions: np.ndarray,
                       points_3d: np.ndarray,
                       images: List[np.ndarray]) -> str:
        """保存patch数据为训练就绪的dataset"""
        print(f"💾 保存patch数据集...")
        
        dataset_info = {
            'total_patches': len(patches),
            'total_gaussians': len(gaussians_positions),
            'total_points': len(points_3d),
            'total_images': len(images),
            'patch_files': [],
            'creation_time': str(np.datetime64('now'))
        }
        
        # 保存每个patch
        for i, patch in enumerate(patches):
            patch_file = f"patch_{patch.patch_id:03d}.pkl"
            patch_path = os.path.join(self.patches_dir, patch_file)
            
            # 提取patch对应的实际数据
            patch_gaussians = gaussians_positions[patch.gaussian_indices]
            patch_points = points_3d[patch.point_indices] if len(patch.point_indices) > 0 else np.array([])
            
            # 提取patch对应的图像数据
            patch_images = {}
            for cam_id, img_patch_info in patch.image_patches.items():
                if cam_id < len(images):
                    bbox_2d = img_patch_info['bbox_2d']
                    x, y, w, h = bbox_2d
                    patch_image = images[cam_id][y:y+h, x:x+w]
                    patch_images[cam_id] = {
                        'image': patch_image,
                        'bbox_2d': bbox_2d
                    }
            
            # 组装patch数据
            patch_data = {
                'patch_id': patch.patch_id,
                'gaussian_positions': patch_gaussians,  # (N, 3)
                'point_cloud': patch_points,            # (M, 3)
                'images': patch_images,                 # {cam_id: {'image': array, 'bbox_2d': tuple}}
                'bbox_3d': patch.bbox_3d,
                'statistics': {
                    'gaussian_count': patch.gaussian_count,
                    'point_count': patch.point_count,
                    'image_patch_count': len(patch.image_patches),
                    'spatial_volume': patch.spatial_volume
                }
            }
            
            # 保存patch文件
            with open(patch_path, 'wb') as f:
                pickle.dump(patch_data, f)
            
            dataset_info['patch_files'].append({
                'file': patch_file,
                'patch_id': patch.patch_id,
                'gaussian_count': patch.gaussian_count,
                'point_count': patch.point_count,
                'image_count': len(patch_images)
            })
            
            print(f"  💾 保存 Patch {patch.patch_id}: {patch.gaussian_count:,} 高斯球, {patch.point_count:,} 点云, {len(patch_images)} 图像")
        
        # 保存dataset元数据
        metadata_path = os.path.join(self.metadata_dir, 'dataset_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # 保存原始数据信息
        raw_data_info = {
            'gaussians_shape': gaussians_positions.shape,
            'points_shape': points_3d.shape,
            'images_info': [img.shape for img in images],
            'data_ranges': {
                'gaussians_min': gaussians_positions.min(axis=0).tolist(),
                'gaussians_max': gaussians_positions.max(axis=0).tolist(),
                'points_min': points_3d.min(axis=0).tolist() if len(points_3d) > 0 else None,
                'points_max': points_3d.max(axis=0).tolist() if len(points_3d) > 0 else None
            }
        }
        
        raw_info_path = os.path.join(self.metadata_dir, 'raw_data_info.json')
        with open(raw_info_path, 'w') as f:
            json.dump(raw_data_info, f, indent=2)
        
        print(f"✅ 数据集保存完成!")
        print(f"  Patch文件: {len(patches)} 个")
        print(f"  元数据: {metadata_path}")
        print(f"  原始数据信息: {raw_info_path}")
        
        return self.output_dir
    
    def create_dataloader_script(self) -> str:
        """创建PyTorch DataLoader脚本"""
        dataloader_script = '''"""
PyTorch DataLoader for Truck Patch Dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import os
import numpy as np

class TruckPatchDataset(Dataset):
    """Truck场景Patch数据集"""
    
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.patches_dir = os.path.join(dataset_dir, "patches")
        self.metadata_dir = os.path.join(dataset_dir, "metadata")
        self.transform = transform
        
        # 加载数据集信息
        with open(os.path.join(self.metadata_dir, 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)
        
        self.patch_files = self.dataset_info['patch_files']
        
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        patch_file = self.patch_files[idx]['file']
        patch_path = os.path.join(self.patches_dir, patch_file)
        
        # 加载patch数据
        with open(patch_path, 'rb') as f:
            patch_data = pickle.load(f)
        
        # 转换为tensor
        sample = {
            'patch_id': patch_data['patch_id'],
            'gaussian_positions': torch.from_numpy(patch_data['gaussian_positions']).float(),
            'point_cloud': torch.from_numpy(patch_data['point_cloud']).float() if len(patch_data['point_cloud']) > 0 else torch.empty(0, 3),
            'images': {cam_id: torch.from_numpy(img_data['image']).float() / 255.0 
                      for cam_id, img_data in patch_data['images'].items()},
            'bbox_3d': patch_data['bbox_3d'],
            'statistics': patch_data['statistics']
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def create_dataloader(dataset_dir, batch_size=4, shuffle=True, num_workers=2):
    """创建DataLoader"""
    dataset = TruckPatchDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                           num_workers=num_workers, collate_fn=patch_collate_fn)
    return dataloader

def patch_collate_fn(batch):
    """自定义collate函数处理变长数据"""
    # 这里需要根据实际训练需求定制
    return batch  # 简化版本直接返回list

if __name__ == "__main__":
    # 测试数据加载
    dataset = TruckPatchDataset("truck_patch_dataset")
    print(f"数据集大小: {len(dataset)}")
    
    sample = dataset[0]
    print(f"样例patch:")
    print(f"  高斯球: {sample['gaussian_positions'].shape}")
    print(f"  点云: {sample['point_cloud'].shape}")
    print(f"  图像: {len(sample['images'])} 个")
'''
        
        script_path = os.path.join(self.output_dir, "patch_dataloader.py")
        with open(script_path, 'w') as f:
            f.write(dataloader_script)
        
        print(f"📝 创建了DataLoader脚本: {script_path}")
        return script_path

def main():
    """主函数：完整的数据加载和dataset生成流程"""
    print("🚛 真实Truck场景Patch数据集生成")
    print("=" * 60)
    
    # 1. 加载真实数据
    loader = RealTruckDataLoader()
    
    # 加载各种数据 (控制规模避免内存爆炸)
    points_3d = loader.load_points_3d()
    gaussian_positions = loader.load_gaussian_positions(max_gaussians=50000)  # 限制5万个
    images = loader.load_images(max_images=8, resolution_scale=4.0)  # 8张图，4x缩放
    cameras = loader.create_mock_cameras(len(images))
    
    # 2. 创建mock高斯球对象
    mock_gaussians = type('MockGaussians', (), {
        'get_xyz': torch.from_numpy(gaussian_positions).float()
    })()
    
    # 3. 创建patch分割器
    sampler = MultiModalPatchSampler(
        gaussians=mock_gaussians,
        points_3d=points_3d,
        images=images,
        cameras=cameras,
        target_gaussian_count=8000,  # 每个patch 8k高斯球
        grid_resolution=4,           # 4x4x4网格
        overlap_ratio=0.1
    )
    
    # 4. 生成patch
    patches = sampler.create_multimodal_patches()
    
    # 5. 生成数据集
    dataset_generator = PatchDatasetGenerator("truck_patch_dataset")
    dataset_dir = dataset_generator.save_patch_data(
        patches, gaussian_positions, points_3d, images
    )
    
    # 6. 创建DataLoader脚本
    dataloader_script = dataset_generator.create_dataloader_script()
    
    print(f"\n🎯 数据集生成完成！")
    print(f"  数据集目录: {dataset_dir}")
    print(f"  Patch数量: {len(patches)}")
    print(f"  DataLoader: {dataloader_script}")
    
    return dataset_dir, patches

if __name__ == "__main__":
    dataset_dir, patches = main()
