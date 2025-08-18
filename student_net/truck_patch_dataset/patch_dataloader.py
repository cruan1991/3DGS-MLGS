"""
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
    dataset = TruckPatchDataset(".")
    print(f"数据集大小: {len(dataset)}")
    
    sample = dataset[0]
    print(f"样例patch:")
    print(f"  高斯球: {sample['gaussian_positions'].shape}")
    print(f"  点云: {sample['point_cloud'].shape}")
    print(f"  图像: {len(sample['images'])} 个")
