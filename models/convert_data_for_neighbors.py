#!/usr/bin/env python3
"""
数据转换脚本：将PLY格式转换为PyTorch格式
===========================================

功能：
- 将COLMAP点云PLY转换为torch.Tensor
- 将高斯球PLY转换为包含xyz和scale的字典
- 保存为.pt格式供邻居预计算使用
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加3DGS路径
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

# 尝试导入3DGS模块
try:
    from scene import dataset_readers
    from plyfile import PlyData
    FULL_3DGS_AVAILABLE = True
    print("✅ 3DGS模块导入成功")
except ImportError as e:
    print(f"⚠️ 3DGS模块导入失败: {e}")
    FULL_3DGS_AVAILABLE = False

def load_colmap_ply(ply_path):
    """加载COLMAP点云PLY文件"""
    print(f"正在加载COLMAP点云: {ply_path}")
    
    if FULL_3DGS_AVAILABLE:
        try:
            point_cloud = dataset_readers.fetchPly(ply_path)
            points = point_cloud.points.astype(np.float32)
            print(f"✅ 使用3DGS加载器成功: {len(points)} 个点")
            return torch.from_numpy(points)
        except Exception as e:
            print(f"⚠️ 3DGS加载器失败: {e}，尝试直接PLY解析")
    
    # 直接PLY解析备用方案
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        x = np.array(vertex['x'])
        y = np.array(vertex['y']) 
        z = np.array(vertex['z'])
        points = np.stack([x, y, z], axis=1).astype(np.float32)
        
        print(f"✅ 直接PLY解析成功: {len(points)} 个点")
        return torch.from_numpy(points)
        
    except Exception as e:
        raise RuntimeError(f"加载COLMAP PLY失败: {e}")

def load_gaussian_ply(ply_path):
    """加载高斯球PLY文件"""
    print(f"正在加载高斯球数据: {ply_path}")
    
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        property_names = [p.name for p in vertex.properties]
        
        # 提取坐标
        x = np.array(vertex['x'])
        y = np.array(vertex['y'])
        z = np.array(vertex['z'])
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        
        # 提取尺度
        scale_props = [p for p in property_names if p.startswith('scale_')]
        scale_props.sort()  # 确保顺序 scale_0, scale_1, scale_2
        
        if len(scale_props) >= 3:
            scale_data = []
            for prop in scale_props[:3]:  # 只取前3个
                scale_data.append(np.array(vertex[prop]))
            scale = np.stack(scale_data, axis=1).astype(np.float32)
        elif len(scale_props) == 1:
            # 如果只有一个尺度值，复制3次
            scale_val = np.array(vertex[scale_props[0]])
            scale = np.stack([scale_val, scale_val, scale_val], axis=1).astype(np.float32)
        else:
            # 如果没有尺度信息，使用默认值
            print("⚠️ 未找到尺度信息，使用默认值 0.01")
            scale = np.full((len(xyz), 3), 0.01, dtype=np.float32)
        
        print(f"✅ 高斯球数据加载成功: {len(xyz)} 个高斯球")
        print(f"   - 坐标范围: [{xyz.min():.3f}, {xyz.max():.3f}]")
        print(f"   - 尺度范围: [{scale.min():.6f}, {scale.max():.6f}]")
        
        return {
            'xyz': torch.from_numpy(xyz),
            'scale': torch.from_numpy(scale)
        }
        
    except Exception as e:
        raise RuntimeError(f"加载高斯球PLY失败: {e}")

def main():
    # 输入文件路径
    colmap_ply = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/sparse/0/points3D.ply"
    gaussian_ply = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    
    # 输出文件路径
    output_dir = Path("batches")
    output_dir.mkdir(exist_ok=True)
    
    colmap_pt = output_dir / "colmap_points.pt"
    gaussian_pt = output_dir / "gaussian_data.pt"
    
    print("🔄 开始数据转换...")
    print("=" * 50)
    
    # 转换COLMAP数据
    try:
        colmap_data = load_colmap_ply(colmap_ply)
        torch.save(colmap_data, colmap_pt)
        print(f"✅ COLMAP数据已保存: {colmap_pt}")
    except Exception as e:
        print(f"❌ COLMAP数据转换失败: {e}")
        return False
    
    # 转换高斯数据
    try:
        gaussian_data = load_gaussian_ply(gaussian_ply)
        torch.save(gaussian_data, gaussian_pt)
        print(f"✅ 高斯数据已保存: {gaussian_pt}")
    except Exception as e:
        print(f"❌ 高斯数据转换失败: {e}")
        return False
    
    print("=" * 50)
    print("🎉 数据转换完成！")
    print(f"COLMAP数据: {colmap_pt} ({colmap_data.shape})")
    print(f"高斯数据: {gaussian_pt} (xyz: {gaussian_data['xyz'].shape}, scale: {gaussian_data['scale'].shape})")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 