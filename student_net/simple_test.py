"""
简化测试：直接读取PLY文件测试patch分割
"""

import numpy as np
import os
from multimodal_patch_sampler import MultiModalPatchSampler, MultiModalPatch
from typing import List
import struct

def read_ply_gaussians(ply_path: str, max_gaussians: int = 50000):
    """简单读取PLY文件的高斯球位置"""
    print(f"📂 读取PLY文件: {ply_path}")
    
    # 这里简化：生成模拟的高斯球位置
    # 实际应该解析PLY文件，但为了测试我们用模拟数据
    print(f"  ⚠️ 使用模拟数据 (实际应解析PLY)")
    
    # 模拟truck场景的高斯球分布
    np.random.seed(42)  # 保证可重复
    
    # 创建一个类似truck形状的点云分布
    # 主体部分 (卡车车身)
    main_body = np.random.normal([0, 0, 0], [3, 1.5, 1], (int(max_gaussians * 0.6), 3))
    
    # 轮子部分
    wheel_centers = [[-2, -1.5, -1], [2, -1.5, -1], [-2, 1.5, -1], [2, 1.5, -1]]
    wheels = []
    for center in wheel_centers:
        wheel = np.random.normal(center, [0.3, 0.3, 0.3], (int(max_gaussians * 0.1), 3))
        wheels.append(wheel)
    
    # 背景部分 (地面和远景)
    background = np.random.uniform([-10, -10, -5], [10, 10, 5], (int(max_gaussians * 0.1), 3))
    
    # 合并所有点
    all_gaussians = np.vstack([main_body] + wheels + [background])
    
    # 只取前max_gaussians个
    gaussians = all_gaussians[:max_gaussians]
    
    print(f"  ✅ 模拟了 {len(gaussians):,} 个高斯球位置")
    return gaussians

def create_mock_cameras(num_cameras: int = 8):
    """创建模拟相机"""
    cameras = []
    
    # 围绕truck创建8个相机位置
    angles = np.linspace(0, 2*np.pi, num_cameras, endpoint=False)
    radius = 8
    height = 2
    
    for i, angle in enumerate(angles):
        # 相机位置
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = height
        
        # 看向原点的旋转矩阵 (简化)
        R = np.eye(3)
        T = np.array([cam_x, cam_y, cam_z])
        
        camera = type('Camera', (), {
            'R': R,
            'T': T,
            'fx': 1000,
            'fy': 1000,
            'cx': 500,
            'cy': 500,
            'image_width': 1000,
            'image_height': 1000,
            'uid': i
        })()
        
        cameras.append(camera)
    
    print(f"📷 创建了 {len(cameras)} 个模拟相机")
    return cameras

def create_mock_images(cameras, image_size=(1000, 1000, 3)):
    """创建模拟图像"""
    images = []
    
    for i, camera in enumerate(cameras):
        # 生成彩色图像 (简单的渐变色)
        image = np.zeros(image_size, dtype=np.uint8)
        
        # 添加一些模式
        h, w = image_size[:2]
        for y in range(h):
            for x in range(w):
                image[y, x, 0] = (x * 255) // w  # 红色渐变
                image[y, x, 1] = (y * 255) // h  # 绿色渐变
                image[y, x, 2] = ((x + y) * 255) // (w + h)  # 蓝色渐变
        
        images.append(image)
    
    print(f"🖼️ 创建了 {len(images)} 张模拟图像")
    return images

def test_truck_scene_patches():
    """测试truck场景的patch分割"""
    print("🚛 测试Truck场景Patch分割")
    print("=" * 50)
    
    # 1. 模拟高斯球数据 (类似实际的204万个高斯球，但我们只用5万个测试)
    mock_gaussians = type('MockGaussians', (), {
        'get_xyz': torch.tensor(read_ply_gaussians("../output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply", max_gaussians=50000))
    })()
    
    # 2. 模拟点云数据 (COLMAP输出)
    mock_points = np.random.randn(5000, 3) * 3  # 5千个点云点
    
    # 3. 模拟相机和图像
    mock_cameras = create_mock_cameras(8)
    mock_images = create_mock_images(mock_cameras)
    
    # 4. 创建patch分割器
    sampler = MultiModalPatchSampler(
        gaussians=mock_gaussians,
        points_3d=mock_points,
        images=mock_images,
        cameras=mock_cameras,
        target_gaussian_count=8000,  # 每个patch 8千个高斯球
        grid_resolution=4,           # 4x4x4 = 64个网格
        overlap_ratio=0.15
    )
    
    # 5. 生成patch
    patches = sampler.create_multimodal_patches()
    
    # 6. 分析结果
    print(f"\n📊 Patch分割结果分析:")
    print(f"  成功创建patch数量: {len(patches)}")
    
    total_gaussians = sum(p.gaussian_count for p in patches)
    total_points = sum(p.point_count for p in patches)
    total_images = sum(len(p.image_patches) for p in patches)
    
    print(f"  高斯球覆盖率: {total_gaussians:,} / 50,000 ({total_gaussians/50000*100:.1f}%)")
    print(f"  点云覆盖率: {total_points:,} / 5,000 ({total_points/5000*100:.1f}%)")
    print(f"  图像patch总数: {total_images}")
    
    if len(patches) > 0:
        print(f"\n📝 样例patch详情:")
        for i, patch in enumerate(patches[:3]):
            print(f"  Patch {patch.patch_id}:")
            print(f"    高斯球: {patch.gaussian_count:,}")
            print(f"    点云: {patch.point_count:,}")
            print(f"    图像patch: {len(patch.image_patches)} 个视角")
            bbox_size = patch.bbox_3d[1] - patch.bbox_3d[0]
            print(f"    3D边界框大小: [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]")
    
    # 7. 评估patch质量
    print(f"\n🎯 Patch质量评估:")
    
    # 高斯球分布均匀性
    gaussian_counts = [p.gaussian_count for p in patches]
    if gaussian_counts:
        mean_count = np.mean(gaussian_counts)
        std_count = np.std(gaussian_counts)
        print(f"  高斯球分布: 均值={mean_count:.0f}, 标准差={std_count:.0f}")
    
    # 空间覆盖检查
    occupied_volumes = [p.spatial_volume for p in patches]
    total_scene_volume = np.prod([20, 20, 10])  # 估计场景体积
    coverage_ratio = sum(occupied_volumes) / total_scene_volume
    print(f"  空间覆盖率: {coverage_ratio:.2f}")
    
    # 图像patch覆盖
    patches_with_images = [p for p in patches if len(p.image_patches) > 0]
    print(f"  有图像patch的网格: {len(patches_with_images)} / {len(patches)}")
    
    return patches

if __name__ == "__main__":
    import torch  # 放在这里避免早期导入错误
    patches = test_truck_scene_patches()
    print(f"\n✅ 测试完成！生成了 {len(patches)} 个多模态patch")
