import open3d as o3d
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import os

# === 路径配置 ===
pcd_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/sparse/0/points3D.ply"
gauss_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck120w/point_cloud/iteration_740000/point_cloud.ply"
out_dir = "./preprocessed_stats"
os.makedirs(out_dir, exist_ok=True)

# === 点云处理 ===
print("[INFO] Loading point cloud...")
pcd = o3d.io.read_point_cloud(pcd_path)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

# 局部密度估计
print("[INFO] Estimating density...")
tree = cKDTree(points)
density = np.array([len(tree.query_ball_point(p, r=0.5)) for p in tqdm(points)])

# 曲率估计（主成分特征值比例）
print("[INFO] Estimating curvature...")
curvature = []
for p in tqdm(points):
    idx = tree.query_ball_point(p, r=0.5)
    if len(idx) < 3:
        curvature.append(0.0)
        continue
    local_pts = points[idx] - p
    cov = np.cov(local_pts.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    c = eigvals[0] / np.sum(eigvals)
    curvature.append(c)
curvature = np.array(curvature)

# === 保存点云特征 ===
np.savez(os.path.join(out_dir, "pointcloud_stats.npz"),
         point_xyz=points,
         point_normal=normals,
         point_density=density,
         point_curvature=curvature)
print("[OK] pointcloud_stats.npz saved.")

# === 高斯处理 ===
print("[INFO] Loading Gaussian point cloud...")
gauss = o3d.io.read_point_cloud(gauss_path)
gauss_xyz = np.asarray(gauss.points)
gauss_rgb = np.asarray(gauss.colors)

# 解析颜色通道存储的 scale + orientation 信息（假设用 RGB 通道编码了 scale+方向，需你替换为真实来源）
# 临时代码用于占位真实高斯数据读取
gauss_scale = np.ones((gauss_xyz.shape[0], 3)) * 0.01
gauss_quat = np.tile([1, 0, 0, 0], (gauss_xyz.shape[0], 1))

# Flatness: min / max(scale)
smin = np.min(gauss_scale, axis=1)
smax = np.max(gauss_scale, axis=1)
gauss_flatness = smin / np.clip(smax, 1e-6, None)

# Volume: product
gauss_volume = np.prod(gauss_scale, axis=1)

# 主轴方向（默认 quat 编码方向）
r = R.from_quat(gauss_quat[:, [1, 2, 3, 0]])
gauss_axis = r.apply(np.tile([[1, 0, 0]], (gauss_quat.shape[0], 1)))

# === 保存高斯特征 ===
np.savez(os.path.join(out_dir, "gaussians_stats.npz"),
         gauss_xyz=gauss_xyz,
         gauss_scale=gauss_scale,
         gauss_flatness=gauss_flatness,
         gauss_axis=gauss_axis,
         gauss_volume=gauss_volume)
print("[OK] gaussians_stats.npz saved.")
