import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

output_dir = "pointcloud_analysis_output"
os.makedirs(output_dir, exist_ok=True)

def estimate_normals(pcd, radius=0.05, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(pcd.normals)
    return normals

def estimate_density(points, k=20):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    dists, _ = nbrs.kneighbors(points)
    density = 1 / (np.mean(dists[:, 1:], axis=1) + 1e-8)
    return density

def estimate_curvature(points, k=20):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, idxs = nbrs.kneighbors(points)
    curvature = []
    for i in range(len(points)):
        neighbors = points[idxs[i]]
        cov = np.cov(neighbors.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))
        c = eigvals[0] / (np.sum(eigvals) + 1e-8)
        curvature.append(c)
    return np.array(curvature)

def visualize_color_map(values, coords, title, filename, cmap='viridis'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap=cmap, s=0.5)
    plt.colorbar(p)
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"[OK] {title} saved to {filename}")

def analyze_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    coords = np.asarray(pcd.points)

    # 1. 空间分布
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=0.01)
    ax.set_title("Point Cloud Spatial Distribution")
    plt.savefig(os.path.join(output_dir, "spatial_distribution.png"))
    plt.close()
    print("[OK] Spatial distribution saved.")

    # 2. 法线估计
    normals = estimate_normals(pcd)
    visualize_color_map(normals[:, 2], coords, "Normal Z Component", "normal_z.png")

    # 3. 局部密度
    density = estimate_density(coords)
    visualize_color_map(density, coords, "Local Density", "density.png")

    # 4. 曲率估计
    curvature = estimate_curvature(coords)
    visualize_color_map(curvature, coords, "Local Curvature", "curvature.png")

    # 5. Edge-like region 估计（高 curvature）
    is_edge = curvature > np.percentile(curvature, 95)
    visualize_color_map(is_edge.astype(int), coords, "Edge Candidates", "edges.png", cmap='hot')

# 运行示例
if __name__ == "__main__":
    analyze_point_cloud("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/data/truck/sparse/0/points3D.ply")
