import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# === 文件路径 ===
base_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/preprocessed_stats"
point_file = os.path.join(base_path, "pointcloud_stats.npz")
gauss_file = os.path.join(base_path, "gaussians_stats.npz")
os.makedirs("correlation_output", exist_ok=True)

# === 加载数据 ===
print("[INFO] Loading data...")
p_data = np.load(point_file)
g_data = np.load(gauss_file)

pxyz = p_data["point_xyz"]
pnorm = p_data["point_normal"]
pdens = p_data["point_density"]
pcurv = p_data["point_curvature"]

gxyz = g_data["gauss_xyz"]
gscale = g_data["gauss_scale"]
gflat = g_data["gauss_flatness"]
gaxis = g_data["gauss_axis"]
gvol = g_data["gauss_volume"]

# === 空间匹配点云和高斯球 ===
print("[INFO] Matching points to nearest Gaussians...")
tree = cKDTree(gxyz)
_, idx = tree.query(pxyz, k=1)

# 获取匹配的高斯属性
matched_axis = gaxis[idx]
matched_volume = gvol[idx]
matched_flatness = gflat[idx]

# === 分析对齐度：法向 vs 高斯主轴方向 ===
align = np.abs(np.sum(pnorm * matched_axis, axis=1))  # cosine similarity

# === 可视化和统计 ===
def plot_and_corr(x, y, xlabel, ylabel, name):
    r, _ = pearsonr(x, y)
    plt.figure()
    plt.scatter(x, y, s=1, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel} (r={r:.3f})")
    plt.savefig(f"correlation_output/{name}.png")
    plt.close()
    print(f"[OK] {name}.png saved. r = {r:.3f}")

plot_and_corr(align, matched_flatness, "Normal-Gaussian Alignment", "Flatness", "align_vs_flatness")
plot_and_corr(align, matched_volume, "Normal-Gaussian Alignment", "Volume", "align_vs_volume")
plot_and_corr(pdens, matched_volume, "Point Density", "Gaussian Volume", "density_vs_volume")
plot_and_corr(pcurv, matched_flatness, "Point Curvature", "Gaussian Flatness", "curvature_vs_flatness")

print("[DONE] All correlations analyzed.")
