import open3d as o3d
import numpy as np
import plotly.graph_objs as go
from matplotlib import cm

# === 读取点云 ===
pcd = o3d.io.read_point_cloud("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/9f7ecccc-5/gaussian_ball/iteration_233910_best_psnr_and_best_loss/gaussian_ball.ply")
points = np.asarray(pcd.points)
print("原始点数：", len(points))

# === 降采样（可选） ===
if len(points) > 1_000_000:
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    points = np.asarray(pcd.points)
    print("降采样后点数：", len(points))

if points.shape[0] == 0:
    raise RuntimeError("点云为空，无法可视化")

# === 计算 scale proxy（距离中心） ===
center = points.mean(axis=0)
scales = np.linalg.norm(points - center, axis=1)

# === 归一化 scale + 上 colormap ===
try:
    normed = (scales - np.min(scales)) / (np.ptp(scales) + 1e-8)
    colors = cm.plasma(normed)[:, :3]
except Exception as e:
    print("颜色生成失败：", e)
    colors = np.ones_like(points) * 0.5  # 灰色 fallback

# === 构造可视化 ===
scatter = go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(size=2, color=colors, opacity=0.8),
)

fig = go.Figure(data=[scatter])
fig.update_layout(title="Gaussian Ball Visualization (Colored by Scale)",
                  scene=dict(aspectmode="data"))
fig.write_html("gaussian_ball_scaled.html")
print("✅ HTML 已保存：gaussian_ball_scaled.html")
