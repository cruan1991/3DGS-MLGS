import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R

output_dir = "gaussian_analysis_output"
os.makedirs(output_dir, exist_ok=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def analyze_gaussians(model):
    means3D = model._xyz.data.cpu().numpy()
    scales = model._scaling.data.cpu().numpy()
    quats = model._rotation.data.cpu().numpy()
    opacities = model._opacity.data.cpu().numpy()
    features_dc = model._features_dc.data.cpu().numpy()
    features_rest = model._features_rest.data.cpu().numpy()
    

    print(f"[INFO] Total Gaussians: {len(means3D)}")

    # Spatial distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(means3D[:, 0], means3D[:, 1], means3D[:, 2], s=0.01)
    ax.set_title("Gaussian Spatial Distribution")
    plt.savefig(os.path.join(output_dir, "spatial_distribution.png"))
    plt.close()

    # XY heatmap
    for axes, name in [((0, 1), "xy"), ((0, 2), "xz"), ((1, 2), "yz")]:
        heatmap, _, _ = np.histogram2d(means3D[:, axes[0]], means3D[:, axes[1]], bins=200)
        fig = plt.figure()
        plt.imshow(heatmap.T, origin='lower', cmap='hot', aspect='auto')
        plt.title(f"Position Density Heatmap ({name.upper()} slice)")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"position_density_heatmap_{name}.png"))
        plt.close()

    # Scale distribution
    fig = plt.figure()
    for i, label in enumerate(['x', 'y', 'z']):
        plt.hist(scales[:, i], bins=100, alpha=0.6, label=label)
    plt.title("Scale Distribution")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "scale_distribution.png"))
    plt.close()

    # Flatness
    safe_max = np.max(scales, axis=1)
    safe_min = np.min(scales, axis=1)
    valid = safe_max > 1e-6
    flatness = np.zeros_like(safe_max)
    flatness[valid] = safe_min[valid] / safe_max[valid]
    flatness = np.clip(flatness, 0, 1)

    plt.hist(flatness, bins=100)
    plt.title("Flatness Ratio Distribution (min/max scale)")
    plt.savefig(os.path.join(output_dir, "flatness_distribution.png"))
    plt.close()

    plt.hist(flatness, bins=100, range=(0, 0.2))
    plt.title("Flatness Ratio Distribution (Zoomed <0.2)")
    plt.savefig(os.path.join(output_dir, "flatness_distribution_zoomed.png"))
    plt.close()

    # Opacity
    sig_opacities = sigmoid(opacities)
    plt.hist(sig_opacities, bins=100)
    plt.title("Opacity Distribution (After Sigmoid)")
    plt.savefig(os.path.join(output_dir, "opacity_distribution.png"))
    plt.close()

    # Feature vector clustering
    features = np.concatenate([features_dc, features_rest], axis=1)
    features_pca = PCA(n_components=2).fit_transform(features)
    kmeans_feat = KMeans(n_clusters=5, random_state=0).fit(features_pca)
    labels_feat = kmeans_feat.labels_

    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels_feat, s=1)
    plt.title("Feature Vector Clustering")
    plt.savefig(os.path.join(output_dir, "feature_clustering.png"))
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(means3D[:, 0], means3D[:, 1], means3D[:, 2], c=labels_feat, s=0.01)
    ax.set_title("Clustered Gaussians in 3D Space")
    plt.savefig(os.path.join(output_dir, "feature_clusters_3d.png"))
    plt.close()

    # Principal Axis Orientation
    r = R.from_quat(quats[:, [1, 2, 3, 0]])
    vecs = r.apply(np.tile([[1, 0, 0]], (quats.shape[0], 1)))
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=0.1)
    ax.set_title("Principal Axis Orientation (Unit Sphere)")
    plt.savefig(os.path.join(output_dir, "principal_axis_sphere.png"))
    plt.close()

    # RGB Distribution
    fig = plt.figure()
    for i, color in enumerate(['R', 'G', 'B']):
        plt.hist(features_dc[:, i], bins=100, alpha=0.6, label=color)
    plt.title("RGB Distribution from Features DC")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "rgb_distribution.png"))
    plt.close()

    # Position-based clustering
    kmeans_xyz = KMeans(n_clusters=6, random_state=42).fit(means3D)
    labels_xyz = kmeans_xyz.labels_

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(means3D[:, 0], means3D[:, 1], means3D[:, 2], c=labels_xyz, s=0.01)
    ax.set_title("Spatial Clustering (by Position)")
    plt.savefig(os.path.join(output_dir, "spatial_clusters_3d.png"))
    plt.close()

    unique, counts = np.unique(labels_xyz, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Spatial Cluster Gaussian Counts")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Gaussians")
    plt.savefig(os.path.join(output_dir, "spatial_cluster_density_map.png"))
    plt.close()

    # Cluster-wise stats
    cluster_ids = np.unique(labels_xyz)
    scale_means = []
    flatness_means = []
    opacity_means = []

    for cid in cluster_ids:
        idx = labels_xyz == cid
        scale_means.append(np.mean(scales[idx], axis=0))
        flatness_means.append(np.mean(flatness[idx]))
        opacity_means.append(np.mean(sig_opacities[idx]))

    scale_means = np.array(scale_means)

    for i, label in enumerate(['X', 'Y', 'Z']):
        plt.bar(cluster_ids, scale_means[:, i], alpha=0.7, label=label)
    plt.title("Cluster-wise Average Scale")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "cluster_avg_scales.png"))
    plt.close()

    plt.bar(cluster_ids, flatness_means)
    plt.title("Cluster-wise Average Flatness")
    plt.savefig(os.path.join(output_dir, "cluster_avg_flatness.png"))
    plt.close()

    plt.bar(cluster_ids, opacity_means)
    plt.title("Cluster-wise Average Opacity (Sigmoid)")
    plt.savefig(os.path.join(output_dir, "cluster_avg_opacity.png"))
    plt.close()

    ### 8. Save boolean masks for each cluster ###
    for cid in cluster_ids:
        mask = (labels_xyz == cid)
        np.save(os.path.join(output_dir, f"cluster_mask_{cid}.npy"), mask)
    print("[OK] Cluster masks saved.")

    ### 9. Extreme Gaussian analysis ###
    volumes = np.prod(scales, axis=1)
    idx_flat = np.argmin(flatness)
    idx_transp = np.argmin(sig_opacities)
    idx_opaque = np.argmax(sig_opacities)
    idx_vol = np.argmax(volumes)

    idxs = [idx_flat, idx_transp, idx_opaque, idx_vol]
    labels = ["Most Flat", "Most Transparent", "Most Opaque", "Largest Volume"]
    colors = ["blue", "cyan", "red", "green"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(means3D[:, 0], means3D[:, 1], means3D[:, 2], s=0.005, alpha=0.2, label='All Gaussians')

    for i, idx in enumerate(idxs):
        ax.scatter(means3D[idx, 0], means3D[idx, 1], means3D[idx, 2], s=50, c=colors[i], label=labels[i])
        print(f"[INFO] {labels[i]} Gaussian:")
        print(f"    Location: {means3D[idx]}")
        print(f"    Scale: {scales[idx]}")
        print(f"    Flatness: {flatness[idx]:.4f}")
        print(f"    Opacity(sigmoid): {sig_opacities[idx]:.4f}")
        print(f"    Volume: {volumes[idx]:.4f}")

    ax.set_title("Extreme Gaussians Highlighted")
    ax.legend()
    plt.savefig(os.path.join(output_dir, "extreme_gaussians.png"))
    plt.close()
    print("[OK] Extreme Gaussians visualized.")
