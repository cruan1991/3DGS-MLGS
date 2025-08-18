import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (forces 3D backend)

__all__ = [
    'heatmap_xy',
    'scale_hist',
    'scatter3d',
    'plot_2d_hist'
]

def heatmap_xy(xyz: np.ndarray, out: str, bins: int = 200):
    """XY spatial density heatmap (top‑down)."""
    h, xedges, yedges = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=bins)
    plt.figure(figsize=(6, 5))
    plt.imshow(h.T, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.xlabel('X'); plt.ylabel('Y'); plt.title('XY Density Heatmap')
    plt.colorbar(label='count')
    plt.tight_layout(); plt.savefig(out); plt.close()


def scale_hist(mags: np.ndarray, out: str, bins: int = 100):
    """Histogram of scale magnitudes (log‑space)."""
    plt.figure(figsize=(6, 4))
    plt.hist(np.log10(mags + 1e-8), bins=bins)
    plt.xlabel('log10(scale)'); plt.ylabel('count'); plt.title('Scale Histogram')
    plt.tight_layout(); plt.savefig(out); plt.close()


def scatter3d(points: np.ndarray, out: str, size: int = 2):
    """Simple 3D scatter for hotspot preview."""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout(); plt.savefig(out); plt.close()


def plot_2d_hist(x: np.ndarray, y: np.ndarray, out: str,
                 xlabel: str = 'x', ylabel: str = 'y', bins: int = 120):
    """General 2D histogram helper."""
    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=bins)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.colorbar(label='count')
    plt.tight_layout(); plt.savefig(out); plt.close()
