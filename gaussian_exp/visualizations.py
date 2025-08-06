# visualizations.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_alpha_histogram(alpha_values, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.hist(alpha_values, bins=100, color='skyblue', edgecolor='black')
    plt.title('Opacity (Alpha) Distribution')
    plt.xlabel('Alpha')
    plt.ylabel('Count')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_gaussian_positions(xyz, alpha, save_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=alpha, cmap='plasma', s=1, alpha=0.7)
    fig.colorbar(p, ax=ax, label='Alpha')
    ax.set_title('Gaussian Positions Colored by Opacity')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()