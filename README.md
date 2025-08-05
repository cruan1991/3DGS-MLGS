# 3DGS-MLGS

This is a research-oriented fork of [3D Gaussian Splatting (CVPR 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), focusing on hierarchical structure modeling, Gaussian insertion strategies, and redundancy-aware pruning for neural 3D scene representation.

## 🚀 Features

- 📐 **Hierarchical Gaussian Splatting**: Multi-resolution modeling for dense and sparse regions.
- 🎯 **Insertable Gaussian Modules**: Inject new Gaussians during training based on structural cues.
- 🧽 **Redundancy-Aware Pruning**: Remove low-activation or ineffective Gaussians post-optimization.
- 🧠 **Pluggable for downstream tasks**: E.g., compression, rendering, or bio-inspired storage.

## 🔧 Getting Started

This project builds upon the original 3DGS implementation. To install dependencies and visualize results:

```bash
conda env create -f environment.yml
conda activate gaussian-splatting
python train.py -s path/to/scene
```

## 🧪 Research Direction

This fork is currently used to explore:

- Structure-aware latent modeling for controllable splatting.
- Integration of point cloud priors and density-aware insertion.
- Visualization and analysis of Gaussian activity levels.

## 📂 Acknowledgements

This repo is based on the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and inherits its core rendering backend. Full credit to the original authors.

