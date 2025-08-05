from gaussian_analysis import analyze_gaussians
import numpy as np
from plyfile import PlyData
import torch

class LoadedGaussianModel:
    def __init__(self, ply_path):
        plydata = PlyData.read(ply_path)
        v = plydata['vertex'].data

        self._xyz = torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1), dtype=torch.float32)
        self._scaling = torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1), dtype=torch.float32)
        self._rotation = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1), dtype=torch.float32)
        self._opacity = torch.tensor(v['opacity'], dtype=torch.float32)

        # SH特征处理（一般前3维是 DC，后面是AC）
        sh_dim = [k for k in v.dtype.names if k.startswith('f_dc_') or k.startswith('f_rest_')]
        features = np.stack([v[k] for k in sh_dim], axis=-1)
        self._features_dc = torch.tensor(features[:, :3], dtype=torch.float32)
        self._features_rest = torch.tensor(features[:, 3:], dtype=torch.float32)

# 使用方法
model = LoadedGaussianModel("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck120w/point_cloud/iteration_740000/point_cloud.ply")
results = analyze_gaussians(model)
