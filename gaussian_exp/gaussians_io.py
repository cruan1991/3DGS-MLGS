# gaussians_io.py
import numpy as np
from plyfile import PlyData
from scipy.special import expit

def load_gaussians_from_ply(ply_path, alpha_mode="expit"):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex'].data

    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    raw_alpha = vertex['opacity']

    if alpha_mode == "expit":
        opacity = expit(raw_alpha)  # sigmoid(logit alpha)
    elif alpha_mode == "exp":
        opacity = np.exp(raw_alpha)  # exp(log alpha)
    else:
        opacity = raw_alpha

    scale = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
    rgb = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)

    return {
        'xyz': xyz,
        'opacity': opacity,
        'scale': scale,
        'rgb': rgb
    }