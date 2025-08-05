import os
import numpy as np
import torch
import torch.nn as nn
from scene import GaussianModel
from scipy.spatial.transform import Rotation as R

# === 参数配置 ===
ckpt_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck120w/checkpoint/chkpnt700000.pth"
output_dir = "./preprocessed_stats_try"
os.makedirs(output_dir, exist_ok=True)

# === 初始化模型 ===
model = GaussianModel(sh_degree=3, optimizer_type='adam')
# (state_dict, it) = torch.load(ckpt_path, map_location="cpu")
(state_dict, it) = torch.load(ckpt_path, map_location='cpu')

if isinstance(state_dict, tuple):  # checkpoint = (model_dict, opt_dict, steps)
    state_dict = state_dict[0]  # 只取模型部分

checkpoint = torch.load(ckpt_path, map_location="cpu")
print("[DEBUG] Type of checkpoint:", type(checkpoint))
print("[DEBUG] Checkpoint content:", checkpoint)
class DummyArgs:
    def __init__(self):
        # 基本参数
        self.percent_dense = 0.01
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.densification_interval = 100
        self.opacity_reset_interval = 100
        self.densify_from_iter = 0
        self.densify_until_iter = 10000
        self.density_thresh = 0.01
        self.max_screen_size = 1e6

        # 学习率相关
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001

        # 训练调度
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.01

        self.iterations = 30000


dummy_args = DummyArgs()
model._exposure = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
model.restore(state_dict, dummy_args)


print("[INFO] xyz:", model._xyz.shape)
print("[INFO] scaling:", model._scaling.shape)
print("[INFO] rotation:", model._rotation.shape)
print(f"[OK] Loaded checkpoint at iteration {it}")

# === 提取高斯参数 ===
means3D = model._xyz.data.cpu().numpy()
scales = model._scaling.data.cpu().numpy()
quats = model._rotation.data.cpu().numpy()

# === Flatness ===
safe_max = np.max(scales, axis=1)
safe_min = np.min(scales, axis=1)
valid = safe_max > 1e-6
flatness = np.zeros_like(safe_max)
flatness[valid] = safe_min[valid] / safe_max[valid]
flatness = np.clip(flatness, 0, 1)

# === Volume ===
volume = np.prod(scales, axis=1)

# === 主轴方向 ===
r = R.from_quat(quats[:, [1, 2, 3, 0]])
principal_axis = r.apply(np.tile([[1, 0, 0]], (quats.shape[0], 1)))

# === 保存 .npz ===
np.savez(os.path.join(output_dir, "gaussians_stats.npz"),
         gauss_xyz=means3D,
         gauss_scale=scales,
         gauss_flatness=flatness,
         gauss_axis=principal_axis,
         gauss_volume=volume)

print("[OK] Saved: gaussians_stats.npz")
