#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F

# ======= PSNR =========
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def cal_psnr(img1, img2, max_val=1.0):
    """与3DGS一致的PSNR计算"""
    # 按batch维度分别计算MSE
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    
    # 处理MSE为0的情况
    psnr_values = torch.where(
        mse > 0,
        20 * torch.log10(max_val / torch.sqrt(mse)),
        torch.tensor(float('inf'), device=img1.device)
    )
    
    return psnr_values  # [batch_size, 1]

# ======= SSIM =========
try:
    from kornia.metrics import ssim as kornia_ssim
    def cal_ssim(img1, img2):
        img1, img2 = prepare(img1), prepare(img2)
        val = kornia_ssim(img1, img2, window_size=11)
        return torch.clamp(val.mean(), 0, 1).item()
except ImportError:
    def cal_ssim(img1, img2):
        return -1  # Placeholder if no SSIM lib

# ======= LPIPS =========
try:
    import lpips
    lpips_model = lpips.LPIPS(net='alex').cuda()
    def cal_lpips(img1, img2):
        img1, img2 = prepare(img1), prepare(img2)
        return lpips_model(img1 * 2 - 1, img2 * 2 - 1).item()
except ImportError:
    def cal_lpips(img1, img2):
        return -1  # Placeholder if LPIPS unavailable

# ======= 图像预处理 =========
def prepare(img):
    """确保 img 是 [B,3,H,W]，且值在 [0,1]"""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.max() > 1.0:
        img = img / 255.0
    return img.clamp(0, 1)



# def psnr(img1, img2):
#     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))
