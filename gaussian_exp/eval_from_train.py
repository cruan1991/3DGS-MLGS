#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_from_train.py
目的：用与 train.py 完全一致的渲染/评估口径，对已有 3DGS 模型做 PSNR 评估。
要点：
- 强制优先使用 COLMAP binary 相机（cameras.bin/images.bin），绕过 cameras.json
- 正确计算 FoVx/FoVy：FoVx=focal2fov(fx,w)，FoVy=focal2fov(fy,h)
- 若 test_cameras 为空，回退到 train_cameras 的小子集（防止 ZeroDivisionError）
- 渲染 pipeline 参数与 train.py 对齐：convert_SHs_python/compute_cov3D_python 等
- 可选：保存第一帧 sanity 渲染图校验相机是否翻转/拉伸
"""

import os, sys
REPO_ROOT = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import os
import sys
import csv
import glob
import argparse
from typing import List, Optional

import torch
from tqdm import tqdm

# === 你的工程根路径（按需改）===
# sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

# === 与 train.py 保持一致的导入 ===
from scene import Scene, GaussianModel
from gaussian_renderer import render as gs_render
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from arguments import ModelParams, PipelineParams
from utils.graphics_utils import focal2fov

# === COLMAP 读取 ===
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary


def find_latest_ply(model_path: str) -> Optional[str]:
    """
    在 model_path 下按常见命名寻找最新的 ply
    兼容两种目录：point_cloud 或 gaussian_ball
    """
    patterns = [
        os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply"),
        os.path.join(model_path, "gaussian_ball", "iteration_*", "gaussian_ball.ply"),
        os.path.join(model_path, "*.ply"),
    ]
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(pat))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def _load_cameras_from_colmap(source_path: str,
                              images_subdir: str = "images",
                              sparse_subdir: str = "sparse/0"):
    """
    从 COLMAP 二进制读取相机（内外参），返回 records 供 Scene 复建。
    不直接构造 Camera 类（不同分支 API 有差异），交由 Scene 内部接口处理。
    """
    sparse_dir = os.path.join(source_path, sparse_subdir)
    images_dir = os.path.join(source_path, images_subdir)

    cams = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    imgs = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))

    # 内参表：cam_id -> dict
    colmap_cams = {}
    for cam_id, intr in cams.items():
        # 适配不同 intr 表达
        w, h = intr.width, intr.height
        if hasattr(intr, "params"):
            fx, fy = intr.params[0], intr.params[1]
            cx = getattr(intr, "params", [None, None, None, None])[2] if len(intr.params) > 2 else None
            cy = getattr(intr, "params", [None, None, None, None])[3] if len(intr.params) > 3 else None
        else:
            fx, fy = intr.fx, intr.fy
            cx, cy = getattr(intr, "cx", None), getattr(intr, "cy", None)
        fovx = focal2fov(fx, w)
        fovy = focal2fov(fy, h)
        colmap_cams[cam_id] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, fovx=fovx, fovy=fovy)

    # 外参 + 关联内参
    records = []
    for _, rec in imgs.items():
        intr = colmap_cams[rec.camera_id]
        records.append(dict(
            image_name=rec.name,
            cam_id=rec.camera_id,
            qvec=rec.qvec,     # world->cam，Scene 内部应处理
            tvec=rec.tvec,
            intr=intr
        ))
    return records, images_dir


def build_dataset_and_pipeline(args):
    """与 train.py 相同的参数对象，保证 pipeline 口径一致。"""
    # 构造一个只含模型与渲染参数的解析器
    parser = argparse.ArgumentParser(add_help=False)
    mp = ModelParams(parser)
    pp = PipelineParams(parser)

    # 这里我们手动塞入必要字段（多数分支允许额外字段不使用）
    tmp_args = parser.parse_args([])
    # ModelParams 里常见需要的字段（视你的分支而定）
    setattr(tmp_args, "model_path", args.model_path)
    setattr(tmp_args, "source_path", args.source_path)
    setattr(tmp_args, "images", "images")
    setattr(tmp_args, "eval", True)
    setattr(tmp_args, "resolution", -1)  # 与训练一致：自动缩放
    setattr(tmp_args, "white_background", args.white_background)
    setattr(tmp_args, "data_device", "cpu")  # 默认

    dataset = mp.extract(tmp_args)
    pipe = pp.extract(tmp_args)

    # 渲染口径与 train.py 对齐
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    # 其他可选 flag（按你分支需要）：
    # pipe.debug = False
    return dataset, pipe


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="输出目录（包含 point_cloud/ 或 gaussian_ball/ 等）")
    parser.add_argument("--source_path", required=True,
                        help="包含 images/ 与 sparse/0/ 的目录（通常等于 model_path 或其上级）")
    parser.add_argument("--ply", default=None, help="显式指定 ply 路径（默认自动寻找最新）")
    parser.add_argument("--n_test", type=int, default=10, help="当没有 test set 时，从 train 里抽多少视角评估")
    parser.add_argument("--stride", type=int, default=2, help="从训练相机抽样的步长")
    parser.add_argument("--save_sanity", action="store_true", help="保存第一帧 sanity 渲染图")
    parser.add_argument("--white_background", action="store_true", help="白背景（默认黑）")
    parser.add_argument("--csv", default=None, help="保存逐帧 PSNR 的 CSV 路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 准备数据与渲染 pipeline（与 train.py 口径一致）
    dataset, pipe = build_dataset_and_pipeline(args)

    # 2) 加载高斯模型
    # 如果你的 GaussianModel 支持第二个参数（看 train.py 第198行）
    try:
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    except Exception:
        gaussians = GaussianModel(dataset.sh_degree)

    ply_path = args.ply or find_latest_ply(args.model_path)
    if ply_path is None:
        print("❌ 没找到 ply，请用 --ply 指定。")
        sys.exit(1)
    print(f"✅ 使用 PLY: {ply_path}")
    gaussians.load_ply(ply_path)

    # 3) 建场景（按 train.py）
    if hasattr(dataset, "data_device"):
        dataset.data_device = "cuda"  # 和 train.py 保持一致
    scene = Scene(dataset, gaussians)

    # 4) 尝试用 COLMAP 二进制覆盖相机（核心修复）
    try:
        cam_recs, images_dir = _load_cameras_from_colmap(args.source_path)
        rebuilt = False
        # 分支A：如果 Scene 有官方重建接口
        if hasattr(scene, "rebuild_from_colmap_records"):
            scene.rebuild_from_colmap_records(cam_recs, images_dir)
            rebuilt = True
        # 分支B：常见分支：提供 _build_cameras_from_colmap
        elif hasattr(scene, "_build_cameras_from_colmap"):
            cams = scene._build_cameras_from_colmap(cam_recs, images_dir)
            # 若没有 test set，则把 train 也用这个重建结果
            scene.train_cameras = cams
            scene.test_cameras = []
            rebuilt = True

        print("✅ 使用 COLMAP 二进制相机" if rebuilt else "ℹ️ 该分支无重建接口，沿用 Scene 默认相机")
    except Exception as e:
        print(f"⚠️  读取 COLMAP 相机失败，沿用 Scene 默认相机：{e}")

    # 5) 取测试相机；若为空则从训练相机抽子集（防止 ZeroDivisionError）
    test_cameras = []
    if hasattr(scene, "getTestCameras"):
        test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("⚠️ 没有测试相机，改用训练相机子集做评估")
        train_cameras = scene.getTrainCameras() if hasattr(scene, "getTrainCameras") else []
        sub = []
        for i in range(0, min(len(train_cameras), args.n_test * args.stride), args.stride):
            sub.append(train_cameras[i])
            if len(sub) >= args.n_test:
                break
        test_cameras = sub

    if not test_cameras:
        print("❌ 无可用相机，退出。")
        sys.exit(1)

    # 背景色
    background = torch.ones(3, device=device) if dataset.white_background else torch.zeros(3, device=device)

    # 6) 评估循环（严格对齐 train.py 的渲染路径）
    l1_test = 0.0
    psnr_test = 0.0
    all_psnr = []

    # 第 1 帧做 sanity 渲染，便于你肉眼看翻转/拉伸
    saved_sanity = False

    # 渲染函数句柄（与 train.py 相同）
    renderFunc = gs_render

    # 可选：保存逐帧 PSNR
    writer = None
    if args.csv:
        writer = csv.writer(open(args.csv, "w", newline=""))
        writer.writerow(["frame_idx", "psnr"])

    for idx, viewpoint in enumerate(tqdm(test_cameras, desc="Evaluating")):
        # 与 train.py 保持一致的调用
        render_pkg = renderFunc(viewpoint, gaussians, pipe, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)

        # 指标
        l1_test += l1_loss(image, gt_image).mean().item()
        ps = psnr(image, gt_image).mean().item()
        psnr_test += ps
        all_psnr.append(ps)

        # 保存 sanity
        if args.save_sanity and not saved_sanity:
            try:
                from utils.image_utils import tensor_to_image
                os.makedirs(os.path.join(args.model_path, "eval_debug"), exist_ok=True)
                tensor_to_image(image).save(os.path.join(args.model_path, "eval_debug", "sanity_view0.png"))
                print("✅ 已保存 sanity_view0.png（检查是否翻转/拉伸）")
            except Exception as e:
                print(f"保存 sanity 图失败：{e}")
            saved_sanity = True

        if writer:
            writer.writerow([idx, ps])

    n = len(test_cameras)
    print(f"✅ L1: {l1_test / n:.6f} | PSNR: {psnr_test / n:.3f} dB | Frames: {n}")

    # 打印前 5 帧 PSNR
    if len(all_psnr) > 5:
        print("样例帧 PSNR（前5）：", [f"{v:.2f}" for v in all_psnr[:5]])


if __name__ == "__main__":
    main()
