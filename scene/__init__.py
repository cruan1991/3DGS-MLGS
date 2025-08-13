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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            gaussian_ball_path = os.path.join(self.model_path,
                                         "gaussian_ball",
                                         "iteration_" + str(self.loaded_iter),
                                         "point_cloud.ply")
            # Try new path first
            if os.path.exists(gaussian_ball_path):
                self.gaussians.load_ply(gaussian_ball_path, args.train_test_exp)
            else:
                # Fall back to old path for backward compatibility
                old_path = os.path.join(self.model_path,
                                      "point_cloud",
                                      "iteration_" + str(self.loaded_iter),
                                      "point_cloud.ply")
                if os.path.exists(old_path):
                    print("Warning: Loading from old point_cloud directory structure")
                    self.gaussians.load_ply(old_path, args.train_test_exp)
                else:
                    raise FileNotFoundError(f"Could not find gaussian ball file at {gaussian_ball_path} or {old_path}")
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration, suffix=""):
        """Save the scene to a file.
        
        Args:
            iteration (int): The current iteration number
            suffix (str, optional): Optional suffix to add to the filename. Defaults to "".
        """
        output_dir = os.path.join(self.model_path, "gaussian_ball", f"iteration_{iteration}{suffix}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PLY format for visualization
        self.gaussians.save_ply(os.path.join(output_dir, "point_cloud.ply"))
        
        # Save NPZ format for training recovery
        import numpy as np
        with torch.no_grad():
            np.savez(
                os.path.join(output_dir, "point_cloud.npz"),
                xyz=self.gaussians.get_xyz.detach().cpu().numpy(),
                features_dc=self.gaussians._features_dc.detach().cpu().numpy(),
                features_rest=self.gaussians._features_rest.detach().cpu().numpy(),
                opacity=self.gaussians.get_opacity.detach().cpu().numpy(),
                rotation=self.gaussians.get_rotation.detach().cpu().numpy(),
                scaling=self.gaussians.get_scaling.detach().cpu().numpy(),
                max_radii2D=self.gaussians.max_radii2D.detach().cpu().numpy()
            )
        
        # Save exposure data with suffix
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        exposure_file = os.path.join(self.model_path, f"exposure{suffix}.json")
        with open(exposure_file, "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
