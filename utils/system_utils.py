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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    """Search for the highest iteration number in the gaussian_ball or point_cloud directory.
    
    Args:
        folder (str): Base directory path (model_path)
        
    Returns:
        int: The highest iteration number found
    """
    # Check both possible directory names
    gaussian_ball_dir = os.path.join(folder, "gaussian_ball")
    point_cloud_dir = os.path.join(folder, "point_cloud")
    
    all_iterations = []
    
    # Check gaussian_ball directory first
    if os.path.exists(gaussian_ball_dir):
        for dirname in os.listdir(gaussian_ball_dir):
            if dirname.startswith("iteration_"):
                # Extract iteration number, handling potential suffixes
                iter_str = dirname.split("_")[1]  # Get the number after "iteration_"
                try:
                    iter_num = int(iter_str)
                    all_iterations.append(iter_num)
                except ValueError:
                    continue
    
    # Fall back to point_cloud directory if needed
    if os.path.exists(point_cloud_dir):
        for dirname in os.listdir(point_cloud_dir):
            if dirname.startswith("iteration_"):
                try:
                    iter_num = int(dirname.split("_")[1])
                    all_iterations.append(iter_num)
                except ValueError:
                    continue
    
    if not all_iterations:
        raise ValueError(f"No valid iteration directories found in {folder}")
        
    return max(all_iterations)
