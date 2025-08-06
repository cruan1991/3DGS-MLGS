import sys
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")
import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement

def filter_gaussians(input_ply, alpha, output_path):
    print(f"🔍 Loading .ply file: {input_ply}")
    plydata = PlyData.read(input_ply)
    vertex = plydata['vertex']
    
    # Get opacity values
    opacities = vertex['opacity']
    
    # Create mask for points with opacity > alpha
    mask = opacities > alpha
    
    # Count points
    total_points = len(opacities)
    kept_points = np.sum(mask)
    removed_points = total_points - kept_points
    
    print(f"📊 Statistics:")
    print(f"  Total points: {total_points}")
    print(f"  Kept points: {kept_points} ({kept_points/total_points*100:.2f}%)")
    print(f"  Removed points: {removed_points} ({removed_points/total_points*100:.2f}%)")
    
    # Filter all vertex properties
    filtered_data = vertex[mask]
    
    # Create new PLY file
    new_plydata = PlyData([PlyElement.describe(filtered_data, 'vertex')], text=True)
    
    # Save filtered PLY
    print(f"💾 Saving filtered .ply to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_plydata.write(output_path)
    print("✅ Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply-path', type=str, required=True, help='Input .ply file path')
    parser.add_argument('--alpha', type=float, required=True, help='Opacity threshold (points with opacity <= alpha will be removed)')
    parser.add_argument('--output-path', type=str, required=True, help='Output .ply file path')
    args = parser.parse_args()
    
    filter_gaussians(args.ply_path, args.alpha, args.output_path) 