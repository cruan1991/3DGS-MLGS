# filter_and_save_ply.py
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
from scipy.special import expit

def filter_and_save_ply(input_path, output_path, alpha_thresh=0.05, alpha_mode="expit"):
    print(f"🔍 Loading {input_path}")
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex'].data

    raw_alpha = vertex['opacity']
    if alpha_mode == "expit":
        opacity = expit(raw_alpha)
    elif alpha_mode == "exp":
        opacity = np.exp(raw_alpha)
    else:
        opacity = raw_alpha

    print(f"📊 Filtering gaussians with alpha > {alpha_thresh}...")
    mask = opacity > alpha_thresh
    filtered_data = vertex[mask]

    print(f"✅ Remaining gaussians: {len(filtered_data)} / {len(vertex)}")
    el = PlyElement.describe(filtered_data, 'vertex')
    PlyData([el], text=True).write(output_path)
    print(f"💾 Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Input PLY file path")
    parser.add_argument("output_path", type=str, help="Filtered output PLY file path")
    parser.add_argument("--alpha_thresh", type=float, default=0.05, help="Alpha threshold")
    parser.add_argument("--alpha_mode", type=str, default="expit", choices=["expit", "exp", "raw"], help="Alpha interpretation mode")
    args = parser.parse_args()

    filter_and_save_ply(args.input_path, args.output_path, args.alpha_thresh, args.alpha_mode)
