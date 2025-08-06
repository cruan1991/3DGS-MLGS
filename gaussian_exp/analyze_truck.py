from gaussians_io import load_gaussians_from_ply
from visualizations import plot_alpha_histogram, plot_gaussian_positions

ply_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
output_dir = "renders/994230_analysis2"

print("[1] Loading gaussians...")
gaussians = load_gaussians_from_ply(ply_path, alpha_mode="expit")

print("[2] Plotting alpha histogram...")
plot_alpha_histogram(gaussians['opacity'], save_path=f"{output_dir}/alpha_hist.png")

print("[3] Plotting Gaussian positions colored by alpha...")
plot_gaussian_positions(gaussians['xyz'], gaussians['opacity'], save_path=f"{output_dir}/positions_alpha.png")
