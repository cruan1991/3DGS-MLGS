import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import logging
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# 设置双重日志输出
class DualLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', buffering=1)  # 行缓冲
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        # 过滤掉 tqdm 的控制字符
        if '\r' not in message and '\033' not in message:
            self.log.write(message)
            self.log.flush()
            
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

class BestResultTracker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.best_psnr = float('-inf')
        self.best_loss = float('inf')
        self.best_psnr_iteration = -1
        self.best_loss_iteration = -1
        self.last_psnr = float('-inf')
        self.last_loss = float('inf')
        
        # Create improvements log file
        self.improvements_log_path = os.path.join(model_path, 'improvements.log')
        self.improvements_logger = logging.getLogger('improvements')
        self.improvements_logger.handlers = []  # Clear any existing handlers
        handler = logging.FileHandler(self.improvements_log_path)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.improvements_logger.addHandler(handler)
        self.improvements_logger.setLevel(logging.INFO)
        self.improvements_logger.propagate = False  # Don't propagate to root logger
        
        # Log header
        self.improvements_logger.info("Training Improvements Log")
        self.improvements_logger.info("-" * 120)
        header = (
            f"{'Iteration':>8} | {'PSNR (dB)':>10} | {'Loss':>10} | {'#Gaussians':>10} | "
            f"{'Best PSNR':>10} | {'Best Loss':>10} | {'Best PSNR Iter':>10} | {'Best Loss Iter':>10} | "
            f"{'Change from Last':>25} | {'Change from Best':>25}"
        )
        self.improvements_logger.info(header)
        self.improvements_logger.info("-" * 120)
        
    def update(self, iteration, psnr, loss, num_gaussians=None):
        """Returns tuple (is_best_psnr, is_best_loss)"""
        is_best_psnr = False
        is_best_loss = False
        should_log = False
        
        # Calculate changes from last iteration
        psnr_change = psnr - self.last_psnr if self.last_psnr != float('-inf') else 0
        loss_change = self.last_loss - loss if self.last_loss != float('inf') else 0
        
        # Calculate changes from best
        psnr_vs_best = psnr - self.best_psnr if self.best_psnr != float('-inf') else 0
        loss_vs_best = self.best_loss - loss if self.best_loss != float('inf') else 0
        
        # Check if this is a new best PSNR
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.best_psnr_iteration = iteration
            is_best_psnr = True
            should_log = True
            
        # Check if this is a new best loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_iteration = iteration
            is_best_loss = True
            should_log = True
        
        # Also log if there's significant improvement from last iteration
        if psnr_change > 0.01 or loss_change > 0.001:  # Thresholds for logging
            should_log = True
            
        # Log if we should
        if should_log:
            msg = (
                f"{iteration:8d} | {psnr:10.4f} | {loss:10.6f} | {num_gaussians:10d} | "
                f"{self.best_psnr:10.4f} | {self.best_loss:10.6f} | {self.best_psnr_iteration:10d} | {self.best_loss_iteration:10d} | "
                f"PSNR: {psnr_change:+7.4f}, Loss: {loss_change:+7.6f} | "
                f"PSNR: {psnr_vs_best:+7.4f}, Loss: {loss_vs_best:+7.6f}"
            )
            
            # Add indicators for best values
            if is_best_psnr:
                msg += " | 🏆 Best PSNR"
            if is_best_loss:
                msg += " | 🏆 Best Loss"
                
            self.improvements_logger.info(msg)
        
        # Update last values
        self.last_psnr = psnr
        self.last_loss = loss
            
        return is_best_psnr, is_best_loss
        
    def save_checkpoint(self, iteration, state, logger=None, is_final=False):
        """Save checkpoint based on type (best_psnr, best_loss, or final)"""
        if is_final:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"final_state.pth")
            # if logger:
            #     logger.info(f"Saving final state checkpoint to: {checkpoint_path}")
        elif iteration == self.best_psnr_iteration:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_psnr.pth")
            # if logger:
                # logger.info(f"Saving best PSNR checkpoint to: {checkpoint_path}")
        elif iteration == self.best_loss_iteration:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_loss.pth")
            # if logger:
                # logger.info(f"Saving best loss checkpoint to: {checkpoint_path}")
        else:
            # Regular checkpoint requested by user
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration}.pth")
            # if logger:
                # logger.info(f"Saving user-requested checkpoint to: {checkpoint_path}")
        
        torch.save(state, checkpoint_path)

def setup_logging(model_path):
    """设置日志系统"""
    # 确保模型目录存在
    os.makedirs(model_path, exist_ok=True)
    
    # 主训练日志
    log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = os.path.join(model_path, log_filename)
    
    # 设置logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 打印日志文件位置
    print(f"Training log: {log_file}")
    print(f"Improvements log: {os.path.join(model_path, 'improvements.log')}")
    
    # 返回logger
    return logging.getLogger(__name__)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    
    # Initialize loggers and best result tracker
    logger = setup_logging(scene.model_path)
    best_tracker = BestResultTracker(scene.model_path)
    logger.info(f"Starting training with model path: {scene.model_path}")
    logger.info(f"Dataset source path: {dataset.source_path if hasattr(dataset, 'source_path') else 'Unknown'}")
    logger.info(f"Optimization settings: iterations={opt.iterations}, densify_from_iter={opt.densify_from_iter}, densify_until_iter={opt.densify_until_iter}")
    
    # Copy original point cloud to output folder if available
    import shutil
    scene_info = dataset if hasattr(dataset, 'ply_path') else None
    if scene_info and hasattr(scene_info, 'ply_path') and os.path.exists(scene_info.ply_path):
        original_ply_name = os.path.basename(scene_info.ply_path)
        copied_ply_path = os.path.join(scene.model_path, f"original_{original_ply_name}")
        shutil.copy2(scene_info.ply_path, copied_ply_path)
        logger.info(f"Original point cloud copied to: {copied_ply_path}")
    
    # Save initial gaussian state
    logger.info("Saving initial gaussian state...")
    scene.save(0, "_initial")  # Save to iteration_0 folder
    logger.info(f"Initial gaussians saved to: {os.path.join(scene.model_path, 'gaussian_ball/iteration_0_initial')}")
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        logger.info(f"Loaded checkpoint from iteration {first_iter}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # 使用 leave=True 确保进度条在完成后保留
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", file=sys.stdout, leave=True)
    first_iter += 1
    
    # Variables to store evaluation results
    last_test_psnr = 0.0
    last_test_loss = 0.0
    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(100)
                
            if iteration == opt.iterations:
                progress_bar.close()

            # Calculate current metrics for improvement tracking
            current_psnr = psnr(image, gt_image).mean().double()
            current_loss = loss.item()
            num_gaussians = gaussians.get_xyz.shape[0]
            
            # Update best tracker with current results
            if best_tracker is not None:
                is_best_psnr, is_best_loss = best_tracker.update(iteration, float(current_psnr), float(current_loss), num_gaussians)
                # 3000次迭代后保存改进的高斯球
                if iteration >= 3000:
                    if is_best_psnr:
                        scene.save(iteration, "_best_psnr")
                    if is_best_loss:
                        scene.save(iteration, "_best_loss")

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                          testing_iterations, scene, render, 
                          (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), 
                          dataset.train_test_exp, logger, best_tracker)
            
            # 按照命令行参数保存
            if iteration in saving_iterations:
                save_msg = f"[ITER {iteration}] Saving Gaussians"
                print(f"\n{save_msg}")
                if logger:
                    logger.info(save_msg)
                scene.save(iteration)

            # 每1000次迭代输出统计信息
            if iteration % 1000 == 0:
                with torch.no_grad():
                    positions = gaussians._xyz.detach().cpu()
                    num_points = positions.shape[0]
                    xyz_min = positions.min(dim=0).values
                    xyz_max = positions.max(dim=0).values
                    xyz_range = xyz_max - xyz_min
                    
                    # 输出高斯球的统计信息
                    stats_info = f"\n{'='*60}\n"
                    stats_info += f"[STATISTICS] Iteration {iteration}\n"
                    stats_info += f"{'='*60}\n"
                    stats_info += f"📊 Training Metrics:\n"
                    stats_info += f"  ➤ Training Loss: {ema_loss_for_log:.7f}\n"
                    stats_info += f"  ➤ Depth Loss: {ema_Ll1depth_for_log:.7f}\n"
                    stats_info += f"\n🔵 Gaussian Statistics:\n"
                    stats_info += f"  ➤ Number of Gaussians: {num_points:,}\n"
                    stats_info += f"  ➤ Position Range:\n"
                    stats_info += f"     x: [{xyz_min[0]:.3f}, {xyz_max[0]:.3f}] (range: {xyz_range[0]:.3f})\n"
                    stats_info += f"     y: [{xyz_min[1]:.3f}, {xyz_max[1]:.3f}] (range: {xyz_range[1]:.3f})\n"
                    stats_info += f"     z: [{xyz_min[2]:.3f}, {xyz_max[2]:.3f}] (range: {xyz_range[2]:.3f})\n"
                    stats_info += f"{'='*60}\n"
                    
                    print(stats_info)
                    if logger:
                        logger.info(stats_info)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                checkpoint_msg = f"[ITER {iteration}] Saving Checkpoint"
                print(f"{checkpoint_msg}")
                logger.info(checkpoint_msg)
                
                # 保存到 model_path/checkpoint/ 目录
                checkpoint_dir = os.path.join(scene.model_path, "checkpoint")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                checkpoint_path = os.path.join(checkpoint_dir, f"chkpnt{iteration}.pth")
                torch.save((gaussians.capture(), iteration), checkpoint_path)
                logger.info(f"Checkpoint saved to: {checkpoint_path}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, logger=None, best_tracker=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Full evaluation on test iterations
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        test_results = None
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        # Convert tensor to numpy array in correct format for tensorboard
                        def prepare_image(img_tensor):
                            # Move to CPU and convert to numpy
                            img_np = img_tensor.detach().cpu().numpy()
                            # Convert from CHW to HWC if needed
                            if img_np.shape[0] in [1, 3, 4]:
                                img_np = np.transpose(img_np, (1, 2, 0))
                            # Scale to 0-255
                            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                            # Ensure 3 channels
                            if len(img_np.shape) == 2:
                                img_np = np.stack([img_np] * 3, axis=-1)
                            elif img_np.shape[-1] == 1:
                                img_np = np.concatenate([img_np] * 3, axis=-1)
                            # Convert back to CHW format for tensorboard
                            img_np = np.transpose(img_np, (2, 0, 1))
                            return img_np

                        # Add images to tensorboard directly as numpy arrays
                        render_np = prepare_image(image)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                          render_np[None], global_step=iteration, dataformats='NCHW')
                        
                        if iteration == testing_iterations[0]:
                            gt_np = prepare_image(gt_image)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                              gt_np[None], global_step=iteration, dataformats='NCHW')
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                if config['name'] == 'test':
                    test_results = {'psnr': float(psnr_test), 'loss': float(l1_test)}
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

        return test_results
    return None

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    # 自动生成 test_iterations 和 save_iterations
    if args.test_iterations is None:
        # 早期的几个关键点 + 从10000开始每5000次迭代
        early_iterations = [1_000, 3_000, 5_000, 7_000]
        auto_iterations = list(range(10_000, args.iterations + 1, 5_000))
        args.test_iterations = early_iterations + auto_iterations
        
    if args.save_iterations is None:
        # 与 test_iterations 保持一致
        args.save_iterations = args.test_iterations.copy()
    
    # 确保最终迭代数也被包含
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")