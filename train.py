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
# ========== 修改1: 使用你的image_utils.py中的函数，避免命名冲突 ==========
from utils.image_utils import cal_psnr as psnr_fn, cal_ssim as ssim_fn, cal_lpips as lpips_fn
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import logging
from datetime import datetime
import csv

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

# ========== 修改2: 替换原有的BestResultTracker为ComprehensiveTracker ==========
class ComprehensiveTracker:
    def __init__(self, model_path):
        self.model_path = model_path
        
        # ========== 分别追踪每个指标的最佳值 ==========
        self.best_psnr = float('-inf')
        self.best_ssim = float('-inf')  
        self.best_lpips = float('inf')
        self.best_loss = float('inf')
        
        # 对应的最佳迭代
        self.best_psnr_iteration = -1
        self.best_ssim_iteration = -1
        self.best_lpips_iteration = -1
        self.best_loss_iteration = -1
        
        # 增量改进跟踪
        self.last_recorded_psnr = float('-inf')
        self.last_recorded_loss = float('inf')
        
        # 高斯球分布统计
        self.densify_events = []
        self.last_gaussian_count = 0
        
        # 创建统一的CSV日志
        self.main_csv_path = os.path.join(model_path, 'training_metrics.csv')
        self.main_csv_file = open(self.main_csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.main_csv_file)
        
        # ========== CSV头部（为每个指标添加best字段） ==========
        headers = [
            "iteration", "train_psnr", "train_ssim", "train_lpips", "train_loss",
            "test_psnr", "test_ssim", "test_lpips", "test_loss",
            "num_gaussians", "gaussian_change", "densify_events_since_last",
            "avg_opacity", "opacity_std", 
            "avg_scale_x", "avg_scale_y", "avg_scale_z",
            "scale_std_x", "scale_std_y", "scale_std_z",
            "position_spread_x", "position_spread_y", "position_spread_z",
            "sh_degree", "learning_rate_xyz", "learning_rate_sh", "significant_improvement",
            "is_best_psnr", "is_best_ssim", "is_best_lpips", "is_best_loss"
        ]
        self.csv_writer.writerow(headers)
        
        # 创建统一的日志文件
        self.log_path = os.path.join(model_path, 'training_complete.log')
        self.logger = logging.getLogger('comprehensive_tracker')
        self.logger.handlers = []
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        self.logger.info("=" * 100)
        self.logger.info("COMPREHENSIVE TRAINING TRACKER INITIALIZED")
        self.logger.info("=" * 100)
        
    def record_densify_event(self, iteration, event_type, count_before, count_after):
        """记录密化事件"""
        event = {
            'iteration': iteration,
            'type': event_type,
            'count_before': count_before,
            'count_after': count_after,
            'change': count_after - count_before
        }
        self.densify_events.append(event)
        
    def compute_gaussian_statistics(self, gaussians):
        """计算高斯球的详细统计信息"""
        with torch.no_grad():
            positions = gaussians.get_xyz
            scales = gaussians.get_scaling
            opacities = gaussians.get_opacity
            
            stats = {
                'num_gaussians': positions.shape[0],
                'avg_opacity': float(opacities.mean()),
                'opacity_std': float(opacities.std()),
                'avg_scale': scales.mean(dim=0).tolist(),
                'scale_std': scales.std(dim=0).tolist(),
                'position_spread': (positions.max(dim=0)[0] - positions.min(dim=0)[0]).tolist()
            }
            return stats
    
    def update(self, iteration, train_metrics, test_metrics=None, gaussians=None, learning_rates=None):
        """更新所有指标"""
        
        # 计算高斯球统计
        gauss_stats = self.compute_gaussian_statistics(gaussians) if gaussians else {}
        gaussian_change = gauss_stats.get('num_gaussians', 0) - self.last_gaussian_count
        
        # 计算自上次记录以来的密化事件
        recent_densify_events = len([e for e in self.densify_events if e['iteration'] > iteration - 200])
        
        # ========== 增量改进检测，调整阈值 ==========
        psnr_change = train_metrics.get('psnr', 0) - self.last_recorded_psnr if self.last_recorded_psnr != float('-inf') else 0
        loss_change = self.last_recorded_loss - train_metrics.get('loss', 0) if self.last_recorded_loss != float('inf') else 0
        
        # 调整阈值：PSNR改进0.5dB以上，或loss改进0.01以上才记录
        significant_improvement = (psnr_change > 0.5 or loss_change > 0.01)
        
        # ========== 分别检查每个指标的最佳值 ==========
        is_best_psnr = False
        is_best_ssim = False  
        is_best_lpips = False
        is_best_loss = False
        
        # 检查训练PSNR和Loss（基于训练数据）
        current_train_psnr = train_metrics.get('psnr', 0)
        current_train_loss = train_metrics.get('loss', float('inf'))
        
        if current_train_psnr > self.best_psnr:
            self.best_psnr = current_train_psnr
            self.best_psnr_iteration = iteration
            is_best_psnr = True
            
        if current_train_loss < self.best_loss:
            self.best_loss = current_train_loss
            self.best_loss_iteration = iteration
            is_best_loss = True
        
        # 检查测试SSIM和LPIPS（基于测试数据，如果有的话）
        if test_metrics:
            current_test_ssim = test_metrics.get('ssim', 0)
            current_test_lpips = test_metrics.get('lpips', float('inf'))
            
            if current_test_ssim > self.best_ssim:
                self.best_ssim = current_test_ssim
                self.best_ssim_iteration = iteration
                is_best_ssim = True
                
            if current_test_lpips < self.best_lpips:
                self.best_lpips = current_test_lpips
                self.best_lpips_iteration = iteration
                is_best_lpips = True
        
        # ========== 写入CSV，包含所有best标志 ==========
        row = [
            iteration,
            train_metrics.get('psnr', ''), train_metrics.get('ssim', ''), 
            train_metrics.get('lpips', ''), train_metrics.get('loss', ''),
            test_metrics.get('psnr', '') if test_metrics else '',
            test_metrics.get('ssim', '') if test_metrics else '',
            test_metrics.get('lpips', '') if test_metrics else '',
            test_metrics.get('loss', '') if test_metrics else '',
            gauss_stats.get('num_gaussians', ''),
            gaussian_change,
            recent_densify_events,
            gauss_stats.get('avg_opacity', ''),
            gauss_stats.get('opacity_std', ''),
            *gauss_stats.get('avg_scale', ['', '', '']),
            *gauss_stats.get('scale_std', ['', '', '']),
            *gauss_stats.get('position_spread', ['', '', '']),
            gaussians.active_sh_degree if gaussians else '',
            learning_rates.get('xyz', '') if learning_rates else '',
            learning_rates.get('sh', '') if learning_rates else '',
            int(significant_improvement),
            int(is_best_psnr),
            int(is_best_ssim),
            int(is_best_lpips),
            int(is_best_loss)
        ]
        self.csv_writer.writerow(row)
        self.main_csv_file.flush()
        
        # ========== 决定是否记录日志（任何best或重要事件） ==========
        should_log = (
            iteration % 1000 == 0 or 
            is_best_psnr or
            is_best_ssim or
            is_best_lpips or 
            is_best_loss or
            recent_densify_events > 0 or 
            significant_improvement
        )
        
        if should_log:
            self.log_comprehensive_update(iteration, train_metrics, test_metrics, gauss_stats, 
                                        recent_densify_events, significant_improvement, psnr_change, loss_change,
                                        is_best_psnr, is_best_ssim, is_best_lpips, is_best_loss)
            
            # 更新上次记录的值
            self.last_recorded_psnr = train_metrics.get('psnr', self.last_recorded_psnr)
            self.last_recorded_loss = train_metrics.get('loss', self.last_recorded_loss)
        
        # 更新状态
        self.last_gaussian_count = gauss_stats.get('num_gaussians', 0)
        
        return is_best_psnr, is_best_ssim, is_best_lpips, is_best_loss  # 返回四个标志
    
    def log_comprehensive_update(self, iteration, train_metrics, test_metrics, gauss_stats, 
                               recent_densify_events, significant_improvement=False, psnr_change=0, loss_change=0,
                               is_best_psnr=False, is_best_ssim=False, is_best_lpips=False, is_best_loss=False):
        """记录综合更新信息"""
        
        msg = f"\n{'='*80}\n"
        msg += f"ITERATION {iteration:,}"
        
        # 添加所有的最佳标志
        best_flags = []
        if significant_improvement:
            best_flags.append("📈 SIGNIFICANT IMPROVEMENT")
        if is_best_psnr:
            best_flags.append("🏆 BEST PSNR")
        if is_best_ssim:
            best_flags.append("🏆 BEST SSIM")
        if is_best_lpips:
            best_flags.append("🏆 BEST LPIPS")
        if is_best_loss:
            best_flags.append("🏆 BEST LOSS")
            
        if best_flags:
            msg += f" {' '.join(best_flags)}"
        msg += f"\n{'='*80}\n"
        
        # 训练指标
        msg += f"🔵 TRAINING METRICS:\n"
        msg += f"  PSNR: {train_metrics.get('psnr', 'N/A'):>8.3f} dB"
        if significant_improvement and psnr_change > 0:
            msg += f" (↑{psnr_change:+.3f})"
        if is_best_psnr:
            msg += f" 🏆"
        msg += f"\n"
        
        msg += f"  SSIM: {train_metrics.get('ssim', 'N/A'):>8.4f}"
        if is_best_ssim and test_metrics:  # SSIM best基于测试数据
            msg += f" 🏆"
        msg += f"\n"
        
        msg += f"  LPIPS: {train_metrics.get('lpips', 'N/A'):>7.4f}"
        if is_best_lpips and test_metrics:  # LPIPS best基于测试数据
            msg += f" 🏆"
        msg += f"\n"
        
        msg += f"  Loss: {train_metrics.get('loss', 'N/A'):>8.6f}"
        if significant_improvement and loss_change > 0:
            msg += f" (↓{loss_change:+.6f})"
        if is_best_loss:
            msg += f" 🏆"
        msg += f"\n"
        
        # ========== 显示所有历史最佳值 ==========
        msg += f"\n📈 HISTORICAL BEST VALUES:\n"
        msg += f"  Best PSNR: {self.best_psnr:.4f} dB (iter {self.best_psnr_iteration})\n"
        msg += f"  Best SSIM: {self.best_ssim:.4f} (iter {self.best_ssim_iteration})\n"
        msg += f"  Best LPIPS: {self.best_lpips:.4f} (iter {self.best_lpips_iteration})\n"
        msg += f"  Best Loss: {self.best_loss:.6f} (iter {self.best_loss_iteration})\n"
        
        # 当前测试指标（如果有）
        if test_metrics:
            msg += f"\n📊 CURRENT TEST METRICS:\n"
            msg += f"  PSNR: {test_metrics['psnr']:>8.3f} dB\n"
            msg += f"  SSIM: {test_metrics['ssim']:>8.4f}\n"
            msg += f"  LPIPS: {test_metrics['lpips']:>7.4f}\n"
            msg += f"  Loss: {test_metrics['loss']:>7.6f}\n"
        
        # 高斯球统计
        msg += f"\n🟡 GAUSSIAN STATISTICS:\n"
        msg += f"  Count: {gauss_stats.get('num_gaussians', 0):>10,} (Δ: {gauss_stats.get('num_gaussians', 0) - self.last_gaussian_count:+,})\n"
        msg += f"  Opacity: μ={gauss_stats.get('avg_opacity', 0):.4f}, σ={gauss_stats.get('opacity_std', 0):.4f}\n"
        if 'avg_scale' in gauss_stats:
            scales = gauss_stats['avg_scale']
            msg += f"  Scale: x={scales[0]:.4f}, y={scales[1]:.4f}, z={scales[2]:.4f}\n"
        
        # 密化事件
        if recent_densify_events > 0:
            msg += f"\n🔄 DENSIFICATION:\n"
            msg += f"  Recent events: {recent_densify_events}\n"
            recent_events = [e for e in self.densify_events if e['iteration'] > iteration - 200][-3:]
            for event in recent_events:
                msg += f"    {event['type']} @ iter {event['iteration']}: {event['count_before']:,} → {event['count_after']:,}\n"
            
        msg += f"{'='*80}\n"
        
        self.logger.info(msg)
    
    def close(self):
        """关闭所有文件句柄"""
        self.main_csv_file.close()
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()

# ========== 修改3: 定义compute_metrics函数 ==========
def compute_metrics(img1, img2):
    """使用image_utils中的函数计算所有指标"""
    return {
        'psnr': psnr_fn(img1, img2).mean().item(),
        'ssim': ssim_fn(img1, img2).mean().item(),
        'lpips': lpips_fn(img1, img2).mean().item()
    }

# ========== 修改4: 增强的测试评估函数 ==========
def run_comprehensive_test(scene, renderFunc, renderArgs, train_test_exp):
    """运行全面的测试评估"""
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        return None
        
    all_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'loss': []}
    
    for viewpoint in test_cameras:
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        
        if train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
        
        metrics = compute_metrics(image, gt_image)
        metrics['loss'] = l1_loss(image, gt_image).mean().item()
        
        for key, value in metrics.items():
            all_metrics[key].append(value)
    
    # 计算平均值
    avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
    return avg_metrics

def setup_logging(model_path):
    """设置日志系统"""
    os.makedirs(model_path, exist_ok=True)
    
    log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = os.path.join(model_path, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print(f"Training log: {log_file}")
    return logging.getLogger(__name__)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    
    # ========== 修改5: 使用新的ComprehensiveTracker ==========
    logger = setup_logging(scene.model_path)
    tracker = ComprehensiveTracker(scene.model_path)
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
    scene.save(0, "_initial")
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

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", file=sys.stdout, leave=True)
    first_iter += 1
    
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

            progress_bar.update(1)

            if iteration % 200 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                
            if iteration == opt.iterations:
                progress_bar.close()

            # ========== 修改6: 计算完整的训练指标 ==========
            train_metrics = compute_metrics(image, gt_image)
            train_metrics['loss'] = loss.item()

            # ========== 修改7: 在测试迭代中计算完整测试指标 ==========
            test_metrics = None
            if iteration in testing_iterations:
                test_metrics = run_comprehensive_test(scene, render, 
                                                    (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), 
                                                    dataset.train_test_exp)

            # 获取学习率信息
            learning_rates = {
                'xyz': gaussians.optimizer.param_groups[0]['lr'],
                'sh': gaussians.optimizer.param_groups[1]['lr'] if len(gaussians.optimizer.param_groups) > 1 else 0
            }

            # ========== 修改8: 使用新的tracker更新，处理四个返回值 ==========
            is_best_psnr, is_best_ssim, is_best_lpips, is_best_loss = tracker.update(iteration, train_metrics, test_metrics, gaussians, learning_rates)

            # Log training metrics to TensorBoard
            tb_writer.add_scalar("train/psnr", train_metrics['psnr'], iteration)
            tb_writer.add_scalar("train/ssim", train_metrics['ssim'], iteration)
            tb_writer.add_scalar("train/lpips", train_metrics['lpips'], iteration)
            tb_writer.add_scalar("train/loss", train_metrics['loss'], iteration)
            tb_writer.add_scalar("train/num_gaussians", gaussians.get_xyz.shape[0], iteration)

            # ========== 为每个最佳指标分别保存检查点 ==========
            if is_best_psnr:
                scene.save(iteration, "_best_psnr")
            if is_best_ssim:
                scene.save(iteration, "_best_ssim")
            if is_best_lpips:
                scene.save(iteration, "_best_lpips")
            if is_best_loss:
                scene.save(iteration, "_best_loss")

            # Regular checkpoint saving
            if iteration in checkpoint_iterations:
                scene.save(iteration)
                
            # Save final state
            if iteration == opt.iterations:
                scene.save(iteration, "_final")
                logger.info(f"Final state saved to: {os.path.join(scene.model_path, f'gaussian_ball/iteration_{iteration}_final')}")
                tracker.close()

            # ========== 修改9: 增强的training_report ==========
            enhanced_training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                                   testing_iterations, scene, render, 
                                   (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), 
                                   dataset.train_test_exp, logger, tracker, test_metrics)
            
            # ========== 新增: 主训练日志每1000次迭代固定输出 ==========
            if iteration % 1000 == 0:
                with torch.no_grad():
                    positions = gaussians._xyz.detach().cpu()
                    num_points = positions.shape[0]
                    xyz_min = positions.min(dim=0).values
                    xyz_max = positions.max(dim=0).values
                    xyz_range = xyz_max - xyz_min

                    # 主训练日志的规律输出
                    stats_info = f"\n{'='*80}\n"
                    stats_info += f"[TRAINING PROGRESS] Iteration {iteration:,}\n"
                    stats_info += f"{'='*80}\n"
                    stats_info += f"📊 Current Metrics:\n"
                    stats_info += f"  ➤ Train PSNR: {train_metrics['psnr']:.4f} dB\n"  
                    stats_info += f"  ➤ Train SSIM: {train_metrics['ssim']:.4f}\n"
                    stats_info += f"  ➤ Train LPIPS: {train_metrics['lpips']:.4f}\n"
                    stats_info += f"  ➤ Train Loss: {ema_loss_for_log:.7f}\n"
                    stats_info += f"  ➤ Depth Loss: {ema_Ll1depth_for_log:.7f}\n"
                    
                    if test_metrics:  # 如果有测试结果
                        stats_info += f"\n📈 Test Metrics:\n"
                        stats_info += f"  ➤ Test PSNR: {test_metrics['psnr']:.4f} dB\n"
                        stats_info += f"  ➤ Test SSIM: {test_metrics['ssim']:.4f}\n"
                        stats_info += f"  ➤ Test LPIPS: {test_metrics['lpips']:.4f}\n"
                        stats_info += f"  ➤ Test Loss: {test_metrics['loss']:.6f}\n"
                        
                    stats_info += f"\n🔵 Gaussian Statistics:\n"
                    stats_info += f"  ➤ Count: {num_points:,}\n"
                    stats_info += f"  ➤ Position Range:\n"
                    stats_info += f"     x: [{xyz_min[0]:.3f}, {xyz_max[0]:.3f}] (range: {xyz_range[0]:.3f})\n"
                    stats_info += f"     y: [{xyz_min[1]:.3f}, {xyz_max[1]:.3f}] (range: {xyz_range[1]:.3f})\n"
                    stats_info += f"     z: [{xyz_min[2]:.3f}, {xyz_max[2]:.3f}] (range: {xyz_range[2]:.3f})\n"
                    
                    stats_info += f"\n📈 Historical Best:\n"
                    stats_info += f"  ➤ Best PSNR: {tracker.best_psnr:.4f} dB (iter {tracker.best_psnr_iteration})\n"
                    stats_info += f"  ➤ Best SSIM: {tracker.best_ssim:.4f} (iter {tracker.best_ssim_iteration})\n"
                    stats_info += f"  ➤ Best LPIPS: {tracker.best_lpips:.4f} (iter {tracker.best_lpips_iteration})\n"
                    stats_info += f"  ➤ Best Loss: {tracker.best_loss:.6f} (iter {tracker.best_loss_iteration})\n"
                    
                    stats_info += f"{'='*80}\n"
                    
                    # 同时输出到控制台和主训练日志
                    print(stats_info)
                    logger.info(stats_info)
                
            if (iteration in saving_iterations):
                save_msg = f"[ITER {iteration}] Saving Gaussians"
                print(f"{save_msg}")
                logger.info(save_msg)
                scene.save(iteration)

            # ========== 修改10: 密化时记录事件 ==========
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # 记录密化前后的高斯数量
                    count_before = gaussians.get_xyz.shape[0]
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    count_after = gaussians.get_xyz.shape[0]
                    
                    # 记录密化事件
                    if count_after != count_before:
                        tracker.record_densify_event(iteration, 'densify_prune', count_before, count_after)
                
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
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def enhanced_training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, train_test_exp, logger=None, tracker=None, test_metrics=None):
    """增强版的training_report，支持完整指标"""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Full evaluation on test iterations
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                all_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'loss': []}
                
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    
                    # 计算完整指标
                    metrics = compute_metrics(image, gt_image)
                    metrics['loss'] = l1_loss(image, gt_image).mean().double().item()
                    
                    for key, value in metrics.items():
                        all_metrics[key].append(value)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                # 计算平均指标
                avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
                
                if tb_writer:
                    for metric_name, metric_value in avg_metrics.items():
                        tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - {metric_name}", metric_value, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()

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
        early_iterations = [1_000, 3_000, 5_000, 7_000]
        auto_iterations = list(range(10_000, args.iterations + 1, 5_000))
        args.test_iterations = early_iterations + auto_iterations
        
    if args.save_iterations is None:
        args.save_iterations = args.test_iterations.copy()
    
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