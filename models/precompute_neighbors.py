#!/usr/bin/env python3
"""
离线预计算 COLMAP 点到 3DGS 高斯中心的邻接关系
===================================================

功能：
- 支持多半径球查询（ball_query）
- 生成 CSR 格式的变长邻接表
- 分块处理大规模数据（2M+ 高斯球）
- 统计邻域内尺度信息
- GPU 加速计算

依赖：
    pip install torch torchvision
    pip install pytorch3d  # 推荐
    # 或者 pip install pointnet2_ops  # 备选

用法：
    python precompute_neighbors.py \
      --colmap data/colmap_xyz.pt \
      --gauss  data/gaussians.pt \
      --radii  0.012 0.039 0.107 0.273 \
      --kmax   128 512 1024 2048 \
      --chunk  200000 \
      --out    cache/neighbors/
"""

import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================= 依赖检查 =============================

def check_and_import_ball_query():
    """检查并导入球查询函数，优先级：pytorch3d > pointnet2_ops > torch_fallback"""
    ball_query_func = None
    backend_name = ""
    
    # 尝试 pytorch3d
    try:
        from pytorch3d.ops import ball_query
        # 测试导入是否真的可用
        torch.manual_seed(42)
        test_p1 = torch.randn(1, 10, 3, device='cuda' if torch.cuda.is_available() else 'cpu')
        test_p2 = torch.randn(1, 20, 3, device='cuda' if torch.cuda.is_available() else 'cpu')
        _ = ball_query(test_p1, test_p2, radius=0.1, K=5)
        ball_query_func = ball_query
        backend_name = "pytorch3d"
        logger.info("✅ 使用 pytorch3d.ops.ball_query")
    except (ImportError, RuntimeError, Exception) as e:
        logger.warning(f"pytorch3d 不可用 ({e})，尝试 pointnet2_ops...")
        
        # 尝试 pointnet2_ops
        try:
            from pointnet2_ops import pointnet2_utils
            def ball_query_wrapper(p1, p2, radius, K):
                # pointnet2_ops 接口适配
                idx = pointnet2_utils.ball_query(radius, K, p2, p1)
                return idx
            ball_query_func = ball_query_wrapper
            backend_name = "pointnet2_ops"
            logger.info("✅ 使用 pointnet2_ops.ball_query")
        except (ImportError, RuntimeError, Exception) as e:
            logger.warning(f"pointnet2_ops 不可用 ({e})，使用纯PyTorch实现...")
            
            # 纯PyTorch实现的球查询
            def torch_ball_query(p1, p2, radius, K):
                """
                内存优化的纯PyTorch球查询实现
                Args:
                    p1: [B, N, 3] 查询点
                    p2: [B, M, 3] 参考点
                    radius: 查询半径
                    K: 最大邻居数
                Returns:
                    [B, N, K] 邻居索引，-1表示无效
                """
                B, N, _ = p1.shape
                _, M, _ = p2.shape
                device = p1.device
                
                # 初始化结果
                neighbor_indices = torch.full((B, N, K), -1, dtype=torch.long, device=device)
                
                # 分批处理查询点以节省内存
                query_batch_size = min(1000, N)  # 每次处理1000个查询点
                
                for b in range(B):
                    for start_n in range(0, N, query_batch_size):
                        end_n = min(start_n + query_batch_size, N)
                        batch_p1 = p1[b, start_n:end_n]  # [batch_size, 3]
                        batch_p2 = p2[b]  # [M, 3]
                        
                        # 计算距离矩阵 [batch_size, M]
                        batch_p1_expanded = batch_p1.unsqueeze(1)  # [batch_size, 1, 3]
                        batch_p2_expanded = batch_p2.unsqueeze(0)  # [1, M, 3]
                        distances = torch.norm(batch_p1_expanded - batch_p2_expanded, dim=2)  # [batch_size, M]
                        
                        # 找到半径内的点
                        valid_mask = distances <= radius  # [batch_size, M]
                        
                        # 为每个查询点找到最近的K个邻居
                        for local_n in range(end_n - start_n):
                            global_n = start_n + local_n
                            valid_neighbors = torch.where(valid_mask[local_n])[0]
                            
                            if len(valid_neighbors) > 0:
                                neighbor_distances = distances[local_n, valid_neighbors]
                                # 排序并选择最近的K个
                                sorted_indices = torch.argsort(neighbor_distances)
                                num_neighbors = min(K, len(valid_neighbors))
                                selected_neighbors = valid_neighbors[sorted_indices[:num_neighbors]]
                                neighbor_indices[b, global_n, :num_neighbors] = selected_neighbors
                
                return neighbor_indices
            
            ball_query_func = torch_ball_query
            backend_name = "torch_fallback"
            logger.info("✅ 使用纯PyTorch球查询实现")
    
    return ball_query_func, backend_name

# ============================= 数据加载 =============================

def load_arrays(path: Union[str, Path]) -> torch.Tensor:
    """加载 .pt 或 .npz 格式的数组数据"""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    if path.suffix == '.pt':
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            # 如果是字典，尝试找到坐标数据
            if 'xyz' in data:
                return data['xyz']
            elif len(data) == 1:
                return list(data.values())[0]
            else:
                raise ValueError(f"字典格式的 .pt 文件需要包含 'xyz' 键: {list(data.keys())}")
        else:
            return data
    elif path.suffix == '.npz':
        data = np.load(path)
        if 'xyz' in data:
            return torch.from_numpy(data['xyz'])
        elif len(data.files) == 1:
            return torch.from_numpy(data[data.files[0]])
        else:
            raise ValueError(f"npz 文件需要包含 'xyz' 键: {list(data.keys())}")
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

def load_gaussian_data(path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载高斯数据，返回 (xyz, scale)"""
    path = Path(path)
    
    if path.suffix == '.pt':
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            xyz = data['xyz']
            scale = data['scale']
        else:
            raise ValueError("高斯数据需要字典格式，包含 'xyz' 和 'scale' 键")
    elif path.suffix == '.npz':
        data = np.load(path)
        xyz = torch.from_numpy(data['xyz'])
        scale = torch.from_numpy(data['scale'])
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    return xyz, scale

# ============================= CSR 处理 =============================

def to_csr(neighbor_indices: torch.Tensor, num_query_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将邻居索引转换为 CSR 格式
    
    Args:
        neighbor_indices: [N, K] 邻居索引，-1 表示无效
        num_query_points: 查询点数量 N
    
    Returns:
        indices: [nnz] 扁平化的有效邻居索引
        row_splits: [N+1] 每行的起始位置
    """
    # 找到有效邻居（非 -1）
    valid_mask = neighbor_indices != -1
    
    # 计算每个查询点的邻居数量
    neighbor_counts = valid_mask.sum(dim=1).int()
    
    # 生成 row_splits
    row_splits = torch.zeros(num_query_points + 1, dtype=torch.int32)
    row_splits[1:] = torch.cumsum(neighbor_counts, dim=0)
    
    # 提取有效的邻居索引
    indices = neighbor_indices[valid_mask].int()
    
    return indices, row_splits

def compute_neighbor_stats(indices: torch.Tensor, row_splits: torch.Tensor, 
                          gauss_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    """计算邻域统计信息"""
    num_points = len(row_splits) - 1
    
    # 计算 log scale magnitudes
    scale_magnitudes = torch.norm(gauss_scale, dim=1)  # [M]
    # 裁剪极端值（99.5分位数）
    scale_p995 = torch.quantile(scale_magnitudes, 0.995)
    scale_magnitudes = torch.clamp(scale_magnitudes, max=scale_p995)
    log_scale_magnitudes = torch.log(scale_magnitudes + 1e-8)  # [M]
    
    # 为每个查询点计算邻域统计
    mean_log_scale = torch.zeros(num_points, dtype=torch.float32)
    scale_sum = torch.zeros(num_points, dtype=torch.float32)
    
    for i in range(num_points):
        start_idx = row_splits[i].item()
        end_idx = row_splits[i + 1].item()
        
        if start_idx < end_idx:
            neighbor_idx = indices[start_idx:end_idx]
            neighbor_log_scales = log_scale_magnitudes[neighbor_idx]
            neighbor_scales = scale_magnitudes[neighbor_idx]
            
            mean_log_scale[i] = neighbor_log_scales.mean()
            scale_sum[i] = neighbor_scales.sum()
        else:
            # 无邻居的情况
            mean_log_scale[i] = 0.0
            scale_sum[i] = 0.0
    
    return {
        'mean_log_scale': mean_log_scale,
        'scale_sum': scale_sum
    }

# ============================= 球查询 =============================

def ball_query_multi_radius(colmap_xyz: torch.Tensor, gauss_xyz: torch.Tensor,
                           radii: List[float], kmax_list: List[int], 
                           chunk_size: int, ball_query_func, backend_name: str) -> Dict[str, Dict]:
    """
    多半径球查询，分块处理大数据
    
    Returns:
        Dict[radius_str, CSRDict] where CSRDict contains:
            - indices: torch.Tensor
            - row_splits: torch.Tensor  
            - count: torch.Tensor
            - mean_log_scale: torch.Tensor
            - scale_sum: torch.Tensor
    """
    device = colmap_xyz.device
    N = len(colmap_xyz)
    M = len(gauss_xyz)
    
    results = {}
    
    # 计算高斯球的尺度信息（用于后续统计）
    logger.info("预计算高斯球尺度信息...")
    if gauss_xyz.shape[1] > 3:
        # 假设后3列是尺度
        gauss_scale = gauss_xyz[:, 3:6] if gauss_xyz.shape[1] >= 6 else gauss_xyz[:, 3:4].repeat(1, 3)
    else:
        # 如果没有尺度信息，使用默认值
        gauss_scale = torch.ones(M, 3, device=device) * 0.01
        logger.warning("未找到尺度信息，使用默认值 0.01")
    
    for radius, kmax in zip(radii, kmax_list):
        logger.info(f"处理半径 {radius:.6f}, 最大邻居数 {kmax}...")
        
        # 存储所有邻居索引
        all_neighbor_indices = []
        
        # 分块处理
        num_chunks = (M + chunk_size - 1) // chunk_size
        
        for chunk_i in range(num_chunks):
            start_idx = chunk_i * chunk_size
            end_idx = min((chunk_i + 1) * chunk_size, M)
            
            gauss_chunk = gauss_xyz[start_idx:end_idx, :3]  # 只取坐标
            
            logger.info(f"  块 {chunk_i+1}/{num_chunks}: 高斯球 {start_idx}-{end_idx}")
            
            # 球查询
            if backend_name == "pytorch3d":
                # pytorch3d: ball_query(p1, p2, radius, K)
                neighbor_idx = ball_query_func(
                    colmap_xyz.unsqueeze(0),  # [1, N, 3]
                    gauss_chunk.unsqueeze(0),  # [1, M_chunk, 3]
                    radius=radius,
                    K=kmax
                )[0].squeeze(0)  # [N, K]
                
            elif backend_name == "pointnet2_ops":
                # pointnet2_ops 需要 [1, N, 3] 和 [1, M_chunk, 3]
                neighbor_idx = ball_query_func(
                    colmap_xyz.unsqueeze(0),  # [1, N, 3]
                    gauss_chunk.unsqueeze(0),  # [1, M_chunk, 3]
                    radius, kmax
                ).squeeze(0)  # [N, K]
                
            elif backend_name == "torch_fallback":
                # 纯PyTorch实现
                neighbor_idx = ball_query_func(
                    colmap_xyz.unsqueeze(0),  # [1, N, 3]
                    gauss_chunk.unsqueeze(0),  # [1, M_chunk, 3]
                    radius, kmax
                ).squeeze(0)  # [N, K]
                
            # 调整索引到全局（对所有后端都适用）
            valid_mask = neighbor_idx != -1
            neighbor_idx[valid_mask] += start_idx
            
            all_neighbor_indices.append(neighbor_idx)
        
        # 合并所有块的结果
        logger.info("合并分块结果...")
        combined_neighbors = torch.cat(all_neighbor_indices, dim=1)  # [N, K*num_chunks]
        
        # 每个点只保留最近的 kmax 个邻居
        if combined_neighbors.shape[1] > kmax:
            # 计算距离并排序
            distances = []
            for i in range(N):
                query_point = colmap_xyz[i:i+1]  # [1, 3]
                neighbor_mask = combined_neighbors[i] != -1
                if neighbor_mask.any():
                    neighbor_points = gauss_xyz[combined_neighbors[i][neighbor_mask], :3]  # [K_valid, 3]
                    dist = torch.norm(neighbor_points - query_point, dim=1)
                    distances.append(dist)
                else:
                    distances.append(torch.tensor([], device=device))
            
            # 重新选择最近的邻居
            final_neighbors = torch.full((N, kmax), -1, dtype=torch.long, device=device)
            for i in range(N):
                neighbor_mask = combined_neighbors[i] != -1
                if neighbor_mask.any():
                    valid_neighbors = combined_neighbors[i][neighbor_mask]
                    valid_distances = distances[i]
                    
                    if len(valid_distances) > kmax:
                        _, sorted_idx = torch.sort(valid_distances)
                        selected_neighbors = valid_neighbors[sorted_idx[:kmax]]
                    else:
                        selected_neighbors = valid_neighbors
                    
                    final_neighbors[i, :len(selected_neighbors)] = selected_neighbors
            
            neighbor_indices = final_neighbors
        else:
            neighbor_indices = combined_neighbors
        
        # 转换为 CSR 格式
        logger.info("转换为 CSR 格式...")
        indices, row_splits = to_csr(neighbor_indices, N)
        
        # 计算邻居数量
        count = row_splits[1:] - row_splits[:-1]
        
        # 计算统计信息
        logger.info("计算邻域统计...")
        stats = compute_neighbor_stats(indices, row_splits, gauss_scale)
        
        # 统计日志
        mean_neighbors = count.float().mean().item()
        median_neighbors = count.float().median().item()
        max_neighbors = count.max().item()
        
        logger.info(f"半径 {radius:.6f} 统计:")
        logger.info(f"  平均邻居数: {mean_neighbors:.2f}")
        logger.info(f"  中位邻居数: {median_neighbors:.2f}")
        logger.info(f"  最大邻居数: {max_neighbors}")
        logger.info(f"  总边数: {len(indices)}")
        
        # 保存结果
        results[f"{radius:.6f}"] = {
            'indices': indices.cpu(),
            'row_splits': row_splits.cpu(),
            'count': count.cpu(),
            'mean_log_scale': stats['mean_log_scale'].cpu(),
            'scale_sum': stats['scale_sum'].cpu(),
            'radius': radius,
            'kmax': kmax,
            'stats': {
                'mean_neighbors': mean_neighbors,
                'median_neighbors': median_neighbors,
                'max_neighbors': max_neighbors,
                'total_edges': len(indices)
            }
        }
    
    return results

# ============================= 主函数 =============================

def main():
    parser = argparse.ArgumentParser(description='预计算 COLMAP-3DGS 邻接关系')
    parser.add_argument('--colmap', type=str, required=True, 
                       help='COLMAP 点坐标文件 (.pt/.npz)')
    parser.add_argument('--gauss', type=str, required=True,
                       help='高斯数据文件 (.pt/.npz，包含 xyz 和 scale)')
    parser.add_argument('--radii', type=float, nargs='+', 
                       default=[0.012, 0.039, 0.107, 0.273],
                       help='查询半径列表')
    parser.add_argument('--kmax', type=int, nargs='+',
                       default=[128, 512, 1024, 2048],
                       help='各半径的最大邻居数')
    parser.add_argument('--chunk', type=int, default=200000,
                       help='高斯球分块大小')
    parser.add_argument('--out', type=str, default='cache/neighbors',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 检查参数
    if len(args.radii) != len(args.kmax):
        raise ValueError("radii 和 kmax 的长度必须相同")
    
    # 创建输出目录
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 检查球查询后端
    ball_query_func, backend_name = check_and_import_ball_query()
    
    # 加载数据
    logger.info("加载数据...")
    colmap_xyz = load_arrays(args.colmap).float()
    
    try:
        gauss_xyz, gauss_scale = load_gaussian_data(args.gauss)
        # 合并坐标和尺度
        gauss_data = torch.cat([gauss_xyz.float(), gauss_scale.float()], dim=1)
    except ValueError:
        # 如果高斯文件只包含坐标
        gauss_data = load_arrays(args.gauss).float()
        logger.warning("高斯文件中未找到尺度信息")
    
    logger.info(f"COLMAP 点数: {len(colmap_xyz):,}")
    logger.info(f"高斯球数: {len(gauss_data):,}")
    
    # 移动到GPU
    colmap_xyz = colmap_xyz.to(device)
    gauss_data = gauss_data.to(device)
    
    # 执行多半径球查询
    logger.info("开始预计算邻接关系...")
    start_time = time.time()
    
    results = ball_query_multi_radius(
        colmap_xyz, gauss_data, args.radii, args.kmax, 
        args.chunk, ball_query_func, backend_name
    )
    
    total_time = time.time() - start_time
    logger.info(f"总耗时: {total_time:.2f} 秒")
    
    # 保存结果
    for radius_str, data in results.items():
        output_path = out_dir / f"neigh_r{radius_str}.pt"
        torch.save(data, output_path)
        logger.info(f"保存: {output_path}")
    
    logger.info("✅ 预计算完成！")

# ============================= 自检测试 =============================

def quick_test():
    """快速自检测试"""
    logger.info("🧪 运行快速自检...")
    
    # 检查球查询后端
    try:
        ball_query_func, backend_name = check_and_import_ball_query()
    except ImportError:
        logger.error("自检失败：缺少球查询后端")
        return False
    
    # 生成测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N_colmap = 1000  # 1k COLMAP 点
    M_gauss = 10000  # 10k 高斯球
    
    torch.manual_seed(42)
    colmap_xyz = torch.randn(N_colmap, 3, device=device) * 2.0
    gauss_xyz = torch.randn(M_gauss, 3, device=device) * 2.0
    gauss_scale = torch.rand(M_gauss, 3, device=device) * 0.1 + 0.01
    gauss_data = torch.cat([gauss_xyz, gauss_scale], dim=1)
    
    # 测试参数
    radii = [0.1, 0.2]
    kmax_list = [32, 64]
    chunk_size = 5000
    
    try:
        results = ball_query_multi_radius(
            colmap_xyz, gauss_data, radii, kmax_list, 
            chunk_size, ball_query_func, backend_name
        )
        
        # 检查结果
        for radius_str, data in results.items():
            indices = data['indices']
            row_splits = data['row_splits']
            count = data['count']
            
            logger.info(f"半径 {radius_str}:")
            logger.info(f"  CSR indices 形状: {indices.shape}")
            logger.info(f"  CSR row_splits 形状: {row_splits.shape}")
            logger.info(f"  count 形状: {count.shape}")
            logger.info(f"  平均邻居数: {count.float().mean():.2f}")
        
        logger.info("✅ 自检通过！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 自检失败: {e}")
        return False

if __name__ == "__main__":
    # 先运行自检
    if not quick_test():
        logger.error("自检失败，请检查环境配置")
        exit(1)
    
    # 运行主程序
    main() 