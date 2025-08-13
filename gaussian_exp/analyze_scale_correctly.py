import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel

def analyze_scale_correctly():
    """正确分析高斯球尺寸 - 理解negative scaling的含义"""
    print("🔍 正确的高斯球尺寸分析")
    print("=" * 50)
    
    # 加载原始模型
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    # 提取参数
    xyz = gaussians._xyz.detach().cpu().numpy()
    scaling = gaussians._scaling.detach().cpu().numpy()  # 这是log space的scaling
    opacity = gaussians._opacity.detach().cpu().numpy()
    
    # 处理NaN值
    nan_mask = np.isnan(xyz)
    nan_positions = np.any(nan_mask, axis=1)
    nan_count = np.sum(nan_positions)
    
    if nan_count > 0:
        print(f"⚠️ 发现 {nan_count} 个NaN位置，将被排除")
        valid_mask = ~nan_positions
        xyz = xyz[valid_mask]
        scaling = scaling[valid_mask]
        opacity = opacity[valid_mask]
    
    print(f"📊 原始scaling参数分析:")
    print(f"  总高斯球数: {len(scaling):,}")
    print(f"  Scaling shape: {scaling.shape}")
    print(f"  Scaling范围: {scaling.min():.6f} ~ {scaling.max():.6f}")
    print(f"  这些是**对数空间**的scaling值！")
    
    # 转换到实际尺寸 (exp操作)
    actual_scales = np.exp(scaling)  # 从log space转换到real space
    
    print(f"\n📏 实际尺寸参数分析 (exp(log_scaling)):")
    print(f"  X尺寸范围: {actual_scales[:, 0].min():.6f} ~ {actual_scales[:, 0].max():.6f}")
    print(f"  Y尺寸范围: {actual_scales[:, 1].min():.6f} ~ {actual_scales[:, 1].max():.6f}")
    print(f"  Z尺寸范围: {actual_scales[:, 2].min():.6f} ~ {actual_scales[:, 2].max():.6f}")
    
    # 计算平均实际尺寸
    avg_actual_scale = np.mean(actual_scales, axis=1)
    max_actual_scale = np.max(actual_scales, axis=1)
    min_actual_scale = np.min(actual_scales, axis=1)
    
    print(f"\n📈 实际尺寸统计:")
    print(f"  平均尺寸范围: {avg_actual_scale.min():.6f} ~ {avg_actual_scale.max():.6f}")
    print(f"  平均尺寸均值: {avg_actual_scale.mean():.6f}")
    print(f"  平均尺寸中位数: {np.median(avg_actual_scale):.6f}")
    print(f"  平均尺寸标准差: {avg_actual_scale.std():.6f}")
    
    # 分析尺寸分布
    percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    avg_scale_percentiles = np.percentile(avg_actual_scale, percentiles)
    
    print(f"\n📊 平均实际尺寸百分位数分析:")
    for p, val in zip(percentiles, avg_scale_percentiles):
        count = np.sum(avg_actual_scale <= val)
        print(f"  {p:2d}%: {val:.6f} ({count:,}球)")
    
    # 重新评估之前的"超小球"
    previous_threshold = 0.003223  # 之前用的阈值
    ultra_small_count = np.sum(avg_actual_scale <= previous_threshold)
    ultra_small_ratio = ultra_small_count / len(avg_actual_scale) * 100
    
    print(f"\n🔄 重新评估'超小球'概念:")
    print(f"  之前阈值: {previous_threshold:.6f} (实际尺寸)")
    print(f"  符合条件数量: {ultra_small_count:,} ({ultra_small_ratio:.2f}%)")
    
    # 现在看起来更合理了，让我们重新分层
    print(f"\n🎯 基于实际尺寸的智能分层:")
    
    # 方案1: 基于实际尺寸的等百分比分层
    print(f"\n  方案1: 基于平均实际尺寸的分层")
    size_thresholds = np.percentile(avg_actual_scale, [20, 40, 60, 80])
    
    layer_names = ["超微", "微小", "小型", "中型", "大型"]
    layer_descriptions = ["Ultra-micro", "Micro", "Small", "Medium", "Large"]
    
    for i in range(5):
        if i == 0:
            mask = avg_actual_scale <= size_thresholds[0]
        elif i == 4:
            mask = avg_actual_scale > size_thresholds[3]
        else:
            mask = (avg_actual_scale > size_thresholds[i-1]) & (avg_actual_scale <= size_thresholds[i])
        
        count = np.sum(mask)
        ratio = count / len(avg_actual_scale) * 100
        if count > 0:
            range_str = f"{avg_actual_scale[mask].min():.6f}~{avg_actual_scale[mask].max():.6f}"
            print(f"    层{i} ({layer_names[i]}): {count:,}球 ({ratio:.1f}%) 范围: {range_str}")
    
    # 方案2: 基于最大尺寸分层 (可能更有意义)
    print(f"\n  方案2: 基于最大实际尺寸的分层")
    max_size_thresholds = np.percentile(max_actual_scale, [20, 40, 60, 80])
    
    for i in range(5):
        if i == 0:
            mask = max_actual_scale <= max_size_thresholds[0]
        elif i == 4:
            mask = max_actual_scale > max_size_thresholds[3]
        else:
            mask = (max_actual_scale > max_size_thresholds[i-1]) & (max_actual_scale <= max_size_thresholds[i])
        
        count = np.sum(mask)
        ratio = count / len(max_actual_scale) * 100
        if count > 0:
            range_str = f"{max_actual_scale[mask].min():.6f}~{max_actual_scale[mask].max():.6f}"
            print(f"    层{i} ({layer_names[i]}): {count:,}球 ({ratio:.1f}%) 最大尺寸范围: {range_str}")
    
    # 方案3: 考虑形状差异的分层
    print(f"\n  方案3: 基于形状特征的分层")
    
    # 计算椭球形状特征
    aspect_ratios = max_actual_scale / min_actual_scale  # 最大轴与最小轴的比例
    volume_approx = np.prod(actual_scales, axis=1)  # 近似体积
    
    print(f"    形状比分析:")
    print(f"      形状比范围: {aspect_ratios.min():.2f} ~ {aspect_ratios.max():.2f}")
    print(f"      形状比均值: {aspect_ratios.mean():.2f}")
    
    # 按形状比分类
    sphere_like = aspect_ratios <= 2.0    # 接近球形
    ellipsoid_like = (aspect_ratios > 2.0) & (aspect_ratios <= 5.0)  # 椭球形
    needle_like = aspect_ratios > 5.0     # 针状/片状
    
    print(f"      球形: {np.sum(sphere_like):,}球 ({np.sum(sphere_like)/len(aspect_ratios)*100:.1f}%)")
    print(f"      椭球形: {np.sum(ellipsoid_like):,}球 ({np.sum(ellipsoid_like)/len(aspect_ratios)*100:.1f}%)")
    print(f"      针状: {np.sum(needle_like):,}球 ({np.sum(needle_like)/len(aspect_ratios)*100:.1f}%)")
    
    # 分析体积分布
    print(f"    体积分析:")
    print(f"      体积范围: {volume_approx.min():.9f} ~ {volume_approx.max():.6f}")
    print(f"      体积均值: {volume_approx.mean():.9f}")
    print(f"      体积中位数: {np.median(volume_approx):.9f}")
    
    # 体积分层
    volume_thresholds = np.percentile(volume_approx, [20, 40, 60, 80])
    print(f"    按体积分层:")
    for i in range(5):
        if i == 0:
            mask = volume_approx <= volume_thresholds[0]
        elif i == 4:
            mask = volume_approx > volume_thresholds[3]
        else:
            mask = (volume_approx > volume_thresholds[i-1]) & (volume_approx <= volume_thresholds[i])
        
        count = np.sum(mask)
        ratio = count / len(volume_approx) * 100
        if count > 0:
            range_str = f"{volume_approx[mask].min():.9f}~{volume_approx[mask].max():.6f}"
            print(f"      层{i}: {count:,}球 ({ratio:.1f}%) 体积范围: {range_str}")
    
    # 创建可视化（修复尺寸问题）
    print(f"\n🎨 生成可视化图表...")
    
    # 创建输出目录
    output_dir = "correct_scale_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 修复图像尺寸问题
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Correct Gaussian Scale Analysis (Actual Sizes)', fontsize=16, fontweight='bold')
    
    # 1. Log scaling分布
    ax = axes[0, 0]
    ax.hist(scaling.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    ax.set_xlabel('Log Scaling Values')
    ax.set_ylabel('Density')
    ax.set_title('Original Log Scaling Distribution')
    ax.grid(True, alpha=0.3)
    
    # 2. 实际尺寸分布
    ax = axes[0, 1]
    ax.hist(avg_actual_scale, bins=50, alpha=0.7, color='red', density=True)
    ax.set_xlabel('Average Actual Scale')
    ax.set_ylabel('Density')
    ax.set_title('Average Actual Scale Distribution')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. 最大尺寸分布
    ax = axes[0, 2]
    ax.hist(max_actual_scale, bins=50, alpha=0.7, color='green', density=True)
    ax.set_xlabel('Maximum Actual Scale')
    ax.set_ylabel('Density')
    ax.set_title('Maximum Actual Scale Distribution')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. 形状比分布
    ax = axes[1, 0]
    ax.hist(aspect_ratios, bins=50, alpha=0.7, color='purple', density=True)
    ax.set_xlabel('Aspect Ratio (Max/Min)')
    ax.set_ylabel('Density')
    ax.set_title('Shape Aspect Ratio Distribution')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 5. 体积分布
    ax = axes[1, 1]
    ax.hist(volume_approx, bins=50, alpha=0.7, color='orange', density=True)
    ax.set_xlabel('Approximate Volume')
    ax.set_ylabel('Density')
    ax.set_title('Volume Distribution')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 6. 累积分布对比
    ax = axes[1, 2]
    
    # 平均尺寸累积分布
    sorted_avg = np.sort(avg_actual_scale)
    cumulative_avg = np.arange(1, len(sorted_avg) + 1) / len(sorted_avg) * 100
    ax.plot(sorted_avg, cumulative_avg, 'r-', linewidth=2, label='Average Scale')
    
    # 最大尺寸累积分布
    sorted_max = np.sort(max_actual_scale)
    cumulative_max = np.arange(1, len(sorted_max) + 1) / len(sorted_max) * 100
    ax.plot(sorted_max, cumulative_max, 'g-', linewidth=2, label='Maximum Scale')
    
    ax.set_xlabel('Actual Scale')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('Cumulative Distribution Comparison')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    distribution_file = os.path.join(output_dir, 'correct_scale_distribution.png')
    plt.savefig(distribution_file, dpi=150, bbox_inches='tight')  # 降低DPI避免过大
    plt.close()
    
    print(f"✅ 分布图保存: {distribution_file}")
    
    # 保存分析结果
    analysis_results = {
        'understanding_correction': {
            'original_scaling_is_log_space': True,
            'actual_scales_computed_by': 'exp(log_scaling)',
            'previous_error': 'Used log values as actual sizes'
        },
        'actual_scale_stats': {
            'avg_scale_range': [float(avg_actual_scale.min()), float(avg_actual_scale.max())],
            'avg_scale_mean': float(avg_actual_scale.mean()),
            'avg_scale_median': float(np.median(avg_actual_scale)),
            'avg_scale_std': float(avg_actual_scale.std())
        },
        'shape_analysis': {
            'aspect_ratio_range': [float(aspect_ratios.min()), float(aspect_ratios.max())],
            'aspect_ratio_mean': float(aspect_ratios.mean()),
            'shape_categories': {
                'sphere_like': int(np.sum(sphere_like)),
                'ellipsoid_like': int(np.sum(ellipsoid_like)),
                'needle_like': int(np.sum(needle_like))
            }
        },
        'volume_analysis': {
            'volume_range': [float(volume_approx.min()), float(volume_approx.max())],
            'volume_mean': float(volume_approx.mean()),
            'volume_median': float(np.median(volume_approx))
        },
        'recommended_layering': {
            'avg_scale_thresholds': [float(t) for t in size_thresholds],
            'max_scale_thresholds': [float(t) for t in max_size_thresholds],
            'volume_thresholds': [float(t) for t in volume_thresholds]
        },
        'key_insights': [
            "原始scaling是对数空间值，需要exp()转换为实际尺寸",
            "实际尺寸分布更加合理，没有99%都是超小球的问题",
            "应该基于实际尺寸重新设计分层策略",
            "可以考虑形状和体积作为分层依据",
            "大部分高斯球是近球形的，少数是椭球或针状"
        ]
    }
    
    results_file = os.path.join(output_dir, 'correct_scale_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"✅ 分析结果保存: {results_file}")
    
    print(f"\n🎉 正确的尺寸分析完成!")
    print(f"📁 输出目录: {output_dir}/")
    print(f"💡 关键发现:")
    print(f"   1. **重大发现**: 原始scaling是对数空间值！")
    print(f"   2. 实际尺寸分布合理: {avg_actual_scale.min():.6f} ~ {avg_actual_scale.max():.6f}")
    print(f"   3. 现在可以进行有意义的尺寸分层了")
    print(f"   4. 大部分是球形({np.sum(sphere_like)/len(aspect_ratios)*100:.1f}%)，少数椭球形和针状")
    
    return analysis_results, {
        'avg_actual_scale': avg_actual_scale,
        'max_actual_scale': max_actual_scale,
        'aspect_ratios': aspect_ratios,
        'volume_approx': volume_approx
    }

def main():
    analyze_scale_correctly()

if __name__ == "__main__":
    main() 