import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel

def analyze_fine_scale_distribution():
    """精细分析高斯球尺寸分布，特别关注超小球的内部分层"""
    print("🔍 精细高斯球尺寸分布分析")
    print("=" * 50)
    
    # 加载原始模型
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    # 提取参数
    xyz = gaussians._xyz.detach().cpu().numpy()
    scaling = gaussians._scaling.detach().cpu().numpy()
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
    
    # 计算平均缩放
    avg_scale = np.mean(scaling, axis=1)
    
    print(f"📊 基本统计:")
    print(f"  总高斯球数: {len(avg_scale):,}")
    print(f"  平均缩放范围: {avg_scale.min():.6f} ~ {avg_scale.max():.6f}")
    print(f"  平均值: {avg_scale.mean():.6f}")
    print(f"  中位数: {np.median(avg_scale):.6f}")
    print(f"  标准差: {avg_scale.std():.6f}")
    
    # 详细百分位数分析
    percentiles = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 99.5, 99.9]
    scale_percentiles = np.percentile(avg_scale, percentiles)
    
    print(f"\n📈 详细百分位数分析:")
    for p, val in zip(percentiles, scale_percentiles):
        count = np.sum(avg_scale <= val)
        print(f"  {p:4.1f}%: {val:.6f} ({count:,}球)")
    
    # 重新定义之前的"超小球"边界
    previous_threshold = 0.0032229693606495857
    ultra_small_count = np.sum(avg_scale <= previous_threshold)
    ultra_small_ratio = ultra_small_count / len(avg_scale) * 100
    
    print(f"\n🔍 之前的'超小球'分析:")
    print(f"  阈值: {previous_threshold:.6f}")
    print(f"  数量: {ultra_small_count:,} ({ultra_small_ratio:.2f}%)")
    print(f"  这确实占了绝大多数！需要进一步细分")
    
    # 对"超小球"进行更细致的分层
    ultra_small_mask = avg_scale <= previous_threshold
    ultra_small_scales = avg_scale[ultra_small_mask]
    
    print(f"\n🎯 超小球内部分析:")
    print(f"  数量: {len(ultra_small_scales):,}")
    print(f"  范围: {ultra_small_scales.min():.6f} ~ {ultra_small_scales.max():.6f}")
    print(f"  均值: {ultra_small_scales.mean():.6f}")
    print(f"  中位数: {np.median(ultra_small_scales):.6f}")
    print(f"  标准差: {ultra_small_scales.std():.6f}")
    
    # 超小球内部细分（10个层级）
    ultra_small_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    ultra_small_thresholds = np.percentile(ultra_small_scales, ultra_small_percentiles)
    
    print(f"\n🔄 超小球内部10层细分:")
    for i in range(10):
        if i == 0:
            mask = ultra_small_scales <= ultra_small_thresholds[0]
            layer_desc = "微观球"
        elif i == 9:
            mask = ultra_small_scales > ultra_small_thresholds[8]
            layer_desc = "准小球"
        else:
            mask = (ultra_small_scales > ultra_small_thresholds[i-1]) & (ultra_small_scales <= ultra_small_thresholds[i])
            layer_desc = f"超小球{i}"
        
        count = np.sum(mask)
        if count > 0:
            layer_range = f"{ultra_small_scales[mask].min():.6f}~{ultra_small_scales[mask].max():.6f}"
            ratio = count / len(ultra_small_scales) * 100
            print(f"  层{i} ({layer_desc}): {count:,}球 ({ratio:.1f}%) 范围: {layer_range}")
    
    # 全局智能分层策略
    print(f"\n🧠 智能全局分层策略:")
    
    # 方案1: 对数分层 (处理幂律分布)
    log_scales = np.log10(avg_scale + 1e-10)  # 避免log(0)
    log_percentiles = [10, 25, 50, 75, 90, 95, 99]
    log_thresholds = np.percentile(log_scales, log_percentiles)
    actual_thresholds = 10**log_thresholds - 1e-10
    
    print(f"\n  方案1: 对数分层 (适合幂律分布)")
    for i in range(len(log_percentiles) + 1):
        if i == 0:
            mask = avg_scale <= actual_thresholds[0]
            layer_name = "nano"
        elif i == len(log_percentiles):
            mask = avg_scale > actual_thresholds[-1]
            layer_name = "giant"
        else:
            mask = (avg_scale > actual_thresholds[i-1]) & (avg_scale <= actual_thresholds[i])
            layer_name = f"scale{i}"
        
        count = np.sum(mask)
        if count > 0:
            ratio = count / len(avg_scale) * 100
            range_str = f"{avg_scale[mask].min():.6f}~{avg_scale[mask].max():.6f}"
            print(f"    层{i} ({layer_name}): {count:,}球 ({ratio:.1f}%) 范围: {range_str}")
    
    # 方案2: 等数量分层
    equal_count_layers = 20  # 20层
    equal_count_thresholds = np.percentile(avg_scale, np.linspace(5, 95, equal_count_layers-1))
    
    print(f"\n  方案2: 等数量分层 ({equal_count_layers}层)")
    for i in range(equal_count_layers):
        if i == 0:
            mask = avg_scale <= equal_count_thresholds[0]
        elif i == equal_count_layers - 1:
            mask = avg_scale > equal_count_thresholds[-1]
        else:
            mask = (avg_scale > equal_count_thresholds[i-1]) & (avg_scale <= equal_count_thresholds[i])
        
        count = np.sum(mask)
        if count > 0:
            ratio = count / len(avg_scale) * 100
            range_str = f"{avg_scale[mask].min():.6f}~{avg_scale[mask].max():.6f}"
            print(f"    层{i:2d}: {count:,}球 ({ratio:.1f}%) 范围: {range_str}")
    
    # 方案3: 混合策略 - 精细+粗糙
    print(f"\n  方案3: 混合策略 (前95%精细分层 + 后5%粗糙分层)")
    
    # 前95%细分为15层
    p95_threshold = np.percentile(avg_scale, 95)
    small_scales = avg_scale[avg_scale <= p95_threshold]
    fine_thresholds = np.percentile(small_scales, np.linspace(6.67, 93.33, 14))  # 15层的14个分界点
    
    # 后5%分为5层
    large_scales = avg_scale[avg_scale > p95_threshold]
    coarse_thresholds = np.percentile(large_scales, [20, 40, 60, 80])
    
    mixed_thresholds = np.concatenate([fine_thresholds, [p95_threshold], coarse_thresholds])
    
    for i in range(20):  # 总共20层
        if i == 0:
            mask = avg_scale <= mixed_thresholds[0]
            layer_type = "精细"
        elif i < 15:
            mask = (avg_scale > mixed_thresholds[i-1]) & (avg_scale <= mixed_thresholds[i])
            layer_type = "精细"
        elif i == 15:
            mask = (avg_scale > mixed_thresholds[i-1]) & (avg_scale <= mixed_thresholds[i])
            layer_type = "过渡"
        else:
            if i == 19:
                mask = avg_scale > mixed_thresholds[i-1]
            else:
                mask = (avg_scale > mixed_thresholds[i-1]) & (avg_scale <= mixed_thresholds[i])
            layer_type = "粗糙"
        
        count = np.sum(mask)
        if count > 0:
            ratio = count / len(avg_scale) * 100
            range_str = f"{avg_scale[mask].min():.6f}~{avg_scale[mask].max():.6f}"
            print(f"    层{i:2d} ({layer_type}): {count:,}球 ({ratio:.1f}%) 范围: {range_str}")
    
    # 创建可视化
    print(f"\n🎨 生成可视化图表...")
    
    # 创建输出目录
    output_dir = "fine_scale_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 尺寸分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fine-Grained Gaussian Scale Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 全局分布
    ax = axes[0, 0]
    ax.hist(avg_scale, bins=100, alpha=0.7, color='blue', density=True)
    ax.set_xlabel('Average Scale')
    ax.set_ylabel('Density')
    ax.set_title('Overall Scale Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 超小球内部分布
    ax = axes[0, 1]
    ax.hist(ultra_small_scales, bins=50, alpha=0.7, color='red', density=True)
    ax.set_xlabel('Average Scale')
    ax.set_ylabel('Density')
    ax.set_title(f'Ultra-Small Gaussians Distribution\n({ultra_small_ratio:.1f}% of total)')
    ax.grid(True, alpha=0.3)
    
    # 对数尺度分布
    ax = axes[1, 0]
    ax.hist(log_scales, bins=50, alpha=0.7, color='green', density=True)
    ax.set_xlabel('Log10(Average Scale)')
    ax.set_ylabel('Density')
    ax.set_title('Log-Scale Distribution')
    ax.grid(True, alpha=0.3)
    
    # 累积分布
    ax = axes[1, 1]
    sorted_scales = np.sort(avg_scale)
    cumulative = np.arange(1, len(sorted_scales) + 1) / len(sorted_scales) * 100
    ax.plot(sorted_scales, cumulative, 'b-', linewidth=2)
    ax.set_xlabel('Average Scale')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('Cumulative Distribution')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 标注关键点
    key_percentiles = [50, 90, 95, 99]
    for p in key_percentiles:
        val = np.percentile(avg_scale, p)
        ax.axvline(val, color='red', linestyle='--', alpha=0.7)
        ax.text(val, p, f'{p}%', rotation=90, ha='right', va='bottom')
    
    plt.tight_layout()
    
    # 保存图像
    distribution_file = os.path.join(output_dir, 'fine_scale_distribution.png')
    plt.savefig(distribution_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 分布图保存: {distribution_file}")
    
    # 保存分析结果
    analysis_results = {
        'basic_stats': {
            'total_gaussians': len(avg_scale),
            'scale_range': [float(avg_scale.min()), float(avg_scale.max())],
            'mean': float(avg_scale.mean()),
            'median': float(np.median(avg_scale)),
            'std': float(avg_scale.std())
        },
        'percentile_analysis': {
            'percentiles': percentiles,
            'values': [float(v) for v in scale_percentiles]
        },
        'ultra_small_analysis': {
            'threshold': previous_threshold,
            'count': int(ultra_small_count),
            'ratio': float(ultra_small_ratio),
            'internal_stats': {
                'range': [float(ultra_small_scales.min()), float(ultra_small_scales.max())],
                'mean': float(ultra_small_scales.mean()),
                'median': float(np.median(ultra_small_scales)),
                'std': float(ultra_small_scales.std())
            }
        },
        'layering_strategies': {
            'logarithmic': {
                'description': '对数分层，适合幂律分布',
                'layers': len(log_percentiles) + 1,
                'thresholds': [float(t) for t in actual_thresholds]
            },
            'equal_count': {
                'description': f'等数量分层，{equal_count_layers}层',
                'layers': equal_count_layers,
                'thresholds': [float(t) for t in equal_count_thresholds]
            },
            'mixed': {
                'description': '混合策略：前95%精细 + 后5%粗糙',
                'layers': 20,
                'thresholds': [float(t) for t in mixed_thresholds],
                'fine_layers': 15,
                'coarse_layers': 5
            }
        },
        'recommendations': [
            "超小球占99%，需要进一步细分",
            "建议使用混合策略：对小球精细分层，对大球粗糙分层", 
            "对数分层可能更适合这种幂律分布",
            "每层球数差异巨大，渐进式评估需要考虑层级重要性"
        ]
    }
    
    results_file = os.path.join(output_dir, 'fine_scale_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"✅ 分析结果保存: {results_file}")
    
    print(f"\n🎉 精细尺寸分析完成!")
    print(f"📁 输出目录: {output_dir}/")
    print(f"💡 关键发现:")
    print(f"   1. 超小球确实占{ultra_small_ratio:.1f}%，需要细分")
    print(f"   2. 尺寸分布呈现明显的幂律特征")
    print(f"   3. 建议使用混合分层策略平衡精细度和实用性")
    
    return analysis_results

def main():
    analyze_fine_scale_distribution()

if __name__ == "__main__":
    main() 