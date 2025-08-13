import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ---- 中文字体设置（任选其一/就近系统）----
import matplotlib
from matplotlib import rcParams

# Windows 常见中文字体
# rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

# macOS 常见中文字体
rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Hiragino Sans GB']

# Linux 常见中文字体（若已安装）
# rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei']

rcParams['axes.unicode_minus'] = False  # 解决负号变方块


# 添加3dgs根目录到path
sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

from scene import GaussianModel

def analyze_gaussian_scales(ply_path, output_dir='scale_analysis'):
    """分析高斯球的尺寸分布"""
    print("🔍 分析高斯球尺寸分布...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载高斯球
    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_path, use_train_test_exp=False)
    
    # 获取缩放参数 (scaling)
    scaling = gaussians.get_scaling.detach().cpu().numpy()  # [N, 3]
    print(f"📊 高斯球数量: {scaling.shape[0]:,}")
    print(f"📏 缩放参数形状: {scaling.shape}")
    
    # 计算各种尺寸指标
    # 1. 平均尺寸 (3个轴的平均)
    avg_scale = np.mean(scaling, axis=1)
    
    # 2. 最大尺寸 (3个轴的最大值)
    max_scale = np.max(scaling, axis=1)
    
    # 3. 最小尺寸 (3个轴的最小值)
    min_scale = np.min(scaling, axis=1)
    
    # 4. 体积 (3个轴的乘积)
    volume = np.prod(scaling, axis=1)
    
    # 5. 各轴独立分析
    scale_x, scale_y, scale_z = scaling[:, 0], scaling[:, 1], scaling[:, 2]
    
    print("\n📈 尺寸统计:")
    metrics = {
        'avg_scale': avg_scale,
        'max_scale': max_scale, 
        'min_scale': min_scale,
        'volume': volume,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'scale_z': scale_z
    }
    
    results = {}
    for name, data in metrics.items():
        stats = {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'q90': float(np.percentile(data, 90)),
            'q95': float(np.percentile(data, 95)),
            'q99': float(np.percentile(data, 99))
        }
        results[name] = stats
        
        print(f"\n{name}:")
        print(f"  范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  均值: {stats['mean']:.6f}, 中位数: {stats['median']:.6f}")
        print(f"  标准差: {stats['std']:.6f}")
        print(f"  分位数: Q25={stats['q25']:.6f}, Q75={stats['q75']:.6f}, Q95={stats['q95']:.6f}")
    
    # 创建可视化
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('高斯球尺寸分布分析', fontsize=16, fontweight='bold')
    
    # 1. 平均尺寸分布
    ax = axes[0, 0]
    ax.hist(avg_scale, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title('平均尺寸分布')
    ax.set_xlabel('平均尺寸')
    ax.set_ylabel('数量')
    ax.set_yscale('log')
    
    # 2. 最大尺寸分布
    ax = axes[0, 1]
    ax.hist(max_scale, bins=100, alpha=0.7, color='red', edgecolor='black')
    ax.set_title('最大尺寸分布')
    ax.set_xlabel('最大尺寸')
    ax.set_ylabel('数量')
    ax.set_yscale('log')
    
    # 3. 体积分布
    ax = axes[0, 2]
    ax.hist(volume, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax.set_title('体积分布')
    ax.set_xlabel('体积')
    ax.set_ylabel('数量')
    ax.set_yscale('log')
    
    # 4. X轴尺寸
    ax = axes[1, 0]
    ax.hist(scale_x, bins=100, alpha=0.7, color='orange', edgecolor='black')
    ax.set_title('X轴尺寸分布')
    ax.set_xlabel('X轴尺寸')
    ax.set_ylabel('数量')
    ax.set_yscale('log')
    
    # 5. Y轴尺寸
    ax = axes[1, 1]
    ax.hist(scale_y, bins=100, alpha=0.7, color='purple', edgecolor='black')
    ax.set_title('Y轴尺寸分布')
    ax.set_xlabel('Y轴尺寸')
    ax.set_ylabel('数量')
    ax.set_yscale('log')
    
    # 6. Z轴尺寸
    ax = axes[1, 2]
    ax.hist(scale_z, bins=100, alpha=0.7, color='brown', edgecolor='black')
    ax.set_title('Z轴尺寸分布')
    ax.set_xlabel('Z轴尺寸')
    ax.set_ylabel('数量')
    ax.set_yscale('log')
    
    # 7. 箱线图对比
    ax = axes[2, 0]
    data_for_box = [avg_scale, max_scale, min_scale]
    labels = ['平均尺寸', '最大尺寸', '最小尺寸']
    ax.boxplot(data_for_box, labels=labels)
    ax.set_title('尺寸指标对比')
    ax.set_ylabel('尺寸值')
    ax.set_yscale('log')
    
    # 8. 累积分布函数
    ax = axes[2, 1]
    sorted_avg = np.sort(avg_scale)
    p = np.arange(len(sorted_avg)) / len(sorted_avg)
    ax.plot(sorted_avg, p, linewidth=2, label='平均尺寸')
    
    sorted_max = np.sort(max_scale)
    ax.plot(sorted_max, p, linewidth=2, label='最大尺寸')
    
    ax.set_title('累积分布函数')
    ax.set_xlabel('尺寸')
    ax.set_ylabel('累积概率')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. 散点图：平均尺寸 vs 体积
    ax = axes[2, 2]
    # 采样显示（避免点太多）
    sample_indices = np.random.choice(len(avg_scale), size=min(10000, len(avg_scale)), replace=False)
    ax.scatter(avg_scale[sample_indices], volume[sample_indices], alpha=0.5, s=1)
    ax.set_title('平均尺寸 vs 体积')
    ax.set_xlabel('平均尺寸')
    ax.set_ylabel('体积')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, 'scale_distribution_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化保存到: {plot_file}")
    
    # 建议分层方案
    print("\n🎯 建议的尺寸分层方案:")
    
    # 基于分位数的分层
    avg_thresholds = [
        np.percentile(avg_scale, 20),   # 超小球 (0-20%)
        np.percentile(avg_scale, 50),   # 小球 (20-50%)
        np.percentile(avg_scale, 80),   # 中球 (50-80%)
        np.percentile(avg_scale, 95),   # 大球 (80-95%)
        # 超大球 (95-100%)
    ]
    
    layer_names = ['超小球', '小球', '中球', '大球', '超大球']
    layer_info = []
    
    for i, name in enumerate(layer_names):
        if i == 0:
            mask = avg_scale <= avg_thresholds[0]
            range_str = f"≤{avg_thresholds[0]:.6f}"
        elif i == len(layer_names) - 1:
            mask = avg_scale > avg_thresholds[-1]
            range_str = f">{avg_thresholds[-1]:.6f}"
        else:
            mask = (avg_scale > avg_thresholds[i-1]) & (avg_scale <= avg_thresholds[i])
            range_str = f"{avg_thresholds[i-1]:.6f}~{avg_thresholds[i]:.6f}"
        
        count = np.sum(mask)
        percentage = count / len(avg_scale) * 100
        
        layer_info.append({
            'layer_id': i,
            'name': name,
            'threshold_range': range_str,
            'count': int(count),
            'percentage': percentage,
            'avg_scale_range': range_str,
            'mask': mask.tolist()  # 用于后续生成PLY文件
        })
        
        print(f"  层{i} ({name}): {range_str}, {count:,}球 ({percentage:.1f}%)")
    
    # 保存结果
    results['layer_suggestions'] = layer_info
    results['thresholds'] = [float(t) for t in avg_thresholds]
    
    results_file = os.path.join(output_dir, 'scale_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ 分析结果保存到: {results_file}")
    
    return results, avg_thresholds

def main():
    print("🔍 高斯球尺寸分布分析")
    print("=" * 50)
    
    # PLY文件路径
    ply_path = "./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
    
    if not os.path.exists(ply_path):
        print(f"❌ PLY文件不存在: {ply_path}")
        return
    
    # 分析尺寸分布
    results, thresholds = analyze_gaussian_scales(ply_path)
    
    print(f"\n🎉 尺寸分析完成!")
    print(f"📁 结果保存在: scale_analysis/")
    print(f"📊 建议的5层分层方案基于平均尺寸分位数")
    print(f"📈 可查看 scale_analysis/scale_distribution_analysis.png 了解分布详情")

if __name__ == "__main__":
    main() 