#!/bin/bash

# 清理冗余文件脚本
# 保留训练备份文件 (*.py.bak)，删除其他冗余文件

echo "🧹 开始清理冗余文件..."
echo "⚠️  训练备份文件 (*.py.bak) 将被保留"

# 切换到项目根目录
cd /shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs

# 记录删除的文件
DELETED_COUNT=0

# 函数：安全删除文件
safe_delete() {
    if [ -f "$1" ]; then
        echo "删除: $1"
        rm "$1"
        ((DELETED_COUNT++))
    fi
}

echo ""
echo "📁 清理 gaussian_exp/ 目录..."

# gaussian_exp/ 目录下的冗余评估脚本
safe_delete "gaussian_exp/eval_all_views_simple.py"
safe_delete "gaussian_exp/eval_all_views.py"
safe_delete "gaussian_exp/eval_direct_scene.py"
safe_delete "gaussian_exp/eval_like_train.py"
safe_delete "gaussian_exp/eval_with_correct_cameras.py"
safe_delete "gaussian_exp/create_complete_eval.py"
safe_delete "gaussian_exp/create_simple_eval.py"

# 调试和测试文件
safe_delete "gaussian_exp/debug_resolution.py"
safe_delete "gaussian_exp/debug_camera_params.py"
safe_delete "gaussian_exp/test_exposure_loading.py"
safe_delete "gaussian_exp/test_truck_eval.py"
safe_delete "gaussian_exp/batch_test_filtered.py"

# 重复的过滤脚本
safe_delete "gaussian_exp/filter_gaussian.py"
safe_delete "gaussian_exp/filter_and_save_ply.py"

# 分析脚本（根据需要删除）
safe_delete "gaussian_exp/analyze_gaussians.py"
safe_delete "gaussian_exp/analyze_truck.py"
safe_delete "gaussian_exp/correlation_analysis.py"
safe_delete "gaussian_exp/analyze_point_vs_gaussian_correlation.py"
safe_delete "gaussian_exp/analyze_point_cloud.py"
safe_delete "gaussian_exp/visualizations.py"
safe_delete "gaussian_exp/gaussians_io.py"
safe_delete "gaussian_exp/inspect_ply.py"
safe_delete "gaussian_exp/extract_gaussian_stats_from_checkpoint.py"

# 学生网络相关（如果不再需要）
safe_delete "gaussian_exp/student_self_supervised.py"
safe_delete "gaussian_exp/student_training.py"
safe_delete "gaussian_exp/student_network_design.py"

echo ""
echo "📁 清理主目录..."

# 重复的评估脚本
safe_delete "evaluate_v04.py"
safe_delete "evaluate_v05_enhanced.py"
safe_delete "proper_evaluation_metrics.py"
safe_delete "analyze_psnr_calculation.py"
safe_delete "quick_performance_test.py"
safe_delete "explain_student_network.py"
safe_delete "progressive_training.py"
safe_delete "advanced_training_monitor.py"
safe_delete "analyze_loss.py"
safe_delete "monitor_training.py"

# 测试文件
safe_delete "test_simplified_network.py"
safe_delete "test_student_network.py"
safe_delete "try_gaussian.py"
safe_delete "check_dependencies.py"
safe_delete "view_gaussian.py"
safe_delete "run_analysis_gs.py"

# 检查脚本
safe_delete "tmux_check.sh"
safe_delete "check_v05_progress.sh"

# 日志文件
safe_delete "v05_training.log"

# 不必要的超大scene.py文件（1GB）
if [ -f "scene.py" ]; then
    file_size=$(stat -f%z "scene.py" 2>/dev/null || stat -c%s "scene.py" 2>/dev/null || echo 0)
    if [ "$file_size" -gt 100000000 ]; then  # 如果大于100MB
        echo "删除超大文件: scene.py ($file_size bytes)"
        rm "scene.py"
        ((DELETED_COUNT++))
    fi
fi

echo ""
echo "🗂️  保留的重要文件:"
echo "   ✅ eval_filtered.py (核心评估脚本)"
echo "   ✅ train.py (主训练脚本)"
echo "   ✅ train*.py.bak (训练备份文件)"
echo "   ✅ filter_gaussians.py (高斯球过滤)"
echo "   ✅ gaussian_analysis.py (分析工具)"
echo "   ✅ scene/ utils/ gaussian_renderer/ arguments/ (核心模块)"

echo ""
echo "✨ 清理完成!"
echo "📊 共删除 $DELETED_COUNT 个冗余文件"
echo ""
echo "💾 建议接下来也清理一些旧的模型检查点："
echo "   - best_model_v02_epoch_*.pth (如果v03/v04更好)"
echo "   - 只保留最近几个epoch的checkpoint"
echo ""
echo "💡 如果确认没问题，可以运行: git add . && git commit -m 'Clean up redundant files'" 