#!/bin/bash

# 示例：如何使用 evaluate_gaussians.py 评估不同的高斯模型

echo "🔍 高斯球模型评估示例"
echo "====================="

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs

# 基础路径设置
BASE_DIR="/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs"
OUTPUT_BASE_DIR="/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w"

# 相机配置文件
CONFIG_PATH="$OUTPUT_BASE_DIR/cameras.json"

echo ""
echo "📋 可用的高斯模型："
echo "1. 原始训练模型: $OUTPUT_BASE_DIR/point_cloud/iteration_30000/point_cloud.ply"
echo "2. 高斯球过滤模型: $OUTPUT_BASE_DIR/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply"
echo "3. 其他迭代检查点..."

echo ""
echo "🚀 开始评估..."

# 示例1: 评估高斯球过滤模型
echo ""
echo "📊 评估高斯球过滤模型..."
cd "$BASE_DIR/gaussian_exp"
python evaluate_gaussians.py \
    --ply-path "$OUTPUT_BASE_DIR/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply" \
    --config-path "$CONFIG_PATH" \
    --output-dir "renders/gaussian_ball_eval"

# 示例2: 评估原始训练模型（如果存在）
if [ -f "$OUTPUT_BASE_DIR/point_cloud/iteration_30000/point_cloud.ply" ]; then
    echo ""
    echo "📊 评估原始训练模型..."
    python evaluate_gaussians.py \
        --ply-path "$OUTPUT_BASE_DIR/point_cloud/iteration_30000/point_cloud.ply" \
        --config-path "$CONFIG_PATH" \
        --output-dir "renders/original_model_eval"
else
    echo "ℹ️  原始训练模型不存在，跳过评估"
fi

# 示例3: 评估其他检查点（如果需要）
echo ""
echo "💡 要评估其他模型，使用以下命令格式："
echo "python evaluate_gaussians.py \\"
echo "    --ply-path /path/to/your/model.ply \\"
echo "    --config-path $CONFIG_PATH \\"
echo "    --output-dir renders/your_model_name"

echo ""
echo "📂 评估结果保存在 renders/ 目录下"
echo "   - 渲染图像: renders/*/rendered_images/"
echo "   - PSNR指标: renders/*/metrics.csv"

echo ""
echo "✅ 评估完成！" 