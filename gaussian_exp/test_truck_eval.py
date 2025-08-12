#!/usr/bin/env python3
# 测试truck数据集和匹配的truck-150w模型
import sys
import os

# 添加路径
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

# 导入eval_like_train
from eval_like_train import evaluate_like_train

def main():
    # 使用匹配的模型和数据集
    model_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w"
    
    # 测试几个不同的PLY文件
    ply_files = [
        "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_12841_best_psnr/gaussian_ball.ply",
        "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_1297_best_loss/gaussian_ball.ply"
    ]
    
    for ply_file in ply_files:
        if os.path.exists(ply_file):
            print(f"\n{'='*80}")
            print(f"🧪 Testing with PLY: {os.path.basename(os.path.dirname(ply_file))}")
            print(f"{'='*80}")
            
            try:
                test_metrics, train_metrics = evaluate_like_train(model_path, ply_file)
                print(f"\n✅ Final Results for {os.path.basename(os.path.dirname(ply_file))}:")
                print(f"  Test PSNR: {test_metrics['psnr']:.2f} dB")
                print(f"  Train PSNR: {train_metrics['psnr']:.2f} dB")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"⚠️ PLY file not found: {ply_file}")

if __name__ == "__main__":
    main() 
# 测试truck数据集和匹配的truck-150w模型
import sys
import os

# 添加路径
sys.path.append("/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs")

# 导入eval_like_train
from eval_like_train import evaluate_like_train

def main():
    # 使用匹配的模型和数据集
    model_path = "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w"
    
    # 测试几个不同的PLY文件
    ply_files = [
        "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_12841_best_psnr/gaussian_ball.ply",
        "/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/gaussian_ball/iteration_1297_best_loss/gaussian_ball.ply"
    ]
    
    for ply_file in ply_files:
        if os.path.exists(ply_file):
            print(f"\n{'='*80}")
            print(f"🧪 Testing with PLY: {os.path.basename(os.path.dirname(ply_file))}")
            print(f"{'='*80}")
            
            try:
                test_metrics, train_metrics = evaluate_like_train(model_path, ply_file)
                print(f"\n✅ Final Results for {os.path.basename(os.path.dirname(ply_file))}:")
                print(f"  Test PSNR: {test_metrics['psnr']:.2f} dB")
                print(f"  Train PSNR: {train_metrics['psnr']:.2f} dB")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"⚠️ PLY file not found: {ply_file}")

if __name__ == "__main__":
    main() 