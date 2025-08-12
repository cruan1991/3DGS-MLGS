#!/usr/bin/env python3
# 批量测试过滤后的高斯球文件
import subprocess
import sys
import os

def test_ply_file(ply_path, name):
    """测试单个PLY文件"""
    print(f"\n{'='*60}")
    print(f"🎯 测试 {name}")
    print(f"📁 文件: {ply_path}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 
        "gaussian_exp/eval_with_correct_cameras.py",
        "--model-path", "./output/truck-150w",
        "--ply-path", ply_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        if result.returncode == 0:
            print("✅ 成功完成")
            # 提取PSNR结果
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "训练风格评估PSNR" in line or "全体相机平均PSNR" in line:
                    print(f"📊 {line.strip()}")
                elif "达成率" in line:
                    print(f"📈 {line.strip()}")
                elif "✅ 加载了" in line and "个高斯球" in line:
                    print(f"🎯 {line.strip()}")
        else:
            print("❌ 执行失败")
            print(f"错误输出: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时（5分钟）")
    except Exception as e:
        print(f"❌ 执行错误: {e}")

def main():
    # 要测试的PLY文件列表
    test_files = [
        ("./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/filtered_alpha005.ply", "Alpha=0.05过滤版本"),
        ("./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/filtered_alpha003.ply", "Alpha=0.03过滤版本"), 
        ("./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/filtered_alpha001.ply", "Alpha=0.01过滤版本"),
        ("./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply", "原始版本（对照）")
    ]
    
    print("🚀 批量测试过滤后的高斯球文件")
    print(f"📊 将测试 {len(test_files)} 个文件")
    
    results = []
    
    for ply_path, name in test_files:
        # 检查文件是否存在
        if not os.path.exists(ply_path):
            print(f"\n❌ 文件不存在: {ply_path}")
            continue
            
        test_ply_file(ply_path, name)
    
    print(f"\n{'='*60}")
    print("🎉 批量测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 
# 批量测试过滤后的高斯球文件
import subprocess
import sys
import os

def test_ply_file(ply_path, name):
    """测试单个PLY文件"""
    print(f"\n{'='*60}")
    print(f"🎯 测试 {name}")
    print(f"📁 文件: {ply_path}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 
        "gaussian_exp/eval_with_correct_cameras.py",
        "--model-path", "./output/truck-150w",
        "--ply-path", ply_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        if result.returncode == 0:
            print("✅ 成功完成")
            # 提取PSNR结果
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "训练风格评估PSNR" in line or "全体相机平均PSNR" in line:
                    print(f"📊 {line.strip()}")
                elif "达成率" in line:
                    print(f"📈 {line.strip()}")
                elif "✅ 加载了" in line and "个高斯球" in line:
                    print(f"🎯 {line.strip()}")
        else:
            print("❌ 执行失败")
            print(f"错误输出: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时（5分钟）")
    except Exception as e:
        print(f"❌ 执行错误: {e}")

def main():
    # 要测试的PLY文件列表
    test_files = [
        ("./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/filtered_alpha005.ply", "Alpha=0.05过滤版本"),
        ("./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/filtered_alpha003.ply", "Alpha=0.03过滤版本"), 
        ("./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/filtered_alpha001.ply", "Alpha=0.01过滤版本"),
        ("./output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply", "原始版本（对照）")
    ]
    
    print("🚀 批量测试过滤后的高斯球文件")
    print(f"📊 将测试 {len(test_files)} 个文件")
    
    results = []
    
    for ply_path, name in test_files:
        # 检查文件是否存在
        if not os.path.exists(ply_path):
            print(f"\n❌ 文件不存在: {ply_path}")
            continue
            
        test_ply_file(ply_path, name)
    
    print(f"\n{'='*60}")
    print("🎉 批量测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 