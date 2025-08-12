#!/usr/bin/env python3
"""
安全的临时文件清理脚本
按重要性分类文件，让用户选择性删除
"""
import os
import sys
from datetime import datetime

def get_file_info(filepath):
    """获取文件信息"""
    stat = os.stat(filepath)
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime)
    return size, mtime

def main():
    gaussian_exp_dir = "gaussian_exp"
    
    # 文件分类
    files_classification = {
        "🚀 核心重要文件 - 绝对保留": [
            "eval_all_views_simple.py",     # 最终完美评估脚本
            "eval_with_correct_cameras.py", # 正确相机参数脚本
            "verify_rendering_pipeline.py", # 验证脚本
            "student_network_design.py",    # Student网络设计
            "student_self_supervised.py",   # 自监督策略
            "student_training.py",          # Student训练
        ],
        
        "📊 分析工具 - 可能有用": [
            "analyze_gaussians.py",         # 高斯球分析
            "filter_gaussians.py",          # 高斯球过滤
            "batch_test_filtered.py",       # 批量测试
            "gaussian_analysis.py",         # 高斯分析
        ],
        
        "🔧 调试文件 - 可考虑删除": [
            "debug_camera_params.py",       # 调试相机参数
            "debug_resolution.py",          # 调试分辨率  
            "test_exposure_loading.py",     # 测试曝光
            "test_train_eval.py",           # 测试训练评估
            "test_truck_eval.py",           # 测试卡车
            "quick_test_resolution.py",     # 快速测试分辨率
        ],
        
        "📝 早期版本 - 建议删除": [
            "eval_direct_colmap.py",        # 早期COLMAP评估
            "eval_from_train.py",           # 早期训练评估
            "eval_train_accurate.py",       # 被替代的准确评估
            "evaluate_gaussians.py",        # 早期高斯评估
            "evaluate_gaussians_simple.py", # 简化评估
            "create_simple_eval.py",        # 创建简单评估
            "create_complete_eval.py",      # 创建完整评估
            "eval_direct_scene.py",         # 直接场景评估
            "eval_like_train.py",           # 模拟训练评估
            "eval_all_views.py",            # 被simple版本替代
        ],
        
        "🗑️ 明确临时文件 - 安全删除": [
            "example_evaluation.sh",        # 示例脚本
        ]
    }
    
    print("🧹 3DGS文件清理工具")
    print("=" * 60)
    
    # 分析每个分类
    for category, file_list in files_classification.items():
        print(f"\n{category}:")
        existing_files = []
        
        for filename in file_list:
            filepath = os.path.join(gaussian_exp_dir, filename)
            if os.path.exists(filepath):
                size, mtime = get_file_info(filepath)
                size_kb = size / 1024
                existing_files.append((filename, size_kb, mtime))
                print(f"  ✓ {filename:<35} ({size_kb:>6.1f}KB, {mtime.strftime('%m-%d %H:%M')})")
            else:
                print(f"  ✗ {filename:<35} (不存在)")
        
        if category in ["🔧 调试文件 - 可考虑删除", "📝 早期版本 - 建议删除", "🗑️ 明确临时文件 - 安全删除"]:
            if existing_files:
                print(f"    💾 总大小: {sum(f[1] for f in existing_files):.1f}KB")
    
    print("\n" + "=" * 60)
    print("📋 建议的清理策略:")
    print("1. 🚀 核心重要文件 - 绝对不删除")
    print("2. 📊 分析工具 - 根据需要保留")  
    print("3. 🔧 调试文件 - 如果调试完成可以删除")
    print("4. 📝 早期版本 - 被更好版本替代，建议删除")
    print("5. 🗑️ 临时文件 - 安全删除")
    
    print("\n要执行自动清理吗？")
    print("选项:")
    print("  1 - 只删除明确的临时文件 (最安全)")
    print("  2 - 删除临时文件 + 早期版本")
    print("  3 - 删除临时文件 + 早期版本 + 调试文件")
    print("  4 - 手动选择要删除的文件")
    print("  0 - 不删除任何文件")
    
    choice = input("\n请选择 (0-4): ").strip()
    
    to_delete = []
    
    if choice == "1":
        to_delete = files_classification["🗑️ 明确临时文件 - 安全删除"]
    elif choice == "2":
        to_delete = (files_classification["🗑️ 明确临时文件 - 安全删除"] + 
                    files_classification["📝 早期版本 - 建议删除"])
    elif choice == "3":
        to_delete = (files_classification["🗑️ 明确临时文件 - 安全删除"] + 
                    files_classification["📝 早期版本 - 建议删除"] + 
                    files_classification["🔧 调试文件 - 可考虑删除"])
    elif choice == "4":
        print("\n手动选择模式:")
        all_deletable = (files_classification["🔧 调试文件 - 可考虑删除"] + 
                        files_classification["📝 早期版本 - 建议删除"] + 
                        files_classification["🗑️ 明确临时文件 - 安全删除"])
        
        for filename in all_deletable:
            filepath = os.path.join(gaussian_exp_dir, filename)
            if os.path.exists(filepath):
                size, mtime = get_file_info(filepath)
                response = input(f"删除 {filename} ({size/1024:.1f}KB)? (y/N): ").strip().lower()
                if response == 'y' or response == 'yes':
                    to_delete.append(filename)
    elif choice == "0":
        print("✅ 没有删除任何文件")
        return
    else:
        print("❌ 无效选择")
        return
    
    # 执行删除
    if to_delete:
        print(f"\n🗑️ 准备删除 {len(to_delete)} 个文件:")
        total_size = 0
        for filename in to_delete:
            filepath = os.path.join(gaussian_exp_dir, filename)
            if os.path.exists(filepath):
                size, _ = get_file_info(filepath)
                total_size += size
                print(f"  - {filename} ({size/1024:.1f}KB)")
        
        print(f"\n💾 总计释放空间: {total_size/1024:.1f}KB")
        
        confirm = input("\n确认删除? (yes/no): ").strip().lower()
        if confirm == 'yes':
            deleted_count = 0
            for filename in to_delete:
                filepath = os.path.join(gaussian_exp_dir, filename)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        print(f"✅ 已删除: {filename}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ 删除失败 {filename}: {e}")
            
            print(f"\n🎉 清理完成! 删除了 {deleted_count} 个文件，释放了 {total_size/1024:.1f}KB 空间")
        else:
            print("❌ 取消删除操作")
    else:
        print("✅ 没有选择要删除的文件")

if __name__ == "__main__":
    main() 