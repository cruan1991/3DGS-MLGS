# 🚀 3DGS项目快速提示词

## 背景
我在做3D Gaussian Splatting项目，已经完成了评估脚本的debug和Student网络的设计。

## 当前状态
- ✅ **修复了eval脚本**: 从10dB PSNR提升到28dB
- ✅ **关键教训**: 相机参数要从COLMAP原始文件读取，不能信任cameras.json
- ✅ **设计了Student网络**: PointNet++ + 自监督学习策略
- ❌ **待实现**: Student网络的具体训练代码

## 项目路径
- **工作目录**: `/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/`
- **核心模型**: `output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply`
- **工作脚本**: `gaussian_exp/eval_with_correct_cameras.py` (28.07 dB PSNR)

## 关键发现
1. **数据集无train/test划分** (`eval=False`)
2. **必须从COLMAP binary读取相机参数** (cameras.json有错)
3. **Exposure bug已修复** (gaussian_model.py line 340)
4. **Student网络用自监督策略** (Teacher 3DGS → 稀疏点云 → Student预测)

## 下一步
请帮我实现Student网络的训练代码，重点是：
1. 从PLY文件加载Teacher参数的工具
2. 稀疏点云采样和数据生成
3. 自监督训练循环

完整记忆档案见：`MEMORY_ARCHIVE.md` 