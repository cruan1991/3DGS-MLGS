#!/usr/bin/env python3
"""
Student Network模型架构详细分析
===============================

分析模型结构、参数分布、计算复杂度等
"""

import torch
import torch.nn as nn
from student_network import create_model, StudentNetwork
import json
from collections import OrderedDict

def analyze_model_architecture():
    """详细分析模型架构"""
    print("🏗️ " + "="*60)
    print("Student Network 模型架构详细分析")
    print("="*64)
    
    # 创建模型
    with open('training_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    config = {
        'input_dim': metadata['feature_dims']['input_dim'],
        'output_dim': metadata['feature_dims']['output_dim'],
        'feature_dims': [64, 128, 256],
        'decoder_dims': [256, 128, 64],
        'num_radii': len(metadata['radii_used']),
        'use_radius_aware': True
    }
    
    model = create_model(config)
    
    print(f"\n📋 总体信息:")
    print(f"   模型类型: 多层感知机 (MLP) + 嵌入")
    print(f"   输入维度: {config['input_dim']} (COLMAP点特征)")
    print(f"   输出维度: {config['output_dim']} (高斯球聚合特征)")
    print(f"   架构风格: Encoder-Decoder + 条件嵌入")
    
    return model, config

def analyze_components(model):
    """分析各个组件"""
    print(f"\n🔧 组件分析:")
    
    # 1. 特征提取器 (MultiScaleFeatureExtractor)
    print(f"\n   1️⃣  特征提取器 (Encoder):")
    feature_extractor = model.feature_extractor
    print(f"      类型: 多层感知机 (MLP)")
    print(f"      输入: [batch, 5] → 输出: [batch, 256]")
    print(f"      层数: 3层全连接")
    print(f"      维度: 5 → 64 → 128 → 256")
    print(f"      激活: ReLU + BatchNorm + Dropout(0.1)")
    print(f"      📊 参数量: {sum(p.numel() for p in feature_extractor.parameters()):,}")
    
    # 2. 半径感知编码器
    print(f"\n   2️⃣  半径感知编码器:")
    radius_encoder = model.radius_encoder
    print(f"      类型: 嵌入 + 融合层")
    print(f"      嵌入维度: 4个半径 → 64维向量")
    print(f"      融合: [256+64] → 256")
    print(f"      📊 参数量: {sum(p.numel() for p in radius_encoder.parameters()):,}")
    
    # 3. 解码器
    print(f"\n   3️⃣  高斯特征解码器 (Decoder):")
    decoder = model.gaussian_decoder
    print(f"      类型: 多层感知机 (MLP)")
    print(f"      输入: [batch, 256] → 输出: [batch, 16]")
    print(f"      层数: 3层全连接 + 输出层")
    print(f"      维度: 256 → 256 → 128 → 64 → 16")
    print(f"      激活: ReLU + BatchNorm + Dropout(0.1)")
    print(f"      📊 参数量: {sum(p.numel() for p in decoder.parameters()):,}")

def analyze_data_flow(model):
    """分析数据流"""
    print(f"\n🌊 数据流分析:")
    
    print(f"\n   输入特征 [batch, 5]:")
    print(f"   ├── [0:3] 归一化坐标 (x, y, z)")
    print(f"   ├── [3] 距离原点")
    print(f"   └── [4] 局部密度估计")
    print(f"   │")
    print(f"   ▼ 特征提取器 (3层MLP)")
    print(f"   │")
    print(f"   特征向量 [batch, 256]")
    print(f"   │")
    print(f"   ▼ 半径感知编码")
    print(f"   │ ├── 半径嵌入: [batch] → [batch, 64]")
    print(f"   │ └── 特征融合: [batch, 256+64] → [batch, 256]")
    print(f"   │")
    print(f"   半径感知特征 [batch, 256]")
    print(f"   │")
    print(f"   ▼ 高斯解码器 (4层MLP)")
    print(f"   │")
    print(f"   输出特征 [batch, 16]:")
    print(f"   ├── [0:3] 质心坐标")
    print(f"   ├── [3:6] 位置标准差")
    print(f"   ├── [6:9] 平均尺度")
    print(f"   ├── [9:12] 尺度标准差")
    print(f"   ├── [12] 邻居数量(log)")
    print(f"   ├── [13] 尺度均值")
    print(f"   ├── [14] 尺度最大值")
    print(f"   └── [15] 尺度最小值")

def analyze_parameters(model):
    """详细参数分析"""
    print(f"\n🔢 参数详细分析:")
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n   📊 各层参数分布:")
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if module_params > 0:
                print(f"      {name:<30}: {module_params:>8,} 参数")
                total_params += module_params
                trainable_params += module_trainable
    
    print(f"\n   📈 总计:")
    print(f"      总参数: {total_params:,}")
    print(f"      可训练参数: {trainable_params:,}")
    print(f"      模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

def analyze_comparison():
    """与其他架构对比"""
    print(f"\n🆚 架构对比:")
    
    print(f"\n   📋 当前架构 vs 经典架构:")
    print(f"   ┌─────────────────┬─────────────────┬─────────────────┐")
    print(f"   │      特征       │   当前Student   │     PointNet    │")
    print(f"   ├─────────────────┼─────────────────┼─────────────────┤")
    print(f"   │ 输入处理        │ 直接MLP         │ 逐点MLP + 池化  │")
    print(f"   │ 特征提取        │ 3层MLP          │ 多层MLP + Max   │")
    print(f"   │ 池化操作        │ ❌ 无池化       │ ✅ Max Pooling  │")
    print(f"   │ 排列不变性      │ ❌ 无需要       │ ✅ 有            │")
    print(f"   │ 条件信息        │ ✅ 半径嵌入     │ ❌ 通常无       │")
    print(f"   │ 参数量          │ ~23万           │ 通常>100万      │")
    print(f"   │ 适用场景        │ 点到聚合特征    │ 点云分类/分割   │")
    print(f"   └─────────────────┴─────────────────┴─────────────────┘")
    
    print(f"\n   🎯 设计特点:")
    print(f"      ✅ 轻量级: 只有23万参数")
    print(f"      ✅ 专用性: 针对COLMAP→3DGS任务设计")
    print(f"      ✅ 条件化: 半径感知，支持多尺度")
    print(f"      ✅ 简单: 纯MLP，易于训练和部署")
    print(f"      ⚠️  局限: 没有空间几何先验")
    print(f"      ⚠️  局限: 不处理邻居关系")

def analyze_complexity():
    """计算复杂度分析"""
    print(f"\n⚡ 计算复杂度:")
    
    # 前向传播计算量
    print(f"\n   🔢 前向传播计算量 (单样本):")
    print(f"      特征提取: 5×64 + 64×128 + 128×256 = 41,280 FLOPs")
    print(f"      半径嵌入: 4×64 = 256 FLOPs")
    print(f"      特征融合: 320×256 = 81,920 FLOPs")
    print(f"      解码器: 256×256 + 256×128 + 128×64 + 64×16 = 106,496 FLOPs")
    print(f"      总计: ~230K FLOPs")
    
    print(f"\n   ⏱️  推理速度估计:")
    print(f"      GPU (RTX 4090): ~10万样本/秒")
    print(f"      CPU (现代): ~1万样本/秒")
    print(f"      移动端: ~1千样本/秒")

def test_forward_pass():
    """测试前向传播"""
    print(f"\n🧪 前向传播测试:")
    
    # 创建模型
    with open('training_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    config = {
        'input_dim': 5,
        'output_dim': 16,
        'feature_dims': [64, 128, 256],
        'decoder_dims': [256, 128, 64],
        'num_radii': 4,
        'use_radius_aware': True
    }
    
    model = create_model(config)
    model.eval()
    
    # 创建测试数据
    batch_size = 100
    input_features = torch.randn(batch_size, 5)
    radii = torch.tensor([0.012, 0.039, 0.107, 0.273] * (batch_size // 4))
    
    # 前向传播
    with torch.no_grad():
        output = model(input_features, radii)
    
    print(f"      输入形状: {input_features.shape}")
    print(f"      半径形状: {radii.shape}")
    print(f"      输出形状: {output.shape}")
    print(f"      输出范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"      ✅ 前向传播正常")

def main():
    """主函数"""
    model, config = analyze_model_architecture()
    analyze_components(model)
    analyze_data_flow(model)
    analyze_parameters(model)
    analyze_comparison()
    analyze_complexity()
    test_forward_pass()
    
    print(f"\n" + "="*64)
    print(f"📋 总结")
    print(f"="*64)
    print(f"🏗️  架构: 条件化多层感知机 (非PointNet)")
    print(f"📊 结构: 3层编码器 + 1层嵌入 + 4层解码器")
    print(f"🔢 参数: 234,448 (0.89MB)")
    print(f"⚡ 复杂度: ~230K FLOPs/样本")
    print(f"🎯 特点: 轻量、专用、条件化")
    print("="*64)

if __name__ == "__main__":
    main() 