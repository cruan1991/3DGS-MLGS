#!/usr/bin/env python3
"""
3DGS Student Network - Version 0.1 MVP
单文件实现：稀疏点云 → 密集高斯球的最简验证

目标：验证核心概念可行性
PSNR目标：10-15dB
时间：2-3天完成
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# ============================================================================
# 1. 数据加载模块
# ============================================================================

class SimpleTeacherLoader:
    """简单的Teacher模型加载器"""
    
    def __init__(self, ply_path):
        self.ply_path = ply_path
        self.gaussians = self.load_gaussians()
        
    def load_gaussians(self):
        """加载Teacher高斯球参数"""
        # 简化：直接从PLY文件读取关键参数
        try:
            # 这里应该是真实的PLY加载逻辑
            # 为了V0.1，我们用简化的加载方式
            print(f"Loading gaussians from {self.ply_path}")
            
            # TODO: 实现真实的PLY加载
            # 暂时用随机数据模拟（你需要替换为真实加载）
            num_gaussians = 150000  # 150万个高斯球
            gaussians = {
                'xyz': torch.randn(num_gaussians, 3) * 2.0,  # 位置
                'scale': torch.exp(torch.randn(num_gaussians, 3) * 0.5),  # 尺度
                'opacity': torch.sigmoid(torch.randn(num_gaussians, 1)),  # 透明度
                'color': torch.sigmoid(torch.randn(num_gaussians, 3))  # 颜色
            }
            
            print(f"Loaded {num_gaussians} gaussians")
            return gaussians
            
        except Exception as e:
            print(f"Error loading gaussians: {e}")
            raise
    
    def sample_sparse_points(self, sparsity=0.15):
        """采样稀疏点云"""
        total_points = len(self.gaussians['xyz'])
        num_sparse = int(total_points * sparsity)
        
        # 基于opacity的重要性采样
        weights = self.gaussians['opacity'].squeeze(-1)
        indices = torch.multinomial(weights, num_sparse, replacement=False)
        
        sparse_points = self.gaussians['xyz'][indices]
        print(f"Sampled {num_sparse} sparse points ({sparsity*100:.1f}%)")
        
        return sparse_points, indices

class SimpleDataset(torch.utils.data.Dataset):
    """V0.1的简单数据集"""
    
    def __init__(self, teacher_loader, num_samples=100):
        self.teacher_loader = teacher_loader
        self.samples = self.generate_samples(num_samples)
        
    def generate_samples(self, num_samples):
        """生成训练样本"""
        print(f"Generating {num_samples} training samples...")
        
        samples = []
        for i in tqdm(range(num_samples)):
            # 每个样本：稀疏点云 + 对应的GT高斯球
            sparse_points, sparse_indices = self.teacher_loader.sample_sparse_points()
            
            # 随机选择200个anchor点（从稀疏点中选择）
            num_anchors = min(200, len(sparse_points))
            anchor_indices = torch.randperm(len(sparse_points))[:num_anchors]
            anchors = sparse_points[anchor_indices]
            
            # 为每个anchor找到最近的30个GT高斯球
            gt_gaussians = self.find_nearest_gaussians(anchors, num_per_anchor=30)
            
            samples.append({
                'sparse_points': anchors,  # [200, 3]
                'gt_gaussians': gt_gaussians  # [200, 30, 7]
            })
            
        return samples
    
    def find_nearest_gaussians(self, anchors, num_per_anchor=30):
        """为每个anchor找到最近的GT高斯球"""
        all_positions = self.teacher_loader.gaussians['xyz']
        
        gt_gaussians = []
        for anchor in anchors:
            # 计算距离
            distances = torch.norm(all_positions - anchor.unsqueeze(0), dim=1)
            
            # 找到最近的30个
            _, nearest_indices = torch.topk(distances, num_per_anchor, largest=False)
            
            # 构建简化的7维高斯参数 [x,y,z, sx,sy,sz, alpha]
            nearest_gaussians = torch.cat([
                self.teacher_loader.gaussians['xyz'][nearest_indices],      # xyz [30,3]
                self.teacher_loader.gaussians['scale'][nearest_indices],    # scale [30,3] 
                self.teacher_loader.gaussians['opacity'][nearest_indices]   # alpha [30,1]
            ], dim=1)  # [30, 7]
            
            gt_gaussians.append(nearest_gaussians)
        
        return torch.stack(gt_gaussians)  # [num_anchors, 30, 7]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# ============================================================================
# 2. 网络模块
# ============================================================================

class SimplePointNet(nn.Module):
    """简化的PointNet编码器"""
    
    def __init__(self, input_dim=3, output_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [B, N, 3] -> [B, 3, N]
        x = x.transpose(2, 1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Global max pooling
        x = torch.max(x, 2)[0]  # [B, 256]
        
        return x

class StudentNetworkV01(nn.Module):
    """V0.1 最简Student网络"""
    
    def __init__(self, num_anchors=200, gaussians_per_anchor=30):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.gaussians_per_anchor = gaussians_per_anchor
        
        # 编码器：稀疏点云 -> 特征
        self.encoder = SimplePointNet(input_dim=3, output_dim=256)
        
        # 解码器：特征 -> 高斯参数
        # 输出：200个anchor × 30个高斯 × 7个参数
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_anchors * gaussians_per_anchor * 7)
        )
        
    def forward(self, sparse_points):
        # sparse_points: [B, num_anchors, 3]
        batch_size = sparse_points.shape[0]
        
        # 编码
        features = self.encoder(sparse_points)  # [B, 256]
        
        # 解码
        raw_output = self.decoder(features)  # [B, 200*30*7]
        
        # 重塑为高斯参数
        gaussians = raw_output.view(batch_size, self.num_anchors, self.gaussians_per_anchor, 7)
        
        # 应用激活函数确保参数合理
        gaussians = self.apply_constraints(gaussians)
        
        return gaussians
    
    def apply_constraints(self, gaussians):
        """应用参数约束"""
        # 分离不同参数
        positions = gaussians[..., :3]  # xyz
        scales = gaussians[..., 3:6]    # scale
        alphas = gaussians[..., 6:7]    # opacity
        
        # 应用约束
        scales = torch.exp(scales)  # 确保scale为正
        alphas = torch.sigmoid(alphas)  # 确保opacity在[0,1]
        
        # 重新组合
        constrained = torch.cat([positions, scales, alphas], dim=-1)
        
        return constrained

# ============================================================================
# 3. 训练模块
# ============================================================================

class SimpleTrainer:
    """V0.1的简单训练器"""
    
    def __init__(self, model, train_loader, val_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            sparse_points = batch['sparse_points']  # [B, 200, 3]
            gt_gaussians = batch['gt_gaussians']    # [B, 200, 30, 7]
            
            # 前向传播
            pred_gaussians = self.model(sparse_points)  # [B, 200, 30, 7]
            
            # 计算损失
            loss = self.criterion(pred_gaussians, gt_gaussians)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch):
        """验证"""
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                sparse_points = batch['sparse_points']
                gt_gaussians = batch['gt_gaussians']
                
                pred_gaussians = self.model(sparse_points)
                loss = self.criterion(pred_gaussians, gt_gaussians)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs=20):
        """完整训练流程"""
        print(f"Starting training for {num_epochs} epochs...")
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate(epoch)
            
            # 学习率调整
            self.scheduler.step()
            
            # 记录最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(f'best_model_epoch_{epoch}.pth')
            
            # 打印进度
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        print("Training completed!")
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filename)
        print(f"Checkpoint saved: {filename}")

# ============================================================================
# 4. 可视化模块
# ============================================================================

class SimpleVisualizer:
    """V0.1的简单可视化"""
    
    @staticmethod
    def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Progress - Version 0.1')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to {save_path}")
    
    @staticmethod
    def visualize_gaussians_3d(gaussians, title="Gaussian Distribution", save_path=None):
        """简单的3D可视化"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取位置信息
        if gaussians.dim() == 4:  # [B, anchors, gaussians_per_anchor, 7]
            positions = gaussians[0, :, :, :3].reshape(-1, 3)  # 取第一个batch
        else:
            positions = gaussians[:, :3]
        
        # 转换为numpy
        pos_np = positions.detach().cpu().numpy()
        
        # 绘制散点图
        ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], 
                  s=1, alpha=0.6, c=pos_np[:, 2], cmap='viridis')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def compare_pred_vs_gt(pred_gaussians, gt_gaussians, save_path='comparison.png'):
        """对比预测和GT"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': '3d'})
        
        # GT
        gt_pos = gt_gaussians[0, :, :, :3].reshape(-1, 3).detach().cpu().numpy()
        ax1.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
                   s=1, alpha=0.6, c='blue')
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Prediction
        pred_pos = pred_gaussians[0, :, :, :3].reshape(-1, 3).detach().cpu().numpy()
        ax2.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
                   s=1, alpha=0.6, c='red')
        ax2.set_title('Prediction')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# 5. 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='3DGS Student Network V0.1')
    parser.add_argument('--teacher-ply', type=str, required=True,
                       help='Path to teacher gaussian_ball.ply file')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of training samples to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 1. 数据准备
        print("=== Step 1: Data Preparation ===")
        teacher_loader = SimpleTeacherLoader(args.teacher_ply)
        
        # 生成数据集
        dataset = SimpleDataset(teacher_loader, num_samples=args.num_samples)
        
        # 划分训练/验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        
        print(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        # 2. 模型构建
        print("=== Step 2: Model Building ===")
        model = StudentNetworkV01(num_anchors=200, gaussians_per_anchor=30)
        model = model.to(device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created with {total_params:,} parameters")
        
        # 3. 训练
        print("=== Step 3: Training ===")
        trainer = SimpleTrainer(model, train_loader, val_loader)
        train_losses, val_losses = trainer.train(num_epochs=args.epochs)
        
        # 4. 可视化结果
        print("=== Step 4: Visualization ===")
        visualizer = SimpleVisualizer()
        
        # 绘制训练曲线
        visualizer.plot_training_curves(train_losses, val_losses)
        
        # 可视化一个预测样本
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            sparse_points = sample_batch['sparse_points'].to(device)
            gt_gaussians = sample_batch['gt_gaussians'].to(device)
            
            pred_gaussians = model(sparse_points)
            
            visualizer.compare_pred_vs_gt(pred_gaussians.cpu(), gt_gaussians.cpu())
        
        # 5. 保存最终模型
        final_model_path = 'student_network_v01_final.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # 6. 简单性能评估
        print("=== Step 5: Performance Summary ===")
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1] if val_losses else 0
        
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        
        # 估算PSNR（简化计算）
        mse = final_val_loss
        if mse > 0:
            psnr_estimate = -10 * np.log10(mse)
            print(f"Estimated PSNR: {psnr_estimate:.2f} dB")
        
        print("=== V0.1 MVP Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()