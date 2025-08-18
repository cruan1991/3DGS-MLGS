#!/usr/bin/env python3
"""
Student Network训练脚本
======================

训练COLMAP → 3DGS特征映射的神经网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 导入自定义模块
from create_training_dataset import StudentNetworkDataset
from student_network import StudentNetwork, StudentNetworkLoss, StudentNetworkEvaluator, create_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StudentNetworkTrainer:
    """Student Network训练器"""
    
    def __init__(self, 
                 model: StudentNetwork,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: StudentNetworkLoss,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler = None,
                 device: str = 'cuda',
                 save_dir: str = 'student_checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info(f"训练器初始化完成，设备: {device}")
        logger.info(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'position_mse': 0.0,
            'position_smooth': 0.0,
            'scale_mse': 0.0,
            'scale_smooth': 0.0,
            'stats_mse': 0.0
        }
        
        num_batches = 0
        progress_bar = tqdm(self.train_loader, desc="训练")
        
        for batch in progress_bar:
            # 数据转移到设备
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            radii = batch['radius'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(inputs, radii)
            
            # 计算损失
            losses = self.criterion(predictions, targets)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            self.optimizer.step()
            
            # 累加损失
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}"
            })
        
        # 平均损失
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        evaluator = StudentNetworkEvaluator(self.model, self.device)
        return evaluator.evaluate(self.val_loader, self.criterion)
    
    def train(self, num_epochs: int, 
              save_every: int = 10,
              early_stopping_patience: int = 20) -> Dict[str, List]:
        """完整训练循环"""
        logger.info(f"开始训练 {num_epochs} 个epochs...")
        
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch()
            
            # 验证
            val_losses = self.validate_epoch()
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_losses['total_loss'])
            self.train_history['val_loss'].append(val_losses['total_loss'])
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 计算时间
            epoch_time = time.time() - start_time
            
            # 日志输出
            logger.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            logger.info(f"  训练损失: {train_losses['total_loss']:.4f}")
            logger.info(f"  验证损失: {val_losses['total_loss']:.4f}")
            logger.info(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch + 1, is_best=True)
                early_stopping_counter = 0
                logger.info(f"  🎉 新的最佳模型! 验证损失: {self.best_val_loss:.4f}")
            else:
                early_stopping_counter += 1
            
            # 定期保存
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
            
            # 早停
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"早停触发! 连续 {early_stopping_patience} 个epochs验证损失未改善")
                break
        
        logger.info(f"训练完成! 最佳模型在epoch {self.best_epoch}, 验证损失: {self.best_val_loss:.4f}")
        
        # 保存训练历史
        self.save_training_history()
        
        return self.train_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history
        }
        
        # 保存当前检查点
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已保存: {best_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        logger.info(f"训练历史已保存: {history_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 损失曲线
        epochs = self.train_history['epoch']
        axes[0].plot(epochs, self.train_history['train_loss'], label='训练损失', color='blue')
        axes[0].plot(epochs, self.train_history['val_loss'], label='验证损失', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练和验证损失')
        axes[0].legend()
        axes[0].grid(True)
        
        # 学习率曲线
        axes[1].plot(epochs, self.train_history['learning_rate'], label='学习率', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('学习率')
        axes[1].set_title('学习率变化')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存: {self.save_dir / 'training_curves.png'}")

def load_datasets(data_dir: str, batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """加载训练、验证和测试数据集"""
    data_dir = Path(data_dir)
    
    # 创建数据集
    train_dataset = StudentNetworkDataset(str(data_dir / "train.h5"))
    val_dataset = StudentNetworkDataset(str(data_dir / "val.h5"))
    test_dataset = StudentNetworkDataset(str(data_dir / "test.h5"))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"数据集加载完成:")
    logger.info(f"  训练: {len(train_dataset)} 样本")
    logger.info(f"  验证: {len(val_dataset)} 样本")
    logger.info(f"  测试: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='训练Student Network')
    parser.add_argument('--data_dir', type=str, default='training_data',
                       help='训练数据目录')
    parser.add_argument('--save_dir', type=str, default='student_checkpoints',
                       help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--save_every', type=int, default=10,
                       help='每隔多少epochs保存一次')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='早停耐心值')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 加载元数据
    metadata_file = Path(args.data_dir) / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # 创建模型配置
    model_config = {
        'input_dim': metadata['feature_dims']['input_dim'],
        'output_dim': metadata['feature_dims']['output_dim'],
        'feature_dims': [64, 128, 256],
        'decoder_dims': [256, 128, 64],
        'num_radii': len(metadata['radii_used']),
        'use_radius_aware': True
    }
    
    # 创建模型
    model = create_model(model_config)
    
    # 加载数据
    train_loader, val_loader, test_loader = load_datasets(
        args.data_dir, 
        args.batch_size, 
        args.num_workers
    )
    
    # 创建损失函数
    criterion = StudentNetworkLoss(
        mse_weight=1.0,
        smooth_l1_weight=0.5,
        position_weight=2.0,
        scale_weight=1.5
    )
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # 创建训练器
    trainer = StudentNetworkTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir
    )
    
    # 开始训练
    training_history = trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping
    )
    
    # 测试最佳模型
    logger.info("测试最佳模型...")
    
    # 加载最佳模型
    best_checkpoint = torch.load(trainer.save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # 测试
    evaluator = StudentNetworkEvaluator(model, device)
    test_losses = evaluator.evaluate(test_loader, criterion)
    
    logger.info("测试结果:")
    for key, value in test_losses.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # 保存最终结果
    final_results = {
        'model_config': model_config,
        'training_args': vars(args),
        'best_epoch': trainer.best_epoch,
        'best_val_loss': trainer.best_val_loss,
        'test_losses': test_losses,
        'training_history': training_history
    }
    
    results_file = trainer.save_dir / "final_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info(f"🎉 训练完成! 结果已保存到: {results_file}")

if __name__ == "__main__":
    main() 