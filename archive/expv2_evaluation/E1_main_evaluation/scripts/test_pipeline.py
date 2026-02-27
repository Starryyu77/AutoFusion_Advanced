#!/usr/bin/env python3
"""
E1 Pipeline测试 - 不依赖CLIP

验证:
1. 架构可以实例化
2. 数据加载器可以工作
3. 训练流程可以运行
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / 'shared'))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent.parent / 'experiment'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """Dummy数据集 - 匹配训练器期望的格式"""
    def __init__(self, num_samples=32):
        self.vision = torch.randn(num_samples, 768)
        self.text = torch.randn(num_samples, 768)
        self.labels = torch.randint(0, 4, (num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'image': self.vision[idx],
            'text': self.text[idx],
            'label': self.labels[idx]
        }


def create_dummy_loaders(batch_size=4):
    """创建dummy数据加载器用于测试"""
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, loader, loader


class DummyEncoder(nn.Module):
    """Dummy编码器用于测试"""
    def forward(self, x):
        return x


def test_training_pipeline():
    """测试训练pipeline"""
    print("\n" + "="*60)
    print("测试训练Pipeline")
    print("="*60)

    from baselines import ConcatMLP
    from evaluation.full_trainer import FullTrainer

    # 创建模型
    fusion = ConcatMLP(vision_dim=768, language_dim=768, hidden_dim=128)
    vision_enc = DummyEncoder()
    text_enc = DummyEncoder()

    # 创建训练器
    trainer = FullTrainer(
        fusion_module=fusion,
        vision_encoder=vision_enc,
        text_encoder=text_enc,
        num_classes=4,
        config={
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 1e-3,
            'mixed_precision': False,
            'early_stopping': False,
            'save_best': False
        }
    )

    # 创建dummy数据
    train_loader, val_loader, test_loader = create_dummy_loaders()

    print("开始训练 (2 epochs)...")
    history = trainer.train(train_loader, val_loader, save_dir=None)

    print(f"✅ 训练完成!")
    print(f"   Best Val Acc: {history['best_val_acc']:.4f}")
    print(f"   Epochs: {history['epochs_trained']}")

    # 测试评估
    test_results = trainer.evaluate(test_loader)
    print(f"   Test Acc: {test_results['accuracy']:.4f}")

    return True


def main():
    print("="*60)
    print("E1 Pipeline测试 (无CLIP依赖)")
    print("="*60)

    try:
        test_training_pipeline()
        print("\n" + "="*60)
        print("✅ 所有测试通过!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
