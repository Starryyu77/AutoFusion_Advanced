#!/usr/bin/env python3
"""
E1本地快速测试 - 验证pipeline

不依赖完整实验环境，只测试架构实例化和基本流程
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / 'shared'))

import torch
import torch.nn as nn


def test_baselines():
    """测试基线架构"""
    print("\n" + "="*60)
    print("测试基线架构")
    print("="*60)

    from baselines import ConcatMLP, BilinearPooling, CrossModalAttention, CLIPFusion, FiLM

    baselines = {
        'ConcatMLP': ConcatMLP(vision_dim=768, language_dim=768, hidden_dim=512),
        'BilinearPooling': BilinearPooling(vision_dim=768, language_dim=768, hidden_dim=512),
        'CrossModalAttention': CrossModalAttention(vision_dim=768, language_dim=768, hidden_dim=512),
        'CLIPFusion': CLIPFusion(vision_dim=768, language_dim=768, output_dim=768),
        'FiLM': FiLM(vision_dim=768, language_dim=768, hidden_dim=512),
    }

    dummy_v = torch.randn(2, 768)
    dummy_t = torch.randn(2, 768)

    for name, model in baselines.items():
        try:
            output = model(dummy_v, dummy_t)
            print(f"✅ {name:25s} output shape: {output.shape}")
        except Exception as e:
            print(f"❌ {name:25s} error: {e}")

    return baselines


def test_discovered():
    """测试发现架构"""
    print("\n" + "="*60)
    print("测试发现架构 (Top 3)")
    print("="*60)

    from discovered import DISCOVERED_ARCHITECTURES

    top3 = ['arch_024', 'arch_019', 'arch_021']
    dummy_v = torch.randn(2, 768)
    dummy_t = torch.randn(2, 768)

    for name in top3:
        try:
            model = DISCOVERED_ARCHITECTURES[name]()
            output = model(dummy_v, dummy_t)
            print(f"✅ {name:15s} output shape: {output.shape}")
        except Exception as e:
            print(f"❌ {name:15s} error: {e}")


def test_metrics():
    """测试指标计算"""
    print("\n" + "="*60)
    print("测试指标计算")
    print("="*60)

    from evaluation.metrics import MetricsCalculator

    calc = MetricsCalculator(device='cpu')

    # 简单模型测试
    model = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Linear(512, 768)
    )

    try:
        metrics = calc.calculate_all(model, input_shape=(1, 768), num_runs=10)
        print(f"✅ FLOPs: {metrics['flops']:.2f}M")
        print(f"✅ Params: {metrics['params']:.2f}M")
        print(f"✅ Latency: {metrics['latency_ms']:.2f}ms")
    except Exception as e:
        print(f"❌ 指标计算错误: {e}")


def main():
    print("="*60)
    print("E1本地测试 - 验证Pipeline")
    print("="*60)

    test_baselines()
    test_discovered()
    test_metrics()

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == '__main__':
    main()
