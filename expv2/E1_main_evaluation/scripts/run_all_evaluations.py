#!/usr/bin/env python3
"""
批量运行所有评估实验

支持:
- 基线评估 (5个传统架构)
- 发现架构评估 (10个NAS架构)
- 完整实验 (100 epochs, 3 runs)
- 快速测试 (10 epochs, 1 run)
"""

import sys
import argparse
import torch
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / 'experiment'))

from evaluation.unified_evaluator import QuickEvaluator, FullEvaluator
from baselines import ConcatMLP, BilinearPooling, CrossModalAttention, CLIPFusion, FiLM
from discovered import DISCOVERED_ARCHITECTURES


def get_baselines():
    """获取所有基线架构"""
    return {
        'ConcatMLP': ConcatMLP(vision_dim=768, language_dim=768, hidden_dim=512),
        'BilinearPooling': BilinearPooling(vision_dim=768, language_dim=768, hidden_dim=512),
        'CrossModalAttention': CrossModalAttention(vision_dim=768, language_dim=768, hidden_dim=512),
        'CLIPFusion': CLIPFusion(vision_dim=768, language_dim=768, output_dim=768),
        'FiLM': FiLM(vision_dim=768, language_dim=768, hidden_dim=512),
    }


def get_discovered_top10():
    """获取Phase 3发现的Top 10架构"""
    architectures = {}

    for name, ArchClass in DISCOVERED_ARCHITECTURES.items():
        try:
            # 尝试实例化
            arch = ArchClass()
            architectures[name] = arch
        except Exception as e:
            print(f"实例化 {name} 失败: {e}")

    return architectures


def run_quick_test(dataset='ai2d', arch_type='all'):
    """快速测试 (10 epochs)"""
    print("=" * 70)
    print("快速测试模式 (10 epochs)")
    print("=" * 70)

    evaluator = QuickEvaluator(dataset=dataset)

    architectures = {}

    if arch_type in ['all', 'baseline']:
        architectures.update(get_baselines())

    if arch_type in ['all', 'discovered']:
        # 只测试Top 3用于快速验证
        top3 = ['arch_024', 'arch_019', 'arch_021']
        discovered = get_discovered_top10()
        for name in top3:
            if name in discovered:
                architectures[name] = discovered[name]

    results = {}
    for name, module in architectures.items():
        try:
            result = evaluator.evaluate(module, name)
            results[name] = result
        except Exception as e:
            print(f"评估 {name} 失败: {e}")
            results[name] = {'error': str(e)}

    return results


def run_full_evaluation(dataset='ai2d', arch_type='all', num_runs=3):
    """完整评估 (100 epochs, 3 runs)"""
    print("=" * 70)
    print(f"完整评估模式 (100 epochs, {num_runs} runs)")
    print("=" * 70)

    evaluator = FullEvaluator(
        dataset=dataset,
        save_dir=f'results/{dataset}_full'
    )

    architectures = {}

    if arch_type in ['all', 'baseline']:
        print("\n加载基线架构...")
        architectures.update(get_baselines())
        print(f"  已加载 {len(get_baselines())} 个基线")

    if arch_type in ['all', 'discovered']:
        print("\n加载发现的架构...")
        discovered = get_discovered_top10()
        architectures.update(discovered)
        print(f"  已加载 {len(discovered)} 个发现架构")

    print(f"\n总共评估 {len(architectures)} 个架构")
    print("=" * 70)

    return evaluator.evaluate_batch(architectures, num_runs=num_runs)


def main():
    parser = argparse.ArgumentParser(description='AutoFusion ExpV2 评估脚本')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full'],
                        help='评估模式: quick (10ep) 或 full (100ep)')
    parser.add_argument('--dataset', type=str, default='ai2d',
                        choices=['ai2d', 'mmmu', 'vsr', 'mathvista'],
                        help='数据集')
    parser.add_argument('--arch-type', type=str, default='all',
                        choices=['all', 'baseline', 'discovered'],
                        help='评估的架构类型')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='每架构运行次数 (仅full模式)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')

    args = parser.parse_args()

    # 设置GPU
    torch.cuda.set_device(args.gpu)
    print(f"使用 GPU: {args.gpu}")

    # 运行评估
    if args.mode == 'quick':
        results = run_quick_test(args.dataset, args.arch_type)
    else:
        results = run_full_evaluation(args.dataset, args.arch_type, args.num_runs)

    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
