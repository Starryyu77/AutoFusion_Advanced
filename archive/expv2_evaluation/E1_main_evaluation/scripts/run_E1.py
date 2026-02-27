#!/usr/bin/env python3
"""
E1: AI2D主实验 - NAS vs Human Design

目标: 在主要数据集上全面评估NAS架构 vs 人工基线
配置: 100 epochs, 3 runs
架构: 10 NAS + 5 Baseline

使用方法:
    # 快速测试 (10 epochs, 1 run)
    python run_E1.py --mode quick --gpu 0

    # 完整实验 (100 epochs, 3 runs)
    python run_E1.py --mode full --gpu 0

    # 只评估基线
    python run_E1.py --mode full --arch-type baseline --gpu 0

    # 只评估发现架构
    python run_E1.py --mode full --arch-type discovered --gpu 0
"""

import sys
import argparse
import torch
from pathlib import Path

# 添加路径
SCRIPT_DIR = Path(__file__).parent
E1_DIR = SCRIPT_DIR.parent
EXPV2_DIR = E1_DIR.parent
SHARED_DIR = EXPV2_DIR / 'shared'
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'experiment'))

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
            arch = ArchClass()
            architectures[name] = arch
        except Exception as e:
            print(f"实例化 {name} 失败: {e}")

    return architectures


def run_quick_test(arch_type='all', gpu=0):
    """快速测试 (10 epochs)"""
    print("=" * 70)
    print("E1: AI2D主实验 - 快速测试 (10 epochs)")
    print("=" * 70)

    evaluator = QuickEvaluator(dataset='ai2d')
    architectures = {}

    if arch_type in ['all', 'baseline']:
        architectures.update(get_baselines())

    if arch_type in ['all', 'discovered']:
        top3 = ['arch_024', 'arch_019', 'arch_021']
        discovered = get_discovered_top10()
        for name in top3:
            if name in discovered:
                architectures[name] = discovered[name]

    results = {}
    result_dir = E1_DIR / 'results' / 'quick_test'
    result_dir.mkdir(parents=True, exist_ok=True)

    for name, module in architectures.items():
        try:
            print(f"\n评估: {name}")
            result = evaluator.evaluate(module, name)
            results[name] = result

            # 保存单个结果
            import json
            with open(result_dir / f'{name}_result.json', 'w') as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            print(f"评估 {name} 失败: {e}")
            results[name] = {'error': str(e)}

    # 保存汇总结果
    import json
    with open(result_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n结果保存至: {result_dir}")
    return results


def run_full_evaluation(arch_type='all', num_runs=3, gpu=0):
    """完整评估 (100 epochs, 3 runs)"""
    print("=" * 70)
    print(f"E1: AI2D主实验 - 完整评估 (100 epochs, {num_runs} runs)")
    print("=" * 70)

    result_dir = E1_DIR / 'results' / f'full_{num_runs}runs'
    evaluator = FullEvaluator(dataset='ai2d', save_dir=str(result_dir))

    architectures = {}

    if arch_type in ['all', 'baseline']:
        print("\n加载基线架构...")
        baselines = get_baselines()
        architectures.update(baselines)
        print(f"  已加载 {len(baselines)} 个基线")

    if arch_type in ['all', 'discovered']:
        print("\n加载发现的架构...")
        discovered = get_discovered_top10()
        architectures.update(discovered)
        print(f"  已加载 {len(discovered)} 个发现架构")

    print(f"\n总共评估 {len(architectures)} 个架构")
    print("=" * 70)

    return evaluator.evaluate_batch(architectures, num_runs=num_runs)


def main():
    parser = argparse.ArgumentParser(description='E1: AI2D主实验')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full'],
                        help='评估模式: quick (10ep) 或 full (100ep)')
    parser.add_argument('--arch-type', type=str, default='all',
                        choices=['all', 'baseline', 'discovered'],
                        help='评估的架构类型')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='每架构运行次数 (仅full模式)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')

    args = parser.parse_args()

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用 GPU: {args.gpu}")
    else:
        print("使用 CPU")

    # 运行评估
    if args.mode == 'quick':
        results = run_quick_test(args.arch_type, args.gpu)
    else:
        results = run_full_evaluation(args.arch_type, args.num_runs, args.gpu)

    print("\n" + "=" * 70)
    print("E1 评估完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
