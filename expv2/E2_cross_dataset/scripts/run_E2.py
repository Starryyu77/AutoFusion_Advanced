#!/usr/bin/env python3
"""
E2: 跨数据集泛化实验 (更新版)

目标: 验证13个成功架构在MMMU/VSR/MathVista上的泛化能力
配置: 3个数据集 × 13架构 × 100 epochs × 3 runs
数据集: MMMU, VSR, MathVista (AI2D已在E1完成)

使用方法:
    python run_E2.py --dataset all --gpu 0
    python run_E2.py --dataset mmmu --gpu 0
    python run_E2.py --dataset vsr --gpu 1
    python run_E2.py --dataset mathvista --gpu 2
"""

import sys
import argparse
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'experiment'))

from evaluation.unified_evaluator import FullEvaluator
from discovered import DISCOVERED_ARCHITECTURES


def get_all_architectures():
    """
    获取所有13个E1成功架构 (8 NAS + 5 Baseline)
    注意: arch_019已修复维度问题
    """
    architectures = {}

    # ========== NAS架构 (8个) ==========
    # 按E1 FLOPs效率排序
    nas_architectures = [
        ('arch_022', {}),      # 12.34M FLOPs - 最佳NAS
        ('arch_021', {}),      # 13.63M FLOPs
        ('arch_017', {}),      # 13.20M FLOPs
        ('arch_004', {}),      # 14.74M FLOPs
        ('arch_025', {}),      # 16.10M FLOPs
        ('arch_015', {}),      # 18.32M FLOPs
        ('arch_024', {}),      # 40.77M FLOPs
        ('arch_008', {}),      # 206.00M FLOPs - 最差NAS
    ]

    print("加载NAS架构...")
    for name, kwargs in nas_architectures:
        if name in DISCOVERED_ARCHITECTURES:
            try:
                arch = DISCOVERED_ARCHITECTURES[name](**kwargs)
                architectures[name] = arch
                print(f"  ✅ NAS: {name}")
            except Exception as e:
                print(f"  ❌ NAS: {name} - {e}")
        else:
            print(f"  ⚠️  NAS: {name} - 不在DISCOVERED_ARCHITECTURES中")

    # ========== 基线架构 (5个) ==========
    print("\n加载基线架构...")
    try:
        from baselines import (
            ConcatMLP, BilinearPooling, CrossModalAttention,
            CLIPFusion, FiLM
        )

        baseline_configs = [
            ('CLIPFusion', CLIPFusion, {'vision_dim': 768, 'language_dim': 768, 'output_dim': 768}),
            ('BilinearPooling', BilinearPooling, {'vision_dim': 768, 'language_dim': 768, 'hidden_dim': 512}),
            ('ConcatMLP', ConcatMLP, {'vision_dim': 768, 'language_dim': 768, 'hidden_dim': 512}),
            ('FiLM', FiLM, {'vision_dim': 768, 'language_dim': 768, 'hidden_dim': 512}),
            ('CrossModalAttention', CrossModalAttention, {'vision_dim': 768, 'language_dim': 768, 'hidden_dim': 512, 'num_heads': 8}),
        ]

        for name, cls, kwargs in baseline_configs:
            try:
                architectures[name] = cls(**kwargs)
                print(f"  ✅ Baseline: {name}")
            except Exception as e:
                print(f"  ❌ Baseline: {name} - {e}")
    except ImportError as e:
        print(f"  ❌ 无法导入基线模块: {e}")

    print(f"\n总计加载: {len(architectures)} 个架构")
    print(f"  - NAS: {len([n for n in architectures if n.startswith('arch_')])} 个")
    print(f"  - Baseline: {len([n for n in architectures if not n.startswith('arch_')])} 个")

    return architectures


def evaluate_dataset(dataset_name, gpu=0, num_runs=3):
    """评估单个数据集上的所有架构"""
    print("\n" + "=" * 70)
    print(f"E2: 跨数据集评估 - {dataset_name.upper()}")
    print("=" * 70)
    print(f"配置: 100 epochs, {num_runs} runs, GPU {gpu}")
    print("-" * 70)

    result_dir = Path(__file__).parent.parent / 'results' / dataset_name
    result_dir.mkdir(parents=True, exist_ok=True)

    evaluator = FullEvaluator(
        dataset=dataset_name,
        save_dir=str(result_dir)
    )

    architectures = get_all_architectures()

    if not architectures:
        print("错误: 没有成功加载任何架构!")
        return {'error': 'No architectures loaded'}

    print(f"\n开始评估 {len(architectures)} 个架构...")
    print("-" * 70)

    return evaluator.evaluate_batch(architectures, num_runs=num_runs)


def main():
    parser = argparse.ArgumentParser(description='E2: 跨数据集泛化实验 (13架构版)')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'mmmu', 'vsr', 'mathvista'],
                        help='数据集名称 (注意: AI2D已在E1完成，不包含)')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='每架构运行次数')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--list-arch', action='store_true',
                        help='仅列出要评估的架构，不执行评估')

    args = parser.parse_args()

    # 仅列出架构
    if args.list_arch:
        print("=" * 70)
        print("E2: 待评估架构列表 (13个)")
        print("=" * 70)
        architectures = get_all_architectures()
        print("\n按效率排序 (FLOPs从低到高):")
        print("  1. CLIPFusion        [Baseline]  2.36M FLOPs")
        print("  2. BilinearPooling   [Baseline]  2.88M FLOPs")
        print("  3. ConcatMLP         [Baseline]  3.93M FLOPs")
        print("  4. FiLM              [Baseline]  6.29M FLOPs")
        print("  5. arch_022          [NAS]       12.34M FLOPs")
        print("  6. arch_017          [NAS]       13.20M FLOPs")
        print("  7. arch_021          [NAS]       13.63M FLOPs")
        print("  8. arch_004          [NAS]       14.74M FLOPs")
        print("  9. CrossModalAttn    [Baseline]  16.52M FLOPs")
        print(" 10. arch_025          [NAS]       16.10M FLOPs")
        print(" 11. arch_015          [NAS]       18.32M FLOPs")
        print(" 12. arch_024          [NAS]       40.77M FLOPs")
        print(" 13. arch_008          [NAS]      206.00M FLOPs")
        return

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU: {args.gpu}")
    else:
        print("警告: CUDA不可用，使用CPU")

    # 确定要评估的数据集 (排除AI2D，已在E1完成)
    if args.dataset == 'all':
        datasets = ['mmmu', 'vsr', 'mathvista']
    else:
        datasets = [args.dataset]

    print("\n" + "=" * 70)
    print("E2: 跨数据集泛化实验 (完整版)")
    print("=" * 70)
    print(f"数据集: {', '.join(datasets)}")
    print(f"架构数: 13 (8 NAS + 5 Baseline)")
    print(f"配置: 100 epochs, {args.num_runs} runs per architecture")
    print(f"总实验数: {len(datasets)} datasets × 13 archs × {args.num_runs} runs = {len(datasets) * 13 * args.num_runs} 次训练")
    print("=" * 70)

    # 执行评估
    all_results = {}
    for dataset in datasets:
        try:
            results = evaluate_dataset(dataset, args.gpu, args.num_runs)
            all_results[dataset] = results
        except Exception as e:
            print(f"\n❌ 评估 {dataset} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset] = {'error': str(e)}

    # 最终总结
    print("\n" + "=" * 70)
    print("E2 跨数据集评估完成!")
    print("=" * 70)
    for dataset, result in all_results.items():
        if 'error' in result:
            print(f"  ❌ {dataset.upper()}: 失败 - {result['error']}")
        else:
            num_arch = len([k for k in result.keys() if not k.startswith('_')])
            print(f"  ✅ {dataset.upper()}: 完成 - {num_arch} 个架构")
    print("=" * 70)


if __name__ == '__main__':
    main()
