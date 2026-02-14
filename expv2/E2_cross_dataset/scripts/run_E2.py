#!/usr/bin/env python3
"""
E2: 跨数据集泛化实验

目标: 验证发现架构在多个数据集上的通用性
配置: 4个数据集 × Top 5架构 × 100 epochs
数据集: AI2D, MMMU, VSR, MathVista

使用方法:
    python run_E2.py --dataset all --gpu 0
    python run_E2.py --dataset mmmu --gpu 0
    python run_E2.py --dataset vsr --gpu 0
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


def get_top5_architectures():
    """获取Top 5发现架构"""
    top5_names = ['arch_024', 'arch_019', 'arch_021', 'arch_012', 'arch_025']
    architectures = {}

    for name in top5_names:
        if name in DISCOVERED_ARCHITECTURES:
            try:
                arch = DISCOVERED_ARCHITECTURES[name]()
                architectures[name] = arch
            except Exception as e:
                print(f"实例化 {name} 失败: {e}")

    return architectures


def evaluate_dataset(dataset_name, gpu=0, num_runs=3):
    """评估单个数据集"""
    print("=" * 70)
    print(f"E2: 跨数据集评估 - {dataset_name.upper()}")
    print("=" * 70)

    result_dir = Path(__file__).parent.parent / 'results' / dataset_name
    evaluator = FullEvaluator(
        dataset=dataset_name,
        save_dir=str(result_dir)
    )

    architectures = get_top5_architectures()
    print(f"评估 {len(architectures)} 个架构")

    return evaluator.evaluate_batch(architectures, num_runs=num_runs)


def main():
    parser = argparse.ArgumentParser(description='E2: 跨数据集泛化实验')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'ai2d', 'mmmu', 'vsr', 'mathvista'],
                        help='数据集名称')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='每架构运行次数')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    datasets = ['ai2d', 'mmmu', 'vsr', 'mathvista'] if args.dataset == 'all' else [args.dataset]

    all_results = {}
    for dataset in datasets:
        try:
            results = evaluate_dataset(dataset, args.gpu, args.num_runs)
            all_results[dataset] = results
        except Exception as e:
            print(f"评估 {dataset} 失败: {e}")
            all_results[dataset] = {'error': str(e)}

    print("\n" + "=" * 70)
    print("E2 跨数据集评估完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
