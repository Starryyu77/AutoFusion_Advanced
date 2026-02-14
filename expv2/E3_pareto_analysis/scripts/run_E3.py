#!/usr/bin/env python3
"""
E3: 效率-性能帕累托分析

目标: 展示NAS架构的多样性优势
可视化: Accuracy vs FLOPs/Params/Latency 帕累托前沿

使用方法:
    python run_E3.py --input-dir ../E1_main_evaluation/results/full_3runs
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent


def load_results(result_dir):
    """加载E1实验结果"""
    results = {}
    result_path = Path(result_dir)

    for arch_dir in result_path.iterdir():
        if arch_dir.is_dir():
            result_file = arch_dir / 'evaluation_results.json'
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    results[arch_dir.name] = data

    return results


def compute_pareto_frontier(points, objectives=['max', 'min']):
    """计算帕累托前沿"""
    pareto_indices = []

    for i, point in enumerate(points):
        is_pareto = True
        for j, other in enumerate(points):
            if i == j:
                continue
            dominated = True
            for p, o, obj in zip(point, other, objectives):
                if obj == 'max':
                    if p > o:
                        dominated = False
                        break
                else:
                    if p < o:
                        dominated = False
                        break
            if dominated and point != other:
                is_pareto = False
                break
        if is_pareto:
            pareto_indices.append(i)

    return pareto_indices


def analyze_pareto(results):
    """分析帕累托前沿"""
    print("=" * 70)
    print("E3: 帕累托前沿分析")
    print("=" * 70)

    # 提取数据
    data = []
    for arch_name, result in results.items():
        if 'runs' in result and len(result['runs']) > 0:
            run = result['runs'][0]
            acc = run.get('test_acc', 0)
            flops = run.get('flops', 0)
            params = run.get('params', 0)
            latency = run.get('latency_ms', 0)
            data.append({
                'name': arch_name,
                'accuracy': acc,
                'flops': flops,
                'params': params,
                'latency': latency
            })

    if not data:
        print("没有有效数据")
        return

    # Accuracy vs FLOPs
    points_flops = [(d['accuracy'], d['flops']) for d in data]
    pareto_flops = compute_pareto_frontier(points_flops, ['max', 'min'])

    # Accuracy vs Params
    points_params = [(d['accuracy'], d['params']) for d in data]
    pareto_params = compute_pareto_frontier(points_params, ['max', 'min'])

    # Accuracy vs Latency
    points_latency = [(d['accuracy'], d['latency']) for d in data]
    pareto_latency = compute_pareto_frontier(points_latency, ['max', 'min'])

    print(f"\n总架构数: {len(data)}")
    print(f"帕累托前沿 (FLOPs): {len(pareto_flops)} 个架构")
    print(f"帕累托前沿 (Params): {len(pareto_params)} 个架构")
    print(f"帕累托前沿 (Latency): {len(pareto_latency)} 个架构")

    print("\n帕累托前沿架构 (FLOPs):")
    for idx in sorted(pareto_flops, key=lambda i: -data[i]['accuracy']):
        d = data[idx]
        print(f"  {d['name']}: Acc={d['accuracy']:.4f}, FLOPs={d['flops']:.2f}M")

    # 保存结果
    output_dir = SCRIPT_DIR.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    pareto_result = {
        'total_architectures': len(data),
        'pareto_flops': [data[i]['name'] for i in pareto_flops],
        'pareto_params': [data[i]['name'] for i in pareto_params],
        'pareto_latency': [data[i]['name'] for i in pareto_latency],
        'all_data': data
    }

    with open(output_dir / 'pareto_analysis.json', 'w') as f:
        json.dump(pareto_result, f, indent=2)

    print(f"\n结果保存至: {output_dir / 'pareto_analysis.json'}")

    return pareto_result


def main():
    parser = argparse.ArgumentParser(description='E3: 帕累托分析')
    parser.add_argument('--input-dir', type=str,
                        default='../E1_main_evaluation/results/full_3runs',
                        help='E1实验结果目录')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 结果目录不存在 {input_dir}")
        print("请先运行E1实验")
        return

    results = load_results(input_dir)
    analyze_pareto(results)


if __name__ == '__main__':
    main()
