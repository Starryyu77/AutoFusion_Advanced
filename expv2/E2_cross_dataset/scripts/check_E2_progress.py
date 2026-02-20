#!/usr/bin/env python3
"""
E2实验进度监控脚本
"""
import json
from pathlib import Path
from datetime import datetime

def check_progress():
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'results'

    print("=" * 70)
    print("E2 Cross-Dataset Experiment Progress")
    print("=" * 70)
    print(f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    datasets = ['mmmu', 'vsr', 'mathvista']
    expected_archs = 13  # 8 NAS + 5 Baseline

    total_completed = 0
    total_expected = len(datasets) * expected_archs

    for dataset in datasets:
        dataset_dir = results_dir / dataset
        if not dataset_dir.exists():
            print(f"[{dataset.upper():12s}] ❌ 未开始")
            continue

        # 统计已完成的架构
        completed = 0
        errors = 0

        for arch_dir in dataset_dir.iterdir():
            if not arch_dir.is_dir():
                continue

            result_file = arch_dir / 'evaluation_results.json'
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                        if 'error' not in data:
                            completed += 1
                        else:
                            errors += 1
                except:
                    pass

        total_completed += completed

        # 显示状态
        if completed >= expected_archs:
            status = "✅ 完成"
        elif completed > 0:
            status = "⏳ 进行中"
        else:
            status = "🔄 等待中"

        progress = completed / expected_archs * 100
        print(f"[{dataset.upper():12s}] {status}: {completed:2d}/{expected_archs} ({progress:.0f}%) {'⚠️ ' + str(errors) + ' errors' if errors > 0 else ''}")

    # 总进度
    print()
    print("-" * 70)
    total_progress = total_completed / total_expected * 100
    print(f"总进度: {total_completed}/{total_expected} ({total_progress:.1f}%)")
    print("=" * 70)

    # 检查日志文件
    print()
    print("日志文件:")
    for log_file in sorted(results_dir.glob('*.log')):
        size = log_file.stat().st_size / 1024  # KB
        print(f"  {log_file.name:40s} ({size:8.1f} KB)")

    print()

if __name__ == '__main__':
    check_progress()
