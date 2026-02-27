#!/usr/bin/env python3
"""
Check experiment status and generate quick summary
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def check_run_status(results_dir: Path):
    """Check status of all runs"""
    print("\n" + "="*60)
    print("Phase 1 Experiment Status")
    print("="*60)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    runs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

    if not runs:
        print("No runs found yet.")
        return

    print(f"\nFound {len(runs)} run(s):\n")

    for run_dir in runs:
        print(f"Run: {run_dir.name}")

        # Check for results
        result_files = list(run_dir.glob("*_results.json"))
        analysis_file = run_dir / "analysis.json"
        report_file = run_dir / "report.md"

        if result_files:
            print(f"  Completed strategies: {len(result_files)}")
            for rf in result_files:
                strategy = rf.stem.replace("_results", "").upper()
                try:
                    with open(rf) as f:
                        data = json.load(f)
                    print(f"    - {strategy}: reward={data.get('best_reward', 0):.3f}, "
                          f"valid={data.get('validity_rate', 0):.1%}")
                except:
                    print(f"    - {strategy}: (error reading)")

        if analysis_file.exists():
            with open(analysis_file) as f:
                analysis = json.load(f)
            print(f"  Winner: {analysis.get('winner', 'N/A')}")

        if report_file.exists():
            print(f"  Report: {report_file}")

        print()


def quick_compare(results_dir: Path):
    """Quick comparison of all runs"""
    print("\n" + "="*60)
    print("Quick Comparison (Best Reward)")
    print("="*60)

    all_data = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        analysis_file = run_dir / "analysis.json"
        if analysis_file.exists():
            with open(analysis_file) as f:
                analysis = json.load(f)

            for strategy, metrics in analysis.get('metrics_by_strategy', {}).items():
                all_data.append({
                    'run': run_dir.name,
                    'strategy': strategy,
                    'best_reward': metrics.get('best_reward', 0),
                    'validity_rate': metrics.get('validity_rate', 0),
                })

    if not all_data:
        print("No completed runs found.")
        return

    # Group by strategy
    from collections import defaultdict
    by_strategy = defaultdict(list)
    for d in all_data:
        by_strategy[d['strategy']].append(d['best_reward'])

    print("\nStrategy Performance (Best Reward):")
    print("-" * 60)
    print(f"{'Strategy':<12} {'Mean':<10} {'Max':<10} {'Runs':<10}")
    print("-" * 60)

    for strategy, rewards in sorted(by_strategy.items()):
        import numpy as np
        mean_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        print(f"{strategy:<12} {mean_reward:<10.3f} {max_reward:<10.3f} {len(rewards):<10}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path(__file__).parent / 'results'

    check_run_status(results_dir)
    quick_compare(results_dir)
