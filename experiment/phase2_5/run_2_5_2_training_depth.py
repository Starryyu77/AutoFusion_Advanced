#!/usr/bin/env python3
"""
Phase 2.5.2: Training Depth Calibration

Tests which training depth (1/3/5/10 epochs) is most cost-effective.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluators.real_data_evaluator import RealDataFewShotEvaluator

TEST_ARCHITECTURES = {
    'attention': '...',  # Simplified
    'transformer': '...',
    'conv': '...',
    'mlp': '...'
}


def run_experiment(dataset='vsr', depths=[1, 3, 5, 10], num_shots=16, data_dir='./data'):
    """Run training depth calibration experiment."""
    print("=" * 60)
    print("Phase 2.5.2: Training Depth Calibration")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Depths: {depths}")

    results = {}

    for depth in depths:
        print(f"\nTesting {depth} epochs...")
        print("-" * 40)

        config = {
            'dataset': dataset,
            'num_shots': num_shots,
            'train_epochs': depth,
            'batch_size': 4,
            'data_dir': data_dir
        }

        evaluator = RealDataFewShotEvaluator(config)
        scores = []

        for arch_name, arch_code in list(TEST_ARCHITECTURES.items())[:3]:
            start = time.time()
            try:
                result = evaluator.evaluate(arch_code)
                scores.append(result.accuracy)
                print(f"  {arch_name}: Acc={result.accuracy:.4f}, Time={time.time()-start:.1f}s")
            except Exception as e:
                print(f"  {arch_name}: FAIL - {e}")

        results[f"epochs_{depth}"] = {
            'mean': np.mean(scores) if scores else 0.0,
            'std': np.std(scores) if scores else 0.0,
            'time_per_eval': (time.time() - start) / len(scores) if scores else 0
        }

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Depth vs Performance")
    print("=" * 60)
    for depth in depths:
        key = f"epochs_{depth}"
        print(f"{depth:2d} epochs: mean={results[key]['mean']:.4f}, time={results[key]['time_per_eval']:.1f}s")

    # Save
    output_dir = SCRIPT_DIR / 'results' / '2_5_2_training_depth'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return results


if __name__ == '__main__':
    run_experiment()
