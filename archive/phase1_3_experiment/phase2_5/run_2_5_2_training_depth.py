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
    'attention_simple': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        attn_out, _ = self.attention(v.unsqueeze(1), l.unsqueeze(1), l.unsqueeze(1))
        return self.norm(attn_out.squeeze(1))
''',
    'conv_fusion': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.vision_conv = nn.Conv1d(vision_dim, hidden_dim, kernel_size=1)
        self.lang_conv = nn.Conv1d(language_dim, hidden_dim, kernel_size=1)
        self.fusion = nn.Sequential(nn.ReLU(), nn.Conv1d(hidden_dim, hidden_dim, 1))

    def forward(self, v, l):
        import torch
        v = self.vision_conv(v.unsqueeze(-1)).squeeze(-1)
        l = self.lang_conv(l.unsqueeze(-1)).squeeze(-1)
        return self.fusion((v + l).unsqueeze(-1)).squeeze(-1)
''',
    'transformer_fusion': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.l_proj = nn.Linear(language_dim, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(hidden_dim, 8, batch_first=True)

    def forward(self, v, l):
        import torch
        v, l = self.v_proj(v).unsqueeze(1), self.l_proj(l).unsqueeze(1)
        return self.transformer(torch.cat([v, l], 1)).mean(1)
''',
    'mlp_simple': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(vision_dim, hidden_dim), nn.ReLU())
        self.l = nn.Sequential(nn.Linear(language_dim, hidden_dim), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, v, l):
        import torch
        return self.fusion(torch.cat([self.v(v), self.l(l)], -1))
'''
}


def run_experiment(dataset='ai2d', depths=[1, 3, 5, 10], num_shots=16, data_dir='./data'):
    """Run training depth calibration experiment."""
    print("=" * 60)
    print("Phase 2.5.2: Training Depth Calibration")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Depths: {depths}")
    print()

    results = {}

    for depth in depths:
        print(f"\nTesting {depth} epochs...")
        print("-" * 40)

        config = {
            'dataset': dataset,
            'num_shots': num_shots,
            'train_epochs': depth,
            'batch_size': 4,
            'backbone': 'clip-vit-l-14',
            'data_dir': data_dir
        }

        evaluator = RealDataFewShotEvaluator(config)
        scores = []
        times = []

        for arch_name, arch_code in list(TEST_ARCHITECTURES.items())[:3]:
            start = time.time()
            try:
                result = evaluator.evaluate(arch_code)
                scores.append(result.accuracy)
                eval_time = time.time() - start
                times.append(eval_time)
                print(f"  {arch_name}: Acc={result.accuracy:.4f}, Time={eval_time:.1f}s")
            except Exception as e:
                print(f"  {arch_name}: FAIL - {e}")

        results[f"epochs_{depth}"] = {
            'mean': np.mean(scores) if scores else 0.0,
            'std': np.std(scores) if scores else 0.0,
            'time_per_eval': np.mean(times) if times else 0
        }

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Depth vs Performance")
    print("=" * 60)
    print(f"{'Epochs':<10} {'Mean Acc':<12} {'Std':<10} {'Time (s)':<10}")
    print("-" * 60)
    for depth in depths:
        key = f"epochs_{depth}"
        r = results[key]
        print(f"{depth:<10} {r['mean']:<12.4f} {r['std']:<10.4f} {r['time_per_eval']:<10.1f}")

    # Find optimal
    best_depth = max(depths, key=lambda d: results[f"epochs_{d}"]['mean'])
    print("\n" + "=" * 60)
    print(f"RECOMMENDATION: {best_depth} epochs")
    print(f"  Mean Accuracy: {results[f'epochs_{best_depth}']['mean']:.4f}")
    print(f"  Time per eval: {results[f'epochs_{best_depth}']['time_per_eval']:.1f}s")
    print("=" * 60)

    # Save
    output_dir = SCRIPT_DIR / 'results' / '2_5_2_training_depth'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Phase 2.5.2: Training Depth Calibration - Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Tested depths: {depths}\n\n")
        f.write("Results:\n")
        for depth in depths:
            key = f"epochs_{depth}"
            r = results[key]
            f.write(f"  {depth} epochs: acc={r['mean']:.4f}, time={r['time_per_eval']:.1f}s\n")
        f.write(f"\nRecommended: {best_depth} epochs\n")

    print(f"\nResults saved to: {output_dir}")
    return results


if __name__ == '__main__':
    run_experiment()
