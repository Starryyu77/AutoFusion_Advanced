#!/usr/bin/env python3
"""
Phase 2.5.1: Dataset Selection Experiment

Tests which dataset (MMMU/VSR/MathVista/AI2D) best predicts final performance.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluators.real_data_evaluator import RealDataFewShotEvaluator

# Test architectures
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
        v = self.vision_conv(v.unsqueeze(-1)).squeeze(-1)
        l = self.lang_conv(l.unsqueeze(-1)).squeeze(-1)
        return self.fusion((v + l).unsqueeze(-1)).squeeze(-1)
''',
    'transformer_fusion': '''
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.l_proj = nn.Linear(language_dim, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(hidden_dim, 8, batch_first=True)

    def forward(self, v, l):
        v, l = self.v_proj(v).unsqueeze(1), self.l_proj(l).unsqueeze(1)
        return self.transformer(torch.cat([v, l], 1)).mean(1)
''',
    'mlp_simple': '''
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(vision_dim, hidden_dim), nn.ReLU())
        self.l = nn.Sequential(nn.Linear(language_dim, hidden_dim), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, v, l):
        return self.fusion(torch.cat([self.v(v), self.l(l)], -1))
'''
}


def run_experiment(datasets=['vsr', 'ai2d', 'mathvista'], num_shots=16, train_epochs=5, data_dir='./data'):
    """Run dataset selection experiment."""
    print("=" * 60)
    print("Phase 2.5.1: Dataset Selection Experiment")
    print("=" * 60)

    results = {}

    for dataset_name in datasets:
        print(f"\nTesting: {dataset_name.upper()}")
        print("-" * 40)

        config = {
            'dataset': dataset_name,
            'num_shots': num_shots,
            'train_epochs': train_epochs,
            'batch_size': 4,
            'backbone': 'clip-vit-l-14',
            'data_dir': data_dir
        }

        evaluator = RealDataFewShotEvaluator(config)
        scores = {}

        for arch_name, arch_code in TEST_ARCHITECTURES.items():
            print(f"  {arch_name}...", end=' ', flush=True)
            try:
                result = evaluator.evaluate(arch_code)
                scores[arch_name] = result.accuracy
                print(f"Acc={result.accuracy:.4f}")
            except Exception as e:
                print(f"FAIL: {e}")
                scores[arch_name] = 0.0

        results[dataset_name] = {
            'scores': scores,
            'mean': np.mean(list(scores.values())),
            'std': np.std(list(scores.values()))
        }

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for dataset_name in datasets:
        print(f"{dataset_name}: mean={results[dataset_name]['mean']:.4f}, std={results[dataset_name]['std']:.4f}")

    # Save results
    output_dir = SCRIPT_DIR / 'results' / '2_5_1_dataset_selection'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return results


if __name__ == '__main__':
    run_experiment()
