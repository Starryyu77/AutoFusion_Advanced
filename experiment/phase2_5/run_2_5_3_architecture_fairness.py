#!/usr/bin/env python3
"""
Phase 2.5.3: Architecture Fairness Testing

Tests if evaluator treats all architecture types fairly.
Uses the optimal config from Phase 2.5.1 and 2.5.2:
- Dataset: AI2D
- Train epochs: 3
- Num shots: 16
"""

import sys
import json
import time
import torch
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluators.real_data_evaluator import RealDataFewShotEvaluator

# Define architecture variants for each type
ARCHITECTURE_TYPES = {
    'attention_based': {
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
        'attention_cross': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.l_proj = nn.Linear(language_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, v, l):
        v, l = self.v_proj(v), self.l_proj(l)
        cross_out, _ = self.cross_attn(v.unsqueeze(1), l.unsqueeze(1), l.unsqueeze(1))
        cross_out = self.norm1(cross_out)
        self_out, _ = self.self_attn(cross_out, cross_out, cross_out)
        return self.norm2(self_out.squeeze(1))
'''
    },
    'conv_based': {
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
        'conv_depthwise': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.lang_proj = nn.Linear(language_dim, hidden_dim)
        self.depth_conv = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=2)
        self.point_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, v, l):
        v, l = self.vision_proj(v), self.lang_proj(l)
        concat = torch.cat([v, l], dim=-1).unsqueeze(-1)
        out = self.depth_conv(concat)
        out = self.point_conv(out)
        return self.norm(out).squeeze(-1)
'''
    },
    'transformer_based': {
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
        v, l = self.v_proj(v).unsqueeze(1), self.l_proj(l).unsqueeze(1)
        return self.transformer(torch.cat([v, l], 1)).mean(1)
''',
        'transformer_cross': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.l_proj = nn.Linear(language_dim, hidden_dim)
        self.cross_transformer = nn.TransformerDecoderLayer(hidden_dim, 8, batch_first=True)
        self.encoder = nn.TransformerEncoderLayer(hidden_dim, 8, batch_first=True)

    def forward(self, v, l):
        v, l = self.v_proj(v).unsqueeze(1), self.l_proj(l).unsqueeze(1)
        cross_out = self.cross_transformer(v, l)
        return self.encoder(cross_out).squeeze(1)
'''
    },
    'mlp_based': {
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
        return self.fusion(torch.cat([self.v(v), self.l(l)], -1))
''',
        'mlp_deep': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.l = nn.Sequential(
            nn.Linear(language_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )

    def forward(self, v, l):
        return self.fusion(torch.cat([self.v(v), self.l(l)], -1))
'''
    },
    'hybrid': {
        'hybrid_attn_conv': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.l_proj = nn.Linear(language_dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, v, l):
        v, l = self.v_proj(v), self.l_proj(l)
        conv_out = self.conv((v + l).unsqueeze(-1)).squeeze(-1)
        attn_out, _ = self.attention(conv_out.unsqueeze(1), conv_out.unsqueeze(1), conv_out.unsqueeze(1))
        return self.norm(attn_out.squeeze(1) + conv_out)
''',
        'hybrid_transformer_mlp': '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=512):
        super().__init__()
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.l_proj = nn.Linear(language_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.transformer = nn.TransformerEncoderLayer(hidden_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, v, l):
        v, l = self.v_proj(v), self.l_proj(l)
        mlp_out = self.mlp(torch.cat([v, l], dim=-1))
        trans_out = self.transformer(mlp_out.unsqueeze(1)).squeeze(1)
        return self.norm(mlp_out + trans_out)
'''
    }
}


def run_experiment(dataset='ai2d', num_shots=16, train_epochs=3, data_dir='./data', seeds=[42, 123, 456]):
    """
    Run architecture fairness experiment.

    Tests multiple architecture types with different seeds to verify:
    1. Fairness: All types get reasonable scores
    2. Stability: Low variance across seeds
    3. Discriminative power: Can distinguish good/bad architectures
    """
    print("=" * 60)
    print("Phase 2.5.3: Architecture Fairness Testing")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Epochs: {train_epochs}")
    print(f"Shots: {num_shots}")
    print(f"Seeds: {seeds}")
    print()

    # Set torch seed for reproducibility
    torch.manual_seed(42)

    results = {}
    all_scores = []

    for arch_type, arch_variants in ARCHITECTURE_TYPES.items():
        print(f"\n{'='*60}")
        print(f"Testing {arch_type.upper()} ({len(arch_variants)} variants)")
        print('='*60)

        type_results = {
            'variants': {},
            'type_mean': 0.0,
            'type_std': 0.0,
            'type_variance': 0.0
        }

        for arch_name, arch_code in arch_variants.items():
            print(f"\n  {arch_name}:")
            variant_scores = []
            variant_times = []

            for seed in seeds:
                config = {
                    'dataset': dataset,
                    'num_shots': num_shots,
                    'train_epochs': train_epochs,
                    'batch_size': 4,
                    'backbone': 'clip-vit-l-14',
                    'data_dir': data_dir
                }

                evaluator = RealDataFewShotEvaluator(config)
                start = time.time()

                try:
                    result = evaluator.evaluate(arch_code)
                    score = result.accuracy
                    eval_time = time.time() - start
                    variant_scores.append(score)
                    variant_times.append(eval_time)
                    print(f"    Seed {seed}: Acc={score:.4f}, Time={eval_time:.1f}s")
                except Exception as e:
                    print(f"    Seed {seed}: FAIL - {e}")
                    variant_scores.append(0.0)
                    variant_times.append(0.0)

            type_results['variants'][arch_name] = {
                'scores': variant_scores,
                'mean': np.mean(variant_scores),
                'std': np.std(variant_scores),
                'variance': np.var(variant_scores),
                'mean_time': np.mean(variant_times)
            }
            all_scores.extend(variant_scores)

        # Calculate aggregate stats for this architecture type
        type_means = [v['mean'] for v in type_results['variants'].values()]
        type_results['type_mean'] = np.mean(type_means)
        type_results['type_std'] = np.std(type_means)
        type_results['type_variance'] = np.var(type_means)

        results[arch_type] = type_results

    # Summary
    print("\n" + "=" * 60)
    print("FAIRNESS ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n{'Architecture Type':<20} {'Mean Acc':<12} {'Std':<10} {'Variance':<10}")
    print("-" * 60)

    for arch_type, data in results.items():
        print(f"{arch_type:<20} {data['type_mean']:<12.4f} {data['type_std']:<10.4f} {data['type_variance']:<10.4f}")

    # Fairness metrics
    print("\n" + "=" * 60)
    print("FAIRNESS METRICS")
    print("=" * 60)

    all_type_means = [data['type_mean'] for data in results.values()]
    overall_mean = np.mean(all_type_means)
    overall_std = np.std(all_type_means)
    max_diff = max(all_type_means) - min(all_type_means)

    print(f"\nOverall mean across types: {overall_mean:.4f}")
    print(f"Overall std across types: {overall_std:.4f}")
    print(f"Max difference between types: {max_diff:.4f}")

    # Fairness assessment
    print("\n" + "=" * 60)
    print("FAIRNESS ASSESSMENT")
    print("=" * 60)

    if overall_std < 0.1:
        fairness_rating = "EXCELLENT"
        print("✅ FAIRNESS: EXCELLENT (std < 0.1)")
    elif overall_std < 0.2:
        fairness_rating = "GOOD"
        print("✅ FAIRNESS: GOOD (std < 0.2)")
    elif overall_std < 0.3:
        fairness_rating = "ACCEPTABLE"
        print("⚠️  FAIRNESS: ACCEPTABLE (std < 0.3)")
    else:
        fairness_rating = "POOR"
        print("❌ FAIRNESS: POOR (std >= 0.3)")

    if max_diff < 0.2:
        print("✅ TYPE BALANCE: EXCELLENT (max_diff < 0.2)")
    elif max_diff < 0.3:
        print("✅ TYPE BALANCE: GOOD (max_diff < 0.3)")
    else:
        print("⚠️  TYPE BALANCE: NEEDS ATTENTION")

    # Per-variant analysis
    print("\n" + "=" * 60)
    print("PER-VARIANT DETAILS")
    print("=" * 60)

    for arch_type, data in results.items():
        print(f"\n{arch_type}:")
        for variant_name, variant_data in data['variants'].items():
            stability = "Stable" if variant_data['std'] < 0.1 else "Variable"
            print(f"  {variant_name:20s}: mean={variant_data['mean']:.4f}, "
                  f"std={variant_data['std']:.4f}, {stability}")

    # Save results
    output_dir = SCRIPT_DIR / 'results' / '2_5_3_architecture_fairness'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare serializable results
    save_results = {
        'config': {
            'dataset': dataset,
            'num_shots': num_shots,
            'train_epochs': train_epochs,
            'seeds': seeds
        },
        'fairness_metrics': {
            'overall_mean': float(overall_mean),
            'overall_std': float(overall_std),
            'max_diff': float(max_diff),
            'fairness_rating': fairness_rating
        },
        'architecture_types': results
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Phase 2.5.3: Architecture Fairness Testing - Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Train epochs: {train_epochs}\n")
        f.write(f"Num shots: {num_shots}\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write("Fairness Metrics:\n")
        f.write(f"  Overall mean: {overall_mean:.4f}\n")
        f.write(f"  Overall std: {overall_std:.4f}\n")
        f.write(f"  Max difference: {max_diff:.4f}\n")
        f.write(f"  Fairness rating: {fairness_rating}\n\n")
        f.write("Results by Architecture Type:\n")
        for arch_type, data in results.items():
            f.write(f"  {arch_type}: mean={data['type_mean']:.4f}, std={data['type_std']:.4f}\n")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print('='*60)

    return save_results


if __name__ == '__main__':
    run_experiment()
