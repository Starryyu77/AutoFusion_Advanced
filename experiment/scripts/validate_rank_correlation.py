#!/usr/bin/env python3
"""
Rank Correlation Validation Script
----------------------------------
Phase 0 mandatory check: Validate that proxy evaluation correlates with full evaluation.

Usage:
    python validate_rank_correlation.py --config path/to/config.yaml --num-architectures 10
"""

import sys
import os
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment.factory import create_evaluator
from utils.rank_correlation import validate_rank_correlation


def generate_mock_architectures(num_architectures: int) -> list:
    """Generate mock architectures for validation"""
    arch_types = ['attention', 'conv', 'mlp', 'transformer', 'hybrid']
    fusion_types = ['early', 'late', 'middle', 'hierarchical']

    architectures = []
    for i in range(num_architectures):
        arch = {
            'type': np.random.choice(arch_types),
            'fusion_type': np.random.choice(fusion_types),
            'hidden_dim': int(np.random.choice([256, 512, 768, 1024])),
            'num_layers': int(np.random.randint(2, 8)),
            'dropout': float(np.random.uniform(0.0, 0.5)),
            'activation': np.random.choice(['gelu', 'relu', 'silu']),
        }
        architectures.append(arch)

    return architectures


def main():
    parser = argparse.ArgumentParser(description='Validate rank correlation between proxy and full evaluation')
    parser.add_argument('--config', type=str, default='phase0_scaffold/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num-architectures', type=int, default=10,
                        help='Number of architectures to evaluate')
    parser.add_argument('--proxy-epochs', type=int, default=5,
                        help='Number of epochs for proxy evaluation')
    parser.add_argument('--full-epochs', type=int, default=20,
                        help='Number of epochs for full evaluation')
    parser.add_argument('--output', type=str, default='results/rank_correlation',
                        help='Output directory for results')
    args = parser.parse_args()

    print("=" * 60)
    print("Rank Correlation Validation (Phase 0 Mandatory Check)")
    print("=" * 60)
    print()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration: {config_path}")
    print(f"Proxy epochs: {args.proxy_epochs}")
    print(f"Full epochs: {args.full_epochs}")
    print(f"Num architectures: {args.num_architectures}")
    print()

    # Create evaluator
    evaluator = create_evaluator('sandbox', config.get('evaluator', {}))

    # Generate mock architectures
    print("Generating architectures...")
    architectures = generate_mock_architectures(args.num_architectures)
    print(f"Generated {len(architectures)} architectures")
    print()

    # Validate rank correlation
    result = validate_rank_correlation(
        architectures=architectures,
        evaluator=evaluator,
        proxy_epochs=args.proxy_epochs,
        full_epochs=args.full_epochs,
        num_samples=args.num_architectures,
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    result_path = output_dir / "validation_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print()
    print(f"Results saved to: {result_path}")

    # Exit code based on recommendation
    if result['recommendation'] == 'continue':
        return 0
    elif result['recommendation'] == 'increase_epochs':
        return 1  # Warning
    else:
        return 2  # Error


if __name__ == '__main__':
    sys.exit(main())
