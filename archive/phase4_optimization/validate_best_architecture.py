"""
Validate Phase 4 Best Architecture
-----------------------------------
Run comprehensive evaluation on the best discovered architecture.

This script:
1. Tests the Bilinear architecture (50% accuracy, 7,936 FLOPs)
2. Runs 5 independent trials for statistical significance
3. Compares against FiLM baseline
4. Generates statistical report

Usage:
    python phase4_optimization/validate_best_architecture.py --gpu 0 --runs 5
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "experiment"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import numpy as np

# Best architecture from Phase 4 discovery_v3
BEST_BILINEAR_CODE = """
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=192):
        super().__init__()
        self.bilinear = nn.Bilinear(vision_dim, language_dim, hidden_dim)
        self.dropout = nn.Dropout(0.18349594814648426)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        fused = self.bilinear(vision_features, language_features)
        fused = self.dropout(fused)
        return self.norm(fused)
"""

# FiLM baseline for comparison
FILM_CODE = """
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=768):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.gamma_proj = nn.Linear(language_dim, hidden_dim)
        self.beta_proj = nn.Linear(language_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        v = self.vision_proj(vision_features)
        gamma = self.gamma_proj(language_features)
        beta = self.beta_proj(language_features)
        fused = gamma * v + beta
        fused = self.dropout(fused)
        return self.norm(fused)
"""


def setup_logging(output_dir: Path):
    """Setup logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def create_evaluator(config: dict):
    """Create evaluator."""
    from evaluator_v2_improved import ImprovedRealDataFewShotEvaluator

    evaluator_config = {
        "dataset": config.get("dataset", "mmmu"),
        "num_shots": config.get("num_shots", 32),
        "train_epochs": config.get("train_epochs", 100),
        "batch_size": config.get("batch_size", 8),
        "backbone": config.get("backbone", "clip-vit-l-14"),
        "data_dir": config.get("data_dir", "./expv2/data"),
        "device": config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        "early_stopping": {
            "enabled": True,
            "patience": 10,
            "min_delta": 0.005,
        },
        "max_training_time": 1800,  # 30 minutes
    }

    return ImprovedRealDataFewShotEvaluator(evaluator_config)


def run_validation(args):
    """Run validation experiment."""
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    logger.info("=" * 70)
    logger.info("Phase 4 Architecture Validation")
    logger.info("=" * 70)
    logger.info(f"Runs per architecture: {args.runs}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Train epochs: {args.train_epochs}")
    logger.info(f"Output: {output_dir}")

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create evaluator
    evaluator = create_evaluator(vars(args))

    # Architectures to validate
    architectures = [
        ("Phase4_Bilinear", BEST_BILINEAR_CODE),
        ("FiLM_Baseline", FILM_CODE),
    ]

    all_results = {}

    for arch_name, arch_code in architectures:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Validating: {arch_name}")
        logger.info(f"{'=' * 70}")

        run_results = []

        for run in range(args.runs):
            logger.info(f"\n--- Run {run + 1}/{args.runs} ---")

            start_time = time.time()

            try:
                # Evaluate
                eval_result = evaluator.evaluate(arch_code)

                elapsed = time.time() - start_time

                result = {
                    "run": run + 1,
                    "accuracy": eval_result.accuracy,
                    "flops": eval_result.flops,
                    "params": eval_result.params,
                    "latency": eval_result.latency,
                    "efficiency": eval_result.efficiency,
                    "compile_success": eval_result.compile_success,
                    "elapsed_time": elapsed,
                    "metadata": eval_result.metadata,
                }

                logger.info(f"  Accuracy: {eval_result.accuracy:.4f}")
                logger.info(f"  FLOPs: {eval_result.flops:,.0f}")
                logger.info(f"  Time: {elapsed:.1f}s")

            except Exception as e:
                logger.error(f"  Evaluation failed: {e}")
                result = {
                    "run": run + 1,
                    "accuracy": 0.0,
                    "error": str(e),
                }

            run_results.append(result)

        all_results[arch_name] = run_results

        # Calculate statistics
        accuracies = [r["accuracy"] for r in run_results if "error" not in r]
        if accuracies:
            logger.info(f"\n{arch_name} Statistics:")
            logger.info(f"  Mean Accuracy: {np.mean(accuracies):.4f}")
            logger.info(f"  Std Accuracy: {np.std(accuracies):.4f}")
            logger.info(f"  Min: {np.min(accuracies):.4f}")
            logger.info(f"  Max: {np.max(accuracies):.4f}")

    # Compare architectures
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPARISON")
    logger.info(f"{'=' * 70}")

    for arch_name, results in all_results.items():
        accuracies = [r["accuracy"] for r in results if "error" not in r]
        if accuracies:
            logger.info(f"{arch_name}:")
            logger.info(
                f"  Accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}"
            )
            logger.info(f"  Runs: {len(accuracies)}/{args.runs} successful")

    # Statistical test (t-test)
    bilinear_accs = [
        r["accuracy"]
        for r in all_results.get("Phase4_Bilinear", [])
        if "error" not in r
    ]
    film_accs = [
        r["accuracy"] for r in all_results.get("FiLM_Baseline", []) if "error" not in r
    ]

    if len(bilinear_accs) >= 2 and len(film_accs) >= 2:
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(bilinear_accs, film_accs)
        logger.info(f"\nStatistical Test (t-test):")
        logger.info(f"  t-statistic: {t_stat:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant at p<0.05: {'YES' if p_value < 0.05 else 'NO'}")

    # Save results
    results_file = output_dir / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 4 Architecture")

    parser.add_argument(
        "--output-dir", type=str, default="./phase4_optimization/results/validation"
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of validation runs")
    parser.add_argument("--dataset", type=str, default="mmmu")
    parser.add_argument("--num-shots", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="./expv2/data")

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results = run_validation(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
