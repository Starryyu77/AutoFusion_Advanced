#!/usr/bin/env python3
"""
Phase 4: Evaluate Top Discovered Architectures
------------------------------------------------
Evaluate the best architectures discovered in Phase 4 search
with full training (100 epochs) on MMMU dataset.

Usage:
    python phase4_optimization/evaluate_top_architectures.py --top-k 5 --gpu 0
    python phase4_optimization/evaluate_top_architectures.py --arch-file results/discovery_v3/top_20_final.json --gpu 0
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Add paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "experiment"))
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from evaluator_v2_improved import ImprovedRealDataFewShotEvaluator


def load_top_architectures(results_file: str, top_k: int = 5):
    """Load top architectures from search results."""
    with open(results_file, "r") as f:
        results = json.load(f)

    # Filter valid (non-rejected) results
    valid = [r for r in results if not r.get("rejected", False)]

    # Sort by reward
    sorted_results = sorted(valid, key=lambda x: x["reward"], reverse=True)

    return sorted_results[:top_k]


def create_module_from_code(code: str, module_name: str = "FusionModule"):
    """Dynamically create module from code string."""
    namespace = {"nn": nn, "torch": torch}
    exec(code, namespace)
    return namespace[module_name]


def evaluate_architecture(arch_info: dict, evaluator, device: str = "cuda"):
    """Evaluate a single architecture with full training."""
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {arch_info['architecture']}")
    print(f"Type: {arch_info['architecture']['fusion_type']}")
    print(
        f"Quick eval: Acc={arch_info['accuracy']:.3f}, FLOPs={arch_info['flops'] / 1e6:.2f}M"
    )
    print(f"{'=' * 70}")

    # Create module from code
    try:
        ModuleClass = create_module_from_code(arch_info["code"])
        module = ModuleClass()
        module.to(device)
    except Exception as e:
        print(f"Failed to create module: {e}")
        # Continue anyway - evaluator will handle code string

    # Run full evaluation - pass code string directly
    try:
        result = evaluator.evaluate(arch_info["code"], arch_info["architecture"])
        return {
            "architecture": arch_info["architecture"],
            "quick_eval": {
                "accuracy": arch_info["accuracy"],
                "flops": arch_info["flops"],
                "reward": arch_info["reward"],
            },
            "full_eval": {
                "accuracy": result.accuracy,
                "flops": result.flops,
                "params": result.params,
                "latency": result.latency,
                "epochs_trained": result.metadata.get("epochs_trained", 0),
                "train_accuracy": result.metadata.get("train_accuracy", 0),
                "val_accuracy": result.accuracy,
                "best_val_accuracy": result.metadata.get("best_val_accuracy", 0),
            },
        }
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Top Architectures")
    parser.add_argument(
        "--arch-file", type=str, default=None, help="Path to architecture results JSON"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top architectures to evaluate"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmmu",
        choices=["mmmu", "ai2d", "vsr", "mathvista"],
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=100,
        help="Training epochs for full evaluation",
    )
    parser.add_argument("--num-shots", type=int, default=32)
    parser.add_argument(
        "--output-dir", type=str, default="./phase4_optimization/results/evaluation"
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument(
        "--max-training-time",
        type=int,
        default=1800,
        help="Max training time per architecture (30 min)",
    )

    args = parser.parse_args()

    # Setup
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default architecture file
    if args.arch_file is None:
        args.arch_file = (
            PROJECT_DIR
            / "phase4_optimization"
            / "results"
            / "discovery_v3"
            / "top_20_final.json"
        )

    # Load architectures
    print(f"\nLoading top {args.top_k} architectures from: {args.arch_file}")
    top_archs = load_top_architectures(args.arch_file, args.top_k)
    print(f"Loaded {len(top_archs)} architectures")

    # Create evaluator with full training config
    eval_config = {
        "dataset": args.dataset,
        "num_shots": args.num_shots,
        "train_epochs": args.train_epochs,
        "batch_size": 8,
        "backbone": "clip-vit-l-14",
        "data_dir": str(PROJECT_DIR / "expv2" / "data"),
        "device": device,
        "early_stopping": {
            "enabled": True,
            "patience": args.early_stopping_patience,
            "min_delta": 0.005,
        },
        "max_training_time": args.max_training_time,
        "eval_every_n_epochs": 1,
    }

    evaluator = ImprovedRealDataFewShotEvaluator(eval_config)

    # Evaluate each architecture
    all_results = []
    for i, arch_info in enumerate(top_archs, 1):
        print(
            f"\n[{i}/{len(top_archs)}] Evaluating architecture from iteration {arch_info['iteration']}"
        )

        result = evaluate_architecture(arch_info, evaluator, device)
        if result:
            all_results.append(result)

            # Save intermediate results
            result_file = (
                output_dir / f"arch_{i}_iteration_{arch_info['iteration']}.json"
            )
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Saved to: {result_file}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "train_epochs": args.train_epochs,
            "num_shots": args.num_shots,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "num_architectures_evaluated": len(all_results),
        "results": all_results,
    }

    summary_file = (
        output_dir
        / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print comparison
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(
        f"{'Iter':<6} {'Type':<20} {'Quick Acc':<12} {'Full Acc':<12} {'FLOPs':<12} {'Epochs':<8}"
    )
    print("-" * 70)

    for r in all_results:
        arch = r["architecture"]
        quick_acc = r["quick_eval"]["accuracy"]
        full_acc = r["full_eval"]["accuracy"]
        flops = r["full_eval"]["flops"] / 1e6
        epochs = r["full_eval"]["epochs_trained"]
        print(
            f"{arch.get('iteration', '?'):<6} {arch['fusion_type']:<20} {quick_acc:.3f}       {full_acc:.3f}       {flops:.2f}M       {epochs}"
        )

    print("\n" + "=" * 70)
    print(f"Summary saved to: {summary_file}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
