"""
Phase 4 with LLM: Architecture Search using DeepSeek
----------------------------------------------------
Run architecture search with LLM-generated code (not templates).

Key differences from run_phase4_search.py:
1. Uses FewShotGenerator (LLM) instead of MockGenerator (templates)
2. Generates architecture descriptions for LLM prompt
3. Uses DeepSeek API for code generation

Usage on cluster:
    cd /projects/tianyu016/AutoFusion_Advanced
    python phase4_optimization/run_phase4_llm.py

Requirements:
    export DEEPSEEK_API_KEY="your-api-key"
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add experiment directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "experiment"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np


# Setup logging
def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase4_llm_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def create_evaluator(config: dict):
    """Create evaluator with Phase 4 configuration."""
    from evaluator_v2_improved import ImprovedRealDataFewShotEvaluator

    evaluator_config = {
        "dataset": config.get("dataset", "mmmu"),
        "num_shots": config.get("num_shots", 32),
        "train_epochs": config.get("train_epochs", 10),
        "batch_size": config.get("batch_size", 8),
        "backbone": config.get("backbone", "clip-vit-l-14"),
        "data_dir": config.get("data_dir", "./expv2/data"),
        "device": config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        "early_stopping": {
            "enabled": config.get("early_stopping_enabled", True),
            "patience": config.get("early_stopping_patience", 3),
            "min_delta": config.get("early_stopping_min_delta", 0.005),
        },
        "max_training_time": config.get("max_training_time", 300),
        "eval_every_n_epochs": config.get("eval_every_n_epochs", 1),
    }

    return ImprovedRealDataFewShotEvaluator(evaluator_config)


def create_reward(config: dict):
    """Create constrained reward function."""
    from reward_v2 import ConstrainedReward

    reward_config = {
        "weights": {
            "accuracy": config.get("weight_accuracy", 1.0),
            "efficiency": config.get("weight_efficiency", 1.5),
            "compile_success": config.get("weight_compile_success", 2.0),
            "complexity": config.get("weight_complexity", 0.3),
        },
        "flops_constraint": {
            "enabled": config.get("flops_constraint_enabled", True),
            "max_flops": config.get("max_flops", 10e6),
            "reject_if_exceed": config.get("reject_if_exceed", True),
        },
        "flops_penalty": {
            "type": config.get("penalty_type", "exponential"),
            "scale": config.get("penalty_scale", 20e6),
        },
        "label_smoothing": config.get("label_smoothing", True),
    }

    return ConstrainedReward(reward_config)


def create_llm_generator(config: dict):
    """Create LLM-based code generator using DeepSeek."""
    from generators.fewshot import FewShotGenerator

    generator_config = {
        "model": config.get("model", "deepseek-chat"),
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 4096),
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "num_examples": config.get("num_examples", 3),
    }

    # Check API key
    if not generator_config["api_key"]:
        raise ValueError(
            "DEEPSEEK_API_KEY not set. Please run:\n"
            "  export DEEPSEEK_API_KEY='your-api-key'\n"
            "Or set it in your environment."
        )

    return FewShotGenerator(llm_client=None, config=generator_config)


class ArchitectureDescriptor:
    """
    Convert search space to architecture description for LLM.

    Instead of just sampling fusion_type, we create rich descriptions
    that guide the LLM to generate diverse architectures.
    """

    def __init__(self, config: dict):
        self.config = config
        np.random.seed(config.get("seed", 42))

    def sample(self) -> dict:
        """Sample an architecture description for LLM prompt."""

        # Sample fusion type with efficiency bias
        fusion_types = [
            ("bilinear", 0.15),  # Simple, efficient
            ("film", 0.20),  # Feature-wise modulation
            ("gated", 0.20),  # Gating mechanism
            ("attention", 0.15),  # Cross-attention
            ("mlp", 0.15),  # Simple MLP
            ("hybrid", 0.15),  # Combination
        ]
        fusion_type = np.random.choice(
            [t[0] for t in fusion_types], p=[t[1] for t in fusion_types]
        )

        # Sample hidden dimension (biased towards smaller)
        hidden_dims = [64, 128, 192, 256]
        hidden_weights = [0.3, 0.3, 0.25, 0.15]  # Prefer smaller
        hidden_dim = np.random.choice(hidden_dims, p=hidden_weights)

        # Sample number of layers (biased towards fewer)
        num_layers = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

        # Sample dropout
        dropout = np.random.uniform(0.0, 0.3)

        # Sample attention heads if needed
        num_heads = np.random.choice([2, 4, 8])

        # Create rich description for LLM
        description = self._create_description(
            fusion_type, hidden_dim, num_layers, dropout, num_heads
        )

        return {
            "type": fusion_type,
            "fusion_type": self._get_fusion_style(fusion_type),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "num_heads": num_heads,
            "description": description,
            "use_residual": True,
        }

    def _create_description(
        self, fusion_type, hidden_dim, num_layers, dropout, num_heads
    ):
        """Create a rich description for LLM."""

        base_desc = f"A {fusion_type} fusion module with {hidden_dim} hidden dimensions"

        if fusion_type == "bilinear":
            return f"{base_desc}. Use bilinear pooling for direct vision-language interaction."
        elif fusion_type == "film":
            return f"{base_desc}. Use Feature-wise Linear Modulation (FiLM) - gamma * vision + beta."
        elif fusion_type == "gated":
            return f"{base_desc}. Use gating mechanism to blend vision and language features."
        elif fusion_type == "attention":
            return f"{base_desc} and {num_heads} attention heads. Use cross-attention between modalities."
        elif fusion_type == "mlp":
            return f"{base_desc} and {num_layers} layers. Simple concat + MLP fusion."
        elif fusion_type == "hybrid":
            return f"{base_desc}. Combine multiple fusion strategies creatively."
        else:
            return base_desc

    def _get_fusion_style(self, fusion_type):
        """Get fusion style description."""
        styles = {
            "bilinear": "efficient",
            "film": "modulation",
            "gated": "adaptive",
            "attention": "cross-modal",
            "mlp": "simple",
            "hybrid": "creative",
        }
        return styles.get(fusion_type, "standard")


def run_llm_architecture_search(args):
    """Run the main architecture search loop with LLM generation."""
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "logs", args.log_level)
    logger.info("=" * 70)
    logger.info("Phase 4 with LLM: Architecture Search")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Check API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        logger.error("DEEPSEEK_API_KEY not set!")
        logger.error("Please run: export DEEPSEEK_API_KEY='your-api-key'")
        return None

    # Create components
    logger.info("Creating components...")

    evaluator = create_evaluator(vars(args))
    logger.info(f"Evaluator: {evaluator.__class__.__name__}")
    logger.info(f"  Dataset: {evaluator.dataset_name}")
    logger.info(f"  Epochs: {evaluator.train_epochs}")
    logger.info(f"  Max time: {evaluator.max_training_time}s")

    reward_fn = create_reward(vars(args))
    logger.info(f"Reward: {reward_fn.__class__.__name__}")
    logger.info(f"  Efficiency weight: {reward_fn.weights['efficiency']}")
    logger.info(f"  Max FLOPs: {reward_fn.max_flops / 1e6:.1f}M")

    descriptor = ArchitectureDescriptor(vars(args))
    logger.info(f"Descriptor: ArchitectureDescriptor (LLM-guided)")

    generator = create_llm_generator(vars(args))
    logger.info(f"Generator: {generator.__class__.__name__}")
    logger.info(f"  Model: {generator.model}")
    logger.info(f"  Temperature: {generator.temperature}")

    # Track results
    all_results = []
    rejected_count = 0
    compile_fail_count = 0
    llm_fail_count = 0

    # Search loop
    logger.info(f"\nStarting search for {args.num_iterations} iterations...")
    logger.info("=" * 70)

    for iteration in range(args.num_iterations):
        iter_start = time.time()
        logger.info(f"\nIteration {iteration + 1}/{args.num_iterations}")
        logger.info("-" * 70)

        # Sample architecture description
        arch_desc = descriptor.sample()
        logger.info(f"Architecture description:")
        logger.info(f"  Type: {arch_desc['type']}")
        logger.info(f"  Hidden dim: {arch_desc['hidden_dim']}")
        logger.info(f"  Layers: {arch_desc['num_layers']}")
        logger.info(f"  Description: {arch_desc['description']}")

        # Generate code using LLM
        logger.info(f"Generating code with LLM...")
        try:
            generation_results = generator.generate(arch_desc, num_samples=1)
            if not generation_results or not generation_results[0].success:
                logger.error(
                    f"LLM generation failed: {generation_results[0].error if generation_results else 'No results'}"
                )
                llm_fail_count += 1
                continue

            generated_code = generation_results[0].code
            logger.info(f"Generated code:\n{generated_code[:500]}...")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            llm_fail_count += 1
            continue

        # Evaluate
        logger.info(f"Evaluating architecture...")
        try:
            eval_result = evaluator.evaluate(generated_code)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            compile_fail_count += 1
            continue

        # Calculate reward
        reward_components = reward_fn.calculate(
            {
                "accuracy": eval_result.accuracy,
                "flops": eval_result.flops,
                "params": eval_result.params,
                "compile_success": eval_result.compile_success,
            }
        )

        scalar_reward = reward_components.to_scalar(reward_fn.weights)

        # Check if rejected
        if hasattr(reward_components, "rejected") and reward_components.rejected:
            rejected_count += 1
            logger.warning(
                f"Architecture REJECTED: {reward_components.rejection_reason}"
            )

        # Log results
        iter_time = time.time() - iter_start
        logger.info(f"Results:")
        logger.info(f"  Accuracy: {eval_result.accuracy:.4f}")
        logger.info(f"  FLOPs: {eval_result.flops / 1e6:.2f}M")
        logger.info(f"  Params: {eval_result.params / 1e6:.2f}M")
        logger.info(f"  Latency: {eval_result.latency:.2f}ms")
        logger.info(f"  Efficiency: {eval_result.efficiency:.4f}")
        logger.info(f"  Reward: {scalar_reward:.4f}")
        logger.info(f"  Time: {iter_time:.1f}s")

        if hasattr(reward_components, "rejected"):
            logger.info(f"  Rejected: {reward_components.rejected}")

        # Store result
        result_record = {
            "iteration": iteration,
            "architecture": arch_desc,
            "code": generated_code,
            "accuracy": eval_result.accuracy,
            "flops": eval_result.flops,
            "params": eval_result.params,
            "latency": eval_result.latency,
            "efficiency": eval_result.efficiency,
            "reward": scalar_reward,
            "rejected": getattr(reward_components, "rejected", False),
            "rejection_reason": getattr(reward_components, "rejection_reason", ""),
            "eval_time": iter_time,
            "metadata": eval_result.metadata,
        }
        all_results.append(result_record)

        # Periodic saving
        if (iteration + 1) % args.save_interval == 0:
            save_results(output_dir, all_results, iteration + 1)

        # Milestone evaluation
        if (iteration + 1) in [10, 20, 30, 50, 100]:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"MILESTONE: {iteration + 1} iterations completed")
            logger.info(f"{'=' * 70}")
            log_milestone_stats(all_results, logger)

    # Final save
    logger.info("\n" + "=" * 70)
    logger.info("Search completed!")
    logger.info("=" * 70)

    save_results(output_dir, all_results, args.num_iterations, final=True)
    log_final_stats(all_results, logger)

    return all_results


def save_results(output_dir: Path, results: list, iteration: int, final: bool = False):
    """Save results to disk."""
    suffix = "final" if final else f"iter_{iteration}"
    results_file = output_dir / f"results_{suffix}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save top architectures
    if len(results) > 0:
        valid_results = [r for r in results if not r.get("rejected", False)]
        if valid_results:
            top_results = sorted(
                valid_results, key=lambda x: x["reward"], reverse=True
            )[:20]
            top_file = output_dir / f"top_20_{suffix}.json"
            with open(top_file, "w") as f:
                json.dump(top_results, f, indent=2, default=str)


def log_milestone_stats(results: list, logger):
    """Log statistics at milestones."""
    valid_results = [r for r in results if not r.get("rejected", False)]
    rejected = len(results) - len(valid_results)

    if not valid_results:
        logger.info("No valid results yet")
        return

    rewards = [r["reward"] for r in valid_results]
    accuracies = [r["accuracy"] for r in valid_results]
    flops = [r["flops"] / 1e6 for r in valid_results]

    logger.info(f"Statistics ({len(valid_results)} valid, {rejected} rejected):")
    logger.info(
        f"  Reward: best={max(rewards):.3f}, mean={sum(rewards) / len(rewards):.3f}"
    )
    logger.info(
        f"  Accuracy: best={max(accuracies):.3f}, mean={sum(accuracies) / len(accuracies):.3f}"
    )
    logger.info(
        f"  FLOPs: min={min(flops):.2f}M, mean={sum(flops) / len(flops):.2f}M, max={max(flops):.2f}M"
    )

    # Top 5
    top_5 = sorted(valid_results, key=lambda x: x["reward"], reverse=True)[:5]
    logger.info(f"  Top 5 architectures:")
    for i, r in enumerate(top_5, 1):
        logger.info(
            f"    {i}. Reward={r['reward']:.3f}, Acc={r['accuracy']:.3f}, FLOPs={r['flops'] / 1e6:.2f}M"
        )


def log_final_stats(results: list, logger):
    """Log final statistics."""
    valid_results = [r for r in results if not r.get("rejected", False)]
    rejected = len(results) - len(valid_results)

    logger.info(f"\nFinal Statistics:")
    logger.info(f"  Total evaluated: {len(results)}")
    logger.info(f"  Valid: {len(valid_results)}")
    logger.info(f"  Rejected: {rejected} ({rejected / len(results) * 100:.1f}%)")

    if valid_results:
        rewards = [r["reward"] for r in valid_results]
        accuracies = [r["accuracy"] for r in valid_results]
        flops = [r["flops"] / 1e6 for r in valid_results]

        logger.info(f"\nReward Statistics:")
        logger.info(f"  Best: {max(rewards):.3f}")
        logger.info(f"  Mean: {sum(rewards) / len(rewards):.3f}")
        logger.info(f"  Std: {np.std(rewards):.3f}")

        logger.info(f"\nAccuracy Statistics:")
        logger.info(f"  Best: {max(accuracies):.3f}")
        logger.info(f"  Mean: {sum(accuracies) / len(accuracies):.3f}")

        logger.info(f"\nFLOPs Statistics:")
        logger.info(f"  Min: {min(flops):.2f}M")
        logger.info(f"  Mean: {sum(flops) / len(flops):.2f}M")
        logger.info(f"  Max: {max(flops):.2f}M")

        # Check against FiLM target
        film_beaters = [
            r for r in valid_results if r["accuracy"] >= 0.46 and r["flops"] <= 6.29e6
        ]
        logger.info(f"\nvs FiLM Baseline (46% acc, 6.29M FLOPs):")
        logger.info(f"  Architectures beating FiLM: {len(film_beaters)}")

        if film_beaters:
            best = max(film_beaters, key=lambda x: x["reward"])
            logger.info(
                f"  Best: Acc={best['accuracy']:.3f}, FLOPs={best['flops'] / 1e6:.2f}M"
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 4 Architecture Search with LLM")

    # Experiment settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./phase4_optimization/results_llm",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=50, help="Number of search iterations"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save results every N iterations"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # Evaluator settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmmu",
        choices=["mmmu", "ai2d", "vsr", "mathvista"],
    )
    parser.add_argument("--num-shots", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone", type=str, default="clip-vit-l-14")
    parser.add_argument("--data-dir", type=str, default="./expv2/data")
    parser.add_argument("--early-stopping-enabled", action="store_true", default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.005)
    parser.add_argument(
        "--max-training-time",
        type=int,
        default=300,
        help="Maximum training time per architecture (seconds)",
    )

    # Reward settings
    parser.add_argument("--weight-accuracy", type=float, default=1.0)
    parser.add_argument("--weight-efficiency", type=float, default=1.5)
    parser.add_argument("--weight-compile-success", type=float, default=2.0)
    parser.add_argument("--weight-complexity", type=float, default=0.3)
    parser.add_argument("--flops-constraint-enabled", action="store_true", default=True)
    parser.add_argument(
        "--max-flops",
        type=float,
        default=10e6,
        help="Maximum FLOPs allowed (10M default)",
    )
    parser.add_argument("--reject-if-exceed", action="store_true", default=True)
    parser.add_argument("--penalty-type", type=str, default="exponential")
    parser.add_argument("--penalty-scale", type=float, default=20e6)

    # LLM Generator settings
    parser.add_argument(
        "--model", type=str, default="deepseek-chat", help="LLM model to use"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--num-examples", type=int, default=3, help="Number of few-shot examples"
    )

    args = parser.parse_args()

    # Run search
    results = run_llm_architecture_search(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
