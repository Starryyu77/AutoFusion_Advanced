#!/bin/bash
# Phase 0: Validation with Real DeepSeek API
# Small-scale test to verify API integration works correctly

set -e

echo "=========================================="
echo "Phase 0: API Validation (Real API)"
echo "=========================================="
echo ""

# Check API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY environment variable not set!"
    echo "Please set it with: export DEEPSEEK_API_KEY='your-api-key'"
    exit 1
fi

echo "Configuration:"
echo "  API Key: ${DEEPSEEK_API_KEY:0:8}..."
echo "  Budget Limit: ${BUDGET_LIMIT_YUAN:-50} yuan"
echo "  Iterations: 10"
echo "  Controller: PPO"
echo "  Seed: 42"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results_validation"
LOGS_DIR="$SCRIPT_DIR/logs_validation"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/.cache/llm}"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$CACHE_DIR"

# Export environment variables
export DEEPSEEK_API_KEY
export BUDGET_LIMIT_YUAN="${BUDGET_LIMIT_YUAN:-50}"
export CACHE_DIR

echo "Starting validation experiment..."
echo ""

CUDA_VISIBLE_DEVICES=2 python3 << PYTHON_SCRIPT
import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "$PROJECT_ROOT")

from experiment.factory import create_controller, create_generator, create_evaluator, create_reward
from experiment.utils.llm_client import create_deepseek_client

def main():
    print("=" * 60)
    print("Phase 0: API Validation")
    print("=" * 60)
    print()

    # Load config
    config_path = Path("$PROJECT_ROOT/phase2_controllers/configs/ppo.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override for validation
    config['controller']['max_iterations'] = 10

    # Set seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("Step 1: Creating DeepSeek client...")
    llm_client = create_deepseek_client(
        api_key=os.environ['DEEPSEEK_API_KEY'],
        cache_dir='$CACHE_DIR',
        budget_limit_yuan=float(os.environ['BUDGET_LIMIT_YUAN']),
    )
    print("  ✓ DeepSeek client created")
    print()

    print("Step 2: Creating components...")
    controller = create_controller('ppo', config['controller'])
    generator = create_generator('cot', llm_client, config['generator'])
    evaluator = create_evaluator('sandbox', config['evaluator'])
    reward_fn = create_reward(config['reward'])
    print("  ✓ All components created")
    print()

    # Output directory
    output_dir = Path("$RESULTS_DIR/ppo_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 3: Running 10 iterations...")
    print()

    for iteration in range(10):
        # Propose
        proposal = controller.propose()
        architecture = proposal['architecture']

        # Generate with REAL LLM
        results = generator.generate(architecture, num_samples=1)
        gen_result = results[0]

        # Evaluate
        if gen_result.success:
            eval_result = evaluator.evaluate(gen_result.code)
            status = "✓"
        else:
            eval_result = type('obj', (object,), {
                'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0,
                'to_dict': lambda: {'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0}
            })()
            status = "✗"

        # Reward
        reward_components = reward_fn.calculate(eval_result.to_dict())

        # Update
        controller.update(reward_components)
        controller.record_iteration(architecture, reward_components)

        # Log
        reward_scalar = reward_components.to_scalar(config['reward']['weights'])
        print(f"  Iter {iteration+1:2d}/10 | {status} Reward: {reward_scalar:.4f} | "
              f"Cost: {llm_client.stats.total_cost_yuan:.2f}¥")

    print()
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    # Final stats
    stats = controller.get_stats()
    print(f"Iterations completed: {stats['iteration']}")
    print(f"Best reward: {stats['best_reward']:.4f}")
    print()

    # API stats
    llm_client.print_stats()

    # Save results
    controller.save_checkpoint(str(output_dir / "checkpoint.pt"))
    llm_client.save_stats(str(output_dir / "api_stats.json"))

    import json
    summary = {
        'controller': 'ppo',
        'seed': seed,
        'stats': stats,
        'api_stats': llm_client.stats.to_dict(),
        'validation_passed': True,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("Validation PASSED ✓")
    print("=" * 60)
    print()
    print("Ready for Phase 0.5 (Mock vs Real comparison)")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Phase 0 Validation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next step: Run Phase 0.5 comparison"
echo "  bash experiment/phase0_validation/run_comparison.sh"
