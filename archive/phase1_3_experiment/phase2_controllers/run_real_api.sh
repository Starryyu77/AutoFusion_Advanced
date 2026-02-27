#!/bin/bash
# Phase 2: Controller Comparison with REAL DeepSeek API
# Fixed: Use actual LLM API instead of mock generation

set -e

echo "=========================================="
echo "Phase 2: Controller Comparison (Real API)"
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
echo "  Budget Limit: ${BUDGET_LIMIT_YUAN:-10000} yuan"
echo "  Cache Dir: ${CACHE_DIR:-.cache/llm}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results_real_api"
LOGS_DIR="$SCRIPT_DIR/logs_real_api"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/.cache/llm}"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$CACHE_DIR"

# Configuration
CONTROLLERS=("ppo" "grpo" "gdpo" "evolution" "cmaes" "random")
SEEDS=(42 123 456 789 1024)

# Budget control
BUDGET_LIMIT_YUAN="${BUDGET_LIMIT_YUAN:-10000}"

echo "Running experiments with REAL API..."
echo "Controllers: ${CONTROLLERS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Results: $RESULTS_DIR"
echo ""

# Function to run single experiment
run_experiment() {
    local controller=$1
    local seed=$2
    local gpu=$3

    local config_file="$SCRIPT_DIR/configs/${controller}.yaml"
    local output_dir="$RESULTS_DIR/${controller}_s${seed}"
    local log_file="$LOGS_DIR/${controller}_s${seed}.log"

    echo "[GPU $gpu] Starting $controller (seed=$seed)..."

    CUDA_VISIBLE_DEVICES=$gpu python3 << PYTHON_SCRIPT
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
    # Load config
    config_path = Path("$config_file")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = $seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create DeepSeek client with budget limit
    budget_limit = float(os.environ.get('BUDGET_LIMIT_YUAN', 10000))
    llm_client = create_deepseek_client(
        api_key=os.environ['DEEPSEEK_API_KEY'],
        cache_dir='$CACHE_DIR',
        budget_limit_yuan=budget_limit,
        model=config['generator'].get('model', 'deepseek-chat'),
        temperature=config['generator'].get('temperature', 0.7),
        max_tokens=config['generator'].get('max_tokens', 4096),
    )

    # Create components
    controller = create_controller('$controller', config['controller'])
    generator = create_generator('cot', llm_client, config['generator'])
    evaluator = create_evaluator('sandbox', config['evaluator'])
    reward_fn = create_reward(config['reward'])

    # Output directory
    output_dir = Path("$output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if exists (for resume)
    checkpoint_path = output_dir / "checkpoint.pt"
    start_iteration = 0
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        controller.load_checkpoint(str(checkpoint_path))
        # Note: You need to implement load_checkpoint in controller

    # Search loop
    max_iterations = config['controller']['max_iterations']
    log_interval = 1  # Log every iteration for real API
    checkpoint_interval = 10  # Save checkpoint every 10 iterations

    print(f"Starting {max_iterations} iterations with REAL API...")
    print(f"Budget limit: {budget_limit} yuan")

    for iteration in range(start_iteration, max_iterations):
        # Propose
        proposal = controller.propose()
        architecture = proposal['architecture']

        # Generate with REAL LLM
        results = generator.generate(architecture, num_samples=1)
        gen_result = results[0]

        # Evaluate
        if gen_result.success:
            eval_result = evaluator.evaluate(gen_result.code)
        else:
            eval_result = type('obj', (object,), {
                'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0,
                'to_dict': lambda: {'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0}
            })()

        # Reward
        reward_components = reward_fn.calculate(eval_result.to_dict())

        # Update
        controller.update(reward_components)
        controller.record_iteration(architecture, reward_components)

        # Log
        reward_scalar = reward_components.to_scalar(config['reward']['weights'])
        print(f"Iter {iteration+1}/{max_iterations} | Reward: {reward_scalar:.4f} | "
              f"API Cost: {llm_client.stats.total_cost_yuan:.2f} yuan")

        # Save checkpoint every N iterations
        if (iteration + 1) % checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_iter{iteration+1}.pt"
            controller.save_checkpoint(str(checkpoint_path))
            # Save API stats
            llm_client.save_stats(str(output_dir / "api_stats.json"))
            print(f"  -> Checkpoint saved")

        # Check budget
        if llm_client.stats.budget_remaining < 0:
            print(f"WARNING: Budget exceeded! Stopping experiment.")
            break

        if controller.should_stop():
            break

    # Save final results
    final_checkpoint = output_dir / "checkpoint.pt"
    controller.save_checkpoint(str(final_checkpoint))

    # Save API stats
    llm_client.print_stats()
    llm_client.save_stats(str(output_dir / "api_stats.json"))

    import json
    summary = {
        'controller': '$controller',
        'seed': seed,
        'stats': controller.get_stats(),
        'api_stats': llm_client.stats.to_dict(),
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Completed: $controller (seed={seed})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
PYTHON_SCRIPT

    echo "[GPU $gpu] Finished $controller (seed=$seed)"
}

# Export environment variables for subprocesses
export DEEPSEEK_API_KEY
export BUDGET_LIMIT_YUAN
export CACHE_DIR

# Function to run experiments on GPU 2
run_gpu2() {
    for controller in ppo grpo evolution; do
        for seed in ${SEEDS[@]}; do
            run_experiment $controller $seed 2
        done
    done
}

# Function to run experiments on GPU 3
run_gpu3() {
    for controller in gdpo cmaes random; do
        for seed in ${SEEDS[@]}; do
            run_experiment $controller $seed 3
        done
    done
}

# Main execution
echo "Starting Real API experiments..."
echo ""

# Check if user wants to run specific experiment
if [ "$1" == "single" ] && [ -n "$2" ] && [ -n "$3" ] && [ -n "$4" ]; then
    # Single experiment mode: run_single.sh single ppo 42 2
    run_experiment $2 $3 $4
else
    # Parallel execution on 2 GPUs
    echo "Launching parallel experiments on GPU 2 and GPU 3..."
    run_gpu2 &
    PID_GPU2=$!
    run_gpu3 &
    PID_GPU3=$!

    echo "GPU2 PID: $PID_GPU2, GPU3 PID: $PID_GPU3"
    echo ""

    # Wait for completion
    wait $PID_GPU2 $PID_GPU3
fi

echo ""
echo "=========================================="
echo "Phase 2 (Real API) Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOGS_DIR"
echo ""
echo "To analyze results:"
echo "  python3 $PROJECT_ROOT/scripts/analyze_controller_results.py --input $RESULTS_DIR"
