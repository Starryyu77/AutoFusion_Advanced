#!/bin/bash
# Phase 1: Prompt Strategy Comparison
# Fixed: PPO Controller, Variable: Prompt Strategy

set -e

echo "=========================================="
echo "Phase 1: Prompt Strategy Comparison"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Configuration
STRATEGIES=("cot" "fewshot" "critic" "shape" "roleplay")
SEEDS=(42 123 456)

# GPU task assignment (for parallel execution)
# GPU 2: cot (3 seeds), fewshot (3 seeds), critic (2 seeds)
# GPU 3: critic (1 seed), shape (3 seeds), roleplay (3 seeds)

echo "Running experiments..."
echo "Strategies: ${STRATEGIES[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""

# Function to run single experiment
run_experiment() {
    local strategy=$1
    local seed=$2
    local gpu=$3

    local config_file="$SCRIPT_DIR/configs/${strategy}.yaml"
    local output_dir="$RESULTS_DIR/${strategy}_s${seed}"
    local log_file="$LOGS_DIR/${strategy}_s${seed}.log"

    echo "[GPU $gpu] Starting $strategy (seed=$seed)..."

    CUDA_VISIBLE_DEVICES=$gpu python3 << PYTHON_SCRIPT
import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "$PROJECT_ROOT")

from experiment.factory import create_controller, create_generator, create_evaluator, create_reward

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

    # Create components
    controller = create_controller('ppo', config['controller'])
    generator = create_generator('$strategy', None, config['generator'])
    evaluator = create_evaluator('sandbox', config['evaluator'])
    reward_fn = create_reward(config['reward'])

    # Output directory
    output_dir = Path("$output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search loop
    max_iterations = config['controller']['max_iterations']
    log_interval = 10

    print(f"Starting {max_iterations} iterations...")

    for iteration in range(max_iterations):
        # Propose
        proposal = controller.propose()
        architecture = proposal['architecture']

        # Generate
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
        if (iteration + 1) % log_interval == 0:
            reward_scalar = reward_components.to_scalar(config['reward']['weights'])
            print(f"Iter {iteration+1}/{max_iterations} | Reward: {reward_scalar:.4f}")

        if controller.should_stop():
            break

    # Save results
    checkpoint_path = output_dir / "checkpoint.pt"
    controller.save_checkpoint(str(checkpoint_path))

    import json
    summary = {
        'strategy': '$strategy',
        'seed': seed,
        'stats': controller.get_stats(),
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Completed: $strategy (seed={seed})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
PYTHON_SCRIPT

    echo "[GPU $gpu] Finished $strategy (seed=$seed)"
}

# Parallel execution on 2 GPUs
# GPU 2 tasks
run_gpu2() {
    for strategy in cot fewshot; do
        for seed in ${SEEDS[@]}; do
            run_experiment $strategy $seed 2
        done
    done
    # critic with 2 seeds
    run_experiment critic 42 2
    run_experiment critic 123 2
}

# GPU 3 tasks
run_gpu3() {
    # critic remaining seed
    run_experiment critic 456 3
    for strategy in shape roleplay; do
        for seed in ${SEEDS[@]}; do
            run_experiment $strategy $seed 3
        done
    done
}

# Run in parallel
echo "Launching parallel experiments..."
run_gpu2 &
PID_GPU2=$!
run_gpu3 &
PID_GPU3=$!

echo "GPU2 PID: $PID_GPU2, GPU3 PID: $PID_GPU3"
echo ""

# Wait for completion
wait $PID_GPU2 $PID_GPU3

echo ""
echo "=========================================="
echo "Phase 1 Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To analyze results:"
echo "  python3 $PROJECT_ROOT/scripts/analyze_prompt_results.py --input $RESULTS_DIR"
