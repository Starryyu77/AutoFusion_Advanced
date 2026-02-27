#!/bin/bash
# Phase 0: Scaffold Validation Script
# PPO + CoT end-to-end test

set -e

echo "=========================================="
echo "Phase 0: Scaffold Validation (PPO + CoT)"
echo "=========================================="
echo ""

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Check Python
echo "Step 0: Environment Check"
echo "-------------------------"
python3 --version
echo "✓ Python available"
echo ""

# Run scaffold test
echo "Step 1: Running Scaffold Test"
echo "-----------------------------"

# Create Python script
PYTHON_SCRIPT="$SCRIPT_DIR/run_scaffold.py"

cat > "$PYTHON_SCRIPT" << 'EOF'
import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

# Get script directory
script_dir = Path(__file__).parent
experiment_dir = script_dir.parent
sys.path.insert(0, str(experiment_dir))

from factory import create_experiment_components, create_controller, create_generator, create_evaluator, create_reward
from base import RewardComponents

def main():
    print("Loading configuration...")
    config_path = script_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed = config['experiment']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"✓ Config loaded (seed={seed})")
    print("")

    # Step 1: Create components
    print("Step 1/5: Creating components...")
    try:
        controller = create_controller('ppo', config['controller'])
        print("  ✓ Controller (PPO) created")

        generator = create_generator('cot', None, config['generator'])
        print("  ✓ Generator (CoT) created")

        evaluator = create_evaluator('sandbox', config['evaluator'])
        print("  ✓ Evaluator (Surgical Sandbox) created")

        reward_fn = create_reward(config['reward'])
        print("  ✓ Reward function created")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("")
    print("Step 2/5: Running search loop...")
    print("-" * 50)

    # Search loop
    max_iterations = config['controller']['max_iterations']
    log_interval = config['experiment']['log_interval']

    for iteration in range(max_iterations):
        # 1. Controller proposes architecture
        proposal = controller.propose()
        architecture = proposal['architecture']

        # 2. Generator generates code
        results = generator.generate(architecture, num_samples=1)
        gen_result = results[0]

        if not gen_result.success:
            # Generation failed, assign penalty
            eval_result = type('obj', (object,), {
                'accuracy': 0.0,
                'efficiency': 0.0,
                'compile_success': 0.0,
                'flops': 0.0,
                'params': 0.0,
                'latency': 0.0,
                'to_dict': lambda: {'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0}
            })()
        else:
            # 3. Evaluator evaluates code
            eval_result = evaluator.evaluate(gen_result.code)

        # 4. Calculate reward
        reward_components = reward_fn.calculate(eval_result.to_dict())

        # 5. Update controller
        controller.update(reward_components)
        controller.record_iteration(architecture, reward_components)

        # Log
        if (iteration + 1) % log_interval == 0 or iteration == 0:
            reward_scalar = reward_components.to_scalar(config['reward']['weights'])
            print(f"Iter {iteration+1:3d}/{max_iterations} | "
                  f"Reward: {reward_scalar:.4f} | "
                  f"Acc: {eval_result.accuracy:.3f} | "
                  f"Eff: {eval_result.efficiency:.3f} | "
                  f"Valid: {eval_result.compile_success:.0f}")

        # Check early stop
        if controller.should_stop():
            print(f"Early stop at iteration {iteration+1}")
            break

    print("-" * 50)
    print("")

    # Summary
    print("Step 3/5: Results Summary")
    print("-------------------------")
    stats = controller.get_stats()
    print(f"Total iterations: {stats['iteration']}")
    print(f"Best reward: {stats['best_reward']:.4f}")
    print(f"Best architecture: {stats.get('best_architecture', 'N/A')}")
    print("")

    # Save results
    print("Step 4/5: Saving results...")
    output_dir = script_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # Save checkpoint
    checkpoint_path = output_dir / "checkpoint.pt"
    controller.save_checkpoint(str(checkpoint_path))
    print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    # Save summary
    summary = {
        'config': config,
        'stats': stats,
        'final_reward': stats['best_reward'],
        'best_architecture': stats.get('best_architecture'),
    }

    import json
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✓ Summary saved: {summary_path}")
    print("")

    print("Step 5/5: Validation Complete")
    print("-----------------------------")
    print("✓ All components working correctly")
    print("✓ Pipeline executed successfully")
    print("")
    print("Next steps:")
    print("  1. Check results/: summary.json and checkpoint.pt")
    print("  2. Run Phase 1: Prompt comparison experiments")
    print("  3. Run Phase 2: Controller comparison experiments")

    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

# Run Python script
python3 "$PYTHON_SCRIPT"
exit_code=$?

# Cleanup
rm "$PYTHON_SCRIPT"

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Phase 0 Validation PASSED"
    echo "=========================================="
    echo ""
    echo "Results saved to: $RESULTS_DIR"
    ls -la "$RESULTS_DIR"
else
    echo ""
    echo "=========================================="
    echo "✗ Phase 0 Validation FAILED"
    echo "=========================================="
    exit 1
fi