#!/bin/bash
# Phase 0.5: Mock vs Real API Comparison
# Compare the same controller with mock and real API generation

set -e

echo "=========================================="
echo "Phase 0.5: Mock vs Real API Comparison"
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
echo "  Budget Limit: ${BUDGET_LIMIT_YUAN:-200} yuan"
echo "  Iterations: 20 per run"
echo "  Controller: PPO"
echo "  Seed: 42"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results_comparison"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/.cache/llm}"

mkdir -p "$RESULTS_DIR"
mkdir -p "$CACHE_DIR"

# Export environment variables
export DEEPSEEK_API_KEY
export BUDGET_LIMIT_YUAN="${BUDGET_LIMIT_YUAN:-200}"
export CACHE_DIR

# Function to run experiment with mock
run_mock() {
    echo "Running MOCK experiment..."
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

def main():
    print("=" * 60)
    print("Mock API Run")
    print("=" * 60)
    print()

    # Load config
    config_path = Path("$PROJECT_ROOT/phase2_controllers/configs/ppo.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['controller']['max_iterations'] = 20

    # Set seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create components (mock - llm_client=None)
    controller = create_controller('ppo', config['controller'])
    generator = create_generator('cot', None, config['generator'])  # Mock!
    evaluator = create_evaluator('sandbox', config['evaluator'])
    reward_fn = create_reward(config['reward'])

    output_dir = Path("$RESULTS_DIR/mock")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running 20 iterations with MOCK generation...")
    print()

    for iteration in range(20):
        proposal = controller.propose()
        architecture = proposal['architecture']

        # Generate with MOCK
        results = generator.generate(architecture, num_samples=1)
        gen_result = results[0]

        if gen_result.success:
            eval_result = evaluator.evaluate(gen_result.code)
            status = "✓"
        else:
            eval_result = type('obj', (object,), {
                'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0,
                'to_dict': lambda: {'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0}
            })()
            status = "✗"

        reward_components = reward_fn.calculate(eval_result.to_dict())
        controller.update(reward_components)
        controller.record_iteration(architecture, reward_components)

        reward_scalar = reward_components.to_scalar(config['reward']['weights'])
        print(f"  Iter {iteration+1:2d}/20 | {status} Reward: {reward_scalar:.4f}")

    print()
    stats = controller.get_stats()
    print(f"Mock Run - Best Reward: {stats['best_reward']:.4f}")
    print()

    # Save results
    controller.save_checkpoint(str(output_dir / "checkpoint.pt"))
    import json
    summary = {
        'type': 'mock',
        'controller': 'ppo',
        'seed': seed,
        'stats': stats,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return 0

if __name__ == "__main__":
    sys.exit(main())
PYTHON_SCRIPT
}

# Function to run experiment with real API
run_real() {
    echo "Running REAL API experiment..."
    echo ""

    CUDA_VISIBLE_DEVICES=3 python3 << PYTHON_SCRIPT
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
    print("Real API Run")
    print("=" * 60)
    print()

    # Load config
    config_path = Path("$PROJECT_ROOT/phase2_controllers/configs/ppo.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['controller']['max_iterations'] = 20

    # Set seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create DeepSeek client
    llm_client = create_deepseek_client(
        api_key=os.environ['DEEPSEEK_API_KEY'],
        cache_dir='$CACHE_DIR',
        budget_limit_yuan=float(os.environ['BUDGET_LIMIT_YUAN']),
    )

    # Create components (real API)
    controller = create_controller('ppo', config['controller'])
    generator = create_generator('cot', llm_client, config['generator'])
    evaluator = create_evaluator('sandbox', config['evaluator'])
    reward_fn = create_reward(config['reward'])

    output_dir = Path("$RESULTS_DIR/real")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running 20 iterations with REAL API generation...")
    print()

    for iteration in range(20):
        proposal = controller.propose()
        architecture = proposal['architecture']

        # Generate with REAL LLM
        results = generator.generate(architecture, num_samples=1)
        gen_result = results[0]

        if gen_result.success:
            eval_result = evaluator.evaluate(gen_result.code)
            status = "✓"
        else:
            eval_result = type('obj', (object,), {
                'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0,
                'to_dict': lambda: {'accuracy': 0.0, 'efficiency': 0.0, 'compile_success': 0.0}
            })()
            status = "✗"

        reward_components = reward_fn.calculate(eval_result.to_dict())
        controller.update(reward_components)
        controller.record_iteration(architecture, reward_components)

        reward_scalar = reward_components.to_scalar(config['reward']['weights'])
        print(f"  Iter {iteration+1:2d}/20 | {status} Reward: {reward_scalar:.4f} | "
              f"Cost: {llm_client.stats.total_cost_yuan:.2f}¥")

    print()
    stats = controller.get_stats()
    print(f"Real API Run - Best Reward: {stats['best_reward']:.4f}")
    print()

    # API stats
    llm_client.print_stats()

    # Save results
    controller.save_checkpoint(str(output_dir / "checkpoint.pt"))
    llm_client.save_stats(str(output_dir / "api_stats.json"))

    import json
    summary = {
        'type': 'real',
        'controller': 'ppo',
        'seed': seed,
        'stats': stats,
        'api_stats': llm_client.stats.to_dict(),
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return 0

if __name__ == "__main__":
    sys.exit(main())
PYTHON_SCRIPT
}

# Function to compare results
compare_results() {
    echo ""
    echo "=========================================="
    echo "Comparison Results"
    echo "=========================================="
    echo ""

    python3 << PYTHON_SCRIPT
import json
from pathlib import Path

results_dir = Path("$RESULTS_DIR")

# Load results
with open(results_dir / "mock" / "summary.json") as f:
    mock_data = json.load(f)

with open(results_dir / "real" / "summary.json") as f:
    real_data = json.load(f)

mock_reward = mock_data['stats']['best_reward']
real_reward = real_data['stats']['best_reward']
diff = real_reward - mock_reward

print("=" * 60)
print("Mock vs Real API Comparison")
print("=" * 60)
print()
print(f"Controller: PPO")
print(f"Seed: 42")
print(f"Iterations: 20")
print()
print("Results:")
print(f"  Mock:      {mock_reward:.4f}")
print(f"  Real:      {real_reward:.4f}")
print(f"  Diff:      {diff:+.4f}")
print(f"  Change:    {(diff/mock_reward)*100:+.1f}%")
print()

# API cost
if 'api_stats' in real_data:
    api_stats = real_data['api_stats']
    print("API Statistics:")
    print(f"  Total Calls:   {api_stats['total_calls']}")
    print(f"  Cache Hits:    {api_stats['cache_hits']}")
    print(f"  Total Cost:    {api_stats['total_cost_yuan']:.2f} yuan")
    print()

print("=" * 60)

# Analysis
if real_reward > mock_reward * 1.1:
    print("Conclusion: Real API significantly outperforms Mock")
elif real_reward < mock_reward * 0.9:
    print("Conclusion: Mock outperforms Real API (unexpected)")
else:
    print("Conclusion: Comparable performance between Mock and Real")

print("=" * 60)
PYTHON_SCRIPT
}

# Main execution
echo "Starting comparison experiments..."
echo ""

# Run mock first (GPU 2)
run_mock &
PID_MOCK=$!

# Run real (GPU 3)
run_real &
PID_REAL=$!

# Wait for both
echo "Waiting for both experiments to complete..."
wait $PID_MOCK $PID_REAL

# Compare results
compare_results

echo ""
echo "=========================================="
echo "Phase 0.5 Comparison Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next step: Run full Phase 2 experiments"
echo "  bash experiment/phase2_controllers/run_real_api.sh"
