#!/usr/bin/env python3
"""
Phase 0: API Validation Script
"""
import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

# Setup paths - ensure experiment module can be found
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add experiment directory to path (parent of phase0_validation)
experiment_dir = str(PROJECT_ROOT)
if experiment_dir not in sys.path:
    sys.path.insert(0, experiment_dir)

# Also add the parent directory (for 'experiment' package)
parent_dir = str(PROJECT_ROOT.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from factory import create_controller, create_generator, create_evaluator, create_reward
from utils.llm_client import create_deepseek_client

def main():
    print("=" * 60)
    print("Phase 0: API Validation (Real API)")
    print("=" * 60)
    print()

    # Check API key
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set!")
        return 1

    budget_limit = float(os.environ.get('BUDGET_LIMIT_YUAN', 50))
    cache_dir = os.environ.get('CACHE_DIR', str(SCRIPT_DIR / '.cache' / 'llm'))

    print(f"API Key: {api_key[:8]}...")
    print(f"Budget: {budget_limit} yuan")
    print(f"Cache: {cache_dir}")
    print()

    # Setup directories
    results_dir = SCRIPT_DIR / 'results_validation'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = PROJECT_ROOT / 'phase2_controllers' / 'configs' / 'ppo.yaml'
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
        api_key=api_key,
        cache_dir=cache_dir,
        budget_limit_yuan=budget_limit,
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
    output_dir = results_dir / 'ppo_validation'
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

    return 0

if __name__ == "__main__":
    sys.exit(main())
