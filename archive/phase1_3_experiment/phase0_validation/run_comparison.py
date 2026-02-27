#!/usr/bin/env python3
"""
Phase 0.5: Mock vs Real API Comparison
Compare the same controller with mock and real API generation.
"""
import sys
import os
import yaml
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add experiment directory to path
experiment_dir = str(PROJECT_ROOT)
if experiment_dir not in sys.path:
    sys.path.insert(0, experiment_dir)

parent_dir = str(PROJECT_ROOT.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from factory import create_controller, create_generator, create_evaluator, create_reward
from utils.llm_client import create_deepseek_client


def run_experiment(use_real_api: bool, gpu_id: int = 0) -> dict:
    """Run a single experiment with mock or real API."""
    exp_type = "Real API" if use_real_api else "Mock"
    print(f"\n{'='*60}")
    print(f"Running: {exp_type}")
    print(f"{'='*60}\n")

    # Load config
    config_path = PROJECT_ROOT / 'phase2_controllers' / 'configs' / 'ppo.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['controller']['max_iterations'] = 20

    # Set seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.set_device(gpu_id)

    # Create LLM client if using real API
    llm_client = None
    if use_real_api:
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            print("ERROR: DEEPSEEK_API_KEY not set!")
            return None

        cache_dir = os.environ.get('CACHE_DIR', str(SCRIPT_DIR / '.cache' / 'llm'))
        budget_limit = float(os.environ.get('BUDGET_LIMIT_YUAN', 200))

        llm_client = create_deepseek_client(
            api_key=api_key,
            cache_dir=cache_dir,
            budget_limit_yuan=budget_limit,
        )
        print(f"DeepSeek client created (budget: {budget_limit} yuan)")

    # Create components
    controller = create_controller('ppo', config['controller'])
    generator = create_generator('cot', llm_client, config['generator'])
    evaluator = create_evaluator('sandbox', config['evaluator'])
    reward_fn = create_reward(config['reward'])

    print(f"Running 20 iterations with {exp_type}...\n")

    results = {
        'type': 'real' if use_real_api else 'mock',
        'rewards': [],
        'compile_success': [],
        'iterations': []
    }

    for iteration in range(20):
        # Propose
        proposal = controller.propose()
        architecture = proposal['architecture']

        # Generate
        gen_results = generator.generate(architecture, num_samples=1)
        gen_result = gen_results[0]

        # Track compile success
        compile_success = gen_result.success
        results['compile_success'].append(compile_success)

        # Evaluate
        if compile_success:
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
        controller.update(reward_components)
        controller.record_iteration(architecture, reward_components)

        # Log
        reward_scalar = reward_components.to_scalar(config['reward']['weights'])
        results['rewards'].append(reward_scalar)
        results['iterations'].append(iteration + 1)

        cost_str = ""
        if use_real_api and llm_client:
            cost_str = f" | Cost: {llm_client.stats.total_cost_yuan:.2f}¥"

        print(f"  Iter {iteration+1:2d}/20 | {status} Reward: {reward_scalar:.4f}{cost_str}")

    # Final stats
    stats = controller.get_stats()
    results['best_reward'] = stats['best_reward']
    results['final_iteration'] = stats['iteration']

    print(f"\n{exp_type} - Best Reward: {stats['best_reward']:.4f}")

    # Save API stats if real
    if use_real_api and llm_client:
        llm_client.print_stats()
        results['api_stats'] = llm_client.stats.to_dict()

    return results


def compare_results(mock_results: dict, real_results: dict):
    """Compare and print results."""
    print("\n" + "="*60)
    print("Mock vs Real API Comparison")
    print("="*60)
    print()
    print(f"Controller: PPO")
    print(f"Seed: 42")
    print(f"Iterations: 20")
    print()

    mock_reward = mock_results['best_reward']
    real_reward = real_results['best_reward']
    diff = real_reward - mock_reward

    print("Results:")
    print(f"  Mock:      {mock_reward:.4f}")
    print(f"  Real:      {real_reward:.4f}")
    print(f"  Diff:      {diff:+.4f}")
    if mock_reward > 0:
        print(f"  Change:    {(diff/mock_reward)*100:+.1f}%")
    print()

    # Compile success rate
    mock_compile = sum(mock_results['compile_success']) / len(mock_results['compile_success'])
    real_compile = sum(real_results['compile_success']) / len(real_results['compile_success'])

    print("Compile Success Rate:")
    print(f"  Mock:      {mock_compile:.1%}")
    print(f"  Real:      {real_compile:.1%}")
    print()

    # API cost
    if 'api_stats' in real_results:
        api_stats = real_results['api_stats']
        print("API Statistics (Real):")
        print(f"  Total Calls:   {api_stats['total_calls']}")
        print(f"  Cache Hits:    {api_stats['cache_hits']}")
        print(f"  Total Cost:    {api_stats['total_cost_yuan']:.2f} yuan")
        print()

    print("="*60)

    # Analysis
    if real_reward > mock_reward * 1.1:
        print("Conclusion: Real API significantly outperforms Mock (+10%+)")
    elif real_reward > mock_reward * 1.05:
        print("Conclusion: Real API moderately outperforms Mock (+5-10%)")
    elif real_reward < mock_reward * 0.95:
        print("Conclusion: Mock outperforms Real API (unexpected)")
    else:
        print("Conclusion: Comparable performance between Mock and Real")

    print("="*60)


def main():
    print("="*60)
    print("Phase 0.5: Mock vs Real API Comparison")
    print("="*60)
    print()

    # Create output directory
    results_dir = SCRIPT_DIR / 'results_comparison'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run Mock experiment
    mock_results = run_experiment(use_real_api=False, gpu_id=2)
    if mock_results:
        with open(results_dir / 'mock_results.json', 'w') as f:
            json.dump(mock_results, f, indent=2)

    # Run Real API experiment
    real_results = run_experiment(use_real_api=True, gpu_id=3)
    if real_results:
        with open(results_dir / 'real_results.json', 'w') as f:
            json.dump(real_results, f, indent=2)

    # Compare
    if mock_results and real_results:
        compare_results(mock_results, real_results)

        # Save combined summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'controller': 'ppo',
            'seed': 42,
            'iterations': 20,
            'mock': mock_results,
            'real': real_results,
        }
        with open(results_dir / 'comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {results_dir}")

    print("\n" + "="*60)
    print("Phase 0.5 Complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
