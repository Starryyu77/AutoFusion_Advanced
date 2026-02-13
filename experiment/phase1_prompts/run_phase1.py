#!/usr/bin/env python3
"""
Phase 1: Prompt Strategy Comparison
-----------------------------------
Compare 5 generator strategies using Evolution controller
and verified RealDataFewShotEvaluator.

Usage:
    python run_phase1.py --run-name phase1_test
    python run_phase1.py --strategy CoT --iterations 10
    python run_phase1.py --mock  # Use mock generation for testing
"""

import json
import time
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENT_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Import components
from utils.llm_client import DeepSeekClient
from generators.cot import ChainOfThoughtGenerator
from generators.fewshot import FewShotGenerator
from generators.critic import CriticGenerator
from generators.shape import ShapeConstraintGenerator
from generators.roleplay import RolePlayGenerator
from controllers.evolution import EvolutionController
from evaluators.real_data_evaluator import RealDataFewShotEvaluator
from base.reward import RewardComponents


# 统一配置
VERIFIED_EVALUATOR_CONFIG = {
    'dataset': 'ai2d',
    'train_epochs': 3,
    'num_shots': 16,
    'batch_size': 4,
    'backbone': 'clip-vit-l-14',
    'data_dir': str(PROJECT_ROOT / 'data' / 'ai2d'),
    'device': 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
}

CONTROLLER_CONFIG = {
    'population_size': 20,
    'num_iterations': 20,
    'mutation_rate': 0.3,
    'crossover_rate': 0.5,
}

GENERATOR_CONFIG = {
    'model': 'deepseek-chat',
    'temperature': 0.7,
    'max_tokens': 4096,
    'top_p': 0.95,
}


def run_single_strategy(
    strategy_name: str,
    generator_class,
    llm_client: DeepSeekClient,
    evaluator_config: Dict,
    controller_config: Dict,
    use_mock: bool = False,
) -> Dict[str, Any]:
    """
    Run a single prompt strategy through the complete search process.

    Returns:
        results: Dictionary containing all experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running Strategy: {strategy_name}")
    print(f"{'='*60}")

    # Initialize components
    if use_mock:
        generator = generator_class(None, GENERATOR_CONFIG)
    else:
        generator = generator_class(llm_client, GENERATOR_CONFIG)

    evaluator = RealDataFewShotEvaluator(evaluator_config)
    controller = EvolutionController(controller_config)

    results = {
        'strategy': strategy_name,
        'iterations': [],
        'start_time': time.time(),
        'config': {
            'evaluator': evaluator_config,
            'controller': controller_config,
            'generator': GENERATOR_CONFIG,
        }
    }

    best_reward = 0.0
    best_architecture = None
    convergence_iteration = None
    total_api_calls = 0

    num_iterations = controller_config.get('num_iterations', 20)

    for iteration in range(num_iterations):
        iter_start = time.time()

        # 1. Controller proposes architecture description
        proposal = controller.propose()
        arch_desc = proposal['architecture']

        # 2. Generator generates code
        gen_start = time.time()
        try:
            gen_results = generator.generate(arch_desc, num_samples=1)
            gen_result = gen_results[0]
            total_api_calls += 1
        except Exception as e:
            print(f"  Generation error: {e}")
            gen_result = type('obj', (object,), {
                'success': False,
                'code': '',
                'prompt': '',
                'error': str(e),
                'metadata': {'strategy': strategy_name.lower()}
            })()

        gen_time = time.time() - gen_start

        # 3. Evaluator evaluates
        eval_result = None
        reward = 0.0

        if gen_result.success and gen_result.code:
            try:
                eval_result = evaluator.evaluate(gen_result.code)
                reward = compute_reward(eval_result)
            except Exception as e:
                print(f"  Evaluation error: {e}")
                eval_result = None
                reward = 0.0

        # 4. Controller updates
        try:
            # Create RewardComponents from evaluation result
            reward_components = RewardComponents(
                accuracy=eval_result.accuracy if eval_result else 0.0,
                efficiency=eval_result.efficiency if eval_result else 0.0,
                compile_success=eval_result.compile_success if eval_result else 0.0,
            )
            controller.update(reward_components)
        except Exception as e:
            print(f"  Controller update error: {e}")

        # 5. Record results
        iter_result = {
            'iteration': iteration,
            'architecture': arch_desc,
            'code_length': len(gen_result.code) if gen_result.code else 0,
            'success': gen_result.success,
            'error': gen_result.error if not gen_result.success else None,
            'reward': float(reward),
            'accuracy': float(eval_result.accuracy) if eval_result else 0.0,
            'efficiency': float(eval_result.efficiency) if eval_result else 0.0,
            'compile_success': float(eval_result.compile_success) if eval_result else 0.0,
            'flops': float(eval_result.flops) if eval_result else 0.0,
            'params': float(eval_result.params) if eval_result else 0.0,
            'latency': float(eval_result.latency) if eval_result else 0.0,
            'generation_time': gen_time,
        }
        results['iterations'].append(iter_result)

        # Update best results
        if reward > best_reward:
            best_reward = reward
            best_architecture = arch_desc
            convergence_iteration = iteration

        # Progress logging
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  Iter {iteration+1}/{num_iterations}: "
                  f"reward={reward:.3f}, valid={gen_result.success}, "
                  f"acc={iter_result['accuracy']:.3f}, time={gen_time:.1f}s")

    # Summary statistics
    results['end_time'] = time.time()
    results['total_time'] = results['end_time'] - results['start_time']
    results['best_reward'] = float(best_reward)
    results['best_architecture'] = best_architecture
    results['convergence_iteration'] = int(convergence_iteration) if convergence_iteration is not None else num_iterations
    results['validity_rate'] = sum(1 for r in results['iterations'] if r['success']) / len(results['iterations'])
    results['avg_generation_time'] = sum(r['generation_time'] for r in results['iterations']) / len(results['iterations'])
    results['total_api_calls'] = total_api_calls
    results['final_accuracy'] = sum(r['accuracy'] for r in results['iterations'][-5:]) / 5  # Last 5 avg

    # Summary
    print(f"\n  Summary for {strategy_name}:")
    print(f"    Best Reward: {results['best_reward']:.3f}")
    print(f"    Validity Rate: {results['validity_rate']:.1%}")
    print(f"    Convergence: Iter {results['convergence_iteration']}")
    print(f"    Avg Gen Time: {results['avg_generation_time']:.1f}s")
    print(f"    Total Time: {results['total_time']:.1f}s")

    return results


def compute_reward(eval_result) -> float:
    """Multi-objective reward calculation (same as Phase 2.1)"""
    if eval_result is None:
        return 0.0

    # Exponential reward sharpening
    acc_reward = np.exp(2 * eval_result.accuracy - 1)
    eff_reward = eval_result.efficiency ** 0.5 if eval_result.efficiency > 0 else 0

    # Validity penalty
    valid_penalty = 0.0 if eval_result.compile_success else 0.5

    return float(acc_reward * 0.7 + eff_reward * 0.3 - valid_penalty)


def analyze_results(all_results: List[Dict]) -> Dict:
    """Cross-strategy comparative analysis"""
    analysis = {
        'rankings': {},
        'metrics_by_strategy': {},
        'winner': None,
    }

    # Sort by different metrics
    for metric in ['best_reward', 'validity_rate', 'final_accuracy',
                   'convergence_iteration', 'avg_generation_time']:
        reverse = metric != 'convergence_iteration' and metric != 'avg_generation_time'
        sorted_results = sorted(all_results,
                               key=lambda x: x.get(metric, 0),
                               reverse=reverse)
        analysis['rankings'][metric] = [
            (r['strategy'], r.get(metric, 0)) for r in sorted_results
        ]

    # Calculate detailed metrics for each strategy
    for result in all_results:
        strategy = result['strategy']
        analysis['metrics_by_strategy'][strategy] = {
            'best_reward': result['best_reward'],
            'validity_rate': result['validity_rate'],
            'final_accuracy': result['final_accuracy'],
            'convergence_iteration': result['convergence_iteration'],
            'avg_generation_time': result['avg_generation_time'],
            'total_time': result['total_time'],
            'total_api_calls': result['total_api_calls'],
        }

    # Overall winner (composite score)
    scores = {}
    for strategy in analysis['metrics_by_strategy']:
        reward_rank = next(i for i, (s, _) in enumerate(analysis['rankings']['best_reward']) if s == strategy)
        validity_rank = next(i for i, (s, _) in enumerate(analysis['rankings']['validity_rate']) if s == strategy)
        convergence_rank = next(i for i, (s, _) in enumerate(analysis['rankings']['convergence_iteration']) if s == strategy)

        # Lower total rank is better
        scores[strategy] = reward_rank + validity_rank + convergence_rank

    analysis['winner'] = min(scores, key=scores.get)
    analysis['composite_scores'] = scores

    return analysis


def generate_markdown_report(analysis: Dict, all_results: List[Dict]) -> str:
    """Generate Markdown format experiment report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = f"""# Phase 1: Prompt Strategy Comparison Report

Generated: {timestamp}

## Summary

| Strategy | Best Reward | Validity Rate | Final Acc | Convergence | Avg Time |
|----------|-------------|---------------|-----------|-------------|----------|
"""

    for result in all_results:
        report += (f"| {result['strategy']} | {result['best_reward']:.3f} | "
                  f"{result['validity_rate']:.1%} | {result['final_accuracy']:.3f} | "
                  f"{result['convergence_iteration']} | "
                  f"{result['avg_generation_time']:.1f}s |\n")

    report += f"\n## 🏆 Winner: **{analysis['winner']}**\n\n"

    report += "### Composite Scores (lower is better)\n\n"
    for strategy, score in sorted(analysis['composite_scores'].items(), key=lambda x: x[1]):
        report += f"- {strategy}: {score}\n"

    report += "\n## Rankings by Metric\n\n"
    for metric, ranking in analysis['rankings'].items():
        report += f"### {metric}\n\n"
        for i, (strategy, value) in enumerate(ranking, 1):
            if isinstance(value, float):
                report += f"{i}. **{strategy}**: {value:.3f}\n"
            else:
                report += f"{i}. **{strategy}**: {value}\n"
        report += "\n"

    report += "## Recommendations\n\n"

    winner_metrics = analysis['metrics_by_strategy'][analysis['winner']]
    report += f"""
Based on the composite scoring across reward, validity rate, and convergence speed:

**Recommended Strategy: {analysis['winner']}**

Key metrics:
- Best Reward: {winner_metrics['best_reward']:.3f}
- Validity Rate: {winner_metrics['validity_rate']:.1%}
- Convergence: Iteration {winner_metrics['convergence_iteration']}
- Average Generation Time: {winner_metrics['avg_generation_time']:.1f}s

### Strategy-Specific Insights

"""

    # Add insights for each strategy
    for strategy, metrics in analysis['metrics_by_strategy'].items():
        report += f"**{strategy}**: "
        if metrics['validity_rate'] > 0.9:
            report += "High code validity; "
        if metrics['best_reward'] > 8.0:
            report += "Excellent architecture quality; "
        if metrics['convergence_iteration'] < 10:
            report += "Fast convergence; "
        report += "\n\n"

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Prompt Strategy Comparison Experiment'
    )
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this run')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Run single strategy (CoT, FewShot, Critic, Shape, RolePlay)')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of iterations per strategy')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock generation (no API calls)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use')

    args = parser.parse_args()

    # Set GPU if specified
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Using GPU {args.gpu}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or f"phase1_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    logs_dir = Path(__file__).parent / 'logs'
    logs_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print("Phase 1: Prompt Strategy Comparison Experiment")
    print(f"{'='*70}")
    print(f"Output Directory: {output_dir}")
    print(f"Mock Mode: {args.mock}")
    print(f"Iterations: {args.iterations}")
    print(f"Device: {VERIFIED_EVALUATOR_CONFIG['device']}")
    print(f"{'='*70}\n")

    # Initialize LLM client
    llm_client = None
    if not args.mock:
        try:
            llm_client = DeepSeekClient(
                model='deepseek-chat',
                temperature=0.7,
            )
            print("LLM Client initialized successfully\n")
        except Exception as e:
            print(f"Warning: Failed to initialize LLM client: {e}")
            print("Falling back to mock mode\n")
            args.mock = True

    # Define strategies to test
    strategies = [
        ('CoT', ChainOfThoughtGenerator),
        ('FewShot', FewShotGenerator),
        ('Critic', CriticGenerator),
        ('Shape', ShapeConstraintGenerator),
        ('RolePlay', RolePlayGenerator),
    ]

    # Filter to single strategy if specified
    if args.strategy:
        strategies = [(name, cls) for name, cls in strategies if name.lower() == args.strategy.lower()]
        if not strategies:
            print(f"Error: Unknown strategy '{args.strategy}'")
            print(f"Available: CoT, FewShot, Critic, Shape, RolePlay")
            return

    # Update controller config with iteration count
    controller_config = CONTROLLER_CONFIG.copy()
    controller_config['num_iterations'] = args.iterations

    # Run all strategies
    all_results = []
    for name, gen_class in strategies:
        result = run_single_strategy(
            strategy_name=name,
            generator_class=gen_class,
            llm_client=llm_client,
            evaluator_config=VERIFIED_EVALUATOR_CONFIG,
            controller_config=controller_config,
            use_mock=args.mock,
        )
        all_results.append(result)

        # Save individual result
        result_file = output_dir / f'{name.lower()}_results.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {result_file}")

    # Analyze results
    if len(all_results) > 1:
        analysis = analyze_results(all_results)

        # Save analysis
        analysis_file = output_dir / 'analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Generate and save report
        report = generate_markdown_report(analysis, all_results)
        report_file = output_dir / 'report.md'
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"\n{'='*70}")
        print("Experiment Complete!")
        print(f"{'='*70}")
        print(f"Results saved to: {output_dir}")
        print(f"\n🏆 Winner: {analysis['winner']}")
        print(f"\nRankings by Best Reward:")
        for i, (strategy, value) in enumerate(analysis['rankings']['best_reward'], 1):
            print(f"  {i}. {strategy}: {value:.3f}")
    else:
        print(f"\n{'='*70}")
        print("Single Strategy Run Complete!")
        print(f"{'='*70}")
        print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
