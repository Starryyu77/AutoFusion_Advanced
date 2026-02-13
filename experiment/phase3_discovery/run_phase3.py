#!/usr/bin/env python3
"""
Phase 3: Architecture Discovery
-------------------------------
Discover novel multimodal fusion architectures using
winning combination from Phases 1 & 2.

Usage:
    python run_phase3.py --run-name discovery_v1
    python run_phase3.py --iterations 100 --population 50
"""

import json
import time
import argparse
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
EXPERIMENT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parent

sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Import components
from utils.llm_client import DeepSeekClient
from generators.fewshot import FewShotGenerator
from controllers.evolution import EvolutionController
from evaluators.real_data_evaluator import RealDataFewShotEvaluator
from base.reward import RewardComponents


# Phase 3 Configuration
PHASE3_CONFIG = {
    'evaluator': {
        'dataset': 'ai2d',
        'train_epochs': 3,
        'num_shots': 16,
        'batch_size': 4,
        'backbone': 'clip-vit-l-14',
        'data_dir': str(PROJECT_ROOT / 'data' / 'ai2d'),
        'device': 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
    },
    'controller': {
        'population_size': 50,
        'num_iterations': 100,
        'mutation_rate': 0.3,
        'crossover_rate': 0.5,
        'search_space': None,  # Will be set dynamically
    },
    'generator': {
        'model': 'deepseek-chat',
        'temperature': 0.7,
        'max_tokens': 4096,
        'top_p': 0.95,
        'api_key': os.environ.get('DEEPSEEK_API_KEY', ''),
        'base_url': 'https://api.deepseek.com/v1',
    }
}


# Extended Search Space for Architecture Discovery
EXTENDED_SEARCH_SPACE = {
    # Fusion Types (use 'type' to match controller expectation)
    'type': ['attention', 'bilinear', 'mlp', 'transformer', 'gated', 'cross_modal', 'hybrid'],

    # Architecture Components
    'num_fusion_layers': {'type': 'int', 'low': 1, 'high': 6},
    'hidden_dim': {'type': 'int', 'low': 128, 'high': 1024, 'step': 64},
    'num_heads': {'type': 'int', 'low': 2, 'high': 16, 'step': 2},

    # Activation & Normalization
    'activation': ['gelu', 'relu', 'silu', 'swish', 'mish'],
    'normalization': ['layer_norm', 'batch_norm', 'instance_norm', 'none'],

    # Regularization
    'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.05},
    'drop_path_rate': {'type': 'float', 'low': 0.0, 'high': 0.3, 'step': 0.05},

    # Special Components
    'use_residual': [True, False],
    'use_gating': [True, False],
    'use_position_embedding': [True, False],
    'use_layer_scale': [True, False],

    # Connectivity Patterns
    'connectivity': ['serial', 'parallel', 'residual_dense', 'densenet_style'],
}


class ArchitectureDiscovery:
    """Main class for architecture discovery experiment"""

    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.top_architectures_dir = output_dir / 'top_architectures'
        self.results_dir = output_dir / 'results'

        # Create directories
        self.top_architectures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        self.search_history = []
        self.top_architectures = []
        self.architecture_counter = 0

        # Threshold for saving architectures
        self.save_threshold = 0.75  # Save architectures with reward > 0.75

    def initialize_components(self, llm_client: Optional[DeepSeekClient] = None):
        """Initialize experiment components"""
        print("Initializing components...")

        # Initialize LLM client
        if llm_client is None:
            try:
                llm_client = DeepSeekClient(
                    model='deepseek-chat',
                    temperature=0.7,
                )
                print("✓ LLM Client initialized")
            except Exception as e:
                print(f"✗ Failed to initialize LLM client: {e}")
                raise

        # Initialize generator with FewShot (Phase 1 Winner)
        generator_config = self.config['generator'].copy()
        self.generator = FewShotGenerator(llm_client, generator_config)
        print("✓ FewShot Generator initialized")

        # Initialize evaluator
        self.evaluator = RealDataFewShotEvaluator(self.config['evaluator'])
        print("✓ RealDataFewShotEvaluator initialized")

        # Initialize controller with extended search space
        controller_config = self.config['controller'].copy()
        controller_config['search_space'] = EXTENDED_SEARCH_SPACE
        self.controller = EvolutionController(controller_config)
        print("✓ Evolution Controller initialized")

        print(f"\nSearch space size: {len(EXTENDED_SEARCH_SPACE)} dimensions")

    def compute_reward(self, eval_result) -> float:
        """Multi-objective reward calculation"""
        if eval_result is None:
            return 0.0

        # Exponential reward sharpening
        acc_reward = np.exp(2 * eval_result.accuracy - 1)
        eff_reward = eval_result.efficiency ** 0.5 if eval_result.efficiency > 0 else 0

        # Validity penalty
        valid_penalty = 0.0 if eval_result.compile_success else 0.5

        return float(acc_reward * 0.7 + eff_reward * 0.3 - valid_penalty)

    def save_architecture(self, arch_id: str, code: str, arch_desc: Dict,
                         eval_result, reward: float, iteration: int):
        """Save discovered architecture"""
        arch_dir = self.top_architectures_dir / f"arch_{arch_id}"
        arch_dir.mkdir(exist_ok=True)

        # Save code
        with open(arch_dir / 'code.py', 'w') as f:
            f.write(code)

        # Save configuration
        config_data = {
            'arch_id': arch_id,
            'iteration': iteration,
            'architecture_description': arch_desc,
            'timestamp': datetime.now().isoformat(),
        }
        with open(arch_dir / 'config.json', 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

        # Save results
        results_data = {
            'reward': reward,
            'accuracy': float(eval_result.accuracy) if eval_result else 0.0,
            'efficiency': float(eval_result.efficiency) if eval_result else 0.0,
            'compile_success': bool(eval_result.compile_success) if eval_result else False,
            'flops': float(eval_result.flops) if eval_result else 0.0,
            'params': float(eval_result.params) if eval_result else 0.0,
            'latency': float(eval_result.latency) if eval_result else 0.0,
        }
        with open(arch_dir / 'results.json', 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"  💾 Saved architecture {arch_id} (reward={reward:.3f})")

    def run_search_iteration(self, iteration: int) -> Dict:
        """Run single search iteration"""
        iter_start = time.time()

        # 1. Controller proposes architecture
        proposal = self.controller.propose()
        arch_desc = proposal['architecture']

        # 2. Generate code with FewShot
        gen_start = time.time()
        try:
            gen_results = self.generator.generate(arch_desc, num_samples=1)
            gen_result = gen_results[0]
        except Exception as e:
            print(f"  Generation error: {e}")
            gen_result = type('obj', (object,), {
                'success': False,
                'code': '',
                'prompt': '',
                'error': str(e),
                'metadata': {}
            })()
        gen_time = time.time() - gen_start

        # 3. Evaluate architecture
        eval_result = None
        reward = 0.0

        if gen_result.success and gen_result.code:
            try:
                eval_result = self.evaluator.evaluate(gen_result.code)
                reward = self.compute_reward(eval_result)
            except Exception as e:
                print(f"  Evaluation error: {e}")
                eval_result = None
                reward = 0.0

        # 4. Update controller
        try:
            reward_components = RewardComponents(
                accuracy=eval_result.accuracy if eval_result else 0.0,
                efficiency=eval_result.efficiency if eval_result else 0.0,
                compile_success=eval_result.compile_success if eval_result else 0.0,
            )
            self.controller.update(reward_components)
        except Exception as e:
            print(f"  Controller update error: {e}")

        # 5. Record iteration
        iter_result = {
            'iteration': iteration,
            'architecture': arch_desc,
            'code_length': len(gen_result.code) if gen_result.code else 0,
            'success': gen_result.success,
            'reward': reward,
            'accuracy': eval_result.accuracy if eval_result else 0.0,
            'efficiency': eval_result.efficiency if eval_result else 0.0,
            'generation_time': gen_time,
            'total_time': time.time() - iter_start,
        }
        self.search_history.append(iter_result)

        # 6. Save top architectures
        if reward > self.save_threshold and gen_result.success:
            self.architecture_counter += 1
            arch_id = f"{self.architecture_counter:03d}"
            self.save_architecture(
                arch_id, gen_result.code, arch_desc,
                eval_result, reward, iteration
            )
            self.top_architectures.append({
                'id': arch_id,
                'reward': reward,
                'iteration': iteration,
            })

        return iter_result

    def run_discovery(self, num_iterations: int = 100):
        """Run architecture discovery"""
        print(f"\n{'='*60}")
        print("Phase 3: Architecture Discovery")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Population: {self.config['controller']['population_size']}")
        print(f"Save threshold: {self.save_threshold}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()
        best_reward = 0.0
        best_iteration = 0

        for iteration in range(num_iterations):
            # Run iteration
            result = self.run_search_iteration(iteration)

            # Track best
            if result['reward'] > best_reward:
                best_reward = result['reward']
                best_iteration = iteration

            # Progress logging
            if (iteration + 1) % 10 == 0 or iteration == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (iteration + 1)
                remaining = avg_time * (num_iterations - iteration - 1)

                print(f"\n{'='*60}")
                print(f"Progress: {iteration+1}/{num_iterations}")
                print(f"Best Reward: {best_reward:.3f} (Iter {best_iteration})")
                print(f"Top Architectures: {len(self.top_architectures)}")
                print(f"Avg Time/Iter: {avg_time:.1f}s")
                print(f"ETA: {remaining/60:.1f} min")
                print(f"{'='*60}\n")

            # Save checkpoint every 20 iterations
            if (iteration + 1) % 20 == 0:
                self.save_checkpoint(iteration)

        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("Discovery Complete!")
        print(f"{'='*60}")
        print(f"Total Time: {total_time/60:.1f} min")
        print(f"Best Reward: {best_reward:.3f} (Iteration {best_iteration})")
        print(f"Top Architectures Discovered: {len(self.top_architectures)}")
        print(f"{'='*60}\n")

        return self.top_architectures

    def save_checkpoint(self, iteration: int):
        """Save search checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'search_history': self.search_history,
            'top_architectures': self.top_architectures,
            'controller_state': self.controller.state.to_dict() if hasattr(self.controller, 'state') else {},
        }

        checkpoint_path = self.results_dir / f'checkpoint_iter_{iteration:03d}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        print(f"  💾 Checkpoint saved: {checkpoint_path.name}")

    def generate_report(self):
        """Generate final discovery report"""
        report_lines = [
            "# Phase 3: Architecture Discovery Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Iterations:** {len(self.search_history)}",
            f"**Top Architectures:** {len(self.top_architectures)}",
            "",
            "## Top Discovered Architectures",
            "",
            "| Rank | ID | Reward | Iteration |",
            "|------|-----|--------|-----------|",
        ]

        # Sort by reward
        sorted_archs = sorted(self.top_architectures, key=lambda x: x['reward'], reverse=True)
        for rank, arch in enumerate(sorted_archs[:10], 1):
            report_lines.append(f"| {rank} | arch_{arch['id']} | {arch['reward']:.3f} | {arch['iteration']} |")

        report_lines.extend([
            "",
            "## Search Statistics",
            "",
        ])

        if self.search_history:
            rewards = [h['reward'] for h in self.search_history]
            report_lines.extend([
                f"- **Mean Reward:** {np.mean(rewards):.3f}",
                f"- **Std Reward:** {np.std(rewards):.3f}",
                f"- **Max Reward:** {np.max(rewards):.3f}",
                f"- **Min Reward:** {np.min(rewards):.3f}",
            ])

        report_path = self.results_dir / 'discovery_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"📊 Report saved: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Architecture Discovery')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this discovery run')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of search iterations')
    parser.add_argument('--population', type=int, default=50,
                       help='Population size for Evolution controller')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Reward threshold for saving architectures')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use')

    args = parser.parse_args()

    # Set GPU if specified
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or f"discovery_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update config
    config = PHASE3_CONFIG.copy()
    config['controller']['population_size'] = args.population
    config['controller']['num_iterations'] = args.iterations

    # Initialize discovery
    discovery = ArchitectureDiscovery(config, output_dir)
    discovery.save_threshold = args.threshold

    try:
        # Initialize components
        discovery.initialize_components()

        # Run discovery
        top_archs = discovery.run_discovery(num_iterations=args.iterations)

        # Generate report
        discovery.generate_report()

        print(f"\n✅ Discovery complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"🏆 Top architectures: {len(top_archs)}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        discovery.save_checkpoint(len(discovery.search_history))
        print(f"Checkpoint saved to: {output_dir}")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        raise


if __name__ == '__main__':
    main()
