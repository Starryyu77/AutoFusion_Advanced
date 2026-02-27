"""
Phase 4: Optimized Architecture Search
--------------------------------------
Run architecture search with improved evaluator and constrained reward.

This script runs on GPU cluster (NTU EEE Cluster) with:
- MMMU dataset evaluator
- Constrained reward (FLOPs < 10M)
- Evolution controller
- 200 iterations search

Usage on cluster:
    cd /projects/tianyu016/AutoFusion_Advanced
    python phase4_optimization/run_phase4_search.py

Submit with Slurm:
    sbatch phase4_optimization/scripts/submit_phase4.sh
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
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiment'))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np

# Setup logging
def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase4_search_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def create_evaluator(config: dict):
    """Create evaluator with Phase 4 configuration."""
    from evaluator_v2_improved import ImprovedRealDataFewShotEvaluator

    evaluator_config = {
        'dataset': config.get('dataset', 'mmmu'),
        'num_shots': config.get('num_shots', 32),
        'train_epochs': config.get('train_epochs', 10),
        'batch_size': config.get('batch_size', 8),
        'backbone': config.get('backbone', 'clip-vit-l-14'),
        'data_dir': config.get('data_dir', './expv2/data'),
        'device': config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'early_stopping': {
            'enabled': config.get('early_stopping_enabled', True),
            'patience': config.get('early_stopping_patience', 3),
            'min_delta': config.get('early_stopping_min_delta', 0.005),
        },
        'max_training_time': config.get('max_training_time', 300),
        'eval_every_n_epochs': config.get('eval_every_n_epochs', 1),
    }

    return ImprovedRealDataFewShotEvaluator(evaluator_config)


def create_reward(config: dict):
    """Create constrained reward function."""
    from reward_v2 import ConstrainedReward

    reward_config = {
        'weights': {
            'accuracy': config.get('weight_accuracy', 1.0),
            'efficiency': config.get('weight_efficiency', 1.5),
            'compile_success': config.get('weight_compile_success', 2.0),
            'complexity': config.get('weight_complexity', 0.3),
        },
        'flops_constraint': {
            'enabled': config.get('flops_constraint_enabled', True),
            'max_flops': config.get('max_flops', 10e6),
            'reject_if_exceed': config.get('reject_if_exceed', True),
        },
        'flops_penalty': {
            'type': config.get('penalty_type', 'exponential'),
            'scale': config.get('penalty_scale', 20e6),
        },
        'label_smoothing': config.get('label_smoothing', True),
    }

    return ConstrainedReward(reward_config)


def create_controller(config: dict, search_space: dict):
    """Create simple random controller."""
    return SimpleController(search_space, config)


class SimpleController:
    """Simple random architecture generator."""

    def __init__(self, search_space: dict, config: dict):
        self.search_space = search_space
        self.config = config
        self.population_size = config.get('population_size', 50)
        self.history = []
        self.iteration = 0
        np.random.seed(config.get('seed', 42))

    def propose(self, context: dict = None) -> dict:
        """Generate random architecture."""
        arch = {}
        for key, spec in self.search_space.items():
            if spec['type'] == 'categorical':
                arch[key] = np.random.choice(spec['choices'])
            elif spec['type'] == 'int':
                low, high = spec['low'], spec['high']
                step = spec.get('step', 1)
                arch[key] = int(np.random.randint(low, high + 1) // step * step)
            elif spec['type'] == 'float':
                low, high = spec['low'], spec['high']
                arch[key] = float(np.random.uniform(low, high))
        self.iteration += 1
        return arch

    def generate_next(self) -> dict:
        """Alias for propose for compatibility."""
        return self.propose()

    def update(self, reward) -> None:
        """Record history."""
        self.history.append(reward)

    def get_stats(self) -> dict:
        """Return stats."""
        return {'history_size': len(self.history)}


def create_generator(config: dict):
    """Create code generator."""
    # Simple mock generator for Phase 4 (generates fusion module code based on config)
    return MockGenerator(config)


class MockGenerator:
    """Mock generator that creates fusion module code from architecture config."""

    def __init__(self, config: dict):
        self.config = config

    def generate(self, arch_config: dict) -> str:
        """Generate fusion module code based on architecture config."""
        fusion_type = arch_config.get('fusion_type', 'attention')
        num_layers = arch_config.get('num_layers', 2)
        hidden_dim = arch_config.get('hidden_dim', 256)
        num_heads = arch_config.get('num_heads', 4)
        dropout = arch_config.get('dropout', 0.1)
        use_residual = arch_config.get('use_residual', True)

        # Ensure hidden_dim is divisible by num_heads for attention
        if fusion_type in ['attention', 'lightweight_attention']:
            # Round hidden_dim down to nearest multiple of num_heads
            hidden_dim = (hidden_dim // num_heads) * num_heads
            if hidden_dim == 0:
                hidden_dim = num_heads  # Minimum size

        # Generate different fusion modules based on type
        if fusion_type == 'attention':
            return self._generate_attention_module(hidden_dim, num_heads, dropout, use_residual, num_layers)
        elif fusion_type == 'bilinear':
            return self._generate_bilinear_module(hidden_dim, dropout, use_residual)
        elif fusion_type == 'mlp':
            return self._generate_mlp_module(hidden_dim, dropout, use_residual, num_layers)
        elif fusion_type == 'gated':
            return self._generate_gated_module(hidden_dim, dropout, use_residual)
        elif fusion_type == 'film':
            return self._generate_film_module(hidden_dim, dropout)
        else:
            return self._generate_attention_module(hidden_dim, num_heads, dropout, use_residual, num_layers)

    def _generate_attention_module(self, hidden_dim: int, num_heads: int, dropout: float, use_residual: bool, num_layers: int) -> str:
        residual_code = "\n        x = x + attn_out" if use_residual else ""
        return f'''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, {num_heads}, batch_first=True, dropout={dropout})
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout({dropout})

    def forward(self, vision_features, language_features):
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        attn_out, _ = self.attn(v.unsqueeze(1), l.unsqueeze(1), l.unsqueeze(1))
        attn_out = self.dropout(attn_out)
        x = self.norm(attn_out.squeeze(1))
        x = self.output_proj(x){residual_code}
        return x
'''

    def _generate_bilinear_module(self, hidden_dim: int, dropout: float, use_residual: bool) -> str:
        return f'''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}):
        super().__init__()
        self.bilinear = nn.Bilinear(vision_dim, language_dim, hidden_dim)
        self.dropout = nn.Dropout({dropout})
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        fused = self.bilinear(vision_features, language_features)
        fused = self.dropout(fused)
        return self.norm(fused)
'''

    def _generate_mlp_module(self, hidden_dim: int, dropout: float, use_residual: bool, num_layers: int) -> str:
        layers = []
        for i in range(num_layers):
            layers.append(f'        self.fc{i+1} = nn.Linear(hidden_dim, hidden_dim)')
        layers_code = '\n'.join(layers)
        return f'''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}):
        super().__init__()
        self.input_proj = nn.Linear(vision_dim + language_dim, hidden_dim)
{layers_code}
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout({dropout})
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        x = torch.cat([vision_features, language_features], dim=-1)
        x = self.input_proj(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        return self.norm(x)
'''

    def _generate_gated_module(self, hidden_dim: int, dropout: float, use_residual: bool) -> str:
        return f'''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout({dropout})
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        concat = torch.cat([v, l], dim=-1)
        g = self.gate(concat)
        fused = g * v + (1 - g) * l
        fused = self.dropout(fused)
        return self.norm(self.output_proj(fused))
'''

    def _generate_film_module(self, hidden_dim: int, dropout: float) -> str:
        return f'''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.gamma_proj = nn.Linear(language_dim, hidden_dim)
        self.beta_proj = nn.Linear(language_dim, hidden_dim)
        self.dropout = nn.Dropout({dropout})
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        v = self.vision_proj(vision_features)
        gamma = self.gamma_proj(language_features)
        beta = self.beta_proj(language_features)
        # FiLM: Feature-wise Linear Modulation
        fused = gamma * v + beta
        fused = self.dropout(fused)
        return self.norm(fused)
'''


def get_search_space():
    """Define Phase 4 search space with strict efficiency constraints."""
    return {
        'fusion_type': {
            'type': 'categorical',
            'choices': [
                'attention',
                'bilinear',
                'mlp',
                'gated',
                'film',
            ]
        },
        'num_layers': {
            'type': 'int',
            'low': 1,
            'high': 3,  # Further reduced
        },
        'hidden_dim': {
            'type': 'int',
            'low': 64,   # Reduced from 128
            'high': 256, # Reduced from 512
            'step': 64,
        },
        'use_residual': {
            'type': 'categorical',
            'choices': [True],
        },
        'num_heads': {
            'type': 'int',
            'low': 1,
            'high': 4,  # Reduced from 8
        },
        'dropout': {
            'type': 'float',
            'low': 0.0,
            'high': 0.3,
        },
    }


def run_architecture_search(args):
    """Run the main architecture search loop."""
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / 'logs', args.log_level)
    logger.info("=" * 70)
    logger.info("Phase 4: Optimized Architecture Search")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create components
    logger.info("Creating components...")
    search_space = get_search_space()
    logger.info(f"Search space: {list(search_space.keys())}")

    evaluator = create_evaluator(vars(args))
    logger.info(f"Evaluator: {evaluator.__class__.__name__}")
    logger.info(f"  Dataset: {evaluator.dataset_name}")
    logger.info(f"  Epochs: {evaluator.train_epochs}")
    logger.info(f"  Max time: {evaluator.max_training_time}s")

    reward_fn = create_reward(vars(args))
    logger.info(f"Reward: {reward_fn.__class__.__name__}")
    logger.info(f"  Efficiency weight: {reward_fn.weights['efficiency']}")
    logger.info(f"  Max FLOPs: {reward_fn.max_flops/1e6:.1f}M")

    controller = create_controller(vars(args), search_space)
    logger.info(f"Controller: {controller.__class__.__name__}")
    logger.info(f"  Population size: {controller.population_size}")

    generator = create_generator(vars(args))
    logger.info(f"Generator: {generator.__class__.__name__}")

    # Initialize population
    logger.info("\nInitializing population...")
    population = []
    for i in range(args.population_size):
        arch = controller.propose(context={'iteration': i})
        population.append(arch)
    logger.info(f"Initialized {len(population)} architectures")

    # Track results
    all_results = []
    rejected_count = 0
    compile_fail_count = 0

    # Search loop
    logger.info(f"\nStarting search for {args.num_iterations} iterations...")
    logger.info("=" * 70)

    for iteration in range(args.num_iterations):
        iter_start = time.time()
        logger.info(f"\nIteration {iteration + 1}/{args.num_iterations}")
        logger.info("-" * 70)

        # Select architecture to evaluate
        if iteration < len(population):
            # Initial population
            arch_config = population[iteration]
        else:
            # Evolve new architecture
            arch_config = controller.generate_next()

        # Generate code
        logger.info(f"Generating code for architecture...")
        try:
            generated_code = generator.generate(arch_config)
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            compile_fail_count += 1
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
        reward_components = reward_fn.calculate({
            'accuracy': eval_result.accuracy,
            'flops': eval_result.flops,
            'params': eval_result.params,
            'compile_success': eval_result.compile_success,
        })

        scalar_reward = reward_components.to_scalar(reward_fn.weights)

        # Check if rejected
        if hasattr(reward_components, 'rejected') and reward_components.rejected:
            rejected_count += 1
            logger.warning(f"Architecture REJECTED: {reward_components.rejection_reason}")

        # Log results
        iter_time = time.time() - iter_start
        logger.info(f"Results:")
        logger.info(f"  Accuracy: {eval_result.accuracy:.4f}")
        logger.info(f"  FLOPs: {eval_result.flops/1e6:.2f}M")
        logger.info(f"  Params: {eval_result.params/1e6:.2f}M")
        logger.info(f"  Latency: {eval_result.latency:.2f}ms")
        logger.info(f"  Efficiency: {eval_result.efficiency:.4f}")
        logger.info(f"  Reward: {scalar_reward:.4f}")
        logger.info(f"  Time: {iter_time:.1f}s")

        if hasattr(reward_components, 'rejected'):
            logger.info(f"  Rejected: {reward_components.rejected}")

        # Store result
        result_record = {
            'iteration': iteration,
            'architecture': arch_config,
            'code': generated_code,
            'accuracy': eval_result.accuracy,
            'flops': eval_result.flops,
            'params': eval_result.params,
            'latency': eval_result.latency,
            'efficiency': eval_result.efficiency,
            'reward': scalar_reward,
            'rejected': getattr(reward_components, 'rejected', False),
            'rejection_reason': getattr(reward_components, 'rejection_reason', ''),
            'eval_time': iter_time,
            'metadata': eval_result.metadata,
        }
        all_results.append(result_record)

        # Update controller
        if iteration >= len(population):
            controller.update(reward_components)

        # Periodic saving
        if (iteration + 1) % args.save_interval == 0:
            save_results(output_dir, all_results, iteration + 1)

        # Milestone evaluation
        if (iteration + 1) in [50, 100, 150, 200]:
            logger.info(f"\n{'='*70}")
            logger.info(f"MILESTONE: {iteration + 1} iterations completed")
            logger.info(f"{'='*70}")
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

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Also save top architectures
    if len(results) > 0:
        valid_results = [r for r in results if not r.get('rejected', False)]
        if valid_results:
            top_results = sorted(valid_results, key=lambda x: x['reward'], reverse=True)[:20]
            top_file = output_dir / f"top_20_{suffix}.json"
            with open(top_file, 'w') as f:
                json.dump(top_results, f, indent=2, default=str)


def log_milestone_stats(results: list, logger):
    """Log statistics at milestones."""
    valid_results = [r for r in results if not r.get('rejected', False)]
    rejected = len(results) - len(valid_results)

    if not valid_results:
        logger.info("No valid results yet")
        return

    rewards = [r['reward'] for r in valid_results]
    accuracies = [r['accuracy'] for r in valid_results]
    flops = [r['flops'] / 1e6 for r in valid_results]

    logger.info(f"Statistics ({len(valid_results)} valid, {rejected} rejected):")
    logger.info(f"  Reward: best={max(rewards):.3f}, mean={sum(rewards)/len(rewards):.3f}")
    logger.info(f"  Accuracy: best={max(accuracies):.3f}, mean={sum(accuracies)/len(accuracies):.3f}")
    logger.info(f"  FLOPs: min={min(flops):.2f}M, mean={sum(flops)/len(flops):.2f}M, max={max(flops):.2f}M")

    # Top 5
    top_5 = sorted(valid_results, key=lambda x: x['reward'], reverse=True)[:5]
    logger.info(f"  Top 5 architectures:")
    for i, r in enumerate(top_5, 1):
        logger.info(f"    {i}. Reward={r['reward']:.3f}, Acc={r['accuracy']:.3f}, FLOPs={r['flops']/1e6:.2f}M")


def log_final_stats(results: list, logger):
    """Log final statistics."""
    valid_results = [r for r in results if not r.get('rejected', False)]
    rejected = len(results) - len(valid_results)

    logger.info(f"\nFinal Statistics:")
    logger.info(f"  Total evaluated: {len(results)}")
    logger.info(f"  Valid: {len(valid_results)}")
    logger.info(f"  Rejected: {rejected} ({rejected/len(results)*100:.1f}%)")

    if valid_results:
        rewards = [r['reward'] for r in valid_results]
        accuracies = [r['accuracy'] for r in valid_results]
        flops = [r['flops'] / 1e6 for r in valid_results]

        logger.info(f"\nReward Statistics:")
        logger.info(f"  Best: {max(rewards):.3f}")
        logger.info(f"  Mean: {sum(rewards)/len(rewards):.3f}")
        logger.info(f"  Std: {np.std(rewards):.3f}")

        logger.info(f"\nAccuracy Statistics:")
        logger.info(f"  Best: {max(accuracies):.3f}")
        logger.info(f"  Mean: {sum(accuracies)/len(accuracies):.3f}")

        logger.info(f"\nFLOPs Statistics:")
        logger.info(f"  Min: {min(flops):.2f}M")
        logger.info(f"  Mean: {sum(flops)/len(flops):.2f}M")
        logger.info(f"  Max: {max(flops):.2f}M")

        # Check against FiLM target
        film_beaters = [r for r in valid_results if r['accuracy'] >= 0.46 and r['flops'] <= 6.29e6]
        logger.info(f"\nvs FiLM Baseline (46% acc, 6.29M FLOPs):")
        logger.info(f"  Architectures beating FiLM: {len(film_beaters)}")

        if film_beaters:
            best = max(film_beaters, key=lambda x: x['reward'])
            logger.info(f"  Best: Acc={best['accuracy']:.3f}, FLOPs={best['flops']/1e6:.2f}M")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Phase 4 Architecture Search')

    # Experiment settings
    parser.add_argument('--output-dir', type=str, default='./phase4_optimization/results/discovery',
                        help='Output directory for results')
    parser.add_argument('--num-iterations', type=int, default=200,
                        help='Number of search iterations')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save results every N iterations')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    # Evaluator settings
    parser.add_argument('--dataset', type=str, default='mmmu',
                        choices=['mmmu', 'ai2d', 'vsr', 'mathvista'])
    parser.add_argument('--num-shots', type=int, default=32)
    parser.add_argument('--train-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--backbone', type=str, default='clip-vit-l-14')
    parser.add_argument('--data-dir', type=str, default='./expv2/data')
    parser.add_argument('--early-stopping-enabled', action='store_true', default=True)
    parser.add_argument('--early-stopping-patience', type=int, default=3)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.005)
    parser.add_argument('--max-training-time', type=int, default=300,
                        help='Maximum training time per architecture (seconds)')

    # Reward settings
    parser.add_argument('--weight-accuracy', type=float, default=1.0)
    parser.add_argument('--weight-efficiency', type=float, default=1.5)
    parser.add_argument('--weight-compile-success', type=float, default=2.0)
    parser.add_argument('--weight-complexity', type=float, default=0.3)
    parser.add_argument('--flops-constraint-enabled', action='store_true', default=True)
    parser.add_argument('--max-flops', type=float, default=10e6,
                        help='Maximum FLOPs allowed (10M default)')
    parser.add_argument('--reject-if-exceed', action='store_true', default=True)
    parser.add_argument('--penalty-type', type=str, default='exponential')
    parser.add_argument('--penalty-scale', type=float, default=20e6)

    # Controller settings
    parser.add_argument('--population-size', type=int, default=50)
    parser.add_argument('--mutation-rate', type=float, default=0.3)
    parser.add_argument('--crossover-rate', type=float, default=0.5)

    # Generator settings
    parser.add_argument('--model', type=str, default='deepseek-chat')
    parser.add_argument('--temperature', type=float, default=0.7)

    args = parser.parse_args()

    # Run search
    results = run_architecture_search(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
