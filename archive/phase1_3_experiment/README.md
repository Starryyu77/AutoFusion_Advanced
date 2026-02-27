# Auto-Fusion Experiment Framework

## Overview

This framework provides a systematic way to compare RL algorithms, prompt strategies, and evaluation protocols for neural architecture search in multimodal fusion.

## Structure

```
experiment/
├── base/                       # Abstract base classes
│   ├── controller.py          # BaseController
│   ├── generator.py           # BaseGenerator
│   ├── evaluator.py           # BaseEvaluator
│   └── reward.py              # MultiObjectiveReward
├── controllers/                # Search algorithm implementations
│   ├── ppo.py                 # PPO + Critic-Free variant
│   ├── grpo.py                # GRPO + Bootstrap variance
│   ├── gdpo.py                # GDPO + variance protection
│   ├── evolution.py           # Evolutionary algorithm
│   ├── cmaes.py               # CMA-ES
│   └── random.py              # Random search baseline
├── generators/                 # Prompt strategy wrappers
│   ├── cot.py                 # Chain-of-Thought
│   ├── fewshot.py             # Few-Shot
│   ├── critic.py              # Critic
│   ├── shape.py               # Shape-Constraints
│   └── roleplay.py            # RolePlay
├── evaluators/                 # Evaluation wrappers
│   └── surgical_sandbox.py    # Surgical Sandbox
├── utils/                      # Utilities
│   ├── oom_handler.py         # OOM protection
│   └── rank_correlation.py    # Rank correlation validation
├── factory.py                  # Component factory
├── phase0_scaffold/           # End-to-end validation (PPO + CoT)
├── phase1_prompts/            # Prompt comparison experiments
├── phase2_controllers/        # RL algorithm comparison
├── phase3_ablation/           # Ablation studies
└── scripts/                   # Helper scripts
    ├── validate_rank_correlation.py
    └── monitor_gpu.sh
```

## Quick Start

### Phase 0: Scaffold Validation

Verify the entire pipeline works:

```bash
cd experiment/phase0_scaffold
./run.sh
```

Expected output:
```
===== Phase 0: Scaffold Validation =====
Step 1/5: Creating components .............. ✓
Step 2/5: Running search loop .............. ✓
Step 3/5: Results Summary .................. ✓
Step 4/5: Saving results ................... ✓
Step 5/5: Validation Complete .............. ✓
```

### Phase 1: Prompt Comparison

Compare 5 prompt strategies (fixed PPO controller):

```bash
cd experiment/phase1_prompts
./run.sh
```

Strategies: CoT, FewShot, Critic, Shape, RolePlay

### Phase 2: Controller Comparison

Compare 6 search algorithms (fixed best prompt from Phase 1):

```bash
cd experiment/phase2_controllers
./run.sh
```

Algorithms: PPO, GRPO, GDPO, Evolution, CMA-ES, Random

## Theory Corrections Implemented

### 1. GDPO Variance Explosion Fix

**Problem**: When `r_valid` variance → 0, normalization explodes

**Solution**:
```python
# In GDPO controller
if std < min_std:
    advantage = values - mean  # Don't divide by std
else:
    advantage = torch.clamp((values - mean) / std, -clip, clip)
```

### 2. PPO Critic-Free Mode

**Problem**: NAS is Contextual Bandit (T=1), not MDP. Critic is redundant.

**Solution**: `use_critic: false` uses Running Mean Baseline instead

### 3. Rank Correlation Validation

**Problem**: Proxy evaluation may not correlate with full evaluation

**Solution**: Mandatory Kendall's tau validation in Phase 0

```bash
python scripts/validate_rank_correlation.py \
    --config phase0_scaffold/config.yaml \
    --num-architectures 10
```

## Key Components

### Controllers

| Algorithm | Key Feature | Use Case |
|-----------|-------------|----------|
| **PPO** | Critic-Free variant | Baseline RL |
| **GRPO** | Bootstrap variance estimation | Group-wise normalization |
| **GDPO** | Decoupled normalization | Multi-objective optimization |
| **Evolution** | Age-based regularization | Discrete search spaces |
| **CMA-ES** | Covariance adaptation | Continuous optimization |
| **Random** | Uniform sampling | Lower bound baseline |

### Generators

| Strategy | Key Feature |
|----------|-------------|
| **CoT** | Step-by-step reasoning |
| **FewShot** | Example-based learning |
| **Critic** | Self-evaluation and refinement |
| **Shape** | Hard tensor constraints |
| **RolePlay** | Expert persona simulation |

## Usage Example

```python
from experiment import create_experiment_components

# Create components
components = create_experiment_components(
    controller_name='gdpo',
    generator_name='cot',
    evaluator_name='sandbox',
    config={
        'controller': {'group_size': 8, 'learning_rate': 1e-5},
        'generator': {'temperature': 0.7},
        'evaluator': {'quick_train_epochs': 5},
        'reward': {'weights': {'accuracy': 1.0, 'efficiency': 0.5}},
    },
)

# Run search
controller = components['controller']
generator = components['generator']
evaluator = components['evaluator']
reward_fn = components['reward']

for iteration in range(100):
    # 1. Controller proposes architecture
    proposal = controller.propose()

    # 2. Generator generates code
    results = generator.generate(proposal['architecture'])

    # 3. Evaluator evaluates code
    eval_result = evaluator.evaluate(results[0].code)

    # 4. Calculate reward
    reward = reward_fn.calculate(eval_result.to_dict())

    # 5. Update controller
    controller.update(reward)
    controller.record_iteration(proposal['architecture'], reward)
```

## OOM Protection

The framework includes comprehensive OOM protection:

```python
from experiment.utils import oom_retry

@oom_retry(max_retries=3, batch_reduction=0.5)
def train_model(config, model, data):
    # Training code here
    pass
```

## GPU Monitoring

```bash
# Monitor GPU memory
./scripts/monitor_gpu.sh [threshold_mb] [interval_sec]

# Example: Alert at 22GB, check every 5 seconds
./scripts/monitor_gpu.sh 22000 5
```

## Requirements

```bash
pip install torch torchvision pyyaml scipy numpy
```

Optional for LLM generation:
```bash
pip install openai  # For DeepSeek API
```

## Citation

```bibtex
@software{autofusion_experiment,
  title={Auto-Fusion: Self-Evolving Multi-Modal Neural Architecture Search},
  author={Auto-Fusion Team},
  year={2025}
}
```
