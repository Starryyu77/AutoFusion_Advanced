# Phase 2: Controller (RL Algorithm) Comparison

## Overview

Compare 6 different search algorithms while keeping the generator (best from Phase 1) and evaluator fixed.

## Search Algorithms

| Algorithm | Type | Key Feature | Config File |
|-----------|------|-------------|-------------|
| PPO | RL | Critic-Free variant | `configs/ppo.yaml` |
| GRPO | RL | Group-wise normalization + Bootstrap | `configs/grpo.yaml` |
| GDPO | RL | Decoupled normalization (Key Innovation) | `configs/gdpo.yaml` |
| Evolution | Evolutionary | Age-based regularization | `configs/evolution.yaml` |
| CMA-ES | Evolutionary | Covariance adaptation | `configs/cmaes.yaml` |
| Random | Baseline | Uniform sampling | `configs/random.yaml` |

## Quick Start

```bash
# Run all controller comparisons
./run.sh

# Or run specific controller
python3 run_single.py --controller gdpo --seed 42
```

## Experiment Design

- **Fixed**: Generator=(Best from Phase 1), Evaluator=Sandbox, Reward=Multi-Objective
- **Variable**: Controller Algorithm
- **Seeds**: 42, 123, 456, 789, 1024 (5 runs per algorithm)
- **Iterations**: 100 per run

## Key Research Question

**Can GDPO achieve better multi-objective optimization than PPO/GRPO?**

GDPO's decoupled normalization should better balance:
- Accuracy (high variance in early search)
- Efficiency (relatively stable)
- Compile Success (binary, can have near-zero variance)

## Expected Results

| Algorithm | Expected Behavior |
|-----------|-------------------|
| PPO | Baseline, may struggle with multi-objective balance |
| GRPO | Better than PPO, but compile_success variance may drown other signals |
| GDPO | Best multi-objective Pareto front (decoupled normalization advantage) |
| Evolution | Good exploration, slower convergence |
| CMA-ES | Sample efficient, may get stuck in local optima |
| Random | Lower bound baseline |

## Output

Results are saved to:
```
results/
├── ppo_s42/
├── ppo_s123/
├── grpo_s42/
├── gdpo_s42/
├── ...
```

## Analysis

After experiments complete, analyze with:
```bash
python3 ../scripts/analyze_controller_results.py \
    --input results/ \
    --output analysis/
```

This will generate:
- Pareto front plots (Accuracy vs FLOPs)
- Convergence curves
- Statistical significance tests
- Best architecture visualization
