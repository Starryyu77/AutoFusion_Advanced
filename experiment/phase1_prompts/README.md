# Phase 1: Prompt Strategy Comparison

## Overview

Compare 5 different prompt strategies while keeping the controller (PPO) and evaluator fixed.

## Prompt Strategies

| Strategy | Description | Config File |
|----------|-------------|-------------|
| CoT | Chain-of-Thought | `configs/cot.yaml` |
| FewShot | Example-based learning | `configs/fewshot.yaml` |
| Critic | Self-evaluation and refinement | `configs/critic.yaml` |
| Shape | Hard tensor dimension constraints | `configs/shape.yaml` |
| RolePlay | Expert persona simulation | `configs/roleplay.yaml` |

## Quick Start

```bash
# Run all prompt comparisons
./run.sh

# Or run specific strategy
python3 run_single.py --prompt cot --seed 42
```

## Experiment Design

- **Fixed**: Controller=PPO, Evaluator=Sandbox, Reward=Multi-Objective
- **Variable**: Generator Prompt Strategy
- **Seeds**: 42, 123, 456 (3 runs per strategy)
- **Iterations**: 100 per run

## Expected Results

The best prompt strategy will be identified based on:
1. Final best reward
2. Convergence speed
3. Code validity rate
4. Architecture diversity

## Output

Results are saved to:
```
results/
├── cot_s42/
├── cot_s123/
├── cot_s456/
├── fewshot_s42/
├── ...
```
