# Phase 0: Scaffold Validation (PPO + CoT)

## Overview

This is the **end-to-end validation phase** for the Auto-Fusion experiment framework. The goal is to verify that the entire pipeline works correctly before running full experiments.

## Pipeline Architecture

```
Controller (PPO) → Generator (CoT) → Evaluator (Surgical Sandbox) → Reward → Update
```

## Configuration

- **Controller**: PPO (Critic-Free mode for Contextual Bandit)
- **Generator**: Chain-of-Thought (CoT)
- **Evaluator**: Surgical Sandbox (5 epochs quick training)
- **Iterations**: 50 (fast validation)

## Quick Start

```bash
# Run the scaffold test
./run.sh
```

## Expected Output

```
===== Phase 0: Scaffold Validation =====
Step 1/5: API Connectivity .................. ✓
Step 2/5: Generator (CoT) ................... ✓
Step 3/5: Evaluator (Sandbox) ............... ✓
Step 4/5: Reward System ..................... ✓
Step 5/5: Full Pipeline (50 iterations) ..... ✓

===== Validation Complete =====
Best Reward: X.XXXX
Best Architecture: {...}
```

## Success Criteria

| Check | Pass Criteria | Status |
|-------|--------------|--------|
| API Connectivity | DeepSeek returns non-empty | ⬜ |
| Generator | Generates valid Python code | ⬜ |
| Evaluator | Compile + train without error | ⬜ |
| Reward | Returns {acc, eff, valid} dict | ⬜ |
| Pipeline | 50 iterations without crash | ⬜ |

## Files

- `config.yaml` - Experiment configuration
- `run.sh` - Main execution script
- `README.md` - This file
