# Auto-Fusion Experiments Documentation

## Overview

This directory contains comprehensive documentation for all Auto-Fusion experiments conducted on the NTU EEE GPU Cluster.

## Experiment Phases

### Phase 0: Scaffold Validation
**Status:** ✅ Completed

End-to-end validation of the complete pipeline (PPO + CoT).

- **Purpose:** Verify all components work together
- **Duration:** ~5 minutes
- **Results:** Pipeline validated successfully

[View Details](phase0_scaffold/)

---

### Phase 1: Prompt Strategy Comparison
**Status:** ⏳ Pending

Compare 5 prompt strategies with fixed PPO controller.

- **Controllers:** PPO (fixed)
- **Generators:** CoT, FewShot, Critic, Shape, RolePlay
- **Seeds:** 5 seeds per strategy
- **Total Runs:** 25 experiments

[View Details](phase1_prompts/)

---

### Phase 2: Controller Comparison ⭐
**Status:** ✅ Completed (February 11, 2025)

Comprehensive comparison of 6 search algorithms with fixed CoT generator.

**Key Results:**
| Rank | Controller | Mean Reward | Best Seed |
|------|------------|-------------|-----------|
| 1 | Random | 3.1831 | 3.2123 |
| 2 | PPO | 2.9896 | 3.2257 |
| 3 | Evolution | 2.7709 | 3.2234 |
| 4 | GDPO | 2.6553 | 2.8961 |
| 5 | CMA-ES | 2.5749 | 3.1490 |
| 6 | GRPO | 2.5067 | 2.6768 |

**Key Finding:** Random search outperformed RL methods, suggesting search space may be too simple or evaluation too noisy.

[View Full Report](phase2_report.md)

---

### Phase 3: Ablation Studies
**Status:** ⏳ Pending

Systematic ablation of key components.

Planned studies:
1. GDPO variance protection ablation
2. PPO Critic-Free vs standard PPO
3. Different group sizes for GRPO/GDPO
4. Reward weight sensitivity

[View Details](phase3_ablation/)

---

## Quick Reference

### Running Experiments

```bash
# Phase 0 - Validation
cd experiment/phase0_scaffold
bash run.sh

# Phase 2 - Controller Comparison
cd experiment/phase2_controllers
python3 run_experiment.py ppo 42 2  # single run
bash run_all.sh                      # all controllers
```

### Results Location

```
experiment/phase2_controllers/
├── results/          # Experiment outputs
│   └── {controller}_s{seed}/
│       ├── checkpoint.pt
│       ├── summary.json
│       └── run.py
├── logs/             # Training logs
└── controller_comparison.png
```

### Analysis Tools

```python
# Load results
import json
from pathlib import Path

result_file = Path('results/ppo_s42/summary.json')
with open(result_file) as f:
    data = json.load(f)

print(f"Best reward: {data['stats']['best_reward']}")
print(f"Iterations: {data['stats']['iteration']}")
```

---

## Theory Corrections

### 1. GDPO Variance Explosion Fix

**Problem:** When `r_valid` variance → 0, normalization explodes

**Solution:**
```python
if std < min_std:
    advantage = values - mean  # Don't divide
else:
    advantage = (values - mean) / std
```

### 2. PPO Critic-Free Mode

**Problem:** NAS is Contextual Bandit (T=1), not MDP

**Solution:** Use Running Mean Baseline instead of Critic network

### 3. Rank Correlation Validation

**Problem:** Proxy evaluation may not correlate with full evaluation

**Solution:** Mandatory Kendall's tau validation (τ > 0.7)

---

## Hardware Environment

- **Server:** NTU GPU43
- **GPUs:** 4x NVIDIA RTX A5000 (24GB each)
- **CPU:** AMD EPYC 7543 32-Core
- **RAM:** 251GB
- **Storage:** /usr1 (3.2TB available)

---

## Citation

```bibtex
@software{autofusion_experiments,
  title={Auto-Fusion: Multi-Modal Neural Architecture Search Experiments},
  author={Auto-Fusion Team},
  year={2025},
  institution={NTU EEE}
}
```
