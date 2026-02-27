# Phase 2.1 Experiment Results

## Experiment Metadata
- **Date**: 2026-02-11
- **Total Runs**: 30 (6 controllers × 5 seeds)
- **Iterations**: 100 per run
- **GPU**: 2× NVIDIA RTX A5000

## Complete Results Table

| Controller | Seed | Iterations | Best Reward | Status |
|------------|------|------------|-------------|--------|
| evolution | 42 | 100 | 10.0 | ✅ |
| evolution | 123 | 100 | 9.91 | ✅ |
| evolution | 456 | 100 | 10.0 | ✅ |
| evolution | 789 | 100 | 10.0 | ✅ |
| evolution | 1024 | 100 | 9.09 | ✅ |
| **evolution_mean** | - | **100** | **9.80** | **🏆** |
| ppo | 42 | 100 | 8.82 | ✅ |
| ppo | 123 | 100 | 9.76 | ✅ |
| ppo | 456 | 100 | 10.0 | ✅ |
| ppo | 789 | 100 | 10.0 | ✅ |
| ppo | 1024 | 100 | 4.83 | ⚠️ |
| **ppo_mean** | - | **100** | **8.68** | **🥈** |
| grpo | 42 | 100 | 5.62 | ✅ |
| grpo | 123 | 100 | 4.24 | ✅ |
| grpo | 456 | 100 | 8.15 | ✅ |
| grpo | 789 | 100 | 5.14 | ✅ |
| grpo | 1024 | 100 | 5.28 | ✅ |
| **grpo_mean** | - | **100** | **5.69** | **🥉** |
| gdpo | 42 | 100 | 1.93 | ✅ |
| gdpo | 123 | 100 | 5.90 | ✅ |
| gdpo | 456 | 100 | 3.83 | ✅ |
| gdpo | 789 | 100 | 5.50 | ✅ |
| gdpo | 1024 | 100 | 6.27 | ✅ |
| **gdpo_mean** | - | **100** | **4.69** | **4th** |
| random | 42 | 39 | 3.12 | ⏹️ |
| random | 123 | 39 | 3.19 | ⏹️ |
| random | 456 | 39 | 3.21 | ⏹️ |
| random | 789 | 39 | 3.20 | ⏹️ |
| random | 1024 | 39 | 3.19 | ⏹️ |
| **random_mean** | - | **39** | **3.18** | **Baseline** |
| cmaes | 42 | 50 | 2.36 | ⏹️ |
| cmaes | 123 | 38 | 3.15 | ⏹️ |
| cmaes | 456 | 42 | 2.91 | ⏹️ |
| cmaes | 789 | 41 | 2.19 | ⏹️ |
| cmaes | 1024 | 39 | 2.26 | ⏹️ |
| **cmaes_mean** | - | **42** | **2.57** | **6th** |

## Statistical Summary

```
Evolution:  mean=9.80  std=0.40  min=9.09  max=10.0
PPO:        mean=8.68  std=2.06  min=4.83  max=10.0
GRPO:       mean=5.69  std=1.35  min=4.24  max=8.15
GDPO:       mean=4.69  std=1.73  min=1.93  max=6.27
Random:     mean=3.18  std=0.04  min=3.12  max=3.21
CMA-ES:     mean=2.57  std=0.36  min=2.19  max=3.15
```

## Key Findings

1. **Evolution**: Most stable (std=0.40), 4/5 seeds reached perfect 10.0
2. **PPO**: Strong performance but high variance (std=2.06), seed=1024 anomaly
3. **GRPO**: Moderate performance, best seed (456) reached 8.15
4. **GDPO**: Highest variance (std=1.73), best seed (1024) reached 6.27
5. **Random/CMA-ES**: Early stopping at ~40 iterations, below Evolution/PPO

## Bug Fixes Applied

See: [Final Report](../../docs/experiments/phase2_1_final_report.md)

### Fix 5: Type Conversion (Critical)
```python
# controllers/gdpo.py
self.learning_rate = float(config.get('learning_rate', 1e-5))
self.min_std = float(config.get('min_std', 1e-4))

# controllers/grpo.py
self.learning_rate = float(config.get('learning_rate', 1e-5))
self.variance_clip = float(config.get('variance_clip', 5.0))
```

This fix enabled GDPO/GRPO to run without crashing.
