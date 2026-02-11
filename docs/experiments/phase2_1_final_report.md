# Phase 2.1 Final Report: Controller Comparison with Fixes

**Date**: 2026-02-11
**Status**: ✅ Completed
**Experiments**: 6 controllers × 5 seeds = 30 runs

---

## Executive Summary

This report documents the final results of Phase 2.1, where we compared 6 different search algorithms with critical bug fixes applied. The key achievement was fixing type conversion issues that previously caused GDPO/GRPO to crash.

### Final Rankings

| Rank | Algorithm | Mean Reward | vs Random | Status |
|------|-----------|-------------|-----------|--------|
| 🥇 1 | **Evolution** | **9.80** | +208% | ✅ Excellent |
| 🥈 2 | **PPO** | **8.68** | +173% | ✅ Very Good |
| 🥉 3 | **GRPO** | **5.69** | +79% | ⚠️ Moderate |
| 4 | **GDPO** | **4.69** | +47% | ⚠️ Moderate |
| 5 | Random | 3.18 | - | Baseline |
| 6 | CMA-ES | 2.57 | -19% | ❌ Poor |

---

## Detailed Results

### Evolution (Winner)
| Seed | Iterations | Best Reward | Notes |
|------|------------|-------------|-------|
| 42 | 100 | **10.0** | Perfect score |
| 123 | 100 | 9.91 | Near perfect |
| 456 | 100 | **10.0** | Perfect score |
| 789 | 100 | **10.0** | Perfect score |
| 1024 | 100 | 9.09 | Very good |
| **Mean** | **100** | **9.80** | Most stable |

**Analysis**: Evolution achieved the highest and most consistent performance across all seeds. 4 out of 5 seeds reached the maximum reward of 10.0.

### PPO (Runner-up)
| Seed | Iterations | Best Reward | Notes |
|------|------------|-------------|-------|
| 42 | 100 | 8.82 | Very good |
| 123 | 100 | 9.76 | Excellent |
| 456 | 100 | **10.0** | Perfect score |
| 789 | 100 | **10.0** | Perfect score |
| 1024 | 100 | 4.83 | ⚠️ Anomaly |
| **Mean** | **100** | **8.68** | Good overall |

**Analysis**: PPO performed excellently on 4/5 seeds but had an anomaly on seed=1024 where it only reached 4.83. This may indicate sensitivity to random initialization.

### GRPO (Fixed ✅)
| Seed | Iterations | Best Reward | Notes |
|------|------------|-------------|-------|
| 42 | 100 | 5.62 | Moderate |
| 123 | 100 | 4.24 | Below average |
| 456 | 100 | **8.15** | Best GRPO result |
| 789 | 100 | 5.14 | Moderate |
| 1024 | 100 | 5.28 | Moderate |
| **Mean** | **100** | **5.69** | Runs successfully |

**Analysis**: After fixing the type conversion bug, GRPO successfully runs all 100 iterations. However, performance is moderate and may benefit from hyperparameter tuning.

### GDPO (Fixed ✅)
| Seed | Iterations | Best Reward | Notes |
|------|------------|-------------|-------|
| 42 | 100 | 1.93 | Poor |
| 123 | 100 | 5.90 | Moderate |
| 456 | 100 | 3.83 | Below average |
| 789 | 100 | 5.50 | Moderate |
| 1024 | 100 | **6.27** | Best GDPO result |
| **Mean** | **100** | **4.69** | Runs successfully |

**Analysis**: GDPO also runs successfully after the bug fix but shows the highest variance in performance. The decoupled normalization may need further tuning.

### Random (Baseline)
| Seed | Iterations | Best Reward | Notes |
|------|------------|-------------|-------|
| 42 | 39 | 3.12 | Early stop |
| 123 | 39 | 3.19 | Early stop |
| 456 | 39 | 3.21 | Early stop |
| 789 | 39 | 3.20 | Early stop |
| 1024 | 39 | 3.19 | Early stop |
| **Mean** | **39** | **3.18** | Baseline |

### CMA-ES
| Seed | Iterations | Best Reward | Notes |
|------|------------|-------------|-------|
| 42 | 50 | 2.36 | Poor |
| 123 | 38 | 3.15 | Below average |
| 456 | 42 | 2.91 | Poor |
| 789 | 41 | 2.19 | Very poor |
| 1024 | 39 | 2.26 | Very poor |
| **Mean** | **42** | **2.57** | Below baseline |

---

## Critical Bug Fix

### Problem
GDPO and GRPO were crashing due to YAML parsing issues where numeric values like `1e-5` were being parsed as strings instead of floats.

### Error Message
```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

### Solution
Added explicit `float()` conversion in controller initialization:

```python
# controllers/gdpo.py
self.learning_rate = float(config.get('learning_rate', 1e-5))
self.min_std = float(config.get('min_std', 1e-4))

# controllers/grpo.py
self.learning_rate = float(config.get('learning_rate', 1e-5))
self.variance_clip = float(config.get('variance_clip', 5.0))
```

### Result
- ✅ GDPO now runs all 100 iterations (previously crashed at ~30)
- ✅ GRPO now runs all 100 iterations (previously crashed at ~30)
- All RL algorithms now complete the full search process

---

## Applied Fixes Summary

### Fix 1: Disable Early Stopping
```yaml
controller:
  early_stopping:
    enabled: false
  max_iterations: 100
```
**Effect**: All RL algorithms now run full 100 iterations

### Fix 2: Exponential Reward Sharpening
```python
R = exp((Acc - baseline) * alpha)
# baseline=2.5, alpha=3.0
```
**Effect**: Better differentiation between good and bad architectures

### Fix 3: Stable Evaluation
```yaml
evaluator:
  quick_train:
    epochs: 10
  num_evaluations: 3
```
**Effect**: Reduced noise in reward estimates

### Fix 4: Penalty Decoupling (GDPO)
```python
# Compile failures get reward=0 (not negative)
# Prevents punishment from overwhelming other signals
```

### Fix 5: Type Conversion (Critical!)
```python
# Added float() conversion for all numeric config parameters
self.learning_rate = float(config.get('learning_rate', 1e-5))
```

---

## Key Findings

1. **Evolution is the most reliable**: Achieved highest mean reward with lowest variance
2. **PPO is strong but has outliers**: Seed=1024 anomaly suggests sensitivity issues
3. **GDPO/GRPO need tuning**: Both run successfully but underperform PPO/Evolution
4. **CMA-ES underperforms**: Even with fixes, performs below random baseline
5. **Fix 5 was critical**: Type conversion bug was blocking GDPO/GRPO entirely

---

## Recommendations

### Immediate Next Steps
1. **Run Phase 1**: Compare prompt strategies using fixed PPO controller
2. **Hyperparameter tuning**: Optimize GDPO/GRPO group_size, beta, learning_rate
3. **Investigate PPO seed=1024**: Understand why this seed performed poorly

### Long-term
1. **Ablation studies** (Phase 3): Test contribution of each fix
2. **Multi-dataset validation**: Verify results generalize
3. **Computational efficiency**: Compare wall-clock time per algorithm

---

## Files Generated

```
experiment/phase2_controllers/
├── results_phase21/           # All experiment results
│   ├── {controller}_s{seed}/
│   │   ├── summary.json
│   │   └── checkpoint.pt
├── logs_phase21/              # Training logs
│   └── {controller}_s{seed}.log
├── run_gdpo_grpo_fixed.sh     # New script for fixed experiments
└── configs/                   # Updated configs
    ├── ppo.yaml
    ├── grpo.yaml
    ├── gdpo.yaml
    └── ...
```

---

## Conclusion

Phase 2.1 successfully fixed critical bugs and established a working baseline for all algorithms. Evolution and PPO are the top performers, while GDPO/GRPO show promise but need hyperparameter optimization. The type conversion fix (Fix 5) was essential for enabling GDPO/GRPO to run at all.

**Phase 2.1 Status**: ✅ **COMPLETE**

---

*Report generated: 2026-02-11*
*Next: Phase 1 (Prompt Comparison)*
