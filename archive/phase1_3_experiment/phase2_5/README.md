# Phase 2.5: Evaluator Verification and Calibration

This phase validates the RealDataFewShotEvaluator to ensure it reliably predicts real-world performance.

## Overview

**Goal**: Find the best (dataset, training depth) combination for the evaluator.

**Three sub-experiments**:
1. **2.5.1**: Dataset Selection - Which dataset best predicts final performance?
2. **2.5.2**: Training Depth Calibration - How many epochs are most cost-effective?
3. **2.5.3**: Architecture Fairness - Does the evaluator treat all architecture types fairly?

## Directory Structure

```
phase2_5/
├── README.md                           # This file
├── run_2_5_1_dataset_selection.py     # Experiment 2.5.1
├── run_2_5_2_training_depth.py        # Experiment 2.5.2
├── run_2_5_3_architecture_fairness.py # Experiment 2.5.3
├── run_all.py                         # Run all experiments
├── results/                           # Experiment results
│   ├── 2_5_1_dataset_selection/
│   ├── 2_5_2_training_depth/
│   └── 2_5_3_architecture_fairness/
└── configs/                           # Recommended configs
    └── evaluator_recommended.yaml
```

## Quick Start

### Run Individual Experiments

```bash
# Experiment 2.5.1: Dataset Selection (Tests 4 datasets)
python experiment/phase2_5/run_2_5_1_dataset_selection.py

# Experiment 2.5.2: Training Depth (Tests 1/3/5/10 epochs)
python experiment/phase2_5/run_2_5_2_training_depth.py

# Experiment 2.5.3: Architecture Fairness (Tests 5 architecture types)
python experiment/phase2_5/run_2_5_3_architecture_fairness.py
```

### Run All Experiments

```bash
python experiment/phase2_5/run_all.py
```

## Evaluation Criteria

### 1. Ranking Correlation (Kendall's τ)

Measures how well the evaluator's ranking matches the ground truth.

```
τ > 0.8: Excellent (fully trustworthy)
τ > 0.7: Good (usable)
τ > 0.5: Fair (use with caution)
τ < 0.5: Poor (must adjust configuration)
```

### 2. Discriminative Power

Measures ability to distinguish good from bad architectures.

```
Good evaluator: Top architecture score >> Bottom architecture score
Poor evaluator: Scores are too close to differentiate
```

### 3. Cost Efficiency

```
Efficiency = τ / Time

Example:
- 5 epochs: τ=0.78, Time=5min → Efficiency=0.156
- 10 epochs: τ=0.81, Time=10min → Efficiency=0.081

→ 5 epochs is more cost-effective
```

## Expected Outputs

Each experiment generates:
- `results.json`: Raw results and metrics
- `summary.json`: Key findings and recommendations
- `correlation_plot.png`: Visualization (if matplotlib available)

## Timeline

| Experiment | Duration | Parallel GPUs |
|------------|----------|---------------|
| 2.5.1 Dataset Selection | ~2 days | 4 GPUs |
| 2.5.2 Training Depth | ~1 day | 4 GPUs |
| 2.5.3 Architecture Fairness | ~1 day | 3 GPUs |
| **Total** | **~4 days** | - |

## Success Criteria

Phase 2.5 is successful when:
1. ✓ At least one dataset achieves τ > 0.7
2. ✓ Optimal training depth is identified
3. ✓ No significant bias against any architecture type
4. ✓ Final recommended configuration is generated

## Final Output

`configs/evaluator_recommended.yaml`:
```yaml
recommended:
  dataset: mmmu  # or best performing
  num_shots: 16
  train_epochs: 5
  expected_tau: 0.78
```

---

*Part of AutoFusion Phase 2.5*
