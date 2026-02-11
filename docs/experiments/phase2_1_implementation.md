# Phase 2.1 Implementation: 4 Critical Fixes

## Overview

Based on Phase 2 results (Random search outperformed RL methods), we implemented 4 key fixes to improve RL controller performance.

---

## Fix 1: Relax Early Stopping (放宽早停)

**Problem:** Early stopping at 20 iterations may cut off RL before it escapes the "exploration valley" and reaches the "exploitation peak".

**Solution:**
- Increased `early_stop_patience` from 20 to 50
- Added `disable_early_stop: true` option to run full 100 iterations

**Files Modified:**
- `experiment/base/controller.py` - Updated default patience and added disable option
- `experiment/phase2_controllers/configs/*.yaml` - All 6 configs updated

**Code Changes:**
```python
# In BaseController.__init__
self.early_stop_patience = config.get('early_stop_patience', 50)  # Was 20
self.disable_early_stop = config.get('disable_early_stop', False)

# In should_stop()
if self.disable_early_stop:
    return False  # Run full iterations
```

---

## Fix 2: Sharpen Reward Function (锐化奖励函数)

**Problem:** All architectures scored around 3.1, making it hard for RL to distinguish good from bad.

**Solution:** Implemented ExponentialReward with exponential scaling.

**Formula:** `R = exp((Acc - Baseline) × α)`

**Example:**
- Baseline = 2.5, α = 3.0
- Acc = 3.0 → exp(0.5×3) = exp(1.5) = **4.48**
- Acc = 2.8 → exp(0.3×3) = exp(0.9) = **2.46**
- Acc = 2.5 → exp(0×3) = **1.0**

**Files Modified:**
- `experiment/base/reward.py` - Added `ExponentialReward` class
- `experiment/factory.py` - Updated `create_reward()` to support types
- `experiment/phase2_controllers/configs/*.yaml` - All configs use exponential reward

**Code:**
```python
class ExponentialReward(MultiObjectiveReward):
    def __init__(self, config):
        super().__init__(config)
        self.baseline = 2.5  # Random baseline
        self.alpha = 3.0     # Sharpening factor

    def calculate(self, evaluation_result):
        components = super().calculate(evaluation_result)
        scalar = components.to_scalar(self.weights)
        sharpened = math.exp((scalar - self.baseline) * self.alpha)
        sharpened = min(sharpened, self.max_sharpened)  # Clip to prevent explosion
        # Redistribute proportionally...
```

---

## Fix 3: Increase Evaluation Stability (增加评估稳定性)

**Problem:** High noise in surgical sandbox evaluation makes it hard for RL to learn consistent policies.

**Solution:**
- Option A: Increased `quick_train_epochs` from 5 to 10
- Option B: Multiple evaluations (default 3) with averaging

**Files Modified:**
- `experiment/evaluators/surgical_sandbox.py` - Added `_evaluate_multiple()` method
- `experiment/phase2_controllers/configs/*.yaml` - Updated evaluator configs

**Code:**
```python
def evaluate(self, code, context=None, num_evals=1):
    if num_evals > 1:
        return self._evaluate_multiple(code, context, num_evals)
    # ... single eval

def _evaluate_multiple(self, code, context, num_evals):
    results = []
    for i in range(num_evals):
        torch.manual_seed(42 + i)  # Different seed each time
        result = self.evaluate(code, context, num_evals=1)
        results.append(result)

    # Return averaged results
    return EvaluationResult(
        accuracy=np.mean([r.accuracy for r in results]),
        efficiency=np.mean([r.efficiency for r in results]),
        ...
    )
```

---

## Fix 4: Decouple Penalty for GDPO (惩罚项解耦)

**Problem:** Treating compilation failure as -100 (extreme negative) discourages exploration. RL should understand "error" means "no score", not "end of world".

**Solution:**
- Compilation failure → `compile_success = 0` (no penalty, just no reward)
- In GDPO, compile_success is normalized separately
- Failed compilations get advantage = 0 for that component only

**Files Modified:**
- `experiment/base/reward.py` - Changed label smoothing: 0 → 0.0 (was 0.1)
- `experiment/controllers/gdpo.py` - Special handling for compile_success in `_compute_decoupled_advantages()`

**Code:**
```python
# In MultiObjectiveReward.calculate()
if self.label_smoothing:
    # Compile failure = 0 (no score), not extreme negative
    compile_success = 0.0 if compile_success < 0.5 else 0.9

# In GDPO._compute_decoupled_advantages()
if key == 'compile_success':
    compile_failed = (values < 0.1).float()
    if compile_failed.sum() > 0:
        # Failed compilations: advantage = 0 (neutral)
        # Successful: normal normalization
        normalized = torch.zeros_like(values)
        success_mask = (values >= 0.1)
        if success_mask.sum() > 0:
            # Normalize only successful ones
            ...
```

---

## Configuration Summary

All Phase 2.1 configs updated:

```yaml
controller:
  disable_early_stop: true      # Fix 1
  early_stop_patience: 50       # Fix 1

evaluator:
  quick_train_epochs: 10        # Fix 3
  num_evals: 3                  # Fix 3

reward:
  type: exponential             # Fix 2
  baseline: 2.5                 # Fix 2
  alpha: 3.0                    # Fix 2
  max_sharpened: 10.0           # Fix 2
```

---

## Expected Outcomes

| Fix | Expected Improvement |
|-----|---------------------|
| Fix 1 | RL has time to escape local optima |
| Fix 2 | Good architectures get much higher rewards |
| Fix 3 | More stable learning signal |
| Fix 4 | RL explores more freely without fear of extreme penalties |

**Combined:** RL methods should outperform Random baseline in Phase 2.1

---

## Testing

Run Phase 2.1 experiments:

```bash
cd experiment/phase2_controllers
python3 run_experiment.py ppo 42 2  # Test single run
bash run_all.sh                      # Full comparison
```

Compare with Phase 2 results to measure improvement.
