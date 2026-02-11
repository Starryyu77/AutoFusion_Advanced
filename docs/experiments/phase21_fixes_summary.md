# Phase 2.1 Fixes Summary

## Overview

基于 Phase 2 实验结果（Random 搜索优于 RL 方法），实施 4 个关键修复，旨在提升 RL 控制器的学习效果。

---

## Fix 1: 放宽早停，增加耐心

**问题:** Early stopping patience=20 可能过早终止，RL 还没跨过"探索低谷"

**修改:**
- `experiment/base/controller.py`:
  - 默认 `early_stop_patience` 从 20 改为 50
  - 新增 `disable_early_stop` 选项

- `experiment/phase2_controllers/configs/*.yaml`:
  - 所有控制器配置 `disable_early_stop: true`
  - 跑满 100 iterations

**预期效果:** 给 RL 足够的时间探索，进入"利用高峰"

---

## Fix 2: 锐化奖励函数 (Sharpen the Reward)

**问题:** 好坏架构奖励差距太小（都在 2.5-3.2 之间），RL 难以区分

**修改:**
- `experiment/base/reward.py`:
  - 新增 `ExponentialReward` 类
  - 公式: `R = exp((scalar - baseline) * alpha)`

**配置:**
```yaml
reward:
  type: exponential
  baseline: 2.5      # Random baseline
  alpha: 3.0         # 锐化系数
  max_sharpened: 10.0  # 防止爆炸
```

**示例:**
| 原始奖励 | 锐化后 | 说明 |
|---------|--------|------|
| 2.5 | 1.0 | baseline |
| 2.8 | 2.46 | +20% |
| 3.0 | 4.48 | +80% |
| 3.2 | 10.0 | capped |

**预期效果:** 好架构获得巨大奖励信号，RL 更容易学习

---

## Fix 3: 增加评估稳定性 (Reduce Noise)

**问题:** Surgical sandbox 评估噪声大，影响 RL 学习稳定性

**修改:**
- `experiment/evaluators/surgical_sandbox.py`:
  - 新增 `_evaluate_multiple()` 方法
  - 支持多次评估取平均

- `experiment/phase2_controllers/configs/*.yaml`:
  - `quick_train_epochs`: 5 → 10
  - `num_evals`: 1 → 3

**代码:**
```python
def evaluate(self, code, context=None, num_evals=3):
    """多次评估取平均"""
    results = []
    for i in range(num_evals):
        torch.manual_seed(42 + i)
        result = self._single_evaluate(code, context)
        results.append(result)

    # 返回平均值
    return EvaluationResult(
        accuracy=np.mean([r.accuracy for r in results]),
        ...
    )
```

**预期效果:** 减少评估噪声，让 RL 学到更稳定的策略

---

## Fix 4: 惩罚项解耦 (GDPO Only)

**问题:** 编译失败时给 -100 分，让 RL 认为"报错"是"世界末日"，不敢探索边界

**修改:**
- `experiment/base/reward.py`:
  - 编译失败时 `compile_success = 0` (而非 0.1)

- `experiment/controllers/gdpo.py`:
  - `_compute_decoupled_advantages()` 中特殊处理 `compile_success`
  - 编译失败的样本，该组件优势 = 0 (不解耦归一化)

**逻辑:**
```python
if key == 'compile_success':
    compile_failed = (values < 0.1)

    if compile_failed.any():
        # 失败的样本: 优势 = 0 (没得分，不是世界末日)
        normalized = torch.zeros_like(values)

        # 成功的样本正常归一化
        success_mask = ~compile_failed
        if success_mask.any():
            normalized[success_mask] = normalize(values[success_mask])
```

**预期效果:** RL 敢于在边界试探，不会因为偶尔失败而过度惩罚

---

## 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `experiment/base/controller.py` | 修改 | 放宽早停参数 |
| `experiment/base/reward.py` | 新增 | ExponentialReward 类 |
| `experiment/evaluators/surgical_sandbox.py` | 修改 | 多次评估取平均 |
| `experiment/controllers/gdpo.py` | 修改 | 编译失败惩罚解耦 |
| `experiment/factory.py` | 修改 | 支持 ExponentialReward |
| `experiment/phase2_controllers/configs/*.yaml` | 修改 | 应用所有修复配置 |
| `experiment/phase2_controllers/run_phase21.sh` | 新增 | Phase 2.1 运行脚本 |

---

## 运行 Phase 2.1

```bash
# SSH to server
ssh ntu-gpu43

# Navigate to experiment
cd ~/AutoFusion_Advanced/experiment/phase2_controllers

# Run Phase 2.1 with all fixes
bash run_phase21.sh

# Or run single experiment
python3 run_experiment.py ppo 42 2
```

---

## 预期结果

| 指标 | Phase 2 | Phase 2.1 (预期) |
|------|---------|------------------|
| PPO mean reward | 2.99 | > 3.0 |
| RL vs Random | -6.1% | +5% |
| 收敛稳定性 | 低 | 高 |
| 编译失败率 | 中等 | 降低 |

---

## 如果仍不理想？

如果 Phase 2.1 仍无法让 RL 超过 Random，考虑：

1. **进一步增加探索:**
   - 增大 entropy coefficient
   - 使用更高的 temperature

2. **调整搜索空间:**
   - 增加架构复杂度
   - 添加更多约束

3. **更长训练:**
   - max_iterations: 100 → 200
   - 更多 seeds

4. **真实数据评估:**
   - 使用真实数据集替代 mock data
   - 端到端训练验证
