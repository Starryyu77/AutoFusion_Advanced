# Phase 5.6: 扩展搜索与基准对比计划

**版本**: 1.0
**创建日期**: 2026-03-02
**目标**: 增加迭代次数、优化评估配置、与 FiLM baseline 对比

---

## 一、背景

### Phase 5.5 成果
- 编译成功率: **100%** ✅ (目标 60%+)
- 最佳 Reward: **3.913** ✅ (超过 Phase 5 的 2.796)
- 最佳架构: Hybrid (attention + gating)
- 有效架构数: 300 个 ✅

### 未达成目标
- 准确率: ~40% (目标 50%+, 超越 FiLM 46%)
- DeepSeek 未完成 (API 超时)

---

## 二、改进方案

### 方案 B: 增加迭代次数

**目标**: 发现更高 Reward 的架构

| 配置 | Phase 5.5 | Phase 5.6 |
|------|-----------|-----------|
| 迭代次数 | 100 | **200** |
| 模型 | 3 个 | 2 个最佳模型 |
| 预计时间 | ~40 分钟 | ~80 分钟 |

**选择最佳模型**:
- Kimi K2.5 (Reward 3.913, 速度最快 39.6分钟)
- Qwen-Max (Reward 3.913, 40.3分钟)

### 方案 C: 优化评估配置

**目标**: 提高评估稳定性和准确率

| 配置 | Phase 5.5 | Phase 5.6 |
|------|-----------|-----------|
| num_shots | 64 | **128** |
| train_epochs | 10 | **15** |
| 评估时间 | ~20s | ~40s |

**预期效果**:
- 评估更稳定
- 准确率更高
- 更好的架构区分度

### 方案 D: Baseline 对比

**目标**: 与 FiLM (人工设计) 对比

| 架构 | 类型 | FLOPs | 准确率 |
|------|------|-------|--------|
| FiLM | 人工设计 | 6.29M | 46% |
| Phase 5.5 最佳 | LLM 生成 | 4.8-8.3M | ~40% |

**对比维度**:
1. 准确率 (MMMU)
2. FLOPs (效率)
3. 参数量
4. 推理延迟
5. 架构复杂度

---

## 三、实验设计

### 3.1 实验矩阵

| 实验 ID | 模型 | 迭代 | shots | epochs | 目标 |
|---------|------|------|-------|--------|------|
| exp_kimi_v3 | Kimi K2.5 | 200 | 128 | 15 | 高 Reward |
| exp_qwen_v3 | Qwen-Max | 200 | 128 | 15 | 高 Reward |
| exp_deepseek_v3 | DeepSeek V3.2 | 100 | 128 | 15 | 完成对比 |
| baseline_film | FiLM | - | 128 | 15 | 基准对比 |

### 3.2 评估流程

```
┌────────────────────────────────────────────────────────────────┐
│                        评估流程                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. NAS 搜索 (200 iterations)                                  │
│      └─→ 生成架构候选                                           │
│                                                                 │
│   2. 初步评估 (128 shots, 15 epochs)                            │
│      └─→ 筛选 Top 10 架构                                       │
│                                                                 │
│  3. 完整评估 (256 shots, 30 epochs)                             │
│      └─→ 准确率、FLOPs、延迟                                    │
│                                                                 │
│   4. Baseline 对比                                              │
│      └─→ 与 FiLM 在相同条件下对比                               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 Baseline 对比实验

**评估配置统一**:
- 数据集: MMMU
- shots: 128
- epochs: 15 (初步), 30 (完整)
- batch_size: 8
- early_stopping: patience=5

**对比架构**:
1. FiLM (人工设计)
2. CLIPFusion (简单)
3. ConcatMLP (简单)
4. Phase 5.6 最佳架构

---

## 四、实现步骤

### Step 1: 创建新配置文件

**文件**: `configs/v3/exp_kimi_v3.yaml`

```yaml
# Kimi K2.5 扩展搜索
experiment:
  name: exp_kimi_v3
  max_iterations: 200
  save_interval: 20
  output_dir: ./results/v3/exp_kimi

llm:
  type: aliyun
  model: kimi-k2.5
  api_key: ${ALIYUN_API_KEY}
  temperature: 0.7
  max_tokens: 2048

improvements:
  use_template_mode: true
  use_error_feedback: true
  max_retries: 3

evaluator:
  dataset: mmmu
  num_shots: 128
  train_epochs: 15
  batch_size: 8
  early_stopping:
    enabled: true
    patience: 5
    min_delta: 0.005
  max_training_time: 600

reward:
  weights:
    accuracy: 1.0
    efficiency: 1.5
    compile_success: 2.0

constraints:
  constraints:
    max_flops: 10000000
    max_params: 50000000
    target_accuracy: 0.5
```

### Step 2: 修改评估器支持 Baseline

**文件**: `src/v3/baseline_evaluator.py`

```python
"""Baseline 评估器 - 与人工设计架构对比"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
import sys
sys.path.insert(0, '/usr1/home/s125mdg43_10/AutoFusion_Advanced')

# 导入 FiLM baseline
from archive.expv2_evaluation.shared.baselines.film import FiLMFusion
from archive.expv2_evaluation.shared.baselines.clip_fusion import CLIPFusion
from archive.expv2_evaluation.shared.baselines.concat_mlp import ConcatMLP

class BaselineEvaluator:
    """人工设计架构评估器"""
    
    BASELINES = {
        'film': FiLMFusion,
        'clip_fusion': CLIPFusion,
        'concat_mlp': ConcatMLP,
    }
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.results = {}
    
    def evaluate_baseline(self, name: str) -> Dict[str, Any]:
        """评估单个 baseline"""
        print(f"\n评估 Baseline: {name}")
        
        # 创建 baseline 模型
        model = self.BASELINE[name]()
        
        # 评估
        result = self.evaluator.evaluate_model(model)
        
        self.results[name] = result
        return result
    
    def evaluate_all_baselines(self) -> Dict[str, Dict]:
        """评估所有 baselines"""
        for name in self.BASELINE:
            self.evaluate_baseline(name)
        
        return self.results
    
    def compare_with_discovered(self, discovered_results: Dict) -> str:
        """生成对比报告"""
        lines = ["# Baseline 对比报告", ""]
        lines.append("## 准确率对比")
        lines.append("")
        lines.append("| 架构 | 类型 | 准确率 | FLOPs |")
        lines.append("|------|------|--------|-------|")
        
        # Baselines
        for name, result in self.results.items():
            acc = result.get('accuracy', 0) * 100
            flops = result.get('flops', 0) / 1e6
            lines.append(f"| {name} | Baseline | {acc:.1f}% | {flops:.1f}M |")
        
        # Discovered
        for name, result in discovered_results.items():
            acc = result.get('accuracy', 0) * 100
            flops = result.get('flops', 0) / 1e6
            lines.append(f"| {name} | Discovered | {acc:.1f}% | {flops:.1f}M |")
        
        return "\n".join(lines)
```

### Step 3: 主脚本更新

**文件**: `src/v3/run_v3.py`

```python
"""Phase 5.6 主脚本 - 扩展搜索 + Baseline 对比"""

import os
import sys
import yaml
import logging

sys.path.insert(0, '/usr1/home/s125mdg43_10/AutoFusion_Advanced')
sys.path.insert(0, '/usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--compare-baselines', action='store_true')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    exp_config = config['experiment']
    eval_config = config.get('evaluator', {})
    
    logger.info("=" * 60)
    logger.info("Phase 5.6: Extended Search + Baseline Comparison")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_config['name']}")
    logger.info(f"Iterations: {exp_config['max_iterations']}")
    logger.info(f"Shots: {eval_config.get('num_shots', 64)}")
    logger.info(f"Epochs: {eval_config.get('train_epochs', 10)}")
    
    # 1. NAS 搜索
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: NAS Search")
    logger.info("=" * 60)
    
    from src.v2.main_loop_v2 import NASControllerV2
    from src.llm_backend import AliyunBailianBackend
    from phase4_optimization.src.evaluator_v2_improved import ImprovedRealDataFewShotEvaluator
    from phase4_optimization.src.reward_v2 import ConstrainedReward
    
    # 初始化组件
    llm_config = config['llm']
    api_key = os.environ.get('ALIYUN_API_KEY', llm_config.get('api_key', ''))
    
    llm = AliyunBailianBackend(
        api_key=api_key,
        model=llm_config.get('model', 'kimi-k2.5'),
        temperature=llm_config.get('temperature', 0.7),
        max_tokens=llm_config.get('max_tokens', 2048),
    )
    
    evaluator = ImprovedRealDataFewShotEvaluator(eval_config)
    reward_fn = ConstrainedReward(config.get('reward', {}))
    
    controller = NASControllerV2(
        llm_backend=llm,
        evaluator=evaluator,
        reward_fn=reward_fn,
        use_template_mode=True,
        use_error_feedback=True,
        max_retries=3,
        output_dir=args.output_dir or exp_config.get('output_dir', './results')
    )
    
    # 运行搜索
    best_result = controller.search(
        max_iterations=exp_config.get('max_iterations', 200),
        save_interval=exp_config.get('save_interval', 20)
    )
    
    # 2. Baseline 对比
    if args.compare_baselines:
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: Baseline Comparison")
        logger.info("=" * 60)
        
        from src.v3.baseline_evaluator import BaselineEvaluator
        
        baseline_eval = BaselineEvaluator(evaluator)
        baseline_results = baseline_eval.evaluate_all_baselines()
        
        # 生成对比报告
        discovered = {'kimi_best': best_result} if best_result else {}
        report = baseline_eval.compare_with_discovered(discovered)
        
        report_path = os.path.join(args.output_dir or '.', 'baseline_comparison.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Baseline comparison saved to: {report_path}")
    
    logger.info("=" * 60)
    logger.info("Experiment completed!")
    if best_result:
        logger.info(f"Best Reward: {best_result.reward}")
        logger.info(f"Best Architecture: {best_result.template}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
```

---

## 五、执行计划

### Week 1: 扩展搜索 (2026-03-02 ~ 2026-03-08)

| Day | 任务 | GPU | 模型 |
|-----|------|-----|------|
| Day 1-2 | Kimi K2.5 扩展 (200 iter) | GPU 2 | Kimi K2.5 |
| Day 3-4 | Qwen-Max 扩展 (200 iter) | GPU 3 | Qwen-Max |
| Day 5-6 | DeepSeek V3.2 重跑 (100 iter) | GPU 2 | DeepSeek V3.2 |
| Day 7 | 结果汇总与分析 | - | - |

### Week 2: Baseline 对比 (2026-03-09 ~ 2026-03-15)

| Day | 任务 |
|-----|------|
| Day 1-2 | 实现 BaselineEvaluator |
| Day 3-4 | 运行 FiLM 等 baseline 评估 |
| Day 5-6 | 完整对比分析 |
| Day 7 | 撰写最终报告 |

---

## 六、预期成果

### 6.1 最低目标

- [ ] Kimi/Qwen 扩展搜索完成 (200 iterations)
- [ ] 编译成功率 ≥ 100%
- [ ] 发现至少 1 个准确率 > 45% 的架构

### 6.2 理想目标

- [ ] 最佳准确率 > 50% (超越 FiLM)
- [ ] FLOPs < 5M
- [ ] 完成与所有 baselines 的对比
- [ ] 生成完整的对比报告

---

## 七、文件清单

### 需要创建的文件

```
phase5_llm_rl/
├── configs/v3/
│   ├── exp_kimi_v3.yaml
│   ├── exp_qwen_v3.yaml
│   └── exp_deepseek_v3.yaml
├── src/v3/
│   ├── __init__.py
│   ├── baseline_evaluator.py
│   └── run_v3.py
└── results/v3/
    └── (运行时创建)
```

### 需要更新的文件

```
docs/
├── experiments/
│   ├── PHASE5.6_RESULTS.md (新建)
│   └── BASELINE_COMPARISON.md (新建)
└── PROJECT_STATUS.md (更新)
```

---

## 八、下一步行动

### 立即执行

1. **创建配置文件** - 更新迭代次数和评估参数
2. **实现 BaselineEvaluator** - 与 FiLM 等对比
3. **更新主脚本** - 支持扩展搜索和 baseline 对比

### 是否开始执行？

请确认：
1. 是否同意这个计划？
2. 是否需要调整参数？
3. 是否现在开始创建文件？