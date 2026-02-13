# Phase 3: Architecture Discovery Experiment Plan

**Date:** 2026-02-13
**Status:** In Progress

---

## 1. 实验目标

使用验证后的最佳组件组合，自动发现高性能的多模态融合架构：
- **Controller**: Evolution (Phase 2.1 Winner)
- **Generator**: FewShot (Phase 1 Winner)
- **Evaluator**: RealDataFewShotEvaluator (Phase 2.5 Verified)

---

## 2. 核心配置

```python
PHASE3_CONFIG = {
    # Controller (Evolution - Phase 2.1 Winner)
    'controller': 'EvolutionController',
    'population_size': 50,
    'num_iterations': 100,
    'mutation_rate': 0.3,
    'crossover_rate': 0.5,

    # Generator (FewShot - Phase 1 Winner)
    'generator': 'FewShotGenerator',
    'model': 'deepseek-chat',
    'temperature': 0.7,
    'max_tokens': 4096,

    # Evaluator (Verified in Phase 2.5)
    'evaluator': 'RealDataFewShotEvaluator',
    'dataset': 'ai2d',
    'train_epochs': 3,
    'num_shots': 16,
    'batch_size': 4,
    'backbone': 'clip-vit-l-14',
}
```

---

## 3. 扩展搜索空间

```python
EXTENDED_SEARCH_SPACE = {
    # 融合类型
    'fusion_type': ['attention', 'bilinear', 'mlp', 'transformer', 'gated', 'cross_modal'],

    # 架构组件
    'num_fusion_layers': {'type': 'int', 'low': 1, 'high': 6},
    'hidden_dim': {'type': 'int', 'low': 128, 'high': 1024},
    'num_heads': {'type': 'int', 'low': 2, 'high': 16},

    # 激活与归一化
    'activation': ['gelu', 'relu', 'silu', 'swish'],
    'normalization': ['layer_norm', 'batch_norm', 'none'],

    # Dropout与正则化
    'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
    'drop_path_rate': {'type': 'float', 'low': 0.0, 'high': 0.3},

    # 特殊组件
    'use_residual': [True, False],
    'use_gating': [True, False],
    'use_position_embedding': [True, False],
}
```

---

## 4. 文件结构

```
experiment/phase3_discovery/
├── PLAN.md                  # 本计划文档
├── run_phase3.py            # 主实验脚本
├── search_space.py          # 扩展搜索空间定义
├── analysis.py              # 结果分析
├── top_architectures/       # 发现的顶级架构
│   ├── arch_001/
│   │   ├── code.py
│   │   ├── config.json
│   │   └── results.json
│   └── ...
├── results/
│   ├── search_history.json
│   └── final_report.md
└── configs/
    └── phase3.yaml
```

---

## 5. 时间线

| 阶段 | 任务 | 预估时间 |
|------|------|----------|
| **Day 1** | 实验代码开发 | 4-6小时 |
| **Day 2** | 搜索运行（100 iterations） | 8-10小时 |
| **Day 3** | Top架构完整评估 | 6-8小时 |
| **Day 4** | 分析与报告撰写 | 3-4小时 |

---

## 6. 预期成果

1. **发现的架构**: 10+ 高性能融合架构
2. **最佳架构**: 在AI2D上达到 SOTA 或接近 SOTA
3. **设计模式**: 识别有效的融合设计模式
4. **消融研究**: 各组件对性能的贡献分析

---

*Created: 2026-02-13*
