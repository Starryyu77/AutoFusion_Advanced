"""
评估框架 - 支持NAS发现架构与传统基线的全面对比

核心组件:
- FullTrainer: 100 epochs完整训练
- Metrics: 指标计算 (Accuracy, FLOPs, Params, Latency)
- UnifiedEvaluator: 统一评估接口
"""

from .full_trainer import FullTrainer
from .metrics import MetricsCalculator
from .unified_evaluator import UnifiedEvaluator

__all__ = [
    'FullTrainer',
    'MetricsCalculator',
    'UnifiedEvaluator',
]
