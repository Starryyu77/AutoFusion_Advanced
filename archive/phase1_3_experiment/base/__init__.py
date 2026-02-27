"""
Auto-Fusion Experiment Base Classes
-----------------------------------
Abstract base classes for Controller, Generator, Evaluator, and Reward.
"""

from .controller import BaseController, SearchState
from .generator import BaseGenerator, GenerationResult
from .evaluator import BaseEvaluator, EvaluationResult
from .reward import BaseReward, RewardComponents, MultiObjectiveReward

__all__ = [
    'BaseController',
    'SearchState',
    'BaseGenerator',
    'GenerationResult',
    'BaseEvaluator',
    'EvaluationResult',
    'BaseReward',
    'RewardComponents',
    'MultiObjectiveReward',
]