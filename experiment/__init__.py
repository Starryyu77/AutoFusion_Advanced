"""
Auto-Fusion Experiment Framework
--------------------------------
Systematic comparison of RL algorithms and prompt strategies for NAS.

Usage:
    from experiment import create_experiment_components
    from experiment.factory import ExperimentFactory

    # Create components
    components = create_experiment_components(
        controller_name='gdpo',
        generator_name='cot',
        evaluator_name='sandbox',
        config=config,
    )

    # Run search
    controller = components['controller']
    generator = components['generator']
    evaluator = components['evaluator']
    reward = components['reward']
"""

__version__ = "1.0.0"

from .factory import (
    create_controller,
    create_generator,
    create_evaluator,
    create_reward,
    create_experiment_components,
    ExperimentFactory,
)

from .base import (
    BaseController,
    BaseGenerator,
    BaseEvaluator,
    MultiObjectiveReward,
    SearchState,
    GenerationResult,
    EvaluationResult,
    RewardComponents,
)

__all__ = [
    # Factory functions
    'create_controller',
    'create_generator',
    'create_evaluator',
    'create_reward',
    'create_experiment_components',
    'ExperimentFactory',
    # Base classes
    'BaseController',
    'BaseGenerator',
    'BaseEvaluator',
    'MultiObjectiveReward',
    # Data classes
    'SearchState',
    'GenerationResult',
    'EvaluationResult',
    'RewardComponents',
]
