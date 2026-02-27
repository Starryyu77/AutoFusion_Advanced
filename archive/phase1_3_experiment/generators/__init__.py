"""
Generators - Prompt Strategies
------------------------------
Implementation of various prompt strategies for code generation.
"""

from .cot import ChainOfThoughtGenerator
from .fewshot import FewShotGenerator
from .critic import CriticGenerator
from .shape import ShapeConstraintGenerator
from .roleplay import RolePlayGenerator

__all__ = [
    'ChainOfThoughtGenerator',
    'FewShotGenerator',
    'CriticGenerator',
    'ShapeConstraintGenerator',
    'RolePlayGenerator',
]