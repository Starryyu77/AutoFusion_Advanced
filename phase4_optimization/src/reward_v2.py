"""
Constrained Reward Function - Phase 4 Optimization
--------------------------------------------------
Enhanced reward function with hard FLOPs constraints and efficiency focus.

Key features:
1. Hard FLOPs constraint (>max_flops gets rejected)
2. Exponential FLOPs penalty
3. Configurable weights (efficiency weight increased to 1.5)
4. Supports rejection tracking for analysis
"""

import math
from typing import Dict, Any
from dataclasses import dataclass, field

# Import from parent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'experiment'))

from base.reward import MultiObjectiveReward, RewardComponents


@dataclass
class ConstrainedRewardComponents(RewardComponents):
    """Extended reward components with rejection tracking"""
    rejected: bool = False
    rejection_reason: str = ""
    flops_ratio: float = 0.0  # FLOPs / max_flops

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        base = super().to_dict()
        base.update({
            'rejected': self.rejected,
            'rejection_reason': self.rejection_reason,
            'flops_ratio': self.flops_ratio
        })
        return base


class ConstrainedReward(MultiObjectiveReward):
    """
    Constrained reward function with hard FLOPs limits.

    Phase 4 improvements:
    - Hard FLOPs constraint (reject if >max_flops)
    - Exponential FLOPs penalty
    - Higher efficiency weight (1.5 vs 0.5)
    - Rejection tracking for analysis
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize constrained reward.

        Args:
            config: Configuration dict with:
                - weights: Dict[str, float] - component weights
                - flops_constraint: Dict with 'enabled', 'max_flops', 'reject_if_exceed'
                - flops_penalty: Dict with 'type', 'scale'
        """
        super().__init__(config)

        # FLOPs hard constraint
        self.flops_constraint = config.get('flops_constraint', {})
        self.use_flops_constraint = self.flops_constraint.get('enabled', False)
        self.max_flops = float(self.flops_constraint.get('max_flops', 1e10))
        self.reject_if_exceed = self.flops_constraint.get('reject_if_exceed', False)

        # Exponential penalty
        self.flops_penalty = config.get('flops_penalty', {})
        self.penalty_type = self.flops_penalty.get('type', 'none')
        self.penalty_scale = float(self.flops_penalty.get('scale', 2e7))

        # Override weights - increase efficiency weight to 1.5
        self.weights = config.get('weights', {
            'accuracy': 1.0,
            'efficiency': 1.5,  # Increased from 0.5
            'compile_success': 2.0,
            'complexity': 0.3,
        })

        print(f"ConstrainedReward initialized:")
        print(f"  FLOPs constraint: {'enabled' if self.use_flops_constraint else 'disabled'}")
        print(f"  Max FLOPs: {self.max_flops/1e6:.1f}M")
        print(f"  Efficiency weight: {self.weights['efficiency']}")
        print(f"  Penalty type: {self.penalty_type}")

    def calculate(self, evaluation_result: Dict[str, Any]) -> ConstrainedRewardComponents:
        """
        Calculate constrained reward.

        Args:
            evaluation_result: Dict with accuracy, flops, params, compile_success, etc.

        Returns:
            ConstrainedRewardComponents with rejection info
        """
        flops = float(evaluation_result.get('flops', 0.0))
        flops_ratio = flops / self.max_flops if self.max_flops > 0 else 0.0

        # Check hard constraint
        if self.use_flops_constraint and self.reject_if_exceed:
            if flops > self.max_flops:
                return ConstrainedRewardComponents(
                    accuracy=0.0,
                    efficiency=0.0,
                    compile_success=0.0 if not evaluation_result.get('compile_success') else 0.9,
                    complexity=0.0,
                    rejected=True,
                    rejection_reason=f'FLOPs {flops/1e6:.1f}M > max {self.max_flops/1e6:.1f}M',
                    flops_ratio=flops_ratio
                )

        # Calculate base reward components
        accuracy = float(evaluation_result.get('accuracy', 0.0))
        accuracy = max(0.0, min(1.0, accuracy))

        # Compile success with label smoothing
        compile_success = float(evaluation_result.get('compile_success', 0.0))
        if self.label_smoothing:
            compile_success = 0.0 if compile_success < 0.5 else 0.9
        compile_success = max(0.0, min(1.0, compile_success))

        # Efficiency (based on FLOPs)
        efficiency = self._compute_efficiency(flops)

        # Complexity (based on params)
        params = float(evaluation_result.get('params', 0.0))
        complexity = self._compute_complexity(params)

        # Apply exponential FLOPs penalty
        if self.penalty_type == 'exponential' and flops > 0:
            penalty = math.exp(-flops / self.penalty_scale)
            accuracy *= penalty
            efficiency *= penalty
            complexity *= penalty

        return ConstrainedRewardComponents(
            accuracy=accuracy,
            efficiency=efficiency,
            compile_success=compile_success,
            complexity=complexity,
            rejected=False,
            rejection_reason="",
            flops_ratio=flops_ratio
        )

    def calculate_scalar(self, evaluation_result: Dict[str, Any]) -> float:
        """
        Calculate scalar reward value (for backward compatibility).

        Returns:
            Scalar reward value (0 if rejected)
        """
        components = self.calculate(evaluation_result)

        if components.rejected:
            return 0.0

        return components.to_scalar(self.weights)

    def _compute_efficiency(self, flops: float) -> float:
        """
        Compute efficiency reward (higher is better, lower FLOPs = higher reward).
        """
        if self.use_log_scale:
            scale = self.max_flops / 10
            efficiency = 1.0 / (1.0 + math.log1p(flops / scale))
        else:
            normalized = flops / self.max_flops
            efficiency = max(0.0, 1.0 - normalized)

        return max(0.0, min(1.0, efficiency))

    def _compute_complexity(self, params: float) -> float:
        """
        Compute complexity reward (lower params = higher reward).
        """
        normalized = params / self.max_params
        complexity = max(0.0, 1.0 - normalized)
        return max(0.0, min(1.0, complexity))

    def get_stats(self) -> Dict[str, Any]:
        """Get reward function statistics for logging."""
        return {
            'type': 'ConstrainedReward',
            'max_flops': self.max_flops,
            'efficiency_weight': self.weights['efficiency'],
            'use_flops_constraint': self.use_flops_constraint,
            'reject_if_exceed': self.reject_if_exceed,
            'penalty_type': self.penalty_type,
            'penalty_scale': self.penalty_scale
        }
