"""
Reward System
-------------
Multi-objective reward calculation supporting both scalar and dict formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass, asdict
import math


@dataclass
class RewardComponents:
    """标准化奖励组件 - 支持 GDPO"""
    accuracy: float = 0.0
    efficiency: float = 0.0
    compile_success: float = 0.0
    complexity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """GDPO: 返回字典用于解耦归一化"""
        return asdict(self)

    def to_scalar(self, weights: Dict[str, float] = None) -> float:
        """PPO/GRPO: 加权求和为标量"""
        if weights is None:
            weights = {
                'accuracy': 1.0,
                'efficiency': 0.5,
                'compile_success': 2.0,
                'complexity': 0.0,
            }
        return sum(
            weights.get(k, 0) * getattr(self, k)
            for k in ['accuracy', 'efficiency', 'compile_success', 'complexity']
        )


class BaseReward(ABC):
    """奖励函数抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = config.get('weights', {
            'accuracy': 1.0,
            'efficiency': 0.5,
            'compile_success': 2.0,
            'complexity': 0.0,
        })
        self.label_smoothing = config.get('label_smoothing', True)

    @abstractmethod
    def calculate(self, evaluation_result: Dict[str, Any]) -> RewardComponents:
        """
        计算奖励

        Args:
            evaluation_result: 评估结果字典

        Returns:
            RewardComponents对象
        """
        pass


class MultiObjectiveReward(BaseReward):
    """多目标奖励计算"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_flops = float(config.get('max_flops', 1e9))
        self.max_params = float(config.get('max_params', 1e8))
        self.use_log_scale = config.get('use_log_scale', True)
        self.use_robust_norm = config.get('use_robust_norm', False)

    def calculate(self, evaluation_result: Dict[str, Any]) -> RewardComponents:
        """
        计算多目标奖励

        Args:
            evaluation_result: 包含 accuracy, flops, params, compile_success 等

        Returns:
            RewardComponents对象
        """
        # 1. 准确率奖励 [0, 1]
        accuracy = evaluation_result.get('accuracy', 0.0)
        accuracy = max(0.0, min(1.0, accuracy))

        # 2. 编译成功奖励 {0, 1} -> 平滑处理 [0.0, 0.9]
        # Fix 4: 编译失败给 0 分，而非负值或极小值
        compile_success = evaluation_result.get('compile_success', 0.0)
        if self.label_smoothing:
            # Label smoothing: 0 -> 0.0, 1 -> 0.9
            # 编译失败 = 0 分 (没得分)，而非"世界末日"
            compile_success = 0.0 if compile_success < 0.5 else 0.9
        compile_success = max(0.0, min(1.0, compile_success))

        # 3. 效率奖励 (基于FLOPs)
        flops = evaluation_result.get('flops', 0.0)
        efficiency = self._compute_efficiency(flops)

        # 4. 复杂度奖励 (基于参数量)
        params = evaluation_result.get('params', 0.0)
        complexity = self._compute_complexity(params)

        return RewardComponents(
            accuracy=accuracy,
            efficiency=efficiency,
            compile_success=compile_success,
            complexity=complexity,
        )

    def _compute_efficiency(self, flops: float) -> float:
        """
        计算效率奖励

        方案A: 对数缩放 (防长尾)
        r_eff = 1.0 / (1.0 + log1p(flops / scale))

        方案B: 线性归一化
        r_eff = 1.0 - flops / max_flops
        """
        if self.use_log_scale:
            # 对数缩放，对高FLOPs更宽容
            scale = self.max_flops / 10
            efficiency = 1.0 / (1.0 + math.log1p(flops / scale))
        else:
            # 线性归一化
            normalized = flops / self.max_flops
            efficiency = max(0.0, 1.0 - normalized)

        return max(0.0, min(1.0, efficiency))

    def _compute_complexity(self, params: float) -> float:
        """
        计算复杂度奖励 (参数越少越好)
        """
        normalized = params / self.max_params
        complexity = max(0.0, 1.0 - normalized)
        return max(0.0, min(1.0, complexity))

    def pareto_rank(self, rewards: list) -> list:
        """
        NSGA-II风格Pareto非支配排序

        Args:
            rewards: RewardComponents列表

        Returns:
            每个个体的Pareto等级列表 (0 = 最优)
        """
        n = len(rewards)
        ranks = [0] * n

        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(rewards[j], rewards[i]):
                    ranks[i] += 1

        return ranks

    def _dominates(self, a: RewardComponents, b: RewardComponents) -> bool:
        """
        判断a是否支配b
        (所有目标都不差，至少一个更好)
        """
        keys = ['accuracy', 'efficiency', 'compile_success']
        better = False
        for k in keys:
            va = getattr(a, k)
            vb = getattr(b, k)
            if va < vb:
                return False
            if va > vb:
                better = True
        return better


@dataclass
class SharpenedRewardComponents(RewardComponents):
    """带锐化标量值的奖励组件"""
    sharpened_scalar: float = 0.0

    def to_scalar(self, weights: Dict[str, float] = None) -> float:
        """返回锐化后的标量值"""
        return self.sharpened_scalar


class ExponentialReward(MultiObjectiveReward):
    """
    指数缩放奖励 - Fix 2: 锐化奖励函数

    公式: R = exp((Acc - Baseline) × alpha)

    目的: 拉大好坏架构的奖励差距，让 RL 更容易学习
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Fix 2: 指数锐化参数
        self.baseline = float(config.get('baseline', 2.5))  # Random baseline ~3.1
        self.alpha = float(config.get('alpha', 3.0))  # 锐化系数
        self.max_sharpened = float(config.get('max_sharpened', 10.0))  # 防止爆炸

    def calculate(self, evaluation_result: Dict[str, Any]) -> RewardComponents:
        """
        计算指数锐化奖励

        步骤:
        1. 先计算标准多目标奖励
        2. 指数锐化拉大差距
        3. 保持各组件比例
        """
        # 1. 计算标准奖励
        components = super().calculate(evaluation_result)

        # 2. 计算标量奖励
        scalar = components.to_scalar(self.weights)

        # 3. 指数锐化: R = exp((scalar - baseline) * alpha)
        # 例如: baseline=2.5, alpha=3.0
        # scalar=3.0 -> exp(0.5*3) = exp(1.5) = 4.48
        # scalar=2.8 -> exp(0.3*3) = exp(0.9) = 2.46
        # scalar=2.5 -> exp(0*3) = 1.0
        sharpened = math.exp((scalar - self.baseline) * self.alpha)

        # 4. Clip 防止数值爆炸
        sharpened = min(sharpened, self.max_sharpened)

        # 5. 保持组件比例，重新分配
        if scalar > 1e-8:
            factor = sharpened / scalar
        else:
            factor = 1.0

        # 使用 SharpenedRewardComponents 保存锐化后的标量值
        return SharpenedRewardComponents(
            accuracy=components.accuracy * factor,
            efficiency=components.efficiency * factor,
            compile_success=components.compile_success * factor,
            complexity=components.complexity * factor,
            sharpened_scalar=sharpened,
        )


class MultiObjectiveReward(BaseReward):
    """多目标奖励计算"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_flops = float(config.get('max_flops', 1e9))
        self.max_params = float(config.get('max_params', 1e8))
        self.use_log_scale = config.get('use_log_scale', True)
        self.use_robust_norm = config.get('use_robust_norm', False)

    def calculate(self, evaluation_result: Dict[str, Any]) -> RewardComponents:
        """
        计算多目标奖励

        Args:
            evaluation_result: 包含 accuracy, flops, params, compile_success 等

        Returns:
            RewardComponents对象
        """
        # 1. 准确率奖励 [0, 1]
        accuracy = evaluation_result.get('accuracy', 0.0)
        accuracy = max(0.0, min(1.0, accuracy))

        # 2. 编译成功奖励 {0, 1} -> 平滑处理 [0.0, 0.9]
        # Fix 4: 编译失败给 0 分，而非负值或极小值
        compile_success = evaluation_result.get('compile_success', 0.0)
        if self.label_smoothing:
            # Label smoothing: 0 -> 0.0, 1 -> 0.9
            # 编译失败 = 0 分 (没得分)，而非"世界末日"
            compile_success = 0.0 if compile_success < 0.5 else 0.9
        compile_success = max(0.0, min(1.0, compile_success))

        # 3. 效率奖励 (基于FLOPs)
        flops = evaluation_result.get('flops', 0.0)
        efficiency = self._compute_efficiency(flops)

        # 4. 复杂度奖励 (基于参数量)
        params = evaluation_result.get('params', 0.0)
        complexity = self._compute_complexity(params)

        return RewardComponents(
            accuracy=accuracy,
            efficiency=efficiency,
            compile_success=compile_success,
            complexity=complexity,
        )

    def _compute_efficiency(self, flops: float) -> float:
        """
        计算效率奖励

        方案A: 对数缩放 (防长尾)
        r_eff = 1.0 / (1.0 + log1p(flops / scale))

        方案B: 线性归一化
        r_eff = 1.0 - flops / max_flops
        """
        if self.use_log_scale:
            # 对数缩放，对高FLOPs更宽容
            scale = self.max_flops / 10
            efficiency = 1.0 / (1.0 + math.log1p(flops / scale))
        else:
            # 线性归一化
            normalized = flops / self.max_flops
            efficiency = max(0.0, 1.0 - normalized)

        return max(0.0, min(1.0, efficiency))

    def _compute_complexity(self, params: float) -> float:
        """
        计算复杂度奖励 (参数越少越好)
        """
        normalized = params / self.max_params
        complexity = max(0.0, 1.0 - normalized)
        return max(0.0, min(1.0, complexity))

    def pareto_rank(self, rewards: list) -> list:
        """
        NSGA-II风格Pareto非支配排序

        Args:
            rewards: RewardComponents列表

        Returns:
            每个个体的Pareto等级列表 (0 = 最优)
        """
        n = len(rewards)
        ranks = [0] * n

        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(rewards[j], rewards[i]):
                    ranks[i] += 1

        return ranks

    def _dominates(self, a: RewardComponents, b: RewardComponents) -> bool:
        """
        判断a是否支配b
        (所有目标都不差，至少一个更好)
        """
        keys = ['accuracy', 'efficiency', 'compile_success']
        better = False
        for k in keys:
            va = getattr(a, k)
            vb = getattr(b, k)
            if va < vb:
                return False
            if va > vb:
                better = True
        return better
