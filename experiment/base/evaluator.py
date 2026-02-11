"""
Evaluator Abstract Base Class
-----------------------------
All evaluation strategies must inherit from BaseEvaluator.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class EvaluationResult:
    """评估结果"""
    accuracy: float
    efficiency: float
    compile_success: float
    flops: float = 0.0
    params: float = 0.0
    latency: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'efficiency': self.efficiency,
            'compile_success': self.compile_success,
            'flops': self.flops,
            'params': self.params,
            'latency': self.latency,
            'metadata': self.metadata,
        }


class BaseEvaluator(ABC):
    """Evaluator 抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quick_train_epochs = config.get('quick_train_epochs', 5)
        self.use_lr_decay = config.get('use_lr_decay', True)
        self.device = config.get('device', 'cuda')

    @abstractmethod
    def evaluate(self, code: str, context: Optional[Dict] = None) -> EvaluationResult:
        """
        评估代码并返回多目标奖励

        Args:
            code: 生成的Python代码
            context: 额外上下文信息

        Returns:
            EvaluationResult对象
        """
        pass

    @abstractmethod
    def compute_flops(self, model: nn.Module, input_shape: tuple = None) -> float:
        """
        计算FLOPs

        Args:
            model: PyTorch模型
            input_shape: 输入形状

        Returns:
            FLOPs数量
        """
        pass

    def compute_params(self, model: nn.Module) -> float:
        """
        计算参数量

        Args:
            model: PyTorch模型

        Returns:
            参数数量
        """
        return sum(p.numel() for p in model.parameters())

    def compile_code(self, code: str) -> Tuple[bool, Any]:
        """
        编译代码并返回模块

        Args:
            code: Python代码

        Returns:
            (是否成功, 模块或错误信息)
        """
        try:
            # 创建局部命名空间
            namespace = {}
            exec(code, namespace)

            # 查找模型类
            model_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, nn.Module):
                    model_class = obj
                    break

            if model_class is None:
                return False, "No nn.Module subclass found in code"

            return True, model_class

        except Exception as e:
            return False, str(e)
