"""
Random Controller
-----------------
Random search baseline for comparison.
"""

from typing import Dict, Any, Optional
import random
import numpy as np
from base import BaseController, RewardComponents


class RandomController(BaseController):
    """随机搜索控制器 - 基线方法"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.search_space = config.get('search_space', {})
        self.seed = config.get('seed', 42)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def propose(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        """随机采样架构"""
        architecture = self._random_sample()
        return {
            'architecture': architecture,
            'instruction': self._build_instruction(architecture),
            'metadata': {'type': 'random'},
        }

    def update(self, reward: RewardComponents) -> None:
        """随机搜索不需要更新"""
        pass

    def _random_sample(self) -> Dict[str, Any]:
        """从搜索空间随机采样"""
        architecture = {}
        for key, space in self.search_space.items():
            if isinstance(space, list):
                architecture[key] = random.choice(space)
            elif isinstance(space, dict):
                if 'type' in space:
                    if space['type'] == 'int':
                        low = space.get('low', 0)
                        high = space.get('high', 10)
                        architecture[key] = random.randint(low, high)
                    elif space['type'] == 'float':
                        low = space.get('low', 0.0)
                        high = space.get('high', 1.0)
                        architecture[key] = random.uniform(low, high)
                    elif space['type'] == 'choice':
                        architecture[key] = random.choice(space['options'])
        return architecture

    def _build_instruction(self, architecture: Dict[str, Any]) -> str:
        """构建生成指令"""
        return f"""Implement a multimodal fusion module with the following configuration:
{architecture}

Requirements:
1. Use PyTorch nn.Module
2. Handle vision and language features
3. Implement forward() method
4. Return fused representation
"""