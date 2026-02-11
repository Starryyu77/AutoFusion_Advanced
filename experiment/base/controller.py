"""
Controller Abstract Base Class
------------------------------
All search algorithms must inherit from BaseController.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import torch
import json


@dataclass
class SearchState:
    """搜索状态"""
    iteration: int = 0
    best_reward: float = 0.0
    best_architecture: Optional[Dict] = None
    history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'best_reward': self.best_reward,
            'best_architecture': self.best_architecture,
            'history': self.history,
        }


class BaseController(ABC):
    """Controller 抽象基类 - 所有搜索算法必须继承"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = SearchState()
        self.max_iterations = config.get('max_iterations', 100)
        self.early_stop_patience = config.get('early_stop_patience', 20)
        self.no_improvement_count = 0

    @abstractmethod
    def propose(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        提出下一个架构候选

        Returns:
            Dict containing:
                - 'architecture': 架构描述
                - 'instruction': 生成指令
                - 'metadata': 额外信息
        """
        pass

    @abstractmethod
    def update(self, reward: "RewardComponents") -> None:
        """
        根据奖励更新策略

        Args:
            reward: RewardComponents 对象 (支持标量或字典访问)
        """
        pass

    def should_stop(self) -> bool:
        """是否停止搜索"""
        # 达到最大迭代次数
        if self.state.iteration >= self.max_iterations:
            return True

        # 早停检查
        if self.no_improvement_count >= self.early_stop_patience:
            return True

        return False

    def record_iteration(self, architecture: Dict, reward: "RewardComponents") -> None:
        """记录迭代结果"""
        reward_value = reward.to_scalar() if hasattr(reward, 'to_scalar') else float(reward)

        self.state.iteration += 1
        self.state.history.append({
            'iteration': self.state.iteration,
            'architecture': architecture,
            'reward': reward.to_dict() if hasattr(reward, 'to_dict') else {'scalar': reward_value},
        })

        # 更新最佳结果
        if reward_value > self.state.best_reward:
            self.state.best_reward = reward_value
            self.state.best_architecture = architecture
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        checkpoint = {
            'state': self.state.to_dict(),
            'config': self.config,
            'controller_type': self.__class__.__name__,
        }
        torch.save(checkpoint, path)

        # 同时保存JSON版本便于查看
        json_path = path.replace('.pt', '.json')
        with open(json_path, 'w') as f:
            json.dump(checkpoint['state'], f, indent=2, default=str)

    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        self.state = SearchState(**checkpoint['state'])
        # 不覆盖config，保持当前配置

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'iteration': self.state.iteration,
            'best_reward': self.state.best_reward,
            'no_improvement_count': self.no_improvement_count,
            'history_length': len(self.state.history),
        }