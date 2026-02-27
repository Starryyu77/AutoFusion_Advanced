"""
CMA-ES Controller
-----------------
Covariance Matrix Adaptation Evolution Strategy.

Features:
- Sample-efficient black-box optimization
- Adaptive step size
- No hyperparameter tuning required
"""

from typing import Dict, Any, List, Optional
import numpy as np
import copy
from base import BaseController, RewardComponents


class CMAESController(BaseController):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

    特点:
    1. 黑盒优化，无需梯度
    2. 自适应步长
    3. 样本效率高

    参考: Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # CMA-ES参数
        self.dim = config.get('dim', 16)  # 搜索空间维度
        self.population_size = config.get('population_size', None)  # 自动计算
        self.sigma = config.get('sigma', 0.5)  # 初始步长
        self.centroid = None
        self.C = None  # 协方差矩阵
        self.pc = None  # 进化路径
        self.ps = None  # 步长路径

        # 自动计算种群大小
        if self.population_size is None:
            self.population_size = 4 + int(3 * np.log(self.dim))

        self.mu = self.population_size // 2  # 父代数量

        # 权重
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        # 自适应参数
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # 初始化
        self._initialize()

        # 当前代
        self.generation = 0
        self.population: List[Dict] = []
        self.current_idx = 0

    def _initialize(self) -> None:
        """初始化CMA-ES状态"""
        self.centroid = np.random.randn(self.dim)
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)

    def propose(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        """提出架构候选"""
        # 生成新一代
        if self.current_idx >= len(self.population):
            self._generate_generation()
            self.current_idx = 0

        individual = self.population[self.current_idx]
        self.current_idx += 1

        return {
            'architecture': individual['architecture'],
            'instruction': self._build_instruction(individual['architecture']),
            'metadata': {
                'generation': self.generation,
                'individual_idx': self.current_idx - 1,
                'sigma': self.sigma,
            },
        }

    def _generate_generation(self) -> None:
        """生成新一代种群"""
        self.population = []

        # 从多元正态分布采样
        for _ in range(self.population_size):
            # 采样: x = centroid + sigma * N(0, C)
            z = np.random.randn(self.dim)
            y = self.centroid + self.sigma * (self.C @ z)

            # 转换为架构
            architecture = self._vector_to_architecture(y)

            self.population.append({
                'vector': y,
                'architecture': architecture,
                'reward': None,
            })

    def update(self, reward: RewardComponents) -> None:
        """更新CMA-ES状态"""
        reward_scalar = reward.to_scalar(self.config.get('reward_weights'))

        # 记录奖励
        idx = self.current_idx - 1
        if 0 <= idx < len(self.population):
            self.population[idx]['reward'] = reward_scalar

        # 当一代评估完成时更新
        if self.current_idx >= len(self.population):
            self._update_generation()

    def _update_generation(self) -> None:
        """更新一代"""
        # 按奖励排序
        sorted_pop = sorted(self.population, key=lambda x: x['reward'] if x['reward'] is not None else -float('inf'), reverse=True)

        # 保存旧中心
        old_centroid = self.centroid.copy()

        # 更新中心
        selected = sorted_pop[:self.mu]
        vectors = np.array([ind['vector'] for ind in selected])
        self.centroid = vectors.T @ self.weights

        # 更新进化路径
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.centroid - old_centroid) / self.sigma

        # 更新步长
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))

        # 更新协方差矩阵
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.centroid - old_centroid) / self.sigma

        # 计算秩更新
        artmp = (vectors - old_centroid) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * np.outer(self.pc, self.pc) +
                  self.cmu * artmp.T @ np.diag(self.weights) @ artmp)

        # 确保协方差矩阵正定
        self.C = (self.C + self.C.T) / 2  # 对称化
        eigvals = np.linalg.eigvalsh(self.C)
        if np.min(eigvals) <= 0:
            self.C += (abs(np.min(eigvals)) + 1e-8) * np.eye(self.dim)

        self.generation += 1

        # 更新最佳奖励
        best_reward = sorted_pop[0]['reward']
        if best_reward > self.state.best_reward:
            self.state.best_reward = best_reward
            self.state.best_architecture = sorted_pop[0]['architecture']

    def _vector_to_architecture(self, vector: np.ndarray) -> Dict[str, Any]:
        """将向量转换为架构描述"""
        arch_types = ['attention', 'conv', 'mlp', 'transformer', 'hybrid']
        fusion_types = ['early', 'late', 'middle', 'hierarchical']
        activations = ['gelu', 'relu', 'silu']

        # 使用向量元素选择架构参数
        arch_idx = int((vector[0] % 1 + 1) / 2 * 100) % len(arch_types)
        fusion_idx = int((vector[1] % 1 + 1) / 2 * 100) % len(fusion_types)
        act_idx = int((vector[5] % 1 + 1) / 2 * 100) % len(activations)

        return {
            'type': arch_types[arch_idx],
            'fusion_type': fusion_types[fusion_idx],
            'hidden_dim': int(256 + abs(vector[2]) * 768),
            'num_layers': int(2 + abs(vector[3]) * 6),
            'dropout': min(0.5, abs(vector[4])),
            'activation': activations[act_idx],
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = super().get_stats()
        stats.update({
            'generation': self.generation,
            'sigma': self.sigma,
            'population_size': self.population_size,
        })
        return stats

    def _build_instruction(self, architecture: Dict[str, Any]) -> str:
        """构建生成指令"""
        return f"""Implement a {architecture['type']} multimodal fusion module.

Configuration:
- Fusion type: {architecture['fusion_type']}
- Hidden dimension: {architecture['hidden_dim']}
- Number of layers: {architecture['num_layers']}
- Dropout: {architecture['dropout']:.2f}
- Activation: {architecture['activation']}

Requirements:
1. Use PyTorch nn.Module
2. Handle vision features (image) and language features (text)
3. Implement forward(vision_features, language_features) method
4. Return fused representation of shape (batch, {architecture['hidden_dim']})
5. Include proper layer normalization and residual connections
"""
