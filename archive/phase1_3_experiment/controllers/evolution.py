"""
Evolution Controller
--------------------
Regularized evolution with age-based regularization.

Features:
- Tournament selection
- Mutation and crossover operations
- Age-based regularization (prevent premature convergence)
- Elite preservation
"""

from typing import Dict, Any, List, Optional
import random
import copy
import numpy as np
from base import BaseController, RewardComponents


class Individual:
    """进化个体"""

    def __init__(self, architecture: Dict[str, Any], age: int = 0):
        self.architecture = architecture
        self.age = age
        self.reward = 0.0
        self.fitness = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'architecture': self.architecture,
            'age': self.age,
            'reward': self.reward,
            'fitness': self.fitness,
        }


class EvolutionController(BaseController):
    """
    正则化进化算法控制器

    特点:
    1. 锦标赛选择 (Tournament Selection)
    2. 变异和交叉操作
    3. 年龄正则化 (防止早熟收敛)
    4. 精英保留
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 进化参数
        self.population_size = config.get('population_size', 50)
        self.tournament_size = config.get('tournament_size', 5)
        self.mutation_rate = config.get('mutation_rate', 0.2)
        self.crossover_rate = config.get('crossover_rate', 0.5)
        self.elite_ratio = config.get('elite_ratio', 0.1)
        self.age_regularization = config.get('age_regularization', True)
        self.age_penalty = config.get('age_penalty', 0.01)
        self.max_age = config.get('max_age', 10)

        # 搜索空间
        self.search_space = config.get('search_space', self._default_search_space())

        # 种群
        self.population: List[Individual] = []
        self.elite: Optional[Individual] = None
        self.current_individual: Optional[Individual] = None
        self.evaluation_count = 0

        # 初始化种群
        self._initialize_population()

    def _default_search_space(self) -> Dict[str, Any]:
        """默认搜索空间"""
        return {
            'type': ['attention', 'conv', 'mlp', 'transformer', 'hybrid'],
            'fusion_type': ['early', 'late', 'middle', 'hierarchical'],
            'hidden_dim': {'type': 'int', 'low': 256, 'high': 1024},
            'num_layers': {'type': 'int', 'low': 2, 'high': 8},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'activation': ['gelu', 'relu', 'silu'],
        }

    def _initialize_population(self) -> None:
        """初始化种群"""
        for _ in range(self.population_size):
            arch = self._random_architecture()
            self.population.append(Individual(arch))

    def _random_architecture(self) -> Dict[str, Any]:
        """随机生成架构"""
        arch = {}
        for key, space in self.search_space.items():
            if isinstance(space, list):
                arch[key] = random.choice(space)
            elif isinstance(space, dict):
                if space['type'] == 'int':
                    arch[key] = random.randint(space['low'], space['high'])
                elif space['type'] == 'float':
                    arch[key] = random.uniform(space['low'], space['high'])
        return arch

    def propose(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        """提出架构候选"""
        # 如果种群未满，从种群中选择
        if self.evaluation_count < self.population_size:
            individual = self.population[self.evaluation_count]
        else:
            # 生成新个体 (通过变异或交叉)
            individual = self._generate_offspring()

        self.current_individual = individual
        self.evaluation_count += 1

        return {
            'architecture': individual.architecture,
            'instruction': self._build_instruction(individual.architecture),
            'metadata': {
                'age': individual.age,
                'population_size': len(self.population),
                'evaluation_count': self.evaluation_count,
            },
        }

    def update(self, reward: RewardComponents) -> None:
        """更新个体适应度"""
        if self.current_individual is None:
            return

        reward_scalar = reward.to_scalar(self.config.get('reward_weights'))
        self.current_individual.reward = reward_scalar

        # 年龄正则化
        if self.age_regularization:
            age_penalty = self.age_penalty * self.current_individual.age
            self.current_individual.fitness = reward_scalar - age_penalty
        else:
            self.current_individual.fitness = reward_scalar

        # 更新精英
        if self.elite is None or reward_scalar > self.elite.reward:
            self.elite = copy.deepcopy(self.current_individual)
            self.state.best_reward = reward_scalar
            self.state.best_architecture = self.current_individual.architecture

        # 种群管理
        if self.evaluation_count >= self.population_size:
            self._manage_population()

        # 增加年龄
        self.current_individual.age += 1

    def _generate_offspring(self) -> Individual:
        """生成后代"""
        if random.random() < self.crossover_rate:
            # 交叉
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            offspring = self._crossover(parent1, parent2)
        else:
            # 变异
            parent = self._tournament_select()
            offspring = self._mutate(parent)

        return offspring

    def _tournament_select(self) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """交叉操作"""
        child_arch = {}
        for key in parent1.architecture:
            if random.random() < 0.5:
                child_arch[key] = parent1.architecture[key]
            else:
                child_arch[key] = parent2.architecture[key]

        return Individual(child_arch, age=0)

    def _mutate(self, parent: Individual) -> Individual:
        """变异操作"""
        child_arch = copy.deepcopy(parent.architecture)

        for key in child_arch:
            if random.random() < self.mutation_rate:
                space = self.search_space[key]
                if isinstance(space, list):
                    child_arch[key] = random.choice(space)
                elif isinstance(space, dict):
                    if space['type'] == 'int':
                        # 高斯变异
                        value = child_arch[key] + int(random.gauss(0, (space['high'] - space['low']) * 0.1))
                        child_arch[key] = max(space['low'], min(space['high'], value))
                    elif space['type'] == 'float':
                        value = child_arch[key] + random.gauss(0, (space['high'] - space['low']) * 0.1)
                        child_arch[key] = max(space['low'], min(space['high'], value))

        return Individual(child_arch, age=0)

    def _manage_population(self) -> None:
        """种群管理 (选择下一代)"""
        # 年龄增长
        for ind in self.population:
            ind.age += 1

        # 移除过老个体
        self.population = [ind for ind in self.population if ind.age < self.max_age]

        # 精英保留
        n_elite = max(1, int(self.elite_ratio * self.population_size))
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population = sorted_pop[:n_elite]

        # 填充种群
        while len(new_population) < self.population_size:
            offspring = self._generate_offspring()
            new_population.append(offspring)

        self.population = new_population[:self.population_size]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = super().get_stats()
        stats.update({
            'population_size': len(self.population),
            'avg_age': sum(ind.age for ind in self.population) / len(self.population) if self.population else 0,
            'avg_fitness': sum(ind.fitness for ind in self.population) / len(self.population) if self.population else 0,
            'elite_reward': self.elite.reward if self.elite else 0,
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