"""
GRPO Controller
---------------
Group Relative Policy Optimization with Bootstrap variance estimation.

Key features:
- Group-wise normalization (advantages computed within group)
- Bootstrap variance estimation (reduce small sample bias)
- Variance clipping (prevent extreme values)
- No Critic needed (group mean as baseline)
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseController, RewardComponents


class GRPOController(BaseController):
    """
    Group Relative Policy Optimization (GRPO)

    Advantages computed as: A_i = (R_i - mean(R)) / (std(R) + epsilon)

    Key differences from PPO:
    1. No Critic needed (group mean serves as baseline)
    2. Group-wise normalization (more stable for small batches)
    3. Bootstrap variance estimation (better for small groups)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # GRPO hyperparameters
        self.group_size = config.get('group_size', 8)
        self.learning_rate = float(config.get('learning_rate', 1e-5))
        self.beta = config.get('beta', 0.04)  # KL penalty coefficient
        self.clip_range = config.get('clip_range', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.n_epochs = config.get('n_epochs', 2)

        # Bootstrap variance estimation
        self.use_bootstrap = config.get('use_bootstrap', True)
        self.n_bootstrap = config.get('n_bootstrap', 100)
        self.variance_clip = float(config.get('variance_clip', 5.0))  # Upper bound for std

        # State dimensions
        self.state_dim = config.get('state_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.action_dim = config.get('action_dim', 16)

        # Initialize policy network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = self._build_policy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Reference policy for KL computation
        self.reference_policy = None

        # Group memory
        self.group_states = []
        self.group_actions = []
        self.group_log_probs = []
        self.group_rewards = []

        self.current_state = torch.randn(self.state_dim).to(self.device)

    def _build_policy(self) -> nn.Module:
        """构建策略网络"""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def propose(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        """提出架构候选"""
        with torch.no_grad():
            action = self.policy(self.current_state.unsqueeze(0)).squeeze(0)
            # Add exploration noise
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1, 1)

        self.group_states.append(self.current_state.clone())
        self.group_actions.append(action.clone())

        # Convert action to architecture
        architecture = self._action_to_architecture(action.cpu().numpy())

        return {
            'architecture': architecture,
            'instruction': self._build_instruction(architecture),
            'metadata': {
                'action': action.cpu().numpy().tolist(),
                'group_size': self.group_size,
            },
        }

    def update(self, reward: RewardComponents) -> None:
        """收集奖励，当组满时更新"""
        reward_scalar = reward.to_scalar(self.config.get('reward_weights'))
        self.group_rewards.append(reward_scalar)

        # 当组满时进行更新
        if len(self.group_rewards) >= self.group_size:
            self._update_group()

    def _update_group(self) -> None:
        """GRPO组更新"""
        if len(self.group_rewards) < self.group_size:
            return

        # 转换为tensor
        states = torch.stack(self.group_states[:self.group_size])
        actions = torch.stack(self.group_actions[:self.group_size])
        rewards = torch.tensor(self.group_rewards[:self.group_size], dtype=torch.float32).to(self.device)

        # 计算标量奖励用于GRPO
        # GRPO: 整体归一化
        mean_reward = rewards.mean()

        # Bootstrap方差估计
        if self.use_bootstrap and len(rewards) >= 4:
            std_reward = self._bootstrap_std(rewards)
        else:
            std_reward = rewards.std()

        # 方差裁剪 (防止极端值)
        std_reward = torch.clamp(std_reward, min=1e-4, max=self.variance_clip)

        # 计算优势
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        # 保存参考策略
        if self.reference_policy is None:
            self.reference_policy = self._copy_policy()

        # GRPO更新
        for _ in range(self.n_epochs):
            # 当前策略动作
            current_actions = self.policy(states)

            # 计算策略损失 (简化的策略梯度)
            # 使用MSE损失鼓励向高奖励动作移动
            weighted_actions = current_actions * advantages.unsqueeze(1)
            policy_loss = -weighted_actions.mean()

            # KL散度惩罚
            with torch.no_grad():
                ref_actions = self.reference_policy(states)
            kl_loss = F.mse_loss(current_actions, ref_actions)

            # 总损失
            loss = policy_loss + self.beta * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # 清空组内存
        self.group_states = self.group_states[self.group_size:]
        self.group_actions = self.group_actions[self.group_size:]
        self.group_rewards = self.group_rewards[self.group_size:]

        # 更新状态
        self.current_state = torch.randn(self.state_dim).to(self.device)

    def _bootstrap_std(self, values: torch.Tensor, n_bootstrap: int = None) -> torch.Tensor:
        """
        Bootstrap方差估计

        Args:
            values: 原始值
            n_bootstrap: Bootstrap采样次数

        Returns:
            Bootstrap估计的标准差
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        n = len(values)
        stds = []

        for _ in range(n_bootstrap):
            indices = torch.randint(0, n, (n,))
            sample = values[indices]
            stds.append(sample.std().item())

        return torch.tensor(stds).mean().to(self.device)

    def _copy_policy(self) -> nn.Module:
        """复制当前策略作为参考"""
        import copy
        ref = copy.deepcopy(self.policy)
        ref.eval()
        for param in ref.parameters():
            param.requires_grad = False
        return ref

    def _action_to_architecture(self, action: np.ndarray) -> Dict[str, Any]:
        """将动作转换为架构描述"""
        arch_types = ['attention', 'conv', 'mlp', 'transformer', 'hybrid']
        fusion_types = ['early', 'late', 'middle', 'hierarchical']

        # 使用action的前几个维度选择架构类型
        arch_idx = int((action[0] + 1) * 0.5 * 100) % len(arch_types)
        fusion_idx = int((action[1] + 1) * 0.5 * 100) % len(fusion_types)

        return {
            'type': arch_types[arch_idx],
            'fusion_type': fusion_types[fusion_idx],
            'hidden_dim': int(256 + (action[2] + 1) * 0.5 * 768),
            'num_layers': int(2 + (action[3] + 1) * 0.5 * 6),
            'dropout': float((action[4] + 1) * 0.25),
            'activation': 'gelu' if action[5] > 0 else 'relu',
        }

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