"""
PPO Controller
--------------
Proximal Policy Optimization with Critic-Free variant for Contextual Bandit.

Key insight: NAS code generation is Contextual Bandit (T=1), not MDP.
PPO-CriticFree uses Running Mean Baseline instead of Critic.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseController, RewardComponents


class PolicyNetwork(nn.Module):
    """策略网络 - 输出架构参数"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    """价值网络 (Critic) - 可选使用"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOController(BaseController):
    """
    PPO Controller with Critic-Free variant

    For NAS (Contextual Bandit, T=1):
    - use_critic=False: Running Mean Baseline (recommended)
    - use_critic=True: Standard PPO with Critic
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PPO hyperparameters
        self.use_critic = config.get('use_critic', False)  # Default: Critic-Free for NAS
        self.clip_range = float(config.get('clip_range', 0.2))
        self.learning_rate = float(config.get('learning_rate', 3e-5))
        self.critic_lr = float(config.get('critic_lr', 1e-4))
        self.entropy_coef = float(config.get('entropy_coef', 0.01))
        self.baseline_momentum = float(config.get('baseline_momentum', 0.9))
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        self.n_epochs = config.get('n_epochs', 4)
        self.batch_size = config.get('batch_size', 32)

        # State dimensions
        self.state_dim = config.get('state_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.action_dim = config.get('action_dim', 16)

        # Initialize networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = PolicyNetwork(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        if self.use_critic:
            self.critic = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        else:
            self.critic = None
            self.baseline = None  # Running mean baseline

        # Memory for batch update
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []

        # Current state
        self.current_state = torch.randn(self.state_dim).to(self.device)

    def propose(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        """提出架构候选"""
        with torch.no_grad():
            mean, std = self.policy(self.current_state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

        self.states.append(self.current_state.clone())
        self.actions.append(action.clone())
        self.log_probs.append(log_prob.clone())

        if self.use_critic:
            with torch.no_grad():
                value = self.critic(self.current_state)
            self.values.append(value.clone())

        # Convert action to architecture description
        architecture = self._action_to_architecture(action.cpu().numpy())

        return {
            'architecture': architecture,
            'instruction': self._build_instruction(architecture),
            'metadata': {
                'action': action.cpu().numpy().tolist(),
                'log_prob': log_prob.item(),
            },
        }

    def update(self, reward: RewardComponents) -> None:
        """根据奖励更新策略"""
        reward_scalar = reward.to_scalar(self.config.get('reward_weights'))
        self.rewards.append(reward_scalar)

        # Batch update when enough samples collected
        if len(self.rewards) >= self.batch_size:
            self._update_policy()

    def _update_policy(self) -> None:
        """PPO策略更新"""
        if len(self.rewards) < self.batch_size:
            return

        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)

        # Compute advantages
        if self.use_critic:
            # Standard PPO: use Critic
            values = torch.stack(self.values).squeeze()
            advantages = rewards - values.detach()
        else:
            # Critic-Free: use Running Mean Baseline
            if self.baseline is None:
                self.baseline = rewards.mean().item()
            else:
                self.baseline = (
                    self.baseline_momentum * self.baseline +
                    (1 - self.baseline_momentum) * rewards.mean().item()
                )
            advantages = rewards - self.baseline

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        for _ in range(self.n_epochs):
            mean, std = self.policy(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # Ratio and clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()
            loss = policy_loss - self.entropy_coef * entropy

            # Update policy
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

            # Update critic if used
            if self.use_critic:
                values_pred = self.critic(states).squeeze()
                critic_loss = F.mse_loss(values_pred, rewards)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        if self.values:
            self.values.clear()

        # Update state for next iteration
        self.current_state = torch.randn(self.state_dim).to(self.device)

    def _action_to_architecture(self, action: np.ndarray) -> Dict[str, Any]:
        """将动作转换为架构描述"""
        # Simplified: use action values to select architecture components
        arch_types = ['attention', 'conv', 'mlp', 'transformer', 'hybrid']
        fusion_types = ['early', 'late', 'middle', 'hierarchical']

        arch_idx = int(abs(action[0]) * 100) % len(arch_types)
        fusion_idx = int(abs(action[1]) * 100) % len(fusion_types)

        return {
            'type': arch_types[arch_idx],
            'fusion_type': fusion_types[fusion_idx],
            'hidden_dim': int(256 + abs(action[2]) * 768),
            'num_layers': int(2 + abs(action[3]) * 6),
            'dropout': float(abs(action[4]) * 0.5),
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