"""
GDPO Controller
---------------
Group Decoupled Policy Optimization with variance explosion protection.

Key innovation: Decoupled normalization per reward component.
- Each reward component normalized independently
- Prevents high-variance components from drowning low-variance ones
- Variance explosion protection (min_std threshold, advantage clipping)

Theory correction: When r_valid has variance -> 0, don't divide by std.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseController, RewardComponents


class GDPOController(BaseController):
    """
    Group Decoupled Policy Optimization (GDPO)

    Key difference from GRPO:
    - GRPO: A_i = (R_i - mean(R)) / (std(R) + epsilon)  [scalar reward]
    - GDPO: A_i = sum_k w_k * (r_k_i - mean(r_k)) / (std(r_k) + epsilon)  [decoupled]

    Advantages:
    1. Each reward component normalized independently
    2. High-variance components (e.g., compile_success) don't drown low-variance ones
    3. Better multi-objective learning

    Variance explosion protection:
    - min_std: Don't divide by std when std < threshold
    - advantage_clip: Clip normalized advantages to prevent extreme values
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # GDPO hyperparameters
        self.group_size = config.get('group_size', 8)
        self.learning_rate = config.get('learning_rate', 1e-5)
        self.beta = config.get('beta', 0.04)
        self.clip_range = config.get('clip_range', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.n_epochs = config.get('n_epochs', 2)

        # Variance explosion protection
        self.advantage_clip = config.get('advantage_clip', 3.0)
        self.min_std = config.get('min_std', 1e-4)
        self.use_robust_norm = config.get('use_robust_norm', False)
        self.auto_weight_scaling = config.get('auto_weight_scaling', True)

        # Reward keys for decoupled normalization
        self.reward_keys = ['accuracy', 'efficiency', 'compile_success', 'complexity']
        self.reward_weights = config.get('reward_weights', {
            'accuracy': 1.0,
            'efficiency': 0.5,
            'compile_success': 2.0,
            'complexity': 0.0,
        })

        # State dimensions
        self.state_dim = config.get('state_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.action_dim = config.get('action_dim', 16)

        # Initialize policy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = self._build_policy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Reference policy
        self.reference_policy = None

        # Group memory (stores dict rewards for GDPO)
        self.group_states = []
        self.group_actions = []
        self.group_rewards = []  # List of dicts for GDPO

        self.current_state = torch.randn(self.state_dim).to(self.device)

    def _build_policy(self) -> nn.Module:
        """构建策略网络"""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh(),
        )

    def propose(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        """提出架构候选"""
        with torch.no_grad():
            action = self.policy(self.current_state.unsqueeze(0)).squeeze(0)
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1, 1)

        self.group_states.append(self.current_state.clone())
        self.group_actions.append(action.clone())

        architecture = self._action_to_architecture(action.cpu().numpy())

        return {
            'architecture': architecture,
            'instruction': self._build_instruction(architecture),
            'metadata': {
                'action': action.cpu().numpy().tolist(),
                'group_size': self.group_size,
                'algorithm': 'GDPO',
            },
        }

    def update(self, reward: RewardComponents) -> None:
        """收集字典形式奖励，当组满时更新"""
        # GDPO: 保存字典形式奖励用于解耦归一化
        reward_dict = reward.to_dict()
        self.group_rewards.append(reward_dict)

        if len(self.group_rewards) >= self.group_size:
            self._update_group()

    def _update_group(self) -> None:
        """GDPO组更新 - 解耦归一化"""
        if len(self.group_rewards) < self.group_size:
            return

        # 转换为tensor
        states = torch.stack(self.group_states[:self.group_size])
        actions = torch.stack(self.group_actions[:self.group_size])

        # GDPO: 对每个奖励分量独立归一化
        advantages = self._compute_decoupled_advantages(
            self.group_rewards[:self.group_size]
        )

        # 保存参考策略
        if self.reference_policy is None:
            self.reference_policy = self._copy_policy()

        # GDPO更新
        for _ in range(self.n_epochs):
            current_actions = self.policy(states)

            # 策略梯度
            weighted_actions = current_actions * advantages.unsqueeze(1)
            policy_loss = -weighted_actions.mean()

            # KL散度惩罚
            with torch.no_grad():
                ref_actions = self.reference_policy(states)
            kl_loss = F.mse_loss(current_actions, ref_actions)

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

    def _compute_decoupled_advantages(self, rewards: List[Dict[str, float]]) -> torch.Tensor:
        """
        解耦归一化优势计算

        A_i = sum_k w_k * normalize(r_k_i)

        其中 normalize 根据配置选择:
        - 标准: (r - mean) / (std + eps)
        - 保护: 当 std < min_std 时不除以 std
        - 稳健: 分位数归一化
        """
        advantages = torch.zeros(self.group_size, device=self.device)

        for key in self.reward_keys:
            if key not in rewards[0]:
                continue

            values = torch.tensor([r[key] for r in rewards], dtype=torch.float32, device=self.device)
            mean = values.mean()
            std = values.std()

            # 方差爆炸保护
            if std < self.min_std:
                # 修正1: 低方差保护 - 只用均值中心化，不除以 std
                normalized = values - mean
            else:
                # 标准归一化
                normalized = (values - mean) / (std + 1e-8)

            # 修正2: Clip 防止极端值
            normalized = torch.clamp(normalized, -self.advantage_clip, self.advantage_clip)

            # 加权
            weight = self.reward_weights.get(key, 1.0)
            advantages += weight * normalized

        return advantages

    def _robust_normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        稳健归一化 (分位数-based)

        使用IQR代替std，对异常值更稳健
        """
        median = values.median()
        q25 = torch.quantile(values, 0.25)
        q75 = torch.quantile(values, 0.75)
        iqr = q75 - q25 + 1e-8

        return (values - median) / iqr

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