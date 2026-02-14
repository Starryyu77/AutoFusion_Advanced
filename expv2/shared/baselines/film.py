"""
FiLM: Feature-wise Linear Modulation

设计思路:
- 一个模态生成调制参数 (gamma, beta)
- 用这些参数对另一个模态进行特征调制
- 条件化的特征变换

参考: "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)
       "FiLM: A General Visual Reasoning Architecture" (arXiv 2017)
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation 融合模块

    架构:
    1. 文本特征生成 gamma 和 beta
    2. 对视觉特征进行调制: gamma * vision + beta
    3. 后处理网络
    """

    def __init__(
        self,
        vision_dim: int = 768,
        language_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            vision_dim: 视觉特征维度
            language_dim: 文本特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 后处理MLP层数
            dropout: Dropout率
        """
        super().__init__()

        # 视觉投影
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # FiLM参数生成器 (从文本生成 gamma 和 beta)
        self.film_generator = nn.Sequential(
            nn.Linear(language_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)  # 输出 gamma 和 beta
        )

        # 后处理网络
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.post_process = nn.Sequential(*layers)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, vision_dim]
            language_features: [batch_size, language_dim]

        Returns:
            fused_features: [batch_size, output_dim]
        """
        # 投影视觉特征
        v = self.vision_proj(vision_features)

        # 生成FiLM参数
        film_params = self.film_generator(language_features)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        # FiLM调制: gamma * v + beta
        v = gamma * v + beta

        # 后处理
        v = self.post_process(v)

        # 输出投影
        output = self.output_proj(v)

        return output
