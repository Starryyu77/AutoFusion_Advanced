"""
ConcatMLP: 最简单的多模态融合基线

设计思路:
- 将视觉和文本特征拼接
- 通过MLP进行融合
- 最常用、最简单的基线方法

参考: Deep Learning for Visual-Language Tasks (2017+)
"""

import torch
import torch.nn as nn


class ConcatMLP(nn.Module):
    """
    拼接 + MLP 融合模块

    架构:
    1. Linear projection to same dimension
    2. Concatenate vision and language features
    3. MLP for fusion
    4. Output projection
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
            vision_dim: 视觉特征维度 (CLIP-ViT-L: 768)
            language_dim: 文本特征维度 (CLIP: 768)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: MLP层数
            dropout: Dropout率
        """
        super().__init__()

        # 投影到相同维度
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

        # 拼接后维度是 2 * hidden_dim
        concat_dim = hidden_dim * 2

        # 构建MLP
        layers = []
        for i in range(num_layers):
            in_dim = concat_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.fusion_mlp = nn.Sequential(*layers)

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
        # 投影
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)

        # 拼接
        fused = torch.cat([v, l], dim=-1)

        # MLP融合
        fused = self.fusion_mlp(fused)

        # 输出投影
        output = self.output_proj(fused)

        return output
