"""
CLIPFusion: CLIP风格的简单线性投影

设计思路:
- 分别投影视觉和文本特征
- 直接相加或相乘
- 极简设计，参数最少

参考: "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
"""

import torch
import torch.nn as nn


class CLIPFusion(nn.Module):
    """
    CLIP风格融合模块

    架构:
    1. Linear projection to same dimension
    2. Simple fusion (add or multiply)
    3. Optional normalization
    """

    def __init__(
        self,
        vision_dim: int = 768,
        language_dim: int = 768,
        output_dim: int = 768,
        fusion_type: str = 'add',  # 'add', 'multiply', 'both'
        normalize: bool = True
    ):
        """
        Args:
            vision_dim: 视觉特征维度
            language_dim: 文本特征维度
            output_dim: 输出维度
            fusion_type: 融合方式 ('add', 'multiply', 'both')
            normalize: 是否进行L2归一化
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.normalize = normalize

        # 投影到输出维度
        self.vision_proj = nn.Linear(vision_dim, output_dim)
        self.language_proj = nn.Linear(language_dim, output_dim)

        # 'both'模式需要额外的投影
        if fusion_type == 'both':
            self.fusion_proj = nn.Linear(output_dim * 2, output_dim)

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

        # L2归一化
        if self.normalize:
            v = torch.nn.functional.normalize(v, p=2, dim=-1)
            l = torch.nn.functional.normalize(l, p=2, dim=-1)

        # 融合
        if self.fusion_type == 'add':
            fused = v + l
        elif self.fusion_type == 'multiply':
            fused = v * l
        elif self.fusion_type == 'both':
            fused = torch.cat([v + l, v * l], dim=-1)
            fused = self.fusion_proj(fused)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return fused
