"""
BilinearPooling: 双线性池化融合

设计思路:
- 使用外积建模模态间交互
- 捕捉元素级别的相关性
- 经典的多模态融合方法

参考: "Compact Bilinear Pooling" (CVPR 2016)
       "Low-rank Bilinear Pooling" (ICCV 2017)
"""

import torch
import torch.nn as nn


class BilinearPooling(nn.Module):
    """
    双线性池化融合模块

    架构:
    1. Linear projection to same dimension
    2. Element-wise multiplication (bilinear interaction)
    3. Normalization
    4. MLP for refinement
    """

    def __init__(
        self,
        vision_dim: int = 768,
        language_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        normalize: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            vision_dim: 视觉特征维度
            language_dim: 文本特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            normalize: 是否进行L2归一化
            dropout: Dropout率
        """
        super().__init__()

        self.normalize = normalize

        # 投影到相同维度
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

        # 双线性交互后的处理
        self.post_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

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

        # L2归一化 (可选)
        if self.normalize:
            v = torch.nn.functional.normalize(v, p=2, dim=-1)
            l = torch.nn.functional.normalize(l, p=2, dim=-1)

        # 双线性交互 (元素级相乘)
        fused = v * l

        # 后处理
        output = self.post_fusion(fused)

        return output
