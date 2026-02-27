"""
传统人工设计的多模态融合基线架构

包含5个经典基线:
1. ConcatMLP - 简单拼接+MLP (最常用)
2. BilinearPooling - 双线性池化
3. CrossModalAttention - 跨模态注意力
4. CLIPFusion - CLIP风格投影
5. FiLM - Feature-wise Linear Modulation
"""

from .concat_mlp import ConcatMLP
from .bilinear_pooling import BilinearPooling
from .cross_modal_attention import CrossModalAttention
from .clip_fusion import CLIPFusion
from .film import FiLM

__all__ = [
    'ConcatMLP',
    'BilinearPooling',
    'CrossModalAttention',
    'CLIPFusion',
    'FiLM',
]
