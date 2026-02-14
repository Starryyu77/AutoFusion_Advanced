"""
CrossModalAttention: 跨模态注意力融合

设计思路:
- 使用Transformer风格的注意力机制
- 一个模态作为Query，另一个作为Key/Value
- 捕捉模态间的细粒度对齐

参考: "Attention is All You Need" (NeurIPS 2017)
       "ViLBERT" (NeurIPS 2019)
       "LXMERT" (EMNLP 2019)
"""

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合模块

    架构:
    1. Linear projection to hidden_dim
    2. Cross-attention (vision attends to language)
    3. Feed-forward network
    4. Residual connections + LayerNorm
    """

    def __init__(
        self,
        vision_dim: int = 768,
        language_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            vision_dim: 视觉特征维度
            language_dim: 文本特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout率
        """
        super().__init__()

        # 投影层
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

        # 跨模态注意力层
        self.attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

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

        # 添加序列维度 [batch, 1, hidden]
        v = v.unsqueeze(1)
        l = l.unsqueeze(1)

        # 跨模态注意力
        for layer in self.attention_layers:
            v = layer(v, l)

        # 移除序列维度
        v = v.squeeze(1)

        # 输出投影
        output = self.output_proj(v)

        return output


class CrossModalAttentionLayer(nn.Module):
    """单层跨模态注意力"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, hidden] - 通常是视觉特征
            key_value: [batch, seq_len, hidden] - 通常是文本特征

        Returns:
            output: [batch, seq_len, hidden]
        """
        # 跨模态注意力 + 残差
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        # 前馈网络 + 残差
        ffn_out = self.ffn(query)
        output = self.norm2(query + ffn_out)

        return output
