"""
Architecture Templates for LLM-Driven NAS
=========================================

预定义的架构模板，LLM 只需选择模板类型和参数。
目标是提高代码编译成功率。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import textwrap


@dataclass
class ArchitectureTemplate:
    """架构模板定义"""

    name: str
    description: str
    params: Dict[str, List[Any]]
    code_template: str


# ============================================================
# 架构模板定义
# ============================================================

ATTENTION_TEMPLATE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    Cross-modal Attention Fusion
    
    使用注意力机制融合视觉和语言特征。
    """
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_heads={num_heads}, dropout={dropout}):
        super().__init__()
        
        # 投影层
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # 交叉注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, vision_features, language_features):
        # 投影
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # 交叉注意力: vision attends to language
        v_expanded = v.unsqueeze(1)  # [B, 1, H]
        l_expanded = l.unsqueeze(1)  # [B, 1, H]
        
        attn_out, _ = self.attention(
            query=v_expanded,
            key=l_expanded,
            value=l_expanded
        )
        
        # 残差连接
        v = self.norm1(v + attn_out.squeeze(1))
        
        # FFN
        v = self.norm2(v + self.ffn(v))
        
        # 输出
        output = self.output_proj(v)
        return output
'''

GATED_TEMPLATE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    Gated Fusion
    
    使用门控机制融合视觉和语言特征。
    """
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, gate_type="{gate_type}"):
        super().__init__()
        
        self.gate_type = gate_type
        
        # 投影层
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.language_proj = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 门控机制
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, vision_features, language_features):
        # 投影
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # 计算门控权重
        concat = torch.cat([v, l], dim=-1)
        gate_logits = self.gate_net(concat)
        
        if self.gate_type == "sigmoid":
            gate = torch.sigmoid(gate_logits)
        elif self.gate_type == "softmax":
            gate = torch.softmax(gate_logits, dim=-1)
        elif self.gate_type == "tanh":
            gate = torch.tanh(gate_logits)
        else:
            gate = torch.sigmoid(gate_logits)
        
        # 门控融合
        fused = gate * v + (1 - gate) * l
        
        # 输出
        output = self.output_proj(fused)
        return output
'''

TRANSFORMER_TEMPLATE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    Transformer-based Fusion
    
    使用 Transformer 编码器融合视觉和语言特征。
    """
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}):
        super().__init__()
        
        # 投影层
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, vision_features, language_features):
        # 投影
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # 拼接为序列
        combined = torch.stack([v, l], dim=1)  # [B, 2, H]
        
        # Transformer 编码
        encoded = self.transformer(combined)
        
        # 取平均
        fused = encoded.mean(dim=1)  # [B, H]
        
        # 输出
        output = self.output_proj(fused)
        return output
'''

MLP_TEMPLATE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    Simple MLP Fusion
    
    使用简单的 MLP 融合视觉和语言特征。
    """
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(vision_dim + language_dim, hidden_dim)
        
        # MLP 层
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        self.mlp = nn.Sequential(*layers)
        
        # 输出
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, vision_features, language_features):
        # 拼接
        combined = torch.cat([vision_features, language_features], dim=-1)
        
        # MLP
        x = self.input_proj(combined)
        x = self.mlp(x)
        
        # 输出
        output = self.output_proj(x)
        return output
'''

HYBRID_TEMPLATE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    Hybrid Fusion (Attention + Gating)
    
    结合注意力和门控机制的混合融合。
    """
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_heads={num_heads}):
        super().__init__()
        
        # 共享投影
        def make_proj(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
        
        self.vision_proj = make_proj(vision_dim)
        self.language_proj = make_proj(language_dim)
        
        # 轻量级注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 输出
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, vision_features, language_features):
        # 投影
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        
        # 交叉注意力
        attended, _ = self.attention(
            query=v.unsqueeze(1),
            key=l.unsqueeze(1),
            value=l.unsqueeze(1)
        )
        attended = self.attn_norm(attended.squeeze(1) + v)
        
        # 门控
        gate_input = torch.cat([attended, l], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * attended + (1 - gate) * l
        
        # 输出
        output = self.output_norm(self.output_proj(fused))
        return output
'''


# ============================================================
# 模板注册表
# ============================================================

ARCHITECTURE_TEMPLATES = {
    "attention": ArchitectureTemplate(
        name="attention",
        description="Cross-modal Attention Fusion - 使用注意力机制融合",
        params={
            "hidden_dim": [32, 48, 64, 96, 128],
            "num_heads": [1, 2, 4],
            "dropout": [0.0, 0.1, 0.2],
        },
        code_template=ATTENTION_TEMPLATE,
    ),
    "gated": ArchitectureTemplate(
        name="gated",
        description="Gated Fusion - 使用门控机制融合",
        params={
            "hidden_dim": [32, 48, 64, 96, 128],
            "gate_type": ["sigmoid", "tanh", "softmax"],
        },
        code_template=GATED_TEMPLATE,
    ),
    "transformer": ArchitectureTemplate(
        name="transformer",
        description="Transformer-based Fusion - 使用 Transformer 编码器",
        params={
            "hidden_dim": [64, 96, 128],
            "num_layers": [1, 2, 3],
            "num_heads": [2, 4],
        },
        code_template=TRANSFORMER_TEMPLATE,
    ),
    "mlp": ArchitectureTemplate(
        name="mlp",
        description="Simple MLP Fusion - 使用简单的 MLP 融合",
        params={"hidden_dim": [64, 96, 128, 192], "num_layers": [1, 2, 3]},
        code_template=MLP_TEMPLATE,
    ),
    "hybrid": ArchitectureTemplate(
        name="hybrid",
        description="Hybrid Fusion (Attention + Gating) - 混合融合机制",
        params={"hidden_dim": [32, 48, 64, 96, 128], "num_heads": [1, 2, 4]},
        code_template=HYBRID_TEMPLATE,
    ),
}


def get_template(name: str) -> Optional[ArchitectureTemplate]:
    """获取指定名称的模板"""
    return ARCHITECTURE_TEMPLATES.get(name)


def get_all_templates() -> Dict[str, ArchitectureTemplate]:
    """获取所有模板"""
    return ARCHITECTURE_TEMPLATES


def generate_code(template_name: str, params: Dict[str, Any]) -> str:
    """
    根据模板和参数生成代码

    Args:
        template_name: 模板名称
        params: 参数字典

    Returns:
        生成的代码字符串
    """
    template = get_template(template_name)
    if template is None:
        raise ValueError(f"Unknown template: {template_name}")

    # 使用参数填充模板
    code = template.code_template.format(**params)

    # 清理多余空行
    code = textwrap.dedent(code).strip()

    return code


def validate_params(template_name: str, params: Dict[str, Any]) -> bool:
    """
    验证参数是否在有效范围内

    Args:
        template_name: 模板名称
        params: 参数字典

    Returns:
        是否有效
    """
    template = get_template(template_name)
    if template is None:
        return False

    for key, value in params.items():
        if key not in template.params:
            return False
        if value not in template.params[key]:
            return False

    return True


def get_default_params(template_name: str) -> Dict[str, Any]:
    """获取模板的默认参数"""
    template = get_template(template_name)
    if template is None:
        return {}

    return {key: values[0] for key, values in template.params.items()}


if __name__ == "__main__":
    # 测试代码生成
    for name in ARCHITECTURE_TEMPLATES:
        print(f"\n{'=' * 60}")
        print(f"Template: {name}")
        print(f"{'=' * 60}")

        params = get_default_params(name)
        print(f"Default params: {params}")

        code = generate_code(name, params)
        print(f"\nGenerated code (first 500 chars):")
        print(code[:500] + "...")

        # 尝试编译
        try:
            exec(code)
            print(f"\n✅ Compilation successful!")
        except Exception as e:
            print(f"\n❌ Compilation failed: {e}")
