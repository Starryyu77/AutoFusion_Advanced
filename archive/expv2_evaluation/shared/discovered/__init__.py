"""
Phase 3 发现的Top 10多模态融合架构

这些架构通过AutoFusion NAS自动发现，经过3 epochs few-shot评估筛选。
现在用于完整100 epochs评估，与传统人工设计基线对比。

排名基于3 epochs奖励 (可能与100 epochs性能不同):
1. arch_024 (0.952) - Hybrid: Bilinear+Transformer
2. arch_019 (0.933) - Attention+MLP
3. arch_021 (0.933) - Pure MLP
4. arch_012 (0.906) - Cross-Modal Transformer
5. arch_025 (0.899) - Hybrid Attention
6. arch_004 (0.873) - MLP+Attention
7. arch_022 (0.873) - Pure MLP
8. arch_015 (0.850) - Gated Attention
9. arch_008 (0.825) - Hybrid Bilinear
10. arch_017 (0.819) - Attention+MLP
"""

from .arch_024 import FusionModule as Arch024
from .arch_019 import FusionModule as Arch019
from .arch_021 import FusionModule as Arch021
from .arch_012 import FusionModule as Arch012
from .arch_025 import FusionModule as Arch025
from .arch_004 import FusionModule as Arch004
from .arch_022 import FusionModule as Arch022
from .arch_015 import FusionModule as Arch015
from .arch_008 import FusionModule as Arch008
from .arch_017 import FusionModule as Arch017

# 架构注册表
DISCOVERED_ARCHITECTURES = {
    'arch_024': Arch024,
    'arch_019': Arch019,
    'arch_021': Arch021,
    'arch_012': Arch012,
    'arch_025': Arch025,
    'arch_004': Arch004,
    'arch_022': Arch022,
    'arch_015': Arch015,
    'arch_008': Arch008,
    'arch_017': Arch017,
}

# 3 epochs评估奖励 (用于参考)
REWARDS_3EPOCHS = {
    'arch_024': 0.952,
    'arch_019': 0.933,
    'arch_021': 0.933,
    'arch_012': 0.906,
    'arch_025': 0.899,
    'arch_004': 0.873,
    'arch_022': 0.873,
    'arch_015': 0.850,
    'arch_008': 0.825,
    'arch_017': 0.819,
}

__all__ = [
    'Arch024', 'Arch019', 'Arch021', 'Arch012', 'Arch025',
    'Arch004', 'Arch022', 'Arch015', 'Arch008', 'Arch017',
    'DISCOVERED_ARCHITECTURES',
    'REWARDS_3EPOCHS',
]
