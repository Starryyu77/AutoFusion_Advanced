"""
Extended Search Space for Architecture Discovery
-----------------------------------------------
Comprehensive search space for multimodal fusion architectures.
"""

from typing import Dict, Any, List
import random


class SearchSpace:
    """Extended search space for Phase 3 architecture discovery"""

    # Fusion operation types
    FUSION_TYPES = [
        'attention',        # Multi-head attention fusion
        'bilinear',         # Bilinear pooling
        'mlp',              # MLP-based fusion
        'transformer',      # Transformer encoder
        'gated',            # Gated fusion
        'cross_modal',      # Cross-modal attention
        'hybrid',           # Combination of above
    ]

    # Activation functions
    ACTIVATIONS = ['gelu', 'relu', 'silu', 'swish', 'mish', 'tanh']

    # Normalization types
    NORMALIZATIONS = ['layer_norm', 'batch_norm', 'instance_norm', 'none']

    # Connectivity patterns
    CONNECTIVITY = ['serial', 'parallel', 'residual_dense', 'densenet_style']

    @classmethod
    def get_full_space(cls) -> Dict[str, Any]:
        """Get full extended search space"""
        return {
            # Fusion type
            'fusion_type': cls.FUSION_TYPES,

            # Architecture dimensions
            'num_fusion_layers': {'type': 'int', 'low': 1, 'high': 6},
            'hidden_dim': {'type': 'int', 'low': 128, 'high': 1024, 'step': 64},
            'num_heads': {'type': 'int', 'low': 2, 'high': 16, 'step': 2},
            'intermediate_ratio': {'type': 'float', 'low': 2.0, 'high': 8.0, 'step': 0.5},

            # Activation & Normalization
            'activation': cls.ACTIVATIONS,
            'normalization': cls.NORMALIZATIONS,

            # Regularization
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.05},
            'drop_path_rate': {'type': 'float', 'low': 0.0, 'high': 0.3, 'step': 0.05},
            'weight_decay': {'type': 'float', 'low': 0.0, 'high': 0.1, 'step': 0.01},

            # Special components
            'use_residual': [True, False],
            'use_gating': [True, False],
            'use_position_embedding': [True, False],
            'use_layer_scale': [True, False],
            'use_stochastic_depth': [True, False],

            # Connectivity
            'connectivity': cls.CONNECTIVITY,

            # Attention-specific
            'attention_dropout': {'type': 'float', 'low': 0.0, 'high': 0.3, 'step': 0.05},
            'attention_type': ['standard', 'linear', 'performer', 'local'],
        }

    @classmethod
    def get_compact_space(cls) -> Dict[str, Any]:
        """Get compact search space (faster search)"""
        return {
            'fusion_type': ['attention', 'mlp', 'transformer', 'gated'],
            'num_fusion_layers': {'type': 'int', 'low': 2, 'high': 4},
            'hidden_dim': {'type': 'int', 'low': 256, 'high': 768, 'step': 128},
            'num_heads': {'type': 'int', 'low': 4, 'high': 8, 'step': 2},
            'activation': ['gelu', 'silu'],
            'normalization': ['layer_norm', 'none'],
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.3, 'step': 0.1},
            'use_residual': [True],
            'use_gating': [True, False],
        }

    @classmethod
    def sample_architecture(cls, search_space: Dict[str, Any] = None) -> Dict[str, Any]:
        """Sample random architecture from search space"""
        if search_space is None:
            search_space = cls.get_full_space()

        arch = {}
        for key, space in search_space.items():
            if isinstance(space, list):
                arch[key] = random.choice(space)
            elif isinstance(space, dict):
                if space['type'] == 'int':
                    low, high = space['low'], space['high']
                    step = space.get('step', 1)
                    arch[key] = random.randrange(low, high + 1, step)
                elif space['type'] == 'float':
                    low, high = space['low'], space['high']
                    step = space.get('step', 0.01)
                    num_steps = int((high - low) / step)
                    arch[key] = low + random.randint(0, num_steps) * step

        return arch

    @classmethod
    def mutate_architecture(cls, arch: Dict[str, Any], mutation_rate: float = 0.3,
                            search_space: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mutate architecture with given probability"""
        if search_space is None:
            search_space = cls.get_full_space()

        mutated = arch.copy()
        for key in mutated:
            if random.random() < mutation_rate:
                space = search_space.get(key)
                if space:
                    if isinstance(space, list):
                        mutated[key] = random.choice(space)
                    elif isinstance(space, dict):
                        if space['type'] == 'int':
                            mutated[key] = random.randrange(
                                space['low'], space['high'] + 1, space.get('step', 1)
                            )
                        elif space['type'] == 'float':
                            num_steps = int((space['high'] - space['low']) / space.get('step', 0.01))
                            mutated[key] = space['low'] + random.randint(0, num_steps) * space.get('step', 0.01)

        return mutated


# Predefined architecture templates
ARCHITECTURE_TEMPLATES = {
    'transformer_fusion': {
        'fusion_type': 'transformer',
        'num_fusion_layers': 4,
        'hidden_dim': 512,
        'num_heads': 8,
        'activation': 'gelu',
        'normalization': 'layer_norm',
        'dropout': 0.1,
        'use_residual': True,
        'use_gating': False,
    },
    'attention_gated': {
        'fusion_type': 'gated',
        'num_fusion_layers': 3,
        'hidden_dim': 768,
        'num_heads': 12,
        'activation': 'silu',
        'normalization': 'layer_norm',
        'dropout': 0.15,
        'use_residual': True,
        'use_gating': True,
    },
    'mlp_baseline': {
        'fusion_type': 'mlp',
        'num_fusion_layers': 2,
        'hidden_dim': 512,
        'activation': 'gelu',
        'normalization': 'none',
        'dropout': 0.1,
        'use_residual': False,
        'use_gating': False,
    },
}


def get_template(name: str) -> Dict[str, Any]:
    """Get predefined architecture template"""
    return ARCHITECTURE_TEMPLATES.get(name, {}).copy()


if __name__ == '__main__':
    # Test search space
    space = SearchSpace()

    print("Full search space dimensions:", len(space.get_full_space()))
    print("Compact search space dimensions:", len(space.get_compact_space()))

    print("\nSample architecture:")
    arch = space.sample_architecture()
    for k, v in arch.items():
        print(f"  {k}: {v}")

    print("\nMutated architecture:")
    mutated = space.mutate_architecture(arch, mutation_rate=0.3)
    for k, v in mutated.items():
        marker = " *" if v != arch[k] else ""
        print(f"  {k}: {v}{marker}")
