# Phase 3: Ablation Studies

## Overview

Systematic ablation studies to understand the contribution of each component.

## Ablation Categories

### 3.1 Search Space Ablation

| ID | Configuration | Description |
|----|--------------|-------------|
| full | Full search space | All operations allowed |
| no_attention | Remove attention | Test attention necessity |
| no_transformer | Remove transformer | Test transformer necessity |
| conv_only | Conv only | Simplest baseline |

### 3.2 Evaluation Strategy Ablation

| ID | Proxy Epochs | Description |
|----|-------------|-------------|
| epochs_50 | 50 epochs | Gold standard (slow) |
| epochs_10 | 10 epochs | Standard proxy |
| epochs_5 | 5 epochs | Fast proxy |
| epochs_2 | 2 epochs | Very fast proxy |
| epochs_1 | 1 epoch | Ultra-fast proxy |

### 3.3 Fusion Architecture Ablation

Compare with mainstream MLLM fusion strategies:

| ID | Method | Description |
|----|--------|-------------|
| llava_mlp | LLaVA-style | Simple MLP projection |
| blip_qformer | BLIP-style | Learnable query (Q-Former) |
| ours_full | Searched (full) | Best architecture from search |
| ours_sparse | Searched (sparse) | Best + sparsity constraint |

## Quick Start

```bash
# Run all ablation studies
./run_all.sh

# Or run specific ablation
./run_ablation.sh search_space    # Search space ablation
./run_ablation.sh eval_epochs     # Evaluation strategy ablation
./run_ablation.sh fusion_arch     # Fusion architecture ablation
```

## Output

Results are saved to:
```
results/
├── search_space/
│   ├── full/
│   ├── no_attention/
│   └── ...
├── eval_epochs/
│   ├── epochs_50/
│   ├── epochs_10/
│   └── ...
└── fusion_arch/
    ├── llava_mlp/
    ├── blip_qformer/
    └── ...
```
