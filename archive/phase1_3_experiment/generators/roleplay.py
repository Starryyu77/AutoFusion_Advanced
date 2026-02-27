"""
RolePlay Generator
------------------
Expert persona simulation for code generation.
"""

from typing import Dict, Any, List
from base import BaseGenerator, GenerationResult


class RolePlayGenerator(BaseGenerator):
    """
    RolePlay Prompt Strategy

    特点:
    1. 专家角色模拟
    2. 多角度思考
    3. 领域专业知识
    """

    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__(llm_client, config)
        self.expert_type = config.get('expert_type', 'pytorch_expert')

    def _get_expert_prompt(self) -> str:
        """获取专家角色设定"""
        experts = {
            'pytorch_expert': """You are a PyTorch core contributor with 10+ years of experience.
Your expertise includes:
- Deep understanding of PyTorch internals
- Optimization techniques for GPU/TPU
- Best practices for production ML code
- Memory-efficient implementations""",

            'multimodal_researcher': """You are a top multimodal ML researcher from Google DeepMind.
Your expertise includes:
- Vision-language models (CLIP, Flamingo, GPT-4V)
- Cross-modal attention mechanisms
- Multimodal fusion architectures
- State-of-the-art benchmark results""",

            'software_architect': """You are a senior software architect specializing in ML systems.
Your expertise includes:
- Clean code architecture
- Design patterns for ML
- Testing and validation
- Performance optimization""",

            'cuda_engineer': """You are a CUDA optimization engineer at NVIDIA.
Your expertise includes:
- GPU kernel optimization
- Memory access patterns
- CUDA-PyTorch integration
- Profiling and benchmarking""",
        }
        return experts.get(self.expert_type, experts['pytorch_expert'])

    def build_prompt(self, architecture_desc: Dict[str, Any]) -> str:
        """构建RolePlay Prompt"""
        arch_type = architecture_desc.get('type', 'attention')
        fusion_type = architecture_desc.get('fusion_type', 'middle')
        hidden_dim = architecture_desc.get('hidden_dim', 512)
        num_layers = architecture_desc.get('num_layers', 4)

        expert_persona = self._get_expert_prompt()

        return f"""{expert_persona}

## Task
Design and implement a state-of-the-art multimodal fusion module.

## Requirements
- Type: {arch_type}
- Fusion Strategy: {fusion_type}
- Hidden Dimension: {hidden_dim}
- Number of Layers: {num_layers}

## Your Approach
As an expert in this field, apply your deep knowledge to:
1. Choose the most appropriate architecture patterns
2. Implement with production-quality code
3. Optimize for both accuracy and efficiency
4. Include proper error handling and edge cases

## Output
Provide the complete PyTorch implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        # Expert implementation

    def forward(self, vision_features, language_features):
        # Expert implementation
        pass
```
"""

    def generate(self, architecture_desc: Dict[str, Any], num_samples: int = 1) -> List[GenerationResult]:
        """生成代码"""
        prompt = self.build_prompt(architecture_desc)

        results = []
        for _ in range(num_samples):
            try:
                if self.llm is not None:
                    code = self._call_llm(prompt)
                else:
                    code = self._mock_generate(architecture_desc)

                code = self.postprocess_code(code)
                is_valid, error = self.validate_code(code)

                results.append(GenerationResult(
                    code=code,
                    prompt=prompt,
                    metadata={'strategy': 'roleplay', 'expert': self.expert_type},
                    success=is_valid,
                    error=error if not is_valid else None
                ))
            except Exception as e:
                results.append(GenerationResult(
                    code="",
                    prompt=prompt,
                    metadata={'strategy': 'roleplay'},
                    success=False,
                    error=str(e)
                ))

        return results

    def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.config.get('api_key', ''),
                base_url=self.config.get('base_url', 'https://api.deepseek.com/v1')
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API call failed: {e}")
            raise RuntimeError(f"DeepSeek API call failed: {e}") from e

    def _mock_generate(self, architecture_desc: Dict[str, Any]) -> str:
        """模拟代码生成"""
        hidden_dim = architecture_desc.get('hidden_dim', 512)
        num_layers = architecture_desc.get('num_layers', 4)

        return f'''import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FusionModule(nn.Module):
    """
    Production-grade multimodal fusion module.
    Optimized for both accuracy and efficiency.
    """

    def __init__(self, vision_dim: int = 768, language_dim: int = 768,
                 hidden_dim: int = {hidden_dim}, num_layers: int = {num_layers},
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Efficient projections with bottleneck
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.language_proj = nn.Sequential(
            nn.Linear(language_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Multi-head cross-attention with flash attention compatible settings
        self.num_heads = 8
        assert hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True,
                bias=False  # More efficient
            )
            for _ in range(num_layers)
        ])

        # Pre-norm architecture for stable training
        self.pre_norm_v = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.pre_norm_l = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.post_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # FFN with SwiGLU-style gating (more efficient than standard FFN)
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),  # Swish activation
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features: torch.Tensor,
                language_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project to common space
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)

        # Iterative cross-modal fusion
        for i in range(len(self.cross_attn)):
            # Pre-norm
            v_norm = self.pre_norm_v[i](v)
            l_norm = self.pre_norm_l[i](l)

            # Cross-attention: language attends to vision
            attn_out, _ = self.cross_attn[i](
                query=l_norm.unsqueeze(1),
                key=v_norm.unsqueeze(1),
                value=v_norm.unsqueeze(1),
                key_padding_mask=attention_mask
            )
            l = l + attn_out.squeeze(1)

            # Post-norm and FFN
            l = l + self.ffn[i](self.post_norm[i](l))

        return self.final_norm(l)
'''