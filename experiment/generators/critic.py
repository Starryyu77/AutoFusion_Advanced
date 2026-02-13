"""
Critic Generator
----------------
Self-evaluation and refinement strategy.
"""

from typing import Dict, Any, List
from base import BaseGenerator, GenerationResult


class CriticGenerator(BaseGenerator):
    """
    Critic Prompt Strategy

    特点:
    1. 自我评估
    2. 迭代改进
    3. 多轮refinement
    """

    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__(llm_client, config)
        self.num_iterations = config.get('num_iterations', 2)

    def build_prompt(self, architecture_desc: Dict[str, Any]) -> str:
        """构建Critic Prompt"""
        arch_type = architecture_desc.get('type', 'attention')
        fusion_type = architecture_desc.get('fusion_type', 'middle')
        hidden_dim = architecture_desc.get('hidden_dim', 512)
        num_layers = architecture_desc.get('num_layers', 4)

        return f"""You are an expert PyTorch developer and code reviewer.

## Task
Implement a multimodal fusion module with the following specification:
- Type: {arch_type}
- Fusion Strategy: {fusion_type}
- Hidden Dimension: {hidden_dim}
- Number of Layers: {num_layers}

## Process

Step 1: Initial Implementation
Write an initial version of the fusion module.

Step 2: Self-Critique
Review your implementation and identify:
1. Potential bugs or issues
2. Performance bottlenecks
3. Missing error handling
4. Suboptimal design choices

Step 3: Refinement
Improve the implementation based on your critique.

## Output Format
Provide the FINAL refined code only:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        # Refined implementation

    def forward(self, vision_features, language_features):
        # Refined implementation
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
                    metadata={'strategy': 'critic', 'iterations': self.num_iterations},
                    success=is_valid,
                    error=error if not is_valid else None
                ))
            except Exception as e:
                results.append(GenerationResult(
                    code="",
                    prompt=prompt,
                    metadata={'strategy': 'critic'},
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

class FusionModule(nn.Module):
    """
    Refined multimodal fusion module with:
    - Proper normalization
    - Residual connections
    - Attention mechanism
    """

    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.language_proj = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Cross-attention layers
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])

        # Feed-forward layers
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])

        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, vision_features, language_features):
        # Project inputs
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)

        # Cross-attention fusion
        fused = l
        for i in range(len(self.cross_attn)):
            # Self-attention on language with vision as context
            attn_out, _ = self.cross_attn[i](
                fused.unsqueeze(1),
                v.unsqueeze(1),
                v.unsqueeze(1)
            )
            fused = self.norm1[i](fused + attn_out.squeeze(1))

            # Feed-forward
            ffn_out = self.ffn[i](fused)
            fused = self.norm2[i](fused + ffn_out)

        return fused
'''
