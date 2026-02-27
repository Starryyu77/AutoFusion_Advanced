"""
Few-Shot Generator
------------------
Example-based code generation.
"""

from typing import Dict, Any, List
from base import BaseGenerator, GenerationResult


class FewShotGenerator(BaseGenerator):
    """
    Few-Shot Prompt Strategy

    特点:
    1. 基于示例学习
    2. 提供多个参考实现
    3. 模仿示例风格
    """

    def __init__(self, llm_client, config: Dict[str, Any]):
        super().__init__(llm_client, config)
        self.num_examples = config.get('num_examples', 3)
        self.examples = self._load_examples()

    def _load_examples(self) -> List[Dict[str, str]]:
        """加载示例"""
        return [
            {
                "description": "Cross-attention fusion with residual connection",
                "code": """
class CrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim, language_dim, hidden_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision, language):
        v = self.vision_proj(vision)
        l = self.language_proj(language)
        attn_out, _ = self.cross_attn(l.unsqueeze(1), v.unsqueeze(1), v.unsqueeze(1))
        return self.norm(attn_out.squeeze(1) + l)
"""
            },
            {
                "description": "Gated multimodal fusion",
                "code": """
class GatedFusion(nn.Module):
    def __init__(self, vision_dim, language_dim, hidden_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, vision, language):
        v = self.vision_proj(vision)
        l = self.language_proj(language)
        gate = self.gate(torch.cat([v, l], dim=-1))
        return gate * v + (1 - gate) * l
"""
            },
            {
                "description": "Transformer-based fusion",
                "code": """
class TransformerFusion(nn.Module):
    def __init__(self, vision_dim, language_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, vision, language):
        v = self.vision_proj(vision).unsqueeze(1)
        l = self.language_proj(language).unsqueeze(1)
        concat = torch.cat([v, l], dim=1)
        return self.transformer(concat).mean(dim=1)
"""
            }
        ]

    def build_prompt(self, architecture_desc: Dict[str, Any]) -> str:
        """构建Few-Shot Prompt"""
        arch_type = architecture_desc.get('type', 'attention')
        fusion_type = architecture_desc.get('fusion_type', 'middle')
        hidden_dim = architecture_desc.get('hidden_dim', 512)
        num_layers = architecture_desc.get('num_layers', 4)

        # 选择相关示例
        selected_examples = self.examples[:self.num_examples]

        prompt = """You are an expert PyTorch developer. Here are some examples of multimodal fusion modules:

"""

        for i, example in enumerate(selected_examples, 1):
            prompt += f"""Example {i}: {example['description']}
{example['code']}

"""

        prompt += f"""Now, implement a new fusion module based on the following specification:

## Target Architecture
- Type: {arch_type}
- Fusion Strategy: {fusion_type}
- Hidden Dimension: {hidden_dim}
- Number of Layers: {num_layers}

Requirements:
1. Follow the style of the examples above
2. Use PyTorch nn.Module
3. Handle vision and language features
4. Return fused representation of shape (batch, {hidden_dim})

Provide ONLY the Python code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        # Implementation

    def forward(self, vision_features, language_features):
        # Implementation
        pass
```
"""
        return prompt

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
                    metadata={'strategy': 'fewshot', 'num_examples': self.num_examples},
                    success=is_valid,
                    error=error if not is_valid else None
                ))
            except Exception as e:
                results.append(GenerationResult(
                    code="",
                    prompt=prompt,
                    metadata={'strategy': 'fewshot'},
                    success=False,
                    error=str(e)
                ))

        return results

    def _call_llm(self, prompt: str) -> str:
        """调用LLM API - 强制使用真实API，失败时抛出异常"""
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
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim*4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)

        # Gated combination
        gate = self.gate(torch.cat([v, l], dim=-1))
        fused = gate * v + (1 - gate) * l

        # Apply fusion layers
        fused = fused.unsqueeze(1)
        for layer in self.fusion_layers:
            fused = layer(fused)

        return self.norm(fused.squeeze(1))
'''
