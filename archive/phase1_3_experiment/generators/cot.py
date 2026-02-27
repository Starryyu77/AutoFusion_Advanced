"""
Chain-of-Thought Generator
--------------------------
Step-by-step reasoning for code generation.
"""

from typing import Dict, Any, List
from base import BaseGenerator, GenerationResult


class ChainOfThoughtGenerator(BaseGenerator):
    """
    Chain-of-Thought (CoT) Prompt Strategy

    特点:
    1. 逐步推理
    2. 显式思考过程
    3. 结构化输出
    """

    def build_prompt(self, architecture_desc: Dict[str, Any]) -> str:
        """构建CoT Prompt"""
        arch_type = architecture_desc.get('type', 'attention')
        fusion_type = architecture_desc.get('fusion_type', 'middle')
        hidden_dim = architecture_desc.get('hidden_dim', 512)
        num_layers = architecture_desc.get('num_layers', 4)
        dropout = architecture_desc.get('dropout', 0.1)
        activation = architecture_desc.get('activation', 'gelu')

        return f"""You are an expert PyTorch developer specializing in multimodal fusion architectures.

Please implement a multimodal fusion module following these steps:

## Architecture Specification
- Type: {arch_type}
- Fusion Strategy: {fusion_type}
- Hidden Dimension: {hidden_dim}
- Number of Layers: {num_layers}
- Dropout: {dropout}
- Activation: {activation}

## Step-by-Step Implementation

Step 1: Analyze Input Modalities
- Vision features: (batch_size, vision_dim) from image encoder
- Language features: (batch_size, language_dim) from text encoder
- Consider how these modalities should interact

Step 2: Design Fusion Mechanism
- How to align different feature spaces?
- What attention/combination mechanism to use?
- How to preserve modality-specific information?

Step 3: Implement Layer Structure
- Design {num_layers} layers with {hidden_dim} hidden units
- Include {activation} activation and {dropout} dropout
- Add residual connections and layer normalization

Step 4: Write Complete PyTorch Code
- Define FusionModule class inheriting from nn.Module
- Implement __init__ and forward methods
- Handle variable input dimensions
- Return fused representation

## Output Format
Provide ONLY the Python code without explanation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        # Your implementation here

    def forward(self, vision_features, language_features):
        # Your implementation here
        # Return: fused_features of shape (batch, {hidden_dim})
        pass
```
"""

    def generate(self, architecture_desc: Dict[str, Any], num_samples: int = 1) -> List[GenerationResult]:
        """生成代码"""
        prompt = self.build_prompt(architecture_desc)

        results = []
        for _ in range(num_samples):
            try:
                # 调用LLM生成代码
                if self.llm is not None:
                    code = self._call_llm(prompt)
                else:
                    # 模拟生成 (用于测试)
                    code = self._mock_generate(architecture_desc)

                # 后处理
                code = self.postprocess_code(code)

                # 验证
                is_valid, error = self.validate_code(code)

                results.append(GenerationResult(
                    code=code,
                    prompt=prompt,
                    metadata={'strategy': 'cot'},
                    success=is_valid,
                    error=error if not is_valid else None
                ))

            except Exception as e:
                results.append(GenerationResult(
                    code="",
                    prompt=prompt,
                    metadata={'strategy': 'cot'},
                    success=False,
                    error=str(e)
                ))

        return results

    def _call_llm(self, prompt: str) -> str:
        """调用LLM API - 强制使用真实API，失败时抛出异常"""
        # 如果传入了 llm_client (DeepSeekClient)，使用它
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # 使用传入的 llm_client 生成
                code = self.llm.generate(prompt, architecture_hash='')
                return code
            except Exception as e:
                print(f"LLM client failed: {e}")
                raise RuntimeError(f"DeepSeek API call failed: {e}") from e

        # 备用：直接调用 OpenAI API
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
                top_p=self.top_p,
            )
            return response.choices[0].message.content
        except Exception as e:
            # API调用失败时抛出异常，不使用mock
            print(f"LLM API call failed: {e}")
            raise RuntimeError(f"DeepSeek API call failed: {e}") from e

    def _mock_generate(self, architecture_desc: Dict[str, Any]) -> str:
        """模拟代码生成 (用于测试)"""
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

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim*4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features, language_features):
        # Project to common space
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)

        # Concatenate and fuse
        fused = v + l  # Simple addition for now

        # Apply transformer layers
        for layer in self.layers:
            fused = layer(fused.unsqueeze(1)).squeeze(1)

        return self.norm(fused)
'''
