"""
Shape Constraint Generator
--------------------------
Hard tensor dimension constraint guided generation.
"""

from typing import Dict, Any, List
from base import BaseGenerator, GenerationResult


class ShapeConstraintGenerator(BaseGenerator):
    """
    Shape-Constraint Prompt Strategy

    特点:
    1. 显式张量形状约束
    2. 维度检查
    3. 自动形状推断
    """

    def build_prompt(self, architecture_desc: Dict[str, Any]) -> str:
        """构建Shape-Constraint Prompt"""
        arch_type = architecture_desc.get('type', 'attention')
        fusion_type = architecture_desc.get('fusion_type', 'middle')
        hidden_dim = architecture_desc.get('hidden_dim', 512)
        num_layers = architecture_desc.get('num_layers', 4)

        return f"""You are an expert PyTorch developer specializing in tensor shape management.

## Task
Implement a multimodal fusion module with STRICT shape constraints.

## Architecture Specification
- Type: {arch_type}
- Fusion Strategy: {fusion_type}
- Hidden Dimension: {hidden_dim}
- Number of Layers: {num_layers}

## Shape Constraints (MUST FOLLOW)

Input:
- vision_features: (batch_size, vision_dim=768)
- language_features: (batch_size, language_dim=768)

Output:
- fused_features: (batch_size, {hidden_dim})

Internal Constraints:
1. After projection: (batch_size, {hidden_dim})
2. After fusion: (batch_size, {hidden_dim})
3. All intermediate tensors must maintain batch dimension

## Shape Verification Checklist

Your code MUST include shape assertions:
```python
assert vision_features.dim() == 2, f"Expected 2D vision, got {{vision_features.dim()}}D"
assert language_features.dim() == 2, f"Expected 2D language, got {{language_features.dim()}}D"
assert fused_output.shape == (vision_features.size(0), {hidden_dim}), f"Output shape mismatch"
```

## Output Format

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        # Initialize layers with explicit shapes

    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:
        # Input shape verification
        batch_size = vision_features.size(0)
        assert vision_features.shape == (batch_size, 768)
        assert language_features.shape == (batch_size, 768)

        # Your implementation with shape comments
        # e.g., # Shape: (batch, 768) -> (batch, {hidden_dim})

        # Output shape verification
        assert fused.shape == (batch_size, {hidden_dim})
        return fused
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
                    metadata={'strategy': 'shape'},
                    success=is_valid,
                    error=error if not is_valid else None
                ))
            except Exception as e:
                results.append(GenerationResult(
                    code="",
                    prompt=prompt,
                    metadata={'strategy': 'shape'},
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
    """Shape-constrained multimodal fusion module."""

    def __init__(self, vision_dim: int = 768, language_dim: int = 768,
                 hidden_dim: int = {hidden_dim}, num_layers: int = {num_layers}):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shape: (batch, 768) -> (batch, {hidden_dim})
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

        # Fusion layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:
        # Input shape verification
        batch_size = vision_features.size(0)
        assert vision_features.dim() == 2, f"Expected 2D vision, got {{vision_features.dim()}}D"
        assert language_features.dim() == 2, f"Expected 2D language, got {{language_features.dim()}}D"
        assert vision_features.shape[1] == 768, f"Expected vision dim 768, got {{vision_features.shape[1]}}"
        assert language_features.shape[1] == 768, f"Expected language dim 768, got {{language_features.shape[1]}}"

        # Projection: (batch, 768) -> (batch, {hidden_dim})
        v = self.vision_proj(vision_features)
        l = self.language_proj(language_features)
        assert v.shape == (batch_size, self.hidden_dim), f"Vision proj shape mismatch: {{v.shape}}"
        assert l.shape == (batch_size, self.hidden_dim), f"Language proj shape mismatch: {{l.shape}}"

        # Fusion: (batch, {hidden_dim}) -> (batch, {hidden_dim})
        fused = v + l  # Element-wise addition

        # Apply layers with shape tracking
        for i, layer in enumerate(self.layers):
            # Input: (batch, {hidden_dim}) -> Output: (batch, {hidden_dim})
            fused = layer(fused.unsqueeze(1)).squeeze(1)
            assert fused.shape == (batch_size, self.hidden_dim), f"Layer {{i}} shape mismatch: {{fused.shape}}"

        # Output normalization
        output = self.output_norm(fused)

        # Final shape verification
        assert output.shape == (batch_size, self.hidden_dim), \
            f"Output shape mismatch: expected ({{batch_size}}, {{self.hidden_dim}}), got {{output.shape}}"

        return output
'''