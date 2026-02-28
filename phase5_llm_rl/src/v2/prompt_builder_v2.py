"""
Prompt Builder V2 - Improved Prompt Engineering
================================================

改进的 Prompt 构建，包含：
1. 代码模板约束
2. Few-Shot 示例增强
3. 结构化输出要求
4. 错误反馈支持
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class Constraints:
    """工程约束"""

    max_flops: Optional[float] = None
    max_params: Optional[float] = None
    max_latency_ms: Optional[float] = None
    target_accuracy: Optional[float] = None
    min_accuracy: Optional[float] = None

    def to_prompt_text(self) -> str:
        lines = ["### 约束条件 (必须满足):", ""]
        if self.max_flops:
            lines.append(f"- 最大 FLOPs: {self.max_flops / 1e6:.1f}M")
        if self.max_params:
            lines.append(f"- 最大参数量: {self.max_params / 1e6:.1f}M")
        if self.max_latency_ms:
            lines.append(f"- 最大延迟: {self.max_latency_ms:.1f}ms")
        if self.min_accuracy:
            lines.append(f"- 最低准确率: {self.min_accuracy * 100:.1f}%")
        if self.target_accuracy:
            lines.append(f"- 目标准确率: {self.target_accuracy * 100:.1f}%")
        return "\n".join(lines)


class PromptBuilderV2:
    """改进的 Prompt 构建器"""

    def __init__(self, use_template: bool = True, use_few_shot: bool = True):
        self.use_template = use_template
        self.use_few_shot = use_few_shot

        # Few-Shot 示例
        self.few_shot_examples = [
            {
                "name": "Attention Fusion",
                "description": "轻量级注意力融合，25% 准确率，4.8M FLOPs",
                "code_type": "attention",
                "params": {"hidden_dim": 48, "num_heads": 1, "dropout": 0.0},
                "performance": {"accuracy": 0.25, "flops": "4.8M", "params": "3.0M"},
            },
            {
                "name": "Gated Fusion",
                "description": "门控融合，最佳 reward 2.797",
                "code_type": "gated",
                "params": {"hidden_dim": 128, "gate_type": "sigmoid"},
                "performance": {"accuracy": 0.25, "flops": "5.2M", "params": "3.2M"},
            },
            {
                "name": "Hybrid Fusion",
                "description": "注意力+门控混合，最优效率",
                "code_type": "hybrid",
                "params": {"hidden_dim": 64, "num_heads": 2},
                "performance": {"accuracy": 0.28, "flops": "5.0M", "params": "3.1M"},
            },
        ]

    def build(
        self,
        strategy: str = "explore",
        constraints: Optional[Constraints] = None,
        history: Optional[List[Dict]] = None,
        best_architecture: Optional[Dict] = None,
        iteration: int = 0,
        previous_error: Optional[str] = None,
        template_mode: bool = True,
    ) -> str:
        """
        构建完整的 Prompt

        Args:
            strategy: 搜索策略 (explore/exploit/refine)
            constraints: 工程约束
            history: 历史反馈
            best_architecture: 当前最佳架构
            iteration: 当前迭代次数
            previous_error: 上次编译错误信息
            template_mode: 是否使用模板模式

        Returns:
            完整的 Prompt 字符串
        """
        sections = []

        # 1. 系统角色
        sections.append(self._build_system_role())

        # 2. 任务描述
        sections.append(self._build_task_description(strategy))

        # 3. 约束条件
        if constraints:
            sections.append(constraints.to_prompt_text())

        # 4. 模板说明 (如果启用模板模式)
        if template_mode and self.use_template:
            sections.append(self._build_template_instructions())

        # 5. Few-Shot 示例
        if self.use_few_shot:
            sections.append(self._build_few_shot_section())

        # 6. 当前状态
        sections.append(self._build_current_state(iteration, best_architecture))

        # 7. 历史反馈
        if history:
            sections.append(self._build_history_section(history))

        # 8. 错误反馈 (如果有)
        if previous_error:
            sections.append(self._build_error_feedback(previous_error))

        # 9. 输出格式
        sections.append(self._build_output_format(template_mode))

        return "\n\n".join(sections)

    def _build_system_role(self) -> str:
        return """## 角色定义

你是一位专业的 PyTorch 架构设计师，专注于多模态融合网络。

你的任务是设计高效的多模态融合架构，融合视觉特征 (768维) 和语言特征 (768维)。

### 设计原则
1. **效率优先**: FLOPs 控制在 10M 以内
2. **结构清晰**: 代码必须有完整的 forward 函数
3. **类型正确**: 所有张量维度必须匹配
4. **可编译**: 代码必须能直接运行"""

    def _build_task_description(self, strategy: str) -> str:
        strategy_hints = {
            "explore": """
### 策略: 探索 (Explore)
- 尝试新的架构模式
- 不要局限于已有设计
- 可以适当冒险，探索未知领域
""",
            "exploit": """
### 策略: 利用 (Exploit)
- 基于成功案例改进
- 微调参数以提升性能
- 在已有方向上深入优化
""",
            "refine": """
### 策略: 精炼 (Refine)
- 微调最佳架构
- 专注于效率优化
- 小幅度调整参数
""",
        }
        return f"""## 任务描述

设计一个多模态融合模块，将视觉特征和语言特征融合为统一表示。
{strategy_hints.get(strategy, "")}"""

    def _build_template_instructions(self) -> str:
        return """## 可用架构模板

你可以选择以下预定义模板之一：

### 1. Attention (注意力融合)
- 参数: hidden_dim, num_heads, dropout
- 特点: 使用交叉注意力机制
- 适用: 需要精细特征交互

### 2. Gated (门控融合)
- 参数: hidden_dim, gate_type (sigmoid/tanh/softmax)
- 特点: 自适应权重融合
- 适用: 需要动态调整融合比例

### 3. Transformer (Transformer融合)
- 参数: hidden_dim, num_layers, num_heads
- 特点: 强大的序列建模能力
- 适用: 复杂特征关系

### 4. MLP (简单MLP融合)
- 参数: hidden_dim, num_layers
- 特点: 简单高效
- 适用: 快速原型

### 5. Hybrid (混合融合)
- 参数: hidden_dim, num_heads
- 特点: 注意力 + 门控的组合
- 适用: 平衡性能和效率"""

    def _build_few_shot_section(self) -> str:
        lines = ["## 参考示例", ""]

        for i, example in enumerate(self.few_shot_examples[:3], 1):
            lines.append(f"### 示例 {i}: {example['name']}")
            lines.append(f"描述: {example['description']}")
            lines.append(f"类型: {example['code_type']}")
            lines.append(f"参数: {json.dumps(example['params'])}")
            lines.append(
                f"性能: 准确率 {example['performance']['accuracy'] * 100:.0f}%, "
                f"FLOPs {example['performance']['flops']}"
            )
            lines.append("")

        return "\n".join(lines)

    def _build_current_state(self, iteration: int, best: Optional[Dict]) -> str:
        lines = [f"## 当前状态", "", f"- 迭代次数: {iteration}"]

        if best:
            lines.append(f"- 最佳 Reward: {best.get('reward', 'N/A')}")
            lines.append(f"- 最佳架构类型: {best.get('type', 'N/A')}")
            lines.append(f"- 最佳准确率: {best.get('accuracy', 'N/A')}")
        else:
            lines.append("- 暂无成功案例")

        return "\n".join(lines)

    def _build_history_section(self, history: List[Dict]) -> str:
        lines = ["## 最近尝试", ""]

        # 只显示最近 5 次
        recent = history[-5:] if len(history) > 5 else history

        for i, h in enumerate(recent, 1):
            status = "✅ 成功" if h.get("compile_success") else "❌ 失败"
            lines.append(f"{i}. {h.get('type', 'unknown')} - {status}")
            if h.get("compile_success"):
                lines.append(
                    f"   准确率: {h.get('accuracy', 0) * 100:.1f}%, "
                    f"FLOPs: {h.get('flops', 0) / 1e6:.1f}M"
                )
            else:
                lines.append(f"   错误: {h.get('error', 'unknown')[:50]}...")

        return "\n".join(lines)

    def _build_error_feedback(self, error: str) -> str:
        return f"""## ⚠️ 上次编译失败

错误信息:
```
{error}
```

请修正上述错误，确保：
1. 所有括号正确匹配
2. 所有变量在使用前定义
3. forward 函数返回正确的输出
4. 张量维度正确"""

    def _build_output_format(self, template_mode: bool) -> str:
        if template_mode:
            return """## 输出格式

请按以下 JSON 格式输出：

```json
{
    "template": "模板名称 (attention/gated/transformer/mlp/hybrid)",
    "params": {
        "hidden_dim": 整数值,
        "其他参数": 值
    },
    "reasoning": "选择理由 (1-2句话)"
}
```

注意：
- 只输出 JSON，不要有其他内容
- 参数值必须在允许范围内
- reasoning 简洁明了"""
        else:
            return """## 输出格式

请直接输出完整的 Python 代码，包含：

```python
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768):
        super().__init__()
        # 你的实现
        
    def forward(self, vision_features, language_features):
        # 你的实现
        return output  # 必须返回融合后的特征
```

注意：
- 代码必须完整可运行
- forward 函数必须返回输出
- 所有变量必须正确初始化"""


def parse_llm_response(response: str, template_mode: bool = True) -> Dict:
    """
    解析 LLM 的响应

    Args:
        response: LLM 返回的文本
        template_mode: 是否为模板模式

    Returns:
        解析后的字典
    """
    if template_mode:
        # 尝试提取 JSON
        try:
            # 查找 JSON 块
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            result = json.loads(json_str)
            return {
                "success": True,
                "template": result.get("template"),
                "params": result.get("params", {}),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "raw_response": response}
    else:
        # 直接返回代码
        return {"success": True, "code": response, "raw_response": response}


if __name__ == "__main__":
    # 测试 Prompt Builder
    builder = PromptBuilderV2()

    # 测试模板模式
    prompt = builder.build(
        strategy="explore",
        constraints=Constraints(max_flops=10e6, target_accuracy=0.5),
        iteration=1,
        template_mode=True,
    )

    print("=" * 60)
    print("Template Mode Prompt:")
    print("=" * 60)
    print(prompt)

    print("\n" + "=" * 60)
    print("Free Code Mode Prompt:")
    print("=" * 60)

    prompt_free = builder.build(
        strategy="exploit",
        constraints=Constraints(max_flops=10e6),
        iteration=10,
        best_architecture={"type": "attention", "reward": 2.5, "accuracy": 0.25},
        template_mode=False,
    )
    print(prompt_free)
