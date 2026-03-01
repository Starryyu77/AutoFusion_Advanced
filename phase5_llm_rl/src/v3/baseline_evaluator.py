"""
Baseline Evaluator - Phase 5.6
==============================

与人工设计架构 (FiLM 等) 对比
"""

import sys
import torch
import torch.nn as nn
from typing import Dict, List, Any

sys.path.insert(0, "/usr1/home/s125mdg43_10/AutoFusion_Advanced")


class BaselineArchitecture(nn.Module):
    """简单的 baseline 架构基类"""

    def __init__(self):
        super().__init__()


class SimpleMLP(BaselineArchitecture):
    """简单 MLP 融合"""

    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(vision_dim + language_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vision_features, language_features):
        x = torch.cat([vision_features, language_features], dim=-1)
        x = self.input_proj(x)
        x = torch.relu(x)
        return self.output_proj(x)


class ConcatLinear(BaselineArchitecture):
    """简单拼接 + 线性"""

    def __init__(self, vision_dim=768, language_dim=768, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(vision_dim + language_dim, hidden_dim)

    def forward(self, vision_features, language_features):
        x = torch.cat([vision_features, language_features], dim=-1)
        return self.proj(x)


class BaselineEvaluator:
    """Baseline 评估器"""

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.results = {}

    def evaluate_baseline(self, name: str, model_class, **kwargs) -> Dict[str, Any]:
        """评估单个 baseline"""
        logger = __import__("logging").getLogger(__name__)
        logger.info(f"\n评估 Baseline: {name}")

        # 创建模型
        model = model_class(**kwargs)

        # 评估
        try:
            result = self.evaluator.evaluate(model)
            self.results[name] = result
            logger.info(f"✓ {name}: Reward={result.get('reward', 0):.3f}")
            return result
        except Exception as e:
            logger.error(f"✗ {name} 评估失败: {e}")
            return None

    def evaluate_all_baselines(self) -> Dict[str, Dict]:
        """评估所有 baselines"""
        baselines = [
            ("simple_mlp", SimpleMLP, {"hidden_dim": 64}),
            ("concat_linear", ConcatLinear, {"hidden_dim": 64}),
        ]

        for name, model_class, kwargs in baselines:
            self.evaluate_baseline(name, model_class, **kwargs)

        return self.results

    def compare_with_discovered(self, discovered_results: Dict) -> str:
        """生成对比报告"""
        lines = [
            "# Baseline 对比报告",
            "",
            "## 性能对比",
            "",
            "| 架构 | 类型 | Reward | FLOPs |",
            "|------|------|--------|-------|",
        ]

        # Baselines
        for name, result in sorted(self.results.items()):
            reward = result.get("reward", 0)
            flops = result.get("flops", 0) / 1e6
            lines.append(f"| {name} | Baseline | {reward:.3f} | {flops:.1f}M |")

        # Discovered
        for name, result in sorted(discovered_results.items()):
            reward = result.get("reward", 0)
            flops = result.get("flops", 0) / 1e6
            lines.append(f"| {name} | Discovered | {reward:.3f} | {flops:.1f}M |")

        lines.append("")
        lines.append("## 结论")
        lines.append("")

        if self.results and discovered_results:
            best_baseline = max(
                self.results.items(), key=lambda x: x[1].get("reward", 0)
            )
            best_discovered = max(
                discovered_results.items(), key=lambda x: x[1].get("reward", 0)
            )

            lines.append(
                f"- 最佳 Baseline: {best_baseline[0]} (Reward: {best_baseline[1].get('reward', 0):.3f})"
            )
            lines.append(
                f"- 最佳 Discovered: {best_discovered[0]} (Reward: {best_discovered[1].get('reward', 0):.3f})"
            )

            if best_discovered[1].get("reward", 0) > best_baseline[1].get("reward", 0):
                lines.append(f"- ✅ Discovered 架构优于 Baseline!")
            else:
                lines.append(f"- ⚠️ Baseline 仍然最优")

        return "\n".join(lines)


if __name__ == "__main__":
    print("BaselineEvaluator module loaded")
    print("Available baselines:")
    print("  - SimpleMLP")
    print("  - ConcatLinear")
