"""
测试改进代码
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture_templates import (
    generate_code,
    validate_params,
    get_default_params,
    get_all_templates,
    ARCHITECTURE_TEMPLATES,
)
from prompt_builder_v2 import PromptBuilderV2, Constraints
from error_feedback import CodeValidator, ErrorAnalyzer


def test_architecture_templates():
    """测试架构模板"""
    print("=" * 60)
    print("测试 1: 架构模板代码生成")
    print("=" * 60)

    validator = CodeValidator()
    results = []

    for name, template in get_all_templates().items():
        params = get_default_params(name)
        code = generate_code(name, params)

        is_valid, errors = validator.validate_all(code)
        results.append((name, is_valid, errors))

        status = "✅ 通过" if is_valid else "❌ 失败"
        print(f"\n模板: {name}")
        print(f"状态: {status}")
        print(f"参数: {params}")

        if errors:
            print(f"错误: {errors}")

        # 检查代码长度
        print(f"代码长度: {len(code)} 字符")

    # 统计
    passed = sum(1 for _, valid, _ in results if valid)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"测试结果: {passed}/{total} 通过")
    print(f"{'=' * 60}")

    return passed == total


def test_prompt_builder():
    """测试 Prompt 构建器"""
    print("\n" + "=" * 60)
    print("测试 2: Prompt 构建器")
    print("=" * 60)

    builder = PromptBuilderV2()
    constraints = Constraints(max_flops=10e6, target_accuracy=0.5)

    # 测试模板模式
    print("\n--- 模板模式 Prompt ---")
    prompt = builder.build(
        strategy="explore", constraints=constraints, iteration=1, template_mode=True
    )
    print(f"Prompt 长度: {len(prompt)} 字符")
    print(f"包含模板说明: {'模板' in prompt}")
    print(f"包含约束条件: {'约束' in prompt}")

    # 测试自由模式
    print("\n--- 自由模式 Prompt ---")
    prompt_free = builder.build(
        strategy="exploit",
        constraints=constraints,
        iteration=10,
        best_architecture={"type": "attention", "reward": 2.5, "accuracy": 0.25},
        template_mode=False,
    )
    print(f"Prompt 长度: {len(prompt_free)} 字符")

    # 测试错误反馈
    print("\n--- 错误反馈 Prompt ---")
    prompt_error = builder.build(
        strategy="exploit",
        constraints=constraints,
        iteration=11,
        previous_error="SyntaxError: invalid syntax (line 10)",
        template_mode=True,
    )
    print(f"包含错误信息: {'编译失败' in prompt_error or '错误' in prompt_error}")

    return True


def test_error_feedback():
    """测试错误反馈机制"""
    print("\n" + "=" * 60)
    print("测试 3: 错误反馈机制")
    print("=" * 60)

    validator = CodeValidator()

    # 测试有效代码
    valid_code = """
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768):
        super().__init__()
        self.proj = nn.Linear(vision_dim + language_dim, 128)
    
    def forward(self, vision_features, language_features):
        x = torch.cat([vision_features, language_features], dim=-1)
        return self.proj(x)
"""

    print("\n--- 测试有效代码 ---")
    is_valid, errors = validator.validate_all(valid_code)
    print(f"验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")

    # 测试无效代码
    invalid_codes = [
        ("缺少导入", "class FusionModule:\n    pass"),
        (
            "缺少 forward",
            "import torch\nimport torch.nn as nn\nclass FusionModule(nn.Module):\n    pass",
        ),
        (
            "语法错误",
            "import torch\nimport torch.nn as nn\nclass FusionModule(nn.Module):\n    def __init__(self:\n        pass",
        ),
    ]

    print("\n--- 测试无效代码 ---")
    for name, code in invalid_codes:
        is_valid, errors = validator.validate_all(code)
        print(f"\n{name}: {'✅ 正确识别为无效' if not is_valid else '❌ 误判为有效'}")
        if errors:
            print(f"错误信息: {errors[0][:80]}")

    # 测试错误分析
    print("\n--- 测试错误分析 ---")
    error_msg = "name 'hidden_dim' is not defined"
    analysis = ErrorAnalyzer.analyze_error(error_msg)
    print(f"原始错误: {analysis['original_error']}")
    print(f"错误类型: {analysis['error_type']}")
    print(f"修复建议: {analysis['suggestion']}")

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Phase 5.5 改进代码测试")
    print("=" * 60)

    results = []

    # 运行测试
    results.append(("架构模板", test_architecture_templates()))
    results.append(("Prompt构建器", test_prompt_builder()))
    results.append(("错误反馈机制", test_error_feedback()))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in results)
    print(f"\n总结果: {'✅ 全部通过' if all_passed else '❌ 存在失败'}")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
