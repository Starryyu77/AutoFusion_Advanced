#!/usr/bin/env python3
"""
简化的实验启动脚本 - Phase 5.5
"""

import os
import sys

# 添加路径
sys.path.insert(0, "/usr1/home/s125mdg43_10/AutoFusion_Advanced")
sys.path.insert(0, "/usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl")

from src.v2.architecture_templates import (
    generate_code,
    get_default_params,
    ARCHITECTURE_TEMPLATES,
)
from src.v2.error_feedback import CodeValidator

print("=" * 60)
print("Phase 5.5 测试 - 架构模板代码生成")
print("=" * 60)

# 测试所有模板
validator = CodeValidator()

for name in ARCHITECTURE_TEMPLATES:
    params = get_default_params(name)
    code = generate_code(name, params)
    is_valid, errors = validator.validate_all(code)

    status = "✅" if is_valid else "❌"
    print(f"\n{status} {name}")
    print(f"   参数: {params}")
    print(f"   代码长度: {len(code)} 字符")
    if errors:
        print(f"   错误: {errors}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
