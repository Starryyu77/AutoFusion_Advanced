"""
Phase 5.5 改进模块
==================

包含:
- architecture_templates: 架构模板库
- prompt_builder_v2: 改进的 Prompt 构建器
- error_feedback: 错误反馈机制
- main_loop_v2: 改进的主循环
"""

from .architecture_templates import (
    ARCHITECTURE_TEMPLATES,
    ArchitectureTemplate,
    generate_code,
    validate_params,
    get_default_params,
    get_template,
    get_all_templates,
)

from .prompt_builder_v2 import PromptBuilderV2, Constraints, parse_llm_response

from .error_feedback import CodeValidator, ErrorFeedbackLoop, ErrorAnalyzer

__all__ = [
    # 架构模板
    "ARCHITECTURE_TEMPLATES",
    "ArchitectureTemplate",
    "generate_code",
    "validate_params",
    "get_default_params",
    "get_template",
    "get_all_templates",
    # Prompt 构建
    "PromptBuilderV2",
    "Constraints",
    "parse_llm_response",
    # 错误反馈
    "CodeValidator",
    "ErrorFeedbackLoop",
    "ErrorAnalyzer",
]
