"""
Generator Abstract Base Class
-----------------------------
All code generation strategies must inherit from BaseGenerator.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """代码生成结果"""
    code: str
    prompt: str
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'prompt': self.prompt,
            'metadata': self.metadata,
            'success': self.success,
            'error': self.error,
        }


class BaseGenerator(ABC):
    """Generator 抽象基类 - 所有代码生成策略必须继承"""

    def __init__(self, llm_client: Any, config: Dict[str, Any]):
        self.llm = llm_client
        self.config = config
        self.model = config.get('model', 'deepseek-chat')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)
        self.top_p = config.get('top_p', 0.95)

    @abstractmethod
    def build_prompt(self, architecture_desc: Dict[str, Any]) -> str:
        """
        构建 Prompt

        Args:
            architecture_desc: 架构描述字典

        Returns:
            构建好的prompt字符串
        """
        pass

    @abstractmethod
    def generate(self, architecture_desc: Dict[str, Any], num_samples: int = 1) -> List[GenerationResult]:
        """
        生成架构代码

        Args:
            architecture_desc: 架构描述
            num_samples: 生成样本数

        Returns:
            GenerationResult列表
        """
        pass

    def postprocess_code(self, code: str) -> str:
        """
        后处理生成的代码

        Args:
            code: 原始代码

        Returns:
            处理后的代码
        """
        # 移除markdown代码块标记
        code = code.strip()
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        return code.strip()

    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        验证代码语法

        Args:
            code: Python代码

        Returns:
            (是否合法, 错误信息)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)