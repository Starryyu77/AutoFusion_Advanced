"""
Error Feedback Mechanism
========================

错误反馈循环，当代码编译失败时，将错误信息反馈给 LLM 进行修正。
"""

from typing import Tuple, Optional, List, Dict
import re
import ast


class CodeValidator:
    """代码验证器"""
    
    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        验证 Python 语法
        
        Args:
            code: Python 代码字符串
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"语法错误 (行 {e.lineno}): {e.msg}"
            if e.text:
                error_msg += f"\n{e.text.rstrip()}"
                error_msg += "\n" + " " * (e.offset or 0) + "^"
            return False, error_msg
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def check_class_definition(code: str) -> Tuple[bool, Optional[str]]:
        """检查是否定义了 FusionModule 类"""
        if "class FusionModule" not in code:
            return False, "未找到 FusionModule 类定义"
        
        if "nn.Module" not in code:
            return False, "FusionModule 必须继承 nn.Module"
        
        return True, None
    
    @staticmethod
    def check_forward_method(code: str) -> Tuple[bool, Optional[str]]:
        """检查是否有 forward 方法"""
        if "def forward" not in code:
            return False, "未找到 forward 方法定义"
        
        # 检查参数
        if "vision_features" not in code or "language_features" not in code:
            return False, "forward 方法需要 vision_features 和 language_features 参数"
        
        # 检查返回值
        if "return" not in code:
            return False, "forward 方法必须有 return 语句"
        
        return True, None
    
    @staticmethod
    def check_imports(code: str) -> Tuple[bool, Optional[str]]:
        """检查必要的导入"""
        required_imports = ["import torch", "import torch.nn"]
        
        for imp in required_imports:
            if imp not in code:
                return False, f"缺少必要的导入: {imp}"
        
        return True, None
    
    def validate_all(self, code: str) -> Tuple[bool, List[str]]:
        """
        执行所有验证
        
        Args:
            code: Python 代码
            
        Returns:
            (是否全部通过, 错误列表)
        """
        errors = []
        
        # 语法检查
        valid, error = self.validate_syntax(code)
        if not valid:
            errors.append(error)
            return False, errors  # 语法错误直接返回
        
        # 类定义检查
        valid, error = self.check_class_definition(code)
        if not valid:
            errors.append(error)
        
        # forward 方法检查
        valid, error = self.check_forward_method(code)
        if not valid:
            errors.append(error)
        
        # 导入检查
        valid, error = self.check_imports(code)
        if not valid:
            errors.append(error)
        
        return len(errors) == 0, errors


class ErrorFeedbackLoop:
    """错误反馈循环"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.validator = CodeValidator()
        self.error_history: List[Dict] = []
    
    def generate_with_feedback(
        self,
        llm_generate_fn,
        initial_prompt: str,
        template_mode: bool = True,
        generate_code_fn=None
    ) -> Tuple[Optional[str], int, List[Dict]]:
        """
        带反馈的代码生成
        
        Args:
            llm_generate_fn: LLM 生成函数
            initial_prompt: 初始 Prompt
            template_mode: 是否使用模板模式
            generate_code_fn: 从模板生成代码的函数
            
        Returns:
            (最终代码, 尝试次数, 错误历史)
        """
        prompt = initial_prompt
        self.error_history = []
        
        for attempt in range(self.max_retries):
            # 1. LLM 生成响应
            response = llm_generate_fn(prompt)
            
            # 2. 解析响应
            if template_mode and generate_code_fn:
                # 模板模式：解析 JSON，生成代码
                from .prompt_builder_v2 import parse_llm_response
                parsed = parse_llm_response(response, template_mode=True)
                
                if not parsed.get("success"):
                    error = f"JSON 解析失败: {parsed.get('error')}"
                    self._record_error(attempt, error, response)
                    prompt = self._build_retry_prompt(initial_prompt, error, attempt)
                    continue
                
                # 从模板生成代码
                try:
                    code = generate_code_fn(
                        parsed["template"],
                        parsed["params"]
                    )
                except Exception as e:
                    error = f"代码生成失败: {str(e)}"
                    self._record_error(attempt, error, response)
                    prompt = self._build_retry_prompt(initial_prompt, error, attempt)
                    continue
            else:
                # 自由模式：直接使用响应作为代码
                code = self._extract_code(response)
            
            # 3. 验证代码
            is_valid, errors = self.validator.validate_all(code)
            
            if is_valid:
                return code, attempt + 1, self.error_history
            
            # 4. 记录错误
            error_msg = "\n".join(errors)
            self._record_error(attempt, error_msg, response[:200])
            
            # 5. 构建重试 Prompt
            prompt = self._build_retry_prompt(initial_prompt, error_msg, attempt)
        
        # 所有尝试都失败
        return None, self.max_retries, self.error_history
    
    def _record_error(self, attempt: int, error: str, response_snippet: str):
        """记录错误"""
        self.error_history.append({
            "attempt": attempt + 1,
            "error": error,
            "response_snippet": response_snippet[:200] if len(response_snippet) > 200 else response_snippet
        })
    
    def _build_retry_prompt(self, initial_prompt: str, error: str, attempt: int) -> str:
        """构建重试 Prompt"""
        return f"""{initial_prompt}

---

## ⚠️ 第 {attempt + 1} 次尝试失败

**错误信息:**
```
{error}
```

**请修正上述错误并重新生成。**

注意：
1. 仔细检查语法错误
2. 确保所有括号匹配
3. 确保所有变量在使用前定义
4. 确保 forward 函数正确返回输出

这是第 {attempt + 1}/{self.max_retries} 次尝试。"""
    
    def _extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        # 尝试提取 ```python ``` 代码块
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.strip()
        
        return code
    
    def get_error_summary(self) -> str:
        """获取错误摘要"""
        if not self.error_history:
            return "无错误记录"
        
        lines = ["错误历史摘要:", ""]
        for entry in self.error_history:
            lines.append(f"尝试 {entry['attempt']}: {entry['error'][:100]}")
        
        return "\n".join(lines)


class ErrorAnalyzer:
    """错误分析器"""
    
    # 常见错误模式
    ERROR_PATTERNS = {
        r"unexpected EOF": "代码不完整，可能缺少括号或引号",
        r"invalid syntax": "语法错误",
        r"name '(\w+)' is not defined": "变量未定义",
        r"indentation error": "缩进错误",
        r"missing required argument": "缺少必要参数",
        r"too many values to unpack": "解包值数量不匹配",
        r"Expected \\)": "可能缺少右括号",
        r"Expected \\]": "可能缺少右方括号",
        r"Expected \\}": "可能缺少右花括号",
    }
    
    @classmethod
    def analyze_error(cls, error_msg: str) -> Dict:
        """
        分析错误信息
        
        Args:
            error_msg: 错误信息
            
        Returns:
            分析结果
        """
        result = {
            "original_error": error_msg,
            "error_type": "unknown",
            "suggestion": None
        }
        
        for pattern, description in cls.ERROR_PATTERNS.items():
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                result["error_type"] = description
                if match.groups():
                    result["suggestion"] = f"请检查 {match.group(1)} 的定义"
                else:
                    result["suggestion"] = description
                break
        
        return result
    
    @classmethod
    def get_common_errors_stats(cls, error_history: List[Dict]) -> Dict[str, int]:
        """统计常见错误"""
        stats = {}
        
        for entry in error_history:
            error = entry.get("error", "")
            for pattern, description in cls.ERROR_PATTERNS.items():
                if re.search(pattern, error, re.IGNORECASE):
                    stats[description] = stats.get(description, 0) + 1
                    break
            else:
                stats["其他错误"] = stats.get("其他错误", 0) + 1
        
        return stats


if __name__ == "__main__":
    # 测试代码验证器
    validator = CodeValidator()
    
    # 测试有效代码
    valid_code = '''
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim=768, language_dim=768):
        super().__init__()
        self.proj = nn.Linear(vision_dim + language_dim, 128)
    
    def forward(self, vision_features, language_features):
        x = torch.cat([vision_features, language_features], dim=-1)
        return self.proj(x)
'''
    
    print("=== 验证有效代码 ===")
    is_valid, errors = validator.validate_all(valid_code)
    print(f"有效: {is_valid}")
    if errors:
        print(f"错误: {errors}")
    
    # 测试无效代码
    invalid_code = '''
class FusionModule:
    def forward(self, x):
        return x
'''
    
    print("\n=== 验证无效代码 ===")
    is_valid, errors = validator.validate_all(invalid_code)
    print(f"有效: {is_valid}")
    print(f"错误: {errors}")
    
    # 测试错误分析器
    print("\n=== 错误分析 ===")
    error_msg = "name 'hidden_dim' is not defined"
    analysis = ErrorAnalyzer.analyze_error(error_msg)
    print(f"分析结果: {analysis}")