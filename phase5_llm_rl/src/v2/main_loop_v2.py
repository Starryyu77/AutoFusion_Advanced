"""
Main Loop V2 - Improved LLM-Driven NAS
======================================

改进的主循环，包含：
1. 模板模式搜索
2. 错误反馈机制
3. 动态策略切换
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加路径
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .architecture_templates import (
    generate_code,
    validate_params,
    get_default_params,
    ARCHITECTURE_TEMPLATES,
)
from .prompt_builder_v2 import PromptBuilderV2, Constraints, parse_llm_response
from .error_feedback import ErrorFeedbackLoop, CodeValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""

    iteration: int
    template: str
    params: Dict[str, Any]
    code: str
    compile_success: bool
    accuracy: float
    flops: float
    params_count: float
    reward: float
    error: Optional[str] = None
    time_taken: float = 0.0


class NASControllerV2:
    """改进的 NAS 控制器"""

    def __init__(
        self,
        llm_backend,
        evaluator,
        reward_fn,
        use_template_mode: bool = True,
        use_error_feedback: bool = True,
        max_retries: int = 3,
        output_dir: str = "./results",
    ):
        self.llm = llm_backend
        self.evaluator = evaluator
        self.reward_fn = reward_fn

        self.use_template_mode = use_template_mode
        self.use_error_feedback = use_error_feedback
        self.max_retries = max_retries
        self.output_dir = output_dir

        # 组件
        self.prompt_builder = PromptBuilderV2(
            use_template=use_template_mode, use_few_shot=True
        )
        self.error_feedback = ErrorFeedbackLoop(max_retries=max_retries)
        self.validator = CodeValidator()

        # 状态
        self.history: List[Dict] = []
        self.best_result: Optional[SearchResult] = None
        self.results: List[SearchResult] = []

        # 统计
        self.stats = {
            "total_iterations": 0,
            "compile_success": 0,
            "compile_failed": 0,
            "best_reward": 0.0,
            "best_accuracy": 0.0,
        }

    def search(
        self,
        max_iterations: int = 50,
        constraints: Optional[Constraints] = None,
        save_interval: int = 5,
    ) -> SearchResult:
        """
        执行搜索

        Args:
            max_iterations: 最大迭代次数
            constraints: 约束条件
            save_interval: 保存间隔

        Returns:
            最佳搜索结果
        """
        logger.info(
            f"开始搜索，模板模式: {self.use_template_mode}, 错误反馈: {self.use_error_feedback}"
        )
        logger.info(f"最大迭代次数: {max_iterations}")

        start_time = time.time()

        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()

            logger.info(f"\n{'=' * 60}")
            logger.info(f"迭代 {iteration}/{max_iterations}")
            logger.info(f"{'=' * 60}")

            # 1. 确定策略
            strategy = self._determine_strategy(iteration, max_iterations)
            logger.info(f"策略: {strategy}")

            # 2. 构建 Prompt
            prompt = self.prompt_builder.build(
                strategy=strategy,
                constraints=constraints,
                history=self.history[-10:],  # 最近 10 次
                best_architecture=self._get_best_info(),
                iteration=iteration,
                template_mode=self.use_template_mode,
            )

            # 3. 生成架构
            if self.use_template_mode:
                result = self._generate_with_template(prompt, iteration)
            else:
                result = self._generate_free(prompt, iteration)

            # 4. 记录结果
            result.time_taken = time.time() - iter_start
            self.results.append(result)
            self._update_stats(result)

            # 5. 更新最佳结果
            if result.compile_success and (
                self.best_result is None or result.reward > self.best_result.reward
            ):
                self.best_result = result
                logger.info(f"🏆 新最佳! Reward: {result.reward:.3f}")

            # 6. 记录历史
            self.history.append(
                {
                    "type": result.template,
                    "compile_success": result.compile_success,
                    "accuracy": result.accuracy,
                    "flops": result.flops,
                    "reward": result.reward,
                    "error": result.error[:50] if result.error else None,
                }
            )

            # 7. 打印状态
            self._print_iteration_summary(result)

            # 8. 定期保存
            if iteration % save_interval == 0:
                self._save_checkpoint(iteration)

        total_time = time.time() - start_time
        logger.info(f"\n{'=' * 60}")
        logger.info("搜索完成!")
        logger.info(f"总耗时: {total_time / 60:.1f} 分钟")
        logger.info(
            f"编译成功率: {self.stats['compile_success']}/{self.stats['total_iterations']} "
            f"({self.stats['compile_success'] / self.stats['total_iterations'] * 100:.1f}%)"
        )
        if self.best_result:
            logger.info(f"最佳 Reward: {self.best_result.reward:.3f}")
            logger.info(
                f"最佳架构: {self.best_result.template}, {self.best_result.params}"
            )

        return self.best_result

    def _determine_strategy(self, iteration: int, max_iterations: int) -> str:
        """确定搜索策略"""
        progress = iteration / max_iterations

        if progress < 0.3:
            return "explore"
        elif progress < 0.7:
            return "exploit"
        else:
            return "refine"

    def _generate_with_template(self, prompt: str, iteration: int) -> SearchResult:
        """使用模板模式生成架构"""

        # 1. LLM 生成响应
        response = self.llm.generate(prompt)

        # 2. 解析响应
        parsed = parse_llm_response(response, template_mode=True)

        if not parsed.get("success"):
            return SearchResult(
                iteration=iteration,
                template="unknown",
                params={},
                code="",
                compile_success=False,
                accuracy=0.0,
                flops=0.0,
                params_count=0.0,
                reward=0.0,
                error=parsed.get("error", "JSON 解析失败"),
            )

        template = parsed["template"]
        params = parsed["params"]

        # 3. 验证参数
        if not validate_params(template, params):
            # 使用默认参数
            params = get_default_params(template)

        # 4. 生成代码
        try:
            code = generate_code(template, params)
        except Exception as e:
            return SearchResult(
                iteration=iteration,
                template=template,
                params=params,
                code="",
                compile_success=False,
                accuracy=0.0,
                flops=0.0,
                params_count=0.0,
                reward=0.0,
                error=f"代码生成失败: {str(e)}",
            )

        # 5. 验证代码
        is_valid, errors = self.validator.validate_all(code)
        if not is_valid:
            return SearchResult(
                iteration=iteration,
                template=template,
                params=params,
                code=code,
                compile_success=False,
                accuracy=0.0,
                flops=0.0,
                params_count=0.0,
                reward=0.0,
                error="\n".join(errors),
            )

        # 6. 评估架构
        return self._evaluate_architecture(iteration, template, params, code)

    def _generate_free(self, prompt: str, iteration: int) -> SearchResult:
        """自由模式生成架构"""

        # 带错误反馈的生成
        code, attempts, error_history = self.error_feedback.generate_with_feedback(
            llm_generate_fn=lambda p: self.llm.generate(p),
            initial_prompt=prompt,
            template_mode=False,
        )

        if code is None:
            return SearchResult(
                iteration=iteration,
                template="unknown",
                params={},
                code="",
                compile_success=False,
                accuracy=0.0,
                flops=0.0,
                params_count=0.0,
                reward=0.0,
                error=f"所有 {attempts} 次尝试均失败",
            )

        # 评估架构
        return self._evaluate_architecture(iteration, "free", {}, code)

    def _evaluate_architecture(
        self, iteration: int, template: str, params: Dict, code: str
    ) -> SearchResult:
        """评估架构"""

        try:
            # 评估
            eval_result = self.evaluator.evaluate(code)

            # 计算奖励
            reward = self.reward_fn.calculate(eval_result)

            return SearchResult(
                iteration=iteration,
                template=template,
                params=params,
                code=code,
                compile_success=True,
                accuracy=eval_result.accuracy,
                flops=eval_result.flops,
                params_count=eval_result.params,
                reward=reward,
            )

        except Exception as e:
            logger.error(f"评估失败: {str(e)}")
            return SearchResult(
                iteration=iteration,
                template=template,
                params=params,
                code=code,
                compile_success=False,
                accuracy=0.0,
                flops=0.0,
                params_count=0.0,
                reward=0.0,
                error=f"评估失败: {str(e)}",
            )

    def _get_best_info(self) -> Optional[Dict]:
        """获取最佳架构信息"""
        if self.best_result is None:
            return None

        return {
            "type": self.best_result.template,
            "params": self.best_result.params,
            "reward": self.best_result.reward,
            "accuracy": self.best_result.accuracy,
        }

    def _update_stats(self, result: SearchResult):
        """更新统计"""
        self.stats["total_iterations"] += 1

        if result.compile_success:
            self.stats["compile_success"] += 1
            if result.reward > self.stats["best_reward"]:
                self.stats["best_reward"] = result.reward
            if result.accuracy > self.stats["best_accuracy"]:
                self.stats["best_accuracy"] = result.accuracy
        else:
            self.stats["compile_failed"] += 1

    def _print_iteration_summary(self, result: SearchResult):
        """打印迭代摘要"""
        status = "✅ 成功" if result.compile_success else "❌ 失败"

        logger.info(f"架构类型: {result.template}")
        logger.info(f"编译状态: {status}")

        if result.compile_success:
            logger.info(f"准确率: {result.accuracy * 100:.1f}%")
            logger.info(f"FLOPs: {result.flops / 1e6:.1f}M")
            logger.info(f"Reward: {result.reward:.3f}")
        else:
            logger.info(f"错误: {result.error[:100] if result.error else 'unknown'}")

        if self.best_result:
            logger.info(f"🏆 当前最佳 Reward: {self.best_result.reward:.3f}")

        logger.info(f"耗时: {result.time_taken:.1f}s")

    def _save_checkpoint(self, iteration: int):
        """保存检查点"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存结果
        results_file = os.path.join(self.output_dir, f"results_iter_{iteration}.json")
        with open(results_file, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)

        # 保存最佳架构代码
        if self.best_result and self.best_result.code:
            best_file = os.path.join(self.output_dir, "best_architecture.py")
            with open(best_file, "w") as f:
                f.write(self.best_result.code)

        logger.info(f"检查点已保存: {results_file}")


if __name__ == "__main__":
    # 测试代码
    print("NASControllerV2 模块已加载")
    print(f"可用模板: {list(ARCHITECTURE_TEMPLATES.keys())}")

    # 测试模板代码生成
    for name in ["attention", "gated", "mlp"]:
        params = get_default_params(name)
        code = generate_code(name, params)

        validator = CodeValidator()
        is_valid, errors = validator.validate_all(code)

        print(f"\n{name}: {'✅ 有效' if is_valid else '❌ 无效'}")
        if errors:
            print(f"  错误: {errors}")
