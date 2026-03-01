#!/usr/bin/env python3
"""
Phase 5.5 实验运行脚本
======================

运行改进的 NAS 搜索实验
"""

import os
import sys
import argparse
import yaml
import logging

# 添加路径
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "phase4_optimization",
    ),
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 5.5 Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    exp_config = config["experiment"]
    llm_config = config["llm"]
    improvements = config.get("improvements", {})
    constraints_config = config.get("constraints", {}).get("constraints", {})

    # 设置输出目录
    output_dir = args.output_dir or exp_config.get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 5.5: Improved LLM-Driven NAS")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_config['name']}")
    logger.info(f"Model: {llm_config['model']}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Template Mode: {improvements.get('use_template_mode', True)}")
    logger.info(f"Error Feedback: {improvements.get('use_error_feedback', True)}")

    # 初始化组件
    logger.info("Initializing components...")

    # 1. LLM Backend
    from src.llm_backend import DeepSeekBackend, AliyunBackend
    
    api_key = os.environ.get("ALIYUN_API_KEY", llm_config.get("api_key", ""))
    
    # 根据类型选择 Backend
    backend_type = llm_config.get("type", "aliyun")
    if backend_type == "deepseek":
        llm = DeepSeekBackend(
            api_key=api_key,
            model=llm_config.get("model", "deepseek-chat"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2048),
        )
    else:  # aliyun
        llm = AliyunBackend(
            api_key=api_key,
            model=llm_config.get("model", "deepseek-v3"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2048),
        )

    # 2. Evaluator (使用 Phase 4 的评估器)
    from src.evaluator_v2_improved import ImprovedRealDataFewShotEvaluator

    eval_config = config.get("evaluator", {})
    evaluator = ImprovedRealDataFewShotEvaluator(
        dataset=eval_config.get("dataset", "mmmu"),
        num_shots=eval_config.get("num_shots", 64),
        train_epochs=eval_config.get("train_epochs", 10),
        batch_size=eval_config.get("batch_size", 8),
        max_time=eval_config.get("max_time", 300),
    )
    logger.info("✓ Evaluator initialized")

    # 3. Reward Function
    from src.reward_v2 import ConstrainedReward

    reward_config = config.get("reward", {})
    reward_fn = ConstrainedReward(
        weights=reward_config.get("weights", {}),
        penalty_type=reward_config.get("penalty_type", "exponential"),
    )
    logger.info("✓ Reward function initialized")

    # 4. 约束条件
    from src.v2.prompt_builder_v2 import Constraints

    constraints = Constraints(
        max_flops=constraints_config.get("max_flops"),
        max_params=constraints_config.get("max_params"),
        target_accuracy=constraints_config.get("target_accuracy"),
    )
    logger.info("✓ Constraints initialized")

    # 5. NAS Controller V2
    from src.v2.main_loop_v2 import NASControllerV2

    controller = NASControllerV2(
        llm_backend=llm,
        evaluator=evaluator,
        reward_fn=reward_fn,
        use_template_mode=improvements.get("use_template_mode", True),
        use_error_feedback=improvements.get("use_error_feedback", True),
        max_retries=improvements.get("max_retries", 3),
        output_dir=output_dir,
    )
    logger.info("✓ NAS Controller V2 initialized")

    logger.info("=" * 60)
    logger.info("Starting search...")
    logger.info("=" * 60)

    # 运行搜索
    best_result = controller.search(
        max_iterations=exp_config.get("max_iterations", 100),
        constraints=constraints,
        save_interval=exp_config.get("save_interval", 10),
    )

    logger.info("=" * 60)
    logger.info("Search completed!")
    if best_result:
        logger.info(f"Best Reward: {best_result.reward:.3f}")
        logger.info(f"Best Architecture: {best_result.template}")
        logger.info(f"Best Params: {best_result.params}")
        logger.info(f"Best Accuracy: {best_result.accuracy * 100:.1f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
