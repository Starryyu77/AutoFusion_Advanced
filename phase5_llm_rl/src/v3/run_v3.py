"""
Phase 5.6 运行脚本
==================

扩展搜索 + Baseline 对比
"""

import os
import sys
import yaml
import logging

sys.path.insert(0, "/usr1/home/s125mdg43_10/AutoFusion_Advanced")
sys.path.insert(0, "/usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_config = config["experiment"]
    eval_config = config.get("evaluator", {})
    llm_config = config.get("llm", {})

    output_dir = args.output_dir or exp_config.get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 5.6: Extended Search")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_config['name']}")
    logger.info(f"Iterations: {exp_config['max_iterations']}")
    logger.info(f"Shots: {eval_config.get('num_shots', 64)}")
    logger.info(f"Epochs: {eval_config.get('train_epochs', 10)}")
    logger.info(f"Output: {output_dir}")

    # 初始化组件
    from src.llm_backend import AliyunBailianBackend
    from phase4_optimization.src.evaluator_v2_improved import (
        ImprovedRealDataFewShotEvaluator,
    )
    from phase4_optimization.src.reward_v2 import ConstrainedReward
    from src.v2.main_loop_v2 import NASControllerV2

    api_key = os.environ.get("ALIYUN_API_KEY", llm_config.get("api_key", ""))

    llm = AliyunBailianBackend(
        api_key=api_key,
        model=llm_config.get("model", "kimi-k2.5"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 2048),
    )
    logger.info("✓ LLM Backend initialized")

    evaluator = ImprovedRealDataFewShotEvaluator(eval_config)
    logger.info("✓ Evaluator initialized")

    reward_fn = ConstrainedReward(config.get("reward", {}))
    logger.info("✓ Reward function initialized")

    controller = NASControllerV2(
        llm_backend=llm,
        evaluator=evaluator,
        reward_fn=reward_fn,
        use_template_mode=True,
        use_error_feedback=True,
        max_retries=3,
        output_dir=output_dir,
    )
    logger.info("✓ Controller initialized")

    logger.info("=" * 60)
    logger.info("Starting search...")
    logger.info("=" * 60)

    # 运行搜索
    best_result = controller.search(
        max_iterations=exp_config.get("max_iterations", 200),
        save_interval=exp_config.get("save_interval", 20),
    )

    logger.info("=" * 60)
    logger.info("Search completed!")
    if best_result:
        logger.info(f"Best Reward: {best_result.reward}")
        logger.info(f"Best Architecture: {best_result.template}")
        logger.info(f"Best Params: {best_result.params}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
