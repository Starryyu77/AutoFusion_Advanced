#!/usr/bin/env python3
"""
Baseline 评估脚本 - Phase 5.6
与 FiLM 等人工设计架构对比
"""

import os
import sys
import json

sys.path.insert(0, "/usr1/home/s125mdg43_10/AutoFusion_Advanced")
sys.path.insert(0, "/usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl")

from expv2.shared.baselines.film import FiLMFusion
from expv2.shared.baselines.clip_fusion import CLIPFusion
from expv2.shared.baselines.concat_mlp import ConcatMLP
from expv2.shared.baselines.bilinear_pooling import BilinearPooling
from phase4_optimization.src.evaluator_v2_improved import (
    ImprovedRealDataFewShotEvaluator,
)

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_baseline(model_class, name, eval_config):
    """评估单个 baseline"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"评估 Baseline: {name}")
    logger.info(f"{'=' * 60}")

    try:
        # 创建模型
        model = model_class()
        logger.info(f"✓ 模型创建成功: {name}")

        # 评估
        evaluator = ImprovedRealDataFewShotEvaluator(eval_config)
        result = evaluator.evaluate(model)

        logger.info(f"✓ 评估完成")
        logger.info(f"  准确率: {result.accuracy * 100:.1f}%")
        logger.info(f"  FLOPs: {result.flops / 1e6:.1f}M")
        logger.info(f"  参数量: {result.params / 1e6:.1f}M")

        return {
            "name": name,
            "accuracy": result.accuracy,
            "flops": result.flops,
            "params": result.params,
            "success": True,
        }

    except Exception as e:
        logger.error(f"✗ 评估失败: {e}")
        return {"name": name, "error": str(e), "success": False}


def main():
    logger.info("=" * 60)
    logger.info("Phase 5.6: Baseline Comparison")
    logger.info("=" * 60)

    # 评估配置 (与 Phase 5.6 相同)
    eval_config = {
        "dataset": "mmmu",
        "num_shots": 128,
        "train_epochs": 15,
        "batch_size": 8,
        "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.005},
        "max_training_time": 600,
    }

    # Baselines 评估
    baselines = [
        (FiLMFusion, "FiLM (人工设计)"),
        (CLIPFusion, "CLIPFusion"),
        (ConcatMLP, "ConcatMLP"),
        (BilinearPooling, "BilinearPooling"),
    ]

    results = []
    for model_class, name in baselines:
        result = evaluate_baseline(model_class, name, eval_config)
        results.append(result)

    # 保存结果
    output_dir = "/usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl/results/v3"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 生成报告
    logger.info("\n" + "=" * 60)
    logger.info("Baseline 对比报告")
    logger.info("=" * 60)

    print("\n| 架构 | 类型 | 准确率 | FLOPs | 参数量 |")
    print("|------|------|--------|-------|--------|")

    for r in results:
        if r.get("success"):
            acc = r["accuracy"] * 100
            flops = r["flops"] / 1e6
            params = r["params"] / 1e6
            print(
                f"| {r['name']} | Baseline | {acc:.1f}% | {flops:.1f}M | {params:.1f}M |"
            )
        else:
            print(f"| {r['name']} | Baseline | 失败 | - | - |")

    logger.info(f"\n结果已保存: {output_dir}/baseline_results.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
