#!/usr/bin/env python3
"""
Baseline 评估脚本 - Phase 5.6 (修复版)
"""

import os
import sys
import json

sys.path.insert(0, "/usr1/home/s125mdg43_10/AutoFusion_Advanced")

from expv2.shared.baselines.film import FiLM
from expv2.shared.baselines.clip_fusion import CLIPFusion
from expv2.shared.baselines.concat_mlp import ConcatMLP
from expv2.shared.baselines.bilinear_pooling import BilinearPooling
from phase4_optimization.src.evaluator_v2_improved import (
    ImprovedRealDataFewShotEvaluator,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Baseline Evaluation Started")

    eval_config = {
        "dataset": "mmmu",
        "num_shots": 128,
        "train_epochs": 15,
        "batch_size": 8,
        "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.005},
        "max_training_time": 600,
    }

    # 评估 FiLM
    logger.info("Evaluating FiLM...")
    try:
        model = FiLM()
        evaluator = ImprovedRealDataFewShotEvaluator(eval_config)
        result = evaluator.evaluate(model)
        logger.info(
            f"FiLM: acc={result.accuracy * 100:.1f}%, flops={result.flops / 1e6:.1f}M"
        )
    except Exception as e:
        logger.error(f"FiLM failed: {e}")

    logger.info("Baseline evaluation completed")


if __name__ == "__main__":
    main()
