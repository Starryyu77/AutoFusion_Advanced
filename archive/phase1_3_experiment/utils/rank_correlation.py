"""
Rank Correlation Validation
---------------------------
Validate proxy evaluation correlation with full evaluation.

Key insight: If proxy_epochs evaluation doesn't correlate with full_epochs,
the search will be misled. This is a Phase 0 mandatory check.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
import torch


def compute_kendall_tau(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    """
    计算Kendall's tau秩相关系数

    Args:
        scores_a: 第一组分数
        scores_b: 第二组分数

    Returns:
        (tau, p_value)
        - tau: 相关系数 (-1 to 1, 1表示完全一致)
        - p_value: 显著性水平
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Length mismatch: {len(scores_a)} vs {len(scores_b)}")

    if len(scores_a) < 3:
        return 0.0, 1.0  # 样本太少，无法计算

    tau, p_value = stats.kendalltau(scores_a, scores_b)
    return tau, p_value


def compute_spearman_rho(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    """
    计算Spearman's rho秩相关系数

    Args:
        scores_a: 第一组分数
        scores_b: 第二组分数

    Returns:
        (rho, p_value)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Length mismatch: {len(scores_a)} vs {len(scores_b)}")

    if len(scores_a) < 3:
        return 0.0, 1.0

    rho, p_value = stats.spearmanr(scores_a, scores_b)
    return rho, p_value


def validate_rank_correlation(
    architectures: List[Dict[str, Any]],
    evaluator,
    proxy_epochs: int = 5,
    full_epochs: int = 20,
    num_samples: int = 10,
) -> Dict[str, Any]:
    """
    验证代理评估的秩相关性

    Args:
        architectures: 架构列表
        evaluator: 评估器实例
        proxy_epochs: 代理评估epoch数
        full_epochs: 完整评估epoch数
        num_samples: 采样的架构数量

    Returns:
        验证结果字典
    """
    # 采样架构
    if len(architectures) > num_samples:
        indices = np.random.choice(len(architectures), num_samples, replace=False)
        sampled_archs = [architectures[i] for i in indices]
    else:
        sampled_archs = architectures

    proxy_scores = []
    full_scores = []

    print(f"Validating rank correlation with {len(sampled_archs)} architectures...")
    print(f"Proxy: {proxy_epochs} epochs, Full: {full_epochs} epochs")

    for i, arch in enumerate(sampled_archs):
        print(f"  [{i+1}/{len(sampled_archs)}] Evaluating {arch.get('type', 'unknown')}...")

        # 代理评估
        evaluator.quick_train_epochs = proxy_epochs
        proxy_result = evaluator.evaluate(arch.get('code', ''))
        proxy_acc = proxy_result.accuracy
        proxy_scores.append(proxy_acc)

        # 完整评估
        evaluator.quick_train_epochs = full_epochs
        full_result = evaluator.evaluate(arch.get('code', ''))
        full_acc = full_result.accuracy
        full_scores.append(full_acc)

        print(f"    Proxy: {proxy_acc:.4f}, Full: {full_acc:.4f}")

    # 计算相关性
    kendall_tau, kendall_p = compute_kendall_tau(proxy_scores, full_scores)
    spearman_rho, spearman_p = compute_spearman_rho(proxy_scores, full_scores)

    # Pearson相关系数 (作为参考)
    pearson_r, pearson_p = stats.pearsonr(proxy_scores, full_scores)

    result = {
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_p,
        'spearman_rho': spearman_rho,
        'spearman_p_value': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p_value': pearson_p,
        'proxy_scores': proxy_scores,
        'full_scores': full_scores,
        'num_samples': len(sampled_archs),
        'proxy_epochs': proxy_epochs,
        'full_epochs': full_epochs,
    }

    # 评估结果
    print("\n" + "="*50)
    print("Rank Correlation Validation Results")
    print("="*50)
    print(f"Kendall's τ: {kendall_tau:.3f} (p={kendall_p:.4f})")
    print(f"Spearman's ρ: {spearman_rho:.3f} (p={spearman_p:.4f})")
    print(f"Pearson's r: {pearson_r:.3f} (p={pearson_p:.4f})")

    if kendall_tau >= 0.7:
        print("✅ Excellent correlation! Proxy evaluation is reliable.")
        result['recommendation'] = 'continue'
    elif kendall_tau >= 0.5:
        print("⚠️  Moderate correlation. Consider increasing proxy_epochs.")
        result['recommendation'] = 'increase_epochs'
    else:
        print("❌ Poor correlation! Proxy evaluation may mislead search.")
        print("   Recommendations:")
        print("   1. Increase proxy_epochs (try 10+)")
        print("   2. Use cosine annealing LR schedule")
        print("   3. Consider early stopping")
        result['recommendation'] = 'significant_improvement_needed'

    print("="*50)

    return result


def check_rank_correlation_threshold(
    result: Dict[str, Any],
    threshold: float = 0.6
) -> bool:
    """
    检查秩相关性是否超过阈值

    Args:
        result: validate_rank_correlation的返回结果
        threshold: 最小可接受阈值

    Returns:
        是否通过检查
    """
    return result.get('kendall_tau', 0) >= threshold


class RankCorrelationTracker:
    """秩相关性追踪器 - 用于监控搜索过程中的相关性变化"""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.proxy_scores = []
        self.full_scores = []

    def add(self, proxy_score: float, full_score: float = None):
        """添加新分数"""
        self.proxy_scores.append(proxy_score)
        if full_score is not None:
            self.full_scores.append(full_score)

        # 保持窗口大小
        if len(self.proxy_scores) > self.window_size:
            self.proxy_scores.pop(0)
        if len(self.full_scores) > self.window_size:
            self.full_scores.pop(0)

    def get_current_correlation(self) -> float:
        """获取当前窗口的相关系数"""
        if len(self.full_scores) < 5:
            return 1.0  # 数据不足，假设良好

        tau, _ = compute_kendall_tau(
            self.proxy_scores[-len(self.full_scores):],
            self.full_scores
        )
        return tau

    def should_trigger_full_eval(self, threshold: float = 0.5) -> bool:
        """判断是否应该触发完整评估"""
        if len(self.full_scores) < 5:
            return False

        return self.get_current_correlation() < threshold
