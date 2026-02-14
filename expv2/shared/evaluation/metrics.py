"""
指标计算模块

计算:
- FLOPs (浮点运算数)
- Parameters (参数量)
- Latency (推理延迟)
- Accuracy (准确率)
- Memory (显存占用)
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Tuple
import numpy as np


class MetricsCalculator:
    """指标计算器"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def calculate_all(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 768),
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            model: 融合模块
            input_shape: 输入特征维度 (batch, dim)
            num_runs: 延迟测试运行次数

        Returns:
            指标字典
        """
        model.eval()
        model.to(self.device)

        metrics = {}

        # 计算FLOPs和参数量
        flops, params = self.calculate_flops_params(model, input_shape)
        metrics['flops'] = flops
        metrics['params'] = params

        # 计算延迟
        latency_mean, latency_std = self.calculate_latency(model, input_shape, num_runs)
        metrics['latency_ms'] = latency_mean
        metrics['latency_std_ms'] = latency_std

        # 计算吞吐量
        throughput = 1000.0 / latency_mean  # items/sec
        metrics['throughput'] = throughput

        # 计算显存占用 (如果可用)
        if torch.cuda.is_available():
            memory = self.calculate_memory(model, input_shape)
            metrics['memory_mb'] = memory

        return metrics

    def calculate_flops_params(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 768)
    ) -> Tuple[float, float]:
        """
        计算FLOPs和参数量

        Returns:
            (flops, params) - 单位: MFLOPs, MParams
        """
        try:
            from thop import profile

            dummy_v = torch.randn(input_shape).to(self.device)
            dummy_t = torch.randn(input_shape).to(self.device)

            flops, params = profile(model, inputs=(dummy_v, dummy_t), verbose=False)

            return flops / 1e6, params / 1e6  # 转换为MFLOPs, MParams

        except ImportError:
            # 如果没有thop，使用简化计算
            params = sum(p.numel() for p in model.parameters()) / 1e6

            # 粗略估计FLOPs (基于参数量和层数)
            flops = self._estimate_flops(model, input_shape)

            return flops, params

    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """简化FLOPs估计"""
        total_flops = 0.0
        batch_size, dim = input_shape

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Linear: 2 * in_features * out_features * batch_size
                flops = 2 * module.in_features * module.out_features * batch_size
                total_flops += flops

            elif isinstance(module, nn.MultiheadAttention):
                # Attention: 4 * seq_len^2 * dim (简化)
                flops = 4 * dim * dim * batch_size
                total_flops += flops

            elif isinstance(module, nn.Bilinear):
                # Bilinear: 2 * in1_features * in2_features * out_features
                flops = 2 * module.in1_features * module.in2_features * module.out_features
                total_flops += flops

        return total_flops / 1e6  # MFLOPs

    def calculate_latency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 768),
        num_runs: int = 100,
        warmup: int = 10
    ) -> Tuple[float, float]:
        """
        计算推理延迟

        Returns:
            (mean_latency_ms, std_latency_ms)
        """
        model.eval()

        dummy_v = torch.randn(input_shape).to(self.device)
        dummy_t = torch.randn(input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_v, dummy_t)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 正式测试
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(dummy_v, dummy_t)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # 转换为ms

        return np.mean(times), np.std(times)

    def calculate_memory(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 768)
    ) -> float:
        """
        计算显存占用

        Returns:
            memory_mb
        """
        if not torch.cuda.is_available():
            return 0.0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dummy_v = torch.randn(input_shape).to(self.device)
        dummy_t = torch.randn(input_shape).to(self.device)

        with torch.no_grad():
            _ = model(dummy_v, dummy_t)

        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        return memory_mb

    def calculate_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        计算准确率

        Args:
            predictions: [N, C] logits 或 [N] 预测类别
            labels: [N] 真实标签

        Returns:
            accuracy (0-1)
        """
        if predictions.dim() == 2:
            predictions = predictions.argmax(dim=1)

        correct = (predictions == labels).sum().item()
        total = labels.size(0)

        return correct / total

    def calculate_topk_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5
    ) -> float:
        """
        计算Top-K准确率

        Args:
            predictions: [N, C] logits
            labels: [N] 真实标签
            k: top-k

        Returns:
            top-k accuracy
        """
        _, pred_k = predictions.topk(k, dim=1)
        correct = pred_k.eq(labels.view(-1, 1).expand_as(pred_k))
        correct_k = correct.any(dim=1).sum().item()

        return correct_k / labels.size(0)


class EfficiencyMetrics:
    """效率指标综合评估"""

    @staticmethod
    def compute_efficiency_score(
        accuracy: float,
        flops: float,
        params: float,
        latency: float,
        weights: Dict[str, float] = None
    ) -> float:
        """
        计算综合效率分数

        公式: score = accuracy / (flops^w1 * params^w2 * latency^w3)

        Args:
            accuracy: 准确率 (0-1)
            flops: MFLOPs
            params: MParams
            latency: ms
            weights: 各指标的权重

        Returns:
            效率分数 (越高越好)
        """
        if weights is None:
            weights = {'flops': 0.3, 'params': 0.3, 'latency': 0.4}

        # 归一化因子 (参考值)
        flops_norm = 100.0  # 100 MFLOPs
        params_norm = 10.0  # 10 MParams
        latency_norm = 10.0  # 10 ms

        efficiency = accuracy / (
            (flops / flops_norm) ** weights['flops'] *
            (params / params_norm) ** weights['params'] *
            (latency / latency_norm) ** weights['latency']
        )

        return efficiency

    @staticmethod
    def compute_pareto_frontier(
        points: list,
        objectives: list = ['max', 'min']  # max accuracy, min flops
    ) -> list:
        """
        计算帕累托前沿

        Args:
            points: [(acc1, flops1), (acc2, flops2), ...]
            objectives: 每个维度的优化目标

        Returns:
            帕累托前沿上的点索引列表
        """
        pareto_indices = []

        for i, point in enumerate(points):
            is_pareto = True

            for j, other in enumerate(points):
                if i == j:
                    continue

                # 检查是否被其他点支配
                dominated = True
                for p, o, obj in zip(point, other, objectives):
                    if obj == 'max':
                        if p > o:
                            dominated = False
                            break
                    else:  # min
                        if p < o:
                            dominated = False
                            break

                if dominated and point != other:
                    is_pareto = False
                    break

            if is_pareto:
                pareto_indices.append(i)

        return pareto_indices
