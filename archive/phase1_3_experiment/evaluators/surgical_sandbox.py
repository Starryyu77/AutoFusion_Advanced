"""
Surgical Sandbox Evaluator
--------------------------
Freeze backbone networks, only train fusion layers for fast evaluation.

Key innovation: Surgical evaluation enables second-level iteration.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from base import BaseEvaluator, EvaluationResult


class SurgicalSandboxEvaluator(BaseEvaluator):
    """
    Surgical Sandbox Evaluator

    特点:
    1. 冻结预训练MLLM主干网络
    2. 只训练融合模块
    3. 快速评估 (秒级而非小时级)
    4. 特征缓存加速

    Theory: NAS不需要端到端训练，只需要评估融合模块质量
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 评估配置
        self.quick_train_epochs = config.get('quick_train_epochs', 5)
        self.use_lr_decay = config.get('use_lr_decay', True)
        self.batch_size = config.get('batch_size', 8)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 模拟数据配置 (用于快速评估)
        self.num_samples = config.get('num_samples', 100)
        self.vision_dim = config.get('vision_dim', 768)
        self.language_dim = config.get('language_dim', 768)
        self.num_classes = config.get('num_classes', 10)

        # 特征缓存
        self.use_cache = config.get('use_cache', True)
        self._cached_features = None

    def evaluate(self, code: str, context: Optional[Dict] = None) -> EvaluationResult:
        """
        评估生成的代码

        Returns:
            EvaluationResult with accuracy, efficiency, compile_success
        """
        # 1. 编译检查
        compile_success, module_or_error = self.compile_code(code)

        if not compile_success:
            return EvaluationResult(
                accuracy=0.0,
                efficiency=0.0,
                compile_success=0.0,
                flops=0.0,
                params=0.0,
                latency=0.0,
                metadata={'error': str(module_or_error)}
            )

        model_class = module_or_error

        try:
            # 2. 实例化模型
            model = model_class(
                vision_dim=self.vision_dim,
                language_dim=self.language_dim
            ).to(self.device)

            # 3. 计算效率指标
            flops = self.compute_flops(model)
            params = self.compute_params(model)

            # 4. 快速训练评估
            accuracy = self._quick_train(model)

            # 5. 计算延迟
            latency = self._measure_latency(model)

            # 6. 计算效率分数
            efficiency = self._compute_efficiency_score(flops, params, latency)

            return EvaluationResult(
                accuracy=accuracy,
                efficiency=efficiency,
                compile_success=1.0,
                flops=flops,
                params=params,
                latency=latency,
                metadata={
                    'epochs_trained': self.quick_train_epochs,
                    'final_loss': self._last_loss if hasattr(self, '_last_loss') else None,
                }
            )

        except Exception as e:
            return EvaluationResult(
                accuracy=0.0,
                efficiency=0.0,
                compile_success=0.0,  # 运行时错误视为编译失败
                flops=0.0,
                params=0.0,
                latency=0.0,
                metadata={'runtime_error': str(e)}
            )

    def _quick_train(self, model: nn.Module) -> float:
        """
        快速训练评估

        使用模拟数据进行快速训练，返回验证准确率
        """
        model.train()

        # 生成模拟数据
        vision_features, language_features, labels = self._generate_mock_data()

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # 学习率调度
        if self.use_lr_decay:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.quick_train_epochs
            )

        # 训练循环
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.quick_train_epochs):
            total_loss = 0
            correct = 0
            total = 0

            # 模拟batch训练
            num_batches = self.num_samples // self.batch_size

            for _ in range(num_batches):
                # 随机采样batch
                indices = torch.randint(0, self.num_samples, (self.batch_size,))
                v_batch = vision_features[indices]
                l_batch = language_features[indices]
                y_batch = labels[indices]

                # 前向传播
                optimizer.zero_grad()
                output = model(v_batch, l_batch)

                # 计算损失
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # 计算准确率
                pred = output.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)

            if self.use_lr_decay:
                scheduler.step()

            avg_loss = total_loss / num_batches
            accuracy = correct / total

            self._last_loss = avg_loss

        # 返回最终验证准确率 (模拟)
        # 实际场景中这里会使用真实验证集
        return accuracy

    def _generate_mock_data(self):
        """生成模拟训练数据"""
        if self._cached_features is not None and self.use_cache:
            return self._cached_features

        vision_features = torch.randn(self.num_samples, self.vision_dim, device=self.device)
        language_features = torch.randn(self.num_samples, self.language_dim, device=self.device)
        labels = torch.randint(0, self.num_classes, (self.num_samples,), device=self.device)

        self._cached_features = (vision_features, language_features, labels)
        return vision_features, language_features, labels

    def _measure_latency(self, model: nn.Module, num_runs: int = 10) -> float:
        """测量推理延迟"""
        model.eval()

        # 生成测试输入
        v = torch.randn(self.batch_size, self.vision_dim, device=self.device)
        l = torch.randn(self.batch_size, self.language_dim, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(v, l)

        # 测量
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(v, l)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()

        avg_latency = (end - start) / num_runs * 1000  # ms
        return avg_latency

    def _compute_efficiency_score(self, flops: float, params: float, latency: float) -> float:
        """
        计算效率分数

        综合考虑FLOPs、参数量和延迟
        """
        # 归一化
        norm_flops = min(1.0, 1e9 / (flops + 1e8))
        norm_params = min(1.0, 1e7 / (params + 1e6))
        norm_latency = min(1.0, 100.0 / (latency + 10))

        # 加权平均
        efficiency = 0.4 * norm_flops + 0.3 * norm_params + 0.3 * norm_latency
        return efficiency

    def compute_flops(self, model: nn.Module, input_shape: tuple = None) -> float:
        """
        估算FLOPs

        简化版本，实际可以使用ptflops或fvcore
        """
        total_flops = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Linear: 2 * input_dim * output_dim (乘加)
                flops = 2 * module.in_features * module.out_features
                total_flops += flops
            elif isinstance(module, nn.MultiheadAttention):
                # Attention: 4 * seq_len * dim^2 (简化)
                flops = 4 * module.embed_dim ** 2
                total_flops += flops
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: 5 * dim
                flops = 5 * module.normalized_shape[0]
                total_flops += flops
            elif isinstance(module, nn.GELU):
                # GELU: 8 * dim (近似)
                if hasattr(module, 'in_features'):
                    flops = 8 * module.in_features
                    total_flops += flops

        return float(total_flops)
