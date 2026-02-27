"""
OOM Handler
-----------
Out-of-memory protection and recovery mechanisms.
"""

import functools
import torch
import gc
import os
from typing import Callable, Any


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_info(gpu_id: int = 0) -> dict:
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return {'available': False}

    return {
        'available': True,
        'total': torch.cuda.get_device_properties(gpu_id).total_memory / 1e9,  # GB
        'allocated': torch.cuda.memory_allocated(gpu_id) / 1e9,
        'reserved': torch.cuda.memory_reserved(gpu_id) / 1e9,
        'free': (torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)) / 1e9,
    }


def oom_retry(max_retries: int = 3, batch_reduction: float = 0.5):
    """
    自动重试并减小batch_size的装饰器

    Args:
        max_retries: 最大重试次数
        batch_reduction: batch_size减小比例

    Example:
        @oom_retry(max_retries=3, batch_reduction=0.5)
        def train_model(config, model, data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试从args或kwargs中获取config
            config = None
            if args and isinstance(args[0], dict):
                config = args[0]
            elif 'config' in kwargs:
                config = kwargs['config']

            original_batch_size = None
            if config and 'batch_size' in config:
                original_batch_size = config['batch_size']

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        print(f"OOM detected! Attempt {attempt + 1}/{max_retries}")
                        clear_gpu_cache()

                        if config and 'batch_size' in config:
                            new_batch_size = int(config['batch_size'] * batch_reduction)
                            if new_batch_size < 1:
                                raise RuntimeError(f"Batch size reduced to 0, cannot continue")
                            print(f"Reducing batch size: {config['batch_size']} -> {new_batch_size}")
                            config['batch_size'] = new_batch_size
                        else:
                            # 无法调整batch_size，直接重试
                            print("Cannot adjust batch_size, retrying...")
                    else:
                        raise

            # 恢复原始batch_size
            if original_batch_size and config:
                config['batch_size'] = original_batch_size

            raise RuntimeError(f"OOM after {max_retries} retries")

        return wrapper
    return decorator


def adaptive_batch_size(max_batch_size: int = 64, min_batch_size: int = 1) -> int:
    """
    根据可用显存自适应选择batch_size

    Args:
        max_batch_size: 最大batch_size
        min_batch_size: 最小batch_size

    Returns:
        推荐的batch_size
    """
    if not torch.cuda.is_available():
        return min_batch_size

    # 获取可用显存 (GB)
    free_memory = get_gpu_memory_info()['free']

    # 简单启发式: 每GB显存约支持8个样本
    estimated_batch = int(free_memory * 8)

    # 限制在合理范围内
    batch_size = max(min_batch_size, min(max_batch_size, estimated_batch))

    # 对齐到2的幂次 (性能优化)
    batch_size = 2 ** int(batch_size.bit_length() - 1)

    return batch_size


class GPUMemoryMonitor:
    """GPU内存监控器"""

    def __init__(self, gpu_id: int = 0, threshold: float = 0.9):
        self.gpu_id = gpu_id
        self.threshold = threshold  # 显存使用阈值
        self.peak_memory = 0

    def check(self) -> bool:
        """检查显存是否超过阈值"""
        if not torch.cuda.is_available():
            return True

        info = get_gpu_memory_info(self.gpu_id)
        usage_ratio = info['allocated'] / info['total']

        self.peak_memory = max(self.peak_memory, info['allocated'])

        return usage_ratio < self.threshold

    def get_peak(self) -> float:
        """获取峰值显存使用 (GB)"""
        return self.peak_memory

    def reset(self):
        """重置峰值记录"""
        self.peak_memory = 0


def enable_gradient_checkpointing(model: torch.nn.Module) -> torch.nn.Module:
    """
    启用梯度检查点以节省显存

    Args:
        model: PyTorch模型

    Returns:
        修改后的模型
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    elif hasattr(model, 'set_gradient_checkpointing'):
        model.set_gradient_checkpointing(True)

    return model
