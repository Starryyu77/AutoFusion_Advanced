"""
统一评估接口

支持对任意融合模块进行完整评估:
- NAS发现的架构
- 传统人工设计基线
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type
from pathlib import Path
import json
import sys

# 添加项目路径
EXP_ROOT = Path(__file__).parent.parent.parent.parent / 'experiment'
sys.path.append(str(EXP_ROOT))

from evaluators.real_data_evaluator import RealDataFewShotEvaluator
from data.dataset_loader import get_dataset_loader


class UnifiedEvaluator:
    """
    统一评估器 - 支持NAS架构和基线的完整评估
    """

    def __init__(
        self,
        dataset: str = 'ai2d',
        epochs: int = 100,
        batch_size: int = 32,
        device: str = 'cuda',
        save_dir: Optional[str] = None
    ):
        """
        Args:
            dataset: 数据集名称
            epochs: 训练轮数 (3 for quick test, 100 for full)
            batch_size: 批次大小
            device: 计算设备
            save_dir: 结果保存目录
        """
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir or f'results/{dataset}'

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def evaluate_architecture(
        self,
        fusion_module: nn.Module,
        arch_name: str,
        num_runs: int = 1,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        评估单个架构

        Args:
            fusion_module: 融合模块
            arch_name: 架构名称
            num_runs: 运行次数 (用于统计显著性)
            seed: 随机种子

        Returns:
            评估结果字典
        """
        from .full_trainer import FullTrainer
        from .metrics import MetricsCalculator

        print(f"\n{'='*60}")
        print(f"评估架构: {arch_name}")
        print(f"数据集: {self.dataset}")
        print(f"Epochs: {self.epochs}")
        print(f"{'='*60}")

        results = {
            'arch_name': arch_name,
            'dataset': self.dataset,
            'epochs': self.epochs,
            'runs': []
        }

        # 多轮运行
        for run_idx in range(num_runs):
            print(f"\n--- Run {run_idx + 1}/{num_runs} ---")

            # 设置随机种子
            current_seed = seed + run_idx
            torch.manual_seed(current_seed)

            # 获取数据加载器
            train_loader, val_loader, test_loader = self._get_data_loaders()

            # 获取编码器 (使用CLIP)
            vision_encoder, text_encoder = self._get_encoders()

            # 获取类别数
            num_classes = self._get_num_classes()

            # 创建训练器
            trainer = FullTrainer(
                fusion_module=fusion_module,
                vision_encoder=vision_encoder,
                text_encoder=text_encoder,
                num_classes=num_classes,
                config={
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'save_best': True,
                    'early_stopping': self.epochs >= 50  # 完整训练时启用早停
                }
            )

            # 训练
            save_path = Path(self.save_dir) / arch_name / f'run_{run_idx}'
            history = trainer.train(train_loader, val_loader, save_dir=str(save_path))

            # 测试集评估
            test_results = trainer.evaluate(test_loader)

            # 计算效率指标
            metrics_calc = MetricsCalculator(device=self.device)
            efficiency_metrics = metrics_calc.calculate_all(
                fusion_module,
                input_shape=(1, 768),
                num_runs=100
            )

            # 汇总结果
            run_result = {
                'run_idx': run_idx,
                'seed': current_seed,
                'best_val_acc': history['best_val_acc'],
                'best_epoch': history['best_epoch'],
                'test_acc': test_results['accuracy'],
                'test_loss': test_results['loss'],
                **efficiency_metrics
            }

            results['runs'].append(run_result)

            print(f"Run {run_idx + 1} 结果:")
            print(f"  Best Val Acc: {history['best_val_acc']:.4f}")
            print(f"  Test Acc: {test_results['accuracy']:.4f}")
            print(f"  FLOPs: {efficiency_metrics['flops']:.2f}M")
            print(f"  Params: {efficiency_metrics['params']:.2f}M")
            print(f"  Latency: {efficiency_metrics['latency_ms']:.2f}ms")

        # 计算统计量
        if num_runs > 1:
            import numpy as np
            test_accs = [r['test_acc'] for r in results['runs']]
            results['mean_test_acc'] = float(np.mean(test_accs))
            results['std_test_acc'] = float(np.std(test_accs))
            results['min_test_acc'] = float(np.min(test_accs))
            results['max_test_acc'] = float(np.max(test_accs))

        # 保存结果
        self._save_results(results, arch_name)

        return results

    def evaluate_multiple(
        self,
        architectures: Dict[str, nn.Module],
        num_runs: int = 1
    ) -> Dict[str, Any]:
        """
        批量评估多个架构

        Args:
            architectures: {name: module} 字典
            num_runs: 每架构运行次数

        Returns:
            所有架构的评估结果
        """
        all_results = {}

        for arch_name, fusion_module in architectures.items():
            try:
                result = self.evaluate_architecture(
                    fusion_module, arch_name, num_runs
                )
                all_results[arch_name] = result
            except Exception as e:
                print(f"评估 {arch_name} 失败: {e}")
                all_results[arch_name] = {'error': str(e)}

        # 保存汇总结果
        summary_path = Path(self.save_dir) / 'summary.json'
        with open(summary_path, 'w') as f:
            # 转换不可序列化的对象
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("评估完成!")
        print(f"结果保存至: {self.save_dir}")
        print(f"{'='*60}")

        return all_results

    def _get_data_loaders(self):
        """获取数据加载器"""
        # 使用已验证的数据集配置
        from data.dataset_loader import get_dataset_loader

        # 对于完整训练(100 epochs)，使用更多样本
        # 对于few-shot快速测试，使用num_shots
        if self.epochs >= 50:
            # 完整训练模式 - 使用更多样本
            loader = get_dataset_loader(
                dataset_name=self.dataset,
                batch_size=self.batch_size,
                num_shots=256,  # 更多样本用于完整训练
                data_dir='./data'
            )
        else:
            # 快速测试模式 - 使用few-shot
            loader = get_dataset_loader(
                dataset_name=self.dataset,
                batch_size=self.batch_size,
                num_shots=16,  # few-shot快速测试
                data_dir='./data'
            )

        # DatasetLoader.load() 返回 (train_loader, test_loader)
        train_loader, test_loader = loader.load()
        # 使用test作为val (数据集通常只有train/test split)
        return train_loader, test_loader, test_loader

    def _get_encoders(self):
        """获取预训练编码器"""
        import clip

        model, _ = clip.load('ViT-L/14', device=self.device)

        class VisionEncoder(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.model = clip_model

            def forward(self, images):
                return self.model.encode_image(images).float()

        class TextEncoder(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.model = clip_model

            def forward(self, texts):
                # texts 已经是 tokenized
                if isinstance(texts, list):
                    texts = clip.tokenize(texts, truncate=True).to(self.model.device)
                return self.model.encode_text(texts).float()

        return VisionEncoder(model), TextEncoder(model)

    def _get_num_classes(self) -> int:
        """获取数据集类别数"""
        dataset_classes = {
            'ai2d': 4,
            'mmmu': 4,
            'vsr': 2,
            'mathvista': 5
        }
        return dataset_classes.get(self.dataset, 4)

    def _save_results(self, results: Dict, arch_name: str):
        """保存评估结果"""
        result_path = Path(self.save_dir) / arch_name / 'evaluation_results.json'
        result_path.parent.mkdir(parents=True, exist_ok=True)

        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)


class QuickEvaluator:
    """
    快速评估器 - 用于本地测试 (10 epochs)
    """

    def __init__(
        self,
        dataset: str = 'ai2d',
        device: str = 'cuda'
    ):
        self.evaluator = UnifiedEvaluator(
            dataset=dataset,
            epochs=10,  # 快速测试
            batch_size=16,
            device=device
        )

    def evaluate(self, fusion_module: nn.Module, arch_name: str) -> Dict:
        """快速评估"""
        return self.evaluator.evaluate_architecture(
            fusion_module, arch_name, num_runs=1
        )


class FullEvaluator:
    """
    完整评估器 - 用于论文实验 (100 epochs, 3 runs)
    """

    def __init__(
        self,
        dataset: str = 'ai2d',
        device: str = 'cuda',
        save_dir: Optional[str] = None
    ):
        self.evaluator = UnifiedEvaluator(
            dataset=dataset,
            epochs=100,
            batch_size=32,
            device=device,
            save_dir=save_dir
        )

    def evaluate(
        self,
        fusion_module: nn.Module,
        arch_name: str,
        num_runs: int = 3
    ) -> Dict:
        """完整评估"""
        return self.evaluator.evaluate_architecture(
            fusion_module, arch_name, num_runs=num_runs, seed=42
        )

    def evaluate_batch(
        self,
        architectures: Dict[str, nn.Module],
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """批量完整评估"""
        return self.evaluator.evaluate_multiple(architectures, num_runs)
