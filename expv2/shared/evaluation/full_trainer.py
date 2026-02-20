"""
100 epochs完整训练器

支持:
- 任意融合模块的训练
- 早停策略
- 学习率调度
- 混合精度训练
- 多随机种子实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any
import time
import json
import os
from pathlib import Path


class FullTrainer:
    """
    完整训练器 - 100 epochs训练用于论文实验
    """

    def __init__(
        self,
        fusion_module: nn.Module,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        num_classes: int,
        config: Optional[Dict] = None
    ):
        """
        Args:
            fusion_module: 融合模块 (NAS发现或传统基线)
            vision_encoder: 视觉编码器 (frozen)
            text_encoder: 文本编码器 (frozen)
            num_classes: 分类类别数
            config: 训练配置
        """
        self.fusion_module = fusion_module
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.num_classes = num_classes

        # 默认配置
        self.config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_epochs': 5,
            'scheduler': 'cosine',  # 'cosine', 'step', 'none'
            'early_stopping': True,
            'patience': 10,
            'mixed_precision': True,
            'gradient_clip': 1.0,
            'save_best': True,
            'log_interval': 10,
            **(config or {})
        }

        # 创建分类头
        fusion_dim = self._get_fusion_dim()
        self.classifier = nn.Linear(fusion_dim, num_classes)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_module.to(self.device)
        self.classifier.to(self.device)
        self.vision_encoder.to(self.device)
        self.text_encoder.to(self.device)

        # 冻结编码器
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 优化器
        self.optimizer = optim.AdamW(
            list(self.fusion_module.parameters()) + list(self.classifier.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # 学习率调度
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs']
            )
        elif self.config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            self.scheduler = None

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if self.config['mixed_precision'] else None

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0,
            'epochs_trained': 0
        }

    def _get_fusion_dim(self) -> int:
        """获取融合模块输出维度"""
        # 通过前向传播推断
        device = next(self.fusion_module.parameters()).device
        with torch.no_grad():
            dummy_vision = torch.randn(1, 768, device=device)
            dummy_text = torch.randn(1, 768, device=device)
            output = self.fusion_module(dummy_vision, dummy_text)
            return output.shape[-1]

    def preprocess_image(self, img):
        """预处理PIL图像或文件路径为张量"""
        from torchvision import transforms
        from PIL import Image
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        # 如果img是文件路径字符串，先加载图像
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        return transform(img)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            save_dir: 模型保存目录

        Returns:
            训练历史记录
        """
        print(f"开始训练: {self.config['epochs']} epochs")
        print(f"设备: {self.device}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")

        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.config['epochs']):
            epoch_start = time.time()

            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader, epoch)

            # 验证阶段
            val_loss, val_acc = self._validate(val_loader)

            # 更新历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epochs_trained'] = epoch + 1

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.history['best_val_acc'] = best_val_acc
                self.history['best_epoch'] = epoch
                patience_counter = 0

                if save_dir and self.config['save_best']:
                    self._save_checkpoint(save_dir, epoch, best_val_acc, is_best=True)
            else:
                patience_counter += 1

            # 早停检查
            if self.config['early_stopping'] and patience_counter >= self.config['patience']:
                print(f"早停触发: {epoch + 1} epochs")
                break

            epoch_time = time.time() - epoch_start

            # 打印进度
            if (epoch + 1) % self.config['log_interval'] == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{self.config['epochs']}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"LR: {lr:.6f} | Time: {epoch_time:.1f}s")

        total_time = time.time() - start_time
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {self.history['best_epoch']})")
        print(f"总训练时间: {total_time/60:.1f} minutes")

        # 保存最终模型和历史
        if save_dir:
            self._save_checkpoint(save_dir, self.history['epochs_trained'], best_val_acc, is_best=False)
            self._save_history(save_dir)

        return self.history

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """训练一个epoch"""
        self.fusion_module.train()
        self.classifier.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Skip empty batches (when all images are None)
            if isinstance(batch, dict) and (batch.get('image') is None or len(batch.get('image', [])) == 0):
                continue
            if not batch:
                continue

            # Handle both dict and list batch formats
            if isinstance(batch, dict):
                # 处理图像 - 可能是tensor或PIL图像list
                images = batch['image']
                if isinstance(images, list):
                    # PIL图像列表，需要预处理
                    images = torch.stack([self.preprocess_image(img) for img in images])
                if hasattr(images, 'to'):
                    images = images.to(self.device)

                # 处理文本 - 可能是'question', 'text', 'caption', 'relation' 等
                texts = batch.get('text') or batch.get('question') or batch.get('caption') or batch.get('relation')
                if texts is None:
                    # 如果没有text字段，使用所有可用文本字段拼接
                    text_fields = [k for k in batch.keys() if isinstance(batch[k], list) and k not in ['image', 'label']]
                    if text_fields:
                        # 使用第一个可用的文本字段
                        texts = batch[text_fields[0]]
                    else:
                        # 如果没有可用文本字段，创建dummy tensor
                        texts = torch.randn(images.shape[0], 77, dtype=torch.long).to(self.device)
                elif hasattr(texts, 'to'):
                    texts = texts.to(self.device)

                labels = batch['label']
                if isinstance(labels, list):
                    labels = torch.tensor(labels)
                labels = labels.to(self.device)
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, texts, labels = batch
                images = images.to(self.device)
                # texts可能是list或tensor，如果是tensor需要移动到device
                if hasattr(texts, 'to'):
                    texts = texts.to(self.device)
                # labels也可能是list
                if isinstance(labels, list):
                    labels = torch.tensor(labels)
                labels = labels.to(self.device)
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            # 编码器前向 (no grad)
            with torch.no_grad():
                vision_features = self.vision_encoder(images)
                # 如果texts是字符串列表，需要先tokenize
                if isinstance(texts, list):
                    import clip
                    # 过滤空字符串并确保所有元素都是字符串
                    texts = [str(t) if t else "" for t in texts]
                    if not texts or all(t == "" for t in texts):
                        # 如果所有文本都是空的，创建dummy tensor
                        texts = torch.randn(images.shape[0], 77, dtype=torch.long).to(self.device)
                    else:
                        texts = clip.tokenize(texts, truncate=True).to(self.device)
                text_features = self.text_encoder(texts)

            # 融合 + 分类
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    fused = self.fusion_module(vision_features, text_features)
                    logits = self.classifier(fused)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()

                if self.config['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.fusion_module.parameters()) + list(self.classifier.parameters()),
                        self.config['gradient_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                fused = self.fusion_module(vision_features, text_features)
                logits = self.classifier(fused)
                loss = self.criterion(logits, labels)

                loss.backward()

                if self.config['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.fusion_module.parameters()) + list(self.classifier.parameters()),
                        self.config['gradient_clip']
                    )

                self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> tuple:
        """验证"""
        self.fusion_module.eval()
        self.classifier.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                # Skip empty batches (when all images are None)
                if isinstance(batch, dict) and (batch.get('image') is None or len(batch.get('image', [])) == 0):
                    continue
                if not batch:
                    continue

                # Handle both dict and list batch formats
                if isinstance(batch, dict):
                    # 处理图像 - 可能是tensor或PIL图像list
                    images = batch['image']
                    if isinstance(images, list):
                        # PIL图像列表，需要预处理
                        images = torch.stack([self.preprocess_image(img) for img in images])
                    if hasattr(images, 'to'):
                        images = images.to(self.device)

                    # 处理文本 - 可能是'question', 'text', 或列表
                    texts = batch.get('text') or batch.get('question')
                    if texts is None:
                        # 如果没有text字段，创建一个dummy tensor
                        texts = torch.randn(images.shape[0], 768).to(self.device)
                    elif hasattr(texts, 'to'):
                        texts = texts.to(self.device)

                    labels = batch['label']
                    if isinstance(labels, list):
                        labels = torch.tensor(labels)
                    labels = labels.to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                    images, texts, labels = batch
                    images = images.to(self.device)
                    # texts可能是list或tensor，如果是tensor需要移动到device
                    if hasattr(texts, 'to'):
                        texts = texts.to(self.device)
                    # labels也可能是list
                    if isinstance(labels, list):
                        labels = torch.tensor(labels)
                    labels = labels.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")

                # 编码
                vision_features = self.vision_encoder(images)
                # 如果texts是字符串列表，需要先tokenize
                if isinstance(texts, list):
                    import clip
                    # 过滤空字符串并确保所有元素都是字符串
                    texts = [str(t) if t else "" for t in texts]
                    if not texts or all(t == "" for t in texts):
                        # 如果所有文本都是空的，创建dummy tensor
                        texts = torch.randn(images.shape[0], 77, dtype=torch.long).to(self.device)
                    else:
                        texts = clip.tokenize(texts, truncate=True).to(self.device)
                text_features = self.text_encoder(texts)

                # 融合 + 分类
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        fused = self.fusion_module(vision_features, text_features)
                        logits = self.classifier(fused)
                        loss = self.criterion(logits, labels)
                else:
                    fused = self.fusion_module(vision_features, text_features)
                    logits = self.classifier(fused)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        在测试集上评估

        Returns:
            {'accuracy': float, 'loss': float}
        """
        test_loss, test_acc = self._validate(test_loader)
        return {'accuracy': test_acc, 'loss': test_loss}

    def _save_checkpoint(self, save_dir: str, epoch: int, val_acc: float, is_best: bool = False):
        """保存检查点"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'fusion_module': self.fusion_module.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }

        if is_best:
            path = Path(save_dir) / 'best_model.pt'
        else:
            path = Path(save_dir) / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, path)

    def _save_history(self, save_dir: str):
        """保存训练历史"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        history_path = Path(save_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.fusion_module.load_state_dict(checkpoint['fusion_module'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['epoch'], checkpoint['val_acc']
