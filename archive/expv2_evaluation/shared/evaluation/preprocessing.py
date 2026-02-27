"""
预处理模块 - 用于数据加载器
"""

from torchvision import transforms
from PIL import Image
import torch


def get_preprocessing_transform(model_type='clip'):
    """
    获取预处理transform

    Args:
        model_type: 模型类型 ('clip', 'clip-vit-l-14')

    Returns:
        torchvision transforms
    """
    if model_type in ['clip', 'clip-vit-l-14']:
        # CLIP预处理
        # 来自CLIP官方: 224x224, ImageNet标准化
        return transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    else:
        # 默认预处理
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def preprocess_text(texts, model_type='clip'):
    """
    预处理文本

    Args:
        texts: 文本列表或单条文本
        model_type: 模型类型

    Returns:
        tokenized texts
    """
    if model_type in ['clip', 'clip-vit-l-14']:
        import clip
        if isinstance(texts, str):
            texts = [texts]
        return clip.tokenize(texts, truncate=True)
    return texts
