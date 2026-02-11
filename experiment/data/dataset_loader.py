"""
Dataset Loader
--------------
Unified interface for loading MMMU, VSR, MathVista, and AI2D datasets.

Usage:
    loader = get_dataset_loader('mmmu', num_shots=16)
    train_loader, val_loader = loader.load()

Supported datasets:
- MMMU: Multi-discipline multimodal understanding
- VSR: Visual spatial reasoning
- MathVista: Visual mathematical reasoning
- AI2D: Diagram understanding
"""

import os
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base dataset class for vision-language tasks."""

    def __init__(self, data: List[Dict[str, Any]], transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image if path is provided
        if 'image_path' in item and os.path.exists(item['image_path']):
            try:
                from PIL import Image
                image = Image.open(item['image_path']).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                item['image'] = image
            except Exception as e:
                logger.warning(f"Failed to load image {item['image_path']}: {e}")
                item['image'] = None

        return item


class DatasetLoader:
    """
    Unified dataset loader for real-data evaluation.

    Supports:
    - MMMU (Multi-discipline multimodal understanding)
    - VSR (Visual spatial reasoning)
    - MathVista (Visual math reasoning)
    - AI2D (Diagram understanding)
    """

    SUPPORTED_DATASETS = ['mmmu', 'vsr', 'mathvista', 'ai2d']

    def __init__(self,
                 dataset_name: str,
                 num_shots: int = 16,
                 batch_size: int = 4,
                 data_dir: str = './data',
                 shot_strategy: str = 'balanced',
                 seed: int = 42):
        """
        Initialize dataset loader.

        Args:
            dataset_name: One of ['mmmu', 'vsr', 'mathvista', 'ai2d']
            num_shots: Number of few-shot samples per class
            batch_size: Batch size for data loader
            data_dir: Directory to store/load data
            shot_strategy: 'balanced', 'stratified', or 'random'
            seed: Random seed
        """
        if dataset_name.lower() not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Choose from {self.SUPPORTED_DATASETS}")

        self.dataset_name = dataset_name.lower()
        self.num_shots = num_shots
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.shot_strategy = shot_strategy
        self.seed = seed

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.data_dir / self.dataset_name
        self.dataset_dir.mkdir(exist_ok=True)

        # Import sampler
        from .few_shot_sampler import FewShotSampler
        self.sampler = FewShotSampler(seed=seed)

    def load(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load dataset and return train/val loaders.

        Returns:
            (train_loader, val_loader): Few-shot train and full validation
        """
        # Load raw data
        train_data, val_data = self._load_raw_data()

        # Sample few-shot training data
        train_data_fewshot = self.sampler.sample(
            train_data,
            num_shots=self.num_shots,
            strategy=self.shot_strategy,
            label_key='label'
        )

        logger.info(f"{self.dataset_name}: Loaded {len(train_data_fewshot)} few-shot train, "
                   f"{len(val_data)} validation samples")

        # Create datasets
        train_dataset = BaseDataset(train_data_fewshot)
        val_dataset = BaseDataset(val_data)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return train_loader, val_loader

    def _load_raw_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load raw data from source."""
        if self.dataset_name == 'mmmu':
            return self._load_mmmu()
        elif self.dataset_name == 'vsr':
            return self._load_vsr()
        elif self.dataset_name == 'mathvista':
            return self._load_mathvista()
        elif self.dataset_name == 'ai2d':
            return self._load_ai2d()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _load_mmmu(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load MMMU dataset.

        MMMU: College-level multi-discipline multimodal understanding.
        """
        try:
            from datasets import load_dataset

            logger.info("Loading MMMU dataset from HuggingFace...")

            # Load validation split (contains all data)
            dataset = load_dataset('MMMU/MMMU', split='validation', cache_dir=str(self.dataset_dir))

            # Process into standard format
            train_data = []
            val_data = []

            for item in dataset:
                processed = {
                    'image': item.get('image'),
                    'image_path': item.get('image_path'),
                    'question': item.get('question'),
                    'choices': item.get('choices', []),
                    'answer': item.get('answer'),
                    'label': item.get('answer_index', 0),  # Use answer index as label
                    'subject': item.get('id', '').split('_')[0] if item.get('id') else 'unknown'
                }

                # Split: 80% for few-shot sampling, 20% for validation
                # This is a simple split; in practice, you might want stratified split
                if hash(processed['question']) % 10 < 8:
                    train_data.append(processed)
                else:
                    val_data.append(processed)

            return train_data, val_data

        except Exception as e:
            logger.error(f"Failed to load MMMU: {e}")
            logger.info("Returning mock data for testing...")
            return self._create_mock_data(num_classes=10)

    def _load_vsr(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load VSR (Visual Spatial Reasoning) dataset.

        VSR: Spatial relationship understanding (left, right, contains, etc.)
        """
        try:
            from datasets import load_dataset

            logger.info("Loading VSR dataset from HuggingFace...")

            dataset = load_dataset('cambridgeltl/vsr_random', split='train', cache_dir=str(self.dataset_dir))

            train_data = []
            val_data = []

            for item in dataset:
                processed = {
                    'image': item.get('image'),
                    'image_path': item.get('image_path'),
                    'caption': item.get('caption'),
                    'relation': item.get('relation'),
                    'label': 1 if item.get('label', False) else 0,  # Binary: 1=correct, 0=incorrect
                    'uuid': item.get('uuid')
                }

                if hash(processed['uuid']) % 10 < 8:
                    train_data.append(processed)
                else:
                    val_data.append(processed)

            return train_data, val_data

        except Exception as e:
            logger.error(f"Failed to load VSR: {e}")
            logger.info("Returning mock data for testing...")
            return self._create_mock_data(num_classes=2)

    def _load_mathvista(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load MathVista dataset.

        MathVista: Visual mathematical reasoning.
        """
        try:
            from datasets import load_dataset

            logger.info("Loading MathVista dataset from HuggingFace...")

            dataset = load_dataset('AI4Math/MathVista', split='test', cache_dir=str(self.dataset_dir))

            train_data = []
            val_data = []

            for item in dataset:
                processed = {
                    'image': item.get('image'),
                    'image_path': item.get('image_path'),
                    'question': item.get('question'),
                    'choices': item.get('choices', []),
                    'answer': item.get('answer'),
                    'label': item.get('answer_index', 0),
                    'task': item.get('task')
                }

                if hash(processed['question']) % 10 < 8:
                    train_data.append(processed)
                else:
                    val_data.append(processed)

            return train_data, val_data

        except Exception as e:
            logger.error(f"Failed to load MathVista: {e}")
            logger.info("Returning mock data for testing...")
            return self._create_mock_data(num_classes=5)

    def _load_ai2d(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load AI2D dataset.

        AI2D: Diagram understanding with text labels and arrows.
        """
        try:
            from datasets import load_dataset

            logger.info("Loading AI2D dataset from HuggingFace...")

            dataset = load_dataset('lmms-lab/AI2D', split='test', cache_dir=str(self.dataset_dir))

            train_data = []
            val_data = []

            for item in dataset:
                processed = {
                    'image': item.get('image'),
                    'image_path': item.get('image_path'),
                    'question': item.get('question'),
                    'choices': item.get('choices', []),
                    'answer': item.get('answer'),
                    'label': item.get('answer_index', 0),
                    'diagram_id': item.get('diagram_id')
                }

                if hash(processed['question']) % 10 < 8:
                    train_data.append(processed)
                else:
                    val_data.append(processed)

            return train_data, val_data

        except Exception as e:
            logger.error(f"Failed to load AI2D: {e}")
            logger.info("Returning mock data for testing...")
            return self._create_mock_data(num_classes=4)

    def _create_mock_data(self, num_classes: int = 10, num_samples: int = 100) -> Tuple[List[Dict], List[Dict]]:
        """Create mock data for testing when dataset loading fails."""
        import numpy as np

        np.random.seed(self.seed)

        train_data = []
        val_data = []

        for i in range(num_samples):
            sample = {
                'image': None,
                'image_path': None,
                'question': f'Mock question {i}',
                'choices': [f'Choice {j}' for j in range(4)],
                'answer': 'Choice 0',
                'label': np.random.randint(0, num_classes),
                'mock': True
            }

            if i < num_samples * 0.8:
                train_data.append(sample)
            else:
                val_data.append(sample)

        return train_data, val_data

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'name': self.dataset_name,
            'num_shots': self.num_shots,
            'shot_strategy': self.shot_strategy,
            'batch_size': self.batch_size,
            'data_dir': str(self.dataset_dir)
        }


def get_dataset_loader(dataset_name: str, **kwargs) -> DatasetLoader:
    """
    Factory function to get dataset loader.

    Args:
        dataset_name: Name of dataset
        **kwargs: Additional arguments for DatasetLoader

    Returns:
        DatasetLoader instance
    """
    return DatasetLoader(dataset_name, **kwargs)
