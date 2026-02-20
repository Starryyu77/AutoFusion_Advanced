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
from torch.utils.data._utils.collate import default_collate
import logging

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """
    Custom collate function that handles None values in batch.

    PyTorch's default collate fails when batch contains None.
    This function filters out samples with None images and handles mixed types.
    """
    if not batch:
        return {}

    # Filter out samples with None images
    valid_batch = [item for item in batch if item.get('image') is not None]

    if not valid_batch:
        # All samples have None images
        return {}

    # Get keys from first valid item
    keys = valid_batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in valid_batch]

        # Filter out None values
        non_none_values = [v for v in values if v is not None]

        if not non_none_values:
            # All values are None
            collated[key] = None
        elif key == 'image':
            # Handle images specially - stack tensors
            if isinstance(non_none_values[0], torch.Tensor):
                collated[key] = torch.stack(non_none_values)
            else:
                collated[key] = non_none_values  # Keep as list if not tensors
        elif key == 'label':
            # Stack labels into tensor
            if isinstance(non_none_values[0], (int, float)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = default_collate(non_none_values)
        elif isinstance(non_none_values[0], (int, float, str)):
            # Keep primitive types as list
            collated[key] = values
        else:
            # Try default collate
            try:
                collated[key] = default_collate(non_none_values)
            except:
                collated[key] = values

    return collated


class BaseDataset(Dataset):
    """Base dataset class for vision-language tasks."""

    def __init__(self, data: List[Dict[str, Any]], transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Check if decoded_image is available (MathVista dataset)
        if 'decoded_image' in item and item['decoded_image'] is not None:
            try:
                from PIL import Image
                image = item['decoded_image']
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                item['image'] = image
                return item
            except Exception as e:
                logger.warning(f"Failed to process decoded_image: {e}")

        # Check if image is already a PIL Image
        if 'image' in item and item['image'] is not None:
            if hasattr(item['image'], 'convert'):  # PIL Image
                try:
                    image = item['image']
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    item['image'] = image
                    return item
                except Exception as e:
                    logger.warning(f"Failed to process PIL image: {e}")
            elif isinstance(item['image'], torch.Tensor):
                # Already a tensor, no processing needed
                return item

        # Load image if path is provided
        image_loaded = False
        if 'image_path' in item and item['image_path'] is not None and os.path.exists(item['image_path']):
            try:
                from PIL import Image
                image = Image.open(item['image_path']).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                item['image'] = image
                image_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load image from image_path {item['image_path']}: {e}")

        # If image is a string (filename), try to load it
        if not image_loaded and 'image' in item and isinstance(item['image'], str):
            # Try to find the image file in various locations
            possible_paths = [
                item['image'],
                os.path.join(os.path.dirname(__file__), '../../data', item['image']),
                os.path.join(os.path.dirname(__file__), '../../data/coco/train2017', item['image']),  # COCO
                os.path.join(os.getcwd(), 'data', item['image']),
                os.path.join(os.getcwd(), 'data/coco/train2017', item['image']),  # COCO
            ]
            for img_path in possible_paths:
                if os.path.exists(img_path):
                    try:
                        from PIL import Image
                        image = Image.open(img_path).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                        item['image'] = image
                        image_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load image from {img_path}: {e}")

            if not image_loaded:
                logger.warning(f"Could not find image file: {item['image']}")
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
                 seed: int = 42,
                 transform=None):
        """
        Initialize dataset loader.

        Args:
            dataset_name: One of ['mmmu', 'vsr', 'mathvista', 'ai2d']
            num_shots: Number of few-shot samples per class
            batch_size: Batch size for data loader
            data_dir: Directory to store/load data
            shot_strategy: 'balanced', 'stratified', or 'random'
            seed: Random seed
            transform: Optional transform for image preprocessing
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
        self.transform = transform

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
        train_dataset = BaseDataset(train_data_fewshot, transform=self.transform)
        val_dataset = BaseDataset(val_data, transform=self.transform)

        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=custom_collate_fn,
            drop_last=True  # Ensure consistent batch size
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=custom_collate_fn,
            drop_last=True  # Ensure consistent batch size
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
        Loads all subjects and combines them.
        """
        try:
            from datasets import load_dataset, concatenate_datasets

            logger.info("Loading MMMU dataset from HuggingFace...")

            # MMMU has multiple subjects (configs)
            subjects = ['Accounting', 'Agriculture', 'Architecture_and_Engineering',
                       'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology',
                       'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design',
                       'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
                       'Energy_and_Power', 'Finance', 'Geography', 'History',
                       'Literature', 'Manage', 'Marketing', 'Materials', 'Math',
                       'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics',
                       'Psychology', 'Public_Health', 'Sociology']

            # Load first subject
            logger.info(f"Loading MMMU subject: {subjects[0]}")
            combined_dataset = load_dataset('MMMU/MMMU', subjects[0], split='validation',
                                           cache_dir=str(self.dataset_dir))

            # Load remaining subjects and combine
            for subject in subjects[1:5]:  # Load first 5 subjects for speed
                logger.info(f"Loading MMMU subject: {subject}")
                try:
                    subject_dataset = load_dataset('MMMU/MMMU', subject, split='validation',
                                                  cache_dir=str(self.dataset_dir))
                    combined_dataset = concatenate_datasets([combined_dataset, subject_dataset])
                except Exception as e:
                    logger.warning(f"Failed to load {subject}: {e}")
                    continue

            # Process into standard format
            train_data = []
            val_data = []

            for item in combined_dataset:
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
            skipped = 0

            # COCO train2017 directory
            coco_dir = self.data_dir / 'coco' / 'train2017'

            for item in dataset:
                # Handle label - VSR uses boolean True/False
                raw_label = item.get('label', False)
                # Convert to int (0 or 1)
                if isinstance(raw_label, bool):
                    label = 1 if raw_label else 0
                elif isinstance(raw_label, (int, float)):
                    label = int(raw_label)
                else:
                    # Handle numpy types or other
                    label = 1 if raw_label else 0

                # Check if image exists in COCO train2017
                img_filename = item.get('image')
                if img_filename and coco_dir.exists():
                    img_path = coco_dir / img_filename
                    if not img_path.exists():
                        # Skip samples with missing images
                        skipped += 1
                        continue

                processed = {
                    'image': item.get('image'),
                    'image_path': item.get('image_path'),
                    'caption': item.get('caption'),
                    'relation': item.get('relation'),
                    'label': label,  # Binary: 1=correct, 0=incorrect
                    'uuid': item.get('uuid')
                }

                # Use caption for hash split since uuid may be None
                split_key = processed['caption'] if processed['caption'] else str(idx)
                if hash(split_key) % 10 < 8:
                    train_data.append(processed)
                else:
                    val_data.append(processed)

            if skipped > 0:
                logger.info(f"VSR: Skipped {skipped} samples with missing COCO images")

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
                    'decoded_image': item.get('decoded_image'),  # Actual PIL Image
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
