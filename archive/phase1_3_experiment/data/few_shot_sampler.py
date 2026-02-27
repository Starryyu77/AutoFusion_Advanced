"""
Few-Shot Sampler
----------------
Sampling strategies for few-shot learning.

Strategies:
- Balanced: Equal samples per class
- Stratified: Proportional to class distribution
- Random: Completely random
"""

import random
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict


class FewShotSampler:
    """
    Few-Shot Sampler with multiple strategies.

    Usage:
        sampler = FewShotSampler()
        few_shot_data = sampler.sample(dataset, num_shots=16, strategy='balanced')
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def sample(self,
               dataset: List[Dict[str, Any]],
               num_shots: int,
               strategy: str = 'balanced',
               label_key: str = 'label') -> List[Dict[str, Any]]:
        """
        Sample few-shot data from dataset.

        Args:
            dataset: List of data samples
            num_shots: Number of shots per class (for balanced) or total (for random)
            strategy: 'balanced', 'stratified', or 'random'
            label_key: Key to access label in sample dict

        Returns:
            List of sampled data
        """
        if strategy == 'balanced':
            return self._balanced_sample(dataset, num_shots, label_key)
        elif strategy == 'stratified':
            return self._stratified_sample(dataset, num_shots, label_key)
        elif strategy == 'random':
            return self._random_sample(dataset, num_shots)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _balanced_sample(self,
                         dataset: List[Dict[str, Any]],
                         num_shots: int,
                         label_key: str) -> List[Dict[str, Any]]:
        """
        Balanced sampling: equal samples per class.

        Args:
            dataset: Full dataset
            num_shots: Number of samples per class
            label_key: Key for label

        Returns:
            Balanced few-shot dataset
        """
        # Group by class
        class_samples = defaultdict(list)
        for sample in dataset:
            label = sample[label_key]
            class_samples[label].append(sample)

        # Sample from each class
        few_shot_data = []
        for label, samples in class_samples.items():
            if len(samples) >= num_shots:
                selected = random.sample(samples, num_shots)
            else:
                # If not enough samples, sample with replacement
                selected = random.choices(samples, k=num_shots)
            few_shot_data.extend(selected)

        random.shuffle(few_shot_data)
        return few_shot_data

    def _stratified_sample(self,
                           dataset: List[Dict[str, Any]],
                           total_shots: int,
                           label_key: str) -> List[Dict[str, Any]]:
        """
        Stratified sampling: proportional to class distribution.

        Args:
            dataset: Full dataset
            total_shots: Total number of samples
            label_key: Key for label

        Returns:
            Stratified few-shot dataset
        """
        # Count class distribution
        class_samples = defaultdict(list)
        for sample in dataset:
            label = sample[label_key]
            class_samples[label].append(sample)

        num_classes = len(class_samples)
        total_samples = len(dataset)

        # Calculate shots per class proportionally
        few_shot_data = []
        for label, samples in class_samples.items():
            # Proportion of this class
            proportion = len(samples) / total_samples
            # Number of shots for this class
            class_shots = max(1, int(total_shots * proportion))

            if len(samples) >= class_shots:
                selected = random.sample(samples, class_shots)
            else:
                selected = random.choices(samples, k=class_shots)
            few_shot_data.extend(selected)

        random.shuffle(few_shot_data)
        return few_shot_data

    def _random_sample(self,
                       dataset: List[Dict[str, Any]],
                       num_shots: int) -> List[Dict[str, Any]]:
        """
        Random sampling: completely random selection.

        Args:
            dataset: Full dataset
            num_shots: Number of samples to select

        Returns:
            Random few-shot dataset
        """
        if len(dataset) >= num_shots:
            return random.sample(dataset, num_shots)
        else:
            return random.choices(dataset, k=num_shots)

    def get_class_distribution(self,
                               dataset: List[Dict[str, Any]],
                               label_key: str = 'label') -> Dict[Any, int]:
        """
        Get class distribution statistics.

        Args:
            dataset: Dataset
            label_key: Key for label

        Returns:
            Dict mapping label to count
        """
        distribution = defaultdict(int)
        for sample in dataset:
            label = sample[label_key]
            distribution[label] += 1
        return dict(distribution)
