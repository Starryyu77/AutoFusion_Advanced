#!/usr/bin/env python3
"""
Download datasets for RealDataFewShotEvaluator.

Usage:
    python scripts/download_datasets.py [--dataset DATASET] [--data_dir DATA_DIR]

Examples:
    # Download all datasets
    python scripts/download_datasets.py

    # Download specific dataset
    python scripts/download_datasets.py --dataset mmmu

    # Download to specific directory
    python scripts/download_datasets.py --data_dir /path/to/data
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset configurations
DATASETS = {
    'mmmu': {
        'name': 'MMMU/MMMU',
        'split': 'validation',
        'description': 'College-level multi-discipline multimodal understanding',
        'size_gb': '~10GB'
    },
    'vsr': {
        'name': 'cambridgeltl/vsr_random',
        'split': 'train',
        'description': 'Visual spatial reasoning',
        'size_gb': '~2GB'
    },
    'mathvista': {
        'name': 'AI4Math/MathVista',
        'split': 'test',
        'description': 'Visual mathematical reasoning',
        'size_gb': '~5GB'
    },
    'ai2d': {
        'name': 'lmms-lab/AI2D',
        'split': 'test',
        'description': 'Diagram understanding',
        'size_gb': '~3GB'
    }
}


def download_dataset(dataset_name: str, data_dir: Path, cache_dir: Path) -> bool:
    """
    Download a single dataset.

    Args:
        dataset_name: Name of dataset (mmmu, vsr, mathvista, ai2d)
        data_dir: Directory to save data
        cache_dir: HuggingFace cache directory

    Returns:
        True if successful, False otherwise
    """
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False

    config = DATASETS[dataset_name]
    dataset_path = config['name']
    split = config['split']

    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading {dataset_name.upper()}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Split: {split}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Estimated size: {config['size_gb']}")
    logger.info(f"{'='*60}\n")

    try:
        # Import datasets library
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not found. Installing...")
            os.system(f"{sys.executable} -m pip install datasets -q")
            from datasets import load_dataset

        # Download dataset
        logger.info(f"Loading {dataset_name} from HuggingFace...")
        logger.info("This may take a while depending on your internet connection...")

        dataset = load_dataset(
            dataset_path,
            split=split,
            cache_dir=str(cache_dir),
            trust_remote_code=True  # Required for some datasets
        )

        logger.info(f"Successfully loaded {dataset_name}")
        logger.info(f"Number of samples: {len(dataset)}")
        logger.info(f"Features: {list(dataset.features.keys())}")

        # Save a small sample for verification
        sample_file = data_dir / dataset_name / 'sample.txt'
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_file, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Path: {dataset_path}\n")
            f.write(f"Split: {split}\n")
            f.write(f"Samples: {len(dataset)}\n")
            f.write(f"Features: {list(dataset.features.keys())}\n")

        logger.info(f"Sample info saved to {sample_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        return False


def download_all_datasets(data_dir: Path, cache_dir: Path) -> Dict[str, bool]:
    """
    Download all datasets.

    Args:
        data_dir: Directory to save data
        cache_dir: HuggingFace cache directory

    Returns:
        Dict mapping dataset name to success status
    """
    results = {}

    logger.info("\n" + "="*60)
    logger.info("Starting download of all datasets")
    logger.info("="*60)

    for dataset_name in DATASETS.keys():
        success = download_dataset(dataset_name, data_dir, cache_dir)
        results[dataset_name] = success

    return results


def print_summary(results: Dict[str, bool], data_dir: Path, cache_dir: Path):
    """Print download summary."""
    logger.info("\n" + "="*60)
    logger.info("Download Summary")
    logger.info("="*60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for dataset_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{dataset_name:12s}: {status}")

    logger.info(f"\nTotal: {success_count}/{total_count} datasets downloaded successfully")

    if success_count < total_count:
        logger.info("\nFailed downloads can be retried later.")

    logger.info(f"\nData directory: {data_dir}")
    logger.info(f"Cache directory: {cache_dir}")

    # Print next steps
    logger.info("\n" + "="*60)
    logger.info("Next Steps")
    logger.info("="*60)
    logger.info("1. Verify datasets are correctly downloaded:")
    logger.info(f"   ls {data_dir}")
    logger.info("\n2. Run evaluator test:")
    logger.info("   python -c \"from experiment.evaluators.real_data_evaluator import RealDataFewShotEvaluator; print('OK')\"")
    logger.info("\n3. Start Phase 2.5 evaluation:")
    logger.info("   python experiment/phase2_5/run_evaluator_verification.py")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for RealDataFewShotEvaluator'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASETS.keys()) + ['all'],
        default='all',
        help='Dataset to download (default: all)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory to save datasets (default: ./data)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='HuggingFace cache directory (default: data_dir/.cache)'
    )

    args = parser.parse_args()

    # Setup directories
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
    else:
        cache_dir = data_dir / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Cache directory: {cache_dir}")

    # Download datasets
    if args.dataset == 'all':
        results = download_all_datasets(data_dir, cache_dir)
    else:
        success = download_dataset(args.dataset, data_dir, cache_dir)
        results = {args.dataset: success}

    # Print summary
    print_summary(results, data_dir, cache_dir)

    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
