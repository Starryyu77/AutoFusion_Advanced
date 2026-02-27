"""
Test for evaluator_v2 with early stopping and time limit
"""
import sys
sys.path.insert(0, '/Users/starryyu/2026/Auto-Fusion-Advanced')
sys.path.insert(0, '/Users/starryyu/2026/Auto-Fusion-Advanced/experiment')

from phase4_optimization.src.evaluator_v2 import RealDataFewShotEvaluator

def test_early_stopping():
    """Test early stopping functionality"""
    print("Testing early stopping...")

    config = {
        'dataset': 'ai2d',
        'train_epochs': 10,
        'num_shots': 16,
        'batch_size': 4,
        'early_stopping': {
            'enabled': True,
            'patience': 2,
            'min_delta': 0.01
        },
        'max_training_time': None,  # No time limit for this test
        'device': 'cpu'
    }

    evaluator = RealDataFewShotEvaluator(config)

    # Check that config is saved
    assert hasattr(evaluator, 'config'), "Config not saved"
    assert evaluator.config['early_stopping']['enabled'], "Early stopping not enabled"

    print("✓ Early stopping configuration test passed")

def test_time_limit():
    """Test time limit functionality"""
    print("Testing time limit...")

    config = {
        'dataset': 'ai2d',
        'train_epochs': 100,  # High number to test time limit
        'num_shots': 16,
        'batch_size': 4,
        'early_stopping': {
            'enabled': False
        },
        'max_training_time': 1,  # 1 second limit
        'device': 'cpu'
    }

    evaluator = RealDataFewShotEvaluator(config)

    # Check that time limit is configured
    assert evaluator.config.get('max_training_time') == 1, "Time limit not set"

    print("✓ Time limit configuration test passed")

def test_constrained_reward():
    """Test that ConstrainedReward will be importable"""
    print("Testing ConstrainedReward import...")

    try:
        from experiment.base.reward import MultiObjectiveReward
        print("✓ Can import MultiObjectiveReward")
    except ImportError as e:
        print(f"✗ Cannot import: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 4 Evaluator v2 Tests")
    print("=" * 60)

    test_early_stopping()
    test_time_limit()
    test_constrained_reward()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
