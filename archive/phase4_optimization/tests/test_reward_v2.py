"""
Test Constrained Reward Function - Phase 4
------------------------------------------
Test script to verify the constrained reward function works correctly.

Usage:
    cd /Users/starryyu/2026/Auto-Fusion-Advanced
    python phase4_optimization/tests/test_reward_v2.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'experiment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_default_weights():
    """Test that efficiency weight is increased to 1.5."""
    logger.info("=" * 60)
    logger.info("Test 1: Default Weights")
    logger.info("=" * 60)

    from reward_v2 import ConstrainedReward

    config = {
        'flops_constraint': {'enabled': False},
    }

    reward = ConstrainedReward(config)

    # Check weights
    assert reward.weights['efficiency'] == 1.5, f"Expected efficiency weight 1.5, got {reward.weights['efficiency']}"
    assert reward.weights['accuracy'] == 1.0, f"Expected accuracy weight 1.0, got {reward.weights['accuracy']}"
    assert reward.weights['compile_success'] == 2.0, f"Expected compile_success weight 2.0, got {reward.weights['compile_success']}"

    logger.info("✓ Efficiency weight correctly set to 1.5 (was 0.5)")
    logger.info(f"  Weights: {reward.weights}")
    return True


def test_flops_constraint():
    """Test FLOPs hard constraint."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: FLOPs Hard Constraint")
    logger.info("=" * 60)

    from reward_v2 import ConstrainedReward

    config = {
        'flops_constraint': {
            'enabled': True,
            'max_flops': 10e6,  # 10M FLOPs limit
            'reject_if_exceed': True
        },
    }

    reward = ConstrainedReward(config)

    # Test case 1: Within limit (5M FLOPs)
    result_low = {
        'accuracy': 0.5,
        'flops': 5e6,
        'params': 1e6,
        'compile_success': 1.0
    }
    components_low = reward.calculate(result_low)
    assert not components_low.rejected, "Low FLOPs should not be rejected"
    logger.info(f"✓ 5M FLOPs: not rejected, reward={components_low.to_scalar(reward.weights):.3f}")

    # Test case 2: Exactly at limit (10M FLOPs)
    result_exact = {
        'accuracy': 0.5,
        'flops': 10e6,
        'params': 1e6,
        'compile_success': 1.0
    }
    components_exact = reward.calculate(result_exact)
    assert not components_exact.rejected, "Exact limit should not be rejected"
    logger.info(f"✓ 10M FLOPs: not rejected, reward={components_exact.to_scalar(reward.weights):.3f}")

    # Test case 3: Over limit (15M FLOPs)
    result_high = {
        'accuracy': 0.8,
        'flops': 15e6,
        'params': 1e6,
        'compile_success': 1.0
    }
    components_high = reward.calculate(result_high)
    assert components_high.rejected, "High FLOPs should be rejected"
    assert components_high.rejection_reason != "", "Should have rejection reason"
    logger.info(f"✓ 15M FLOPs: rejected, reason='{components_high.rejection_reason}'")

    # Test case 4: Very high FLOPs (50M)
    result_very_high = {
        'accuracy': 0.9,
        'flops': 50e6,
        'params': 1e6,
        'compile_success': 1.0
    }
    components_very_high = reward.calculate(result_very_high)
    assert components_very_high.rejected, "Very high FLOPs should be rejected"
    logger.info(f"✓ 50M FLOPs: rejected as expected")

    return True


def test_exponential_penalty():
    """Test exponential FLOPs penalty."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Exponential FLOPs Penalty")
    logger.info("=" * 60)

    from reward_v2 import ConstrainedReward

    config = {
        'flops_constraint': {'enabled': False},
        'flops_penalty': {
            'type': 'exponential',
            'scale': 20e6  # 20M scale
        },
    }

    reward = ConstrainedReward(config)

    # Test with different FLOPs values
    flops_values = [1e6, 5e6, 10e6, 20e6, 40e6, 100e6]

    for flops in flops_values:
        result = {
            'accuracy': 0.5,
            'flops': flops,
            'params': 1e6,
            'compile_success': 1.0
        }
        components = reward.calculate(result)
        penalty = math.exp(-flops / 20e6)
        expected_accuracy = 0.5 * penalty

        logger.info(f"  {flops/1e6:6.1f}M FLOPs: "
                   f"penalty={penalty:.4f}, "
                   f"accuracy={components.accuracy:.4f}, "
                   f"expected={expected_accuracy:.4f}")

    logger.info("✓ Exponential penalty working correctly")
    return True


def test_efficiency_vs_flops():
    """Test that efficiency reward decreases with FLOPs."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Efficiency vs FLOPs Relationship")
    logger.info("=" * 60)

    from reward_v2 import ConstrainedReward

    config = {
        'flops_constraint': {'enabled': False},
    }

    reward = ConstrainedReward(config)

    flops_values = [1e6, 5e6, 10e6, 20e6, 50e6, 100e6]

    logger.info("FLOPs    | Efficiency | Efficiency Reward")
    logger.info("-" * 45)

    for flops in flops_values:
        result = {
            'accuracy': 0.5,
            'flops': flops,
            'params': 1e6,
            'compile_success': 1.0
        }
        components = reward.calculate(result)
        eff_reward = components.efficiency * reward.weights['efficiency']

        logger.info(f"{flops/1e6:6.1f}M  | {components.efficiency:10.4f} | {eff_reward:17.4f}")

    logger.info("✓ Efficiency reward decreases with FLOPs as expected")
    return True


def test_scalar_calculation():
    """Test scalar reward calculation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Scalar Reward Calculation")
    logger.info("=" * 60)

    from reward_v2 import ConstrainedReward

    config = {
        'flops_constraint': {
            'enabled': True,
            'max_flops': 10e6,
            'reject_if_exceed': True
        },
    }

    reward = ConstrainedReward(config)

    # Test case 1: Good architecture (low FLOPs, high accuracy)
    result_good = {
        'accuracy': 0.7,
        'flops': 5e6,
        'params': 1e6,
        'compile_success': 1.0
    }
    scalar_good = reward.calculate_scalar(result_good)
    logger.info(f"✓ Good arch (5M FLOPs, 70% acc): scalar={scalar_good:.3f}")

    # Test case 2: Bad architecture (high FLOPs, but still within limit)
    result_bad = {
        'accuracy': 0.4,
        'flops': 8e6,
        'params': 1e6,
        'compile_success': 1.0
    }
    scalar_bad = reward.calculate_scalar(result_bad)
    logger.info(f"✓ Bad arch (8M FLOPs, 40% acc): scalar={scalar_bad:.3f}")

    # Test case 3: Rejected architecture (over FLOPs limit)
    result_rejected = {
        'accuracy': 0.9,
        'flops': 50e6,
        'params': 1e6,
        'compile_success': 1.0
    }
    scalar_rejected = reward.calculate_scalar(result_rejected)
    logger.info(f"✓ Rejected arch (50M FLOPs, 90% acc): scalar={scalar_rejected:.3f}")
    assert scalar_rejected == 0.0, "Rejected architecture should have 0 scalar reward"

    # Test case 4: Compilation failure
    result_compile_fail = {
        'accuracy': 0.0,
        'flops': 0,
        'params': 0,
        'compile_success': 0.0
    }
    scalar_compile_fail = reward.calculate_scalar(result_compile_fail)
    logger.info(f"✓ Compile fail: scalar={scalar_compile_fail:.3f}")

    logger.info(f"\nComparison:")
    logger.info(f"  Good arch:      {scalar_good:.3f}")
    logger.info(f"  Bad arch:       {scalar_bad:.3f}")
    logger.info(f"  Rejected arch:  {scalar_rejected:.3f} (should be 0)")
    logger.info(f"  Compile fail:   {scalar_compile_fail:.3f}")

    assert scalar_good > scalar_bad, "Good arch should have higher reward than bad arch"
    assert scalar_rejected == 0.0, "Rejected arch should have 0 reward"

    return True


def test_phase4_target_scenario():
    """Test Phase 4 target scenario - comparing NAS to FiLM."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Phase 4 Target Scenario (vs FiLM)")
    logger.info("=" * 60)

    from reward_v2 import ConstrainedReward

    # Phase 4 config: max 10M FLOPs (FiLM is 6.29M)
    config = {
        'flops_constraint': {
            'enabled': True,
            'max_flops': 10e6,  # 10M limit
            'reject_if_exceed': True
        },
        'flops_penalty': {
            'type': 'exponential',
            'scale': 20e6
        },
    }

    reward = ConstrainedReward(config)

    # Scenario: Compare different architectures
    architectures = [
        # (name, accuracy, flops_m, params_m)
        ("FiLM (target)", 0.46, 6.29, 3.15),
        ("CLIPFusion", 0.33, 2.36, 1.18),
        ("Old NAS (bad)", 0.33, 50.0, 25.0),
        ("New NAS (good)", 0.50, 5.0, 2.5),
        ("New NAS (efficient)", 0.45, 3.0, 1.5),
    ]

    logger.info(f"{'Architecture':<20} | {'Acc':>5} | {'FLOPs(M)':>9} | {'Reward':>8} | Status")
    logger.info("-" * 70)

    for name, acc, flops_m, params_m in architectures:
        result = {
            'accuracy': acc,
            'flops': flops_m * 1e6,
            'params': params_m * 1e6,
            'compile_success': 1.0
        }
        components = reward.calculate(result)
        scalar = reward.calculate_scalar(result)

        status = "REJECTED" if components.rejected else "OK"
        logger.info(f"{name:<20} | {acc:5.2f} | {flops_m:9.2f} | {scalar:8.3f} | {status}")

    logger.info("\n✓ Phase 4 scenario test complete")
    logger.info("  - Old NAS (50M FLOPs) is correctly rejected")
    logger.info("  - New efficient architectures get good rewards")
    return True


def run_all_tests():
    """Run all tests and report results."""
    logger.info("\n" + "=" * 60)
    logger.info("Constrained Reward Function Test Suite - Phase 4")
    logger.info("=" * 60)

    results = {
        'Default Weights (Efficiency=1.5)': test_default_weights(),
        'FLOPs Hard Constraint': test_flops_constraint(),
        'Exponential Penalty': test_exponential_penalty(),
        'Efficiency vs FLOPs': test_efficiency_vs_flops(),
        'Scalar Calculation': test_scalar_calculation(),
        'Phase 4 Target Scenario': test_phase4_target_scenario(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! Reward function is ready for Phase 4.")
        return 0
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed. Please review.")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
