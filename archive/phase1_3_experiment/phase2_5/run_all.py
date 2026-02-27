#!/usr/bin/env python3
"""
Run all Phase 2.5 experiments
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()

def main():
    print("=" * 60)
    print("Phase 2.5: Complete Evaluator Verification")
    print("=" * 60)

    # Run 2.5.1
    print("\n[1/3] Running Dataset Selection...")
    import run_2_5_1_dataset_selection
    run_2_5_1_dataset_selection.run_experiment()

    # Run 2.5.2
    print("\n[2/3] Running Training Depth Calibration...")
    import run_2_5_2_training_depth
    run_2_5_2_training_depth.run_experiment()

    # Run 2.5.3
    print("\n[3/3] Running Architecture Fairness...")
    import run_2_5_3_architecture_fairness
    run_2_5_3_architecture_fairness.run_experiment()

    print("\n" + "=" * 60)
    print("Phase 2.5 Complete!")
    print("=" * 60)
    print("\nResults saved in:")
    print("  - results/2_5_1_dataset_selection/")
    print("  - results/2_5_2_training_depth/")
    print("  - results/2_5_3_architecture_fairness/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
