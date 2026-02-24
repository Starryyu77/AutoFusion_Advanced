#!/usr/bin/env python3
"""
AutoFusion 统一实验入口

用法:
    # 运行Phase 1 (Prompt对比)
    python run.py --experiment phase1 --config configs/phase1.yaml

    # 运行Phase 3 (架构发现)
    python run.py --experiment phase3 --config configs/phase3.yaml --gpu 0

    # 运行E1 (完整评估)
    python run.py --experiment E1 --mode full --gpu 0

    # 运行E2 (跨数据集)
    python run.py --experiment E2 --dataset mmmu --gpu 0

    # 查看状态
    python run.py --status

    # 查看结果
    python run.py --results --experiment E1
"""

import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'experiment'))
sys.path.insert(0, str(PROJECT_ROOT / 'expv2'))


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_phase1(config: dict, gpu: int = None):
    """运行Phase 1: Prompt策略对比"""
    from experiment.phase1_prompts.run_phase1 import run_experiment

    print("=" * 60)
    print("Phase 1: Prompt Strategy Comparison")
    print("=" * 60)

    results = run_experiment(
        strategies=config['strategies'],
        iterations=config.get('iterations', 20),
        gpu=gpu
    )

    # 保存结果
    output_dir = Path('results/phase1')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("Phase 1 Results:")
    print("=" * 60)
    for strategy, result in sorted(results.items(), key=lambda x: x[1]['best_reward'], reverse=True):
        print(f"  {strategy:15s}: Reward={result['best_reward']:.3f}, Valid={result['validity_rate']:.1%}")

    return results


def run_phase3(config: dict, gpu: int = None):
    """运行Phase 3: 架构发现"""
    from experiment.phase3_discovery.run_phase3 import ArchitectureDiscovery

    print("=" * 60)
    print("Phase 3: Architecture Discovery")
    print("=" * 60)

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"results/phase3/discovery_{timestamp}")

    # 初始化并运行
    discovery = ArchitectureDiscovery(config, output_dir)
    discovery.initialize_components()
    results = discovery.run_discovery(num_iterations=config.get('iterations', 100))

    print("\n" + "=" * 60)
    print(f"Phase 3 Complete. Top architectures saved to: {output_dir}")
    print("=" * 60)

    return results


def run_E1(mode: str = 'quick', gpu: int = None):
    """运行E1: 完整评估"""
    from expv2.E1_main_evaluation.scripts.run_E1 import main as run_E1_main

    print("=" * 60)
    print(f"E1: Full Evaluation ({mode} mode)")
    print("=" * 60)

    import sys
    old_argv = sys.argv
    try:
        sys.argv = ['run_E1.py', '--mode', mode]
        if gpu is not None:
            sys.argv.extend(['--gpu', str(gpu)])
        run_E1_main()
    finally:
        sys.argv = old_argv

    print("\n" + "=" * 60)
    print("E1 Complete. Results saved to: expv2/E1_main_evaluation/results/")
    print("=" * 60)


def run_E2(dataset: str = 'all', gpu: int = None):
    """运行E2: 跨数据集评估"""
    from expv2.E2_cross_dataset.scripts.run_E2 import main as run_E2_main

    print("=" * 60)
    print(f"E2: Cross-Dataset Evaluation ({dataset})")
    print("=" * 60)

    import sys
    old_argv = sys.argv
    try:
        sys.argv = ['run_E2.py', '--dataset', dataset]
        if gpu is not None:
            sys.argv.extend(['--gpu', str(gpu)])
        run_E2_main()
    finally:
        sys.argv = old_argv

    print("\n" + "=" * 60)
    print("E2 Complete. Results saved to: expv2/E2_cross_dataset/results/")
    print("=" * 60)


def show_status():
    """显示所有实验状态"""
    print("=" * 70)
    print("AutoFusion 实验状态")
    print("=" * 70)

    experiments = [
        ('Phase 0/0.5', 'API验证', '✅ 完成'),
        ('Phase 1', 'Prompt策略对比', '✅ 完成'),
        ('Phase 2.1', 'Controller对比', '✅ 完成'),
        ('Phase 2.5', '评估器验证', '✅ 完成'),
        ('Phase 3', '架构发现', '✅ 完成'),
        ('E1', 'AI2D完整评估', '✅ 完成'),
        ('E2', '跨数据集评估', '✅ 完成'),
        ('E3', '帕累托分析', '📋 未开始'),
        ('E4', '相关性分析', '📋 未开始'),
        ('E5', '消融实验', '📋 未开始'),
        ('E6', '设计模式', '📋 未开始'),
        ('E7', '统计检验', '📋 未开始'),
    ]

    for exp_id, name, status in experiments:
        print(f"  {exp_id:12s} | {name:25s} | {status}")

    print("=" * 70)

    # 检查结果文件
    print("\n结果文件状态:")
    result_paths = [
        ('Phase 1', 'results/phase1/results.json'),
        ('Phase 3', 'results/phase3'),
        ('E1', 'expv2/E1_main_evaluation/results/summary.json'),
        ('E2', 'expv2/E2_cross_dataset/results'),
    ]

    for name, path in result_paths:
        exists = Path(path).exists()
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"  {name:12s} | {path:50s} | {status}")


def show_results(experiment: str = None):
    """显示实验结果"""
    if experiment == 'E1':
        result_file = Path('expv2/E1_main_evaluation/results/summary.json')
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            print("\nE1 Results:")
            print("-" * 70)
            for arch_name, results in data.items():
                mean_acc = results.get('mean_test_acc', 0)
                flops = results.get('runs', [{}])[0].get('flops', 0)
                print(f"  {arch_name:20s}: Acc={mean_acc:.3f}, FLOPs={flops:.1f}M")
        else:
            print("E1 results not found. Run: python run.py --experiment E1")

    elif experiment == 'phase1':
        result_file = Path('results/phase1/results.json')
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            print("\nPhase 1 Results:")
            print("-" * 70)
            for strategy, result in sorted(data.items(), key=lambda x: x[1]['best_reward'], reverse=True):
                print(f"  {strategy:15s}: Reward={result['best_reward']:.3f}")
        else:
            print("Phase 1 results not found. Run: python run.py --experiment phase1 --config configs/phase1.yaml")

    else:
        print(f"Results for {experiment} not implemented yet.")


def main():
    parser = argparse.ArgumentParser(
        description='AutoFusion Unified Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1
  python run.py --experiment phase1 --config configs/phase1.yaml

  # Run Phase 3 with GPU
  python run.py --experiment phase3 --config configs/phase3.yaml --gpu 0

  # Run E1 evaluation
  python run.py --experiment E1 --mode full --gpu 0

  # Run E2 on MMMU
  python run.py --experiment E2 --dataset mmmu --gpu 0

  # Check status
  python run.py --status

  # View results
  python run.py --results --experiment E1
        """
    )

    parser.add_argument('--experiment', '-e', type=str,
                        choices=['phase1', 'phase3', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'],
                        help='Which experiment to run')

    parser.add_argument('--config', '-c', type=str,
                        help='Path to config file (for phase1/phase3)')

    parser.add_argument('--mode', '-m', type=str, default='quick',
                        choices=['quick', 'full'],
                        help='E1 mode: quick (10ep) or full (100ep)')

    parser.add_argument('--dataset', '-d', type=str, default='all',
                        choices=['all', 'ai2d', 'mmmu', 'vsr', 'mathvista'],
                        help='E2 dataset')

    parser.add_argument('--gpu', '-g', type=int,
                        help='GPU ID to use')

    parser.add_argument('--status', '-s', action='store_true',
                        help='Show experiment status')

    parser.add_argument('--results', '-r', action='store_true',
                        help='Show results for specified experiment')

    args = parser.parse_args()

    # 处理状态查询
    if args.status:
        show_status()
        return

    # 处理结果展示
    if args.results:
        if not args.experiment:
            print("Error: --results requires --experiment")
            return
        show_results(args.experiment)
        return

    # 必须指定experiment
    if not args.experiment:
        parser.print_help()
        return

    # 运行实验
    if args.experiment == 'phase1':
        if not args.config:
            print("Error: phase1 requires --config")
            return
        config = load_config(args.config)
        run_phase1(config, args.gpu)

    elif args.experiment == 'phase3':
        if not args.config:
            print("Error: phase3 requires --config")
            return
        config = load_config(args.config)
        run_phase3(config, args.gpu)

    elif args.experiment == 'E1':
        run_E1(args.mode, args.gpu)

    elif args.experiment == 'E2':
        run_E2(args.dataset, args.gpu)

    else:
        print(f"Experiment {args.experiment} not implemented yet.")


if __name__ == '__main__':
    main()
