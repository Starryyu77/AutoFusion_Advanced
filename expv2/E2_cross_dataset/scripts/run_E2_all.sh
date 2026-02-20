#!/bin/bash
# E2 Cross-Dataset Experiment Batch Runner
# 在NTU GPU集群上并行运行3个数据集的实验

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPV2_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$EXPV2_DIR"

# 创建结果目录
mkdir -p E2_cross_dataset/results

echo "=========================================="
echo "E2: Cross-Dataset Experiment Launcher"
echo "=========================================="
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: Python未找到"
    exit 1
fi

echo "Python版本: $(python --version)"
echo ""

# 先测试列出架构
echo "测试架构加载..."
python E2_cross_dataset/scripts/run_E2.py --list-arch
echo ""

# 数据集和GPU映射
declare -A DATASET_GPU
declare -A DATASET_LOG

DATASET_GPU["mmmu"]=0
DATASET_GPU["vsr"]=1
DATASET_GPU["mathvista"]=2

# 日志文件
LOG_DIR="E2_cross_dataset/results"
mkdir -p "$LOG_DIR"

# 为每个数据集启动一个后台进程
echo "启动并行实验..."
echo ""

for dataset in mmmu vsr mathvista; do
    gpu="${DATASET_GPU[$dataset]}"
    log_file="$LOG_DIR/${dataset}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$dataset] -> GPU $gpu -> $log_file"

    # 使用nohup在后台运行
    nohup python E2_cross_dataset/scripts/run_E2.py \
        --dataset "$dataset" \
        --num-runs 3 \
        --gpu "$gpu" \
        > "$log_file" 2>&1 &

    pid=$!
    echo "  PID: $pid"

    # 保存PID
    echo $pid > "$LOG_DIR/${dataset}.pid"

    sleep 3
done

echo ""
echo "=========================================="
echo "所有数据集已启动!"
echo "=========================================="
echo ""
echo "监控命令:"
echo "  查看日志:  tail -f E2_cross_dataset/results/*.log"
echo "  查看GPU:   nvidia-smi"
echo "  查看进程:  ps aux | grep run_E2"
echo "  检查进度:  python E2_cross_dataset/scripts/check_E2_progress.py"
echo ""
echo "停止命令:"
echo "  停止所有:  kill $(cat E2_cross_dataset/results/*.pid)"
echo ""
