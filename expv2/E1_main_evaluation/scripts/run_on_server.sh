#!/bin/bash
# E1: AI2D主实验 - 服务器运行脚本

GPU_ID=${1:-2}
EXP_NAME="E1_main_evaluation"
RESULTS_DIR="/usr1/home/s125mdg43_10/AutoFusion_Advanced/expv2/${EXP_NAME}/results"

echo "=========================================="
echo "E1: AI2D主实验"
echo "GPU: $GPU_ID"
echo "开始时间: $(date)"
echo "=========================================="

# 创建结果目录
mkdir -p $RESULTS_DIR

# 找到conda并初始化
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_ENV="anaconda3"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_ENV="miniconda3"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    CONDA_ENV="opt/conda"
else
    echo "警告: 未找到conda初始化脚本，尝试直接激活"
fi

# 激活环境
conda activate autofusion 2>/dev/null || source activate autofusion 2>/dev/null || echo "使用默认Python环境"

# 验证Python
which python3 || which python || (echo "错误: 未找到Python" && exit 1)
PYTHON=$(which python3 || which python)
echo "使用Python: $PYTHON"

# 设置路径
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced/expv2
export PYTHONPATH=/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment:$PYTHONPATH

# 1. 快速测试 (验证pipeline)
echo ""
echo "[1/3] E1快速测试 (10 epochs)..."
$PYTHON E1_main_evaluation/scripts/run_E1.py \
    --mode quick \
    --arch-type all \
    --gpu $GPU_ID \
    2>&1 | tee $RESULTS_DIR/quick_test.log

# 2. 完整评估 (基线)
echo ""
echo "[2/3] E1基线完整评估 (100 epochs, 3 runs)..."
$PYTHON E1_main_evaluation/scripts/run_E1.py \
    --mode full \
    --arch-type baseline \
    --num-runs 3 \
    --gpu $GPU_ID \
    2>&1 | tee $RESULTS_DIR/baseline_full.log

# 3. 完整评估 (发现架构)
echo ""
echo "[3/3] E1发现架构完整评估 (100 epochs, 3 runs)..."
$PYTHON E1_main_evaluation/scripts/run_E1.py \
    --mode full \
    --arch-type discovered \
    --num-runs 3 \
    --gpu $GPU_ID \
    2>&1 | tee $RESULTS_DIR/discovered_full.log

echo ""
echo "=========================================="
echo "E1 实验完成!"
echo "结束时间: $(date)"
echo "结果目录: $RESULTS_DIR"
echo "=========================================="
