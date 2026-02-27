#!/bin/bash
# GDPO/GRPO Fixed Experiments
set -e

echo "=========================================="
echo "GDPO/GRPO Fixed Experiments"
echo "=========================================="
echo ""

SCRIPT_DIR='/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase2_controllers'
cd $SCRIPT_DIR

SEEDS=(42 123 456 789 1024)

run_experiment() {
    local controller=$1
    local seed=$2
    local gpu=$3

    echo "[GPU $gpu] Starting $controller (seed=$seed)..."

    export PYTHONPATH='/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment'

    CUDA_VISIBLE_DEVICES=$gpu python3 run_experiment_phase21.py $controller $seed $gpu > logs_phase21/${controller}_s${seed}.log 2>&1

    echo "[GPU $gpu] Finished $controller (seed=$seed)"
}

# GPU 2: GDPO
run_gpu2() {
    for seed in 42 123 456 789 1024; do
        run_experiment gdpo $seed 2
    done
    for seed in 42 123 456; do
        run_experiment grpo $seed 2
    done
}

# GPU 3: GRPO remaining
run_gpu3() {
    for seed in 789 1024; do
        run_experiment grpo $seed 3
    done
}

echo "Launching experiments..."
run_gpu2 &
PID_GPU2=$!
run_gpu3 &
PID_GPU3=$!

echo "GPU2 PID: $PID_GPU2, GPU3 PID: $PID_GPU3"
echo ""

wait $PID_GPU2 $PID_GPU3

echo ""
echo "=========================================="
echo "GDPO/GRPO Fixed Experiments Complete!"
echo "=========================================="
