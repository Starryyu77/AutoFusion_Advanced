#!/bin/bash
# Phase 4 with LLM - Run Script for GPU Cluster
# Usage: bash run_llm_search.sh [gpu_id] [num_iterations]

GPU_ID=${1:-0}
NUM_ITERATIONS=${2:-50}

echo "========================================"
echo "Phase 4 with LLM: Architecture Search"
echo "========================================"
echo "GPU: $GPU_ID"
echo "Iterations: $NUM_ITERATIONS"
echo "Output: ./phase4_optimization/results_llm/"
echo "========================================"

# Set environment
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH=/usr1/home/s125mdg43_10/AutoFusion_Advanced:$PYTHONPATH

# Check API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY not set!"
    echo "Please run: export DEEPSEEK_API_KEY='your-api-key'"
    exit 1
fi

# Run search
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced

python phase4_optimization/run_phase4_llm.py \
    --output-dir ./phase4_optimization/results_llm \
    --num-iterations $NUM_ITERATIONS \
    --dataset mmmu \
    --num-shots 32 \
    --train-epochs 10 \
    --batch-size 8 \
    --max-training-time 300 \
    --weight-efficiency 1.5 \
    --max-flops 10000000 \
    --temperature 0.7 \
    --num-examples 3 \
    --save-interval 10 \
    2>&1 | tee ./phase4_optimization/results_llm/search.log

echo "========================================"
echo "Search completed!"
echo "Results saved to: ./phase4_optimization/results_llm/"
echo "========================================"