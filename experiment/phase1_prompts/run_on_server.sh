#!/bin/bash
# Phase 1 Prompt Comparison - Server Run Script
# For NTU GPU43 server

set -e

echo "=========================================="
echo "Phase 1: Prompt Strategy Comparison"
echo "Running on NTU GPU43"
echo "=========================================="

# Default configuration
GPU_ID=${GPU_ID:-3}  # Default to GPU 3 (usually less busy)
ITERATIONS=${ITERATIONS:-20}
RUN_NAME=${RUN_NAME:-"phase1_$(date +%Y%m%d_%H%M%S)"}

# Project paths
PROJECT_DIR="/usr1/home/s125mdg43_10/AutoFusion_Advanced"
EXPERIMENT_DIR="$PROJECT_DIR/experiment/phase1_prompts"
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$EXPERIMENT_DIR/results"
LOGS_DIR="$EXPERIMENT_DIR/logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# Environment setup
echo "Setting up environment..."
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Check for API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "Warning: DEEPSEEK_API_KEY not set!"
    echo "Please set it with: export DEEPSEEK_API_KEY='your-key'"
    exit 1
fi

echo "Configuration:"
echo "  GPU: $GPU_ID"
echo "  Iterations: $ITERATIONS"
echo "  Run Name: $RUN_NAME"
echo "  Output: $OUTPUT_DIR/$RUN_NAME"
echo ""

# Navigate to experiment directory
cd "$EXPERIMENT_DIR"

# Run the experiment
LOG_FILE="$LOGS_DIR/${RUN_NAME}.log"
echo "Starting experiment... (logging to $LOG_FILE)"
echo "Use: tail -f $LOG_FILE to monitor progress"
echo ""

python run_phase1.py \
    --run-name "$RUN_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --gpu "$GPU_ID" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "Results: $OUTPUT_DIR/$RUN_NAME"
echo "Log: $LOG_FILE"
echo "=========================================="
