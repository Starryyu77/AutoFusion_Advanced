#!/bin/bash
#SBATCH --job-name=phase4_nas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=phase4_%j.out
#SBATCH --error=phase4_%j.err

# Phase 4: Optimized Architecture Search
# Run on NTU EEE GPU Cluster

echo "=========================================="
echo "Phase 4 Architecture Search"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module load cuda/11.8
module load python/3.10

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate autofusion

# Navigate to project directory
cd /projects/tianyu016/AutoFusion_Advanced || exit 1

echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# Run Phase 4 search
echo ""
echo "Starting Phase 4 architecture search..."
echo "=========================================="

python phase4_optimization/run_phase4_search.py \
    --output-dir ./phase4_optimization/results/discovery \
    --num-iterations 200 \
    --dataset mmmu \
    --num-shots 32 \
    --train-epochs 10 \
    --batch-size 8 \
    --max-training-time 300 \
    --weight-efficiency 1.5 \
    --max-flops 10000000 \
    --population-size 50 \
    --save-interval 50 \
    --log-level INFO

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
