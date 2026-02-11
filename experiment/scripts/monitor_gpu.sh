#!/bin/bash
# GPU Memory Monitor
# Real-time GPU memory monitoring with alerts

THRESHOLD_MB=${1:-22000}  # Default: 22GB (24GB * 0.9)
INTERVAL=${2:-10}         # Default: 10 seconds

echo "========================================"
echo "GPU Memory Monitor"
echo "========================================"
echo "Threshold: ${THRESHOLD_MB}MB"
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    for GPU in 2 3; do
        # Check if GPU exists
        if ! nvidia-smi --id=$GPU > /dev/null 2>&1; then
            continue
        fi

        # Get memory info
        USED=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits)
        TOTAL=$(nvidia-smi --id=$GPU --query-gpu=memory.total --format=csv,noheader,nounits)
        UTIL=$(nvidia-smi --id=$GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits)

        # Calculate percentage
        PERCENT=$((USED * 100 / TOTAL))

        # Status indicator
        if [ "$USED" -gt "$THRESHOLD_MB" ]; then
            STATUS="⚠️  WARNING"
        else
            STATUS="✓ OK"
        fi

        # Print status
        echo "[$(date '+%H:%M:%S')] GPU$GPU: ${USED}MB / ${TOTAL}MB (${PERCENT}%) | Util: ${UTIL}% | $STATUS"
    done

    echo "----------------------------------------"
    sleep $INTERVAL
done
