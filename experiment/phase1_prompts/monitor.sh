#!/bin/bash
# Monitor Phase 1 experiment status

LOG_FILE="/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase1_prompts/logs/phase1_gpu2_20260213_153324.log"
RESULTS_DIR="/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase1_prompts/results"

echo "=========================================="
echo "Phase 1 Experiment Monitor"
echo "=========================================="
echo ""

# Check process
echo "Process Status:"
ps aux | grep run_phase1 | grep -v grep
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi | grep -E "GPU|A5000"
echo ""

# Check latest log
echo "Latest Log Entries:"
tail -20 "$LOG_FILE" 2>/dev/null || echo "No log file yet"
echo ""

# Check results
echo "Results Directory:"
ls -la "$RESULTS_DIR" 2>/dev/null || echo "No results yet"
echo ""

echo "=========================================="
echo "To monitor continuously: watch -n 10 bash monitor.sh"
echo "=========================================="
