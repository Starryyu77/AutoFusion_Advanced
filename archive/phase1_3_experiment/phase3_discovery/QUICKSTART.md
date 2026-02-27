# Phase 3: Architecture Discovery - Quick Start Guide

## Overview

Use the winning combination from Phases 1 & 2 to discover novel multimodal fusion architectures.

**Winning Configuration:**
- Controller: Evolution (Phase 2.1 Winner)
- Generator: FewShot (Phase 1 Winner, Best Reward 0.873)
- Evaluator: RealDataFewShotEvaluator (AI2D, 3 epochs)

---

## Quick Start

### Local Testing

```bash
cd experiment/phase3_discovery

# Quick test (10 iterations)
python run_phase3.py --run-name test_run --iterations 10

# Standard discovery (100 iterations)
python run_phase3.py --run-name discovery_v1 --iterations 100

# Extended search (200 iterations, larger population)
python run_phase3.py --run-name discovery_deep --iterations 200 --population 100
```

### On NTU GPU43 Server

```bash
# Login
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg

# Navigate to project
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase3_discovery

# Set API key
export DEEPSEEK_API_KEY="your-api-key"

# Run discovery
bash run_on_server.sh

# Or with custom parameters
GPU_ID=2 ITERATIONS=100 POPULATION=50 RUN_NAME=my_discovery bash run_on_server.sh
```

---

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--run-name` | auto-generated | Name for this discovery run |
| `--iterations` | 100 | Number of search iterations |
| `--population` | 50 | Population size for Evolution |
| `--threshold` | 0.75 | Reward threshold for saving architectures |
| `--output-dir` | results | Output directory |
| `--gpu` | None | GPU device ID |

---

## Output Structure

```
results/discovery_20260213_120000/
├── top_architectures/          # Saved architectures (reward > threshold)
│   ├── arch_001/
│   │   ├── code.py            # Architecture code
│   │   ├── config.json        # Architecture description
│   │   └── results.json       # Evaluation results
│   ├── arch_002/
│   └── ...
├── results/
│   ├── checkpoint_iter_020.json
│   ├── checkpoint_iter_040.json
│   └── discovery_report.md    # Final report
└── logs/
    └── discovery_20260213_120000.log
```

---

## Monitoring Progress

```bash
# View real-time log
tail -f results/discovery_*/logs/*.log

# Check number of discovered architectures
ls results/discovery_*/top_architectures/ | wc -l

# View top architectures
ls results/discovery_*/top_architectures/arch_*/results.json
```

---

## Expected Results

- **Duration:** ~8-10 hours for 100 iterations
- **API Calls:** 100 (one per iteration)
- **Cost:** ~¥50-80 (depends on prompt length)
- **Top Architectures:** 10-20 (with threshold 0.75)

---

## Next Steps After Discovery

1. **Analyze Top Architectures:**
   ```bash
   python analysis.py --results-dir results/discovery_*/
   ```

2. **Full Evaluation:**
   Evaluate top 10 architectures with 100 epochs on all datasets

3. **Ablation Studies:**
   Study contribution of different components

---

## Tips

- **Lower threshold** (e.g., 0.70) to save more architectures
- **Higher threshold** (e.g., 0.80) to save only the best
- Use **GPU 2 or 3** on GPU43 (usually less busy)
- Run **multiple seeds** for robustness (vary run-name)

---

*For detailed documentation, see PLAN.md*
