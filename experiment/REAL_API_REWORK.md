# Real API Rework - Implementation Guide

## Status: Phase A Complete ✅

Phase A (Infrastructure) and Phase B preparation are complete. Ready for validation experiments.

---

## Phase A: Infrastructure (Complete)

### A1. DeepSeek API Client ✅
**File:** `experiment/utils/llm_client.py`

Features implemented:
- ✅ DeepSeek API wrapper using OpenAI-compatible interface
- ✅ Request caching (MD5 hash-based, JSON storage)
- ✅ Exponential backoff retry (default: 3 retries)
- ✅ Real-time cost tracking (prompt/completion tokens)
- ✅ Budget enforcement (configurable limit)
- ✅ API call logging to `.cache/llm/api_calls.log`

Usage:
```python
from experiment.utils.llm_client import create_deepseek_client

client = create_deepseek_client(
    api_key=os.environ['DEEPSEEK_API_KEY'],
    cache_dir='.cache/llm',
    budget_limit_yuan=10000.0
)

code = client.generate(prompt, architecture_hash)
client.print_stats()  # Show cost statistics
```

### A2. Modified Experiment Scripts ✅
**Files:**
- `experiment/phase2_controllers/run_real_api.sh` - Main real API runner
- `experiment/utils/__init__.py` - Export llm_client

Changes:
- ✅ Import and initialize DeepSeekClient
- ✅ Pass llm_client to create_generator() instead of None
- ✅ Environment variable support (DEEPSEEK_API_KEY, BUDGET_LIMIT_YUAN)
- ✅ Single experiment mode: `run_real_api.sh single ppo 42 2`

### A3. Checkpoint Mechanism ✅
**Status:** Already implemented in BaseController

Features:
- ✅ Auto-save every 10 iterations
- ✅ Load checkpoint if exists (for resume)
- ✅ JSON + PyTorch checkpoint formats
- ✅ API stats saved alongside checkpoint

### A4. Cost Monitoring ✅
**Integrated into:**
- `llm_client.py` - Real-time cost tracking
- `run_real_api.sh` - Budget checks and logging

Features:
- ✅ Per-call cost calculation (DeepSeek-V3 pricing)
- ✅ Cumulative cost tracking
- ✅ Budget limit enforcement
- ✅ Cache hit rate statistics

---

## Phase B: Validation (Ready)

### B1. Phase 0: API Validation
**Script:** `experiment/phase0_validation/run_validation.sh`

Configuration:
- Controller: PPO
- Seed: 42
- Iterations: 10
- Budget: ~20 yuan
- GPU: 2

Run:
```bash
export DEEPSEEK_API_KEY="your-api-key"
bash experiment/phase0_validation/run_validation.sh
```

Success Criteria:
- [ ] 10 iterations complete
- [ ] API calls successful
- [ ] Code generated (not template)
- [ ] Evaluator compiles and runs
- [ ] Cost < 50 yuan

### B2. Phase 0.5: Mock vs Real Comparison
**Script:** `experiment/phase0_validation/run_comparison.sh`

Configuration:
- Controller: PPO
- Seed: 42
- Iterations: 20 per run
- Budget: ~60 yuan
- GPUs: 2 (mock), 3 (real)

Run:
```bash
export DEEPSEEK_API_KEY="your-api-key"
bash experiment/phase0_validation/run_comparison.sh
```

Output:
- Side-by-side reward comparison
- API cost analysis
- Performance difference percentage

---

## Quick Start Commands

### 1. Set Environment
```bash
export DEEPSEEK_API_KEY="sk-..."
export BUDGET_LIMIT_YUAN=10000
export CACHE_DIR=".cache/llm"
```

### 2. Phase 0 (Validation)
```bash
cd experiment/phase0_validation
bash run_validation.sh
```

### 3. Phase 0.5 (Comparison)
```bash
cd experiment/phase0_validation
bash run_comparison.sh
```

### 4. Single Experiment (Test)
```bash
cd experiment/phase2_controllers
bash run_real_api.sh single ppo 42 2
```

### 5. Full Phase 2 (All Controllers)
```bash
cd experiment/phase2_controllers
bash run_real_api.sh
```

---

## File Structure

```
experiment/
├── utils/
│   ├── llm_client.py              # NEW: DeepSeek API client
│   └── __init__.py                # MODIFIED: Export llm_client
├── phase0_validation/
│   ├── README.md                  # NEW: Validation guide
│   ├── run_validation.sh          # NEW: Phase 0 script
│   └── run_comparison.sh          # NEW: Phase 0.5 script
├── phase2_controllers/
│   ├── run.sh                     # ORIGINAL: Mock runner
│   └── run_real_api.sh            # NEW: Real API runner
└── ...
```

---

## Cost Estimation

| Phase | Experiments | Iterations | Est. Cost |
|-------|-------------|------------|-----------|
| 0 | 1 | 10 | ~20 yuan |
| 0.5 | 2 | 40 | ~60 yuan |
| 1 | 15 | 1500 | ~3000 yuan |
| 2 | 30 | 3000 | ~6000 yuan |
| 3 | ~10 | ~1000 | ~2000 yuan |
| **Total** | - | - | **~11000 yuan** |

With caching (50% hit rate): **~5000-6000 yuan**

---

## Next Steps

1. **Run Phase 0** to validate API integration
2. **Run Phase 0.5** to compare mock vs real
3. **Analyze results** to confirm real API provides value
4. **Run full experiments** (Phase 1-3) if validation passes

---

## Troubleshooting

### Import Error: `openai`
```bash
pip install openai pyyaml torch numpy
```

### Budget Exceeded
Increase limit or check `api_stats.json` for usage.

### Cache Issues
Clear cache: `rm -rf .cache/llm/*.json`

### GPU Memory
Reduce batch size in config or use different GPU.

---

*Last Updated: 2026-02-11*
*Status: Phase A Complete, Phase B Ready*
