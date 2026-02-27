# Phase 0 & 0.5: Real API Validation

This directory contains validation scripts for testing the real DeepSeek API integration.

## Quick Start

### 1. Set API Key
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

### 2. Run Phase 0 (Validation)
Small-scale test (10 iterations) to verify API works:
```bash
bash run_validation.sh
```

**Expected:**
- 10 iterations complete successfully
- API cost: ~10-20 yuan
- Validation passes

### 3. Run Phase 0.5 (Mock vs Real Comparison)
Compare mock and real API performance:
```bash
bash run_comparison.sh
```

**Expected:**
- 20 iterations each (mock + real)
- Side-by-side comparison results
- API cost: ~40-60 yuan

## Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | - | Required. Your DeepSeek API key |
| `BUDGET_LIMIT_YUAN` | 50 (validation), 200 (comparison) | Maximum budget in yuan |
| `CACHE_DIR` | `.cache/llm` | Cache directory for API responses |

### Examples

**With custom budget:**
```bash
export BUDGET_LIMIT_YUAN=100
bash run_validation.sh
```

**With custom cache location:**
```bash
export CACHE_DIR=/path/to/cache
bash run_comparison.sh
```

## Output

Results are saved to:
- `results_validation/ppo_validation/` - Phase 0 results
- `results_comparison/mock/` - Mock experiment results
- `results_comparison/real/` - Real API experiment results

Each result directory contains:
- `checkpoint.pt` - Controller checkpoint
- `api_stats.json` - API usage statistics (real only)
- `summary.json` - Experiment summary

## Troubleshooting

### API Key Error
```
ERROR: DEEPSEEK_API_KEY environment variable not set!
```
**Fix:** Set the environment variable before running scripts.

### Budget Exceeded
```
WARNING: Budget exceeded! Stopping experiment.
```
**Fix:** Increase `BUDGET_LIMIT_YUAN` or check `api_stats.json` for usage.

### Import Error
```
ImportError: openai package required
```
**Fix:** Install dependencies:
```bash
pip install openai pyyaml torch numpy
```

## Next Steps

After successful validation:
1. Run full Phase 2 experiments:
   ```bash
   cd ../phase2_controllers
   bash run_real_api.sh
   ```

2. Or run single experiment for testing:
   ```bash
   export DEEPSEEK_API_KEY="your-key"
   export BUDGET_LIMIT_YUAN=100
   bash run_real_api.sh single ppo 42 2
   ```
