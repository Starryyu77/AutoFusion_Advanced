# Phase 1 - NTU GPU43 服务器运行指南

## 1. 登录服务器

```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
```

## 2. 进入项目目录

```bash
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced
```

## 3. 拉取最新代码

```bash
git pull origin main
```

## 4. 设置环境变量

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
export PYTHONPATH="/usr1/home/s125mdg43_10/AutoFusion_Advanced:$PYTHONPATH"
```

## 5. 运行实验

### 方法 A：使用脚本（推荐）

```bash
cd experiment/phase1_prompts

# 使用 GPU 3（通常较空闲）
GPU_ID=3 ITERATIONS=20 RUN_NAME=phase1_gpu3 bash run_on_server.sh
```

### 方法 B：手动运行

```bash
cd experiment/phase1_prompts

# 创建日志目录
mkdir -p logs

# 运行实验
CUDA_VISIBLE_DEVICES=3 python run_phase1.py \
    --run-name phase1_gpu3 \
    --iterations 20 \
    --gpu 3 \
    2>&1 | tee logs/phase1_gpu3_$(date +%Y%m%d_%H%M%S).log
```

### 方法 C：后台运行（nohup）

```bash
cd experiment/phase1_prompts

nohup bash -c '
    export PYTHONPATH="/usr1/home/s125mdg43_10/AutoFusion_Advanced:$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES=3
    python run_phase1.py \
        --run-name phase1_full \
        --iterations 20 \
        --gpu 3
' > logs/phase1_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看进程
tail -f logs/phase1_full_*.log
```

## 6. 监控运行

### 查看日志

```bash
# 实时查看
tail -f experiment/phase1_prompts/logs/*.log

# 查看最新日志
ls -lt experiment/phase1_prompts/logs/ | head -5
```

### 查看 GPU 状态

```bash
# 监控 GPU
watch -n 1 nvidia-smi

# 或
nvidia-smi -l 1
```

### 查看进程

```bash
ps aux | grep run_phase1

# 查看特定用户的进程
ps aux | grep s125mdg43_10 | grep python
```

### 检查状态

```bash
cd experiment/phase1_prompts
python check_status.py results
```

## 7. 多 GPU 并行运行

如果有多块空闲 GPU，可以并行运行多个策略：

```bash
cd experiment/phase1_prompts

# GPU 2 - CoT
CUDA_VISIBLE_DEVICES=2 python run_phase1.py --strategy CoT --gpu 2 --run-name cot_gpu2 > logs/cot.log 2>&1 &

# GPU 3 - FewShot
CUDA_VISIBLE_DEVICES=3 python run_phase1.py --strategy FewShot --gpu 3 --run-name fewshot_gpu3 > logs/fewshot.log 2>&1 &

# 查看所有后台任务
jobs -l
```

## 8. 多次运行取平均

```bash
cd experiment/phase1_prompts

for i in 1 2 3; do
    echo "Starting run $i..."
    CUDA_VISIBLE_DEVICES=3 python run_phase1.py \
        --run-name phase1_run_$i \
        --iterations 20 \
        --gpu 3 \
        > logs/phase1_run_$i.log 2>&1
    echo "Run $i complete!"
done

# 汇总结果
python check_status.py results
```

## 9. 下载结果

在本地机器上运行：

```bash
# 使用 scp 下载结果
scp -r s125mdg43_10@gpu43.dynip.ntu.edu.sg:/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase1_prompts/results ./phase1_results

# 或使用 rsync
rsync -avz s125mdg43_10@gpu43.dynip.ntu.edu.sg:/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase1_prompts/results/ ./phase1_results/
```

## 10. 故障排除

### 问题：ImportError

```bash
# 确保 PYTHONPATH 设置正确
export PYTHONPATH="/usr1/home/s125mdg43_10/AutoFusion_Advanced:$PYTHONPATH"
```

### 问题：CUDA out of memory

```bash
# 减小 batch_size（在 run_phase1.py 中修改 VERIFIED_EVALUATOR_CONFIG）
# 或使用不同的 GPU
CUDA_VISIBLE_DEVICES=2  # 尝试 GPU 2
```

### 问题：API key 错误

```bash
# 检查环境变量
echo $DEEPSEEK_API_KEY

# 重新设置
export DEEPSEEK_API_KEY="your-key"
```

### 问题：进程被杀

可能是内存不足，尝试：
- 使用更小的 batch_size
- 减少 iterations
- 使用 mock 模式测试（--mock）

## 11. 预期运行时间

- 每个策略：约 15-20 分钟（20 iterations）
- 完整实验（5 个策略）：约 1.5-2 小时
- 多次运行（3 次）：约 4-6 小时

## 12. 预期成本

- 每次 API 调用约 0.02-0.05 元
- 20 iterations × 5 strategies = 100 API calls
- 预计成本：2-5 元

---

**注意**：
1. 确保 DEEPSEEK_API_KEY 已设置
2. 使用 `nohup` 或 `tmux` 防止 SSH 断开导致实验中断
3. 定期保存结果（结果会自动保存到 results 目录）
