# Phase 4 集群部署指南

## 目标
在 NTU EEE GPU Cluster 上运行 Phase 4 优化实验

---

## Step 1: 本地准备

### 1.1 确保所有代码已提交

```bash
cd /Users/starryyu/2026/Auto-Fusion-Advanced
git add .
git commit -m "feat(phase4): Add improved evaluator, constrained reward, and search script"
git push
```

### 1.2 创建部署包

```bash
# 在项目根目录执行
 tar -czf phase4_deploy.tar.gz \
  experiment/ \
  phase4_optimization/ \
  expv2/data/ \
  requirements.txt \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git'
```

---

## Step 2: 集群环境准备

### 2.1 连接到集群

```bash
ssh ntu-cluster
# 或
ssh tianyu016@10.97.216.128
```

### 2.2 创建项目目录

```bash
cd /projects/tianyu016
mkdir -p AutoFusion_Advanced
cd AutoFusion_Advanced
```

### 2.3 上传代码

**方式A: 通过 scp**

```bash
# 在本地执行
scp phase4_deploy.tar.gz ntu-cluster:/projects/tianyu016/AutoFusion_Advanced/

# 在集群上解压
ssh ntu-cluster "cd /projects/tianyu016/AutoFusion_Advanced && tar -xzf phase4_deploy.tar.gz"
```

**方式B: 通过 git clone**

```bash
# 在集群上执行
cd /projects/tianyu016
git clone https://github.com/Starryyu77/AutoFusion_Advanced.git
cd AutoFusion_Advanced
```

### 2.4 创建 Conda 环境

```bash
cd /projects/tianyu016/AutoFusion_Advanced

# 创建环境
conda create -n autofusion python=3.10 -y
conda activate autofusion

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets clip
pip install numpy pandas matplotlib seaborn
pip install openai tiktoken  # For DeepSeek API

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2.5 准备数据集

```bash
# 数据集应该已经在 expv2/data/ 中
# 验证数据集存在
ls -la expv2/data/
# 应该看到: ai2d, mmmu, vsr, mathvista
```

---

## Step 3: 运行实验

### 3.1 快速测试（验证环境）

```bash
# 登录到计算节点 (交互式)
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

# 加载环境
module load cuda/11.8
conda activate autofusion

cd /projects/tianyu016/AutoFusion_Advanced

# 运行快速测试 (10 iterations)
python phase4_optimization/run_phase4_search.py \
    --num-iterations 10 \
    --dataset ai2d \
    --num-shots 16 \
    --train-epochs 3 \
    --output-dir ./test_output

# 退出交互式会话
exit
```

### 3.2 提交完整实验

```bash
cd /projects/tianyu016/AutoFusion_Advanced
sbatch phase4_optimization/scripts/submit_phase4.sh
```

### 3.3 监控任务

```bash
# 查看任务状态
squeue -u tianyu016

# 查看输出
tail -f phase4_*.out

# 查看错误
tail -f phase4_*.err
```

---

## Step 4: 结果获取

### 4.1 在集群上查看结果

```bash
cd /projects/tianyu016/AutoFusion_Advanced/phase4_optimization/results/discovery
ls -la

# 查看统计
cat results_final.json | python -m json.tool | head -100
```

### 4.2 同步到本地

```bash
# 在本地执行
rsync -avz ntu-cluster:/projects/tianyu016/AutoFusion_Advanced/phase4_optimization/results/ \
  ./phase4_optimization/results_cluster/
```

---

## 配置说明

### 评估器配置

```yaml
dataset: mmmu              # 使用 MMMU 数据集
num_shots: 32              # 32-shot learning
train_epochs: 10           # 最多10 epochs
max_training_time: 300     # 5分钟时间限制
early_stopping: true       # 启用早停
```

### 奖励函数配置

```yaml
weights:
  accuracy: 1.0
  efficiency: 1.5          # 提升效率权重
  compile_success: 2.0

flops_constraint:
  enabled: true
  max_flops: 10000000      # 10M FLOPs 上限
  reject_if_exceed: true   # 超出直接拒绝
```

### 搜索算法配置

```yaml
controller: evolution
population_size: 50
num_iterations: 200
```

---

## 预期运行时间

| 配置 | 单次评估时间 | 总时间 (200 iter) |
|------|-------------|-------------------|
| MMMU + 10ep + 早停 | ~3-5 min | ~10-16 小时 |
| AI2D + 3ep (快速测试) | ~0.5 min | ~1-2 小时 |

**注意**: 早停会在验证集不提升时提前停止，实际时间可能更短。

---

## 故障排除

### 问题1: CUDA 内存不足

```bash
# 解决方案: 减小 batch_size
python phase4_optimization/run_phase4_search.py --batch-size 4
```

### 问题2: MMMU 数据集加载失败

```bash
# 验证数据集存在
ls -la expv2/data/mmmu/

# 如果缺失，需要重新下载
python -c "from datasets import load_dataset; load_dataset('MMMU/MMMU', 'Accounting', split='validation', cache_dir='./expv2/data/mmmu')"
```

### 问题3: DeepSeek API 调用失败

```bash
# 检查 API key 设置
echo $DEEPSEEK_API_KEY

# 如果没有设置，添加到 ~/.bashrc
export DEEPSEEK_API_KEY="your-api-key"
```

### 问题4: 任务被杀死 (OOM)

```bash
# 申请更多内存
sbatch --mem=64G phase4_optimization/scripts/submit_phase4.sh
```

---

## 检查点

### Checkpoint 1: 环境验证
- [ ] SSH 连接成功
- [ ] Conda 环境创建成功
- [ ] PyTorch + CUDA 正常工作
- [ ] 快速测试 (10 iter) 完成

### Checkpoint 2: 小规模搜索
- [ ] 50 iterations 完成
- [ ] 平均 FLOPs < 15M
- [ ] 没有频繁拒绝 (>50%)

### Checkpoint 3: 完整搜索
- [ ] 200 iterations 完成
- [ ] 至少 1 个架构 > 46% on MMMU
- [ ] 同一架构 < 6.29M FLOPs

---

## 联系人

如有问题，请联系:
- 集群支持: NTU EEE Cluster Documentation
- 项目维护: (你的名字)

---

*Last Updated: 2026-02-24*
