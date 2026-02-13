# Phase 1: Prompt Strategy Comparison Experiment Plan

## 1. 实验目标

系统性比较 5 种 Prompt 策略在 NAS 代码生成任务中的表现：

| 策略 | 核心思想 | 预期优势 |
|------|----------|----------|
| **CoT** | 逐步推理 | 逻辑清晰，易于调试 |
| **FewShot** | 示例学习 | 风格一致，减少幻觉 |
| **Critic** | 自我评估改进 | 代码质量更高 |
| **Shape** | 显式形状约束 | 减少运行时错误 |
| **RolePlay** | 专家角色模拟 | 领域知识更丰富 |

## 2. 实验设计

### 2.1 控制变量

为公平比较，固定以下参数：

| 参数 | 值 | 说明 |
|------|----|------|
| **Controller** | Evolution | Phase 2.1 中表现最佳 |
| **Evaluator** | RealDataFewShotEvaluator | Phase 2.5 验证通过 |
| **Dataset** | AI2D | 最优数据集 |
| **Train Epochs** | 3 | 最优训练深度 |
| **Num Shots** | 16 | 标准 few-shot 设置 |
| **Search Budget** | 20 iterations | 控制总成本 |
| **LLM Model** | deepseek-chat | 统一模型 |

### 2.2 评估指标

```python
METRICS = {
    # 1. 代码有效性
    'validity_rate': '成功编译的代码比例',
    'syntax_errors': '每轮平均语法错误数',

    # 2. 架构质量
    'final_accuracy': '最终验证准确率',
    'best_arch_reward': '最佳架构奖励',
    'convergence_iteration': '收敛所需迭代次数',

    # 3. 多样性
    'architecture_diversity': '生成架构的多样性（KL散度）',
    'operator_coverage': '使用的算子种类数',

    # 4. 效率
    'time_per_generation': '每次生成耗时',
    'tokens_per_prompt': 'Prompt token 数',
    'api_calls': 'API 调用次数',

    # 5. 稳定性
    'inter_run_variance': '多次运行的方差',
}
```

## 3. 假设（Hypotheses）

| 假设 | 预期结果 | 验证方式 |
|------|----------|----------|
| H1 | **Shape** 策略有最高代码有效性 | 比较 validity_rate |
| H2 | **Critic** 策略产生最高奖励 | 比较 best_reward |
| H3 | **FewShot** 收敛最快 | 比较 convergence_iteration |
| H4 | **CoT** 生成时间最长（token 多） | 比较 avg_generation_time |

## 4. 实验执行步骤

### 4.1 预实验（Pre-test）

```bash
# 1. Mock 测试 - 验证脚本逻辑
python run_phase1.py --mock --iterations 3

# 2. 单策略测试 - 验证单个策略
python run_phase1.py --strategy CoT --iterations 5

# 3. 成本估算 - 估算 API 费用
python run_phase1.py --dry-run
```

### 4.2 正式实验

```bash
# 运行完整实验（约 20 iterations × 5 strategies = 100 API calls）
python run_phase1.py \
    --run-name "phase1_final_$(date +%Y%m%d)" \
    --output-dir "results"

# 运行多次取平均（推荐 3 次）
for i in {1..3}; do
    python run_phase1.py \
        --run-name "phase1_run_$i" \
        --output-dir "results"
done
```

## 5. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| API 成本高 | 预算超支 | 限制 iterations 到 20，使用缓存 |
| 运行时间长 | 实验超时 | 每个策略单独运行，支持断点续跑 |
| 结果方差大 | 结论不可靠 | 运行 3 次取平均，使用统计检验 |
| 某些策略失效 | 数据不完整 | 添加 fallback 到 mock 生成 |

## 6. 验收标准

实验成功需满足：

- [ ] 所有 5 个策略成功运行完成
- [ ] 每个策略至少 80% 代码有效性
- [ ] 生成可复现的对比报告
- [ ] 识别出最佳 Prompt 策略

## 7. 服务器运行配置

### NTU GPU43 配置

```bash
# 登录服务器
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg

# 项目路径
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase1_prompts

# 设置环境变量
export DEEPSEEK_API_KEY="your-api-key"
export PYTHONPATH="/usr1/home/s125mdg43_10/AutoFusion_Advanced:$PYTHONPATH"

# 使用 GPU 3 (通常较空闲)
export CUDA_VISIBLE_DEVICES=3

# 运行实验
nohup python run_phase1.py --run-name phase1_gpu3 > logs/phase1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 监控运行

```bash
# 查看日志
tail -f logs/phase1_*.log

# 查看 GPU 使用
nvidia-smi -l 1

# 查看进程
ps aux | grep run_phase1
```

---

*Created: 2026-02-13*
*Phase: 1 - Prompt Strategy Comparison*
