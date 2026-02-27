# Phase 5 实验方案设计（四大模型对比版）
## LLM-Driven NAS - 阿里云百炼四大模型对比

> **目标**: 对比阿里云百炼集成的四大顶尖模型在 NAS 任务上的效果

---

## 一、实验目标

### 核心目标

| 目标 | 描述 | 成功标准 |
|------|------|----------|
| **1. 跑通流程** | 验证 Phase 5 框架端到端可用 | 完成 20 iterations 无报错 |
| **2. Baseline对比** | 对比人工设计 vs LLM生成 | 达到或超越 FiLM (46% acc) |
| **3. 四大模型对比** | 对比 DeepSeek/MiniMax/GLM/Kimi | 找到最佳模型 |

### 不做的内容
- ❌ API 提供商对比（全部使用阿里云百炼）
- ❌ 模型规模对比（直接对比四大顶尖模型）

---

## 二、实验设计

### 实验组：四大模型对比

| 实验 | 模型 | 厂商 | 配置 | 迭代数 | GPU |
|------|------|------|------|--------|-----|
| E1 | **DeepSeek-V3.2** | DeepSeek | `exp_ds_v32.yaml` | 20 | 0 |
| E2 | **MiniMax-M2.5** | MiniMax | `exp_minimax_m25.yaml` | 20 | 1 |
| E3 | **GLM-5** | 智谱 | `exp_glm5.yaml` | 20 | 2 |
| E4 | **Kimi-K2.5** | Moonshot | `exp_kimi_k25.yaml` | 20 | 3 |

**对比维度**:
- 最佳准确率
- 收敛速度
- 代码编译成功率
- 架构多样性
- API 调用成本

---

### Baseline 评估（固定对照组）

| 实验 | 方法 | 说明 |
|------|------|------|
| B1 | **FiLM (人工)** | Phase 4 基准: 46% acc, 6.29M FLOPs |
| B2 | **CLIPFusion (人工)** | Phase 4 轻量级基准 |
| B3 | **MockGenerator** | Phase 4 模板方法 |

---

## 三、实验流程（2-3 周计划）

### Week 1: 四大模型并行实验

**Day 1**: 同时启动 4 个实验
```bash
# 4 个 GPU 同时运行四大模型对比
CUDA_VISIBLE_DEVICES=0 bash scripts/run_local.sh configs/exp_ds_v32.yaml &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_local.sh configs/exp_minimax_m25.yaml &
CUDA_VISIBLE_DEVICES=2 bash scripts/run_local.sh configs/exp_glm5.yaml &
CUDA_VISIBLE_DEVICES=3 bash scripts/run_local.sh configs/exp_kimi_k25.yaml &
```

**Day 2-3**: 监控运行
- 每个实验约 3-4 小时（20 iterations）
- 监控日志，确保正常运行

**Day 4**: 数据分析
- 对比 4 个模型的 Best Accuracy
- 对比收敛速度
- 计算 API 调用成本

**Day 5**: 确定最佳模型
- 综合准确率、成本、速度
- 选择最优模型进行完整搜索

### Week 2: 完整搜索 + Baseline对比

**Day 1-3**: 使用最佳模型跑 50 iterations
```bash
# 使用 Week 1 确定的最佳模型
bash scripts/run_local.sh configs/exp_full_best.yaml
```

**Day 4**: 评估 Baseline
- FiLM 直接评估
- CLIPFusion 评估
- MockGenerator 20 iterations

**Day 5**: 对比分析
- LLM-Driven vs Baseline
- 撰写分析报告

---

## 四、配置文件清单

| 文件 | 模型 | 说明 |
|------|------|------|
| `exp_ds_v32.yaml` | DeepSeek-V3.2 | 阿里云百炼 DeepSeek |
| `exp_minimax_m25.yaml` | MiniMax-M2.5 | 阿里云百炼 MiniMax |
| `exp_glm5.yaml` | GLM-5 | 阿里云百炼 智谱 |
| `exp_kimi_k25.yaml` | Kimi-K2.5 | 阿里云百炼 Moonshot |
| `exp_full_best.yaml` | [Week1最佳] | 完整 50 iterations |

---

## 五、快速开始

### 1. 配置环境
```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
cd ~/AutoFusion_Advanced/phase5_llm_rl

# 阿里云百炼 API Key 已配置在 yaml 中
```

### 2. 立即启动四大模型对比实验
```bash
# 并行运行
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_local.sh configs/exp_ds_v32.yaml > results/exp_ds_v32.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_local.sh configs/exp_minimax_m25.yaml > results/exp_minimax_m25.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_local.sh configs/exp_glm5.yaml > results/exp_glm5.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_local.sh configs/exp_kimi_k25.yaml > results/exp_kimi_k25.log 2>&1 &
```

### 3. 监控进度
```bash
tail -f results/exp_*.log
grep "Best" results/exp_*.log
```

---

## 六、预期结果

| 模型 | 预期准确率 | 预期 FLOPs | 特点 |
|------|-----------|-----------|------|
| DeepSeek-V3.2 | > 45% | < 10M | 开源最强，推理能力强 |
| MiniMax-M2.5 | > 43% | < 10M | 对话优化，速度快 |
| GLM-5 | > 44% | < 10M | 智谱旗舰，中文强 |
| Kimi-K2.5 | > 44% | < 10M | 长文本，逻辑推理 |

**最佳模型**: 预计是 DeepSeek-V3.2 或 GLM-5

---

## 七、交付物

### 代码（已完成 ✅）
- [x] Phase 5 框架代码
- [x] 阿里云百炼支持
- [x] 四大模型配置文件

### 实验结果（待执行）
- [ ] 四大模型对比报告
- [ ] 完整搜索报告（50 iterations）
- [ ] Baseline 对比报告

### 推荐
- [ ] 最佳模型推荐
- [ ] 最优架构代码

---

**已准备就绪，可以立即执行四大模型对比实验！**

*Last Updated: 2026-02-27*
