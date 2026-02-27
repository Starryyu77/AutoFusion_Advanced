# Phase 5 实验方案设计（简化版）
## LLM-Driven NAS 核心实验

> **目标**: 跑通流程 + Baseline 对比 + 模型影响分析

---

## 一、实验目标（简化后）

### 核心目标

| 目标 | 描述 | 成功标准 |
|------|------|----------|
| **1. 跑通流程** | 验证 Phase 5 框架端到端可用 | 完成 20 iterations 无报错 |
| **2. Baseline对比** | 对比人工设计 vs LLM生成 | 达到或超越 FiLM (46% acc, 6.29M FLOPs) |
| **3. 模型影响** | 测试不同规模模型对 NAS 效果的影响 | 找到性价比最高的模型 |

### 不做的内容
- ❌ API 提供商对比（如阿里云 vs DeepSeek官方）
- ❌ 跨平台测试

---

## 二、实验设计（简化版）

### 实验 1: 跑通流程验证（使用阿里云百炼）

**目的**: 验证框架可用性

| 配置 | 值 |
|------|-----|
| API | 阿里云百炼 |
| 模型 | DeepSeek-V3 |
| 迭代数 | 10（快速验证） |
| 评估器 | Mock（先验证代码逻辑） |

**成功标准**:
- [x] 10 iterations 完成无报错
- [x] Controller 正常更新
- [x] Few-Shot DB 正常添加示例
- [x] 结果正确保存

---

### 实验 2: 模型规模对比（使用阿里云百炼）

**目的**: 找到性价比最高的模型

| 实验 | 模型 | 规模 | 迭代数 | GPU |
|------|------|------|--------|-----|
| B1 | DeepSeek-V3 | Large | 20 | 0 |
| B2 | Qwen-Max | Large | 20 | 1 |
| B3 | Qwen-Plus | Medium | 20 | 2 |
| B4 | Qwen-Turbo | Small | 20 | 3 |

**对比维度**:
- 最佳准确率
- 收敛速度（多少轮达到最佳）
- API 调用成本
- 代码编译成功率

---

### 实验 3: Baseline 对比 + 完整搜索

**目的**: 证明 LLM-Driven 超越传统方法

| 实验 | 方法 | 迭代数 | 说明 |
|------|------|--------|------|
| C1 | **LLM-Driven** | **50** | 使用实验2中最佳的模型 |
| C2 | MockGenerator | 20 | Phase 4 的模板生成方法 |
| C3 | FiLM (人工) | - | 直接评估，不搜索 |
| C4 | CLIPFusion (人工) | - | 直接评估，不搜索 |

**成功标准**:
- LLM-Driven 准确率 > FiLM (46%)
- LLM-Driven FLOPs < FiLM (6.29M)
- LLM-Driven > MockGenerator

---

## 三、实验流程（4 周计划）

### Week 1: 跑通流程 + Baseline 评估

**Day 1-2**: 框架验证
```
- 使用阿里云百炼 DeepSeek-V3
- Mock 评估器跑 10 iterations
- 验证所有组件正常工作
```

**Day 3-4**: 集成真实评估器
```
- 复用 Phase 4 的 ImprovedRealDataFewShotEvaluator
- 接入 MMMU 数据集
- 跑 5 iterations 测试
```

**Day 5**: Baseline 评估
```
- 评估 FiLM (已有结果: 46%, 6.29M)
- 重新运行 CLIPFusion 评估
- 记录完整指标
```

### Week 2: 模型对比实验

**Day 1-2**: 并行跑 B1-B4
- 4 个实验同时在 4 个 GPU 上运行
- 各 20 iterations

**Day 3-4**: 数据分析
- 对比 4 个模型的效果
- 选择最佳模型

**Day 5**: 确定最佳模型
- 综合考虑准确率、成本、速度

### Week 3: 完整搜索

**Day 1-5**: 运行实验 C1
- 使用 Week 2 确定的最佳模型
- 跑 50 iterations
- 每日监控进度

### Week 4: 对比分析 + 报告

**Day 1-2**: 运行 C2 (MockGenerator)
- 与 C1 对比

**Day 3-4**: 数据分析
- C1 vs C2 vs C3 vs C4
- 可视化结果

**Day 5**: 撰写报告
- 实验结论
- 架构推荐

---

## 四、配置文件

### 实验 1: 快速验证
```yaml
# exp1_quick_test.yaml
llm:
  type: "aliyun"
  model: "deepseek-v3"
  api_key: "${ALIYUN_API_KEY}"

experiment:
  max_iterations: 10
  output_dir: "./results/exp1"

evaluator:
  type: "mock"  # 先用 mock 验证
```

### 实验 2: 模型对比
```yaml
# exp2_b1_deepseek.yaml, exp2_b2_qwen_max.yaml, etc.
llm:
  type: "aliyun"
  model: "deepseek-v3"  # 或 qwen-max, qwen-plus, qwen-turbo

experiment:
  max_iterations: 20
```

### 实验 3: 完整搜索
```yaml
# exp3_full_search.yaml
llm:
  type: "aliyun"
  model: "[WEEK2_BEST_MODEL]"

experiment:
  max_iterations: 50
  output_dir: "./results/exp3_full"
```

---

## 五、风险管理（简化）

| 风险 | 概率 | 应对措施 |
|------|------|----------|
| API 限额 | 中 | 控制并发数，准备重试机制 |
| 生成代码编译率低 | 中 | Week 1 快速发现并修复 Prompt |
| GPU 资源不足 | 低 | Week 2 顺序运行 B1-B4 即可 |

---

## 六、交付物

### 代码
- [x] Phase 5 框架代码（已完成）
- [x] 阿里云百炼支持（已完成）
- [ ] 评估器集成（Week 1）

### 实验结果
- [ ] 实验 1: 跑通验证报告
- [ ] 实验 2: 模型对比报告
- [ ] 实验 3: 完整搜索报告
- [ ] 最终对比: LLM-Driven vs Baseline

### 推荐
- [ ] 最佳模型推荐
- [ ] 最佳架构推荐
- [ ] 后续优化建议

---

## 七、快速开始（确认后立即执行）

### 1. 配置环境
```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
cd ~/AutoFusion_Advanced/phase5_llm_rl
export ALIYUN_API_KEY="sk-fa81e2c1077c4bf5a159c2ca5ddcf200"
```

### 2. 运行实验 1（快速验证）
```bash
bash scripts/run_local.sh configs/exp1_quick_test.yaml
```

### 3. 验证通过后，并行跑实验 2
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/run_local.sh configs/exp2_b1_deepseek.yaml &

# GPU 1  
CUDA_VISIBLE_DEVICES=1 bash scripts/run_local.sh configs/exp2_b2_qwen_max.yaml &

# GPU 2
CUDA_VISIBLE_DEVICES=2 bash scripts/run_local.sh configs/exp2_b3_qwen_plus.yaml &

# GPU 3
CUDA_VISIBLE_DEVICES=3 bash scripts/run_local.sh configs/exp2_b4_qwen_turbo.yaml &
```

---

**确认后开始执行？**

1. ✅ 确认简化版实验方案
2. ⏳ 先跑实验 1（快速验证）
3. ⏳ 同时集成 Phase 4 评估器
4. ⏳ 并行跑实验 2（模型对比）
5. ⏳ 跑实验 3（完整搜索）

需要我先做什么？
