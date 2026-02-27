# Phase 5 实验方案设计
## LLM-Driven NAS 对比实验

> **目标**: 验证 LLM 作为 Controller 的有效性，并对比不同 LLM API 的性能差异

---

## 一、实验目标

### 1.1 核心目标

| 目标 | 描述 | 成功标准 |
|------|------|----------|
| **跑通流程** | 验证 Phase 5 框架端到端可用 | 完成 20 iterations 无报错 |
| **Baseline对比** | 对比人工设计 (FiLM) vs LLM生成 | 超越 FiLM (46% acc, 6.29M FLOPs) |
| **API对比** | 对比不同 LLM API 的生成效果 | 找到最佳 API/模型组合 |
| **模型影响** | 分析模型规模对 NAS 效果的影响 | 量化大模型 vs 小模型差异 |

### 1.2 评估指标

**主要指标**:
- MMMU 验证集准确率 (target: > 46%)
- FLOPs (target: < 6.29M)
- 参数量
- 推理延迟

**次要指标**:
- 生成代码编译成功率
- 每轮迭代耗时
- API 调用成本
- 架构多样性

---

## 二、实验设计

### 2.1 实验分组

#### 实验组 A: API 对比 (固定模型)

测试不同 API 提供商，使用相似的模型规模

| 实验 | API | 模型 | 配置 | 迭代数 |
|------|-----|------|------|--------|
| A1 | 阿里云百炼 | DeepSeek-V3 | `aliyun_deepseek.yaml` | 20 |
| A2 | DeepSeek官方 | DeepSeek-V3 | `deepseek_official.yaml` | 20 |
| A3 | MiniMax | MiniMax-Text-01 | `minimax.yaml` | 20 |
| A4 | 阿里云百炼 | Qwen-Max | `aliyun_qwen.yaml` | 20 |

**对比维度**: API 稳定性、响应速度、生成质量

#### 实验组 B: 模型规模对比 (固定 API)

使用阿里云百炼API，测试不同规模的模型

| 实验 | 模型 | 规模 | 配置 | 迭代数 |
|------|------|------|------|--------|
| B1 | DeepSeek-V3 | Large | `ds_v3.yaml` | 20 |
| B2 | Qwen-Max | Large | `qwen_max.yaml` | 20 |
| B3 | Qwen-Plus | Medium | `qwen_plus.yaml` | 20 |
| B4 | Qwen-Turbo | Small | `qwen_turbo.yaml` | 20 |

**对比维度**: 模型规模 vs 生成效果、成本效益

#### 实验组 C: Baseline 对比

| 实验 | 方法 | 配置 | 迭代数 |
|------|------|------|--------|
| C1 | **LLM-Driven (我们的)** | 最佳API+模型 | 50 |
| C2 | MockGenerator (Phase 4) | `phase4_mock.yaml` | 20 |
| C3 | 人工设计 FiLM | 直接评估 | - |
| C4 | 人工设计 CLIPFusion | 直接评估 | - |

### 2.2 控制变量

**固定参数**:
- 数据集: MMMU (32-shot)
- 评估器: ImprovedRealDataFewShotEvaluator
- 约束: max_flops=10M, target_accuracy=50%
- Few-Shot: 3 examples, similarity 选择
- Temperature: 0.7 (动态调整)

**变化参数**:
- LLM Backend (API + 模型)
- 随机种子 (每个实验跑 3 个 seed)

---

## 三、实验流程

### Phase 1: 验证跑通 (Week 1)

**Day 1-2**: 跑通流程
1. 使用 Mock 模式验证代码无bug
2. 使用阿里云百炼 API 跑 5 iterations 测试
3. 修复发现的问题

**Day 3-4**: Baseline 评估
1. 评估 FiLM (已有结果: 46%, 6.29M)
2. 重新运行 CLIPFusion 评估
3. 确认评估器一致性

**Day 5**: 准备完整实验
1. 配置所有 API key
2. 准备运行脚本
3. 设计监控方案

### Phase 2: API 对比 (Week 2)

并行运行实验组 A:
- A1: 阿里云百炼 DeepSeek (GPU 0)
- A2: DeepSeek 官方 (GPU 1)
- A3: MiniMax (GPU 2)
- A4: 阿里云百炼 Qwen (GPU 3)

每实验 20 iterations，预计 2-3 小时完成

### Phase 3: 模型对比 (Week 3)

顺序运行实验组 B (避免阿里云并发限制):
- B1: DeepSeek-V3
- B2: Qwen-Max
- B3: Qwen-Plus
- B4: Qwen-Turbo

### Phase 4: 完整搜索 (Week 4)

使用最佳 API+模型，运行 50 iterations:
- C1: LLM-Driven 完整搜索
- 与 C2, C3, C4 对比

---

## 四、实施方案

### 4.1 新增阿里云百炼支持

需要修改 `llm_backend.py` 添加阿里云百炼 API:

```python
class AliyunBailianBackend(LLMBackend):
    """阿里云百炼 API 后端"""
    # 兼容 OpenAI 接口格式
    # 支持模型: DeepSeek-V3, Qwen-Max, Qwen-Plus, Qwen-Turbo
```

### 4.2 实验配置矩阵

创建 8 个配置文件:
- `exp_a1_aliyun_deepseek.yaml`
- `exp_a2_deepseek_official.yaml`
- `exp_a3_minimax.yaml`
- `exp_a4_aliyun_qwen.yaml`
- `exp_b1_ds_v3.yaml`
- `exp_b2_qwen_max.yaml`
- `exp_b3_qwen_plus.yaml`
- `exp_b4_qwen_turbo.yaml`
- `exp_c1_best.yaml` (50 iterations)

### 4.3 批量运行脚本

```bash
# run_all_experiments.sh
# 自动顺序运行所有实验，保存结果
```

---

## 五、数据分析计划

### 5.1 对比维度

1. **准确率对比**: 各实验组的 Best Accuracy
2. **效率对比**: FLOPs vs Accuracy 帕累托图
3. **收敛速度**: Reward vs Iteration 曲线
4. **成本分析**: API 调用费用 vs 效果

### 5.2 可视化

- 柱状图: 不同 API 的最佳准确率
- 散点图: FLOPs-Accuracy 帕累托前沿
- 曲线图: 收敛过程对比
- 热力图: API × 模型效果矩阵

---

## 六、预期结果

### 6.1 假设

**H1**: LLM-Driven 能超越传统 MockGenerator  
**H2**: 大模型 (DeepSeek-V3, Qwen-Max) 生成效果更好  
**H3**: 阿里云百炼响应速度 > DeepSeek 官方  
**H4**: 经过 50 iterations，能发现超越 FiLM 的架构

### 6.2 成功标准

| 指标 | 目标 | 最低要求 |
|------|------|----------|
| MMMU Accuracy | > 50% | > 46% (FiLM) |
| FLOPs | < 5M | < 6.29M (FiLM) |
| 编译成功率 | > 80% | > 60% |
| 收敛迭代数 | < 30 | < 50 |

---

## 七、风险管理

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| API 限额/超时 | 高 | 中 | 实现重试机制，准备备选API |
| 评估器集成失败 | 中 | 高 | 先使用 Mock 验证流程 |
| 生成代码编译率低 | 中 | 高 | 优化 Prompt，增加代码验证 |
| GPU 资源不足 | 低 | 中 | 顺序运行实验，避免并发 |

---

## 八、时间线

| 周 | 任务 | 交付物 |
|----|------|--------|
| W1 | 框架验证 + Baseline | 跑通的代码，Baseline结果 |
| W2 | API 对比实验 | A1-A4 结果 |
| W3 | 模型对比实验 | B1-B4 结果 |
| W4 | 完整搜索 + 分析 | C1 结果，对比分析报告 |

---

**请确认此实验方案后，我将开始实施：**
1. 添加阿里云百炼 API 支持
2. 创建所有配置文件
3. 准备批量运行脚本
4. 集成 Phase 4 评估器

需要调整的地方请告诉我！
