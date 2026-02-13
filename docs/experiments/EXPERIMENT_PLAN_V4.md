# AutoFusion 实验计划 V4 (含 RealDataFewShot 评估器)

**Version**: 4.0
**Date**: 2026-02-11
**Status**: Phase 1 完成 ✅

---

## 核心变更

### 新评估器: RealDataFewShotEvaluator

**问题**: 原 SurgicalSandboxEvaluator 使用模拟数据（随机特征），无法反映真实任务性能。

**解决方案**:
- 真实数据集 few-shot 验证 (16/32/64 shots)
- 冻结预训练 MLLM，只训练融合层
- 支持可变训练深度 (1/3/5/10 epochs)
- 支持多数据集对比 (MMMU/VSR/MathVista/AI2D)

---

## 完整实验架构

```
Phase 0: 基础设施验证 (已完成 ✅)
  ├─ 0.0: API 连接验证
  └─ 0.5: Mock vs Real API 对比

Phase 1: Prompt 策略对比 (已完成 ✅)
  ├─ Generator: Real API (DeepSeek-V3)
  ├─ Evaluator: 使用 Phase 2.5 验证后的配置
  └─ Winner: FewShot (Best Reward 0.873)

Phase 2: Controller 对比 + 评估器校准
  ├─ 2.1: Controller 算法对比 (Mock) ✅
  │     └─ Evolution > PPO > GRPO > GDPO > Random
  │
  └─ 2.5: 评估器验证与校准 (NEW) ⭐
        ├─ 2.5.1: 数据集选择实验
        ├─ 2.5.2: 训练深度校准实验
        ├─ 2.5.3: 架构适配性实验
        └─ 2.5.4: 最终配置确定

Phase 3: 架构发现 (规划中 📋)
  ├─ 3.1: 最佳组合探索
  ├─ 3.2: 消融实验
  └─ 3.3: SOTA 刷榜 (MMMU Leaderboard)
```

---

## Phase 2.5: 评估器验证与校准

### 目标
找到最佳 (数据集, 训练深度, 架构) 组合，建立可靠的评估器配置。

### 2.5.1: 数据集选择实验

**问题**: 哪个真实数据集最能区分架构好坏？

**设计**:
```
输入: 8 个代表性架构
      ├─ Evolution Top-4
      ├─ PPO Top-3
      └─ Random × 1

变量: 数据集 ∈ {MMMU, VSR, MathVista, AI2D}
固定: 训练深度 = 5 epochs

输出: 各数据集与 Full Training 的相关性 (Kendall's τ)
```

**指标**:
- Ranking Correlation (τ)
- Discriminative Power (好vs差架构 gap)
- Evaluation Time
- Stability (variance across seeds)

**预期**: 确定 1-2 个最佳数据集

### 2.5.2: 训练深度校准实验

**问题**: 多少 epochs 最经济有效？

**设计**:
```
输入: 8 个代表性架构
固定: 数据集 = 2.5.1 最佳

变量: 训练深度 ∈ {1, 3, 5, 10, 20} epochs

输出: 深度-相关性曲线
```

**指标**:
- 与 Full Training (100 epochs) 的相关性
- 训练时间
- 收敛稳定性

**预期**: 找到"甜点"深度 (如 3 或 5 epochs)

### 2.5.3: 架构适配性实验

**问题**: 评估器对不同架构类型是否公平？

**设计**:
```
输入: 5 种架构类型
      ├─ Attention-based
      ├─ Conv-based
      ├─ Transformer-based
      ├─ MLP-based
      └─ Hybrid

固定: 数据集 + 训练深度 = 最佳配置
变量: 3 random seeds

输出: 架构-性能矩阵
```

**指标**:
- 各类型架构的排名稳定性
- 是否存在评估器偏见

**预期**: 识别潜在偏见，必要时为不同架构推荐不同配置

### 2.5.4: 最终配置确定

**输出**:
```yaml
# configs/evaluator_recommended.yaml
recommended:
  primary:
    dataset: mmmu  # 或 2.5.1 最佳
    num_shots: 16
    train_epochs: 5  # 或 2.5.2 最佳

  fallback:
    dataset: vsr  # 第二选择
    train_epochs: 3  # 快速筛选
```

---

## 基础设施开发

### 新增组件

| 组件 | 文件 | 功能 |
|------|------|------|
| RealDataFewShotEvaluator | `evaluators/real_data_evaluator.py` | 真实数据评估 |
| MultiDatasetEvaluator | `evaluators/multi_dataset_evaluator.py` | 多数据集对比 |
| DatasetLoader | `data/dataset_loader.py` | 统一数据加载 |
| FewShotSampler | `data/few_shot_sampler.py` | few-shot 采样 |
| TrainingProtocol | `protocols/training_protocol.py` | 训练协议 |

### 数据准备

```bash
# 下载 4 个验证数据集
python scripts/download_datasets.py

# 数据集统计
- MMMU: ~11K samples, multi-choice QA
- VSR: ~10K samples, spatial relation classification
- MathVista: ~6K samples, visual math reasoning
- AI2D: ~4K samples, diagram understanding
```

---

## 时间线

| 阶段 | 任务 | 时间 | 并行度 |
|------|------|------|--------|
| **Week 1** | | | |
| | 基础设施开发 (评估器类) | 2-3 天 | 1人 |
| | 数据集下载与预处理 | 1-2 天 | 并行 |
| **Week 1-2** | | | |
| | 2.5.1 数据集选择实验 | 2 天 | 4 GPUs |
| | 2.5.2 训练深度校准实验 | 2 天 | 4 GPUs |
| | 2.5.3 架构适配性实验 | 1 天 | 3 GPUs |
| | 2.5.4 最终配置确定 | 0.5 天 | - |
| **Week 2-3** | | | |
| | Phase 1 Prompt 对比 | 3-5 天 | 4 GPUs |
| **Week 3-4** | | | |
| | Phase 3 架构发现 | 5-7 天 | 4 GPUs |
| **Week 4-5** | | | |
| | 论文撰写 | 7 天 | - |

---

## 关键决策点

### 决策 1: 预训练模型选择
**选项**:
- A: CLIP-ViT-L/14 (轻量，快)
- B: BLIP-2 (平衡)
- C: LLaVA-1.5 (强，慢)

**建议**: B (BLIP-2)，平衡性能和速度

### 决策 2: Few-shot 样本数
**选项**: 16 / 32 / 64 shots

**建议**: 16 shots (平衡效率和性能)

### 决策 3: 相关性阈值
**目标**: Kendall's τ > 0.7 (强相关)

**应对**:
- 若 τ < 0.5: 增加训练深度
- 若 τ = 0.5-0.7: 可用但需标注不确定性
- 若 τ > 0.7: 通过

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 数据集下载失败 | Medium | High | 使用 HuggingFace 镜像，提前下载 |
| 显存不足 | High | High | 梯度累积，batch_size=1 |
| 训练时间过长 | Medium | Medium | 4 GPUs 并行，early stopping |
| 相关性不足 | Low | High | 增加 shots 或 epochs |

---

## 预期产出

### 技术组件
1. `RealDataFewShotEvaluator` 类
2. `MultiDatasetEvaluator` 类
3. `configs/evaluator_recommended.yaml`
4. 数据加载器套件

### 实验报告
1. 数据集选择报告 (含相关性分析)
2. 训练深度校准曲线
3. 架构适配性热力图
4. 最终推荐配置

### 下游影响
- Phase 1: 使用验证后的评估器
- Phase 3: 使用验证后的评估器
- 论文: Method 部分包含评估器验证

---

## 下一步行动

待确认后：

1. **Day 1-2**: 实现基础设施
   - RealDataFewShotEvaluator
   - DatasetLoader
   - FewShotSampler

2. **Day 2-3**: 数据准备
   - 下载 4 个数据集
   - 验证格式
   - 创建缓存

3. **Day 4-9**: 运行验证实验
   - 2.5.1 数据集选择
   - 2.5.2 训练深度校准
   - 2.5.3 架构适配性

4. **Day 10**: 确定最终配置
   - 生成推荐配置
   - 更新所有实验脚本

---

*Last Updated: 2026-02-11*
*Status: 待确认*
