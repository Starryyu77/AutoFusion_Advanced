# 评估器验证详解：如何评估 + 好坏标准

**Date**: 2026-02-11

---

## 一、评估流程详解（具体怎么做）

### 1.1 当前评估器的问题

```python
# 当前 SurgicalSandboxEvaluator
def _generate_mock_data(self):
    # 问题：使用随机特征！
    vision_features = torch.randn(100, 768)      # 随机噪声
    language_features = torch.randn(100, 768)    # 随机噪声
    labels = torch.randint(0, 10, (100,))        # 随机标签
```

**问题**：在随机数据上表现好 ≠ 在真实任务上表现好

### 1.2 新评估流程（RealDataFewShot）

```
输入：生成的融合模块代码 (FusionModule.py)

Step 1: 编译检查
   ├─ 代码是否能运行？
   ├─ 是否包含 nn.Module 子类？
   └─ 是否接受 (vision_features, language_features) 输入？

Step 2: 加载预训练 MLLM（冻结）
   ├─ 加载 BLIP-2 或 CLIP（已预训练）
   ├─ 冻结视觉编码器
   ├─ 冻结文本编码器
   └─ 冻结输出头

Step 3: 插入融合模块（唯一可训练）
   ├─ 将生成的 FusionModule 插入视觉和文本之间
   └─ 只解冻融合模块的参数

Step 4: Few-Shot 训练
   ├─ 从真实数据集采样 16 shots per class
   │   例：MMMU 有 10 个学科，每学科 16 题 = 160 题
   ├─ 只训练融合模块（不是整个模型）
   └─ 训练 N 个 epochs（1/3/5/10）

Step 5: 验证评估
   ├─ 在完整验证集上测试（非 few-shot）
   ├─ 计算准确率
   └─ 返回 EvaluationResult

输出：{accuracy: 0.75, efficiency: 0.8, ...}
```

### 1.3 关键区别

| 环节 | 旧评估器 | 新评估器 |
|------|----------|----------|
| 数据 | 随机噪声 `torch.randn()` | 真实图像+文本（MMMU等） |
| 训练目标 | 拟合随机标签 | 真实分类任务 |
| 预训练 | 无 | 使用 BLIP-2/CLIP 预训练权重 |
| 意义 | 只能验证代码能跑 | 能预测真实任务性能 |

---

## 二、判断评估器好坏的 4 个标准

### 标准 1：Ranking Correlation（排名相关性）⭐ 最重要

**核心问题**：评估器的排序是否与真实性能排序一致？

#### 举例说明

```
假设有 8 个架构，真实性能（Full Training 100 epochs）排名：

真实排名 R_full = [A, B, C, D, E, F, G, H]
                 （A最好，H最差）

评估器 A（Quick Eval 5 epochs）给出的排名：
R_eval_A = [A, B, C, D, E, F, G, H]  → τ = 1.0（完美！）

评估器 B（Quick Eval 5 epochs）给出的排名：
R_eval_B = [A, C, B, D, E, F, H, G]  → τ = 0.8（良好）

评估器 C（Quick Eval 1 epoch）给出的排名：
R_eval_C = [B, A, D, C, F, E, H, G]  → τ = 0.6（一般）

评估器 D（Random）给出的排名：
R_eval_D = [H, G, F, E, D, C, B, A]  → τ = -1.0（完全相反！）
```

#### Kendall's Tau 解释

```
τ = 1.0  → 完全正相关（评估器排序 = 真实排序）
τ = 0.8  → 强相关（偶尔错序，但大体正确）
τ = 0.5  → 中等相关（很多错序，仅供参考）
τ = 0.0  → 无关（随机）
τ = -1.0 → 完全负相关（反向指标）
```

#### 通过标准

| τ 范围 | 评级 | 结论 |
|--------|------|------|
| > 0.8 | 优秀 | 完全可以信任 |
| 0.7-0.8 | 良好 | 可以使用 |
| 0.5-0.7 | 一般 | 可用但需谨慎 |
| < 0.5 | 不可信 | 必须调整配置 |

### 标准 2：Discriminative Power（区分能力）

**核心问题**：评估器能否区分好架构和差架构？

#### 举例说明

```
好评估器：
  好架构得分 = 8.5
  差架构得分 = 3.2
  Gap = 5.3 ✓（能清晰区分）

差评估器：
  好架构得分 = 6.1
  差架构得分 = 5.8
  Gap = 0.3 ✗（几乎无法区分）
```

#### 量化指标

```python
# Top-K 准确率
top_architectures = 选择真实最好的 3 个架构
评估器是否将这 3 个排在 top-3？

# Score Gap
gap = mean(好架构分数) - mean(差架构分数)
gap > 2.0 视为良好
```

### 标准 3：Cost Efficiency（性价比）

**核心问题**：获得的信息是否值得花费的时间？

#### 计算方式

```
性价比 = 信息增益 / 时间成本

配置 A：5 epochs
  - τ = 0.75
  - 时间 = 5 min
  - 性价比 = 0.75 / 5 = 0.15

配置 B：10 epochs
  - τ = 0.78
  - 时间 = 10 min
  - 性价比 = 0.78 / 10 = 0.078

虽然 B 的相关性略高，但 A 的性价比更好！
```

### 标准 4：Stability（稳定性）

**核心问题**：不同随机种子下结果是否稳定？

#### 举例说明

```
稳定的评估器（好）：
  Architecture X, Seed 42:  8.5
  Architecture X, Seed 123: 8.3
  Architecture X, Seed 456: 8.6
  方差 = 0.15 ✓

不稳定的评估器（差）：
  Architecture X, Seed 42:  8.5
  Architecture X, Seed 123: 6.2
  Architecture X, Seed 456: 9.1
  方差 = 1.45 ✗（结果不可复现）
```

---

## 三、具体实验如何运作

### 实验 2.5.1：数据集选择

**目标**：哪个真实数据集最能预测最终性能？

```
准备阶段：
  1. 选择 8 个代表性架构
     ├─ Evolution 发现的 Top-4（好架构）
     ├─ PPO 发现的 Top-3（中等架构）
     └─ Random 采样 1 个（差架构）

  2. 获取 Ground Truth
     └─ 每个架构跑 Full Training（100 epochs）
     └─ 记录真实排名 R_full

实验阶段：
  对每个数据集 D in [MMMU, VSR, MathVista, AI2D]:
      对每个架构 A in 8 architectures:
          在 D 上评估 A（5 epochs, 16 shots）

      得到评估器排名 R_D
      计算 τ(R_D, R_full)

输出：
  MMMU:    τ = 0.82  ✓ 最佳
  MathVista: τ = 0.75  ✓ 第二
  VSR:     τ = 0.68  △ 可用
  AI2D:    τ = 0.71  ✓ 第三

结论：选择 MMMU 作为主要评估数据集
```

### 实验 2.5.2：训练深度校准

**目标**：多少 epochs 最经济有效？

```
准备阶段：
  - 使用实验 2.5.1 选出的最佳数据集（MMMU）
  - 相同的 8 个架构
  - 相同的 Ground Truth R_full

实验阶段：
  对深度 E in [1, 3, 5, 10, 20]:
      对每个架构 A in 8 architectures:
          在 MMMU 上评估 A（E epochs）

      得到排名 R_E
      计算 τ(R_E, R_full)
      记录时间 T_E

输出：
  1 epoch:   τ = 0.55,  T = 1 min
  3 epochs:  τ = 0.72,  T = 3 min  ✓ 甜点
  5 epochs:  τ = 0.78,  T = 5 min  ✓ 推荐
  10 epochs: τ = 0.81,  T = 10 min △ 略好但慢
  20 epochs: τ = 0.82,  T = 20 min △ 收益递减

结论：
  - 快速筛选：3 epochs
  - 精确评估：5 epochs
```

### 实验 2.5.3：架构适配性

**目标**：评估器对不同架构类型是否公平？

```
准备阶段：
  - 使用最佳配置（MMMU + 5 epochs）

实验阶段：
  对每种架构类型 T in [Attention, Conv, Transformer, MLP, Hybrid]:
      选择 3 个该类型的架构

      对 seed in [42, 123, 456]:
          评估并记录分数

      计算方差 Var_T

输出分析：
  Attention:   Var = 0.12 ✓ 稳定
  Conv:        Var = 0.15 ✓ 稳定
  Transformer: Var = 0.18 ✓ 稳定
  MLP:         Var = 0.89 ✗ 不稳定！
  Hybrid:      Var = 0.21 ✓ 稳定

结论：
  - MLP 架构在 few-shot 设置下不稳定
  - 可能需要为 MLP 调整配置（更多 shots？）
  - 或在论文中标注此限制
```

---

## 四、最终输出示例

### 推荐配置（实验 2.5.4 产出）

```yaml
# configs/evaluator_recommended.yaml

# 主要配置（推荐用于大多数情况）
primary:
  dataset: mmmu
  num_shots: 16
  train_epochs: 5
  expected_tau: 0.78
  eval_time: ~5min
  use_for: [architecture_search, ranking]

# 快速筛选配置（用于大量候选）
fast:
  dataset: mmmu
  num_shots: 16
  train_epochs: 3
  expected_tau: 0.72
  eval_time: ~3min
  use_for: [initial_screening]

# 精确验证配置（用于最终候选）
precise:
  dataset: mmmu
  num_shots: 32
  train_epochs: 10
  expected_tau: 0.85
  eval_time: ~15min
  use_for: [final_validation]

# 多数据集验证（用于重要架构）
multi:
  datasets: [mmmu, mathvista]
  num_shots: 16
  train_epochs: 5
  use_for: [cross_validation]

# 已知限制
limitations:
  - MLP architectures may need 32 shots for stability
  - VSR not recommended as primary dataset (lower tau)
```

### 验证报告（论文 Method 部分）

```markdown
## Evaluator Verification

To ensure our evaluator reliably predicts real-world performance,
we conducted systematic verification across three dimensions:

### Dataset Selection
We evaluated 8 representative architectures on 4 datasets (MMMU,
MathVista, VSR, AI2D) and computed Kendall's τ correlation with
full training results (100 epochs).

Results:
- MMMU: τ = 0.82 (selected as primary)
- MathVista: τ = 0.75
- VSR: τ = 0.68
- AI2D: τ = 0.71

### Training Depth Calibration
Using MMMU, we tested 5 training depths:

| Epochs | τ   | Time | Efficiency |
|--------|-----|------|------------|
| 1      | 0.55| 1min | 0.55       |
| 3      | 0.72| 3min | 0.24       |
| 5      | 0.78| 5min | 0.156      |
| 10     | 0.81| 10min| 0.081      |
| 20     | 0.82| 20min| 0.041      |

We selected 5 epochs as the default configuration, balancing
accuracy (τ = 0.78) and efficiency.

### Architecture Fairness
Testing across 5 architecture types with 3 random seeds:
- Attention, Conv, Transformer, Hybrid: σ² < 0.2 (stable)
- MLP: σ² = 0.89 (higher variance, noted in limitations)
```

---

## 五、关键洞察

### 为什么这样设计？

1. **Few-shot 而非 Full Training**
   - Full training 100 epochs = 几小时
   - Few-shot 5 epochs = 5 分钟
   - 如果排序一致，few-shot 足够

2. **冻结主干**
   - 预训练 MLLM 已经很强
   - 架构搜索的重点是"如何融合"
   - 只训融合层 = 快速评估融合质量

3. **真实数据**
   - 随机数据无法测试视觉理解能力
   - MMMU 需要真实学科知识
   - MathVista 需要几何推理

### 失败案例警示

```
❌ 不好的评估器配置示例：
   - 数据集：随机噪声
   - 训练：5 epochs
   - 结果：τ = 0.3（与真实性能无关）
   - 后果：搜索出的架构在真实任务上表现差

❌ 不好的评估器配置示例：
   - 数据集：MMMU
   - 训练：1 epoch
   - 结果：τ = 0.5（太浅，不稳定）
   - 后果：好架构和差架构得分接近，无法区分

✅ 好的评估器配置示例：
   - 数据集：MMMU
   - 训练：5 epochs
   - 结果：τ = 0.78（强相关）
   - 效果：评估器排名 ≈ 真实排名
```

---

*Last Updated: 2026-02-11*
