# 论文素材大纲

## 标题建议

**主标题**: LLM-Driven Neural Architecture Search for Multimodal Fusion: An Empirical Study

**副标题**: From Code Generator to Controller - Challenges and Opportunities

---

## Abstract (摘要)

### 背景
多模态融合架构设计是视觉语言模型的关键挑战。传统 NAS 方法依赖复杂的搜索算法和大量计算资源。

### 方法
本文探索将 Large Language Models (LLM) 作为神经架构搜索的控制器 (Controller)，实现闭环反馈的自进化搜索。

### 实验
在 MMMU、VSR、AI2D 等多模态数据集上，系统性对比了:
- 4 种 LLM 后端 (DeepSeek-V3, GLM-5, Kimi-K2.5, Qwen-Max)
- 5 种 Prompt 策略 (CoT, FewShot, Critic, Shape, RolePlay)
- 6 种搜索算法 (PPO, GRPO, GDPO, Evolution, CMA-ES, Random)

### 发现
1. 简单评估器 (AI2D) 无法预测复杂任务 (MMMU) 性能
2. 人工设计架构在复杂任务上表现更鲁棒 (FiLM 46% > NAS 33%)
3. LLM 代码生成能力差异显著 (DeepSeek 24% vs Qwen 0% 编译成功)
4. LLM-Driven NAS 有潜力但需改进代码质量

### 结论
评估器设计是 NAS 成功的关键，LLM 作为 Controller 展现了新可能但面临代码生成挑战。

---

## 1. Introduction (引言)

### 1.1 研究背景
- 多模态融合的重要性
- 传统 NAS 方法的局限性
- LLM 的代码生成能力

### 1.2 研究动机
- 能否让 LLM 自主设计神经网络架构？
- LLM 生成的架构能否超越人工设计？
- 评估器如何影响搜索结果？

### 1.3 研究问题
1. LLM 代码生成能力如何影响 NAS？
2. 简单评估器能否预测复杂任务性能？
3. 如何改进 LLM-Driven NAS？

### 1.4 主要贡献
1. LLM-Driven NAS 框架 (Phase 5)
2. 系统性 LLM 对比实验
3. 评估器设计洞察
4. 人工设计 vs NAS 对比方法论

---

## 2. Related Work (相关工作)

### 2.1 Neural Architecture Search
- RL-based NAS (Zoph & Le, 2017)
- Differentiable NAS (DARTS, 2019)
- Evolution-based NAS (Real et al., 2019)

### 2.2 LLM for Code Generation
- Codex (Chen et al., 2021)
- StarCoder (Li et al., 2023)
- DeepSeek-Coder (Guo et al., 2024)

### 2.3 LLM for Architecture Design
- AutoML-GPT (Wang et al., 2023)
- GPT-NAS (Recent works)

### 2.4 Multimodal Fusion
- FiLM (Perez et al., 2018)
- CLIP (Radford et al., 2021)
- BLIP (Li et al., 2022)

---

## 3. Method (方法)

### 3.1 System Architecture
```
LLM Controller → Prompt Builder → Code Generation → Evaluator → Reward → Feedback
```

### 3.2 LLM Backend
- 支持 DeepSeek, GLM, Kimi, Qwen 等
- OpenAI-compatible API 接口

### 3.3 Prompt Builder
- 约束注入
- Few-Shot 示例选择
- 历史反馈整合

### 3.4 Evaluator
- RealDataFewShotEvaluator
- 支持 MMMU, VSR, AI2D 数据集

### 3.5 Reward Function
- Accuracy + Efficiency + Compile Success
- 多目标加权

---

## 4. Experiments (实验)

### 4.1 Experimental Setup
- 数据集: MMMU, VSR, AI2D
- 评估配置: 32-shot, 5 epochs
- 搜索迭代: 50 iterations

### 4.2 Phase 1: Prompt Strategy Comparison
- FewShot 最佳 (Reward 0.873)
- CoT 并列最佳

### 4.3 Phase 2: Controller Comparison
- Evolution 最佳 (Reward 9.80)
- PPO 次优

### 4.4 E1/E2: Full Evaluation
- AI2D 过于简单 (100% 所有架构)
- MMMU 上人工设计 > NAS (46% vs 33%)

### 4.5 Phase 5: LLM-Driven RL Loop
- 4 模型对比
- 编译成功率分析
- 最佳架构分析

---

## 5. Results (结果)

### 5.1 Key Findings

#### 发现 1: 评估器陷阱
| 数据集 | 难度 | 区分能力 |
|--------|------|----------|
| AI2D | 简单 | ❌ 无法区分 |
| MMMU | 困难 | ✅ 可区分 |

#### 发现 2: 人工设计优势
| 架构 | MMMU 准确率 | FLOPs |
|------|-------------|-------|
| FiLM (人工) | **46%** | 6.29M |
| arch_017 (NAS) | 33% | 13.20M |

#### 发现 3: LLM 代码生成差异
| 模型 | 编译成功率 | Best Reward |
|------|-----------|-------------|
| DeepSeek-V3 | **24%** | 2.796 |
| GLM-5 | 6% | **2.797** |
| Kimi-K2.5 | 2% | 2.539 |
| Qwen-Max | 0% | 0.500 |

### 5.2 Analysis

#### 为什么准确率低？
1. Few-shot 评估不稳定 (32 shots)
2. 训练 epochs 不足 (5 epochs)
3. 架构设计缺乏关键模式
4. LLM 生成的代码有语法错误

#### 为什么编译成功率低？
1. Prompt 缺少代码模板
2. LLM 对 PyTorch API 理解不完整
3. 错误反馈机制缺失

---

## 6. Discussion (讨论)

### 6.1 Implications

#### 对 NAS 研究的启示
1. **评估器设计至关重要**: 简单评估器导致搜索偏差
2. **搜索空间需要约束**: 效率约束缺失导致低效架构
3. **代码质量影响结果**: LLM 代码生成能力是瓶颈

#### 对 LLM 研究的启示
1. **代码生成需要专门训练**: 通用 LLM 不够
2. **结构化输出有帮助**: 代码模板可提高成功率
3. **错误反馈有价值**: 可用于迭代改进

### 6.2 Limitations

#### 方法局限
1. 编译成功率低 (最高 24%)
2. 准确率未超越 baseline
3. 模型选择有限 (4 个)

#### 实验局限
1. 数据集覆盖不全
2. 训练配置可能不是最优
3. 搜索迭代次数有限

### 6.3 Future Work

#### 短期改进 (1-2 周)
1. 优化 Prompt 设计
2. 增加代码模板
3. 实现错误反馈机制

#### 中期改进 (1-2 月)
1. Fine-tune LLM on architecture code
2. 混合搜索 (LLM + 传统 NAS)
3. 多任务学习

#### 长期方向
1. 架构代码生成 benchmark
2. 自动化架构优化系统
3. 跨领域迁移学习

---

## 7. Conclusion (结论)

### 7.1 Summary
本文探索了 LLM-Driven NAS 的可行性，发现了:
1. 评估器设计是 NAS 成功的关键
2. LLM 代码生成能力是主要瓶颈
3. 人工设计在复杂任务上仍有优势

### 7.2 Takeaway Messages
1. **不要低估评估器**: 简单评估器 = 错误搜索方向
2. **LLM 有潜力但需改进**: 代码生成能力需要提升
3. **人工设计仍有价值**: 专家知识难以完全替代

---

## 8. References (参考文献)

### NAS 相关
- Zoph & Le, "Neural Architecture Search with Reinforcement Learning", ICLR 2017
- Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019
- Real et al., "Regularized Evolution for Image Classifier Architecture Search", AAAI 2019

### LLM 代码生成
- Chen et al., "Evaluating Large Language Models Trained on Code", arXiv 2021
- Li et al., "StarCoder: May the source be with you!", arXiv 2023
- Guo et al., "DeepSeek-Coder: When the large language model meets programming", arXiv 2024

### 多模态融合
- Perez et al., "FiLM: Visual Reasoning with a Conditioned Feature Layer", AAAI 2018
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- Li et al., "BLIP: Bootstrapping Language-Image Pre-training", ICML 2022

### 数据集
- Yue et al., "MMMU: A Massive Multi-discipline Multimodal Understanding Benchmark", arXiv 2023
- Liu et al., "Visual Spatial Reasoning", TACL 2023
- Kembhavi et al., "AI2D: Diagrams in Science Question Answering", AAAI 2016

---

## 附录

### A. 实验配置详情
- GPU: 4 × NVIDIA RTX A5000 (24GB)
- Python: 3.8.10
- PyTorch: 2.4.1

### B. 代码仓库
- GitHub: https://github.com/Starryyu77/AutoFusion_Advanced

### C. 补充实验
- 详细实验日志
- 更多架构分析
- 错误案例分析

---

*Outline Generated: 2026-02-28*