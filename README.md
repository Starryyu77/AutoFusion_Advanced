# Auto-Fusion Experiment Framework

自进化多模态神经架构搜索(NAS)系统实验框架

## 项目概述

AutoFusion 是一个用于自动设计多模态融合架构的神经网络架构搜索(NAS)系统，系统性比较 RL 算法、提示策略对多模态架构生成的影响。

**核心流程**: Controller → Generator(LLM) → Evaluator(Sandbox) → Reward

---

## 项目结构

```
experiment/
├── base/                 # 抽象基类
├── controllers/          # 6个搜索算法 (PPO/GRPO/GDPO/Evolution/CMA-ES/Random)
├── generators/           # 5个Prompt策略 (CoT/FewShot/Critic/Shape/RolePlay)
├── evaluators/           # RealDataFewShotEvaluator (已验证)
├── data/                 # 数据集加载器 (MMMU/VSR/MathVista/AI2D)
├── utils/                # OOM防护 + 秩相关验证 + LLM Client
├── phase0_validation/    # API验证 (✅ PASSED)
├── phase2_controllers/   # Controller对比实验 (✅ 完成)
├── phase2_5/             # 评估器验证 (✅ 完成)
│   ├── run_2_5_1_dataset_selection.py
│   ├── run_2_5_2_training_depth.py
│   ├── run_2_5_3_architecture_fairness.py
│   └── results/
├── phase1_prompts/       # Prompt对比实验 (⏳ 待运行)
└── phase3_ablation/      # 架构发现 (⏳ 规划中)

docs/
├── experiments/          # 实验报告
│   ├── PHASE_2_5_1_REPORT.md  # 数据集选择
│   ├── PHASE_2_5_2_REPORT.md  # 训练深度校准
│   └── PHASE_2_5_3_REPORT.md  # 架构公平性
└── design/               # 设计文档
```

---

## 实验进度

| Phase | 名称 | 状态 | 关键结果 |
|-------|------|------|----------|
| 0/0.5 | API验证 | ✅ | Mock ≈ Real (τ验证通过) |
| 2.1 | Controller对比 | ✅ | Evolution(9.8) > PPO(8.68) > GRPO(5.69) > GDPO(4.69) |
| **2.5** | **评估器验证** | **✅** | **AI2D + 3 epochs + EXCELLENT公平性** |
| 1 | Prompt对比 | ⏳ | 使用验证后的评估器 |
| 3 | 架构发现 | 📋 | 使用验证后的评估器 |

---

## ✅ Phase 2.5 评估器验证完成

### 验证结果

| 验证项 | 结果 | 关键指标 |
|--------|------|----------|
| **数据集选择** (2.5.1) | **AI2D** | 准确率 0.25 (最高) |
| **训练深度校准** (2.5.2) | **3 epochs** | 时间 2.7s (最快) |
| **架构公平性** (2.5.3) | **EXCELLENT** | std=0.056 (< 0.1) |

### 验证后的评估器配置

```python
verified_evaluator_config = {
    'dataset': 'ai2d',              # 最优数据集
    'train_epochs': 3,              # 最优训练深度
    'num_shots': 16,                # few-shot样本数
    'batch_size': 4,                # 批大小
    'backbone': 'clip-vit-l-14',    # 预训练骨干
}
```

### 实验报告

- [PHASE_2_5_1_REPORT.md](docs/experiments/PHASE_2_5_1_REPORT.md) - 数据集选择实验
- [PHASE_2_5_2_REPORT.md](docs/experiments/PHASE_2_5_2_REPORT.md) - 训练深度校准实验
- [PHASE_2_5_3_REPORT.md](docs/experiments/PHASE_2_5_3_REPORT.md) - 架构公平性测试

---

## 快速开始

### 环境设置

```bash
# 安装依赖
pip install torch torchvision transformers datasets pillow numpy pandas

# 验证环境
cd experiment/phase0_validation
python run_val.py
```

### 运行 Phase 2.5 验证实验

```bash
# 数据集选择 (2.5.1)
python experiment/phase2_5/run_2_5_1_dataset_selection.py

# 训练深度校准 (2.5.2)
python experiment/phase2_5/run_2_5_2_training_depth.py

# 架构公平性测试 (2.5.3)
python experiment/phase2_5/run_2_5_3_architecture_fairness.py
```

---

## 特性

- **理论修正**: GDPO方差爆炸保护、PPO Critic-Free模式
- **多目标优化**: 准确率、效率、编译成功率
- **OOM防护**: 自动batch size调整
- **秩相关验证**: Kendall's tau验证代理评估
- **真实数据评估**: RealDataFewShotEvaluator (AI2D, 3 epochs)

---

## 技术架构

| 组件 | 实现 | 说明 |
|------|------|------|
| Controllers | PPO, GRPO, GDPO, Evolution, CMA-ES, Random | 6种搜索算法 |
| Generators | CoT, FewShot, Critic, Shape, RolePlay | 5种提示策略 |
| Evaluator | RealDataFewShotEvaluator | 真实数据few-shot评估 |
| Reward | MultiObjective + Exponential | 准确率+效率+有效性 |

---

## 服务器配置

- **Host**: `ntu-gpu43` / `gpu43.dynip.ntu.edu.sg`
- **GPU**: 4 × NVIDIA RTX A5000 (24GB)
- **项目路径**: `/usr1/home/s125mdg43_10/AutoFusion_Advanced/`

---

## GitHub 仓库

https://github.com/Starryyu77/AutoFusion_Advanced

---

*Last Updated: 2026-02-11*
*Status: Phase 2.5 Complete ✅, Phase 1 Ready ⏳*
