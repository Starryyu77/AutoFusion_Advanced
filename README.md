# Auto-Fusion Experiment Framework

自进化多模态神经架构搜索(NAS)系统实验框架

## 结构

```
experiment/
├── base/              # 抽象基类
├── controllers/       # 6个搜索算法 (PPO/GRPO/GDPO/Evolution/CMA-ES/Random)
├── generators/        # 5个Prompt策略 (CoT/FewShot/Critic/Shape/RolePlay)
├── evaluators/        # Surgical Sandbox评估器
├── utils/             # OOM防护 + 秩相关验证
├── phase0_scaffold/   # 端到端验证 (✅ PASSED)
├── phase1_prompts/    # Prompt对比实验
├── phase2_controllers/ # Controller对比实验
├── phase3_ablation/   # 消融实验
└── scripts/           # 辅助脚本
```

## 快速开始

```bash
cd experiment/phase0_scaffold
./run.sh
```

## Phase 0 测试结果

- **状态**: ✅ PASSED
- **迭代**: 11 (早停)
- **奖励**: 0.7000
- **配置**: PPO + CoT

## 特性

- **理论修正**: GDPO方差爆炸保护、PPO Critic-Free模式
- **多目标优化**: 准确率、效率、编译成功率
- **OOM防护**: 自动batch size调整
- **秩相关验证**: Kendall's tau验证代理评估
