# Auto-Fusion: LLM-Driven Neural Architecture Search

**LLM驱动的多模态神经架构搜索系统**

## 项目简介

Auto-Fusion 是一个自进化的多模态神经架构搜索(NAS)系统，核心创新是将 LLM 从静态代码生成器升级为全局控制器(Controller)，实现真正的 LLM-Driven NAS。

### 核心特性

- **LLM as Controller**: LLM 直接作为搜索策略，自主决定架构改进方向
- **闭环反馈**: 评估结果反馈给 LLM，形成持续进化的搜索循环
- **多模型支持**: 支持 DeepSeek-V3、GLM-5、Kimi-K2.5、Qwen-Max 等
- **真实评估**: 使用 MMMU、VSR 等真实数据集进行 few-shot 评估

## 目录结构

```
AutoFusion_Advanced/
├── phase5_llm_rl/           # Phase 5: LLM-Driven RL Loop (核心)
│   ├── src/                 # 源代码
│   │   ├── llm_backend.py   # LLM 后端抽象
│   │   ├── rl_controller.py # RL 控制器
│   │   ├── prompt_builder.py# Prompt 构建
│   │   ├── few_shot_db.py   # Few-Shot 数据库
│   │   └── main_loop.py     # 主循环
│   ├── configs/             # 实验配置
│   └── results/             # 实验结果
│
├── configs/                 # 全局配置
├── docs/                    # 文档
│   ├── plans/               # 实验计划
│   ├── experiments/         # 实验报告
│   └── design/              # 设计文档
│
├── archive/                 # 旧实验归档
│   ├── phase1_3_experiment/ # Phase 1-3
│   ├── phase4_optimization/ # Phase 4
│   └── expv2_evaluation/    # E1-E7 评估
│
├── scripts/                 # 运行脚本
└── run.py                   # 入口文件
```

## 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/Starryyu77/AutoFusion_Advanced.git
cd AutoFusion_Advanced

# 安装依赖
pip install -r requirements.txt
```

### 运行 Phase 5 实验

```bash
cd phase5_llm_rl

# 配置 API Key
export ALIYUN_API_KEY="your-api-key"

# 运行搜索 (DeepSeek-V3)
python src/main_loop.py --config configs/exp_deepseek.yaml --output-dir results/exp_deepseek

# 或使用脚本
bash scripts/run_experiment.sh deepseek 0
```

### 多模型对比

```bash
# 同时运行多个模型 (不同 GPU)
bash scripts/run_all_models.sh
```

## 实验结果

### Phase 5: LLM-Driven RL Loop

| 模型 | Best Reward | 架构类型 | 编译成功率 |
|------|-------------|----------|-----------|
| **GLM-5** | **2.797** | gated | 6% |
| DeepSeek-V3 | 2.796 | attention | **24%** |
| Kimi-K2.5 | 2.539 | attention | 2% |
| Qwen-Max | 0.500 | - | 0% |

### 最佳架构 (GLM-5)

```python
{
    'type': 'gated',
    'hidden_dim': 128,
    'num_layers': 2
}
```

## 核心组件

### 1. LLM Backend

支持多种 LLM 后端，通过统一接口调用：

```python
from llm_backend import LLMBackend

backend = LLMBackend.create("aliyun", model="glm-5", api_key="...")
response = backend.generate(prompt)
```

### 2. RL Controller

LLM 作为搜索策略，基于历史反馈提出新架构：

```python
controller = RLController(llm_backend, prompt_builder, few_shot_db)
proposal = controller.propose(context)
```

### 3. Prompt Builder

动态构建 Prompt，支持约束注入和 Few-Shot 示例：

```python
prompt = prompt_builder.build(
    task="design_efficient_fusion",
    constraints=Constraints(max_flops=10M),
    few_shot_examples=db.get_top_k(3),
    history=controller.get_recent_history(5)
)
```

## 服务器配置

- **Host**: `gpu43.dynip.ntu.edu.sg`
- **GPU**: 4 × NVIDIA RTX A5000 (24GB)
- **项目路径**: `/usr1/home/s125mdg43_10/AutoFusion_Advanced/`

## 文档

- [Phase 5 计划](docs/plans/PHASE5_RL_LOOP_PLAN.md)
- [Phase 5 进度](docs/plans/PHASE5_PROGRESS.md)
- [实验总结](docs/EXPERIMENTS_SUMMARY.md)

## 旧实验归档

- `archive/phase1_3_experiment/`: Phase 1-3 架构搜索实验
- `archive/phase4_optimization/`: Phase 4 LLM 搜索
- `archive/expv2_evaluation/`: E1-E7 完整评估实验

## 参考资料

- [FiLM: Visual Reasoning with a Conditioned Feature Layer](https://arxiv.org/abs/1709.07871)
- [MMMU: A Massive Multi-discipline Multimodal Understanding Benchmark](https://arxiv.org/abs/2311.16502)

## License

MIT License

---

*Last Updated: 2026-02-28*
*Status: Phase 5 In Progress*