# Auto-Fusion Phase 5: LLM-Driven Reinforcement Learning Loop

> **项目定位**: 将 LLM 从静态代码生成器升级为全局控制器(Controller)，实现真正的 LLM-Driven NAS

---

## 一、系统架构概览

### 1.1 核心创新

| 旧架构 (Phase 1-4) | 新架构 (Phase 5) |
|-------------------|-----------------|
| LLM = 代码生成器 | **LLM = 控制器 + 决策者** |
| RL Controller = 神经网络策略 | **LLM Controller = 语言模型策略** |
| 单向生成，无反馈 | **闭环反馈，自主进化** |
| 静态 Few-Shot 示例 | **动态上下文学习** |

### 1.2 数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM-Driven RL Closed Loop                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────────┐                      ┌──────────────────┐           │
│    │   RLController   │◄────── Reward ◄──────│    Evaluator     │           │
│    │    (LLM Agent)   │                      │   (GPU/CPU)      │           │
│    └────────┬─────────┘                      └────────▲─────────┘           │
│             │                                         │                      │
│             │ Architecture Code                       │ Metrics              │
│             │ (self-modifying)                        │ (accuracy, FLOPs)    │
│             ▼                                         │                      │
│    ┌──────────────────┐                              │                      │
│    │  PromptBuilder   │                              │                      │
│    │  + Few-Shot DB   │                              │                      │
│    │  + Constraints   │──────────────────────────────┘                      │
│    └──────────────────┘                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心模块设计

### 2.1 模块职责划分

| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **RLController** | LLM 驱动的架构搜索策略 | 历史反馈、约束条件 | 新架构代码 |
| **PromptBuilder** | 动态构建 Prompt | 搜索状态、Few-Shot 示例 | 完整 Prompt |
| **FewShotDatabase** | 管理优秀架构示例 | 评估结果 | 动态选择的示例 |
| **Evaluator** | 执行评估并提取指标 | 架构代码 | 准确率、FLOPs、延迟 |
| **RewardFunction** | 计算奖励信号 | 评估指标 | 标量/向量奖励 |
| **ConstraintManager** | 管理工程约束 | 用户配置 | 约束条件注入 |
| **LLMBackend** | 抽象 LLM API 调用 | Prompt | 生成的代码/决策 |

### 2.2 核心类接口

```python
# ===== RLController (LLM as Controller) =====
class RLController:
    """LLM 驱动的强化学习控制器"""
    
    def propose(self, context: SearchContext) -> ArchitectureProposal:
        """基于历史反馈提出新架构"""
        pass
    
    def update(self, feedback: Feedback) -> None:
        """根据评估反馈更新策略（Few-Shot DB 更新）"""
        pass

# ===== PromptBuilder =====
class PromptBuilder:
    """动态 Prompt 构建，支持约束注入和 Few-Shot 示例"""
    
    def build(
        self,
        task: str,
        constraints: Constraints,
        few_shot_examples: List[Example],
        history: List[Feedback],
    ) -> str:
        pass

# ===== Evaluator =====
class Evaluator:
    """评估器抽象，支持多数据集"""
    
    def evaluate(
        self,
        code: str,
        dataset: str = "mmmu",
        num_shots: int = 32,
    ) -> EvaluationResult:
        pass

# ===== RewardFunction =====
class RewardFunction:
    """多目标奖励函数"""
    
    def calculate(self, result: EvaluationResult) -> Reward:
        pass

# ===== LLMBackend (可插拔) =====
class LLMBackend(ABC):
    """LLM 后端抽象，支持快速切换"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class DeepSeekBackend(LLMBackend): ...
class MiniMaxBackend(LLMBackend): ...
class OpenAIBackend(LLMBackend): ...
```

---

## 三、关键创新设计

### 3.1 LLM 作为 Controller（而非 Generator）

**旧设计问题**：
```python
# 旧架构：LLM 只是代码生成器
controller = PPOController()  # 神经网络策略
architecture_desc = controller.propose()  # 提出架构描述
code = llm_generator.generate(architecture_desc)  # LLM 只是翻译
```

**新设计**：
```python
# 新架构：LLM 直接作为 Controller
controller = RLController(
    llm_backend=MiniMaxBackend(),
    prompt_builder=PromptBuilder(),
    few_shot_db=FewShotDatabase(),
)
proposal = controller.propose(context)  # LLM 直接输出架构代码
# LLM 看到历史反馈，自主决定如何改进
```

### 3.2 闭环反馈机制

```python
# 每轮迭代，LLM 看到：
context = SearchContext(
    current_iteration=10,
    best_architecture=best_arch,  # 当前最佳架构
    best_reward=0.75,
    recent_feedbacks=[  # 最近 5 次尝试的反馈
        Feedback(arch, accuracy=0.72, flops=15M, reward=0.68),
        Feedback(arch, accuracy=0.75, flops=20M, reward=0.72),
        ...
    ],
    constraints=Constraints(max_flops=10M, max_latency=50ms),
)
```

### 3.3 约束注入到 Prompt

```python
prompt = prompt_builder.build(
    task="design_efficient_fusion",
    constraints=Constraints(
        max_flops=10_000_000,      # 最大 FLOPs
        max_params=50_000_000,     # 最大参数量
        max_latency_ms=50,         # 最大推理延迟
        target_accuracy=0.80,      # 目标准确率
    ),
    few_shot_examples=db.get_top_k(k=3),
    history=controller.get_recent_history(n=5),
)
```

### 3.4 LLM 后端可插拔

```yaml
# config.yaml
llm_backend:
  type: "minimax"  # 可选: deepseek, openai, minimax, anthropic
  model: "MiniMax-Text-01"
  api_key: "${MINIMAX_API_KEY}"
  temperature: 0.7
  max_tokens: 4096
```

---

## 四、服务器实验状态

### 4.1 当前运行状态

| 项目 | 状态 | 详情 |
|------|------|------|
| **Phase 4 MockGenerator** | ✅ 完成 | Arch #3: 50% acc, 3.55M FLOPs (超越 FiLM) |
| **Phase 4 LLM** | ✅ 完成 | 20 iterations, 最佳 37.5% acc, 17.71M FLOPs |
| **GPU 0-3** | 部分占用 | GPU 3 可用于新实验 |

### 4.2 服务器路径

```
/usr1/home/s125mdg43_10/AutoFusion_Advanced/
├── experiment_template/     # 新创建的通用模板
├── phase5_llm_rl/           # 本实验目录 (待创建)
├── experiment/              # 原有框架
└── phase4_optimization/     # Phase 4 结果
```

---

## 五、实施计划

### Phase 5.1: 框架重构（Week 1）

- [ ] 创建 `phase5_llm_rl/` 目录结构
- [ ] 实现 `LLMBackend` 抽象层（支持 DeepSeek/MiniMax/OpenAI）
- [ ] 实现 `RLController` (LLM as Controller)
- [ ] 实现 `PromptBuilder` with constraint injection

### Phase 5.2: 评估器升级（Week 2）

- [ ] 接入 VSR 数据集
- [ ] 接入 MMMU 数据集
- [ ] 实现 few-shot evaluation protocol
- [ ] 优化评估速度（early stopping + caching）

### Phase 5.3: Few-Shot 动态选择（Week 3）

- [ ] 实现 `FewShotDatabase`
- [ ] 基于搜索状态动态选择示例
- [ ] 实现架构相似度计算
- [ ] 自动更新 Top-K 示例库

### Phase 5.4: 闭环实验（Week 4）

- [ ] 运行 50 iterations 闭环搜索
- [ ] 对比 LLM-Controller vs MockGenerator
- [ ] 分析 Few-Shot 动态选择效果
- [ ] 撰写实验报告

---

## 六、预期成果

| 目标 | 指标 | 基线 |
|------|------|------|
| **超越 FiLM** | MMMU > 50% | FiLM: 46% |
| **效率优化** | FLOPs < 5M | FiLM: 6.29M |
| **LLM 智能进化** | 后期迭代 > 前期 | 证明闭环有效 |
| **架构多样性** | 发现 > 5 种有效架构 | - |

---

## 七、变更日志

| 2026-02-27 | 创建 Phase 5 规划文档 | ✅ 完成 |
| 2026-02-27 | 创建 `experiment_template/` 目录 | ✅ 完成 |
| 2026-02-27 | 实现 LLM Backend (DeepSeek/MiniMax/OpenAI) | ✅ 完成 |
| 2026-02-27 | 实现 Prompt Builder | ✅ 完成 |
| 2026-02-27 | 实现 RL Controller | ✅ 完成 |
| 2026-02-27 | 实现 Few-Shot Database | ✅ 完成 |
| 2026-02-27 | 实现 Constraint Manager | ✅ 完成 |
| 2026-02-27 | 实现 Main Loop | ✅ 完成 |
| 2026-02-27 | 创建配置文件和运行脚本 | ✅ 完成 |
| 2026-02-27 | 上传到服务器 | ✅ 完成 |
| - | 运行测试实验 | ⏳ 待执行 |
| - | 集成真实评估器 | ⏳ 待执行 |
| - | 完整 50 iterations 搜索 | ⏳ 待执行 |
---

*Last Updated: 2026-02-27*

---

## 附录 A: 数据集基线 (搜索结果)

### VSR Dataset

| Split | Train | Dev | Test |
|-------|-------|-----|------|
| Random | 7,083 | 1,012 | 2,024 |

**Baseline Accuracy:**
| Model | Random Split | Zero-Shot |
|-------|-------------|-----------|
| Human | 95.4% | 95.4% |
| LXMERT | 72.5% | 63.2% |
| ViLT | 71.0% | 62.4% |

### MMMU Dataset

| Metric | Value |
|--------|-------|
| Total Samples | 11,550 |
| Disciplines | 6 |
| Subjects | 30 |

**Baseline Accuracy (Validation):**
| Model | Accuracy |
|-------|----------|
| Human Expert | 88.6% |
| GPT-4o | 69.1% |
| GPT-4V | 56.8% |
| Random Choice | 26.8% |

**我们的目标**: 超越 50% (当前 FiLM baseline: 46%)

---

## 附录 B: 现有可复用组件

### 接口继承关系

```
BaseController (abstract)
├── PPOController      # Critic-Free variant for NAS
├── GRPOController     # Group-relative normalization
├── GDPOController     # Decoupled normalization
├── EvolutionController # Tournament selection
├── CMAESController    # Covariance adaptation
└── RandomController   # Baseline

BaseGenerator (abstract)
├── CoTGenerator       # Chain-of-thought
├── FewShotGenerator   # Example-based
├── CriticGenerator    # Self-evaluation
├── ShapeGenerator     # Tensor constraints
└── RolePlayGenerator  # Expert persona

BaseEvaluator (abstract)
├── RealDataFewShotEvaluator  # VSR/MMMU/AI2D
└── SurgicalSandboxEvaluator  # Code validation

BaseReward (abstract)
├── MultiObjectiveReward  # Accuracy + Efficiency
└── ExponentialReward     # Sharpened rewards
```

### 核心数据类

| 类名 | 字段 |
|------|------|
| `SearchState` | iteration, best_reward, best_architecture, history |
| `GenerationResult` | code, prompt, metadata, success, error |
| `EvaluationResult` | accuracy, efficiency, compile_success, flops, params, latency |
| `RewardComponents` | accuracy, efficiency, compile_success, complexity |

---

## 附录 C: Few-Shot Prompt 最佳实践

### 示例数量
- **推荐**: 2-3 个示例 (diminishing returns beyond 4-5)
- **多样性**: 覆盖不同架构模式 (REST, GraphQL, WebSocket)

### 约束注入方法

```python
# Method 1: Structured Outputs (JSON Schema)
class ArchitectureSpec(BaseModel):
    pattern: str
    components: list[str]
    flops_budget: int

# Method 2: Delimiter-Based Constraints
"""### OUTPUT FORMAT
## Component: <name>
- Type: <service|repository|controller>
- FLOPs: <estimate>
"""
```

### 动态示例选择

```python
def select_examples(query, example_pool, top_k=3):
    # 1. Semantic similarity
    # 2. Task relevance
    # 3. Complexity match
    return sorted_examples[:top_k]
```

---

*Last Updated: 2026-02-27*
*Status: Planning Phase*