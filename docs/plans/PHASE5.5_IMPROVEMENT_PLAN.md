# Phase 5.5: LLM-Driven NAS 改进计划

**版本**: 1.0
**创建日期**: 2026-02-28
**目标**: 提高 LLM 编译成功率，发现超越 FiLM baseline 的架构

---

## 一、改进目标

### 1.1 核心目标

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 编译成功率 | 24% (最高) | **60%+** |
| 最佳准确率 | ~25% | **50%+** (超越 FiLM 46%) |
| 有效架构数 | 12/50 | **30+/50** |
| FLOPs 效率 | 4.8M | **< 6M** |

### 1.2 测试模型

| 模型 | API 提供商 | 模型 ID |
|------|-----------|---------|
| DeepSeek V3.2 | 阿里云百炼 | `deepseek-v3` |
| GLM-5 | 阿里云百炼 | `glm-5` |
| MiniMax 2.5/2.1 | 阿里云百炼 | `abab6.5s-chat` 或类似 |
| Kimi K2.5 | 阿里云百炼 | `kimi-k2.5` |

---

## 二、改进方案详解

### 2.1 改进 1: Prompt 工程优化 (高优先级)

#### 问题描述
LLM 生成的代码经常有语法错误，导致编译失败。

#### 改进方案

**方案 A: 代码模板约束**

```python
# 定义代码模板，LLM 只需填充关键部分
FUSION_TEMPLATE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    多模态融合模块
    vision_dim: 视觉特征维度 (固定 768)
    language_dim: 语言特征维度 (固定 768)
    """
    def __init__(self, vision_dim=768, language_dim=768, hidden_dim={hidden_dim}, num_layers={num_layers}):
        super().__init__()
        # === 融合逻辑开始 ===
        {fusion_code}
        # === 融合逻辑结束 ===
        
    def forward(self, vision_features, language_features):
        """
        输入:
            vision_features: [batch, 768]
            language_features: [batch, 768]
        输出:
            fused: [batch, hidden_dim]
        """
        {forward_code}
        return fused
'''

# LLM 只需要生成 fusion_code 和 forward_code
```

**方案 B: 结构化输出**

```python
# 要求 LLM 输出 JSON 格式
OUTPUT_FORMAT = '''
{
    "architecture_type": "attention|gated|transformer|mlp",
    "hidden_dim": 64-256,
    "num_layers": 1-4,
    "components": {
        "projection": "linear|mlp|none",
        "fusion": "attention|gating|concat|bilinear",
        "normalization": "layer_norm|batch_norm|none"
    }
}
'''
```

**方案 C: Few-Shot 示例增强**

```python
# 提供 3-5 个高质量示例
FEW_SHOT_EXAMPLES = [
    {
        "description": "Attention-based fusion",
        "code": ATTENTION_CODE,
        "accuracy": 0.46,
        "flops": "6.29M"
    },
    {
        "description": "Gated fusion",
        "code": GATED_CODE,
        "accuracy": 0.42,
        "flops": "4.8M"
    },
    ...
]
```

#### 实施步骤

```
Week 1:
  Day 1-2: 实现 Prompt 改进 (方案 A + B)
  Day 3: 测试新 Prompt 在 DeepSeek 上的效果
  Day 4: 迭代优化
  Day 5: 部署到所有模型
```

#### 预期效果
- 编译成功率: 24% → **50%+**

---

### 2.2 改进 2: 错误反馈机制 (高优先级)

#### 问题描述
编译失败后，错误信息未反馈给 LLM，导致重复错误。

#### 改进方案

```python
class ErrorFeedbackLoop:
    """错误反馈循环"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.error_history = []
    
    def generate_with_feedback(self, llm, initial_prompt, code_validator):
        """带反馈的代码生成"""
        prompt = initial_prompt
        
        for attempt in range(self.max_retries):
            # 1. LLM 生成代码
            code = llm.generate(prompt)
            
            # 2. 验证代码
            is_valid, error = code_validator.validate(code)
            
            if is_valid:
                return code, attempt + 1
            
            # 3. 记录错误
            self.error_history.append({
                "attempt": attempt + 1,
                "error": error,
                "code_snippet": code[:200]
            })
            
            # 4. 构建反馈 prompt
            prompt = f"""
{initial_prompt}

=== 之前的尝试失败 ===
尝试 {attempt + 1}/{self.max_retries}
错误信息: {error}

请修复上述错误，生成正确的代码。
注意：
1. 确保所有括号匹配
2. 确保所有变量在使用前定义
3. 确保 forward 函数返回正确的输出
"""
        
        return None, self.max_retries
```

#### 实施步骤

```
Week 1:
  Day 3-4: 实现 ErrorFeedbackLoop 类
  Day 5: 集成到 main_loop.py
  Day 6-7: 测试并优化
```

#### 预期效果
- 编译成功率额外提升: **10-15%**

---

### 2.3 改进 3: 评估器升级 (高优先级)

#### 问题描述
当前使用 MMMU 32-shot 5 epochs，可能不够稳定。

#### 改进方案

**方案 A: 增加评估配置**

```yaml
# 新评估配置
evaluator:
  # 主配置
  primary:
    dataset: "mmmu"
    num_shots: 64      # 从 32 增加到 64
    train_epochs: 10   # 从 5 增加到 10
    
  # 快速配置 (用于早期迭代)
  quick:
    dataset: "mmmu"
    num_shots: 32
    train_epochs: 3
    
  # 混合数据集
  multi_dataset:
    datasets: ["mmmu", "vsr"]
    weights: [0.7, 0.3]
```

**方案 B: 使用验证集子集**

```python
# 使用更稳定的验证集
MMMU_VALIDATION_SUBSET = [
    "Accounting",      # 相对简单
    "Art",             # 视觉为主
    "Biology",         # 知识密集
]
```

#### 实施步骤

```
Week 2:
  Day 1: 实现 multi-dataset evaluator
  Day 2: 测试不同配置的稳定性
  Day 3: 确定最优配置
  Day 4-5: 部署到搜索流程
```

#### 预期效果
- 评估稳定性提升
- 搜索方向更准确

---

### 2.4 改进 4: 架构模板库 (中优先级)

#### 问题描述
LLM 需要从零生成完整代码，容易出错。

#### 改进方案

```python
# 预定义架构模板
ARCHITECTURE_TEMPLATES = {
    "attention": {
        "description": "Cross-modal attention fusion",
        "template": ATTENTION_TEMPLATE,
        "params": {
            "hidden_dim": [32, 64, 128, 256],
            "num_heads": [1, 2, 4, 8],
            "dropout": [0.0, 0.1, 0.2]
        }
    },
    "gated": {
        "description": "Gating mechanism fusion",
        "template": GATED_TEMPLATE,
        "params": {
            "hidden_dim": [32, 64, 128, 256],
            "gate_type": ["sigmoid", "softmax", "tanh"]
        }
    },
    "transformer": {
        "description": "Transformer-based fusion",
        "template": TRANSFORMER_TEMPLATE,
        "params": {
            "hidden_dim": [64, 128, 256],
            "num_layers": [1, 2, 3],
            "num_heads": [2, 4, 8]
        }
    },
    "mlp": {
        "description": "Simple MLP fusion",
        "template": MLP_TEMPLATE,
        "params": {
            "hidden_dim": [64, 128, 256],
            "num_layers": [1, 2, 3]
        }
    }
}

# LLM 只需要选择模板和参数
PROMPT_FOR_TEMPLATE = """
基于当前搜索状态，选择最优的架构模板和参数：

可用模板:
1. attention - Cross-modal attention fusion
2. gated - Gating mechanism fusion
3. transformer - Transformer-based fusion
4. mlp - Simple MLP fusion

请输出 JSON 格式:
{
    "template": "模板名称",
    "params": {
        "参数名": 参数值
    },
    "reasoning": "选择理由"
}
"""
```

#### 实施步骤

```
Week 2-3:
  Day 1-2: 定义 4 种架构模板
  Day 3-4: 实现模板选择 Prompt
  Day 5: 测试模板模式搜索
  Day 6-7: 与自由生成模式对比
```

#### 预期效果
- 编译成功率: **80%+**
- 搜索更稳定可控

---

### 2.5 改进 5: 混合搜索策略 (中优先级)

#### 问题描述
纯 LLM 搜索不稳定，需要结合传统方法。

#### 改进方案

```python
class HybridSearchStrategy:
    """混合搜索策略"""
    
    def __init__(self, llm_controller, traditional_optimizer):
        self.llm = llm_controller
        self.optimizer = traditional_optimizer
        
    def search_iteration(self, context):
        """单次搜索迭代"""
        
        # Step 1: LLM 选择架构类型 (高层决策)
        arch_type = self.llm.select_architecture_type(context)
        
        # Step 2: 传统优化器搜索超参数 (低层优化)
        best_params = self.optimizer.optimize(
            arch_type=arch_type,
            search_space=ARCHITECTURE_TEMPLATES[arch_type]["params"],
            evaluator=context.evaluator
        )
        
        # Step 3: LLM 精调架构细节 (微调)
        refined_arch = self.llm.refine_architecture(
            arch_type=arch_type,
            params=best_params,
            feedback=context.history
        )
        
        return refined_arch
```

#### 实施步骤

```
Week 3:
  Day 1-2: 实现 HybridSearchStrategy
  Day 3-4: 集成 Evolution 优化器
  Day 5-7: 测试混合搜索效果
```

#### 预期效果
- 结合 LLM 创造力和传统 NAS 稳定性

---

## 三、实验计划

### 3.1 实验时间表

**每次只使用 2 块 GPU，分批次测试**

```
Week 1 (2026-03-01 ~ 2026-03-07):
├── Day 1-2: Prompt 改进实现
├── Day 3-4: 错误反馈机制
├── Day 5: 部署测试
└── Day 6-7: 第一批测试 (2 GPUs)
    ├── GPU 0: DeepSeek V3.2 (100 iterations)
    └── GPU 1: GLM-5 (100 iterations)

Week 2 (2026-03-08 ~ 2026-03-14):
├── Day 1-4: 第二批测试 (2 GPUs)
│   ├── GPU 0: MiniMax 2.5 (100 iterations)
│   └── GPU 1: Kimi K2.5 (100 iterations)
├── Day 5-6: 结果汇总分析
└── Day 7: 评估器升级部署

Week 3 (2026-03-15 ~ 2026-03-21):
├── Day 1-2: 架构模板实现
├── Day 3-4: 混合搜索实现
├── Day 5-6: 第三轮测试 (最优模型)
│   ├── GPU 0: 最优模型 A (100 iterations)
│   └── GPU 1: 最优模型 B (100 iterations)
└── Day 7: 最终报告
```

### 3.2 单模型测试流程

```bash
# 1. 配置文件准备
cat > configs/exp_${MODEL}_v2.yaml << EOF
experiment:
  name: "exp_${MODEL}_v2"
  max_iterations: 100
  
llm:
  model: "${MODEL_ID}"
  
evaluator:
  dataset: "mmmu"
  num_shots: 64
  train_epochs: 10
  
# 改进配置
improvements:
  use_template: true
  error_feedback: true
  max_retries: 3
EOF

# 2. 启动实验
python src/main_loop.py --config configs/exp_${MODEL}_v2.yaml

# 3. 监控进度
tail -f results/exp_${MODEL}_v2/run.log

# 4. 分析结果
python scripts/analyze_results.py --exp exp_${MODEL}_v2
```

### 3.3 评估指标

| 指标 | 计算方法 | 目标 |
|------|----------|------|
| 编译成功率 | 成功编译数 / 总生成数 | > 60% |
| 最佳准确率 | 成功架构的最高准确率 | > 50% |
| 平均准确率 | 所有成功架构的平均准确率 | > 35% |
| 效率指标 | FLOPs < 6M 的架构比例 | > 50% |
| 搜索效率 | 发现有效架构的迭代次数 | < 30 |

---

## 四、资源配置

### 4.1 GPU 分配

**每次只使用 2 块 GPU，分批次测试**

| 时间段 | GPU 0 | GPU 1 | 状态 |
|--------|-------|-------|------|
| Week 1 Day 6-7 | DeepSeek V3.2 | GLM-5 | 第一批 |
| Week 2 Day 1-4 | MiniMax 2.5 | Kimi K2.5 | 第二批 |
| Week 3 Day 5-6 | 最优模型 A | 最优模型 B | 第三轮 |

**说明**: 
- 每批次同时运行 2 个模型，每个模型 100 iterations
- 等一批完成后，再启动下一批
- GPU 2、GPU 3 保留备用或供其他任务使用
### 4.2 API 配额

| 模型 | 预估调用次数 | 预估费用 |
|------|-------------|----------|
| DeepSeek V3.2 | ~500 次 | ~¥50 |
| GLM-5 | ~500 次 | ~¥50 |
| MiniMax 2.5 | ~500 次 | ~¥50 |
| Kimi K2.5 | ~500 次 | ~¥50 |
| **总计** | ~2000 次 | ~¥200 |

---

## 五、风险与应对

### 5.1 风险列表

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| API 限流 | 中 | 高 | 准备多个 API key，错峰调用 |
| 编译成功率仍低 | 中 | 高 | 增加模板约束，减少自由度 |
| 评估不稳定 | 中 | 中 | 增加评估次数，取平均 |
| GPU 资源不足 | 低 | 高 | 使用云 GPU 备用 |
| 模型 API 变更 | 低 | 中 | 监控 API 状态，及时调整 |

### 5.2 回滚方案

如果改进效果不佳：
1. 回退到 Phase 5 原始配置
2. 使用 Phase 3 发现的最佳架构
3. 直接使用 FiLM baseline

---

## 六、成功标准

### 6.1 最低标准

- [ ] 编译成功率 > 40% (至少 2 个模型达到)
- [ ] 发现至少 1 个准确率 > 40% 的架构
- [ ] 完成 4 个模型的对比测试

### 6.2 理想标准

- [ ] 编译成功率 > 60%
- [ ] 最佳准确率 > 50% (超越 FiLM)
- [ ] FLOPs < 6M
- [ ] 4 个模型中至少 2 个发现有效架构

### 6.3 验收测试

```python
def validate_improvement():
    """验证改进效果"""
    
    # 1. 编译成功率测试
    success_rates = {}
    for model in ["deepseek", "glm5", "minimax", "kimi"]:
        total, success = count_compilation(model)
        success_rates[model] = success / total
    
    # 2. 准确率测试
    best_accuracies = {}
    for model in ["deepseek", "glm5", "minimax", "kimi"]:
        best_acc = get_best_accuracy(model)
        best_accuracies[model] = best_acc
    
    # 3. 判断是否达标
    min_success_rate = min(success_rates.values())
    max_accuracy = max(best_accuracies.values())
    
    print(f"最低编译成功率: {min_success_rate:.1%}")
    print(f"最高准确率: {max_accuracy:.1%}")
    
    return min_success_rate > 0.4 and max_accuracy > 0.4
```

---

## 七、交付物

### 7.1 代码交付

- [ ] `src/prompt_builder_v2.py` - 改进的 Prompt 构建
- [ ] `src/error_feedback.py` - 错误反馈机制
- [ ] `src/architecture_templates.py` - 架构模板库
- [ ] `src/hybrid_search.py` - 混合搜索策略
- [ ] `configs/exp_*_v2.yaml` - 新配置文件

### 7.2 文档交付

- [ ] `docs/experiments/PHASE5.5_RESULTS.md` - 实验结果报告
- [ ] `docs/experiments/PHASE5.5_COMPARISON.md` - 模型对比报告
- [ ] 更新 `docs/PROJECT_STATUS.md`

### 7.3 数据交付

- [ ] `results/exp_*_v2/` - 各模型实验结果
- [ ] `results/phase5.5_summary.json` - 汇总数据

---

## 八、下一步行动

### 立即执行 (Day 1)

1. **创建改进代码目录**
   ```bash
   mkdir -p phase5_llm_rl/src/v2
   ```

2. **实现 Prompt 改进**
   - 编辑 `src/prompt_builder_v2.py`
   - 添加代码模板约束

3. **准备配置文件**
   - 创建 4 个模型的配置文件

### 本周目标 (Week 1)

- [ ] 完成 Prompt 改进实现
- [ ] 完成错误反馈机制
- [ ] DeepSeek V3.2 测试完成

---

*计划创建: 2026-02-28*
*计划执行: 2026-03-01 开始*
*预计完成: 2026-03-21*