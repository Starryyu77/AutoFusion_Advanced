# 评估器 V2 设计方案：真实数据 Few-Shot 验证

**Version**: 2.0
**Date**: 2026-02-11
**Status**: 待确认

---

## 1. 设计目标

### 核心问题
当前 SurgicalSandboxEvaluator 使用**模拟数据**（随机特征），无法反映真实任务性能。

### 新目标
构建 **RealDataFewShotEvaluator**，通过真实数据集的 few-shot 验证，找到：
- **最佳数据集**：哪个数据集最能区分架构好坏？
- **最佳训练深度**：1/3/5/10 epochs 哪个最经济有效？
- **架构适配性**：不同架构类型在各数据集上的表现

---

## 2. 新评估器架构

### 2.1 类层次结构

```
BaseEvaluator (抽象基类)
    ├── SurgicalSandboxEvaluator (模拟数据，用于快速筛选)
    └── RealDataFewShotEvaluator (真实数据，用于精确评估) ⭐ NEW
            └── MultiDatasetEvaluator (支持多数据集对比) ⭐ NEW
```

### 2.2 RealDataFewShotEvaluator 设计

```python
class RealDataFewShotEvaluator(BaseEvaluator):
    """
    真实数据 Few-Shot 评估器

    特点:
    1. 使用真实数据集 (MMMU/VSR/MathVista/AI2D)
    2. Few-shot 学习 (k=16/32/64 shots)
    3. 冻结主干，只训练融合层
    4. 支持可变训练深度
    """

    def __init__(self, config: Dict[str, Any]):
        # 数据集配置
        self.dataset_name = config['dataset']  # 'mmmu', 'vsr', 'mathvista', 'ai2d'
        self.num_shots = config.get('num_shots', 16)  # few-shot 样本数

        # 训练深度配置 (关键参数)
        self.train_epochs = config.get('train_epochs', 5)
        self.warmup_epochs = config.get('warmup_epochs', 1)

        # 数据加载
        self.data_loader = self._build_data_loader()

    def evaluate(self, code: str) -> EvaluationResult:
        # 1. 编译代码
        # 2. 加载预训练 MLLM (冻结)
        # 3. 插入生成的融合模块
        # 4. Few-shot 训练 (只训融合层)
        # 5. 在验证集上评估
        # 6. 返回 metrics
```

### 2.3 关键组件

#### A. 数据加载器 (DatasetLoader)

```python
class DatasetLoader:
    """统一的数据集加载接口"""

    SUPPORTED_DATASETS = ['mmmu', 'vsr', 'mathvista', 'ai2d']

    def __init__(self, dataset_name: str, num_shots: int = 16):
        self.dataset_name = dataset_name
        self.num_shots = num_shots

    def load(self) -> Tuple[DataLoader, DataLoader]:
        """
        返回: (train_loader, val_loader)
        train_loader: few-shot 训练数据 (num_shots per class)
        val_loader: 完整验证集
        """
        if self.dataset_name == 'mmmu':
            return self._load_mmmu()
        elif self.dataset_name == 'vsr':
            return self._load_vsr()
        elif self.dataset_name == 'mathvista':
            return self._load_mathvista()
        elif self.dataset_name == 'ai2d':
            return self._load_ai2d()
```

#### B. Few-Shot 采样器 (FewShotSampler)

```python
class FewShotSampler:
    """
    Few-Shot 采样策略

    支持:
    - Balanced: 每个类别样本数相同
    - Stratified: 按比例采样
    - Random: 完全随机
    """

    def sample(self, dataset, num_shots: int, strategy: str = 'balanced'):
        if strategy == 'balanced':
            return self._balanced_sample(dataset, num_shots)
        elif strategy == 'stratified':
            return self._stratified_sample(dataset, num_shots)
```

#### C. 训练协议 (TrainingProtocol)

```python
class TrainingProtocol:
    """
    Few-Shot 训练协议

    冻结主干网络，只训练融合层
    """

    def __init__(self, epochs: int, lr: float = 1e-4):
        self.epochs = epochs
        self.lr = lr

    def train(self, model, train_loader, val_loader=None):
        """
        训练流程:
        1. 冻结所有参数
        2. 解冻融合层参数
        3. 训练指定 epochs
        4. 返回验证准确率
        """
        # 冻结主干
        for param in model.backbone.parameters():
            param.requires_grad = False

        # 只优化融合层
        optimizer = Adam(model.fusion_module.parameters(), lr=self.lr)

        # 训练
        for epoch in range(self.epochs):
            # ...

        return best_val_accuracy
```

---

## 3. 验证实验设计

### 3.1 实验目标

找到最佳 (数据集, 训练深度, 架构) 组合，使得：
- **Ranking Correlation**: 与 Full Training (100 epochs) 的排名相关性最高
- **Discriminative Power**: 能区分好坏架构的能力最强
- **Cost Efficiency**: 性价比最高 (时间 vs 准确度)

### 3.2 实验矩阵

#### 实验 1: 数据集选择 (Dataset Selection)

**设计**:
```
固定: 训练深度=5 epochs, 架构=8个代表性架构
变量: 数据集 ∈ {MMMU, VSR, MathVista, AI2D}

评估指标:
- 与 Full Training 的排名相关性 (Kendall's τ)
- 架构区分度 (好架构 vs 差架构的 gap)
- 评估时间
```

**预期结果**:
- 确定哪个数据集最能预测最终性能
- 可能需要组合多个数据集

#### 实验 2: 训练深度校准 (Training Depth Calibration)

**设计**:
```
固定: 数据集=实验1最佳, 架构=8个代表性架构
变量: 训练深度 ∈ {1, 3, 5, 10, 20} epochs

评估指标:
- 与 Full Training 的相关性
- 训练时间
- 收敛稳定性 (不同 seed 的 variance)
```

**预期结果**:
- 找到"甜点"训练深度 (性价比高)
- 可能采用自适应策略 (先 1 epoch 筛选，再 5 epoch 精修)

#### 实验 3: 架构适配性 (Architecture Adaptivity)

**设计**:
```
固定: 数据集=实验1最佳, 训练深度=实验2最佳
变量: 架构类型 ∈ {Attention, Conv, Transformer, MLP, Hybrid}

评估指标:
- 不同架构在各数据集上的表现一致性
- 某些数据集是否对特定架构有偏见
```

**预期结果**:
- 识别评估器偏见
- 为不同架构类型推荐最佳评估配置

### 3.3 验证流程

```
Step 1: 准备 8 个代表性架构
    ├─ 来自 Phase 2.1 的 Evolution Top-4
    ├─ 来自 Phase 2.1 的 PPO Top-3
    └─ 1 个 Random 基线

Step 2: Full Training 基准 (Ground Truth)
    └─ 每个架构跑 100 epochs (或在完整数据集上训练)
    └─ 记录排名 R_full

Step 3: 网格搜索验证配置
    ├─ 4 数据集 × 5 训练深度 = 20 种配置
    └─ 每种配置评估 8 个架构
    └─ 计算与 R_full 的相关性

Step 4: 选择最佳配置
    └─ 综合相关性、时间、稳定性
    └─ 输出推荐配置
```

---

## 4. 基础设施修改

### 4.1 文件结构

```
experiment/
├── evaluators/
│   ├── __init__.py
│   ├── base.py                      # BaseEvaluator
│   ├── surgical_sandbox.py          # 模拟数据评估器
│   ├── real_data_evaluator.py       # ⭐ NEW: 真实数据评估器
│   └── multi_dataset_evaluator.py   # ⭐ NEW: 多数据集对比
│
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py            # ⭐ NEW: 统一数据加载
│   ├── few_shot_sampler.py          # ⭐ NEW: few-shot 采样
│   └── datasets/                    # ⭐ NEW: 数据集目录
│       ├── mmmu_loader.py
│       ├── vsr_loader.py
│       ├── mathvista_loader.py
│       └── ai2d_loader.py
│
└── protocols/
    ├── __init__.py
    └── training_protocol.py         # ⭐ NEW: 训练协议
```

### 4.2 配置更新

```yaml
# configs/evaluator_real.yaml
evaluator:
  type: real_data_few_shot

  # 数据集选择
  dataset: mmmu  # 或 'vsr', 'mathvista', 'ai2d', 'multi'

  # Few-shot 配置
  num_shots: 16  # 16/32/64
  shot_strategy: balanced  # balanced/stratified/random

  # 训练深度 (关键参数)
  train_epochs: 5  # 1/3/5/10/20
  warmup_epochs: 1

  # 优化配置
  learning_rate: 1e-4
  batch_size: 4  # few-shot 用小 batch

  # 主干网络 (冻结)
  backbone:
    name: clip-vit-l-14  # 或 'blip', 'llava'
    freeze: true

  # 验证协议
  validation:
    metric: accuracy
    patience: 3  # 早停耐心值
```

### 4.3 数据准备脚本

```bash
# scripts/download_datasets.sh
#!/bin/bash
# 下载所有验证数据集

echo "Downloading MMMU..."
python -c "from datasets import load_dataset; load_dataset('MMMU/MMMU')"

echo "Downloading VSR..."
python -c "from datasets import load_dataset; load_dataset('cambridgeltl/vsr_random')"

echo "Downloading MathVista..."
python -c "from datasets import load_dataset; load_dataset('AI4Math/MathVista')"

echo "Downloading AI2D..."
python -c "from datasets import load_dataset; load_dataset('lmms-lab/AI2D')"

echo "All datasets downloaded!"
```

---

## 5. 更新后的实验计划

### Phase 2.5: 评估器验证与校准 (NEW)

#### 2.5.1: 数据集选择实验
- **目标**: 确定最佳验证数据集
- **配置**: 8 架构 × 4 数据集 × 5 epochs
- **输出**: 数据集排名相关性对比
- **时间**: 2-3 天

#### 2.5.2: 训练深度校准实验
- **目标**: 确定最佳训练深度
- **配置**: 8 架构 × 1 数据集 × 5 深度
- **输出**: 深度-相关性曲线
- **时间**: 2-3 天

#### 2.5.3: 架构适配性实验
- **目标**: 验证评估器对各类型的公平性
- **配置**: 5 类型 × 3 seeds × 最佳配置
- **输出**: 架构-数据集热力图
- **时间**: 1-2 天

#### 2.5.4: 最终配置确定
- **目标**: 输出推荐配置
- **输出**: `configs/evaluator_recommended.yaml`
- **时间**: 0.5 天

### 完整流程

```
Phase 0/0.5: ✅ 完成
Phase 2.1:   ✅ 完成 (算法对比)
Phase 2.5:   ⏳ 评估器验证 (新增，7-9天)
  ├─ 2.5.1: 数据集选择
  ├─ 2.5.2: 训练深度校准
  ├─ 2.5.3: 架构适配性
  └─ 2.5.4: 最终配置
Phase 1:     ⏳ Prompt 对比 (使用验证后的评估器)
Phase 3:     📋 架构发现 (使用验证后的评估器)
```

---

## 6. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| 数据集下载失败 | High | 提前准备，使用 HuggingFace datasets 镜像 |
| 显存不足 (Few-shot 也需加载大模型) | High | 使用梯度累积，减小 batch_size |
| 数据集标签不一致 | Medium | 统一标签格式，添加数据预处理层 |
| 训练时间过长 | Medium | 并行评估 (4 GPUs 同时跑 4 数据集) |

---

## 7. 下一步行动

待确认后：

1. **基础设施开发** (2-3 天)
   - 实现 `RealDataFewShotEvaluator`
   - 实现 `DatasetLoader` 和 `FewShotSampler`
   - 实现 `TrainingProtocol`

2. **数据准备** (并行，1-2 天)
   - 下载 MMMU/VSR/MathVista/AI2D
   - 验证数据格式
   - 创建数据缓存

3. **实验运行** (7-9 天)
   - 运行 2.5.1/2.5.2/2.5.3
   - 生成对比报告
   - 确定最终配置

4. **更新下游实验**
   - Phase 1 使用新评估器
   - Phase 3 使用新评估器

---

## 8. 预期输出

### 技术产出
- `RealDataFewShotEvaluator` 类
- `configs/evaluator_recommended.yaml`
- 评估器验证报告 (含相关性分析)

### 实验产出
- 最佳数据集推荐
- 最佳训练深度推荐
- 架构适配性报告

---

*Design Date: 2026-02-11*
*Status: 待确认*
