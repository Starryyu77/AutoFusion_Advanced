# Phase 5 实验进度追踪

## 📋 项目概述

**实验名称**: Phase 5 - LLM-Driven Reinforcement Learning Loop  
**目标**: 将 LLM 从静态代码生成器升级为全局控制器，实现真正的闭环 NAS  
**服务器路径**: `gpu43.dynip.ntu.edu.sg:~/AutoFusion_Advanced/phase5_llm_rl/`

---

## ✅ 已完成工作 (2026-02-27)

### 核心框架模块 (已完成 ✓)

| 模块 | 文件 | 代码行数 | 状态 |
|------|------|----------|------|
| LLM Backend | `llm_backend.py` | ~280 | ✅ |
| Prompt Builder | `prompt_builder.py` | ~120 | ✅ |
| RL Controller | `rl_controller.py` | ~390 | ✅ |
| Few-Shot DB | `few_shot_db.py` | ~290 | ✅ |
| Constraint Manager | `constraint_manager.py` | ~280 | ✅ |
| Main Loop | `main_loop.py` | ~390 | ✅ |
| Configs | `config.yaml`, `deepseek_config.yaml` | ~140 | ✅ |
| Scripts | `run_local.sh`, `submit_cluster.sh` | ~100 | ✅ |
| **总计** | | **~2000** | ✅ |

### 关键特性实现

1. **✅ LLM 后端可插拔**: 支持 DeepSeek / MiniMax / OpenAI / Mock
2. **✅ 动态 Few-Shot 选择**: 支持 random / top_k / similarity / diversity 策略
3. **✅ 约束注入**: FLOPs / 延迟 / 参数量 / 准确率约束
4. **✅ 闭环反馈**: Controller 根据历史反馈自主改进
5. **✅ 搜索策略切换**: explore / exploit / refine 三种模式
6. **✅ 状态保存/恢复**: 支持断点续传

---

## 🔄 服务器实验状态

### 当前运行状态

| 实验 | 状态 | 结果 |
|------|------|------|
| Phase 4 MockGenerator | ✅ 完成 | Arch #3: 50% acc, 3.55M FLOPs |
| Phase 4 LLM | ✅ 完成 | 37.5% acc, 17.71M FLOPs |
| **Phase 5** | 🆕 **新建** | **待运行** |

### 服务器资源

```
Host: gpu43.dynip.ntu.edu.sg
GPU: 4x NVIDIA RTX A5000 (24GB each)
Project: /usr1/home/s125mdg43_10/AutoFusion_Advanced/
```

---

## 📁 文件结构

```
phase5_llm_rl/
├── src/
│   ├── __init__.py              ✅ 模块初始化
│   ├── llm_backend.py           ✅ LLM 后端抽象
│   ├── prompt_builder.py        ✅ 动态 Prompt 构建
│   ├── rl_controller.py         ✅ RL 控制器核心
│   ├── few_shot_db.py          ✅ Few-Shot 数据库
│   ├── constraint_manager.py    ✅ 约束管理
│   └── main_loop.py             ✅ 主循环
├── configs/
│   ├── config.yaml              ✅ MiniMax 配置
│   └── deepseek_config.yaml     ✅ DeepSeek 配置
├── scripts/
│   ├── run_local.sh             ✅ 本地运行脚本
│   └── submit_cluster.sh        ✅ 集群提交脚本
├── results/                     📂 实验结果目录
└── docs/                        📂 文档目录
```

---

## 🚀 下一步行动计划

### Step 1: 测试运行
- [ ] SSH 登录服务器
- [ ] 运行 `bash scripts/run_local.sh configs/config.yaml`
- [ ] 验证 Mock 模式正常工作
- [ ] 检查日志输出

### Step 2: API 配置
- [ ] 确认 MiniMax API Key 可用
- [ ] 或切换到 DeepSeek API
- [ ] 测试真实 LLM 调用

### Step 3: 集成评估器
- [ ] 复用 Phase 4 的 ImprovedRealDataFewShotEvaluator
- [ ] 集成 VSR / MMMU 数据集
- [ ] 测试完整评估流程

### Step 4: 完整实验
- [ ] 运行 50 iterations 搜索
- [ ] 对比 MiniMax vs DeepSeek 效果
- [ ] 分析 Few-Shot 动态选择效果
- [ ] 对比 Phase 4 结果

---

## 📊 预期结果

| 指标 | 目标 | Phase 4 基线 |
|------|------|--------------|
| MMMU 准确率 | > 50% | 46% (FiLM) |
| FLOPs | < 5M | 6.29M (FiLM) |
| 发现架构数 | > 5 | 3 (有效) |
| 搜索效率 | 提升 20% | - |

---

## 📝 变更日志

| 日期 | 变更 | 状态 |
|------|------|------|
| 2026-02-27 | 创建 Phase 5 框架代码 | ✅ 完成 |
| 2026-02-27 | 上传到服务器 | ✅ 完成 |
| 2026-02-27 | 创建本地进度文档 | ✅ 完成 |
| - | 运行首次测试 | ⏳ 待执行 |
| - | 集成真实评估器 | ⏳ 待执行 |
| - | 完整实验 | ⏳ 待执行 |

---

## 🔗 相关链接

- **Phase 5 规划文档**: `~/AutoFusion_Advanced/PHASE5_RL_LOOP_PLAN.md`
- **Phase 5 代码目录**: `~/AutoFusion_Advanced/phase5_llm_rl/`
- **Phase 4 结果**: `~/AutoFusion_Advanced/phase4_optimization/results/`
- **项目 README**: `~/AutoFusion_Advanced/README.md`

---

## 💡 快速命令

```bash
# SSH 登录
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg

# 进入目录
cd ~/AutoFusion_Advanced/phase5_llm_rl

# 本地测试（Mock 模式）
bash scripts/run_local.sh configs/config.yaml

# 集群提交
sbatch scripts/submit_cluster.sh configs/config.yaml

# 查看日志
tail -f results/raw/*.log
```

---

*Last Updated: 2026-02-27*  
*Status: Framework Ready for Testing*

## 🆕 更新 (2026-02-27 晚)

### 实验方案确认

**已确认**: 跳过 API 对比实验，专注于：
1. ✅ 跑通流程验证
2. ✅ Baseline 对比
3. ✅ 模型规模影响（阿里云百炼内部）

### 新增配置文件

| 文件 | 实验 | 说明 |
|------|------|------|
| `exp1_quick_test.yaml` | 实验 1 | 快速验证（10 iterations） |
| `exp2_b1_deepseek.yaml` | 实验 2-B1 | DeepSeek-V3 |
| `exp2_b2_qwen_max.yaml` | 实验 2-B2 | Qwen-Max |
| `exp2_b3_qwen_plus.yaml` | 实验 2-B3 | Qwen-Plus |
| `exp2_b4_qwen_turbo.yaml` | 实验 2-B4 | Qwen-Turbo |

### API Key 已配置
- 阿里云百炼 API Key: `sk-fa81e2...` ✅
- 支持模型: DeepSeek-V3, Qwen-Max/Plus/Turbo

### 下一步（等待执行）

**立即可以执行**:
```bash
# SSH 登录服务器
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
cd ~/AutoFusion_Advanced/phase5_llm_rl

# 运行实验 1（快速验证）
bash scripts/run_local.sh configs/exp1_quick_test.yaml
```

**实验 1 成功后并行运行实验 2**:
```bash
# 4 个 GPU 同时运行
CUDA_VISIBLE_DEVICES=0 bash scripts/run_local.sh configs/exp2_b1_deepseek.yaml &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_local.sh configs/exp2_b2_qwen_max.yaml &
CUDA_VISIBLE_DEVICES=2 bash scripts/run_local.sh configs/exp2_b3_qwen_plus.yaml &
CUDA_VISIBLE_DEVICES=3 bash scripts/run_local.sh configs/exp2_b4_qwen_turbo.yaml &
```

---

*Last Updated: 2026-02-27 晚*
*Status: Ready for Experiment 1*

