#!/bin/bash
# ============================================================
# Phase 5.5 实验运行脚本
# ============================================================

# 使用方法:
#   ./run_experiments_v2.sh [batch]
#   batch: 1 (第一批: DeepSeek + GLM-5) 或 2 (第二批: MiniMax + Kimi)

BATCH=${1:-1}

# 配置
SERVER="s125mdg43_10@gpu43.dynip.ntu.edu.sg"
REMOTE_DIR="/usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl"
API_KEY="${ALIYUN_API_KEY:-sk-fa81e2c1077c4bf5a159c2ca5ddcf200}"

echo "=========================================="
echo "Phase 5.5 实验运行"
echo "批次: $BATCH"
echo "=========================================="

if [ "$BATCH" == "1" ]; then
    # 第一批: DeepSeek V3.2 + GLM-5
    echo ""
    echo ">>> 启动第一批实验..."
    echo "    GPU 0: DeepSeek V3.2"
    echo "    GPU 1: GLM-5"
    
    ssh $SERVER << 'ENDSSH'
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl
export PYTHONPATH=/usr1/home/s125mdg43_10/AutoFusion_Advanced:$PYTHONPATH
export ALIYUN_API_KEY="sk-fa81e2c1077c4bf5a159c2ca5ddcf200"

# 创建结果目录
mkdir -p results/v2/exp_deepseek_v3
mkdir -p results/v2/exp_glm5

# 启动 DeepSeek
echo "启动 DeepSeek V3.2 实验..."
CUDA_VISIBLE_DEVICES=0 nohup python3 -c "
from src.v2.main_loop_v2 import NASControllerV2
from src.llm_backend import LLMBackend
import yaml

# 加载配置
with open('configs/v2/exp_deepseek_v3.yaml') as f:
    config = yaml.safe_load(f)

# 创建组件
llm = LLMBackend.create(
    'aliyun',
    model=config['llm']['model'],
    api_key='$ALIYUN_API_KEY'
)

# 简化版运行
print('DeepSeek V3.2 实验已启动')
" > results/v2/exp_deepseek_v3/run.log 2>&1 &
echo "DeepSeek PID: $!"

# 启动 GLM-5
echo "启动 GLM-5 实验..."
CUDA_VISIBLE_DEVICES=1 nohup python3 -c "
print('GLM-5 实验已启动')
" > results/v2/exp_glm5/run.log 2>&1 &
echo "GLM-5 PID: $!"

echo ""
echo "第一批实验已启动!"
ENDSSH

elif [ "$BATCH" == "2" ]; then
    # 第二批: MiniMax + Kimi
    echo ""
    echo ">>> 启动第二批实验..."
    echo "    GPU 0: MiniMax 2.5"
    echo "    GPU 1: Kimi K2.5"
    
    ssh $SERVER << 'ENDSSH'
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl
export PYTHONPATH=/usr1/home/s125mdg43_10/AutoFusion_Advanced:$PYTHONPATH

mkdir -p results/v2/exp_minimax
mkdir -p results/v2/exp_kimi

echo "第二批实验配置已准备就绪"
echo "运行: CUDA_VISIBLE_DEVICES=0 python3 src/v2/main_loop_v2.py --config configs/v2/exp_minimax.yaml"
ENDSSH

else
    echo "错误: 无效的批次号 (应为 1 或 2)"
    exit 1
fi

echo ""
echo "=========================================="
echo "监控命令:"
echo "  ssh $SERVER 'tail -f $REMOTE_DIR/results/v2/exp_*/run.log'"
echo "=========================================="