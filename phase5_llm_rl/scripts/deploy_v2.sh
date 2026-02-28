#!/bin/bash
# ============================================================
# Phase 5.5 部署脚本
# ============================================================

set -e

# 配置
SERVER="s125mdg43_10@gpu43.dynip.ntu.edu.sg"
REMOTE_DIR="/usr1/home/s125mdg43_10/AutoFusion_Advanced/phase5_llm_rl"
LOCAL_DIR="./phase5_llm_rl"

echo "=========================================="
echo "Phase 5.5 部署脚本"
echo "=========================================="

# 1. 检查本地文件
echo ""
echo ">>> 检查本地文件..."
if [ ! -d "$LOCAL_DIR/src/v2" ]; then
    echo "错误: 本地 v2 目录不存在"
    exit 1
fi

# 2. 创建远程目录
echo ""
echo ">>> 创建远程目录..."
ssh $SERVER "mkdir -p $REMOTE_DIR/src/v2 $REMOTE_DIR/configs/v2 $REMOTE_DIR/results/v2"

# 3. 同步代码
echo ""
echo ">>> 同步代码到服务器..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    $LOCAL_DIR/src/v2/ \
    $SERVER:$REMOTE_DIR/src/v2/

# 4. 同步配置
echo ""
echo ">>> 同步配置文件..."
rsync -avz $LOCAL_DIR/configs/v2/ $SERVER:$REMOTE_DIR/configs/v2/

# 5. 测试连接
echo ""
echo ">>> 测试服务器连接..."
ssh $SERVER "cd $REMOTE_DIR && python3 -c 'from src.v2.architecture_templates import ARCHITECTURE_TEMPLATES; print(f\"模板数量: {len(ARCHITECTURE_TEMPLATES)}\")'"

echo ""
echo "=========================================="
echo "部署完成!"
echo "=========================================="
echo ""
echo "下一步: 运行测试脚本"
echo "  ssh $SERVER"
echo "  cd $REMOTE_DIR"
echo "  python3 src/v2/test_improvements.py"