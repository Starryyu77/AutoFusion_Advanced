#!/bin/bash
# 在服务器上启动E1实验
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced/expv2
nohup bash E1_main_evaluation/scripts/run_on_server.sh 2 > E1_main_evaluation/results/E1_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "E1实验已在GPU 2上后台启动"
echo "查看日志: tail -f E1_main_evaluation/results/*.log"
