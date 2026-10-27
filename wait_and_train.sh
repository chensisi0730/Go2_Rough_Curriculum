#!/bin/bash
# 等待并发的 Geesun 训练释放 GPU 后，自动以 10288 环境 / 20000 迭代启动 Go2 训练。
set -u

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"   # 自动取脚本所在工程目录（即当前工程）
WAIT_LOG=$PROJ/wait_launch.log
TRAIN_LOG=$PROJ/train_run.log
GEESUN_MATCH="Unitree-GeesunDog-LingsiD30W-Velocity"
GPU_FREE_THRESHOLD_MIB=6000   # 显存低于此值视为已释放

# 环境准备
source ~/miniconda3/etc/profile.d/conda.sh
conda activate go2_rough_curriculum || { echo "[$(date)] conda 激活失败" >> "$WAIT_LOG"; exit 1; }

export OMNI_KIT_ACCEPT_EULA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 双保险：即使 go2_demo 未 editable 安装，也能通过 PYTHONPATH 导入
export PYTHONPATH="$PROJ/source/go2_demo${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date)] 守护脚本启动，等待 Geesun 训练 ($GEESUN_MATCH) 释放 GPU (阈值 ${GPU_FREE_THRESHOLD_MIB}MiB) ..." | tee -a "$WAIT_LOG"

while true; do
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    GEESUN_RUNNING=$(pgrep -f "$GEESUN_MATCH" > /dev/null && echo yes || echo no)

    if [ "$GEESUN_RUNNING" = "no" ] && [ -n "$GPU_USED" ] && [ "$GPU_USED" -lt "$GPU_FREE_THRESHOLD_MIB" ]; then
        echo "[$(date)] 检测到 GPU 已空闲 (used=${GPU_USED}MiB, Geesun=$GEESUN_RUNNING)，开始启动 Go2 20000 迭代训练。" | tee -a "$WAIT_LOG"
        break
    fi

    echo "[$(date)] 仍占用: GPU_used=${GPU_USED}MiB, Geesun=$GEESUN_RUNNING，继续等待(5分钟)..." | tee -a "$WAIT_LOG"
    sleep 300
done

cd "$PROJ"
# 迁移学习起点：来自另一工程 go2_Complex_terrain（2026-10-26_14-19-52 run，2500 迭代）
PRETRAINED="/home/css/work/unitree/rl/zhouzhiguang_repo/Go2_Rough_Curriculum/logs/rsl_rl/go2_Complex_terrain/2026-10-26_14-19-52/model_2500.pt"
setsid nohup python -u scripts/rsl_rl/train.py \
    --task=Go2-velocity-v0 \
    --num_envs=10288 \
    --headless \
    --max_iterations=20000 \
    --pretrained "$PRETRAINED" \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
NEW_PID=$!
echo "[$(date)] Go2 训练已启动 PID=$NEW_PID，日志: $TRAIN_LOG" | tee -a "$WAIT_LOG"
