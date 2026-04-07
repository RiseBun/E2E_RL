#!/bin/bash

# ============================================
# 运行 DiffusionDrive 数据导出脚本
# ============================================

cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

# 设置 PYTHONPATH
export PYTHONPATH=/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/DiffusionDrive:$PYTHONPATH

# 设置环境变量
export NUPLAN_MAPS_ROOT="/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/DiffusionDrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/DiffusionDrive/navsim_workspace/exp"

echo "=========================================="
echo " DiffusionDrive 数据导出"
echo "=========================================="
echo ""

# 运行 dump 脚本
python scripts/dump_diffusiondrive_inference.py \
    --diffusiondrive_root /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/DiffusionDrive \
    --agent_config /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/DiffusionDrive/navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml \
    --checkpoint /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/DiffusionDrive/download/ckpt/diffusiondrive_navsim_88p1_PDMS \
    --data_path /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/DiffusionDrive/navsim_workspace/dataset/navsim_logs/trainval \
    --sensor_path /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/DiffusionDrive/navsim_workspace/dataset/sensor_blobs/trainval \
    --output_dir data/diffusiondrive_dumps \
    --max_samples 100 \
    --device cuda

echo ""
echo "=========================================="
echo " 导出完成！"
echo "=========================================="
