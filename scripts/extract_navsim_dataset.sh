#!/bin/bash
# 解压 NAVSIM 数据集脚本

set -e

DATASET_DIR="/mnt/datasets/e2e-navsim/20260302"
OUTPUT_BASE="/mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive"

echo "========================================="
echo "NAVSIM 数据集解压脚本"
echo "========================================="

# 创建输出目录
mkdir -p "$OUTPUT_BASE/data"

# 解压训练集
echo ""
echo "[1/2] 解压训练集 (navsim_train.tar)..."
echo "源文件: $DATASET_DIR/navsim_train.tar"
echo "目标目录: $OUTPUT_BASE/data/"

tar -xf "$DATASET_DIR/navsim_train.tar" \
    -C "$OUTPUT_BASE/data/" \
    --strip-components=4 \
    "home/ma-user/work/gongxijie/navsim_train/navsim_logs" \
    "home/ma-user/work/gongxijie/navsim_train/sensor_blobs" \
    "home/ma-user/work/gongxijie/navsim_train/nuplan-maps-v1.0"

echo "✓ 训练集解压完成"

# 重命名为标准格式
if [ -d "$OUTPUT_BASE/data/navsim_logs" ]; then
    mv "$OUTPUT_BASE/data/navsim_logs" "$OUTPUT_BASE/data/trainval_navsim_logs"
fi

if [ -d "$OUTPUT_BASE/data/sensor_blobs" ]; then
    mv "$OUTPUT_BASE/data/sensor_blobs" "$OUTPUT_BASE/data/trainval_sensor_blobs"
fi

echo ""
echo "========================================="
echo "✓ 解压完成！"
echo "========================================="
echo ""
echo "数据结构:"
echo "  $OUTPUT_BASE/data/"
echo "  ├── trainval_navsim_logs/"
echo "  ├── trainval_sensor_blobs/"
echo "  └── nuplan-maps-v1.0/"
echo ""
echo "下一步:"
echo "  1. 配置 DiffusionDrive 环境"
echo "  2. 运行数据导出脚本"
echo "========================================="
