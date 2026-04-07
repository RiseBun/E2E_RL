# DiffusionDrive 快速开始指南

## 📋 概述

本指南帮助你从 `/mnt/datasets/e2e-navsim/20260302` 数据集快速导出数据，用于 E2E_RL 训练。

---

## 🚀 快速流程 (3 步)

### **Step 1: 解压数据集**

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

# 运行解压脚本
bash scripts/extract_navsim_dataset.sh
```

**预期输出**:
```
=========================================
NAVSIM 数据集解压脚本
=========================================

[1/2] 解压训练集 (navsim_train.tar)...
✓ 训练集解压完成

=========================================
✓ 解压完成！
=========================================

数据结构:
  /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/data/
  ├── trainval_navsim_logs/
  ├── trainval_sensor_blobs/
  └── nuplan-maps-v1.0/
```

---

### **Step 2: 配置 DiffusionDrive 环境**

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive

# 创建环境 (如果还没有)
conda env create -f environment.yml
conda activate diffusiondrive

# 安装
pip install -e .

# 测试导入
python -c "from navsim.common.dataloader import SceneLoader; print('✓ 环境正常')"
```

---

### **Step 3: 导出数据**

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

# 先测试 10 个样本
python scripts/dump_diffusiondrive_simple.py \
    --checkpoint /path/to/diffusiondrive.pth \
    --data_dir /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/data \
    --output_dir data/diffusiondrive_raw \
    --max_samples 10

# 如果成功，导出更多
python scripts/dump_diffusiondrive_simple.py \
    --checkpoint /path/to/diffusiondrive.pth \
    --data_dir /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/data \
    --output_dir data/diffusiondrive_raw \
    --max_samples 1000
```

**参数说明**:
- `--checkpoint`: DiffusionDrive 权重路径 (必需)
- `--data_dir`: 解压后的数据目录
- `--output_dir`: 输出目录
- `--max_samples`: 样本数 (先用小值测试)

---

## 📦 完整流程

### **1. 准备 DiffusionDrive 权重**

如果你还没有权重，从 HuggingFace 下载：

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive

# 下载权重
wget https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_stage2.pth

# 或使用其他版本
# wget https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_nuscenes.pth
```

### **2. 解压数据集**

```bash
bash scripts/extract_navsim_dataset.sh
```

### **3. 导出原始数据**

```bash
python scripts/dump_diffusiondrive_simple.py \
    --checkpoint diffusiondrive_stage2.pth \
    --data_dir /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/data \
    --output_dir data/diffusiondrive_raw \
    --max_samples 1000
```

### **4. 转换为标准格式**

```bash
python scripts/convert_diffusiondrive_dump.py \
    --input_dir data/diffusiondrive_raw \
    --output_dir data/diffusiondrive_dumps \
    --pool_mode grid \
    --grid_size 4
```

### **5. 数据增强**

```bash
python scripts/augment_vad_data.py \
    --input_dir data/diffusiondrive_dumps \
    --output_dir data/diffusiondrive_dumps_full \
    --samples_per_original 50 \
    --noise_scale 0.1 \
    --max_samples 5000
```

### **6. 验证数据**

```bash
python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader(
    'data/diffusiondrive_dumps_full',
    adapter_type='diffusiondrive',
    batch_size=8
)
print(f'✓ 数据集: {len(loader.dataset)} samples')
for batch in loader:
    print(f'✓ Batch 加载成功')
    break
"
```

---

## 🔧 常见问题

### **Q1: 找不到模型权重**

**解决方案**:
```bash
# 下载权重
cd /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive
wget https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_stage2.pth

# 或指定完整路径
python scripts/dump_diffusiondrive_simple.py \
    --checkpoint /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/diffusiondrive_stage2.pth \
    ...
```

### **Q2: 环境导入失败**

**解决方案**:
```bash
cd /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive

# 确保激活了正确的环境
conda activate diffusiondrive

# 重新安装
pip install -e .

# 测试
python -c "from navsim.common.dataloader import SceneLoader; print('OK')"
```

### **Q3: 数据路径错误**

**检查数据结构**:
```bash
ls -lh /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/data/
# 应该看到:
# ├── trainval_navsim_logs/
# ├── trainval_sensor_blobs/
# └── nuplan-maps-v1.0/
```

### **Q4: 数据集还没解压**

**运行解压**:
```bash
bash scripts/extract_navsim_dataset.sh
```

---

## 📊 数据规模参考

| 样本数 | 推理时间 | 存储大小 | 用途 |
|--------|---------|---------|------|
| 10 | ~1 分钟 | ~400KB | 测试流程 |
| 100 | ~10 分钟 | ~4MB | 调试 |
| 1,000 | ~1-2 小时 | ~40MB | 小实验 |
| 5,000 | ~5-10 小时 | ~200MB | 正式训练 |

---

## ✅ 检查清单

开始前确保：

- [ ] 数据集已解压 (`/mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/data/`)
- [ ] DiffusionDrive 环境已配置
- [ ] 模型权重已下载
- [ ] 有足够磁盘空间 (输出约 40MB/1000 样本)

---

## 🎯 预期输出

### **成功输出示例**:

```
============================================================
DiffusionDrive 数据导出工具 (简化版)
============================================================

Step 1: 检查 DiffusionDrive 环境...
✓ DiffusionDrive 环境正常

Step 2: 查找模型权重...
✓ 找到权重: /path/to/diffusiondrive.pth

Step 3: 构建 DiffusionDrive Agent...
✓ Agent 创建成功

Step 4: 构建 SceneLoader...
数据路径:
  日志: /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/data/trainval_navsim_logs
  传感器: /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive/data/trainval_sensor_blobs
✓ SceneLoader 创建成功
  Token 数量: 100

Step 5: 运行推理并导出...
开始导出 100 个样本...
输出目录: data/diffusiondrive_raw

[10/100] ✓ 000000.pt (0.52s)
[20/100] ✓ 000001.pt (0.48s)
...
[100/100] ✓ 000099.pt (0.51s)

============================================================
✓ 导出完成！
  成功: 100 个样本
  失败: 0 个样本
  输出目录: data/diffusiondrive_raw
============================================================
```

---

## 📞 需要帮助？

如果遇到问题，请检查：

1. 数据集是否正确解压
2. 环境是否正确配置
3. 权重文件是否存在
4. 错误日志的具体信息
