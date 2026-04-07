# DiffusionDrive 数据准备指南

## 📋 概述

本文档说明如何为 DiffusionDrive 模型准备训练数据，用于训练 UpdateEvaluator 和 CorrectionPolicy。

---

## 🎯 数据需求

### **必需数据集**

| 数据集 | 用途 | 样本数 | 说明 |
|--------|------|--------|------|
| **nuScenes v1.0** | 推理和训练 | 6,019 (val) | 标准 nuScenes 数据集 |

### **数据格式**

最终需要的数据格式：
```
data/diffusiondrive_dumps_full/
├── 000000.pt  # 包含:
│   ├─ ego_fut_trajs: [T, 2]          # GT 轨迹
│   ├─ interface_grid: {               # PlanningInterface
│   │   ├─ scene_token: [D]
│   │   ├─ reference_plan: [T, 2]
│   │   ├─ plan_confidence: [1]
│   │   └─ safety_*: [...]
│   └─ ...
├── 000001.pt
└── ... (5,000+ 个样本)
```

---

## 🚀 完整流程

### **方案 A: 使用已有的 DiffusionDrive 原始输出（推荐）**

适用场景：别人已经跑过 DiffusionDrive，有 `.pt` 输出文件

#### **Step 1: 准备原始输出**

确保原始输出包含以下字段：
```python
{
    'planner_outputs': {  # 或 'outputs' 或直接是顶层
        'trajectory': [T, 3],           # 必需: (x, y, heading)
        'bev_semantic_map': [C, H, W],  # 必需: BEV 语义图
        'agent_states': [A, 5],         # 可选
        'agent_labels': [A],            # 可选
    },
    'gt_plan': [T, 2],                  # 必需: GT 轨迹
}
```

#### **Step 2: 转换为标准格式**

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

python scripts/convert_diffusiondrive_dump.py \
    --input_dir /path/to/diffusiondrive_raw_outputs \
    --output_dir data/diffusiondrive_dumps \
    --pool_mode grid \
    --grid_size 4 \
    --max_samples 1000
```

**参数说明**:
- `--input_dir`: 原始 DiffusionDrive 输出目录
- `--output_dir`: 转换后的标准格式目录
- `--pool_mode`: BEV 池化方式 (`mean` / `grid` / `ego_local`)
- `--grid_size`: grid 池化的分块数 (默认 4)
- `--max_samples`: 最大样本数 (可选)

**预期输出**:
```
找到 1000 个样本
转换完成: 1000 samples -> data/diffusiondrive_dumps
```

#### **Step 3: 数据增强（可选但推荐）**

```bash
python scripts/augment_vad_data.py \
    --input_dir data/diffusiondrive_dumps \
    --output_dir data/diffusiondrive_dumps_full \
    --samples_per_original 50 \
    --noise_scale 0.1 \
    --max_samples 5000
```

#### **Step 4: 验证数据**

```bash
python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader(
    'data/diffusiondrive_dumps_full',
    adapter_type='diffusiondrive',
    batch_size=8
)
print(f'数据集大小: {len(loader.dataset)} samples')
for batch in loader:
    print(f'Batch - GT: {batch[\"gt_plan\"].shape}')
    print(f'Batch - Interface: {batch[\"interface\"].reference_plan.shape}')
    break
"
```

---

### **方案 B: 从 DiffusionDrive 模型推理收集数据**

适用场景：你有 DiffusionDrive 权重和完整环境

#### **Step 1: 配置 DiffusionDrive 环境**

确保你能成功运行 DiffusionDrive 的推理：
```bash
# 测试 DiffusionDrive 推理
cd /path/to/DiffusionDrive
python tools/test.py \
    --config configs/diffusiondrive.py \
    --checkpoint diffusiondrive.pth \
    --data_root /data/nuscenes
```

#### **Step 2: 修改推理脚本**

编辑 `scripts/dump_diffusiondrive_inference.py`，根据 DiffusionDrive 的实际代码调整以下函数：

1. `build_model()`: 模型加载逻辑
2. `build_dataloader()`: 数据加载逻辑
3. `extract_gt()`: GT 提取逻辑
4. 推理和输出提取部分

#### **Step 3: 运行推理**

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

python scripts/dump_diffusiondrive_inference.py \
    --config /path/to/diffusiondrive_config.py \
    --checkpoint /path/to/diffusiondrive.pth \
    --data_root /data/nuscenes \
    --output_dir data/diffusiondrive_raw \
    --max_samples 100  # 先用少量样本测试
```

#### **Step 4: 转换和增强**

```bash
# 转换为标准格式
python scripts/convert_diffusiondrive_dump.py \
    --input_dir data/diffusiondrive_raw \
    --output_dir data/diffusiondrive_dumps \
    --pool_mode grid

# 数据增强
python scripts/augment_vad_data.py \
    --input_dir data/diffusiondrive_dumps \
    --output_dir data/diffusiondrive_dumps_full \
    --samples_per_original 50 \
    --max_samples 5000
```

---

## 📊 DiffusionDrive 输出字段说明

### **必需字段**

| 字段 | 形状 | 说明 |
|------|------|------|
| `trajectory` | [T, 3] | 最优轨迹 (x, y, heading)，T=8 |
| `bev_semantic_map` | [C, H, W] | BEV 语义分割图，C=7, H=128, W=256 |

### **可选字段**

| 字段 | 形状 | 说明 |
|------|------|------|
| `agent_states` | [A, 5] | 检测到的车辆 (x, y, heading, length, width) |
| `agent_labels` | [A] | 车辆有效性分数 (logits) |
| `all_poses_reg` | [M, T, 3] | 全部候选轨迹 (多模态) |
| `all_poses_cls` | [M] | 候选轨迹分类分数 |

---

## 🔧 常见问题

### **Q1: 我应该收集多少样本？**

- **最小**: 100 帧真实推理 + 数据增强 → 5,000 样本
- **推荐**: 500 帧真实推理 + 数据增强 → 25,000 样本
- **理想**: 1000+ 帧真实推理

### **Q2: 使用哪种 pool_mode？**

| Pool Mode | scene_token 维度 | 特点 | 推荐场景 |
|-----------|-----------------|------|---------|
| `mean` | 7 | 全局均值，最紧凑 | 快速验证 |
| `grid` | 112 (4×4×7) | 保留空间结构 | **推荐，平衡** |
| `ego_local` | 可变 | 以自我为中心 | 需要局部感知 |

### **Q3: 数据增强会不会影响质量？**

- 数据增强添加的是**小幅噪声** (noise_scale=0.1)
- GT 轨迹**保持不变**
- 只扰动 `reference_plan` 和 `scene_token`
- 实测：101 帧 → 5,001 样本，效果良好

### **Q4: 如何验证数据质量？**

```bash
# 1. 检查文件数量
ls data/diffusiondrive_dumps_full/*.pt | wc -l

# 2. 检查数据内容
python -c "
import torch
data = torch.load('data/diffusiondrive_dumps_full/000000.pt')
print('Keys:', data.keys())
print('GT shape:', data['ego_fut_trajs'].shape)
print('Interface keys:', data['interface_grid'].keys())
"

# 3. 加载 dataloader
python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader(
    'data/diffusiondrive_dumps_full',
    adapter_type='diffusiondrive',
    batch_size=8
)
for batch in loader:
    print('Batch loaded successfully')
    break
"
```

---

## 📁 目录结构

```
E2E_RL/
├── data/
│   ├── diffusiondrive_raw/          # [方案 B] DiffusionDrive 原始推理输出
│   ├── diffusiondrive_dumps/        # 转换后的标准格式 (少量)
│   └── diffusiondrive_dumps_full/   # 增强后的训练数据 (5000+)
├── scripts/
│   ├── dump_diffusiondrive_inference.py   # [方案 B] 推理脚本
│   └── convert_diffusiondrive_dump.py     # [方案 A/B] 转换脚本
├── planning_interface/
│   └── adapters/
│       └── diffusiondrive_adapter.py      # Adapter (已完成)
└── configs/
    └── update_evaluator.yaml              # 训练配置
```

---

## 🎓 与 VAD 的对比

| 特性 | VAD | DiffusionDrive |
|------|-----|----------------|
| **数据集** | nuScenes | nuScenes |
| **Adapter** | `VADPlanningAdapter` | `DiffusionDrivePlanningAdapter` ✅ |
| **转换脚本** | `dump_vad_inference.py` | `convert_diffusiondrive_dump.py` ✅ |
| **scene_token** | BEV embedding | BEV semantic map |
| **reference_plan** | [T, 2] | [T, 2] (从 [T, 3] 提取) |
| **时间步长 T** | 6 | 8 |
| **BEV 类别 C** | - | 7 |

---

## ✅ 检查清单

在开始训练前，确保：

- [ ] nuScenes 数据集已下载并配置
- [ ] DiffusionDrive 原始输出已准备（方案 A）或推理脚本已修改（方案 B）
- [ ] 转换脚本运行成功
- [ ] 数据增强完成（可选）
- [ ] 数据加载验证通过
- [ ] 至少 100 个真实样本（增强后 5000+）

---

## 🚀 下一步

数据准备完成后：

1. **训练 UpdateEvaluator**:
   ```bash
   python scripts/train_evaluator_v2.py \
       --data_dir data/diffusiondrive_dumps_full \
       --output_dir experiments/diffusiondrive_evaluator
   ```

2. **训练 CorrectionPolicy**:
   ```bash
   python scripts/expC_relaxed.py \
       --data_dir data/diffusiondrive_dumps_full \
       --evaluator_ckpt experiments/diffusiondrive_evaluator/evaluator_epoch_30.pth \
       --output_dir experiments/diffusiondrive_correction
   ```

---

## 📞 需要帮助？

如果遇到问题，请检查：
1. DiffusionDrive 输出格式是否符合要求
2. Adapter 是否正确处理了所有字段
3. 数据加载是否正常工作
4. 参考 VAD 的数据准备流程
