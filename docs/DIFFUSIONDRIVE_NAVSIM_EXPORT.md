# DiffusionDrive (NAVSIM) 数据导出指南

## 📋 概述

DiffusionDrive 基于 **NAVSIM 框架**，与 VAD 使用不同的数据格式和加载方式。本文档说明如何从 DiffusionDrive 模型导出数据用于 E2E_RL 训练。

---

## 🎯 关键区别

| 特性 | VAD | DiffusionDrive |
|------|-----|----------------|
| **框架** | mmdet3d | NAVSIM |
| **数据集** | nuScenes 原始格式 | NAVSIM 处理后的格式 |
| **数据加载** | 自定义 DataLoader | SceneLoader |
| **配置系统** | Python 配置文件 | Hydra YAML 配置 |
| **推理脚本** | `dump_vad_inference.py` | `dump_diffusiondrive_navsim.py` |

---

## 📦 数据需求

### **NAVSIM 数据集**

DiffusionDrive 不使用原始的 nuScenes 格式，而是使用 NAVSIM 处理后的数据：

```
/path/to/navsim_data/
├── navsim_logs/
│   ├── train/
│   ├── val/
│   └── test/
└── sensor_blobs/
    ├── train/
    ├── val/
    └── test/
```

**下载 NAVSIM 数据**:

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive

# 下载验证集
bash download/download_test.sh

# 或下载完整训练集
bash download/download_trainval.sh

# 下载地图数据
bash download/download_maps.sh
```

---

## 🚀 完整流程

### **Step 1: 配置 DiffusionDrive 环境**

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive

# 创建环境
conda env create -f environment.yml
conda activate diffusiondrive

# 安装
pip install -e .
```

### **Step 2: 准备配置文件**

DiffusionDrive 使用 Hydra 配置系统。你需要一个 Agent 配置文件：

```yaml
# configs/diffusiondrive_agent.yaml
_target_: navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent

config:
  # TransfuserConfig 的参数
  num_bounding_boxes: 50
  trajectory_sampling:
    num_poses: 8
    time_horizon: 4.0
  # ... 其他配置

lr: 1e-4
checkpoint_path: /path/to/diffusiondrive.pth
```

### **Step 3: 运行数据导出**

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

python scripts/dump_diffusiondrive_navsim.py \
    --agent_config /path/to/diffusiondrive_agent.yaml \
    --checkpoint /path/to/diffusiondrive.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/diffusiondrive_raw \
    --max_samples 100 \
    --device cuda
```

**参数说明**:
- `--agent_config`: DiffusionDrive Agent 的 Hydra 配置文件
- `--checkpoint`: 预训练权重路径
- `--data_path`: NAVSIM 日志数据路径
- `--sensor_path`: NAVSIM 传感器数据路径
- `--output_dir`: 输出目录
- `--max_samples`: 最大样本数（调试用）
- `--device`: 计算设备 (cuda/cpu)

### **Step 4: 转换为标准格式**

```bash
python scripts/convert_diffusiondrive_dump.py \
    --input_dir data/diffusiondrive_raw \
    --output_dir data/diffusiondrive_dumps \
    --pool_mode grid \
    --grid_size 4 \
    --max_samples 1000
```

### **Step 5: 数据增强**

```bash
python scripts/augment_vad_data.py \
    --input_dir data/diffusiondrive_dumps \
    --output_dir data/diffusiondrive_dumps_full \
    --samples_per_original 50 \
    --noise_scale 0.1 \
    --max_samples 5000
```

### **Step 6: 验证数据**

```bash
python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader(
    'data/diffusiondrive_dumps_full',
    adapter_type='diffusiondrive',
    batch_size=8
)
print(f'✓ 数据集大小: {len(loader.dataset)} samples')
for batch in loader:
    print(f'✓ Batch 加载成功')
    print(f'  GT shape: {batch[\"gt_plan\"].shape}')
    print(f'  Interface shape: {batch[\"interface\"].reference_plan.shape}')
    break
"
```

---

## 🔧 问题排查

### **问题 1: ModuleNotFoundError: No module named 'navsim'**

**解决方案**:
```bash
cd /mnt/cpfs/prediction/lipeinan/RL/DiffusionDrive
pip install -e .
```

### **问题 2: 找不到配置文件**

**解决方案**:
检查 Hydra 配置路径，可能需要创建自定义配置文件：
```bash
mkdir -p configs
cat > configs/diffusiondrive_agent.yaml << 'EOF'
# 根据你的实际需求调整
_target_: navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent
lr: 1e-4
EOF
```

### **问题 3: 数据路径错误**

**确认数据结构**:
```bash
# 检查 NAVSIM 数据
ls -lh /path/to/navsim_logs/val/
ls -lh /path/to/sensor_blobs/val/

# 应该看到:
# navsim_logs/val/  -> .parquet 或 .json 文件
# sensor_blobs/val/ -> 图像文件夹
```

### **问题 4: 推理失败**

**调试步骤**:
```python
# 测试 Agent 加载
python << 'EOF'
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/diffusiondrive_agent.yaml')
print("配置加载成功:", cfg)

# 测试初始化
agent = TransfuserAgent(
    config=...,  # 根据你的配置
    lr=1e-4,
    checkpoint_path='/path/to/checkpoint.pth'
)
print("Agent 创建成功")

agent.initialize()
print("Agent 初始化成功")
EOF
```

---

## 📊 输出格式

### **原始输出 (dump_diffusiondrive_navsim.py)**

```python
# data/diffusiondrive_raw/000000.pt
{
    'planner_outputs': {
        'trajectory': [T, 3],           # (x, y, heading)
        # 可能包含其他字段
    },
    'gt_plan': [T, 2],                  # GT 轨迹 (x, y)
    'token': 'scene_token_string',
    'sample_idx': 0,
}
```

### **标准格式 (convert_diffusiondrive_dump.py)**

```python
# data/diffusiondrive_dumps/000000.pt
{
    'sample_idx': 0,
    'source_path': '/path/to/raw/000000.pt',
    'ego_fut_trajs': [T, 2],
    'interface_grid': {
        'scene_token': [D],
        'reference_plan': [T, 2],
        'plan_confidence': [1],
        'safety_*': [...],
    },
}
```

---

## 🎓 与 VAD 导出的对比

| 步骤 | VAD | DiffusionDrive |
|------|-----|----------------|
| **推理脚本** | `dump_vad_inference.py` | `dump_diffusiondrive_navsim.py` |
| **框架依赖** | mmdet3d | NAVSIM |
| **配置方式** | Python config | Hydra YAML |
| **数据加载** | 自定义 DataLoader | SceneLoader |
| **转换脚本** | 不需要（直接输出标准格式） | `convert_diffusiondrive_dump.py` |
| **数据增强** | `augment_vad_data.py` | `augment_vad_data.py` (相同) |

---

## ✅ 检查清单

在开始训练前，确保：

- [ ] NAVSIM 数据集已下载
- [ ] DiffusionDrive 环境已配置
- [ ] Agent 配置文件已准备
- [ ] 预训练权重已下载
- [ ] 推理脚本运行成功
- [ ] 转换脚本运行成功
- [ ] 数据增强完成
- [ ] 数据加载验证通过

---

## 🚀 快速开始 (最小示例)

```bash
# 1. 导出 10 个样本测试
python scripts/dump_diffusiondrive_navsim.py \
    --agent_config configs/dd_agent.yaml \
    --checkpoint dd.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/dd_raw \
    --max_samples 10

# 2. 转换
python scripts/convert_diffusiondrive_dump.py \
    --input_dir data/dd_raw \
    --output_dir data/dd_dumps \
    --pool_mode grid

# 3. 增强
python scripts/augment_vad_data.py \
    --input_dir data/dd_dumps \
    --output_dir data/dd_dumps_full \
    --samples_per_original 50 \
    --max_samples 500

# 4. 验证
python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader('data/dd_dumps_full', adapter_type='diffusiondrive')
print(f'数据集: {len(loader.dataset)} samples')
"
```

---

## 📞 需要帮助？

如果遇到问题：

1. 检查 NAVSIM 数据是否正确下载
2. 确认 DiffusionDrive 环境配置正确
3. 测试 Agent 能否正常加载
4. 查看错误日志，确认是哪个步骤失败
5. 参考 VAD 的导出流程作为对比
