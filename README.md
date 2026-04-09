# E2E_RL: 模型无关的端到端强化学习轨迹修正框架

通过强化学习学习最优的轨迹修正策略，提升任意 E2E 自动驾驶模型的规划质量。

**核心特性**: 🎯 **训练和推理对所有 E2E 规划模型完全模型无关**

---

## 📋 目录

- [核心理念](#核心理念)
- [已集成模型](#已集成模型)
- [双环境工作流](#双环境工作流)
- [标准使用流程](#标准使用流程)
  - [阶段 1: 导出数据](#阶段-1-导出数据)
  - [阶段 2: 验证数据](#阶段-2-验证数据)
  - [阶段 3: 训练模型](#阶段-3-训练模型)
  - [阶段 4: 推理验证](#阶段-4-推理验证)
- [各模型独特参数](#各模型独特参数)
- [项目结构](#项目结构)
- [集成新模型](#集成新模型)
- [坐标系约定](#坐标系约定)
- [实验结果](#实验结果)
- [常见问题](#常见问题)

---

## 核心理念

### 🎯 模型无关设计

```
┌─────────────────────────────────────────────────────────────┐
│  任何 E2E 模型 (VAD, SparseDrive, DiffusionDrive, ...)       │
│         ↓ 导出 .pt 文件                                      │
│  PlanningInterface (统一接口层)                              │
│         ↓ 完全解耦                                           │
│  E2E_RL 训练和推理 (模型无关)                                  │
└─────────────────────────────────────────────────────────────┘
```

**关键优势**:
- ✅ **一次实现，处处运行**: 训练和推理代码对所有模型完全相同
- ✅ **快速集成新模型**: 只需实现 Adapter，无需修改训练代码
- ✅ **公平对比**: 相同的训练流程，公平的性能对比
- ✅ **灵活扩展**: 随时添加新模型

---

## 已集成模型

### ✅ 完全集成（7个）

| 模型 | 论文 | 框架 | Adapter | Dump 脚本 | 状态 |
|------|------|------|---------|-----------|------|
| **VAD** | ICLR 2024 | mmdet3d | `VADPlanningAdapter` | `dump_vad_inference.py` | ✅ |
| **VADv2** | ICLR 2026 | mmdet3d | `VADv2PlanningAdapter` | `dump_vadv2_inference.py` | ✅ |
| **SparseDrive** | ECCV 2024 | mmdet3d | `SparseDrivePlanningAdapter` | `dump_sparsedrive_inference.py` | ✅ |
| **SparseDriveV2** | arXiv 2024 | mmdet3d | `SparseDriveV2PlanningAdapter` | `dump_sparsedrivev2_inference.py` | ✅ |
| **DiffusionDrive** | arXiv 2024 | NAVSIM | `DiffusionDrivePlanningAdapter` | `dump_diffusiondrive_inference.py` | ✅ |
| **DiffusionDriveV2** | arXiv 2025 | NAVSIM | `DiffusionDriveV2PlanningAdapter` | `dump_diffusiondrivev2_inference.py` | ✅ |
| **UniAD** | CVPR 2023 | mmdet3d | `UniADPlanningAdapter` | `dump_uniad_inference.py` | ✅ |

### ⏳ 延迟集成（1个）

| 模型 | 原因 | 预计工作量 |
|------|------|-----------|
| **TCP** | 框架不兼容（CARLA vs nuScenes） | 2-3天 |

> 📖 详细信息: [多模型集成完成报告](docs/多模型集成完成报告.md)

---

## 双环境工作流

### 🔑 核心原则

本项目使用**两个独立的 conda 环境**，分工明确：

```
环境 1: 原 E2E 模型环境
  ↓ 用途: 运行模型推理，导出数据
  ↓ 包含: mmdet3d, NAVSIM 等模型特定依赖
  
环境 2: E2E_RL 环境
  ↓ 用途: 数据处理、训练、推理、分析
  ↓ 包含: torch, numpy 等基础依赖
```

### 📊 环境分工

| 阶段 | 使用环境 | 原因 |
|------|---------|------|
| **导出数据** | 原模型环境 | 需要原模型的依赖和代码 |
| **数据验证** | E2E_RL 环境 | 只需读取 .pt 文件 |
| **数据增强** | E2E_RL 环境 | 只需读取 .pt 文件 |
| **训练模型** | E2E_RL 环境 | 你的训练代码 |
| **推理验证** | E2E_RL 环境 | 你的推理代码 |
| **对比实验** | E2E_RL 环境 | 你的分析代码 |

### 🛠️ 环境配置

#### 1. E2E_RL 环境（必需）

```bash
# 创建 E2E_RL 核心环境
conda create -n e2e_rl python=3.10 -y
conda activate e2e_rl

# 安装 PyTorch（根据您的 CUDA 版本调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy matplotlib scipy pyyaml tqdm
```

#### 2. 原模型环境（按需创建）

**每个模型可能需要独立环境**（如果依赖冲突）：

```bash
# 示例：VAD 环境
conda create -n vad python=3.10 -y
conda activate vad
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
cd projects/VAD && pip install -r requirements.txt

# 示例：DiffusionDrive 环境（NAVSIM）
conda create -n diffusiondrive python=3.10 -y
conda activate diffusiondrive
cd projects/DiffusionDrive
# 参考 projects/DiffusionDrive/README.md 安装依赖
```

> 💡 **提示**: 如果多个模型的依赖不冲突，可以共用一个环境。建议先尝试共用，遇到冲突再创建独立环境。

---

## 标准使用流程

> 以下流程对所有模型完全相同，只需替换模型名称和数据路径。

### 阶段 1: 导出数据

```bash
# 1. 激活原模型环境
conda activate <model_env>  # 如 vad, sparsedrive, diffusiondrive

# 2. 设置 PYTHONPATH（如果需要）
export PYTHONPATH=projects/<ModelName>:$PYTHONPATH

# 3. 运行导出脚本，注意查看多模型集成完成报告查看用法
cd ~/E2E_RL
python scripts/dump_<model>_inference.py \
    --output_dir data/<model>_dumps \
    --max_samples 10  # 收集样本数量

# 4. 导出完成后，切换回 E2E_RL 环境
conda deactivate
conda activate e2e_rl
```

### 阶段 2: 验证数据

```bash
# 在 E2E_RL 环境中运行
cd ~/E2E_RL

python -c "
from data.dataloader import build_planner_dataloader

# 加载数据
loader = build_planner_dataloader(
    'data/<model>_dumps',
    adapter_type='<model>',
    batch_size=8
)

# 验证数据质量
for batch in loader:
    gt = batch['gt_plan']
    ref = batch['interface'].reference_plan
    
    # 检查坐标系（期望 15-25m）
    gt_end = gt[:, -1, :].norm(dim=-1).mean()
    ref_end = ref[:, -1, :].norm(dim=-1).mean()
    
    print(f'✅ GT终点: {gt_end:.2f}m')
    print(f'✅ Ref终点: {ref_end:.2f}m')
    print(f'✅ Scene token: {batch[\"interface\"].scene_token.shape}')
    break
"
```

### 阶段 3: 训练模型

#### 3.1 数据增强（可选）

```bash
cd ~/E2E_RL

python scripts/augment_vad_data.py \
    --input_dir data/<model>_dumps \
    --output_dir data/<model>_dumps_full \
    --samples_per_original 50 \
    --noise_scale 0.1 \
    --max_samples 5000
```

#### 3.2 训练 UpdateEvaluator

```bash
# 方式 1: 使用默认配置
python scripts/train_evaluator_v2.py

# 方式 2: 使用命令行参数（推荐）
python scripts/train_evaluator_v2.py \
    --data_dir data/<model>_dumps_full \
    --output_dir experiments/<model>_evaluator \
    --epochs 50 \
    --lr 5e-5

# 方式 3: 查看所有可用参数
python scripts/train_evaluator_v2.py --help
```

**常用参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 数据目录 | `data/vad_dumps_full` |
| `--output_dir` | 输出目录 | `experiments/update_evaluator` |
| `--epochs` | 训练轮数 | 50 |
| `--lr` | 学习率 | 5e-5 |
| `--batch_size` | 数据 batch size | 16 |
| `--train_batch_size` | 训练 batch size | 128 |
| `--scene_dim` | 场景特征维度 | 256 |
| `--plan_len` | 轨迹长度 | 6 |

#### 3.3 训练 CorrectionPolicy

```bash
# 方式 1: 使用默认配置
python scripts/expC_relaxed.py

# 方式 2: 使用命令行参数（推荐）
python scripts/expC_relaxed.py \
    --data_dir data/<model>_dumps_full \
    --evaluator_ckpt experiments/<model>_evaluator/evaluator_final.pth \
    --output_dir experiments/<model>_policy \
    --rl_epochs 50

# 方式 3: 查看所有可用参数
python scripts/expC_relaxed.py --help
```

**常用参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 数据目录 | `data/vad_dumps_full` |
| `--evaluator_ckpt` | UpdateEvaluator 检查点 | `experiments/update_evaluator/evaluator_final.pth` |
| `--output_dir` | 输出目录 | `experiments/expC_learned_gate` |
| `--bc_epochs` | Behavioral Cloning 轮数 | 3 |
| `--rl_epochs` | Reinforcement Learning 轮数 | 15 |
| `--lr` | 学习率 | 3e-4 |

### 阶段 4: 推理验证

```bash
cd ~/E2E_RL

python scripts/inference_with_correction.py \
    --checkpoint experiments/<model>_policy/policy_final.pth \
    --evaluator experiments/<model>_evaluator/evaluator_final.pth \
    --data_dir data/<model>_dumps \
    --output_dir outputs/<model>_inference
```

---

## 各模型独特参数

> 以下仅列出每个模型 dump 脚本的**独特参数**。通用参数（如 `--output_dir`, `--max_samples`）已在上面说明。

### VAD / VADv2 / UniAD / SparseDrive / SparseDriveV2

这些模型基于 mmdet3d 框架，参数相同：

```bash
python scripts/dump_<model>_inference.py \
    --config projects/<ModelName>/projects/configs/<config>.py \
    --checkpoint /path/to/<model>.pth \
    --data_root /path/to/nuscenes/
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | 模型配置文件路径 | `projects/VAD/projects/configs/VAD/VAD_base_e2e.py` |
| `--checkpoint` | 模型权重路径 | `/path/to/vad_epoch_24.pth` |
| `--data_root` | nuScenes 数据根目录 | `/path/to/nuscenes/` |

---

### DiffusionDrive / DiffusionDriveV2

基于 NAVSIM 框架，参数不同：

```bash
python scripts/dump_diffusiondrive_inference.py \
    --agent_config projects/DiffusionDrive/navsim_agent/diffusiondrive_agent.yaml \
    --checkpoint /path/to/diffusiondrive.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --convert \
    --pool_mode grid
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `--agent_config` | Agent YAML 配置 | `projects/DiffusionDrive/.../diffusiondrive_agent.yaml` |
| `--checkpoint` | 模型权重路径 | `/path/to/diffusiondrive.pth` |
| `--data_path` | NAVSIM 日志路径 | `/path/to/navsim_logs/val` |
| `--sensor_path` | 传感器数据路径 | `/path/to/sensor_blobs/val` |
| `--convert` | 直接转换为标准格式（推荐） | 启用后一步完成导出+转换 |
| `--pool_mode` | BEV 池化方式 | `mean` / `grid` / `ego_local` |

---

## 项目结构

```
E2E_RL/
├── projects/                          # E2E 模型项目
│   ├── VAD/                         # VAD 项目
│   ├── VADv2/                       # VADv2 项目
│   ├── SparseDrive/                 # SparseDrive 项目
│   ├── SparseDriveV2/               # SparseDriveV2 项目
│   ├── DiffusionDrive/              # DiffusionDrive 项目
│   ├── DiffusionDriveV2/            # DiffusionDriveV2 项目
│   ├── UniAD/                       # UniAD 项目
│   └── TCP/                         # TCP 项目（延迟集成）
│
├── planning_interface/                # 统一接口层（模型无关）
│   ├── interface.py                  # PlanningInterface 定义
│   └── adapters/                     # 模型适配器
│       ├── base_adapter.py           # 抽象基类
│       ├── vad_adapter.py            # VAD 适配器
│       ├── vadv2_adapter.py          # VADv2 适配器
│       ├── sparsedrive_adapter.py    # SparseDrive 适配器
│       ├── sparsedrivev2_adapter.py  # SparseDriveV2 适配器
│       ├── diffusiondrive_adapter.py # DiffusionDrive 适配器
│       ├── diffusiondrivev2_adapter.py # DiffusionDriveV2 适配器
│       └── uniad_adapter.py          # UniAD 适配器
│
├── data/
│   └── dataloader.py                 # 通用 DataLoader
│
├── correction_policy/                 # RL Policy（模型无关）
│   ├── policy.py                     # CorrectionPolicy
│   ├── actor.py                      # GaussianCorrectionActor
│   └── losses.py                     # PPO Loss
│
├── update_selector/                   # 门控选择器（模型无关）
│   ├── safety_guard.py               # SafetyGuard（硬底线）
│   ├── stapo_gate.py                 # STAPOGate（规则）
│   └── update_evaluator.py           # UpdateEvaluator + LearnedUpdateGate
│
├── scripts/
│   ├── dump_*.py                     # 各模型数据导出脚本
│   ├── augment_vad_data.py           # 数据增强
│   ├── train_evaluator_v2.py         # 训练 Evaluator
│   ├── train_correction_policy.py    # 训练 Policy
│   ├── expA_relaxed.py               # 实验 A: SafetyGuard only
│   ├── expB_relaxed.py               # 实验 B: SafetyGuard + STAPOGate
│   ├── expC_relaxed.py               # 实验 C: SafetyGuard + LearnedUpdateGate
│   ├── inference_with_correction.py  # 在线推理
│   └── ...                           # 其他分析脚本
│
├── data/                              # 数据目录
│   ├── vad_dumps/                    # VAD 导出数据
│   ├── sparsedrive_dumps/            # SparseDrive 导出数据
│   ├── diffusiondrive_dumps/         # DiffusionDrive 导出数据
│   └── ...
│
├── experiments/                       # 实验输出
│   └── ...
│
└── docs/                              # 文档
    ├── 多模型集成完成报告.md
    ├── Scripts脚本完整分析.md
    └── ...
```

---

## 集成新模型

### 集成步骤

集成新模型只需 **4 步**，工作量约 2-6 小时：

```
步骤 1: 创建 Dump 脚本（1-2小时）
  ↓
步骤 2: 实现 Adapter（1-2小时）
  ↓
步骤 3: 注册到 dataloader（5分钟）
  ↓
步骤 4: 验证数据质量（1-2小时）
```

### 详细指南

#### 步骤 1: 创建 Dump 脚本

参考 `scripts/dump_vad_inference.py`，为您的模型创建导出脚本。

**输出要求**: `.pt` 文件需包含：
```python
{
    'sample_idx': int,           # 样本索引
    'scene_token': str,          # 场景标识
    'planner_outputs': {...},    # 模型原始输出
    'gt_plan': [T, 2],          # GT 轨迹（ego-centric 绝对坐标）
    'ego_fut_cmd': [M],          # 驾驶命令（可选）
}
```

#### 步骤 2: 实现 Adapter

创建 `planning_interface/adapters/my_model_adapter.py`：

```python
from .base_adapter import BasePlanningAdapter

class MyModelPlanningAdapter(BasePlanningAdapter):
    """MyModel → PlanningInterface 适配器。"""
    
    def extract_scene_token(self, planner_outputs):
        """提取场景特征 [D]"""
        # 从 planner_outputs 中提取并池化特征
        ...
    
    def extract_reference_plan(self, planner_outputs, ego_fut_cmd=None):
        """提取参考轨迹 [T, 2]（ego-centric 绝对坐标）"""
        # 注意：
        # - 位移增量需要 cumsum 转绝对坐标
        # - 全局坐标需要减去起点
        ...
    
    def extract_plan_confidence(self, planner_outputs, ego_fut_cmd=None):
        """提取置信度 [K]（可选）"""
        ...
    
    def extract_safety_features(self, planner_outputs):
        """提取安全特征（可选）"""
        ...
```

#### 步骤 3: 注册 Adapter

在 `data/dataloader.py` 的 `_get_adapter_class()` 函数中添加：

```python
from E2E_RL.planning_interface.adapters.my_model_adapter import (
    MyModelPlanningAdapter,
)

adapter_map = {
    'vad': VADPlanningAdapter,
    # ... 其他 adapter
    'mymodel': MyModelPlanningAdapter,  # 新增
}
```

#### 步骤 4: 验证

```bash
# 1. 导出数据（在原模型环境中）
conda activate my_model_env
python scripts/dump_mymodel_inference.py --max_samples 10

# 2. 验证数据（在 E2E_RL 环境中）
conda activate e2e_rl
python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader('data/mymodel_dumps', adapter_type='mymodel')
for batch in loader:
    gt_end = batch['gt_plan'][:, -1, :].norm(dim=-1).mean()
    ref_end = batch['interface'].reference_plan[:, -1, :].norm(dim=-1).mean()
    print(f'GT: {gt_end:.2f}m, Ref: {ref_end:.2f}m')
    break
"
```

> 📖 详细教程: [Scripts脚本完整分析.md](docs/Scripts脚本完整分析.md)

---

## 坐标系约定

所有数据使用 **ego-centric 绝对坐标**：
- 原点 = 自车当前位置 (t=0)
- 坐标系 = 自车朝向

### 验证方法

```python
# GT 终点应该在 15-25m（正常驾驶）
gt_end_dist = gt[:, -1, :].norm(dim=-1)
print(f'GT终点距原点: {gt_end_dist.mean():.2f}m')  # 期望 15-25m

# Ref 终点应该接近 GT
ref_end_dist = ref[:, -1, :].norm(dim=-1)
print(f'Ref终点距原点: {ref_end_dist.mean():.2f}m')  # 期望 15-25m

# Correction = GT - Ref 应该在 0-15m
correction = (gt - ref)[:, -1, :].norm(dim=-1)
print(f'Correction: {correction.mean():.2f}m')  # 期望 0-15m
```

### 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| GT 终点 > 30m | 使用了全局坐标 | 减去起点坐标 |
| GT 终点 < 5m | 使用了位移增量 | 添加 cumsum |
| Ref 和 GT 差距大 | 坐标系不一致 | 统一为 ego-centric |

---

## 实验结果

### UpdateEvaluator 性能

| 指标 | 值 | 说明 |
|------|-----|------|
| Spearman (gain) | 0.717 | 预测排序与真实排序高度相关 |
| Kendall Tau | 0.561 | 56.1% 的成对比较正确 |
| Spearman (risk) | 0.974 | 风险预测几乎完美 |

### A/B 对比实验

| 实验 | 配置 | retained_adv | retention |
|------|------|-------------|-----------|
| A | SafetyGuard only | -1.1054 | 46.51% |
| B | SafetyGuard + STAPOGate | -1.0830 | 46.79% |
| **C** | **SafetyGuard + LearnedUpdateGate** | **-1.0784** | **45.93%** |

### 三层防御体系

```
┌─────────────────────────────────────────┐
│  1. SafetyGuard（硬底线）                │
│     物理约束检查，任何违规 → 直接拒绝    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  2. LearnedUpdateGate（主判断）         │
│     预测 gain/risk，筛选高质量修正       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  3. 接受修正                             │
└─────────────────────────────────────────┘
```

---

## 常见问题

### Q: 新模型需要修改哪些文件？

**A**: 只需要：
1. **Dump 脚本**: 在原模型项目中运行推理，保存 .pt 文件
2. **Adapter**: 实现 `planning_interface/adapters/my_model_adapter.py`
3. **注册**: 在 `data/dataloader.py` 添加一行

训练和推理代码**完全不需要修改**。

---

### Q: 为什么需要两个 conda 环境？

**A**: 
- **原模型环境**: 包含 mmdet3d, NAVSIM 等模型特定依赖
- **E2E_RL 环境**: 只包含基础依赖（torch, numpy）

导出阶段需要原模型的代码和依赖，而训练/推理阶段只需要读取 .pt 文件，无需原模型依赖。

---

### Q: 如何配置训练参数？

**A**: 训练脚本完全支持命令行参数！

```bash
# 查看所有可用参数
python scripts/train_evaluator_v2.py --help
python scripts/expC_relaxed.py --help

# 使用命令行参数
python scripts/train_evaluator_v2.py \
    --data_dir data/my_model_dumps_full \
    --output_dir experiments/my_model_evaluator \
    --epochs 100 \
    --lr 1e-4
```

如果命令行参数不够用，仍然可以编辑脚本中的 `DEFAULT_CONFIG` 字典。

---

### Q: 如何快速集成同类框架的模型？

**A**: 如果新模型与已有模型框架相同（如 VAD → VADv2），可以直接复用：

```bash
# 1. 复制文件
cp scripts/dump_vad_inference.py scripts/dump_vadv2_inference.py
cp planning_interface/adapters/vad_adapter.py planning_interface/adapters/vadv2_adapter.py

# 2. 修改类名和路径
sed -i 's/VAD/VADv2/g' scripts/dump_vadv2_inference.py
sed -i 's/VADPlanningAdapter/VADv2PlanningAdapter/g' planning_interface/adapters/vadv2_adapter.py

# 3. 注册到 dataloader.py
# 添加 import 和 adapter_map 条目

# 4. 测试
```

集成时间：**2 小时**（vs 新框架 4-6 小时）

---

### Q: 数据坐标系不一致怎么办？

**A**: 检查 Adapter 的 `extract_reference_plan()` 方法：
- 位移增量 → 使用 `cumsum()` 转绝对坐标
- 全局坐标 → 减去起点坐标
- 验证终点距离在 15-25m

---

### Q: advantage 全为负？

**A**: 正常现象。LearnedUpdateGate 筛选**相对较好**的样本，不是让所有样本变好。实测正 gain 样本仅占 26%。

---

### Q: 如何对比不同模型的性能？

**A**: 使用相同的训练流程：

```bash
# 1. 为每个模型导出数据
python scripts/dump_vad_inference.py --output_dir data/vad_dumps
python scripts/dump_sparsedrive_inference.py --output_dir data/sparsedrive_dumps

# 2. 使用相同的训练脚本
python scripts/train_evaluator_v2.py  # 修改 CONFIG['data']['data_dir']

# 3. 对比实验结果
python scripts/compare_experiments.py
```

---

## 📚 相关文档

- [多模型集成完成报告](docs/多模型集成完成报告.md) - 集成状态总览
- [Scripts脚本完整分析](docs/Scripts脚本完整分析.md) - 30个脚本的详细说明
- [DiffusionDrive数据处理改进](docs/DiffusionDrive数据处理改进.md) - 一步导出+转换
- [E2E模型集成计划](docs/E2E模型集成计划.md) - 分阶段集成计划

---

**项目状态**: ✅ 活跃开发中  
**已集成模型**: 7 个  
**核心特性**: 🎯 模型无关设计  
**最后更新**: 2025-04-09
