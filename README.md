# E2E_RL: 模型无关的端到端强化学习微调框架

通过强化学习直接微调 E2E 规划模型，提升规划能力。

**核心流程**: 冻结 Backbone → 直接微调轨迹头 → 知识蒸馏部署

---

## 📋 目录

- [核心流程](#核心流程)
- [Phase 1 (可选): 外挂验证器](#phase-1-可选-外挂验证器)
- [Phase 2: 端到端微调](#phase-2-端到端微调)
- [Phase 3: 知识蒸馏](#phase-3-知识蒸馏)
- [快速开始](#快速开始)
- [已集成模型](#已集成模型)

---

## 核心流程

```
原始 E2E 模型
     │
     ▼
┌─────────────────────────────────────────┐
│  冻结 Backbone，只微调轨迹头 (LoRA)       │
│        │
│                                         │
│  1. 采样多条轨迹                         │
│  2. 计算 reward (PDM Score)             │
│  3. 计算 advantage: (r - μ) / σ         │
│  4. GRPO Loss: -exp(logp - logp.detach()) * adv │
│  5. 结合 IL Loss (L1 与 GT 距离)         │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  知识蒸馏: Teacher → Student             │
│  去除训练期外挂模块，得到纯模型            │
└─────────────────────────────────────────┘
     │
     ▼
  部署模型
```

---

## Phase 1 (可选): 外挂验证器

Phase 1 是**可选**的，用于快速验证和产生活动修正样本。

### 使用场景

- 想快速验证修正策略是否有效
- 需要产生活动修正样本用于分析
- 作为 Phase 2 的前置探索

### 训练

```bash
# 训练 UpdateEvaluator
python scripts/train_evaluator_v2.py \
    --data_dir data/vad_dumps_full \
    --output_dir experiments/vad_evaluator

# 训练 CorrectionPolicy
python scripts/expC_relaxed.py \
    --data_dir data/vad_dumps_full \
    --evaluator_ckpt experiments/vad_evaluator/evaluator_final.pth \
    --output_dir experiments/vad_policy
```

**注意**: Phase 1 训练后的模型是"外挂"形式，如需提升模型本体能力，请使用 Phase 2。

---

## Phase 2: Conservative E2E Post-Training

### 核心思想

**将有益修正内化为 E2E 模型本体的参数更新**，而非依赖外挂修正器。

### 关键组件

#### 1. Reward-Cost 分离设计

```python
class RewardCostSeparator:
    """
    不是简单加权和，而是区分:
    
    Reward 分支 (追求目标):
    - progress: 轨迹终点接近 GT
    - efficiency: 行驶效率
    - route_completion: 路线完成度
    
    Cost 分支 (必须约束的边界):
    - collision: 碰撞风险
    - offroad: 离道风险
    - comfort: 舒适度
    
    优化目标: maximize reward under bounded cost
    """
```

#### 2. Beneficial Update Filter

```python
class BeneficialUpdateFilter:
    """
    只保留同时满足以下条件的更新:
    1. reward_improvement > reward_margin
    2. cost_increase < cost_increase_threshold  
    3. kl_drift < kl_bound
    
    这确保了:
    - 真正有益 (不是假阳性)
    - 不引入额外风险 (cost 不上升)
    - 更新幅度受控 (不偏离原始策略太远)
    """
```

#### 3. LoRA Finetuning

```python
class HydraTrajHeadE2E:
    """
    增强规划头:
    - LoRA: 只训练低秩适配器，不动原始权重
    - Value Head: 估计 V(s)，用于 advantage 计算
    - 与 PlanningInterface 统一接口兼容
    """
```

### 训练流程

```bash
# Phase 2 E2E 微调
python scripts/train_e2e_finetuning.py \
    --model_type vad \
    --checkpoint /path/to/vad.pth \
    --lora_rank 16 \
    --epochs 50

# 或不使用 LoRA (全量微调)
python scripts/train_e2e_finetuning.py \
    --no_lora \
    --lr 1e-5
```

### Conservative RL 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_rank` | 16 | LoRA rank |
| `kl_target` | 0.01 | 目标 KL 散度 |
| `reward_margin` | 0.0 | 正增益阈值 |
| `cost_increase_threshold` | 0.1 | cost 允许的最大增量 |
| `reference_alpha` | 0.5 | 参考模型权重 |

---

## Phase 3: 知识蒸馏

### 核心思想

Phase 2 训练后的模型带有 Value Head、Reference Model 等训练期模块，Phase 3 将知识蒸馏回纯 E2E 模型。

### 蒸馏损失

```python
distill_loss = (
    λ_trajectory * L2(traj_student, traj_teacher) +
    λ_score * KL(score_student, score_teacher) +
    λ_feature * MSE(feature_student, feature_teacher)
)
```

详见 [蒸馏设计文档](docs/distillation_design.md)。

---

## 已集成模型

### 完全集成（7个）

| 模型 | 论文 | 框架 | Adapter | Dump 脚本 | 状态 |
|------|------|------|---------|-----------|------|
| **VAD** | ICLR 2024 | mmdet3d | `VADPlanningAdapter` | `dump_vad_inference.py` | ✅ |
| **VADv2** | ICLR 2026 | mmdet3d | `VADv2PlanningAdapter` | `dump_vadv2_inference.py` | ✅ |
| **SparseDrive** | ECCV 2024 | mmdet3d | `SparseDrivePlanningAdapter` | `dump_sparsedrive_inference.py` | ✅ |
| **SparseDriveV2** | arXiv 2024 | mmdet3d | `SparseDriveV2PlanningAdapter` | `dump_sparsedrivev2_inference.py` | ✅ |
| **DiffusionDrive** | arXiv 2024 | NAVSIM | `DiffusionDrivePlanningAdapter` | `dump_diffusiondrive_inference.py` | ✅ |
| **DiffusionDriveV2** | arXiv 2025 | NAVSIM | `DiffusionDriveV2PlanningAdapter` | `dump_diffusiondrivev2_inference.py` | ✅ |
| **UniAD** | CVPR 2023 | mmdet3d | `UniADPlanningAdapter` | `dump_uniad_inference.py` | ✅ |

---

## 环境配置

### E2E_RL 环境（必需）

```bash
conda create -n e2e_rl python=3.10 -y
conda activate e2e_rl

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scipy pyyaml tqdm
```

### 原模型环境（按需）

```bash
# VAD 环境
conda create -n vad python=3.10 -y
conda activate vad
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
cd projects/VAD && pip install -r requirements.txt

# DiffusionDrive 环境
conda create -n diffusiondrive python=3.10 -y
conda activate diffusiondrive
cd projects/DiffusionDrive
```

---

## 快速开始

### 1. 导出数据

```bash
# 激活模型环境
conda activate vad

# 导出 VAD 数据
python scripts/dump_vad_inference.py \
    --config projects/VAD/projects/configs/VAD/VAD_base_e2e.py \
    --checkpoint projects/VAD/VAD_base.pth \
    --data_root /path/to/nuscenes/ \
    --output_dir data/vad_dumps
```

### 2. Phase 2: 端到端微调 (核心)

```bash
conda activate e2e_rl

# LoRA 微调 (推荐)
python scripts/train_e2e_finetuning.py \
    --model_type vad \
    --checkpoint projects/VAD/VAD_base.pth \
    --lora_rank 16 \
    --epochs 50 \
    --output_dir experiments/e2e_finetuned
```

### 3. Phase 3: 知识蒸馏

```bash
python scripts/distill_e2e_rl.py \
    --model_type vad \
    --teacher_checkpoint experiments/e2e_finetuned/final_model.pth \
    --student_checkpoint projects/VAD/VAD_base.pth \
    --data_dir data/vad_dumps \
    --output_dir experiments/distilled
```

---

## VAD 完整跑通结果

### 数据准备
- Checkpoint: `projects/VAD/VAD_base.pth` (675MB)
- 数据集: `data/vad_dumps/` (100 样本)

### Phase 2: E2E 微调 (5 epochs)

```
[Epoch 0] loss=-0.6104 reward=0.8121 cost=0.7324 safety=0.89%
[Epoch 1] loss=-0.7247 reward=0.8303 cost=0.7140 safety=0.89%
[Epoch 2] loss=-0.6987 reward=0.7906 cost=0.7015 safety=0.89%
[Epoch 3] loss=-0.7093 reward=0.8206 cost=0.7225 safety=0.89%
[Epoch 4] loss=-0.5883 reward=0.8074 cost=0.7171 safety=0.89%
[Val] reward=0.8113 cost=0.7010 FDE=6.96m
```

**输出**: `experiments/vad_finetuned/final_model.pth`

### Phase 3: 知识蒸馏 (3 epochs)

```
Epoch 1 完成 | 平均 Loss: 0.0054
Epoch 2 完成 | 平均 Loss: 0.0020
Epoch 3 完成 | 平均 Loss: 0.0008  ← 收敛良好
```

**输出**: `experiments/vad_distilled/`
```
├── checkpoint_epoch_3.pth
├── distilled_model_final.pth
└── distilled_model_merged.pth  # 可直接部署
```

### 训练曲线趋势
- Loss 从 -0.61 逐渐收敛
- Reward 稳定在 0.79~0.83
- 蒸馏 loss 从 0.0054 降至 0.0008

---

## 项目结构

```
E2E_RL/
├── projects/                          # E2E 模型项目
│   ├── VAD/                           # VAD
│   ├── DiffusionDrive/                 # DiffusionDrive
│   └── ...
│
├── planning_interface/                # 统一接口层
│   ├── interface.py                   # PlanningInterface
│   └── adapters/                      # 模型适配器
│
├── correction_policy/                 # Phase 1 策略
│   ├── policy.py                      # CorrectionPolicy
│   └── actor.py                       # GaussianCorrectionActor
│
├── update_selector/                   # Phase 1 筛选器
│   ├── safety_guard.py               # SafetyGuard
│   ├── stapo_gate.py                  # STAPOGate
│   └── update_evaluator.py            # UpdateEvaluator
│
├── e2e_finetuning/                    # Phase 2 端到端微调
│   ├── reward.py                      # 闭环奖励计算
│   ├── conservative_rl.py             # 保守 RL 更新
│   └── hydra_traj_head_e2e.py         # 增强规划头
│
├── refinement/                        # 奖励代理
│   └── reward_proxy.py                # 离线奖励函数
│
├── scripts/                           # 训练脚本
│   ├── dump_*.py                      # 数据导出
│   ├── train_evaluator_v2.py          # 训练 Evaluator
│   ├── expC_relaxed.py                # 训练 Policy
│   └── train_e2e_finetuning.py        # Phase 2 E2E 微调
│
└── experiments/                       # 实验输出
```

---

## 集成新模型

只需 4 步：

1. **创建 Dump 脚本** (`scripts/dump_my_model_inference.py`)
2. **实现 Adapter** (`planning_interface/adapters/my_model_adapter.py`)
3. **注册 Adapter** (`data/dataloader.py`)
4. **验证数据**

---

## 常见问题

### Q: Phase 1 和 Phase 2 的区别？

| 方面 | Phase 1 | Phase 2 |
|------|---------|---------|
| 模型参数 | Frozen | Trainable |
| 梯度流 | 停在 reference_plan | 完整回传 |
| 部署 | 需要外挂模块 | 单模型 |
| 训练复杂度 | 较低 | 较高 |

### Q: 为什么需要 LoRA？

- 减少可训练参数
- 防止灾难性遗忘
- 加速训练收敛

### Q: Phase 3 的作用？

去除训练期的辅助模块（Value Head、Reference Model），得到可直接部署的纯 E2E 模型。

---

**项目状态**: ✅ 活跃开发中 (Phase 2 核心已完成, Phase 3 已完成, Phase 1 可选)  
**已集成模型**: 7 个  
**核心特性**: 🎯 冻结 Backbone → 直接微调轨迹头 → 知识蒸馏部署
