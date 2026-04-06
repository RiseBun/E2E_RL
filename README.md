# E2E_RL: 端到端强化学习轨迹修正框架

通过强化学习学习最优的轨迹修正策略，提升自动驾驶规划质量。**训练和推理对任何 E2E 规划模型都是模型无关的**。

---

## 目录

- [Pipeline](#pipeline)
- [运行示例 (VAD)](#运行示例-vad)
- [项目结构](#项目结构)
- [集成新模型](#集成新模型)
- [坐标系约定](#坐标系约定)
- [实验结果](#实验结果)
- [推理流程](#推理流程)
- [常见问题](#常见问题)


---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1: Dump 数据（模型特定）                                     │
│   └─ 在 E2E 模型项目中运行推理脚本                               │
│   └─ 输出: .pt 文件 (scene_token, ego_fut_preds, ego_fut_trajs) │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2: Adapter（模型无关）                                       │
│   └─ 解析 .pt 文件 → PlanningInterface                           │
│   └─ 新模型只需实现 Adapter                                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3: RL 训练和推理（模型无关）                                 │
│   └─ 所有模块通过 PlanningInterface 解耦                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 基于vad的运行示例，其他模型的过程一致 

### Step 1: 放置 VAD 项目

```bash
cd ~/E2E_RL/projects

# 克隆 VAD
git clone  VAD项目路径

# 安装依赖
cd VAD && pip install -r requirements.txt
```

### Step 2: Dump 数据

运行提供的 dump 脚本，可以收集模型原始输出：

```bash
cd ~/E2E_RL

# 设置 PYTHONPATH，让 dump 脚本能找到模型代码
export PYTHONPATH=~/E2E_RL/projects/VAD:$PYTHONPATH

# 运行 dump（脚本在 E2E_RL/scripts/ 中）
python scripts/dump_vad_inference.py \
    --config ~/E2E_RL/projects/VAD/projects/configs/VAD/VAD_base_e2e.py \
    --checkpoint /path/to/vad_epoch_xxx.pth \
    --output_dir data/vad_dumps \
    --max_samples 5000
```

### Step 3: 验证数据加载（数据增强后也建议验证，修改路径即可）

```bash
cd ~/E2E_RL

python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader('data/vad_dumps', adapter_type='vad', batch_size=8)
for batch in loader:
    gt = batch['gt_plan']
    ref = batch['interface'].reference_plan
    print(f'GT终点距原点: {gt[:, -1, :].norm(dim=-1)[:3]}')
    print(f'ReF终点距原点: {ref[:, -1, :].norm(dim=-1)[:3]}')
    break
"
```

期望输出（ego-centric 坐标系）：
```
GT终点距原点: tensor([17.82, 18.37, 19.64])
Ref终点距原点: tensor([5.52, 27.34, 22.59])
```

### Step 4: 数据增强（可选）

```bash
cd ~/E2E_RL

python scripts/augment_vad_data.py \
    --input_dir data/vad_dumps \
    --output_dir data/vad_dumps_full \
    --samples_per_original 50 \
    --noise_scale 0.1 \
    --max_samples 5000
```

**参数说明**:
- `--input_dir`: 原始真实数据目录（dump_vad_inference.py 的输出）
- `--output_dir`: 增强后的数据目录
- `--samples_per_original`: 每个原始样本生成多少个增强样本（默认 50）
- `--noise_scale`: 噪声尺度，控制扰动大小（默认 0.1）
- `--max_samples`: 最大样本数（默认 5000）

**增强策略**:
```python
# 对 reference_plan 添加高斯噪声
noise = torch.randn_like(reference_plan) * noise_scale
augmented_reference_plan = reference_plan + noise

# 对 scene_token 添加高斯噪声
noise = torch.randn_like(scene_token) * noise_scale
augmented_scene_token = scene_token + noise

# GT 保持不变
gt_plan = gt_plan  # 不添加噪声
```

**预期输出**:
```
========================================
VAD Dump 数据扩充
========================================
输入目录: data/vad_dumps
输出目录: data/vad_dumps_full
每个原始样本扩充: 50 个
噪声尺度: 0.1
目标样本数: 5000
========================================

找到 101 个原始样本
扩充数据: 100%|██████████| 101/101 [00:30<00:00]

扩充完成！共生成 5001 个样本
输出目录: data/vad_dumps_full
```

> **⚠️ 注意**: 
> - 数据增强是可选的，如果已经有足够的真实数据（> 1000 帧），可以跳过
> - 增强数据包含人工噪声，最终性能可能略低于纯真实数据
> - 推荐：先用增强数据快速验证，再用真实数据微调

### Step 5: 训练 UpdateEvaluator

**UpdateEvaluator 预测修正的 gain 和 risk，用于筛选高质量训练样本。**
> 实测：正 gain 样本仅占 26%，73% 的修正是无效的，必须筛选。

```bash
cd ~/E2E_RL

python scripts/train_evaluator_v2.py \
    --data_dir data/vad_dumps_full \
    --output_dir experiments/update_evaluator_v4_5k_samples \
    --num_epochs 50
```

### Step 6: 训练 CorrectionPolicy

三种实验配置可选：

```bash
cd ~/E2E_RL

# ==========================================
# 实验 A: SafetyGuard only (Baseline)
# ==========================================
python scripts/expA_relaxed.py \
    --data_dir data/vad_dumps_full \
    --output_dir experiments/ab_comparison_v2/expA_safety_guard_only \
    --num_epochs 15 \
    --bc_epochs 3

# ==========================================
# 实验 B: SafetyGuard + STAPOGate
# ==========================================
python scripts/expB_relaxed.py \
    --data_dir data/vad_dumps_full \
    --output_dir experiments/ab_comparison_v2/expB_stapo_gate \
    --num_epochs 15 \
    --bc_epochs 3

# ==========================================
# 实验 C: SafetyGuard + LearnedUpdateGate (✅ 推荐)
# ==========================================
python scripts/expC_relaxed.py \
    --data_dir data/vad_dumps_full \
    --output_dir experiments/ab_comparison_v2/expC_learned_gate \
    --evaluator_ckpt experiments/update_evaluator_v4_5k_samples/evaluator_epoch_30.pth \
    --num_epochs 15 \
    --bc_epochs 3
```

### Step 7: 推理

```bash
cd ~/E2E_RL

python scripts/inference_with_correction.py \
    --checkpoint experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth \
    --evaluator experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth \
    --data_dir data/vad_dumps
```

---

## 项目结构

```
E2E_RL/
├── projects/                      # 放置 E2E 模型项目
│   ├── VAD/                     # VAD 项目
│   │   ├── projects/            # VAD 配置文件
│   │   ├── tools/               # VAD 工具脚本
│   │   └── ...
│   └── DiffusionDrive/          # DiffusionDrive 项目（可选）
├── planning_interface/            # 统一接口层（模型无关）
│   ├── interface.py              # PlanningInterface 定义
│   └── adapters/                # 模型适配器
│       ├── base_adapter.py      # 抽象基类
│       ├── vad_adapter.py       # VAD 适配器
│       └── diffusiondrive_adapter.py  # DiffusionDrive 适配器
├── data/
│   ├── dataloader.py            # 通用 DataLoader
│   └── vad_dumps/               # dump 数据目录
├── correction_policy/            # RL Policy
│   ├── policy.py                # CorrectionPolicy
│   ├── actor.py                 # GaussianCorrectionActor
│   └── losses.py                # PPO Loss
├── update_selector/              # 门控选择器
│   ├── safety_guard.py          # SafetyGuard（硬底线）
│   ├── stapo_gate.py            # STAPOGate（规则）
│   └── update_evaluator.py      # UpdateEvaluator + LearnedUpdateGate
├── scripts/
│   ├── train_evaluator_v2.py   # 训练 Evaluator
│   ├── expC_relaxed.py         # 训练 Policy
│   └── inference_with_correction.py  # 在线推理
└── experiments/                 # 实验输出
```

---

## 集成新模型

对于新模型，需要做两件事：

```
阶段 1: Dump 数据（模型特定）
  └─ 在模型项目中写 dump 脚本，保存输出为 .pt 文件

阶段 2: Adapter（模型无关）
  └─ 实现 Adapter，解析 .pt 文件
```

### 1. Dump 数据（模型特定）

**方式 A**：参考 `E2E_RL/scripts/dump_vad_inference.py` 编写 dump 脚本
**方式 B**：如果有原始模型输出，手动转换格式

Dump 输出 `.pt` 文件需要包含：
- `scene_token`: 场景特征
- `ego_fut_preds`: 规划轨迹（位移增量）
- `ego_fut_trajs`: GT 轨迹
- 其他元信息

### 2. Adapter（模型无关）

参考 `planning_interface/adapters/vad_adapter.py`：

```python
from .base_adapter import BaseAdapter

class MyModelAdapter(BaseAdapter):

    def extract_scene_token(self, planner_outputs: Dict) -> torch.Tensor:
        # 提取场景特征
        ...

    def extract_reference_plan(
        self,
        planner_outputs: Dict,
        ego_fut_cmd=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 提取参考轨迹（ego-centric 绝对坐标）
        # 注意：位移增量需要 cumsum，全局坐标需要减去起点
        ...

    def extract_plan_confidence(...) -> Optional[torch.Tensor]:
        # 提取置信度（可选）
        ...

    def extract_safety_features(...) -> Optional[Dict]:
        # 提取安全特征（可选）
        ...
```

### 3. 注册 Adapter

在 `data/dataloader.py` 中注册：

```python
adapter_map = {
    'vad': VADPlanningAdapter,
    'diffusiondrive': DiffusionDrivePlanningAdapter,
    'mymodel': MyModelAdapter,  # 新增
}
```

### 4. 验证

```bash
python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader('data/my_model_dumps', adapter_type='mymodel')
for batch in loader:
    gt = batch['gt_plan']
    ref = batch['interface'].reference_plan
    print(f'GT终点距原点: {gt[:, -1, :].norm(dim=-1).mean():.2f}m')
    break
"
```

---

## 坐标系约定

所有数据使用 **ego-centric 绝对坐标**：
- 原点 = 自车当前位置 (t=0)
- 坐标系 = 自车朝向

检查方法：
```python
# GT 终点应该在 15-25m（正常驾驶）
gt_end_dist = gt[:, -1, :].norm(dim=-1)
print(f'GT终点距原点: {gt_end_dist.mean():.2f}m')  # 期望 15-25m

# GT correction = gt - ref 应该在 0-15m
correction = (gt - ref)[:, -1, :].norm(dim=-1)
print(f'GT correction: {correction.mean():.2f}m')  # 期望 0-15m
```

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

## 推理流程

### 基本用法

```bash
cd ~/E2E_RL

# 完整三层防御推理
python scripts/inference_with_correction.py \
    --checkpoint experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth \
    --evaluator experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth \
    --data_dir data/vad_dumps \
    --max_samples 100

# 只用 SafetyGuard
python scripts/inference_with_correction.py \
    --checkpoint experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth \
    --data_dir data/vad_dumps \
    --disable_learned_gate

# 保存结果
python scripts/inference_with_correction.py \
    --checkpoint experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth \
    --evaluator experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth \
    --data_dir data/vad_dumps \
    --output_dir outputs/inference_results \
    --max_samples 1000
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--checkpoint` | CorrectionPolicy 模型路径 |
| `--evaluator` | UpdateEvaluator 模型路径（可选） |
| `--data_dir` | 数据目录 |
| `--batch_size` | 批大小，默认 1 |
| `--max_samples` | 最大处理样本数 |
| `--output_dir` | 输出目录（保存 JSON 结果） |
| `--disable_learned_gate` | 禁用 LearnedUpdateGate |

---

## 常见问题

### Q: 新模型需要修改哪些文件？

**A**: 只需要：
1. Dump 脚本：在模型项目中运行推理，保存 .pt 文件
2. Adapter：实现 `planning_interface/adapters/my_model_adapter.py`
3. 注册：在 `data/dataloader.py` 添加一行

### Q: 数据坐标系不一致？

**A**: 检查 Adapter 的 `extract_reference_plan()`：
- 位移增量 → cumsum
- 全局坐标 → 减去起点

### Q: advantage 全为负？

**A**: 正常现象。LearnedUpdateGate 筛选相对较好的样本，不是让所有样本变好。

### Q: 训练和推理对不同 E2E 模型是否通用？

**A**: 完全通用。所有模块通过 PlanningInterface 解耦，只需实现对应的 Adapter。
