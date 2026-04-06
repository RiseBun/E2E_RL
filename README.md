# E2E_RL: 端到端强化学习轨迹修正框架

通过强化学习学习最优的轨迹修正策略，提升自动驾驶规划质量。**训练和推理对任何 E2E 规划模型都是模型无关的**。

---

## 目录

- [快速开始](#快速开始)
- [Pipeline](#pipeline)
- [运行示例 (VAD)](#运行示例-vad)
- [项目结构](#项目结构)
- [集成新模型](#集成新模型)
- [坐标系约定](#坐标系约定)
- [关键文件说明](#关键文件说明)
- [实验结果](#实验结果)
- [推理流程](#推理流程)
- [常见问题](#常见问题)

---

## 快速开始

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

# 1. 准备 dump 数据
python scripts/dump_vad_inference.py --config projects/configs/VAD/VAD_base_e2e.py \
    --checkpoint /path/to/vad_epoch_xxx.pth --output_dir data/vad_dumps

# 2. 训练 UpdateEvaluator（必需）
python scripts/train_evaluator_v2.py --output_dir experiments/update_evaluator

# 3. 训练 CorrectionPolicy
python scripts/expC_relaxed.py --output_dir experiments/correction_policy

# 4. 推理
python scripts/inference_with_correction.py \
    --checkpoint experiments/correction_policy/policy_final.pth \
    --evaluator experiments/update_evaluator/update_evaluator_final.pth \
    --data_dir data/vad_dumps
```

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     E2E 规划模型 (VAD / DiffusionDrive / ...)    │
└─────────────────────────────────────────────────────────────────┘
                                ↓ dump 数据
┌─────────────────────────────────────────────────────────────────┐
│                     Adapter (模型无关)                            │
│    统一 PlanningInterface: scene_token, reference_plan, ...      │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     RL 训练流程                                   │
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │ Update       │ →  │ Correction   │ →  │ 三层防御     │     │
│   │ Evaluator    │    │ Policy       │    │ Gate         │     │
│   │ (预测 gain   │    │ (学习修正)   │    │ (SafetyGuard │     │
│   │  和 risk)    │    │              │    │  + Learned)  │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     在线推理                                      │
│    Policy 输出 correction → 三层防御检查 → 修正后轨迹             │
└─────────────────────────────────────────────────────────────────┘
```
## 筛选器模型工作原理
```
UpdateEvaluator = 评分器/裁判
  ├─ 训练出来一个神经网络
  ├─ 输入: 候选修正
  └─ 输出: 这个修正有多好 (gain) 和有多危险 (risk)

LearnedUpdateGate = 筛选器/门卫
  ├─ 使用 UpdateEvaluator 的评分
  ├─ 逻辑: 只放行评分高的修正
  └─ 策略: 保留 top 30%，过滤掉 70%

整体 = 智能筛选系统
  ├─ UpdateEvaluator 负责"判断好坏"
  └─ LearnedUpdateGate 负责"执行筛选"
```

**核心设计**：所有模块通过 `PlanningInterface` 解耦，新增模型只需实现 Adapter。

---

## 运行示例 (VAD)

### Step 1: 准备 dump 数据

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

python scripts/dump_vad_inference.py \
    --config projects/configs/VAD/VAD_base_e2e.py \
    --checkpoint /path/to/vad_epoch_xxx.pth \
    --output_dir data/vad_dumps \
    --max_samples 5000
```

### Step 2: 验证数据加载

```bash
python -c "
from data.dataloader import build_vad_dataloader
loader = build_vad_dataloader('data/vad_dumps', batch_size=8)
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

### Step 3: 训练 UpdateEvaluator（必需）

**UpdateEvaluator 预测修正的 gain（收益）和 risk（风险），用于筛选高质量训练样本。**
> 实测：正 gain 样本仅占 26%，73% 的修正是无效的，必须筛选才能有效训练。

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

python scripts/train_evaluator_v2.py \
    --output_dir experiments/update_evaluator_v4_5k_samples \
    --num_epochs 50
```

### Step 4: 训练 CorrectionPolicy

三种实验配置可选：

```bash
# 实验 A: SafetyGuard only（baseline）
python scripts/expA_relaxed.py --num_epochs 15 --bc_epochs 3

# 实验 B: SafetyGuard + STAPOGate
python scripts/expB_relaxed.py --num_epochs 15 --bc_epochs 3

# 实验 C: SafetyGuard + LearnedUpdateGate（✅ 推荐配置）
python scripts/expC_relaxed.py --num_epochs 15 --bc_epochs 3
```

### Step 5: 推理

```bash
python scripts/inference_with_correction.py \
    --checkpoint experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth \
    --evaluator experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth \
    --data_dir data/vad_dumps
```

---

## 项目结构

```
E2E_RL/
├── planning_interface/              # 统一接口层（模型无关）
│   ├── interface.py                  # PlanningInterface 定义
│   └── adapters/                     # 模型适配器
│       ├── base_adapter.py          # 抽象基类
│       ├── vad_adapter.py           # VAD 适配器
│       └── diffusiondrive_adapter.py # DiffusionDrive 适配器
├── data/
│   ├── dataloader.py                # 通用 DataLoader
│   ├── vad_dataset.py               # VAD 数据集
│   └── vad_dumps/                    # dump 数据目录
├── correction_policy/                 # RL Policy
│   ├── policy.py                     # CorrectionPolicy
│   ├── actor.py                      # GaussianCorrectionActor
│   └── losses.py                     # PPO Loss
├── update_selector/                  # 门控选择器
│   ├── safety_guard.py              # SafetyGuard（硬底线）
│   ├── stapo_gate.py                # STAPOGate（规则）
│   └── update_evaluator.py         # UpdateEvaluator（学习）
├── refinement/                        # 奖励计算
│   └── reward_proxy.py
├── rl_trainer/                        # 训练器
│   └── correction_policy_trainer.py
├── scripts/                           # 脚本
│   ├── dump_vad_inference.py        # 数据准备
│   ├── train_evaluator_v2.py        # 训练 Evaluator
│   ├── expC_relaxed.py              # 训练 Policy
│   └── inference_with_correction.py  # 在线推理
└── experiments/                       # 实验输出
```

---

## 集成新模型

### 为什么模型无关？

所有模块通过 `PlanningInterface` 解耦：
- Adapter 负责模型输出 → Interface 的转换
- 训练器和推理代码只依赖 Interface，不依赖具体模型

### 集成步骤

**1. 创建 Adapter**

在 `planning_interface/adapters/` 下创建新文件：

```python
from .base_adapter import BaseAdapter

class MyModelAdapter(BaseAdapter):
    """MyModel 适配器"""

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

**2. 注册 Adapter**

在 `data/dataloader.py` 中注册：

```python
def _get_adapter_class(adapter_type: str):
    from .vad_adapter import VADPlanningAdapter
    from .diffusiondrive_adapter import DiffusionDrivePlanningAdapter
    from .my_model_adapter import MyModelAdapter  # 新增

    adapter_map = {
        'vad': VADPlanningAdapter,
        'diffusiondrive': DiffusionDrivePlanningAdapter,
        'mymodel': MyModelAdapter,  # 新增
    }
    ...
```

**3. 验证**

```bash
python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader('data/my_model_dumps', adapter_type='mymodel')
for batch in loader:
    gt = batch['gt_plan']
    ref = batch['interface'].reference_plan
    # 检查：GT 终点应该在 15-25m 范围
    print(f'GT终点距原点: {gt[:, -1, :].norm(dim=-1).mean():.2f}m')
    break
"
```

### Adapter 要求

| 方法 | 必需 | 说明 |
|------|------|------|
| `extract_scene_token` | ✅ | 场景特征 [D] |
| `extract_reference_plan` | ✅ | 参考轨迹 [T, 2]，ego-centric 绝对坐标 |
| `extract_plan_confidence` | ❌ | 置信度 [1]，用于策略输入 |
| `extract_safety_features` | ❌ | 安全特征，用于 SafetyGuard |

---

## 坐标系约定

**重要**：所有数据使用 **ego-centric 绝对坐标**。

```
ego-centric 绝对坐标：
- 原点 = 自车当前位置 (t=0)
- 坐标系 = 自车朝向
- 参考轨迹 = cumsum(位移增量) 或 全局坐标 - 起点
```

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

## 关键文件说明

| 文件 | 说明 |
|------|------|
| `planning_interface/interface.py` | PlanningInterface 定义 |
| `planning_interface/adapters/base_adapter.py` | Adapter 抽象基类 |
| `data/dataloader.py` | 通用 DataLoader |
| `correction_policy/policy.py` | CorrectionPolicy |
| `correction_policy/actor.py` | GaussianCorrectionActor |
| `update_selector/safety_guard.py` | SafetyGuard |
| `update_selector/update_evaluator.py` | UpdateEvaluator + LearnedUpdateGate |
| `scripts/inference_with_correction.py` | 在线推理脚本 |

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

**实验 C 优势**：
- retained_adv 比 baseline 提升 **2.4%**
- 收敛速度提升 **2-3 倍**
- 碰撞率降低 **30-50%**

### 三层防御体系

```
┌─────────────────────────────────────────┐
│  1. SafetyGuard（硬底线）                │
│     - 物理约束检查                       │
│     - 任何违规 → 直接拒绝                │
└─────────────────────────────────────────┘
                    ↓ 通过
┌─────────────────────────────────────────┐
│  2. LearnedUpdateGate（主判断）          │
│     - 预测 gain 和 risk                  │
│     - gain < tau_gain → 拒绝            │
│     - risk > tau_risk → 拒绝            │
└─────────────────────────────────────────┘
                    ↓ 通过
┌─────────────────────────────────────────┐
│  3. 接受修正                             │
└─────────────────────────────────────────┘
```

---

## 推理流程

### 基本用法

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL

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
| `--disable_stapo_gate` | 禁用 STAPOGate |

### 训练好的模型

| 模型 | 路径 |
|------|------|
| CorrectionPolicy (Exp C) | `experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth` |
| UpdateEvaluator | `experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth` |

---

## 常见问题

### Q: 新模型需要修改哪些文件？

**A**: 只需要：
1. 创建 `planning_interface/adapters/my_model_adapter.py`
2. 在 `data/dataloader.py` 注册

### Q: 数据坐标系不一致？

**A**: 检查 Adapter 的 `extract_reference_plan()`：
- 位移增量 → cumsum
- 全局坐标 → 减去起点

### Q: advantage 全为负？

**A**: 正常现象。LearnedUpdateGate 筛选相对较好的样本，不是让所有样本变好。

### Q: 如何判断训练有效？

**A**: 检查 `retained_adv > filtered_adv`

### Q: 训练和推理对不同 E2E 模型是否通用？

**A**: 完全通用。所有模块通过 PlanningInterface 解耦，只需实现对应的 Adapter。
