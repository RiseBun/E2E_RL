# E2E_RL: 端到端强化学习轨迹修正框架

通过强化学习学习最优的轨迹修正策略，提升自动驾驶规划质量。

---

## 目录

- [Pipeline](#pipeline)
- [项目结构](#项目结构)
- [集成新模型](#集成新模型)
- [坐标系约定](#坐标系约定)
- [关键文件说明](#关键文件说明)
- [常见问题](#常见问题)

---

## Pipeline

### 整体流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据准备                                       │
│  模型在线推理 → dump 数据（包含预测轨迹、GT、scene_token）              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Adapter (模型相关)                              │
│  将模型输出转换为统一的 PlanningInterface                               │
│  - 提取 scene_token: [B, D] 场景编码                                 │
│  - 提取 reference_plan: [B, T, 2] ego-centric 绝对坐标               │
│  - 提取 plan_confidence: [B, 1] 置信度                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PlanningInterface (统一接口)                      │
│  封装 scene_token, reference_plan, plan_confidence 等字段            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CorrectionPolicy (策略网络)                         │
│  高斯策略网络，输入 state，输出修正量 correction ~ N(μ, σ)           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         三层防御级联                                    │
│  1. SafetyGuard: 物理约束检查（残差范数、单步位移、速度）            │
│  2. STAPOGate: 过滤虚假更新（正 advantage + 低概率 + 低熵）         │
│  3. LearnedUpdateGate: 基于 Evaluator 预测选择高质量更新              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Policy Gradient 训练                              │
│  Stage 1: BC 预热 - 最大化 GT correction 的对数概率                  │
│  Stage 2: RL 训练 - masked_pg_loss - entropy_coef * H(π)           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
训练阶段:
    batch['interface'] → policy.sample() → correction
    → 计算 advantage = r(ref + correction) - r(ref)
    → SafetyGuard → STAPOGate → LearnedUpdateGate
    → masked loss → backward

推理阶段:
    interface → policy.act() → correction → corrected_plan
```

---

## 项目结构

```
E2E_RL/
├── planning_interface/              # 统一接口层（模型无关）
│   ├── interface.py                # PlanningInterface 定义
│   └── adapters/                  # 模型适配器（模型相关）
│       ├── base_adapter.py         # 抽象基类
│       ├── vad_adapter.py          # VAD 适配器
│       └── diffusiondrive_adapter.py  # DiffusionDrive 适配器
│
├── data/                          # 数据加载
│   └── dataloader.py               # 通用 DataLoader（自动选择 Adapter）
│
├── correction_policy/              # 核心策略
│   ├── actor.py                    # GaussianCorrectionActor
│   ├── policy.py                  # CorrectionPolicy 统一接口
│   └── losses.py                   # BC + PG loss
│
├── update_selector/               # 策略学习门控
│   ├── safety_guard.py             # SafetyGuard
│   ├── stapo_gate.py               # STAPOGate
│   ├── learned_update_gate.py      # LearnedUpdateGate
│   └── update_evaluator.py         # UpdateEvaluator（预测 gain/risk）
│
├── rl_trainer/                    # 训练器
│   └── correction_policy_trainer.py
│
├── refinement/                    # 奖励计算
│   └── reward_proxy.py
│
├── scripts/                      # 脚本
│   ├── dump_vad_inference.py     # VAD 数据 dump
│   ├── expA_relaxed.py          # 实验 A (SafetyGuard only)
│   ├── expB_relaxed.py          # 实验 B (SafetyGuard + STAPOGate)
│   ├── expC_relaxed.py          # 实验 C (SafetyGuard + LearnedUpdateGate)
│   └── train_evaluator_v2.py    # 训练 UpdateEvaluator
│
└── configs/                     # 配置文件
    ├── correction_policy.yaml
    └── update_evaluator.yaml
```

---

## 集成新模型

### 核心思想

**所有模型相关的处理都在 Adapter 中完成**。dataloader、训练器、推理代码都只依赖 `PlanningInterface`，无需修改。

```
新模型 → 只需写一个 Adapter → 打通训练 + 推理
```

### Adapter 要求

Adapter 必须继承 `BasePlanningAdapter`，实现以下四个方法：

```python
class BasePlanningAdapter(ABC):
    @abstractmethod
    def extract_scene_token(self, planner_outputs) -> torch.Tensor:
        """提取场景编码

        Returns:
            torch.Tensor: [B, D] 场景特征向量
        """
        pass

    @abstractmethod
    def extract_reference_plan(
        self,
        planner_outputs,
        ego_fut_cmd=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """提取参考轨迹

        Args:
            planner_outputs: 模型原始输出字典
            ego_fut_cmd: 可选的命令信号

        Returns:
            (reference_plan, candidate_plans):
                - reference_plan: [B, T, 2] ego-centric 绝对坐标
                - candidate_plans: [B, M, T, 2] 或 None（可选）
        """
        pass

    @abstractmethod
    def extract_plan_confidence(self, planner_outputs, ego_fut_cmd=None) -> Optional[torch.Tensor]:
        """提取置信度

        Returns:
            torch.Tensor: [B, 1] 置信度，或 None
        """
        pass

    @abstractmethod
    def extract_safety_features(self, planner_outputs) -> Optional[Dict[str, torch.Tensor]]:
        """提取安全特征（可选）

        Returns:
            Dict[str, torch.Tensor] 或 None
        """
        pass
```

### Adapter 实现示例

```python
# planning_interface/adapters/my_model_adapter.py
from .base_adapter import BasePlanningAdapter
import torch
from typing import Any, Dict, Optional

class MyModelAdapter(BasePlanningAdapter):
    """MyModel → PlanningInterface 适配器"""

    def __init__(self, scene_pool: str = 'mean'):
        self.scene_pool = scene_pool

    def extract_scene_token(self, planner_outputs: Dict) -> torch.Tensor:
        """从 BEV 特征池化得到场景 token"""
        if 'bev_feature' in planner_outputs:
            bev = planner_outputs['bev_feature']  # [B, C, H, W]
            return bev.mean(dim=(-2, -1))         # [B, C]
        raise KeyError('无法提取 scene_token')

    def extract_reference_plan(
        self,
        planner_outputs: Dict,
        ego_fut_cmd=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """提取参考轨迹"""
        trajectory = planner_outputs['trajectory']  # [B, T, 2] 或 [T, 2]

        # 如果是位移增量，需要 cumsum
        if self._is_delta(planner_outputs):
            trajectory = torch.cumsum(trajectory, dim=-2)

        # 确保 [B, T, 2]
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)

        return trajectory, None

    def extract_plan_confidence(self, planner_outputs, ego_fut_cmd=None) -> Optional[torch.Tensor]:
        """提取置信度"""
        if 'confidence' in planner_outputs:
            return planner_outputs['confidence']
        return torch.ones(1, 1)

    def extract_safety_features(self, planner_outputs) -> Optional[Dict]:
        return None

    def _is_delta(self, planner_outputs) -> bool:
        """判断是否是位移增量格式"""
        return planner_outputs.get('is_delta', False)
```

### 注册 Adapter

修改 `data/dataloader.py` 的 `_get_adapter_class()` 函数：

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
    # ...
```

### 训练

```bash
# 使用新的 adapter_type
python scripts/expC_relaxed.py \
    --data_dir data/my_model_dumps \
    --adapter_type mymodel \
    --num_epochs 15
```

---

## 坐标系约定

### ego-centric 绝对坐标

所有轨迹必须使用 ego-centric 绝对坐标：

| 属性 | 值 |
|------|-----|
| 原点 | 自车当前位置 (t=0) |
| X轴 | 前进方向 |
| Y轴 | 左侧方向 |

```
                    Y
                    ↑
                    |
                    |
                    |--------→ X
                   ego
```

### Adapter 坐标系处理

| 模型输出格式 | Adapter 处理 |
|-------------|-------------|
| 位移增量 [T, 2] | cumsum 转绝对坐标 |
| 全局绝对坐标 [T, 2] | 减去起点转 ego-centric |
| ego-centric 绝对坐标 | 直接使用 |

### GT 坐标系

| 模型 | GT 坐标系 | gt_in_ego_frame |
|------|-----------|----------------|
| VAD | 全局坐标 | False |
| DiffusionDrive | ego-centric | True |

---

## 关键文件说明

| 文件 | 说明 |
|------|------|
| `planning_interface/interface.py` | PlanningInterface 定义，包含字段说明 |
| `planning_interface/adapters/base_adapter.py` | Adapter 抽象基类，定义接口规范 |
| `planning_interface/adapters/vad_adapter.py` | VAD Adapter 实现，参考学习 |
| `data/dataloader.py` | 通用 DataLoader，`build_planner_dataloader()` |
| `correction_policy/actor.py` | GaussianCorrectionActor 网络结构 |
| `update_selector/learned_update_gate.py` | LearnedUpdateGate 实现 |
| `refinement/reward_proxy.py` | 奖励计算函数 |

---

## 常见问题

### Q: 新模型需要修改哪些文件？

**A**: 只需要：
1. 创建 `planning_interface/adapters/my_model_adapter.py`
2. 在 `data/dataloader.py` 注册

**不需要修改**：训练器、推理代码、Policy 等。

### Q: 如何判断 Adapter 实现是否正确？

**A**: 运行验证代码检查坐标系：

```python
from data.dataloader import build_planner_dataloader

loader = build_planner_dataloader('data/my_model_dumps', adapter_type='mymodel')
for batch in loader:
    gt = batch['gt_plan']           # [B, T, 2]
    ref = batch['interface'].reference_plan  # [B, T, 2]

    # 检查：GT 终点应该在 15-25m 范围（正常驾驶）
    gt_end_dist = gt[:, -1, :].norm(dim=-1)
    print(f'GT终点距原点: {gt_end_dist.mean():.2f}m (期望 15-25m)')

    # 检查：GT correction = gt - ref 应该在 0-15m 范围
    correction = (gt - ref)[:, -1, :].norm(dim=-1)
    print(f'GT correction: {correction.mean():.2f}m (期望 0-15m)')
    break
```

### Q: 数据坐标系不一致导致训练失败？

**A**: 检查 Adapter 的 `extract_reference_plan()` 是否正确处理：
- 位移增量 → cumsum
- 全局坐标 → 减去起点

### Q: advantage 全为负？

**A**: 正常现象。LearnedUpdateGate 的作用是**选择相对较好的样本**，不是让所有样本变好。

### Q: 如何判断训练有效？

**A**: 检查：
1. BC 阶段 loss 下降
2. `retained_adv > filtered_adv`
