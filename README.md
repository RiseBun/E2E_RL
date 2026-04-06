# E2E_RL: 端到端强化学习修正框架

## 概述

E2E_RL 是一个端到端强化学习修正框架，通过学习最优的轨迹修正策略来提升自动驾驶规划的质量。

**核心思想**：将修正问题建模为强化学习问题，策略网络学习在给定参考轨迹和场景上下文的情况下，输出最优的修正量。

---

## 核心架构

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         输入数据                                 │
│  scene_token (场景编码) + reference_plan (参考轨迹)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CorrectionPolicy                              │
│              (高斯策略网络 GaussianCorrectionActor)               │
│  输出: correction ~ N(μ, σ)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   三层防御级联 (Policy Learning Gate)             │
│                                                                 │
│  Layer 1: SafetyGuard  ─ 硬性物理约束检查                        │
│           • 残差范数限制                                         │
│           • 单步位移限制                                         │
│           • 速度上限                                             │
│                                                                 │
│  Layer 2: STAPOGate   ─ 虚假更新过滤                             │
│           • 正 advantage + 低概率 + 低熵 → 静音                  │
│                                                                 │
│  Layer 3: LearnedUpdateGate ─ 预测收益排序                        │
│           • UpdateEvaluator 预测 gain/risk                       │
│           • 保留 top-k 高收益更新                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Policy Gradient 更新                           │
│  loss = masked_pg_loss - entropy_coef * H(π)                    │
└─────────────────────────────────────────────────────────────────┘
```

### 推理流程

```
scene_token, ref_plan, conf = Adapter.extract(planner_output)
correction = Policy.act(scene_token, ref_plan, conf)  # 取均值
corrected_plan = ref_plan + correction
```

---

## 项目结构

```
E2E_RL/
├── correction_policy/          # 核心策略模块
│   ├── actor.py               # GaussianCorrectionActor (高斯策略网络)
│   ├── policy.py              # CorrectionPolicy (统一策略接口)
│   ├── losses.py              # BC + PG loss
│   └── __init__.py
│
├── update_selector/            # 策略学习门控模块
│   ├── update_evaluator.py    # UpdateEvaluator (预测 gain/risk)
│   ├── safety_guard.py        # SafetyGuard (硬性约束)
│   ├── stapo_gate.py          # STAPOGate (虚假更新过滤)
│   ├── learned_update_gate.py  # LearnedUpdateGate (学习型门控)
│   ├── candidate_generator.py  # 候选修正生成器
│   └── __init__.py
│
├── rl_trainer/                 # RL 训练器
│   ├── correction_policy_trainer.py
│   └── __init__.py
│
├── planning_interface/         # 统一接口层
│   ├── interface.py           # PlanningInterface
│   ├── extractor.py           # BaseExtractor
│   └── adapters/              # 规划器适配器 (VADAdapter 等)
│
├── data/                      # 数据加载
│   ├── vad_dataset.py
│   └── dataloader.py
│
├── refinement/                 # 奖励计算
│   └── reward_proxy.py
│
├── scripts/
│   ├── train_correction_policy.py   # 策略训练入口
│   ├── train_evaluator_v2.py       # Evaluator 训练入口
│   └── train_with_learned_gate.py   # 集成训练入口
│
└── configs/
    ├── correction_policy.yaml
    └── update_evaluator.yaml
```

---

## 核心模块详解

### 1. CorrectionPolicy (策略接口)

统一策略接口，封装高斯策略网络：

| 方法 | 说明 |
|------|------|
| `sample(interface)` | 训练时采样修正量 |
| `evaluate(interface, action)` | 计算 log_prob + entropy |
| `act(interface, deterministic=True)` | 推理时输出确定性修正 |
| `get_corrected_plan(interface)` | 直接输出修正后轨迹 |

### 2. GaussianCorrectionActor (高斯策略网络)

```
输入:
  - scene_token: [B, 256] 场景编码
  - reference_plan: [B, T, 2] 参考轨迹
  - plan_confidence: [B, 1] 置信度

输出:
  - mean: [B, T, 2] 修正量均值
  - log_std: [B, T, 2] 对数标准差
  - entropy: 策略熵
```

### 3. UpdateEvaluator (评估网络)

多目标回归网络，预测候选修正的收益和风险：

**网络结构**：
```
输入: scene_token + reference_plan + correction
  ↓
共享编码层 (MLP)
  ↓
多任务预测头:
  - pred_gain: [B, 1] 奖励增益
  - pred_collision: [B, 1] 碰撞风险变化
  - pred_offroad: [B, 1] 偏离道路风险变化
  - pred_comfort: [B, 1] 舒适度变化
  - pred_drift: [B, 1] 漂移风险变化
```

**Loss 函数**：
```
loss = α * MSE(pred_gain, y_gain) + β * Σ(risk_weight_i * MSE(pred_risk_i, y_risk_i))
```

**训练指标**：
| 指标 | 目标 | 含义 |
|------|------|------|
| spearman_gain | > 0.3 | 预测 gain 排序相关性 |
| kendall_gain | > 0.3 | 预测 gain 排序相关性 |
| retained_gain > filtered_gain | ✓ | 保留样本 gain 更高 |
| retained_risk < filtered_risk | ✓ | 保留样本 risk 更低 |

**当前训练结果**：
- Spearman: 0.736
- Kendall: 0.581
- 训练数据: 9600 样本
- 正 gain 比例: ~26%

### 4. SafetyGuard (硬性约束)

物理约束检查，违规则丢弃更新：

| 约束项 | 说明 |
|--------|------|
| residual_norm | 修正量 L2 范数上限 |
| step_limit | 单步位移上限 |
| velocity_limit | 速度上限 |
| total_displacement | 总位移上限 |

### 5. STAPOGate (虚假更新过滤)

识别并过滤虚假有益更新：

```
虚假更新条件 = (advantage > 0) AND (π(a|s) < τ_prob) AND (H < τ_entropy)
```

这类更新虽然看起来有益（正 advantage），但实际上是策略过尖导致的噪声。

### 6. LearnedUpdateGate (学习型门控)

基于 UpdateEvaluator 预测进行更新选择：

1. 对每个候选修正调用 UpdateEvaluator 获取预测 gain
2. 按预测 gain 排序
3. 保留 top-k (retention_ratio) 的更新
4. 其余 mask 为 0

### 7. CandidateCorrector (候选生成器)

生成训练所需的候选修正：

| 策略 | 权重 | 说明 |
|------|------|------|
| zero | 10% | 零修正（无操作基线） |
| deterministic | 5% | 确定性策略输出 |
| policy_sample | 25% | 策略采样 |
| gt_directed | 60% | GT 导向采样（关键） |

**GT-Directed 采样**：
```
direction = gt_plan - ref_plan
scale ~ Uniform(0.1, 0.5)  # 小幅扰动
correction = direction * scale + GaussianNoise
```

---

## 模型无关接口

### PlanningInterface

所有规划器通过统一接口接入：

```python
@dataclass
class PlanningInterface:
    """统一规划接口"""
    scene_token: torch.Tensor      # [B, D] 场景编码
    reference_plan: torch.Tensor    # [B, T, 2] 参考轨迹 (UTM)
    plan_confidence: torch.Tensor  # [B, 1] 置信度
    velocity: torch.Tensor          # [B, T, 2] 速度 (可选)
```

### Adapter 模式

新增规划器只需实现适配器：

```python
class YourModelAdapter(BaseExtractor):
    """适配器模板"""

    def extract(self, model_output: dict) -> PlanningInterface:
        """从模型输出提取 PlanningInterface"""
        ...

    def restore(self, corrected_plan: torch.Tensor) -> dict:
        """将修正后轨迹转回模型格式"""
        ...
```

---

## 训练流程

### Stage 1: Behavioral Cloning 预热

```
GT correction = gt_plan - ref_plan
Loss = -log π(GT_correction | state)
```

### Stage 2: Policy Gradient + 三层门控

```
1. 采样 correction ~ π(·|state)
2. 计算 advantage = safe_reward(ref + correction) - safe_reward(ref)
3. SafetyGuard: 硬约束检查
4. STAPOGate: 过滤虚假更新
5. LearnedUpdateGate: 预测收益排序
6. filtered_loss = masked_pg_loss * (n_total / n_active)
7. total_loss = filtered_loss - entropy_coef * H(π)
8. backward + step
```

---

## 配置说明

### CorrectionPolicy 配置

```yaml
# configs/correction_policy.yaml
policy:
  scene_dim: 256
  plan_len: 6
  hidden_dim: 256
  dropout: 0.1

training:
  bc_epochs: 10
  rl_epochs: 100
  bc_lr: 1e-4
  rl_lr: 5e-5
  entropy_coef: 0.01

gates:
  safety_guard:
    enabled: true
    residual_limit: 2.0
    step_limit: 1.0
  stapo_gate:
    enabled: true
    prob_threshold: 0.3
    entropy_threshold: 0.5
  learned_update_gate:
    enabled: false
    retention_ratio: 0.3
    evaluator_ckpt: null
```

### UpdateEvaluator 配置

```yaml
# configs/update_evaluator.yaml
model:
  scene_dim: 256
  plan_len: 6
  hidden_dim: 256
  dropout: 0.1

risk_weights:
  collision: 2.0
  offroad: 1.0
  comfort: 0.5
  drift: 1.0

training:
  batch_size: 128
  lr: 5e-5
  weight_decay: 1e-4
  grad_clip: 1.0
  epochs: 50
  eval_every: 5
```

---

## 训练命令

### 1. 训练 CorrectionPolicy

```bash
export PYTHONPATH="/mnt:$PYTHONPATH"

python -m E2E_RL.scripts.train_correction_policy \
    --config /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/correction_policy.yaml \
    --output_dir /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/correction_policy
```

### 2. 训练 UpdateEvaluator

```bash
python /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/scripts/train_evaluator_v2.py
```

训练输出：
```
[Evaluator Ranking] spearman_gain=0.736 kendall=0.581 spearman_risk=0.974
[Evaluator Filtering] retained_gain=-0.003 filtered_gain=-0.052 gain_diff=0.049
```

### 3. 集成训练

```bash
# 先更新配置中的 evaluator_ckpt
python /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/scripts/train_with_learned_gate.py
```

---

## 集成新规划器

### 步骤 1: 实现 Adapter

```python
# planning_interface/adapters/your_model_adapter.py
from E2E_RL.planning_interface.extractor import BaseExtractor
from E2E_RL.planning_interface.interface import PlanningInterface

class YourModelAdapter(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract(self, model_output: dict) -> PlanningInterface:
        """提取 PlanningInterface"""
        return PlanningInterface(
            scene_token=self._encode_scene(model_output['obs']),
            reference_plan=model_output['trajectory'],
            plan_confidence=model_output.get('confidence', torch.ones(1, 1)),
        )

    def restore(self, corrected_plan: torch.Tensor) -> dict:
        """恢复为模型输出格式"""
        return {'trajectory': corrected_plan}
```

### 步骤 2: 注册 Adapter

```python
# 在 planning_interface/__init__.py 中注册
ADAPTERS = {
    'vad': VADAdapter,
    'your_model': YourModelAdapter,
}
```

### 步骤 3: 使用

```python
adapter = get_adapter('your_model')
interface = adapter.extract(model_output)
correction = policy.act(interface)
corrected = interface.reference_plan + correction
output = adapter.restore(corrected)
```

---

## 注意事项

### 候选采样

- **GT-Directed 权重 60%** 确保生成足够多正 gain 样本
- **Scale 范围 0.1-0.5** 生成小幅修正，效果更稳定
- **Zero 采样 10%** 提供无操作基线用于对比

### 三层门控配置

| 配置 | 推荐值 | 说明 |
|------|--------|------|
| SafetyGuard | enabled | 物理安全底线 |
| STAPOGate | 先禁用 | 观察效果后再启用 |
| LearnedUpdateGate | retention=1.0 起步 | 观察预测分布后再调整 |

### 数据集要求

训练数据需包含：
- `reference_plan`: [T, 2] 参考轨迹
- `gt_plan`: [T, 2] 地面真值轨迹
- `scene_token`: [D] 场景编码

---

## 模块依赖关系

```
PlanningInterface (接口层)
       │
       ▼
CorrectionPolicy (策略)
       │
       ▼
┌──────────────────────────────────────────┐
│           三层门控 (可配置开关)            │
├──────────────┬──────────────┬─────────────┤
│ SafetyGuard  │ STAPOGate   │ LearnedGate │
│ (硬约束)     │ (虚假过滤)  │ (收益排序)  │
└──────────────┴──────────────┴─────────────┘
       │
       ▼
CorrectionPolicyTrainer (训练器)

UpdateEvaluator (独立训练)
       │
       ▼
LearnedUpdateGate (可选集成)
```
