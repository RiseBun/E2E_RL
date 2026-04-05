# E2E_RL: 端到端强化学习修正框架

**架构**: Correction Policy (高斯策略) + 三层防御级联 (SafetyGuard → STAPOGate → LearnedUpdateGate)

不同于旧的 "Refiner + 后验裁决器" 架构，E2E_RL 现在是一个真正的 RL 修正框架。

---

## 核心架构

### 旧架构（废弃）→ 新架构

```
旧: Refiner → Scorer 评估 → HUF 兜底 → 接受/拒绝修正
新: Policy 采样修正 → 三层门控筛选 → 梯度更新 Policy
```

**核心区别**: 筛选器服务于 **policy update selection**（决定哪些梯度更新进 loss），而非 **correction acceptance**（推理时裁决接不接受修正）。

### 新架构 Pipeline

```
Stage 1: Behavioral Cloning 预热
  GT correction = gt_plan - ref_plan
  Loss = -log π(gt_correction | state)

Stage 2: Policy Gradient + 三层门控
  1. Policy 采样 correction ~ π(·|state)
  2. corrected_plan = ref_plan + correction
  3. UpdateEvaluator 预测 gain/risk
  4. SafetyGuard: 硬约束检查 → 违规 mask
  5. STAPOGate: 正 A + 低 π + 低 H → 虚假更新静音
  6. LearnedUpdateGate: 预测 gain 排序 → 选择 top-k 更新
  7. filtered_loss = 重归一化(masked loss)
  8. total_loss = filtered_loss - entropy_coef * H(π)
  9. backward + step
```

### 推理流程

```
scene_token, ref_plan, conf = Adapter.extract(planner_output)
correction = Policy.act(scene_token, ref_plan, conf)  # 确定性 mean
corrected_plan = ref_plan + correction
```

**推理时没有 Scorer，没有 HUF，没有 accept/reject。Policy 自身就是修正专家。**

---

## 项目结构

```
E2E_RL/
├── correction_policy/          # 核心策略模块
│   ├── actor.py               # GaussianCorrectionActor
│   ├── policy.py              # 统一策略接口
│   ├── losses.py              # BC + PG loss
│   └── __init__.py
│
├── update_selector/            # 三层门控模块
│   ├── update_evaluator.py    # UpdateEvaluator (预测 gain/risk)
│   ├── safety_guard.py        # 硬性物理约束
│   ├── stapo_gate.py           # STAPO 门控
│   ├── learned_update_gate.py  # LearnedUpdateGate (基于 Evaluator)
│   ├── candidate_generator.py  # 候选修正生成器
│   └── __init__.py
│
├── rl_trainer/                 # RL 训练器
│   ├── correction_policy_trainer.py
│   └── __init__.py
│
├── planning_interface/         # 统一接口（模型无关）
│   ├── interface.py           # PlanningInterface
│   ├── extractor.py           # 提取器
│   └── adapters/              # 规划器适配器
│
├── data/                      # 数据加载
│   ├── vad_dataset.py
│   └── dataloader.py
│
├── refinement/                 # 奖励计算
│   └── reward_proxy.py        # safe_reward 函数
│
├── scripts/
│   ├── train_correction_policy.py   # 策略训练入口
│   ├── train_evaluator_v2.py       # UpdateEvaluator 训练入口
│   └── train_with_learned_gate.py   # 保守集成训练
│
└── configs/
    ├── correction_policy.yaml     # 策略配置
    └── update_evaluator.yaml      # Evaluator 配置
```

---

## 核心模块说明

### 1. GaussianCorrectionActor

高斯策略网络：
- 输入: scene_token + reference_plan + plan_confidence
- 输出: 高斯分布参数 (mean, std)
- 训练时采样，推理时取均值
- 正确的高斯熵计算: `H = 0.5 * log(2πe * σ²)`

### 2. CorrectionPolicy

统一策略接口：
- `sample()`: 训练时采样
- `evaluate()`: 给定 action 计算 log_prob + entropy
- `act()`: 推理时确定性输出
- `get_corrected_plan()`: 直接输出修正后轨迹

### 3. UpdateEvaluator (新增)

多目标回归网络，预测修正的收益和风险：

**预测目标**：
| 预测头 | 说明 | 权重 |
|--------|------|------|
| pred_gain | 修正带来的奖励增益 | 1.0 |
| pred_collision | 碰撞风险变化 | 2.0 |
| pred_offroad | 偏离道路风险变化 | 1.0 |
| pred_comfort | 舒适度变化 | 0.5 |
| pred_drift | 漂移风险变化 | 1.0 |

**Loss 函数**：
```
loss = α * MSE(pred_gain, y_gain) + β * (risk_loss)
```

**输入特征**：
- reference_plan: [B, T, 2] 参考轨迹
- correction: [B, T, 2] 候选修正
- scene_token: [B, 256] 场景编码

### 4. SafetyGuard

硬性物理约束检查：
- 残差范数限制
- 单步位移限制
- 速度上限
- 总位移限制

### 5. STAPOGate

STAPO 门控：
- 识别虚假有益更新：正 A + 低 π + 低 H
- 在 policy gradient loss 内部静音
- 重归一化保持梯度期望

### 6. LearnedUpdateGate (新增)

基于 UpdateEvaluator 的学习型门控：
- 使用预测 gain 排序候选修正
- 只保留 top-k 更新用于梯度更新
- 需要先训练好 UpdateEvaluator

### 7. CandidateCorrector (新增)

候选修正生成器，支持多种采样策略：

| 策略 | 权重 | 说明 |
|------|------|------|
| zero | 10% | 零修正（无操作基线） |
| deterministic | 5% | 确定性策略输出 |
| policy_sample | 25% | 策略采样 |
| gt_directed | 60% | GT 导向采样（小幅扰动） |

**GT-Directed 采样**：
```
direction = gt_plan - ref_plan
scale = uniform(0.1, 0.5)
correction = direction * scale + noise
```

---

## 三层防御级联

```
┌─────────────────────────────────────────────────────────┐
│                    候选修正候选                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 1: SafetyGuard                       │
│  检查: 残差范数、单步位移、速度上限、总位移                │
│  违规 → mask=0 (丢弃)                                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 2: STAPOGate                          │
│  检查: 正 advantage + 低 π + 低 entropy                 │
│  虚假更新 → mask=0 (静音)                               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 3: LearnedUpdateGate                  │
│  预测 gain 排序 → 保留 top-k                            │
│  低 gain → mask=0                                       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Policy Gradient 更新                    │
└─────────────────────────────────────────────────────────┘
```

---

## 模型无关接口 (PlanningInterface)

### 接口定义

```python
@dataclass
class PlanningInterface:
    """统一的规划器输出接口"""
    scene_token: torch.Tensor      # [B, D] 场景编码
    reference_plan: torch.Tensor    # [B, T, 2] 参考轨迹 (UTM 坐标系)
    plan_confidence: torch.Tensor   # [B, 1] 规划置信度
    velocity: torch.Tensor          # [B, T, 2] 速度 (可选)
```

### 适配器模式

新增规划器只需实现适配器：

```python
class VADAdapter:
    """VAD 规划器适配器"""

    def extract(self, vad_output: dict) -> PlanningInterface:
        """从 VAD 输出提取 PlanningInterface"""
        ...

    def restore(self, corrected_plan: torch.Tensor) -> dict:
        """将修正后轨迹转回 VAD 格式"""
        ...
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
    enabled: false  # 保守集成时设为 true
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

### 1. CorrectionPolicy 训练

```bash
# 激活环境
conda activate e2e_rl

# 设置路径
export PYTHONPATH="/mnt:$PYTHONPATH"

# 完整训练（BC + RL）
python -m E2E_RL.scripts.train_correction_policy \
    --config /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/correction_policy.yaml \
    --output_dir /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/correction_policy
```

### 2. UpdateEvaluator 训练

```bash
# 训练 UpdateEvaluator（使用 5000 样本增强数据集）
python /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/scripts/train_evaluator_v2.py
```

### 3. 保守集成训练

```bash
# SafetyGuard + STAPOGate + LearnedUpdateGate
python /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/scripts/train_with_learned_gate.py
```

### 4. Dry run

```bash
# 仅构建模型验证
python -m E2E_RL.scripts.train_correction_policy \
    --config /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/correction_policy.yaml \
    --dry_run
```

---

## 训练注意事项

### 1. UpdateEvaluator 训练

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| 正 gain 比例过低 | < 30% | 调整 gt_directed scale，使用加权采样 |
| Spearman 相关性低 | < 0.3 | 增加训练数据量，检查标签质量 |
| 过拟合 | val loss 上升 | 减少 epochs，增加 dropout |

### 2. 三层门控配置

**保守集成配置（推荐起步）**：
```yaml
safety_guard:
  enabled: true
stapo_gate:
  enabled: false  # 先禁用观察效果
learned_update_gate:
  enabled: true
  retention_ratio: 1.0  # 初始 100% 保留，观察预测分布
```

### 3. 候选采样策略

- **GT-Directed 采样权重 60%** 是关键，确保生成足够多正 gain 样本
- **Scale 范围 0.1-0.5** 产生小幅修正，更接近真实有效修正
- **Zero 采样 10%** 提供无操作基线

### 4. 数据集要求

- 原始 VAD dump 数据需包含: `reference_plan`, `gt_plan`, `scene_token`
- 增强数据集通过噪声注入扩展样本量
- manifest.json 记录样本列表

---

## 新旧模块对应关系

| 旧模块（废弃） | 新模块 | 说明 |
|--------------|--------|------|
| `InterfaceRefiner` | `CorrectionPolicy` | 确定性 → 高斯策略 |
| `supervised_refinement_loss` | `behavioral_cloning_loss` | BC 预热 |
| `reward_weighted_refinement_loss` | `policy_gradient_loss` | PG loss |
| `HarmfulUpdateFilter` | `SafetyGuard` + `STAPOGate` + `LearnedUpdateGate` | 后验裁决 → 训练时梯度过滤 |
| `UpdateReliabilityScorer` | `UpdateEvaluator` | 新增学习型评分器 |
| `ReliabilityNet` | 移除 | 不再需要 |
| `InterfaceRefinerTrainer` | `CorrectionPolicyTrainer` | 重写训练循环 |

---

## STAPO Gate 设计说明

STAPO (Self-Taught Policy Optimization) 的核心思想：

**识别虚假有益更新**：具有正 advantage 但概率低、熵低的更新。这类更新看起来"有益"，但实际上是策略分布过尖导致的噪声。

**处理方式**：在 policy gradient loss 内部静音这类更新，并重归一化 loss。

**公式**：
```
虚假有益更新 = (A > 0) AND (π(a|s) < τ_prob) AND (H < τ_entropy)
filtered_loss = mask * pg_loss * (n_total / n_active)
```

---

## UpdateEvaluator Ranking Metrics

训练过程中监控以下指标：

| 指标 | 目标 | 说明 |
|------|------|------|
| spearman_gain | > 0.3 | 预测 gain 与真实 gain 的 Spearman 相关性 |
| kendall_gain | > 0.3 | 预测 gain 与真实 gain 的 Kendall tau |
| retained_gain > filtered_gain | ✓ | 保留样本的 gain 应高于丢弃样本 |
| retained_risk < filtered_risk | ✓ | 保留样本的 risk 应低于丢弃样本 |

**达标参考（v4_5k_samples）**：
- Spearman: 0.736 ✓
- Kendall: 0.581 ✓
- 训练数据: 9600 样本
- 正 gain 比例: ~26%

---

## 集成新 E2E 模型

只需新建一个 Adapter，参考 `planning_interface/adapters/` 下的示例。

```python
# 新建 your_model_adapter.py
from E2E_RL.planning_interface.extractor import BaseExtractor

class YourModelAdapter(BaseExtractor):
    """YourModel 规划器适配器"""

    def extract(self, model_output: dict) -> PlanningInterface:
        """从 YourModel 输出提取 PlanningInterface"""
        return PlanningInterface(
            scene_token=self._encode_scene(model_output),
            reference_plan=model_output['trajectory'],
            plan_confidence=model_output['confidence'],
        )
```
