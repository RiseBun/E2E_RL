# E2E_RL: 端到端强化学习修正框架

## 📑 目录

- [概述](#概述)
- [Pipeline 流程](#pipeline-流程)
  - [架构图](#架构图)
  - [推理流程](#推理流程)
  - [训练流程](#训练流程)
- [如何集成新的 E2E 模型](#如何集成新的-e2e-模型)
  - [第一步：分析新模型的输出格式](#第一步分析新模型的输出格式)
  - [第二步：创建 Adapter 适配器](#第二步创建-adapter-适配器)
  - [第三步：注册 Adapter](#第三步注册-adapter)
  - [第四步：准备训练数据](#第四步准备训练数据)
  - [第五步：配置训练参数](#第五步配置训练参数)
  - [第六步：开始训练](#第六步开始训练)
  - [第七步：训练 UpdateEvaluator](#第七步可选训练-updateevaluator)
  - [第八步：推理部署](#第八步推理部署)
  - [关键注意事项](#关键注意事项)
- [项目结构](#项目结构)
- [核心模块详解](#核心模块详解)
- [模型无关接口](#模型无关接口)
- [配置说明](#配置说明)
- [训练命令](#训练命令)
- [注意事项](#注意事项)
- [模块依赖关系](#模块依赖关系)

---

## 概述

E2E_RL 是一个端到端强化学习修正框架，通过学习最优的轨迹修正策略来提升自动驾驶规划的质量。

**核心思想**：将修正问题建模为强化学习问题，策略网络学习在给定参考轨迹和场景上下文的情况下，输出最优的修正量。

---

## Pipeline 流程

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

### 训练流程

#### Stage 1: Behavioral Cloning 预热

```
GT correction = gt_plan - ref_plan
Loss = -log π(GT_correction | state)
```

#### Stage 2: Policy Gradient + 三层门控

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

## 如何集成新的 E2E 模型

当你拿到一个新的 E2E 模型（如 DiffusionDrive、UniAD 等）时，按照以下步骤将其集成到 E2E_RL 框架中。

### 第一步：分析新模型的输出格式

首先需要了解新模型的 `forward()` 方法输出的字典包含哪些字段。以 DiffusionDrive 为例：

```python
# 示例：DiffusionDrive 输出
model_output = {
    'trajectory': [B, T, 3],           # 最优轨迹 (x, y, heading)
    'bev_semantic_map': [B, C, H, W],  # BEV语义图
    'agent_states': [B, A, 5],         # 检测到的车辆状态
    'agent_labels': [B, A],            # 检测置信度
    # ... 其他字段
}
```

**关键问题**：
- 轨迹字段名称是什么？（如 `trajectory`、`plan`、`path`）
- 轨迹维度是 `[B, T, 2]` 还是 `[B, T, 3]`（是否包含 heading）？
- 是否有 BEV 特征或场景编码？
- 是否有候选轨迹或多模态输出？
- 是否有置信度分数？

### 第二步：创建 Adapter 适配器

在 `planning_interface/adapters/` 目录下创建新的适配器文件，例如 `your_model_adapter.py`。

**核心任务**：继承 `BasePlanningAdapter` 并实现 4 个抽象方法：

```python
from .base_adapter import BasePlanningAdapter
import torch
from typing import Any, Dict, Optional

class YourModelAdapter(BasePlanningAdapter):
    """YourModel -> PlanningInterface 适配器"""
    
    def __init__(self, scene_pool: str = 'mean', **kwargs):
        self.scene_pool = scene_pool
    
    def extract_scene_token(self, planner_outputs: Dict[str, Any]) -> torch.Tensor:
        """从模型输出提取场景编码 [B, D]
        
        通常从 BEV 特征池化得到，如果没有 BEV 特征，
        可以从轨迹和检测结果构建伪场景编码。
        """
        # 示例：从 BEV 语义图池化
        if 'bev_semantic_map' in planner_outputs:
            bev_map = planner_outputs['bev_semantic_map']  # [B, C, H, W]
            return bev_map.mean(dim=(-2, -1))  # [B, C]
        
        # 回退方案：从轨迹构建
        trajectory = planner_outputs.get('trajectory')
        if trajectory is not None:
            return trajectory.flatten(1)  # [B, T*3]
        
        raise KeyError("无法提取 scene_token")
    
    def extract_reference_plan(
        self, 
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """提取参考轨迹 [B, T, 2]
        
        Returns:
            (reference_plan, candidate_plans)
            - reference_plan: [B, T, 2] ego-centric 绝对坐标
            - candidate_plans: [B, M, T, 2] 或 None
        """
        if 'trajectory' not in planner_outputs:
            raise KeyError('模型输出中缺少 trajectory')
        
        trajectory = planner_outputs['trajectory']  # [B, T, 3] 或 [B, T, 2]
        
        # 如果包含 heading，只取 (x, y)
        if trajectory.shape[-1] == 3:
            reference_plan = trajectory[..., :2]  # [B, T, 2]
        else:
            reference_plan = trajectory  # [B, T, 2]
        
        # 候选轨迹（如果有）
        candidate_plans = None
        if 'all_poses_reg' in planner_outputs:
            all_poses = planner_outputs['all_poses_reg']  # [B, M, T, 3]
            candidate_plans = all_poses[..., :2]  # [B, M, T, 2]
        
        return reference_plan, candidate_plans
    
    def extract_plan_confidence(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """提取规划置信度 [B, 1]
        
        可选，如果没有显式置信度，可以返回 None 或使用启发式方法。
        """
        # 策略 1：多模态分类分数
        if 'all_poses_cls' in planner_outputs:
            cls_scores = planner_outputs['all_poses_cls']  # [B, M]
            probs = torch.softmax(cls_scores, dim=-1)
            confidence = probs.max(dim=-1).values.unsqueeze(-1)  # [B, 1]
            return confidence
        
        # 回退：均匀置信度
        trajectory = planner_outputs.get('trajectory')
        if trajectory is not None:
            batch_size = trajectory.shape[0]
            return torch.ones((batch_size, 1), device=trajectory.device)
        
        return None
    
    def extract_safety_features(
        self,
        planner_outputs: Dict[str, Any]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """提取安全相关特征
        
        可选，用于增强安全性判断。
        """
        safety: Dict[str, torch.Tensor] = {}
        
        # 示例：检测物体密度
        if 'agent_labels' in planner_outputs:
            labels = planner_outputs['agent_labels']  # [B, A]
            obj_prob = torch.sigmoid(labels)
            safety['object_density'] = obj_prob.mean(dim=-1, keepdim=True)
        
        return safety if safety else None
```

**参考现有实现**：
- [VAD Adapter](file:///mnt/cpfs/prediction/lipeinan/RL/E2E_RL/planning_interface/adapters/vad_adapter.py) - 从 BEV embedding 提取 scene_token
- [DiffusionDrive Adapter](file:///mnt/cpfs/prediction/lipeinan/RL/E2E_RL/planning_interface/adapters/diffusiondrive_adapter.py) - 从 BEV 语义图池化提取 scene_token

### 第三步：注册 Adapter

修改 [`planning_interface/extractor.py`](file:///mnt/cpfs/prediction/lipeinan/RL/E2E_RL/planning_interface/extractor.py)，在 `from_config` 方法中添加新类型：

```python
# 1. 导入你的适配器
from .adapters.your_model_adapter import YourModelAdapter

# 2. 在 from_config 中添加分支
@classmethod
def from_config(
    cls,
    adapter_type: str = 'vad',
    scene_pool: str = 'mean',
    debug: bool = False,
    **adapter_kwargs: Any,
) -> 'PlanningInterfaceExtractor':
    if adapter_type == 'vad':
        adapter = VADPlanningAdapter(scene_pool=scene_pool, **adapter_kwargs)
    elif adapter_type == 'diffusiondrive':
        adapter = DiffusionDrivePlanningAdapter(scene_pool=scene_pool, **adapter_kwargs)
    elif adapter_type == 'your_model':  # 新增
        adapter = YourModelAdapter(scene_pool=scene_pool, **adapter_kwargs)
    else:
        raise ValueError(f'未知的 adapter_type: {adapter_type}')
    
    return cls(adapter=adapter, debug=debug)
```

### 第四步：准备训练数据

数据集需要包含以下关键字段：

```python
data_sample = {
    'reference_plan': [T, 2],      # 模型输出的参考轨迹 (x, y)
    'gt_plan': [T, 2],             # 地面真值轨迹 (x, y)
    # scene_token 由 Adapter 自动从模型输出中提取
}
```

**关键点**：
- ✅ GT correction = `gt_plan - reference_plan`
- ✅ 确保轨迹坐标系一致（ego-centric 绝对坐标）
- ✅ 去掉 heading 信息，只保留 (x, y) 位置
- ✅ 时间步长 T 需与配置中的 `plan_len` 一致

### 第五步：配置训练参数

根据新模型调整配置文件 [`configs/correction_policy.yaml`](file:///mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/correction_policy.yaml)：

```yaml
policy:
  scene_dim: 256          # ⚠️ 需与 Adapter 输出的 scene_token 维度匹配
  plan_len: 6             # 轨迹长度 T（需与数据集一致）
  hidden_dim: 256
  dropout: 0.1

training:
  bc_epochs: 10           # BC 预热轮数
  rl_epochs: 100          # RL 训练轮数
  bc_lr: 1e-4
  rl_lr: 5e-5
  entropy_coef: 0.01      # 熵正则系数
  batch_size: 64

gates:
  safety_guard:
    enabled: true         # 始终启用硬性约束
    residual_limit: 2.0
    step_limit: 1.0
  stapo_gate:
    enabled: false        # 初期建议禁用，观察效果后再启用
    prob_threshold: 0.3
    entropy_threshold: 0.5
  learned_update_gate:
    enabled: false        # 需要先训练 UpdateEvaluator
    retention_ratio: 0.3
    evaluator_ckpt: null
```

**⚠️ 重要**：如果 Adapter 输出的 `scene_token` 维度与配置的 `scene_dim` 不一致，需要在 Actor 网络中添加投影层。

### 第六步：开始训练

#### Stage 1 & 2: BC 预热 + RL 强化训练

```bash
export PYTHONPATH="/mnt:$PYTHONPATH"

python -m E2E_RL.scripts.train_correction_policy \
    --config configs/correction_policy.yaml \
    --output_dir experiments/your_model_policy \
    --adapter_type your_model  # 指定使用你的 Adapter
```

**训练流程**：
1. **BC 阶段**（前 10 epochs）：让策略学会模仿 GT correction
2. **RL 阶段**（后续 100 epochs）：通过策略梯度优化，结合三层门控筛选高质量更新

**监控指标**：
- `bc_loss`: BC 阶段的负对数似然
- `rl_loss`: RL 阶段的策略梯度损失
- `entropy`: 策略熵（应保持适中，避免过早收敛）
- `advantage_mean`: 平均优势值（应逐渐增大）
- `gate_mask_ratio`: 被门控过滤的比例

### 第七步：(可选)训练 UpdateEvaluator

如果想启用学习型门控（LearnedUpdateGate），需要先训练评估网络：

```bash
python scripts/train_evaluator_v2.py \
    --config configs/update_evaluator.yaml \
    --adapter_type your_model
```

**作用**：预测每个候选修正的 gain/risk，用于筛选高质量更新

**训练完成后**：
1. 记录 checkpoint 路径
2. 更新 `correction_policy.yaml` 中的 `evaluator_ckpt`
3. 设置 `learned_update_gate.enabled: true`
4. 重新训练 CorrectionPolicy

### 第八步：推理部署

```python
from E2E_RL.planning_interface.extractor import PlanningInterfaceExtractor
from E2E_RL.correction_policy.policy import CorrectionPolicy

# 1. 创建提取器
extractor = PlanningInterfaceExtractor.from_config(
    adapter_type='your_model',
    debug=False  # 生产环境关闭调试
)

# 2. 加载训练好的策略
policy = CorrectionPolicy.load_from_checkpoint('experiments/your_model_policy/best_model.pth')
policy.eval()

# 3. 推理流程
with torch.no_grad():
    model_output = your_model.forward(inputs)
    interface = extractor.extract(model_output)
    correction = policy.act(interface, deterministic=True)  # 取均值
    corrected_plan = interface.reference_plan + correction

# 4. 将修正后轨迹转回模型格式（如果需要）
# final_output = your_model.restore(corrected_plan)
```

### 关键注意事项

#### 1. Scene Token 维度一致性

Adapter 输出的 `scene_token` 维度必须与配置中的 `scene_dim` 一致。

**如果不一致**，有两种解决方案：
- **方案 A**：调整 Adapter 的池化方式，使输出维度匹配
- **方案 B**：在 Actor 网络中添加线性投影层

```python
# 在 GaussianCorrectionActor 中
if scene_token.shape[-1] != self.scene_dim:
    self.scene_projection = nn.Linear(scene_token.shape[-1], self.scene_dim)
    scene_token = self.scene_projection(scene_token)
```

#### 2. 轨迹坐标系

- ✅ 统一使用 **ego-centric 绝对坐标** (x, y)
- ✅ 去掉 heading 信息，只保留位置
- ❌ 不要使用相对坐标或极坐标

#### 3. 候选采样策略

训练时的候选生成权重（默认）：

| 策略 | 权重 | 说明 |
|------|------|------|
| GT-Directed | 60% | ⭐ 关键，保证正 gain 样本 |
| Policy Sample | 25% | 策略采样探索 |
| Zero | 10% | 无操作基线 |
| Deterministic | 5% | 确定性输出 |

**GT-Directed 采样**：
```python
direction = gt_plan - ref_plan
scale ~ Uniform(0.1, 0.5)  # 小幅扰动
correction = direction * scale + GaussianNoise(σ=0.1)
```

#### 4. 三层门控调优顺序

**推荐顺序**：
1. **初期**：只开 SafetyGuard（硬约束底线）
2. **中期**：观察效果后再考虑 STAPOGate
3. **后期**：启用 LearnedUpdateGate（需先训练 Evaluator）

| 配置 | 推荐值 | 说明 |
|------|--------|------|
| SafetyGuard | `enabled: true` | 物理安全底线 |
| STAPOGate | `enabled: false` → `true` | 观察效果后再启用 |
| LearnedUpdateGate | `retention_ratio: 1.0` 起步 | 观察预测分布后再调整 |

#### 5. 调试技巧

```python
# 开启调试模式查看提取结果
extractor = PlanningInterfaceExtractor.from_config(
    adapter_type='your_model',
    debug=True  # 会打印 shape 信息
)

# 检查提取的接口
interface = extractor.extract(model_output)
print(interface.describe())
# 输出示例：
# scene_token: (1, 256)
# reference_plan: (1, 6, 2)
# plan_confidence: (1, 1)
```

#### 6. 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| scene_token 维度不匹配 | Adapter 输出与配置不一致 | 检查 `scene_dim` 配置 |
| 训练 loss 不下降 | 学习率过大或数据问题 | 降低学习率，检查数据质量 |
| 熵快速衰减 | 探索不足 | 增大 `entropy_coef` |
| advantage 全为负 | 奖励函数问题 | 检查 reward_proxy 实现 |
| 门控过滤比例过高 | 阈值过严 | 放宽门控参数 |

### 完整集成流程图

```
新 E2E 模型
    ↓
[Step 1] 分析输出格式 (trajectory, bev_feature, ...)
    ↓
[Step 2] 创建 Adapter (继承 BasePlanningAdapter)
    ↓
[Step 3] 注册到 from_config
    ↓
[Step 4] 准备数据集 (reference_plan + gt_plan)
    ↓
[Step 5] 调整配置 (scene_dim, plan_len)
    ↓
[Step 6] BC 预热训练 → RL 强化训练
    ↓
[Step 7] (可选)训练 UpdateEvaluator
    ↓
[Step 8] 推理部署
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

---

## 实验结果 (2026-04-06)

### 数据坐标系修复

**问题**：GT (ego_fut_trajs) 和 VAD 预测 (ego_fut_preds) 在不同坐标系中。

**发现**：
- `ego_fut_trajs` (GT)：全局坐标，dump 时已 cumsum
- `ego_fut_preds` (VAD)：ego-centric 位移增量，需 cumsum 转绝对坐标

**修复**：
```python
# 修复前
reference_plan = ego_fut_preds[0]  # 增量，错误
gt_plan = ego_fut_trajs             # 全局坐标，错误

# 修复后
reference_plan = ego_fut_preds[0].cumsum(dim=0)  # ego-centric 绝对坐标
gt_plan = gt_plan_global - gt_plan_global[0]    # ego-centric 绝对坐标
```

**结果**：GT correction 从 20m 降到 10-13m（合理范围）

### A/B/C 实验对比

| 实验 | 配置 | retained_adv | retention | filtered_adv |
|------|------|-------------|-----------|--------------|
| A | SafetyGuard only | -1.1054 | 46.51% | -1.6717 |
| B | SafetyGuard + STAPOGate | -1.0830 | 46.79% | -1.6468 |
| **C** | **SafetyGuard + LearnedUpdateGate** | **-1.0784** | 45.93% | -1.6586 |

### 关键结论

1. **LearnedUpdateGate 有效**：实验 C 的 retained_adv (-1.0784) 比实验 A (-1.1054) 提高了 **2.4%**

2. **所有 advantage 仍为负**：VAD baseline 本身误差较大（3秒 FDE 约 10m），policy 修正能力有限

3. **Policy 修正效果**：
   - 部分样本 FDE 从 10m 降到 6m（有效修正）
   - 部分样本 FDE 从 13m 变成 14m（修正方向错误）
   - 整体泛化能力有待提升

4. **训练配置**：
   - BC epochs: 3
   - RL epochs: 15
   - action_scale: 2.0
   - reward: progress + collision + offroad + comfort (w_comfort=0.01, fde_scale=5.0)

### 下一步优化建议

1. **增大 action_scale**：当前 2.0 限制修正量，可尝试 5.0
2. **增加 BC epochs**：3 epochs 不足以充分学习 GT correction
3. **调整 reward 权重**：增大 progress_reward 权重或减小 fde_scale
4. **改善 policy 泛化**：增加训练数据或使用数据增强
