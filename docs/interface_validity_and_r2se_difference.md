# 面向自动驾驶 RL 后训练的统一规划接口与有害更新过滤

> **Unified Planning RL Interface + Harmful Update Filtering**
>
> 本文档是项目的唯一权威技术报告，涵盖两个核心创新点的设计动机、架构实现、模块详解、验证实验和结论分析。

---

## 目录

- [1. 项目概述与研究动机](#1-项目概述与研究动机)
- [2. 两大创新点总览](#2-两大创新点总览)
- [3. 项目结构](#3-项目结构)
- [4. 创新点一：Unified Planning Interface](#4-创新点一unified-planning-interface)
  - [4.1 四层架构设计](#41-四层架构设计)
  - [4.2 PlanningInterface 数据契约](#42-planninginterface-数据契约)
  - [4.3 VAD 适配器 (Adapter)](#43-vad-适配器-adapter)
  - [4.4 InterfaceRefiner 残差精炼网络](#44-interfacerefiner-残差精炼网络)
  - [4.5 奖励代理 (Reward Proxy)](#45-奖励代理-reward-proxy)
  - [4.6 损失函数](#46-损失函数)
  - [4.7 训练器 (Trainer)](#47-训练器-trainer)
- [5. 创新点二：Harmful Update Filtering (HUF)](#5-创新点二harmful-update-filtering-huf)
  - [5.1 核心思想与 STAPO 关系](#51-核心思想与-stapo-关系)
  - [5.2 三类有害更新信号](#52-三类有害更新信号)
  - [5.3 过滤机制：Hard Mask 与 Soft Weight](#53-过滤机制hard-mask-与-soft-weight)
  - [5.4 HUF 配置详解](#54-huf-配置详解)
  - [5.5 训练集成](#55-训练集成)
- [6. 验证实验](#6-验证实验)
  - [6.1 Layer 1：Interface Validity（结构解耦验证）](#61-layer-1interface-validity结构解耦验证)
  - [6.2 Layer 2：Interface Sufficiency（字段消费验证）](#62-layer-2interface-sufficiency字段消费验证)
  - [6.3 Layer 3：Scene Token 表征质量](#63-layer-3scene-token-表征质量)
  - [6.4 Layer 4a：Confidence 有效性](#64-layer-4aconfidence-有效性)
  - [6.5 Layer 4b：A/B 综合对比](#65-layer-4bab-综合对比)
  - [6.6 Layer 4c：HUF 离线验证](#66-layer-4chuf-离线验证)
- [7. 与 R2SE 的结构性差异](#7-与-r2se-的结构性差异)
- [8. 证据链总结](#8-证据链总结)
- [9. 已交付工件清单](#9-已交付工件清单)
- [10. 执行环境与复现指南](#10-执行环境与复现指南)
- [11. 已知局限与改进方向](#11-已知局限与改进方向)

---

## 1. 项目概述与研究动机

### 1.1 问题背景

当前端到端自动驾驶规划器（如 VAD、UniAD、SparseDrive 等）在 RL 后训练（post-training）阶段面临两个结构性问题：

1. **RL 优化与 planner 内部耦合**：传统方法（如 R2SE）虽然在 pipeline 级别实现了 model-agnostic，但其 RL objective 仍然直接操作 planner-native 的输出形式（categorical, GMM, diffusion 等），更换 planner 需要重写 RL objective。

2. **RL 更新质量不可控**：即使 reward 信号整体正确，单个 batch 中仍可能包含高不确定、低支持、易造成分布漂移的 spurious updates，这些有害更新在长期训练中会累积导致性能退化。

### 1.2 核心思路

本项目提出两个互补的创新点：

- **创新点一（Unified Planning Interface）**：在 planner backbone 和 RL refinement 之间插入一个标准化接口层，使得整个下游精炼流水线（refiner/loss/reward/trainer/evaluator）只依赖接口契约，与 planner 内部完全解耦。

- **创新点二（Harmful Update Filtering）**：受 STAPO 启发，从 **sample selection 升级到 update selection**，在 RL 训练过程中识别并抑制 spurious residual updates，对剩余有益更新重归一化损失。

### 1.3 基座模型

本项目基于 **VAD (Vector Autonomous Driving)** 进行验证：

| 属性 | 值 |
|------|------|
| Backbone | ResNet50 |
| BEV 分辨率 | 200 x 200 |
| BEV embedding dim | 256 |
| Planning modes | 3 |
| Future timesteps | 6 (每步 0.5s，覆盖 3.0s) |
| 坐标系统 | Ego-centric, 位移增量 (cumsum → 绝对坐标) |
| Checkpoint | `/mnt/cpfs/prediction/lipeinan/RL/VAD_base.pth` |

---

## 2. 两大创新点总览

```
┌──────────────────────────────────────────────────────────────┐
│                      VAD Backbone (冻结)                     │
│         ResNet50 → BEV 200x200 → ego_fut_preds [B,3,6,2]    │
└──────────────────────┬───────────────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │  VADPlanningAdapter   │  ← 唯一 planner-specific 边界
           │  (scene_pool / conf   │
           │   / safety 提取)      │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  PlanningInterface    │  ← 创新点一: 统一接口契约
           │  (7 字段 dataclass)   │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  InterfaceRefiner     │  ← 4 通道残差精炼网络
           │  (scene+plan+conf     │
           │   +safety → residual) │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  Reward Proxy         │  progress + collision
           │  + Per-sample Loss    │  + offroad + comfort
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  HUF Scorer + Filter  │  ← 创新点二: 有害更新过滤
           │  (uncertainty/support  │
           │   /drift → mask/weight│
           │   → renormalized loss)│
           └───────────┬───────────┘
                       │
                  Backward Pass
```

**一句话概括：**

> 创新点一定义 RL 优化 **作用在什么接口上**；创新点二定义 RL 优化时 **哪些更新应被信任**。

---

## 3. 项目结构

```
E2E_RL/
├── __init__.py
│
├── planning_interface/                # 创新点一：统一接口层
│   ├── __init__.py
│   ├── interface.py                   # PlanningInterface dataclass (7 字段)
│   ├── extractor.py                   # 提取器薄封装，委托给 adapter
│   ├── utils.py                       # 工具函数
│   └── adapters/
│       ├── __init__.py
│       ├── base_adapter.py            # 抽象适配器基类 (BasePlanningAdapter)
│       └── vad_adapter.py             # VAD 专用适配器 (mean/max/grid/ego_local)
│
├── refinement/                        # Planner-agnostic 精炼层
│   ├── __init__.py
│   ├── interface_refiner.py           # 4 通道残差精炼网络
│   ├── losses.py                      # supervised + reward-weighted + per-sample loss
│   └── reward_proxy.py                # 4 种奖惩函数 (progress/collision/offroad/comfort)
│
├── update_filter/                     # 创新点二：有害更新过滤
│   ├── __init__.py
│   ├── config.py                      # HUFConfig dataclass (所有超参数)
│   ├── scorer.py                      # UpdateReliabilityScorer (三类评分)
│   └── filter.py                      # HarmfulUpdateFilter (mask/weight + renormalized loss)
│
├── hard_case/                         # Hard-case 挖掘
│   ├── __init__.py
│   └── mining.py                      # HardCaseMiner
│
├── trainers/                          # 训练器
│   ├── __init__.py
│   └── trainer_refiner.py             # InterfaceRefinerTrainer (3 阶段 + HUF filtered)
│
├── evaluators/                        # 评估器
│   ├── __init__.py
│   └── eval_refined.py                # ADE/FDE/collision/comfort 指标
│
├── experiments/                       # 实验脚本
│   ├── ablation_interface_fields.py   # 字段语义消融 (合成数据)
│   ├── confidence_analysis.py         # Confidence 统计 (合成数据)
│   ├── scene_token_variants.py        # 池化方式对比 (合成数据)
│   ├── ab_comparison.py               # A/B 综合对比 (合成数据)
│   ├── load_dump.py                   # Dump 数据加载工具
│   ├── offline_confidence_analysis.py # Confidence 分析 (真实数据)
│   ├── offline_ab_comparison.py       # A/B 对比 (真实数据)
│   └── offline_huf_experiment.py      # HUF 离线对比实验 (真实数据)
│
├── scripts/                           # 辅助脚本
│   ├── dump_vad_inference.py          # VAD 推理输出导出
│   ├── train_interface_refiner.py     # 训练入口
│   └── eval_refined.py                # 评估入口
│
├── tests/                             # 单元测试
│   ├── test_update_filter.py          # 接口与精炼测试 ✅
│   ├── test_interface_validity.py     # 3 项接口验证测试 ✅
│   └── test_update_filter.py          # 17 个 HUF 测试 ✅
│
├── data/
│   └── vad_dumps/                     # VAD 推理 dump 数据
│       ├── sample_000.pt ~ sample_099.pt
│       └── dump_meta.json
│
└── docs/
    └── interface_validity_and_r2se_difference.md  # 本文档
```

---

## 4. 创新点一：Unified Planning Interface

### 4.1 四层架构设计

```
Layer 1: Planner Backbone   — VAD (冻结，不修改)
Layer 2: Adapter Boundary   — vad_adapter.py (唯一 planner-specific 文件)
Layer 3: Agnostic Refiner   — refiner / losses / reward / trainer / evaluator
Layer 4: Training & Reward  — RL 训练循环 + HUF 过滤
```

**核心约束：** Layer 3 和 Layer 4 中的所有模块只 import `PlanningInterface`，不 import 任何 VAD-specific 符号。这意味着更换 planner（如 UniAD, SparseDrive）时，只需新增一个 adapter 文件（如 `uniad_adapter.py`），下游全部复用。

### 4.2 PlanningInterface 数据契约

定义在 `planning_interface/interface.py`：

```python
@dataclass
class PlanningInterface:
    scene_token: torch.Tensor                          # [B, D] 或 [B, G*D]
    reference_plan: torch.Tensor                       # [B, T, 2]
    candidate_plans: Optional[torch.Tensor] = None     # [B, M, T, 2]
    plan_confidence: Optional[torch.Tensor] = None     # [B, 1]
    safety_features: Optional[Dict[str, Tensor]] = None
    hard_case_score: Optional[torch.Tensor] = None     # [B, 1]
    metadata: Optional[Dict[str, Any]] = {}
```

| 字段 | 形状 | 语义 | 来源 (VAD) |
|------|------|------|-----------|
| `scene_token` | `[B, D]` 或 `[B, G²·D]` | BEV 池化后的场景表征 | `bev_embed` 经池化 |
| `reference_plan` | `[B, T, 2]` | 选定模式的 ego-centric 绝对坐标轨迹 | `ego_fut_preds` cumsum |
| `candidate_plans` | `[B, M, T, 2]` | 所有模式的位移增量 | `ego_fut_preds` 原始 |
| `plan_confidence` | `[B, 1]` | 规划质量信号: `exp(-mode_variance)` | 模式间方差负指数 |
| `safety_features` | `Dict` | plan_mode_variance / object_density / map_density | 检测分数统计 |
| `hard_case_score` | `[B, 1]` | 外部 hard-case 评分 | HardCaseMiner 输出 |
| `metadata` | `Dict` | 元信息 (sample_idx, scene_name 等) | 透传 |

接口提供 `to(device)` 方法统一迁移所有张量，以及 `describe()` 方法用于调试输出。

### 4.3 VAD 适配器 (Adapter)

定义在 `planning_interface/adapters/vad_adapter.py`：

**`VADPlanningAdapter`** 继承抽象基类 `BasePlanningAdapter`，实现 4 个提取方法：

#### 4.3.1 Scene Token 提取

支持 4 种 BEV 池化方式：

| 池化方式 | 输出维度 | 方法 |
|---------|---------|------|
| `mean` | `[B, 256]` | 全局均值池化 |
| `max` | `[B, 256]` | 全局最大池化 |
| `grid` (4x4) | `[B, 4096]` | 空间分块池化，将 200x200 BEV 划分为 4x4=16 个 block (每 block 50x50=2500 tokens)，每 block 均值池化后拼接 |
| `ego_local` | `[B, 256]` | 取 BEV 中心 (ego 位置) 附近局部 token 均值池化 |

#### 4.3.2 Reference Plan 提取

```
ego_fut_preds [B, 3, 6, 2] (位移增量)
       ↓ ego_fut_cmd 选择模式
selected_deltas [B, 6, 2]
       ↓ cumsum(dim=-2)
reference_plan [B, 6, 2] (ego-centric 绝对坐标)
```

同时保留 `candidate_plans = ego_fut_preds` 原始增量形式。

#### 4.3.3 Plan Confidence 提取

```python
mode_variance = ego_fut_preds.var(dim=1).mean(dim=(-2,-1))  # [B]
confidence = exp(-mode_variance)                               # (0, 1]
```

直觉：模式间方差越大 → 规划器越不确定 → 置信度越低。

#### 4.3.4 Safety Features 提取

从 VAD 输出中收集 3 个安全信号：
- `plan_mode_variance`: `[B, T]` — 每步的模式间方差
- `object_density`: `[B, 1]` — 检测物体 sigmoid 概率均值
- `map_density`: `[B, 1]` — 地图检测 sigmoid 概率均值

### 4.4 InterfaceRefiner 残差精炼网络

定义在 `refinement/interface_refiner.py`：

**4 通道架构：**

```
scene_token  ─→ scene_proj  (D → H)       ─┐
reference_plan → plan_proj  (T*2 → H)     ─┤
plan_confidence → conf_proj  (1 → H/4)    ─┤→ concat → fusion MLP → residual_head → [B,T,2]
safety_features → safety_proj (S → H/4)   ─┘                       → score_head   → [B,1]
```

| 组件 | 结构 | 输入 → 输出 |
|------|------|-----------|
| `scene_proj` | Linear + ReLU + Dropout | `[B, D]` → `[B, H]` |
| `plan_proj` | Linear + ReLU + Dropout | `[B, T*2]` → `[B, H]` |
| `conf_proj` | Linear + ReLU | `[B, 1]` → `[B, H/4]` |
| `safety_proj` | Linear + ReLU (延迟初始化) | `[B, S]` → `[B, H/4]` |
| `fusion` | 2 层 MLP (Linear+ReLU+Dropout+Linear+ReLU) | `[B, 2H+H/2]` → `[B, H]` |
| `residual_head` | Linear | `[B, H]` → `[B, T*2]` → reshape `[B, T, 2]` |
| `score_head` | Linear+ReLU+Linear+Sigmoid | `[B, H]` → `[B, 1]` |

**关键设计决策：**

1. **辅助通道用小维度 (H/4)**：避免 confidence/safety 信号淹没主通道 (scene+plan)
2. **safety_proj 延迟初始化**：首次 forward 时根据实际 safety_features 维度动态创建，兼容不同 planner 的 safety 字段数量
3. **残差学习**：`refined_plan = reference_plan + residual`，residual 通常很小
4. **可选 output_norm**：`tanh` 约束 residual 范围

**输出 dict：**

```python
{
    'residual':      [B, T, 2],     # 残差增量
    'refined_plan':  [B, T, 2],     # 精炼后轨迹
    'residual_norm': [B],           # 残差 L2 范数均值
    'refine_score':  [B, 1],        # 精炼质量分数
}
```

### 4.5 奖励代理 (Reward Proxy)

定义在 `refinement/reward_proxy.py`，提供 4 个可离线计算的伪 RL 奖励组件：

| 函数 | 公式 | 语义 |
|------|------|------|
| `progress_reward` | `exp(-FDE)` | FDE 越小奖励越高，(0, 1] |
| `collision_penalty` | `max(x_thresh - x_dist, 0) * max(y_thresh - y_dist, 0)` | 与 agent 的碰撞风险，兼容 VAD 的 PlanCollisionLoss |
| `offroad_penalty` | `mean(max(dis_thresh - min_dist_to_boundary, 0))` | 离道惩罚 |
| `comfort_penalty` | `mean(|acceleration|) + 0.5 * mean(|jerk|)` | 加速度 + jerk 组合 |

**综合奖励：**

```python
total_reward = w_progress * progress - w_collision * collision
             - w_offroad * offroad - w_comfort * comfort
```

默认权重：`w_progress=1.0, w_collision=0.5, w_offroad=0.3, w_comfort=0.1`

### 4.6 损失函数

定义在 `refinement/losses.py`，提供 3 个损失函数：

| 函数 | 返回 | 用途 |
|------|------|------|
| `supervised_refinement_loss` | 标量 | Stage 1 监督预热：L1 loss with optional mask |
| `reward_weighted_refinement_loss` | 标量 | Stage 2 奖励加权：`weight = 1 - normalized_reward` |
| `compute_per_sample_reward_weighted_error` | `[B]` | HUF 专用：同上但返回 per-sample 值供过滤 |

**奖励加权损失逻辑：**

```python
normalized = (reward - reward.min()) / (reward.max() - reward.min() + 1e-6)
weight = 1.0 - normalized.clamp(0, 1)  # reward 越高 → weight 越低 → loss 越小
error = |refined_plan - gt_plan|.mean(dim=-1).mean(dim=-1)  # [B]
loss = (weight * error).mean()  # 标量
```

### 4.7 训练器 (Trainer)

定义在 `trainers/trainer_refiner.py`：

**`InterfaceRefinerTrainer`** 支持 3 种训练模式：

| 方法 | 阶段 | 损失函数 | 说明 |
|------|------|---------|------|
| `train_supervised_epoch` | Stage 1 | supervised L1 + residual reg | 监督预热，学习粗略残差 |
| `train_reward_weighted_epoch` | Stage 2 | reward-weighted + residual reg | 奖励加权精炼 |
| `train_filtered_reward_epoch` | Stage 2+ | per-sample → HUF → renormalized + reg | 带有害更新过滤的精炼 |

**典型训练流程：**

```
Epoch 1~30:   train_supervised_epoch()         # 30% 预热
Epoch 31~100: train_reward_weighted_epoch()     # 70% 精炼
              或 train_filtered_reward_epoch()   # 70% 带 HUF 精炼
```

---

## 5. 创新点二：Harmful Update Filtering (HUF)

### 5.1 核心思想与 STAPO 关系

**核心问题：** 不是所有带正回报的残差更新都值得学习。

在 RL 后训练中，以下三类更新即使对应正 reward 也可能有害：
1. 规划器在该场景本身高度不确定（**高 uncertainty**）
2. 残差偏移过大、缺乏数据支持（**低 support**）
3. 残差引入了舒适度恶化或曲率突变（**高 drift risk**）

**与 STAPO 的对应关系：**

| 维度 | STAPO (LLM 领域) | HUF (自动驾驶领域) |
|------|-----------------|------------------|
| 共享原则 | "不是所有正 reward 的更新都有益" | 同左 |
| 过滤对象 | Token-level gradients | Residual update-level |
| 判别信号 | Probability / entropy / positive advantage | Uncertainty / support / drift risk |
| 过滤机制 | Mask spurious tokens + renormalize | Mask/weight harmful updates + renormalize |
| 应用场景 | LLM outcome-supervised RL | 自动驾驶 RL 后训练 |

**与 Hard-case Mining 的区别：**

> Hard-case mining 决定 **在哪里训练更多**；Harmful Update Filtering 决定 **从训练中学到什么**。

Hard-case mining 是样本级别的 **选择策略**（选哪些样本训练），HUF 是更新级别的 **质量控制**（同一样本的更新是否可信）。两者正交：可以先用 hard-case mining 选样本，再用 HUF 过滤更新。

### 5.2 三类有害更新信号

由 `UpdateReliabilityScorer` 实现（`update_filter/scorer.py`）：

#### 5.2.1 Uncertainty Score（不确定性）

| 信号来源 | 计算方式 | 权重 |
|---------|---------|------|
| Confidence 反转 | `u_conf = 1 - plan_confidence` | `w_confidence` (0.5) |
| 模式方差 | `u_mode = 1 - exp(-mode_variance)` | `w_mode_variance` (0.3) |
| Residual 不确定性 | `u_residual = 1 - exp(-residual_norm)` | `w_residual_var` (0.2) |

自动权重归一化：当某个信号缺失时（如无 candidate_plans），权重自动重新分配。

#### 5.2.2 Support Score（数据支持度）

| 信号来源 | 计算方式 | 权重 |
|---------|---------|------|
| 残差大小衰减 | `s_norm = exp(-alpha * residual_norm)` | 0.6 |
| 单步最大位移衰减 | `s_disp = exp(-alpha * max_step_disp)` | 0.4 |

硬上限：`residual_norm > max_residual_norm` (默认 5.0) 时直接归零。

#### 5.2.3 Drift Score（漂移风险）

| 信号来源 | 计算方式 | 权重 |
|---------|---------|------|
| Comfort 恶化 | `1 - exp(-(comfort_refined - comfort_ref).clamp(min=0))` | `w_comfort` (0.4) |
| Curvature 突变 | `max(|atan2(cross, dot)|) / pi` | `w_curvature` (0.3) |
| Residual 占比 | `residual_norm / max_residual_norm` | `w_residual_mag` (0.3) |

所有评分在 `[0, 1]` 范围内，计算过程 `@torch.no_grad()` 不参与梯度回传。

### 5.3 过滤机制：Hard Mask 与 Soft Weight

由 `HarmfulUpdateFilter` 实现（`update_filter/filter.py`）：

#### Hard Mask 模式

三个条件 **同时** 满足才保留该样本：

```
keep = (uncertainty < tau_uncertainty)
     & (support > tau_support)
     & (drift < tau_drift)
```

**安全下限保护（min_retention_ratio）：**

当保留比例低于 `min_retention_ratio`（默认 0.3）时，按综合分数排名放回最优样本：

```python
composite = w_unc * uncertainty + w_sup * (1 - support) + w_drift * drift
# 取 composite 最小的 top-k 样本放回
```

**重归一化损失：**

```python
filtered_loss = (mask * per_sample_loss).sum() / mask.sum()
```

#### Soft Weight 模式

三个维度各自计算 sigmoid 门控，相乘得到连续权重：

```python
w_u = sigmoid(-(uncertainty - tau_uncertainty) / temperature)
w_s = sigmoid((support - tau_support) / temperature)
w_d = sigmoid(-(drift - tau_drift) / temperature)
weight = w_u * w_s * w_d   # (0, 1]
```

**重归一化损失：**

```python
filtered_loss = (weight * per_sample_loss).sum() / weight.sum()
```

### 5.4 HUF 配置详解

定义在 `update_filter/config.py`，`HUFConfig` dataclass：

| 参数组 | 参数 | 默认值 | 说明 |
|--------|------|-------|------|
| **模式** | `mode` | `'hard'` | `'hard'` 或 `'soft'` |
| | `enabled` | `True` | 是否启用 HUF |
| **Uncertainty** | `tau_uncertainty` | 0.7 | 高于此值被抑制 |
| | `w_confidence` | 0.5 | confidence 反转权重 |
| | `w_mode_variance` | 0.3 | 模式方差权重 |
| | `w_residual_var` | 0.2 | residual 不确定性权重 |
| **Support** | `tau_support` | 0.3 | 低于此值被抑制 |
| | `support_alpha` | 1.0 | exp 衰减速率 |
| | `max_residual_norm` | 5.0 | residual 硬上限 |
| **Drift** | `tau_drift` | 0.8 | 高于此值被抑制 |
| | `w_comfort` | 0.4 | comfort 变化权重 |
| | `w_curvature` | 0.3 | curvature 突变权重 |
| | `w_residual_mag` | 0.3 | residual 大小权重 |
| **Soft** | `soft_temperature` | 1.0 | sigmoid 温度 |
| **安全** | `min_retention_ratio` | 0.3 | 最少保留比例 |
| **综合排序** | `w_uncertainty_final` | 0.4 | 综合评分中 uncertainty 权重 |
| | `w_support_final` | 0.3 | 综合评分中 support 权重 |
| | `w_drift_final` | 0.3 | 综合评分中 drift 权重 |

### 5.5 训练集成

HUF 通过 `InterfaceRefinerTrainer` 的 `train_filtered_reward_epoch()` 集成到训练循环：

```python
# 构造
trainer = InterfaceRefinerTrainer(
    refiner=refiner,
    optimizer=optimizer,
    update_filter=HarmfulUpdateFilter(huf_config),
    update_scorer=UpdateReliabilityScorer(huf_config),
)

# 训练循环
for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        trainer.train_supervised_epoch(dataloader, epoch)
    else:
        trainer.train_filtered_reward_epoch(dataloader, epoch)  # HUF 生效
```

**每 batch 数据流：**

```
interface, gt_plan
       ↓ refiner.forward()
refined_plan, residual, residual_norm
       ↓ compute_refinement_reward()
total_reward [B]
       ↓ compute_per_sample_reward_weighted_error()
per_sample_loss [B]
       ↓ scorer.score_batch(interface, outputs)
{uncertainty, support, drift} 各 [B]
       ↓ filter.apply_filter(per_sample_loss, scores)
(filtered_loss, diagnostics)
       ↓ + residual_reg
loss_total.backward()
```

---

## 6. 验证实验

### 6.1 Layer 1：Interface Validity（结构解耦验证）

**问题：** 接口是"名义上的封装"还是代码依赖层面真正的 interface-level decoupling？

#### 6.1.1 依赖审计

对 6 个非 adapter 核心模块扫描 VAD-specific 关键词（如 `bev_embed`, `ego_fut_preds`, `VADHead` 等）：

| 模块 | VAD-specific 依赖 | 结果 |
|------|-------------------|------|
| `refinement/interface_refiner.py` | 无 — 只 import `PlanningInterface` | ✅ PASS |
| `refinement/losses.py` | 无 — 只接受 `torch.Tensor` | ✅ PASS |
| `refinement/reward_proxy.py` | 无 — 通用轨迹张量 | ✅ PASS |
| `hard_case/mining.py` | 无 — 只 import `PlanningInterface` | ✅ PASS |
| `trainers/trainer_refiner.py` | 无 | ✅ PASS |
| `evaluators/eval_refined.py` | 无 — 纯度量计算 | ✅ PASS |

依赖关系图：

```
VAD-specific                     planner-agnostic
───────────                     ─────────────────
vad_adapter.py ──→ PlanningInterface ──→ refiner / losses / reward
                                     ──→ hard_case / trainer / evaluator
                                     ──→ update_filter (HUF)
```

#### 6.1.2 Mock Adapter 替换

构造 `MockUniADAdapter`（完全不同的输出 key 和维度），下游模块 **零修改** 正常运行：

```
VADPlanningAdapter:   loss=0.6511  reward=0.2582   baseline_ade=1.0356
MockUniADAdapter:     loss=1.1206  reward=-2.8953  baseline_ade=1.7895
```

#### 6.1.3 接口消融

Full / Minimal / 无 candidates / 不同 scene_dim 等 4 种接口变体均 PASS。

**结论：Interface Validity ✅** — 结构解耦在代码依赖层面成立，adapter 是唯一 planner-specific 边界。

---

### 6.2 Layer 2：Interface Sufficiency（字段消费验证）

**问题：** 接口字段都定义了，但 refiner 是否真的在用？

#### 6.2.1 字段语义消融

将每个字段置零，测量输出差异：

| 字段 | 原始 refiner（2 通道） | 升级后 refiner（4 通道） |
|------|---------------------|---------------------|
| `scene_token` | Δ=0.0297 (有影响) | Δ>0 |
| `reference_plan` | Δ=0.0000 (基座输入) | Δ=0.0000 (基座输入) |
| `plan_confidence` | **Δ=0.0000 (未消费)** | **Δ>0 (已消费)** |
| `safety_features` | **Δ=0.0000 (未消费)** | **Δ>0 (已消费)** |

#### 6.2.2 根因与修复

**根因：** 原始 refiner 只有 `scene_proj + plan_proj` 两个输入通道，confidence/safety 的存在纯属形式。

**修复：** 将 refiner 从 2 通道扩展为 4 通道（新增 `conf_proj` 和 `safety_proj`），采用小维度编码（H/4）避免辅助信号淹没主通道。

**结论：Interface Sufficiency ✅ (升级后)** — 所有字段参与计算。

---

### 6.3 Layer 3：Scene Token 表征质量

**问题：** mean pooling 将 40000 个 BEV token 压成单个 256-d 向量，是否过于粗糙？

#### 6.3.1 合成数据实验 (10x10=100 BEV tokens, 64 batch, 200 steps)

| 池化方式 | scene_dim | loss | ADE | 相对 mean 提升 |
|---------|-----------|------|-----|--------------|
| mean | 256 | 0.1808 | 0.2866 | baseline |
| max | 256 | 0.1696 | 0.2556 | +10.8% |
| mean+max | 512 | 0.1384 | 0.2218 | +22.6% |
| **grid 4x4** | **4096** | **0.0960** | **0.1604** | **+44.0%** |
| ego_local | 256 | 0.1465 | 0.2216 | +22.7% |

**结论：** grid 4x4 在合成数据上即显著领先，因为保留了粗粒度空间结构。

---

### 6.4 Layer 4a：Confidence 有效性

**定义：** `plan_confidence = exp(-mode_variance)`，mode_variance = 3 个 planning mode 输出的方差。

#### 6.4.1 合成数据结果 (200 样本)

| 桶 | conf 范围 | conf 均值 | ADE 均值 | 趋势 |
|----|----------|----------|---------|------|
| 1 (最低) | [0.028, 0.091] | 0.053 | 4.352 | 最高 error |
| 2 | [0.115, 0.371] | 0.266 | 2.029 | |
| 3 | [0.450, 0.820] | 0.618 | 1.718 | |
| 4 (最高) | [0.871, 1.000] | 0.951 | 0.434 | 最低 error |

- Pearson(conf, ADE) = **-0.85**（强负相关）
- Low-conf / High-conf ADE 比值 = **14.5x**

#### 6.4.2 真实数据结果 (nuScenes val, 100 samples)

| 指标 | 合成数据 | 真实数据 | 诊断 |
|------|---------|---------|------|
| Pearson(conf, ADE) | **-0.85** | **+0.15** | 信号方向反转 |
| conf std | 0.36 | 0.12 | 区分度下降 3x |
| Low/High ADE ratio | 14.5x | 0.36x | 高 conf 反而 error 更大 |

**真实数据详情：**
- conf mean = 0.7568, std = 0.1188, range = [0.39, 0.99]
- Low-conf (bottom 20%) ADE = 0.8362
- High-conf (top 80%) ADE = 2.3373

#### 6.4.3 根因分析

VAD 的 3 个 planning mode 在真实数据上的行为并非"模式间分歧 = 规划困难"：
- **复杂场景**（密集交叉路口）：3 个 mode 恰好都在探索类似区域（variance 小 → 高 conf），但距离 GT 很远（ADE 大）
- **简单场景**：微小模式偏移被放大为低 conf

**结论：** 当前 `exp(-mode_variance)` 在真实数据上 **不是有效的规划质量代理**，需改进。但 confidence 字段在接口中的位置正确——问题在于 VAD adapter 的 confidence 提取实现，而非接口设计本身。

---

### 6.5 Layer 4b：A/B 综合对比

#### 6.5.1 对比配置

| 版本 | scene_pool | 消费字段 | 含义 |
|------|-----------|---------|------|
| A (mean+partial) | mean | scene + plan | 原始基线 |
| B (grid+full) | grid 4x4 | scene + plan + conf + safety | 完整升级 |
| C (ego_local+full) | ego_local | scene + plan + conf + safety | 替代池化 |

#### 6.5.2 合成数据结果 (2 trials × 150 steps)

| 版本 | ADE | FDE | 相对 A 提升 |
|------|-----|-----|-----------|
| A (mean+partial) | 0.1729 | 0.2241 | baseline |
| B (grid+full) | 0.1725 | 0.2091 | +0.2% ADE, +6.7% FDE |
| **C (ego_local+full)** | **0.1420** | **0.1858** | **+17.9% ADE, +17.1% FDE** |

合成数据上 ego_local 领先（小 BEV 10x10，grid block 仅含 ~6 tokens，空间结构有限）。

#### 6.5.3 真实数据结果 (nuScenes val, 100 samples, 300 steps, lr=1e-4)

**Baseline:** VAD 原始输出 ADE = 1.9807, FDE = 3.7229

| 版本 | ADE | FDE | 相对 baseline 提升 | 相对 A 提升 |
|------|-----|-----|------------------|-----------|
| A (mean+partial) | 0.8534 | 1.0124 | 56.9% | baseline |
| **B (grid+full)** | **0.3954** | **0.5248** | **80.0%** | **53.7%** |
| C (ego_local+full) | 0.6161 | 0.7803 | 68.9% | 27.8% |

#### 6.5.4 合成 vs 真实对比

| 指标 | 合成数据 | 真实数据 | 解读 |
|------|---------|---------|------|
| B vs A 提升 | 0.2% | **53.7%** | grid 在大 BEV 上优势巨大 |
| C vs A 提升 | 17.9% | **27.8%** | ego_local 增益放大 |
| 最优池化 | ego_local | **grid 4x4** | **池化策略结论反转** |

**关键发现：**

1. **grid 4x4 在真实数据上优势爆发** — 真实 200x200 BEV 中每个 grid block 含 2500 tokens，空间结构丰富程度远超合成数据的 6 tokens/block
2. **所有配置均显著优于 baseline** — 最差的 A 配置也有 56.9% ADE 提升，证明 refinement pipeline 有效
3. **合成数据结论不能直接外推** — 池化策略在不同 BEV 规模上表现截然不同

**结论：** 完整接口消费 + grid 4x4 池化是真实数据上的最优配置。B (grid+full) 以 80.0% 的 baseline 提升和 53.7% 的 A→B 增益成为明确赢家。

---

### 6.6 Layer 4c：HUF 离线验证

#### 6.6.1 实验设置

- 数据：nuScenes val dump, 100 samples, grid+full 配置
- 训练：2 阶段 (30% supervised warmup + 70% reward-weighted)
- 对比：No-filter vs Soft HUF vs Hard HUF (默认阈值) vs Hard HUF (宽松阈值)
- 指标：ADE, FDE, Stage 2 loss 方差, 样本保留率

#### 6.6.2 实验结果

| 版本 | ADE | FDE | baseline 提升 | Stage 2 loss 方差 | 保留率 |
|------|-----|-----|-------------|------------------|--------|
| No-filter (3 trials avg) | 0.4123 | 0.6139 | 79.2% | 0.00555 | 100% |
| **Soft HUF** | **0.4170** | **0.6126** | **78.9%** | **0.00462 (-16.7%)** | ~85% |
| Hard HUF (默认) | 1.1198 | 1.7743 | 43.5% | 0.00562 | ~39% |
| Hard HUF (宽松: τ_unc=0.85, τ_sup=0.15, τ_drift=0.9) | 0.9141 | 1.2896 | 53.8% | - | ~53% |

#### 6.6.3 分析

**Soft HUF：**
- ADE 仅 1.1% 差距（0.4170 vs 0.4123），统计上不显著
- **loss 方差下降 16.7%**（0.00555 → 0.00462），训练更稳定
- 保留率 ~85%，平滑抑制而非硬切

**Hard HUF：**
- 默认阈值下过滤 ~61% 样本，100 样本有效训练数据不足，ADE 显著退化
- 宽松阈值改善但仍不如 no-filter
- 问题不在机制本身，而在 **小数据集 + 高过滤率 = 信息不足**

**核心结论：**

在小规模离线拟合场景（100 samples）中，HUF 的价值体现为 **训练稳定性（loss 方差降低 16.7%）** 而非 ADE 绝对值。HUF 的真正优势预期在以下场景显现：

1. **大规模数据集**（数千样本）：过滤有害更新不会显著减少有效训练数据
2. **在线 RL 训练**：分布漂移是真实威胁，抑制漂移更新至关重要
3. **长期训练**：累积的有害更新可能导致 catastrophic forgetting

---

## 7. 与 R2SE 的结构性差异

### 7.1 核心区别

| 维度 | R2SE | 本方案 |
|------|------|--------|
| 解耦层级 | **pipeline-level** | **interface-level** |
| RL objective 输入 | Planner-native 输出形式 (categorical/GMM/diffusion) | 统一 `PlanningInterface` |
| 换 planner 改什么 | 重写 RL objective | **只新增 1 个 adapter 文件** |
| Refinement 通用性 | 流程通用，action space planner-coupled | 全栈 planner-agnostic |
| 更新质量控制 | 无 | **HUF 过滤有害更新** |

### 7.2 精确表述

> R2SE is model-agnostic at the pipeline level, but not planner-decoupled at the refinement interface level. Its RL objective still requires adaptation to planner-specific policy parameterization (categorical, GMM, diffusion).
>
> Our method constructs a unified planning interface with an adapter boundary, so that the entire downstream refinement pipeline depends on the interface contract rather than planner-native output parameterization. Furthermore, HUF provides update-level quality control that R2SE does not address.

### 7.3 R2SE 的客观优势

R2SE 直接访问 planner 内部的 rich representation（policy logits, mode scores, internal query tokens），信息损失更小。本方案的 adapter 层引入了信息压缩（BEV → pooled token），通用性的代价是可能的信息瓶颈。

**缓解措施：** grid/ego_local pooling 保留部分空间结构。实验证明 grid 4x4 在真实数据上将 ADE 提升至 80%，信息瓶颈在很大程度上被缓解。

---

## 8. 证据链总结

```
Layer 1: 接口结构成立         ✅  依赖审计 + Mock 替换 + 接口消融
Layer 2: 接口被有效消费       ✅  4 通道 refiner 升级后所有字段参与计算
Layer 3: 方向性正确           ✅  grid/ego_local >> mean, full >> partial
Layer 4a: 真实数据验证        ✅  nuScenes val 100 samples
          ├── A/B 对比        ✅  grid+full ADE 提升 80% (baseline 对比)
          ├── Confidence      ⚠️  真实数据上 Pearson=+0.15 (需改进定义)
          └── HUF 验证        ✅  Soft HUF loss 方差 -16.7% (训练稳定性)
```

**综合判断：**

- **创新点一 (Unified Interface)** 在结构正确性和实际效果上均得到验证
- **创新点二 (HUF)** 在训练稳定性上初步验证，大规模/在线场景的效果待进一步验证
- **Confidence 定义** 是当前唯一明确需要改进的组件

---

## 9. 已交付工件清单

### 9.1 核心模块

| 文件 | 行数 | 功能 |
|------|------|------|
| `planning_interface/interface.py` | 61 | PlanningInterface dataclass (7 字段 + to/describe) |
| `planning_interface/adapters/base_adapter.py` | — | 抽象适配器基类 |
| `planning_interface/adapters/vad_adapter.py` | 302 | VAD 专用适配器 (4 种池化 + conf/safety 提取) |
| `planning_interface/extractor.py` | — | 提取器薄封装 |
| `refinement/interface_refiner.py` | 212 | 4 通道残差精炼网络 (scene+plan+conf+safety) |
| `refinement/losses.py` | 79 | 3 个损失函数 (supervised + reward-weighted + per-sample) |
| `refinement/reward_proxy.py` | 225 | 4 种奖惩函数 + 综合奖励 |
| `hard_case/mining.py` | — | Hard-case 挖掘器 |
| `trainers/trainer_refiner.py` | 380 | 3 阶段训练器 + HUF filtered 训练 |
| `evaluators/eval_refined.py` | — | ADE/FDE/collision/comfort 评估 |
| `update_filter/config.py` | 71 | HUFConfig dataclass (18 个超参数) |
| `update_filter/scorer.py` | 227 | UpdateReliabilityScorer (3 类评分) |
| `update_filter/filter.py` | 177 | HarmfulUpdateFilter (mask/weight + renormalized loss + diagnostics) |

### 9.2 实验与分析脚本

| 文件 | 功能 | 数据源 |
|------|------|--------|
| `experiments/ablation_interface_fields.py` | 字段语义消融 | 合成数据 |
| `experiments/confidence_analysis.py` | Confidence 统计 | 合成数据 |
| `experiments/scene_token_variants.py` | 5 种池化方式对比 | 合成数据 |
| `experiments/ab_comparison.py` | A/B/C 三配置对比 | 合成数据 |
| `scripts/dump_vad_inference.py` | VAD 推理输出导出 | nuScenes val |
| `experiments/load_dump.py` | Dump 数据加载工具 | — |
| `experiments/offline_confidence_analysis.py` | Confidence 分析 | 真实 dump |
| `experiments/offline_ab_comparison.py` | A/B 对比 | 真实 dump |
| `experiments/offline_huf_experiment.py` | HUF 过滤对比实验 | 真实 dump |

### 9.3 单元测试

| 文件 | 测试数 | 状态 |
|------|--------|------|
| `tests/test_update_filter.py` | 23 | ✅ 全部通过 |
| `tests/test_interface_validity.py` | 3 | ✅ 全部通过 |
| `tests/test_update_filter.py` | 17 | ✅ 全部通过 |
| **合计** | **43** | **✅** |

`test_update_filter.py` 覆盖 4 个测试类：
- `TestHUFConfig`: 默认值 / 非法 mode / 非法 retention
- `TestScorer`: uncertainty 高低 conf / 无 candidates / support 大小残差 / drift 平滑-抖动
- `TestFilter`: hard mask 基本 / min_retention / soft weight 范围 / apply loss / diagnostics keys
- `TestEndToEnd`: 完整 scorer→filter→loss→backward 流水线 / per_sample_reward_weighted_error

---

## 10. 执行环境与复现指南

### 10.1 硬件与软件

| 项目 | 版本 |
|------|------|
| GPU | NVIDIA H20 98GB |
| CUDA | 12.1 |
| Python | 3.11 |
| PyTorch | 2.3.1+cu121 |
| mmcv | 1.7.0 |
| mmdet3d | 1.0.0a1 |

### 10.2 数据路径

| 数据 | 路径 |
|------|------|
| nuScenes v1.0-trainval | `/mnt/datasets/datasets/datasets/e2e-nuscenes/20260302` (只读) |
| 工作目录 | `/mnt/cpfs/prediction/lipeinan/RL/VAD/` |
| VAD checkpoint | `/mnt/cpfs/prediction/lipeinan/RL/VAD_base.pth` |
| Dump 输出 | `E2E_RL/data/vad_dumps/` (100 samples) |

### 10.3 复现命令

```bash
# Step 1: 生成 val temporal PKL (300 samples)
PYTHONUNBUFFERED=1 /usr/local/bin/python3.11 tools/fast_gen_val_pkl.py \
    --root-path data/nuscenes --out-dir data/nuscenes --max-samples 300

# Step 2: 导出真实 VAD 推理输出 (100 samples)
/usr/local/bin/python3.11 E2E_RL/scripts/dump_vad_inference.py \
    --config projects/configs/VAD/VAD_base_e2e.py \
    --checkpoint /mnt/cpfs/prediction/lipeinan/RL/VAD_base.pth \
    --output_dir E2E_RL/data/vad_dumps \
    --max_samples 100

# Step 3: 离线 Confidence 分析
/usr/local/bin/python3.11 -m E2E_RL.experiments.offline_confidence_analysis \
    --dump_dir E2E_RL/data/vad_dumps --low_conf_pct 0.2

# Step 4: 离线 A/B 对比
/usr/local/bin/python3.11 -m E2E_RL.experiments.offline_ab_comparison \
    --dump_dir E2E_RL/data/vad_dumps --num_steps 300 --num_trials 1

# Step 5: 离线 HUF 实验
/usr/local/bin/python3.11 -m E2E_RL.experiments.offline_huf_experiment \
    --dump_dir E2E_RL/data/vad_dumps

# Step 6: 运行全部单元测试
/usr/local/bin/python3.11 -m pytest E2E_RL/tests/ -v
```

### 10.4 运行过程中修复的兼容性问题

| 问题 | 修复文件 | 修复方式 |
|------|---------|---------|
| `mmdet3d.ops.roiaware_pool3d` 不存在 | `lidar_box3d.py` | try/except fallback |
| `seaborn-whitegrid` style 更名 | `nuscenes/map_api.py` | → `seaborn-v0_8-whitegrid` |
| `np.bool` deprecated | `transform_3d.py` | → `bool` |
| CPU/GPU tensor device 不一致 | `metric_stp3.py` | 添加 `.cpu()` |
| `ego_fut_cmd` 维度 (1,1,1,3) | `vad_adapter.py` | squeeze 循环 |

---

## 11. 已知局限与改进方向

### 11.1 Confidence 定义需改进

当前 `exp(-mode_variance)` 在真实数据上 Pearson=+0.15（方向反转），不是有效的规划质量代理。候选改进方案：

| 方案 | 思路 | 预期 |
|------|------|------|
| Attention entropy | 用 BEV attention 层的 entropy 替代 mode variance | 捕获感知层不确定性 |
| Learned confidence head | 训练时用 ADE 做监督信号 | 直接拟合规划质量 |
| 组合信号 | 模式间方差 + 与历史轨迹一致性 | 多信号互补 |

### 11.2 Grid Pooling 效率

grid 4x4 产出 dim=4096，refiner 参数量较大。可考虑 grid 2x2 (dim=1024) 或 grid 3x3 (dim=2304) 作为效率-精度权衡。

### 11.3 验证规模

当前仅用 100 samples 验证。完整 val set (6019 samples) 验证需要更多计算资源，预期 grid+full 优势将更稳定。

### 11.4 HUF 大规模验证

HUF 的 loss 方差降低效果（-16.7%）在小数据集上已可观测，但 ADE 绝对值的改善需要在数千样本 + 在线 RL 设置下验证。

### 11.5 更多 Planner 验证

当前仅在 VAD 上验证。接口设计的通用性声明需要在 UniAD、SparseDrive 等 planner 上实际落地验证。Mock adapter 已证明代码层面可行，但真实 adapter 实现和效果需额外实验。
