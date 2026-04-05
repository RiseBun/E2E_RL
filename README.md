# E2E_RL 项目说明

## 一、项目定位

E2E_RL 是一个**模型无关的外挂式轨迹精炼框架**，专为自动驾驶端到端(E2E)规划器设计。

**核心价值**：无需修改你的 E2E 模型，只需添加一个轻量级 Adapter，即可获得：
- 轨迹精炼能力（Refiner）
- 智能筛选能力（Scorer）
- 安全兜底保障（HUF）

---

## 二、核心问题与解决方案

### 问题
```
E2E 规划器输出的轨迹不一定最优，需要修正
但不是所有修正都是好的，有些修正反而会让轨迹变差
```

### 解决方案：训练一个"智能筛选器"

```
E2E 规划器 → Refiner 生成修正 → Scorer 评估 → HUF 兜底 → 接受/拒绝修正
                                    ↑
                              核心创新点
```

**创新点**：不是简单接受所有修正，而是让模型学会判断"哪些修正值得应用，哪些应该拒绝"。

---

## 三、Pipeline 整体流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           E2E_RL 完整训练 Pipeline                           │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────────┐      ┌────────────────────┐
  │   E2E Planner │ ──→ │  Adapter (提取器) │ ──→ │  PlanningInterface │
  │ (任意规划器)  │      │   (新建一个)      │      │    (统一接口)      │
  └──────────────┘      └──────────────────┘      └─────────┬──────────┘
                                                             │
                    ┌────────────────────────────────────────┼────────────────┐
                    │                                        │                │
                    ▼                                        ▼                ▼
           ┌────────────────┐                      ┌───────────────┐  ┌──────────┐
           │ InterfaceRefiner│                      │  RewardProxy  │  │  Scorer  │
           │   (残差网络)    │                      │  (奖励计算)   │  │ (可靠性) │
           └───────┬────────┘                      └───────────────┘  └────┬─────┘
                   │                                                   │
                   │                    ┌──────────────────────────────┘
                   │                    │
                   ▼                    ▼
           ┌────────────────┐    ┌──────────────┐
           │  refined_plan   │    │   scores     │
           │   (精炼轨迹)     │    │  (评分)      │
           └────────────────┘    └──────┬───────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │ HarmfulUpdateFilter │
                            │    (有害更新过滤)    │
                            └──────────┬──────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │  Filtered Loss   │
                              │   (过滤后损失)    │
                              └──────────────────┘
```

### 三层筛选架构

```
输入: PlanningInterface (原始场景 + 轨迹)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Refiner (生成层)                                  │
│  职责: 生成候选修正                                          │
│  输出: refined_plan = reference_plan + residual             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Scorer (智能筛选层) ← 核心创新                    │
│  职责: 预测 Gain↑ Risk↓                                    │
│  架构: MLP 双头回归 (Gain Head + Risk Head)                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: HUF (规则兜底层)                                  │
│  职责: 硬性安全规则检查                                      │
│  检查: 碰撞 / 偏离道路 / 突变幅度 / 舒适度                   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
最终输出: 接受安全的修正 → 更优轨迹
```

---

## 四、项目结构

```
E2E_RL/
├── configs/                          # 配置文件
│   ├── refiner_debug.yaml            # Refiner + HUF
│   ├── refiner_only.yaml            # 纯 Refiner
│   ├── refiner_plus_scorer.yaml     # Refiner + Scorer
│   └── refiner_scorer_huf.yaml      # 完整三层系统
│
├── data/                             # 数据
│   ├── vad_dataset.py               # VAD 数据集
│   ├── dataloader.py               # 数据加载器
│   └── vad_dumps/                  # 预处理数据 (100 samples)
│
├── planning_interface/               # 统一接口
│   ├── interface.py                 # PlanningInterface 定义
│   ├── extractor.py                 # 提取器 (需修改注册)
│   └── adapters/                    # 规划器适配器 (需新建)
│       ├── base_adapter.py          # 基类
│       ├── vad_adapter.py           # VAD 适配器
│       └── diffusiondrive_adapter.py
│
├── refinement/                       # 轨迹精炼
│   ├── interface_refiner.py         # InterfaceRefiner 模型
│   └── reward_proxy.py              # 奖励代理
│
├── update_filter/                    # 筛选机制
│   ├── scorer.py                    # UpdateReliabilityScorer
│   ├── model.py                     # ReliabilityNet (MLP)
│   ├── huf.py                       # HarmfulUpdateFilter
│   └── config.py                    # HUF 配置
│
├── trainers/                         # 训练器
│   └── trainer_refiner.py          # 两阶段训练
│
├── scripts/                          # 脚本
│   ├── train_interface_refiner.py   # 训练入口
│   ├── train_scorer.py             # 单独训练 Scorer
│   └── compare_experiments.py       # 实验对比
│
└── experiments/                      # 实验结果
    ├── refiner_only/               # 纯 Refiner
    ├── refiner_plus_scorer/        # Refiner + Scorer
    ├── refiner_debug/              # Refiner + HUF
    ├── refiner_scorer_huf/         # 完整三层系统
    └── scorer_training/            # 单独训练的 Scorer
```

---

## 五、核心模块说明

### 1. PlanningInterface (统一接口)

封装所有规划器的输出为统一格式：

```python
@dataclass
class PlanningInterface:
    scene_token: torch.Tensor      # [B, D] 场景特征
    reference_plan: torch.Tensor   # [B, T, 2] 参考轨迹
    plan_confidence: torch.Tensor  # [B, 1] 置信度
    candidate_plans: torch.Tensor   # [B, M, T, 2] 多模态候选
    safety_features: Dict          # 安全特征
```

### 2. InterfaceRefiner (残差网络)

学习对原始轨迹的小幅修正：

```python
refined_plan = reference_plan + residual
# residual = Refiner(interface)  # 预测的修正量
```

### 3. ReliabilityScorer (学习型筛选器) ← 核心创新

预测修正的好坏（双头回归）：

```python
pred_gain, pred_risk = scorer(interface, residual)
# Gain: 预期奖励提升 (越大越好)
# Risk: 预期风险 (越小越好)
```

### 4. HarmfulUpdateFilter (规则筛选器)

基于规则的硬性安全检查：

- 碰撞检测：修正后会不会撞车？
- 偏离检测：修正后会不会离开道路？
- 突变检测：修正幅度是否过大？

---

## 六、伪 RL 训练范式

### 本质：离线奖励加权监督学习

```
传统 RL (PPO/SAC):          E2E_RL (你的框架):
┌─────────┐                 ┌─────────┐
│ Policy  │ ──→ Env ──→ Reward ──→ Policy 更新
│  (需要在线探索)            │ Planner │ ──→ Refiner ──→ Reward ──→ Refiner 更新
└─────────┘                  │ (冻结)  │ ──→ (不需要环境交互)
     ↑                        └─────────┘
     └──────────────── 离线
```

### 两阶段训练

| 阶段 | 方法 | 目标 |
|------|------|------|
| Stage 1 | 监督预热 | L1 Loss + 残差正则化 |
| Stage 2 | 奖励加权 | 高奖励样本权重↑，低奖励样本权重↓ |

### 奖励函数

```
total_reward = w_progress × r_progress 
             - w_collision × p_collision 
             - w_offroad × p_offroad 
             - w_comfort × p_comfort
```

---

## 七、消融实验验证结果

### 实验设计

| 实验 | Scorer | HUF | 说明 |
|------|--------|-----|------|
| **refiner_only** | ❌ | ❌ | 纯 Refiner，无筛选 |
| **refiner_plus_scorer** | ✅ | ❌ | 学习型筛选，无规则兜底 |
| **refiner_debug** | ❌ | ✅ | 规则筛选，无学习模型 |
| **refiner_scorer_huf** | ✅ | ✅ | 完整三层架构 |

### 性能对比

| 排名 | 实验 | Scorer | HUF | Reward | Stage 1 Loss |
|------|------|--------|-----|--------|--------------|
| 🥇 1 | **refiner_only** | ❌ | ❌ | **-0.43** | 2.42 |
| 🥈 2 | **refiner_scorer_huf** | ✅ | ✅ | -0.45 | 2.55 |
| 🥉 3 | **refiner_plus_scorer** | ✅ | ❌ | -0.59 | 2.58 |
| 4 | **refiner_debug** | ❌ | ✅ | -0.68 | 2.74 |

### 关键发现

1. **无筛选效果最好**：纯 Refiner 表现最优，Reward = -0.43

2. **Scorer 的负面影响**：
   - refiner_plus_scorer (-0.59) 比 refiner_only (-0.43) 差 37%
   - 可能原因：Scorer 过于保守，拒绝了太多"好的修正"

3. **HUF 的负面影响**：
   - refiner_debug (-0.68) 比 refiner_only (-0.43) 差 58%
   - 可能原因：规则过于严格

4. **组合有一定补偿**：
   - refiner_scorer_huf (-0.45) 接近最优
   - Scorer + HUF 的组合比单独使用好

### 结论

```
筛选机制验证结论:
├── ❌ 当前筛选策略过于保守
├── ❌ 拒绝了太多有价值的修正
├── ✅ Scorer + HUF 组合优于单独使用
└── 💡 建议：调整阈值，降低保守程度
```

---

## 八、训练与评估

### 训练命令

```bash
# 激活环境
conda activate e2e_rl

# 设置路径
export PYTHONPATH="/mnt:$PYTHONPATH"

# 纯 Refiner (Baseline)
python -m E2E_RL.scripts.train_interface_refiner \
    --config /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/refiner_only.yaml

# Refiner + Scorer
python -m E2E_RL.scripts.train_interface_refiner \
    --config /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/refiner_plus_scorer.yaml

# Refiner + HUF
python -m E2E_RL.scripts.train_interface_refiner \
    --config /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/refiner_debug.yaml

# 完整三层系统
python -m E2E_RL.scripts.train_interface_refiner \
    --config /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/refiner_scorer_huf.yaml
```

### 查看实验对比

```bash
python /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/scripts/compare_experiments.py
```

---

## 九、核心设计原则

| 设计点 | 说明 |
|--------|------|
| **模型无关** | 通过 Adapter 模式适配任意 E2E 规划器 |
| **外挂框架** | 无需修改你的模型，只需新建 Adapter |
| **统一接口** | PlanningInterface 是所有规划器输出的统一表示 |
| **残差学习** | Refiner 不从头预测，只学习微调 |
| **离线可用** | 奖励代理支持离线计算 |
| **安全优先** | 多层筛选机制确保安全 |

---

## 十、集成新 E2E 模型指南

### 4步快速接入

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     E2E_RL 外挂框架 - 4步集成                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: 分析你的模型输出格式
         ↓
Step 2: 新建 Adapter 文件 (1个文件)
         ↓
Step 3: 注册 Adapter (修改2处)
         ↓
Step 4: 新建配置文件 (1个文件)

其余全部复用，无需修改！
```

### Step 1: 分析模型输出

```python
# 你的模型输出结构（任意格式）
your_model_outputs = {
    'bev_features': torch.Tensor,      # [B, H, W, C]
    'trajectory': torch.Tensor,        # [B, T, 3]
    'mode_probs': torch.Tensor,       # [B, M]
    'object_boxes': torch.Tensor,     # 检测框
    'map_polygons': torch.Tensor,     # 地图多边形
}
```

### Step 2: 新建 Adapter

```python
# 文件: E2E_RL/planning_interface/adapters/your_model_adapter.py

from .base_adapter import BasePlanningAdapter
import torch

class YourModelAdapter(BasePlanningAdapter):
    """YourModel → PlanningInterface 适配器"""
    
    def __init__(self, scene_pool='mean', ego_fut_mode=3, fut_ts=6):
        self.scene_pool = scene_pool
        self.ego_fut_mode = ego_fut_mode
        self.fut_ts = fut_ts
    
    def extract_scene_token(self, planner_outputs):
        """提取场景 token: [B, D]"""
        bev = planner_outputs['bev_features']  # [B, H, W, C]
        
        # 转为 [B, N, D]
        if bev.dim() == 4:
            B, H, W, C = bev.shape
            bev = bev.reshape(B, H * W, C)
        
        # 池化
        if self.scene_pool == 'mean':
            return bev.mean(dim=1)
        elif self.scene_pool == 'ego_local':
            center = H // 2
            k = 16
            return bev[:, (center-k)*W + center:(center+k)*W + center, :].mean(dim=1)
        return bev.mean(dim=1)
    
    def extract_reference_plan(self, planner_outputs, ego_fut_cmd=None):
        """提取轨迹: [B, T, 2]"""
        traj = planner_outputs['trajectory']  # [B, T, 3]
        xy = traj[..., :2]  # 取 x, y
        
        if 'mode_probs' in planner_outputs:
            probs = planner_outputs['mode_probs']
            return xy, torch.cat([xy.unsqueeze(1)] * probs.shape[1], dim=1)
        
        return xy, None
    
    def extract_plan_confidence(self, planner_outputs, ego_fut_cmd=None):
        """提取置信度: [B, 1]"""
        if 'mode_probs' in planner_outputs:
            probs = planner_outputs['mode_probs']
            return probs.max(dim=-1).values.unsqueeze(-1)
        return torch.ones(planner_outputs['trajectory'].shape[0], 1)
    
    def extract_safety_features(self, planner_outputs):
        """提取安全特征"""
        safety = {}
        if 'object_boxes' in planner_outputs:
            n_objects = planner_outputs['object_boxes'].shape[1]
            safety['object_density'] = torch.tensor([n_objects])
        return safety if safety else None
```

### Step 3: 注册 Adapter

编辑 `adapters/__init__.py`:
```python
from .your_model_adapter import YourModelAdapter
```

编辑 `planning_interface/extractor.py`:
```python
# 在 PlanningInterfaceExtractor.from_config 中添加
elif adapter_type == 'your_model':
    from .adapters import YourModelAdapter
    return cls(YourModelAdapter(**kwargs))
```

### Step 4: 新建配置文件

```yaml
# 文件: configs/refiner_your_model.yaml

model:
  scene_dim: 256
  plan_len: 12
  hidden_dim: 256

adapter:
  type: your_model       # ← 你的适配器类型
  scene_pool: ego_local

data:
  data_dir: /path/to/your/dumps
  batch_size: 4

training:
  supervised:
    epochs: 5
    lr: 0.001
  reward_weighted:
    epochs: 3
    lr: 0.0005

output_dir: /path/to/your/experiment
```

### 完整文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新建** | `adapters/your_model_adapter.py` | Adapter 实现 |
| **修改** | `adapters/__init__.py` | 导入你的 Adapter |
| **修改** | `extractor.py` | 注册适配器类型 |
| **新建** | `configs/refiner_your_model.yaml` | 配置文件 |

**其余文件全部复用，无需修改！**

---

## 十一、总结

E2E_RL 是一个**模型无关的外挂式轨迹精炼框架**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         E2E_RL 是什么？                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ 模型无关外挂: 通过 Adapter 适配任意 E2E 规划器                            │
│  ✓ 三层筛选架构: Refiner → Scorer → HUF                                    │
│  ✓ 混合训练范式: 监督预热 + 奖励加权精炼                                     │
│  ✓ 离线可用: 奖励代理支持离线计算                                            │
│  ✓ 安全优先: 多层筛选机制确保安全                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

集成新模型只需要:
┌─────────────────────────────────────────────────────────────┐
│  1. 新建 1 个 Adapter 文件                                   │
│  2. 修改 2 处注册代码                                        │
│  3. 新建 1 个配置文件                                        │
│                                                             │
│  其余全部复用，无需修改！                                    │
└─────────────────────────────────────────────────────────────┘
```
