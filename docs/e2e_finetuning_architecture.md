# E2E_RL End-to-End Finetuning Architecture

## Overview

This document describes the architecture for **Phase 2** of Route C: 
**End-to-End RL Finetuning** to enhance the E2E model itself.

## Architecture Design

```
┌─────────────────────────────────────────────────────────────────┐
│              End-to-End RL Finetuning Architecture               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  E2E Model (Backbone + Perception + Planning Head)      │  │
│  │                                                          │  │
│  │  ┌─────────┐  ┌──────────┐  ┌────────────────────────┐  │  │
│  │  │ Images  │→ │ Backbone │→ │ BEV Feature (bev_embed) │  │  │
│  │  └─────────┘  └──────────┘  └────────────────────────┘  │  │
│  │                                         ↓                │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Planning Head (LoRA Finetunable)               │   │  │
│  │  │  - Scene Feature Extraction                     │   │  │
│  │  │  - Trajectory Generation                        │   │  │
│  │  │  - PlanningInterface Output                     │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │                         ↓                                │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  PlanningInterface (Unified Output)              │   │  │
│  │  │  - scene_token: [D]  (scene feature)            │   │  │
│  │  │  - reference_plan: [T, 2] (trajectory)         │   │  │
│  │  │  - plan_confidence: [K] (optional)              │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Closed-Loop Reward Computation                         │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────┐    │  │
│  │  │  Reward Components:                             │    │  │
│  │  │  - collision_penalty: collision risk           │    │  │
│  │  │  - offroad_penalty: off-road violation         │    │  │
│  │  │  - progress_reward: forward progress (route completion)  │    │  │
│  │  │  - comfort_penalty: acceleration/jerk          │    │  │
│  │  │  -TTC_reward: time-to-collision                │    │  │
│  │  └────────────────────────────────────────────────┘    │  │
│  │                         ↓                                │  │
│  │  total_reward = Σ(w_i * component_i)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RL Loss Computation (PPO-style)                         │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Conservative Update Mechanisms:                  │  │  │
│  │  │  1. KL Divergence Constraint                      │  │  │
│  │  │  2. Reference Model Anchoring                    │  │  │
│  │  │  3. STAPO-style Update Filtering                  │  │  │
│  │  │  4. Conservative RL Step Size                     │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Gradient Backpropagation                               │  │
│  │                                                          │  │
│  │  loss → planning_head → perception → backbone            │  │
│  │  (Full E2E gradient flow for model enhancement)          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. PlanningInterface Enhancement

Extended from E2E_RL to support end-to-end training:

```python
@dataclass
class PlanningInterface:
    """Unified planning interface with gradient support."""
    
    # Core outputs (require grad for E2E training)
    scene_token: torch.Tensor  # [D] - scene feature for RL
    reference_plan: torch.Tensor  # [T, 2] - trajectory (trainable)
    
    # Optional outputs
    candidate_plans: Optional[torch.Tensor] = None
    plan_confidence: Optional[torch.Tensor] = None
    safety_features: Optional[Dict[str, torch.Tensor]] = None
    
    # NEW: RL-specific outputs
    log_prob: Optional[torch.Tensor] = None  # trajectory selection probability
    value: Optional[torch.Tensor] = None  # state value estimation
```

### 2. Closed-Loop Reward Module

```python
class ClosedLoopReward:
    """Compute reward based on closed-loop simulation."""
    
    def compute(
        self,
        trajectory: torch.Tensor,  # [B, T, 2]
        gt_trajectory: torch.Tensor,  # [B, T, 2]
        map_features: Dict,  # drivable area, etc.
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            - collision_penalty: [B]
            - offroad_penalty: [B]
            - progress_reward: [B]
            - comfort_penalty: [B]
            - total_reward: [B]
        """
```

### 3. Conservative Update Mechanism

```python
class ConservativeRLUpdate:
    """Safe policy update with multiple constraints."""
    
    def __init__(
        self,
        kl_target: float = 0.01,
        reference_model: nn.Module = None,
        update_filter: Callable = None,
        max_step: float = 0.1,
    ):
        ...
    
    def compute_loss(
        self,
        policy_output: PlanningInterface,
        reward: torch.Tensor,
        old_log_prob: torch.Tensor,
        reference_plan: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute constrained RL loss."""
```

## Training Pipeline

```python
def train_e2e_finetuning(model, dataloader, optimizer, config):
    """End-to-end RL finetuning loop."""
    
    for batch in dataloader:
        # 1. Forward pass through E2E model
        result = model(
            img=batch['img'],
            command=batch['command'],
            sdc_planning_past=batch['sdc_planning_past'],
            ...
        )
        
        # 2. Extract PlanningInterface
        interface = model.extract_planning_interface(result)
        
        # 3. Compute closed-loop reward
        reward_dict = closed_loop_reward.compute(
            trajectory=interface.reference_plan,
            gt_trajectory=batch['gt_trajectory'],
            map_features=batch['map_features'],
        )
        
        # 4. Compute RL loss with conservative constraints
        loss_dict = conservative_rl.compute_loss(
            policy_output=interface,
            reward=reward_dict['total_reward'],
            old_log_prob=interface.log_prob,
        )
        
        # 5. Backpropagation (full E2E gradient flow)
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # 6. Gradient clipping and update
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
```

## Integration with R2SE Model

### Modified HydraTrajEnsHead

```python
@HEADS.register_module()
class HydraTrajEnsHeadE2E(HydraTrajEnsHead):
    """Extended HydraTrajEnsHead with E2E RL finetuning support."""
    
    def __init__(self, *args, 
                 enable_e2e_rl=False,  # NEW
                 closed_loop_reward_config=None,  # NEW
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_e2e_rl = enable_e2e_rl
        self.closed_loop_reward = ClosedLoopReward(**closed_loop_reward_config)
        
        # NEW: Value head for advantage estimation
        if enable_e2e_rl:
            self.value_head = nn.Sequential(
                nn.Linear(kwargs['d_model'], kwargs['d_ffn']),
                nn.ReLU(),
                nn.Linear(kwargs['d_ffn'], 1),
            )
    
    def extract_planning_interface(self, bev_embed, **kwargs) -> PlanningInterface:
        """Extract PlanningInterface with gradient support."""
        # ... extract scene_token, reference_plan, etc.
        return PlanningInterface(
            scene_token=scene_feature,
            reference_plan=trajectory,
            log_prob=log_prob,  # for PPO
            value=value_estimate,  # for advantage
        )
```

## Configuration Example

```python
# configs/paradrive/exp_100pct/e2e_rl_finetuning.py

model = dict(
    # ... existing R2SE config ...
    
    planning_head=dict(
        type='HydraTrajEnsHeadE2E',  # Changed
        enable_e2e_rl=True,  # NEW
        closed_loop_reward_config=dict(
            collision_weight=0.5,
            offroad_weight=0.5,
            progress_weight=8.0,
            comfort_weight=2.0,
        ),
        # ... existing config ...
    ),
)

# NEW: Conservative RL settings
conservative_rl = dict(
    kl_target=0.01,
    use_reference_anchor=True,
    use_stapo_filter=True,
    max_step_size=0.1,
)
```

## Comparison with E2E_RL External Approach

| Aspect | E2E_RL External | E2E RL Finetuning (This) |
|--------|-----------------|-------------------------|
| Model Parameters | Frozen | Trainable |
| Gradient Flow | Stops at reference_plan | Full E2E |
| Reward Source | Pre-computed from dump | Closed-loop |
| Deployment | Requires external module | Standalone |
| Training Complexity | Lower | Higher |

## Future Work (Phase 3)

After Phase 2, the trained model can be further distilled:

1. **Knowledge Distillation**: Transfer learned knowledge to student model
2. **Weight Compression**: Prune/quantize if needed
3. **Deployment Optimization**: Export to ONNX/TorchScript
