"""
E2E Finetuning 模块 - Phase 2: 端到端 RL 微调

核心组件:
- conservative_rl: 保守 RL 更新 (Reward-Cost 分离, Beneficial Update Filter)
- hydra_traj_head_e2e: 通用增强规划头 (LoRA, Value Head)
- vad_e2e_wrapper: VAD 模型 E2E 包装器
- diffusiondrive_e2e_wrapper: DiffusionDrive E2E 包装器
- reward: 闭环奖励计算
"""

from .reward import ClosedLoopReward, RewardConfig, RewardNormalizer
from .conservative_rl import (
    ConservativeRLConfig,
    ConservativeRLUpdate,
    ConservativeE2ETrainer,
    RewardCostSeparator,
    BeneficialUpdateFilter,
)
from .hydra_traj_head_e2e import (
    LoRAConfig,
    LoRALinear,
    ValueHead,
    HydraTrajHeadE2E,
    E2EFinetuningWrapper,
)
from .vad_e2e_wrapper import (
    VADE2EConfig,
    VADHeadE2E,
    VADModelE2E,
    wrap_vad_head,
    wrap_vad_model,
)
from .diffusiondrive_e2e_wrapper import (
    DiffusionDriveE2EConfig,
    DiffusionDriveHeadE2E,
    wrap_diffusiondrive_head,
)

__all__ = [
    # Reward
    'ClosedLoopReward',
    'RewardConfig',
    'RewardNormalizer',
    # Conservative RL
    'ConservativeRLConfig',
    'ConservativeRLUpdate',
    'ConservativeE2ETrainer',
    'RewardCostSeparator',
    'BeneficialUpdateFilter',
    # E2E Head (通用)
    'LoRAConfig',
    'LoRALinear',
    'ValueHead',
    'HydraTrajHeadE2E',
    'E2EFinetuningWrapper',
    # VAD Wrapper
    'VADE2EConfig',
    'VADHeadE2E',
    'VADModelE2E',
    'wrap_vad_head',
    'wrap_vad_model',
    # DiffusionDrive Wrapper
    'DiffusionDriveE2EConfig',
    'DiffusionDriveHeadE2E',
    'wrap_diffusiondrive_head',
]