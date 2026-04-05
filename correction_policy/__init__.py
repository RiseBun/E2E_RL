"""
Correction Policy 模块 - 真正的 RL Policy

核心组件：
- GaussianCorrectionActor: 高斯策略网络
- DeterministicCorrectionActor: 确定性策略（对比用）
- CorrectionPolicy: 统一策略接口
- losses: BC + Policy Gradient 损失函数
"""

from .actor import GaussianCorrectionActor, DeterministicCorrectionActor
from .policy import CorrectionPolicy
from .losses import (
    behavioral_cloning_loss,
    policy_gradient_loss,
    compute_advantage,
    entropy_bonus_loss,
    combined_policy_loss,
)

__all__ = [
    'GaussianCorrectionActor',
    'DeterministicCorrectionActor',
    'CorrectionPolicy',
    'behavioral_cloning_loss',
    'policy_gradient_loss',
    'compute_advantage',
    'entropy_bonus_loss',
    'combined_policy_loss',
]
