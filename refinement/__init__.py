# Refinement 模块（部分保留）
# 保留 reward_proxy 用于计算 safe_reward
# InterfaceRefiner 和旧 losses 已废弃，由 correction_policy 模块替代

from .reward_proxy import compute_refinement_reward

__all__ = [
    'compute_refinement_reward',
]
