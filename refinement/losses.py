from __future__ import annotations

import torch


def supervised_refinement_loss(
        refined_plan: torch.Tensor,
        gt_plan: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = 'mean',
) -> torch.Tensor:
    """Compute imitation-style L1 loss for refined planning output."""
    error = torch.abs(refined_plan - gt_plan)
    if mask is not None:
        mask = mask.unsqueeze(-1).float()
        error = error * mask
        if reduction == 'mean':
            return error.sum() / mask.sum().clamp(min=1.0)
    if reduction == 'sum':
        return error.sum()
    return error.mean()


def reward_weighted_refinement_loss(
        refined_plan: torch.Tensor,
        gt_plan: torch.Tensor,
        reward: torch.Tensor,
        mask: torch.Tensor | None = None,
        baseline: float = 0.0,
) -> torch.Tensor:
    """Compute a pseudo-RL loss using reward-weighted imitation.

    reward is expected to be higher for better plans. The loss scales
    trajectory error by (1 - normalized_reward).
    """
    if reward.dim() == 2 and reward.shape[1] == 1:
        reward = reward.squeeze(-1)

    normalized = (reward - reward.min()) / (reward.max() - reward.min() + 1e-6)
    weight = 1.0 - normalized.clamp(0.0, 1.0)
    error = torch.abs(refined_plan - gt_plan)
    if mask is not None:
        mask = mask.unsqueeze(-1).float()
        error = error * mask
    error = error.mean(dim=-1).mean(dim=-1)
    return (weight * error).mean()


def compute_per_sample_reward_weighted_error(
        refined_plan: torch.Tensor,
        gt_plan: torch.Tensor,
        reward: torch.Tensor,
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算 per-sample 奖励加权误差，供 HUF 过滤使用。

    与 reward_weighted_refinement_loss 逻辑一致，但返回 [B] 而非标量。

    Args:
        refined_plan: [B, T, 2] 精炼后轨迹
        gt_plan: [B, T, 2] GT 轨迹
        reward: [B] 或 [B, 1] 奖励信号
        mask: [B, T] 有效步掩码

    Returns:
        [B] per-sample 加权误差
    """
    if reward.dim() == 2 and reward.shape[1] == 1:
        reward = reward.squeeze(-1)

    normalized = (reward - reward.min()) / (reward.max() - reward.min() + 1e-6)
    weight = 1.0 - normalized.clamp(0.0, 1.0)
    error = torch.abs(refined_plan - gt_plan)
    if mask is not None:
        mask = mask.unsqueeze(-1).float()
        error = error * mask
    error = error.mean(dim=-1).mean(dim=-1)  # [B]
    return weight * error  # [B]
