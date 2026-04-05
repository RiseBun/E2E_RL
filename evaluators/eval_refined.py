"""精炼轨迹评估模块：对比 baseline 与 refined 的规划指标。

支持指标:
- Planning L2 (逐步欧氏距离)
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- Collision rate proxy
- Comfort proxy
- Hard-case 子集性能
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _compute_planning_l2(
    plan: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """逐步 L2 距离均值。

    Args:
        plan: [B, T, 2]
        gt: [B, T, 2]
        mask: [B, T]

    Returns:
        [B] 每个样本的平均 L2
    """
    dist = torch.norm(plan - gt, dim=-1)  # [B, T]
    if mask is not None:
        dist = dist * mask.float()
        return dist.sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1.0)
    return dist.mean(dim=-1)


def _compute_ade(
    plan: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average Displacement Error。"""
    return _compute_planning_l2(plan, gt, mask)


def _compute_fde(
    plan: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Final Displacement Error。

    Returns:
        [B] 每个样本的 FDE
    """
    if mask is not None:
        valid_counts = mask.float().sum(dim=-1).long().clamp(min=1)
        batch_range = torch.arange(plan.shape[0], device=plan.device)
        last_idx = valid_counts - 1
        pred_end = plan[batch_range, last_idx]
        gt_end = gt[batch_range, last_idx]
    else:
        pred_end = plan[:, -1]
        gt_end = gt[:, -1]
    return torch.norm(pred_end - gt_end, dim=-1)


def _compute_collision_rate(
    plan: torch.Tensor,
    agent_positions: Optional[torch.Tensor] = None,
    agent_future_trajs: Optional[torch.Tensor] = None,
    x_thresh: float = 1.5,
    y_thresh: float = 3.0,
) -> torch.Tensor:
    """碰撞率代理。

    Returns:
        [B] 每个样本是否碰撞 (0 或 1)
    """
    if agent_positions is None or agent_future_trajs is None:
        return torch.zeros(plan.shape[0], device=plan.device)

    # agent 绝对位置
    agent_abs = agent_positions.unsqueeze(2) + agent_future_trajs.cumsum(dim=2)
    ego = plan.unsqueeze(1)  # [B, 1, T, 2]

    x_dist = torch.abs(ego[..., 0] - agent_abs[..., 0])
    y_dist = torch.abs(ego[..., 1] - agent_abs[..., 1])

    collision = (x_dist < x_thresh) & (y_dist < y_thresh)  # [B, A, T]
    has_collision = collision.any(dim=-1).any(dim=-1).float()  # [B]
    return has_collision


def _compute_comfort(
    plan: torch.Tensor,
    dt: float = 0.5,
) -> torch.Tensor:
    """舒适度指标（加速度幅值均值）。

    Returns:
        [B] 加速度幅值
    """
    velocity = torch.diff(plan, dim=1) / dt
    acceleration = torch.diff(velocity, dim=1) / dt
    acc_mag = torch.norm(acceleration, dim=-1).mean(dim=-1)
    return acc_mag


def evaluate_refined_plans(
    baseline_plan: torch.Tensor,
    refined_plan: torch.Tensor,
    gt_plan: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    agent_positions: Optional[torch.Tensor] = None,
    agent_future_trajs: Optional[torch.Tensor] = None,
    dt: float = 0.5,
    hard_case_indices: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """对比 baseline 和 refined 轨迹的全面评估。

    Args:
        baseline_plan: [B, T, 2] 原始规划器输出（绝对坐标）
        refined_plan: [B, T, 2] 精炼后轨迹（绝对坐标）
        gt_plan: [B, T, 2] GT 轨迹（绝对坐标）
        mask: [B, T] 有效步掩码
        agent_positions: [B, A, 2]
        agent_future_trajs: [B, A, T, 2] 增量
        dt: 时间步间隔
        hard_case_indices: 困难样本索引

    Returns:
        包含 baseline_* 和 refined_* 前缀的指标字典
    """
    results: Dict[str, float] = {}

    for prefix, plan in [('baseline', baseline_plan), ('refined', refined_plan)]:
        ade = _compute_ade(plan, gt_plan, mask)
        fde = _compute_fde(plan, gt_plan, mask)
        l2 = _compute_planning_l2(plan, gt_plan, mask)
        col = _compute_collision_rate(
            plan, agent_positions, agent_future_trajs
        )
        comfort = _compute_comfort(plan, dt)

        results[f'{prefix}_ade'] = ade.mean().item()
        results[f'{prefix}_fde'] = fde.mean().item()
        results[f'{prefix}_l2'] = l2.mean().item()
        results[f'{prefix}_collision_rate'] = col.mean().item()
        results[f'{prefix}_comfort'] = comfort.mean().item()

        # hard-case 子集
        if hard_case_indices is not None and hard_case_indices.numel() > 0:
            hc = hard_case_indices
            results[f'{prefix}_hard_ade'] = ade[hc].mean().item()
            results[f'{prefix}_hard_fde'] = fde[hc].mean().item()
            results[f'{prefix}_hard_l2'] = l2[hc].mean().item()

    # 改进幅度
    for metric in ('ade', 'fde', 'l2', 'collision_rate'):
        base_val = results[f'baseline_{metric}']
        ref_val = results[f'refined_{metric}']
        if base_val > 0:
            improvement = (base_val - ref_val) / base_val * 100
            results[f'improvement_{metric}_pct'] = improvement

    logger.info('=== 精炼评估结果 ===')
    for key, val in results.items():
        logger.info(f'  {key}: {val:.4f}')

    return results
