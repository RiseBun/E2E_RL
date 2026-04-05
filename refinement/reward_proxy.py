"""奖励代理模块：为 refinement 提供离线可计算的伪 RL 奖励信号。

主要奖励组成:
- progress_reward: 沿 GT 方向的前进奖励
- collision_penalty: 碰撞惩罚（与其他 agent 的距离）
- offroad_penalty: 离道惩罚（与道路边界的距离）
- comfort_penalty: 舒适度惩罚（加速度 / 曲率变化）

设计要求:
- 所有输入使用 ego-centric 坐标
- 输出为 [B] 或 [B, 1] 的标量奖励
- 可直接替换为真实 RL 奖励
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def progress_reward(
    refined_plan: torch.Tensor,
    gt_plan: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算沿 GT 方向的前进奖励。

    轨迹越接近 GT 终点，奖励越高。

    Args:
        refined_plan: [B, T, 2] 精炼后的轨迹（绝对坐标）
        gt_plan: [B, T, 2] GT 轨迹（绝对坐标）
        mask: [B, T] 有效时间步掩码

    Returns:
        [B] 的进度奖励
    """
    # 使用终点距离: 越接近 GT 终点，奖励越高
    if mask is not None:
        # 找到每个 batch 的最后有效时间步
        valid_counts = mask.sum(dim=-1).long().clamp(min=1)  # [B]
        batch_size = refined_plan.shape[0]
        last_idx = valid_counts - 1  # [B]
        batch_range = torch.arange(batch_size, device=refined_plan.device)
        pred_end = refined_plan[batch_range, last_idx]  # [B, 2]
        gt_end = gt_plan[batch_range, last_idx]  # [B, 2]
    else:
        pred_end = refined_plan[:, -1]  # [B, 2]
        gt_end = gt_plan[:, -1]  # [B, 2]

    # FDE 越小，奖励越高
    fde = torch.norm(pred_end - gt_end, dim=-1)  # [B]
    reward = torch.exp(-fde)  # (0, 1]
    return reward


def collision_penalty(
    refined_plan: torch.Tensor,
    agent_positions: Optional[torch.Tensor] = None,
    agent_future_trajs: Optional[torch.Tensor] = None,
    x_thresh: float = 1.5,
    y_thresh: float = 3.0,
) -> torch.Tensor:
    """计算碰撞惩罚。

    与 VAD 的 PlanCollisionLoss 保持一致的距离阈值。

    Args:
        refined_plan: [B, T, 2] 精炼后的轨迹（绝对坐标）
        agent_positions: [B, A, 2] agent 当前位置
        agent_future_trajs: [B, A, T, 2] agent 未来轨迹增量

    Returns:
        [B] 的碰撞惩罚（非负，越大越危险）
    """
    if agent_positions is None or agent_future_trajs is None:
        return torch.zeros(
            refined_plan.shape[0], device=refined_plan.device
        )

    batch_size, time_steps, _ = refined_plan.shape

    # agent 绝对位置: 当前位置 + 未来增量的 cumsum
    agent_abs = agent_positions.unsqueeze(2) + agent_future_trajs.cumsum(dim=2)
    # agent_abs: [B, A, T, 2]

    # ego 轨迹展开: [B, 1, T, 2]
    ego_expanded = refined_plan.unsqueeze(1)

    # 计算每个时间步到每个 agent 的 x/y 距离
    x_dist = torch.abs(ego_expanded[..., 0] - agent_abs[..., 0])  # [B, A, T]
    y_dist = torch.abs(ego_expanded[..., 1] - agent_abs[..., 1])  # [B, A, T]

    # 在阈值内的部分计算惩罚
    x_penalty = torch.clamp(x_thresh - x_dist, min=0.0)
    y_penalty = torch.clamp(y_thresh - y_dist, min=0.0)
    col_score = x_penalty * y_penalty  # [B, A, T]

    # 取所有 agent 和时间步的最大碰撞分数
    penalty = col_score.amax(dim=(1, 2))  # [B]
    return penalty


def offroad_penalty(
    refined_plan: torch.Tensor,
    lane_boundaries: Optional[torch.Tensor] = None,
    dis_thresh: float = 1.0,
) -> torch.Tensor:
    """计算离道惩罚。

    与 VAD 的 PlanMapBoundLoss 保持一致。

    Args:
        refined_plan: [B, T, 2] 精炼后的轨迹（绝对坐标）
        lane_boundaries: [B, N, P, 2] 道路边界线段点

    Returns:
        [B] 的离道惩罚
    """
    if lane_boundaries is None:
        return torch.zeros(
            refined_plan.shape[0], device=refined_plan.device
        )

    batch_size, time_steps, _ = refined_plan.shape
    n_lanes = lane_boundaries.shape[1]
    n_pts = lane_boundaries.shape[2]

    # ego: [B, T, 1, 1, 2], boundary: [B, 1, N, P, 2]
    ego = refined_plan.unsqueeze(2).unsqueeze(3)
    boundary = lane_boundaries.unsqueeze(1)

    # 点到边界点的最近距离
    dist = torch.norm(ego - boundary, dim=-1)  # [B, T, N, P]
    min_dist = dist.amin(dim=(2, 3))  # [B, T]

    # 低于阈值则有惩罚
    offroad = torch.clamp(dis_thresh - min_dist, min=0.0)  # [B, T]
    penalty = offroad.mean(dim=-1)  # [B]
    return penalty


def comfort_penalty(
    refined_plan: torch.Tensor,
    dt: float = 0.5,
) -> torch.Tensor:
    """计算舒适度惩罚（基于加速度和 jerk）。

    Args:
        refined_plan: [B, T, 2] 精炼后的轨迹（绝对坐标）
        dt: 相邻时间步的间隔 (s)

    Returns:
        [B] 的舒适度惩罚
    """
    # 速度: [B, T-1, 2]
    velocity = torch.diff(refined_plan, dim=1) / dt
    # 加速度: [B, T-2, 2]
    acceleration = torch.diff(velocity, dim=1) / dt
    # jerk: [B, T-3, 2]
    jerk = torch.diff(acceleration, dim=1) / dt

    # 加速度幅值惩罚
    acc_mag = torch.norm(acceleration, dim=-1)  # [B, T-2]
    jerk_mag = torch.norm(jerk, dim=-1)  # [B, T-3]

    # 组合惩罚: 加速度 + jerk
    penalty = acc_mag.mean(dim=-1) + 0.5 * jerk_mag.mean(dim=-1)  # [B]
    return penalty


def compute_refinement_reward(
    refined_plan: torch.Tensor,
    gt_plan: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    agent_positions: Optional[torch.Tensor] = None,
    agent_future_trajs: Optional[torch.Tensor] = None,
    lane_boundaries: Optional[torch.Tensor] = None,
    dt: float = 0.5,
    w_progress: float = 1.0,
    w_collision: float = 0.5,
    w_offroad: float = 0.3,
    w_comfort: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """计算综合奖励信号。

    Args:
        refined_plan: [B, T, 2] 精炼后的轨迹（绝对坐标）
        gt_plan: [B, T, 2] GT 轨迹（绝对坐标）
        mask: [B, T] 有效时间步掩码
        agent_positions: [B, A, 2]
        agent_future_trajs: [B, A, T, 2]
        lane_boundaries: [B, N, P, 2]
        dt: 时间步间隔
        w_progress: 进度奖励权重
        w_collision: 碰撞惩罚权重
        w_offroad: 离道惩罚权重
        w_comfort: 舒适度惩罚权重

    Returns:
        字典包含 total_reward 和各分项
    """
    r_progress = progress_reward(refined_plan, gt_plan, mask)
    p_collision = collision_penalty(
        refined_plan, agent_positions, agent_future_trajs
    )
    p_offroad = offroad_penalty(refined_plan, lane_boundaries)
    p_comfort = comfort_penalty(refined_plan, dt)

    total = (
        w_progress * r_progress
        - w_collision * p_collision
        - w_offroad * p_offroad
        - w_comfort * p_comfort
    )

    return {
        'total_reward': total,
        'progress_reward': r_progress,
        'collision_penalty': p_collision,
        'offroad_penalty': p_offroad,
        'comfort_penalty': p_comfort,
    }
