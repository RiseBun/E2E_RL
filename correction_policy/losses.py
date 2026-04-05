"""
Correction Policy Loss Functions

包含三种损失函数：
1. behavioral_cloning_loss    - BC 预热：最大化 GT correction 的概率
2. policy_gradient_loss       - 基础 PG：-A * log π(a|s)
3. compute_advantage          - Advantage 计算：safe_reward(corrected) - safe_reward(ref)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.correction_policy.policy import CorrectionPolicy


def behavioral_cloning_loss(
    policy: CorrectionPolicy,
    interface: PlanningInterface,
    gt_correction: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Behavioral Cloning 损失函数。

    Stage 1 使用：最大化 policy 在 GT correction 上的对数概率。
    Loss = -E[log π(gt_correction | state)]

    直观理解：让 policy 学会在给定场景下输出接近 GT 的修正。

    Args:
        policy: CorrectionPolicy
        interface: PlanningInterface 输入
        gt_correction: [B, T, 2] GT 修正量 = gt_plan - reference_plan
        mask: [B, T] 可选掩码，对无效时间步不计算损失

    Returns:
        标量损失
    """
    eval_result = policy.evaluate(interface, gt_correction)
    log_prob = eval_result['log_prob']  # [B]

    if mask is not None:
        # 对每个样本，按有效时间步数归一化
        valid_counts = mask.sum(dim=-1).float().clamp(min=1.0)  # [B]
        loss = -(log_prob / valid_counts).mean()
    else:
        loss = -log_prob.mean()

    return loss


def policy_gradient_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample Policy Gradient 损失。

    Loss_i = -A_i * log π(a_i | s_i)

    注意：advantage 使用 .detach() 断开梯度，只作为加权因子。

    Args:
        log_probs: [B] 每个样本的对数概率
        advantages: [B] 每个样本的 advantage
        mask: [B] 可选掩码

    Returns:
        [B] per-sample PG loss（后续会被 STAPO gate 过滤和重归一化）
    """
    # advantage 断开梯度，只作为权重
    advantage_weight = advantages.detach()

    per_sample_loss = -advantage_weight * log_probs  # [B]

    if mask is not None:
        per_sample_loss = per_sample_loss * mask.float()

    return per_sample_loss


def compute_advantage(
    corrected_plan: torch.Tensor,
    reference_plan: torch.Tensor,
    gt_plan: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reward_config: Optional[Dict] = None,
) -> torch.Tensor:
    """计算 Advantage。

    A = safe_reward(corrected) - safe_reward(reference)

    即修正后轨迹相对于参考轨迹的 safe reward 增益。

    Args:
        corrected_plan: [B, T, 2] 修正后的轨迹
        reference_plan: [B, T, 2] 参考轨迹
        gt_plan: [B, T, 2] GT 轨迹
        mask: [B, T] 可选掩码
        reward_config: 奖励计算的参数字典

    Returns:
        [B] advantage
    """
    from E2E_RL.refinement.reward_proxy import compute_refinement_reward

    reward_config = reward_config or {}

    with torch.no_grad():
        # 修正后轨迹的 reward
        r_corrected = compute_refinement_reward(
            refined_plan=corrected_plan,
            gt_plan=gt_plan,
            mask=mask,
            **reward_config,
        )
        r_corrected_total = r_corrected['total_reward']

        # 参考轨迹的 reward（作为 baseline）
        r_reference = compute_refinement_reward(
            refined_plan=reference_plan,
            gt_plan=gt_plan,
            mask=mask,
            **reward_config,
        )
        r_reference_total = r_reference['total_reward']

    advantage = r_corrected_total - r_reference_total  # [B]

    return advantage


def ppo_clipped_surrogate_loss(
    policy: CorrectionPolicy,
    interface: PlanningInterface,
    sampled_correction: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """PPO Clipped Surrogate Loss。

    L = -min(A * r, A * clip(r, 1-ε, 1+ε))

    其中 r = π_new(a|s) / π_old(a|s)

    用于防止 policy 更新过大。

    Args:
        policy: CorrectionPolicy
        interface: PlanningInterface
        sampled_correction: [B, T, 2] 本次采样的修正
        old_log_probs: [B] 上次策略的对数概率（来自 buffer）
        advantages: [B] advantage
        clip_epsilon: PPO clip 范围

    Returns:
        标量损失
    """
    # 计算新的对数概率
    eval_result = policy.evaluate(interface, sampled_correction)
    new_log_probs = eval_result['log_prob']  # [B]

    # 概率比
    ratio = torch.exp(new_log_probs - old_log_probs.detach())

    # Clipped surrogate
    advantage = advantages.detach()
    surr1 = advantage * ratio
    surr2 = advantage * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # 取较小值
    loss = -torch.min(surr1, surr2).mean()

    return loss


def entropy_bonus_loss(entropies: torch.Tensor) -> torch.Tensor:
    """Entropy 正则化损失。

    L_entropy = -H(π)

    鼓励策略保持探索，防止熵崩溃到 0。

    Args:
        entropies: [B] 每个样本的熵

    Returns:
        标量损失（负值鼓励增加熵）
    """
    return -entropies.mean()


def combined_policy_loss(
    policy: CorrectionPolicy,
    interface: PlanningInterface,
    gt_correction: torch.Tensor,
    advantages: torch.Tensor,
    entropy_coef: float = 0.01,
    use_bc: bool = False,
    bc_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """组合策略损失（BC + PG + Entropy）。

    用于 Stage 2 训练，支持：
    1. 纯 PG：L = -A * log π
    2. BC + PG 混合：L = bc_weight * L_bc + (1 - bc_weight) * L_pg - entropy_coef * H

    Args:
        policy: CorrectionPolicy
        interface: PlanningInterface
        gt_correction: [B, T, 2] GT 修正（用于 BC）
        advantages: [B] advantage
        entropy_coef: 熵正则化系数
        use_bc: 是否使用 BC 混合
        bc_weight: BC 权重（当 use_bc=True 时生效）

    Returns:
        dict with:
            - total_loss: 组合损失
            - bc_loss: BC 损失（如果 use_bc=True）
            - pg_loss: PG 损失
            - entropy_loss: 熵损失
            - log_prob: 当前 log_prob
            - entropy: 当前 entropy
    """
    # 采样（用于 PG）
    sample_result = policy.sample(interface)
    correction = sample_result['correction']
    log_prob = sample_result['log_prob']
    entropy = sample_result['entropy']

    # BC 损失
    if use_bc:
        bc_loss = behavioral_cloning_loss(policy, interface, gt_correction)
    else:
        bc_loss = torch.tensor(0.0, device=interface.scene_token.device)

    # PG 损失
    pg_loss = -(advantages.detach() * log_prob).mean()

    # Entropy 损失
    entropy_loss = -entropy.mean() * entropy_coef

    # 组合
    if use_bc:
        total_loss = bc_weight * bc_loss + (1 - bc_weight) * pg_loss + entropy_loss
    else:
        total_loss = pg_loss + entropy_loss

    return {
        'total_loss': total_loss,
        'bc_loss': bc_loss,
        'pg_loss': pg_loss,
        'entropy_loss': entropy_loss,
        'log_prob': log_prob,
        'entropy': entropy,
    }
