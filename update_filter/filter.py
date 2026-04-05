"""有害更新过滤器：基于可靠性分数生成 mask/weight，提供 filtered loss。

核心 STAPO 启发步骤:
1. 抑制有害更新 (mask 或 soft weight)
2. 对保留的有益更新重归一化损失
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.update_filter.config import HUFConfig


class HarmfulUpdateFilter:
    """有害更新过滤器。

    支持两种模式:
    - hard: 二值 mask，完全排除有害更新
    - soft: 连续权重，平滑抑制有害更新
    """

    def __init__(self, config: HUFConfig) -> None:
        self.cfg = config

    def compute_mask(
        self,
        scores: Dict[str, torch.Tensor],
        interface: PlanningInterface,
        refiner_outputs: Dict[str, torch.Tensor],
        reward: Optional[torch.Tensor] = None,
        ref_reward: Optional[torch.Tensor] = None,
        reward_info: Optional[Dict[str, torch.Tensor]] = None,
        ref_reward_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """计算最终更新掩码。

        STAPO 启发逻辑:
        1. 物理底线 (Hard Guards): 无论 reward 如何，违规即 Mask。
        2. 优势门控 (Advantage Gating): 只有当 delta_safe_reward > margin 时，才启动模型审查。
        3. 模型审查 (Learned Scorer): 拦截 Gain 低或 Risk 高的更新。
        """
        # --- 1. 硬物理底线 (Hard Guards) ---
        residual = refiner_outputs['residual']
        res_norm = torch.norm(residual, dim=-1).max(dim=-1)[0]
        step_disp = torch.norm(residual[:, 1:] - residual[:, :-1], dim=-1).max(dim=-1)[0]
        
        # 如果提供了 reward 详情，可以检查碰撞增加
        # 这里先检查基本的运动学指标
        hard_invalid = (res_norm > self.cfg.max_residual_norm) | (step_disp > self.cfg.max_step_disp)

        # --- 2. 优势门控 (Advantage Gating) ---
        if reward is not None and ref_reward is not None:
            # 更新为安全加权门控，避免仅用 total reward 导致“投机”更新。
            if reward_info is not None and ref_reward_info is not None:
                delta_progress = (reward_info.get('progress_reward', reward) - ref_reward_info.get('progress_reward', ref_reward))
                delta_collision = (reward_info.get('collision_penalty', torch.zeros_like(reward)) - ref_reward_info.get('collision_penalty', torch.zeros_like(ref_reward))).clamp(min=0)
                delta_offroad = (reward_info.get('offroad_penalty', torch.zeros_like(reward)) - ref_reward_info.get('offroad_penalty', torch.zeros_like(ref_reward))).clamp(min=0)
                delta_comfort = (reward_info.get('comfort_penalty', torch.zeros_like(reward)) - ref_reward_info.get('comfort_penalty', torch.zeros_like(ref_reward))).clamp(min=0)
                # drift 近似用 comfort 变化作为 proxy
                delta_drift = delta_comfort

                delta_safe_reward = (
                    delta_progress
                    - self.cfg.lambda_collision * delta_collision
                    - self.cfg.lambda_offroad * delta_offroad
                    - self.cfg.lambda_comfort * delta_comfort
                    - self.cfg.lambda_drift * delta_drift
                )
                is_positive_adv = delta_safe_reward > self.cfg.delta_safe_margin
            else:
                delta_reward = reward - ref_reward
                is_positive_adv = delta_reward > self.cfg.delta_margin
        else:
            is_positive_adv = torch.ones_like(res_norm, dtype=torch.bool)

        # --- 3. 模型/规则联合审查 (Learned Scorer) ---
        pred_gain = scores.get('pred_gain', torch.zeros_like(res_norm))
        pred_risk = scores.get('pred_risk', torch.zeros_like(res_norm))

        pred_risk_collision = scores.get('pred_risk_collision', torch.zeros_like(res_norm))
        pred_risk_offroad = scores.get('pred_risk_offroad', torch.zeros_like(res_norm))
        pred_risk_comfort = scores.get('pred_risk_comfort', torch.zeros_like(res_norm))
        pred_risk_drift = scores.get('pred_risk_drift', torch.zeros_like(res_norm))

        is_harmful_by_risk = (
            (pred_risk > self.cfg.tau_risk) |
            (pred_risk_collision > self.cfg.tau_risk) |
            (pred_risk_offroad > self.cfg.tau_risk) |
            (pred_risk_comfort > self.cfg.tau_risk) |
            (pred_risk_drift > self.cfg.tau_risk)
        )

        is_harmful = (pred_gain < self.cfg.tau_gain) | is_harmful_by_risk

        # --- 4. STAPO (规划域): 正优势 ∧ 低 π_sel ∧ 低熵 → 视为虚假有益更新，予以静音 ---
        is_stapo_spurious = torch.zeros_like(res_norm, dtype=torch.bool)
        if self.cfg.stapo_enabled:
            stapo_pi = scores.get('stapo_pi')
            stapo_h = scores.get('stapo_entropy_norm')
            if stapo_pi is not None and stapo_h is not None:
                is_stapo_spurious = (
                    is_positive_adv
                    & (stapo_pi < self.cfg.stapo_tau_pi)
                    & (stapo_h < self.cfg.stapo_tau_entropy)
                )

        mask = ~(hard_invalid | (is_positive_adv & is_harmful) | is_stapo_spurious)

        uncertainty = scores.get('uncertainty_score', torch.zeros_like(res_norm))
        support = scores.get('support_score', torch.ones_like(res_norm))
        drift = scores.get('drift_score', torch.zeros_like(res_norm))

        batch_size = uncertainty.shape[0]
        retention = mask.float().mean()
        if retention < self.cfg.min_retention_ratio:
            composite = (
                self.cfg.w_uncertainty_final * uncertainty
                + self.cfg.w_support_final * (1.0 - support)
                + self.cfg.w_drift_final * drift
            )
            eligible = ~hard_invalid
            composite = composite.masked_fill(~eligible, float('inf'))
            eligible_count = int(eligible.sum().item())
            if eligible_count > 0:
                k = math.ceil(batch_size * self.cfg.min_retention_ratio)
                k = min(k, eligible_count)
                _, topk_idx = composite.topk(k, largest=False)
                mask = torch.zeros_like(mask)
                mask[topk_idx] = True

        return mask

    def compute_weight(self, scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算 soft weight [B] (float)，范围 (0, 1]。

        每个维度用 sigmoid 门控，三个门控相乘。
        """
        uncertainty = scores['uncertainty_score']
        support = scores['support_score']
        drift = scores['drift_score']
        temp = self.cfg.soft_temperature

        w_u = torch.sigmoid(-(uncertainty - self.cfg.tau_uncertainty) / temp)
        w_s = torch.sigmoid((support - self.cfg.tau_support) / temp)
        w_d = torch.sigmoid(-(drift - self.cfg.tau_drift) / temp)

        weight = w_u * w_s * w_d
        return weight.clamp(min=1e-6)

    def apply_filter(
        self,
        per_sample_loss: torch.Tensor,
        scores: Dict[str, torch.Tensor],
        interface: PlanningInterface,
        refiner_outputs: Dict[str, torch.Tensor],
        reward: Optional[torch.Tensor] = None,
        ref_reward: Optional[torch.Tensor] = None,
        reward_info: Optional[Dict[str, torch.Tensor]] = None,
        ref_reward_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """应用过滤并返回重归一化后的损失。

        Args:
            per_sample_loss: [B] 每个样本的损失值
            scores: scorer 输出的各维度评分
            interface: 当前 PlanningInterface
            refiner_outputs: Refiner 输出
            reward: [B] 当前样本修正后的 reward
            ref_reward: [B] 当前样本原始参考轨迹的 reward
            reward_info / ref_reward_info: 分项奖励（用于安全优势与 STAPO 门控）

        Returns:
            (filtered_loss, diagnostics):
            - filtered_loss: 标量损失
            - diagnostics: 诊断信息 dict
        """
        if not self.cfg.enabled:
            return per_sample_loss.mean(), {'retention_ratio': 1.0, 'huf_enabled': False}

        if self.cfg.mode == 'hard':
            mask = self.compute_mask(
                scores, interface, refiner_outputs,
                reward=reward, ref_reward=ref_reward,
                reward_info=reward_info, ref_reward_info=ref_reward_info,
            )
            mask_float = mask.float()
            n_kept = mask_float.sum().clamp(min=1.0)
            filtered_loss = (mask_float * per_sample_loss).sum() / n_kept
            diagnostics = self._get_diagnostics(scores, mask)
        else:
            weight = self.compute_weight(scores)
            filtered_loss = (weight * per_sample_loss).sum() / weight.sum().clamp(min=1e-6)
            diagnostics = self._get_diagnostics(scores, weight)

        return filtered_loss, diagnostics

    def _get_diagnostics(
        self,
        scores: Dict[str, torch.Tensor],
        mask_or_weight: torch.Tensor,
    ) -> Dict[str, Any]:
        """生成诊断信息。"""
        uncertainty = scores['uncertainty_score']
        support = scores['support_score']
        drift = scores['drift_score']
        batch_size = uncertainty.shape[0]

        if mask_or_weight.dtype == torch.bool:
            # Hard mask 模式
            kept = mask_or_weight
            filtered = ~mask_or_weight
            retention_ratio = kept.float().mean().item()

            diag: Dict[str, Any] = {
                'huf_enabled': True,
                'huf_mode': 'hard',
                'retention_ratio': retention_ratio,
                'n_kept': int(kept.sum().item()),
                'n_filtered': int(filtered.sum().item()),
            }

            if kept.any():
                diag['mean_uncertainty_kept'] = uncertainty[kept].mean().item()
                diag['mean_support_kept'] = support[kept].mean().item()
                diag['mean_drift_kept'] = drift[kept].mean().item()
            if filtered.any():
                diag['mean_uncertainty_filtered'] = uncertainty[filtered].mean().item()
                diag['mean_support_filtered'] = support[filtered].mean().item()
                diag['mean_drift_filtered'] = drift[filtered].mean().item()

            # 分项统计: 哪个条件导致了过滤
            diag['filter_by_uncertainty'] = int(
                (uncertainty >= self.cfg.tau_uncertainty).sum().item()
            )
            diag['filter_by_support'] = int(
                (support <= self.cfg.tau_support).sum().item()
            )
            diag['filter_by_drift'] = int(
                (drift >= self.cfg.tau_drift).sum().item()
            )

        else:
            # Soft weight 模式
            weight = mask_or_weight
            diag = {
                'huf_enabled': True,
                'huf_mode': 'soft',
                'retention_ratio': weight.mean().item(),
                'mean_weight': weight.mean().item(),
                'min_weight': weight.min().item(),
                'max_weight': weight.max().item(),
                'mean_uncertainty': uncertainty.mean().item(),
                'mean_support': support.mean().item(),
                'mean_drift': drift.mean().item(),
            }

        return diag
