"""更新可靠性评分器：计算 uncertainty / support / drift 三类分数。

所有评分在 [0, 1] 范围内，计算过程 detach 不参与梯度回传。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.refinement.reward_proxy import comfort_penalty
from E2E_RL.update_filter.config import HUFConfig
from E2E_RL.update_filter.model import ReliabilityNet


class UpdateReliabilityScorer:
    """基于规则或模型的更新可靠性评分器。

    消费 PlanningInterface + refiner 输出，输出可靠性评分。
    """

    def __init__(
        self,
        config: HUFConfig,
        model: Optional[ReliabilityNet] = None,
        model_path: Optional[str] = None,
        model_plan_len: Optional[int] = None,  # Scorer 期望的 plan_len（用于下采样）
    ) -> None:
        self.cfg = config
        self.model = model
        self.model_plan_len = model_plan_len
        if model_path is not None and self.model is not None:
            ckpt = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(ckpt)
            self.model.eval()
            print(f"Loaded ReliabilityNet from {model_path}")

    @torch.no_grad()
    def score_batch(
        self,
        interface: PlanningInterface,
        refiner_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """计算 batch 内每个样本的可靠性分数。"""
        # 1. 基础启发式分数 (Uncertainty, Support, Drift)
        uncertainty = self._compute_uncertainty(interface, refiner_outputs)
        support = self._compute_support(interface, refiner_outputs)
        drift = self._compute_drift(interface, refiner_outputs)

        # 2. STAPO 驾驶域代理：选中模态概率 π_sel、模态分布归一化熵 H_norm
        stapo_pi, stapo_entropy_norm = self._compute_stapo_signals(interface)

        # 3. 模型预测分数 (Learned Scorer - 双头回归)
        h_scores = torch.stack([uncertainty, support, drift], dim=-1)
        if self.model is not None:
            # 处理 plan_len 不匹配的情况（下采样）
            ref_plan = interface.reference_plan
            residual = refiner_outputs['residual']
            
            if self.model_plan_len is not None and ref_plan.shape[1] != self.model_plan_len:
                # 下采样到 Scorer 期望的 plan_len
                src_len = ref_plan.shape[1]
                indices = torch.linspace(0, src_len - 1, self.model_plan_len).long()
                ref_plan = ref_plan[:, indices, :]
                residual = residual[:, indices, :]
            
            p_gain, p_risk_total, p_risk_components = self.model(
                interface.scene_token,
                ref_plan,
                residual,
                plan_confidence=interface.plan_confidence,
                safety_features=None,
                heuristic_scores=h_scores,
            )
            pred_gain = p_gain.squeeze(-1)
            pred_risk = p_risk_total.squeeze(-1)
            pred_risk_collision = p_risk_components[:, 0]
            pred_risk_offroad = p_risk_components[:, 1]
            pred_risk_comfort = p_risk_components[:, 2]
            pred_risk_drift = p_risk_components[:, 3]
        else:
            pred_gain = torch.full_like(uncertainty, 1e6)
            pred_risk = torch.zeros_like(uncertainty)
            pred_risk_collision = torch.zeros_like(uncertainty)
            pred_risk_offroad = torch.zeros_like(uncertainty)
            pred_risk_comfort = torch.zeros_like(uncertainty)
            pred_risk_drift = torch.zeros_like(uncertainty)
        return {
            'uncertainty_score': uncertainty,
            'support_score': support,
            'drift_score': drift,
            'stapo_pi': stapo_pi,
            'stapo_entropy_norm': stapo_entropy_norm,
            'pred_gain': pred_gain,
            'pred_risk': pred_risk,
            'pred_risk_collision': pred_risk_collision,
            'pred_risk_offroad': pred_risk_offroad,
            'pred_risk_comfort': pred_risk_comfort,
            'pred_risk_drift': pred_risk_drift,
        }

    def _compute_stapo_signals(
        self,
        interface: PlanningInterface,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """STAPO 在规划上的类比：多模态 「token」= 离散规划模态。

        - π_sel: 当前 reference 对应模态在隐式分布下的概率（高 = 主模式与基座一致）
        - H_norm: 模态分布熵 / log(M) ∈ [0,1]（低 = 过尖 / 过度确信某一模态）

        若 M<2 或无 candidate，则返回 (1, 1)，不参与虚假更新判定。
        若 metadata['ego_mode_logits'] 存在 [B,M]，优先用作策略 logits。
        """
        device = interface.scene_token.device
        b = interface.scene_token.shape[0]
        ones = torch.ones(b, device=device)
        temp = max(self.cfg.stapo_softmax_temp, 1e-6)

        meta: Dict[str, Any] = interface.metadata or {}
        lm = meta.get('ego_mode_logits')
        if lm is not None and isinstance(lm, torch.Tensor):
            lm = lm.to(device)
            while lm.dim() > 2:
                lm = lm.squeeze(0)
            if lm.dim() == 1:
                lm = lm.unsqueeze(0).expand(b, -1)
            m = lm.shape[-1]
            if m < 2:
                return ones, ones
            p = torch.softmax(lm / temp, dim=-1)
            sel = self._resolve_executed_mode_index(meta, lm.argmax(dim=-1), b, device)
            pi_sel = p[torch.arange(b, device=device), sel]
            ent = -(p * (p + 1e-8).log()).sum(dim=-1)
            h_norm = (ent / math.log(m)).clamp(0.0, 1.0)
            return pi_sel.clamp(0.0, 1.0), h_norm

        cand = interface.candidate_plans
        if cand is None or cand.dim() != 4 or cand.shape[1] < 2:
            return ones, ones

        abs_plans = cand.cumsum(dim=2)
        ref = interface.reference_plan.unsqueeze(1)
        dist = (abs_plans - ref).norm(dim=-1).mean(dim=-1)
        logits = -dist / temp
        p = torch.softmax(logits, dim=-1)
        sel = self._resolve_executed_mode_index(meta, dist.argmin(dim=-1), b, device)
        pi_sel = p[torch.arange(b, device=device), sel]
        ent = -(p * (p + 1e-8).log()).sum(dim=-1)
        h_norm = (ent / math.log(cand.shape[1])).clamp(0.0, 1.0)
        return pi_sel.clamp(0.0, 1.0), h_norm

    @staticmethod
    def _resolve_executed_mode_index(
        meta: Dict[str, Any],
        default_sel: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """实际执行/监督的模态索引 [B]；来自 dataset 或规划器命令，而非仅几何最近。"""
        em = meta.get('executed_mode_index')
        if em is None:
            return default_sel
        if isinstance(em, int):
            return torch.full((batch_size,), em, device=device, dtype=torch.long)
        if isinstance(em, torch.Tensor):
            t = em.to(device).long().view(-1)
            if t.numel() == 1:
                return t.expand(batch_size)
            return t[:batch_size]
        return default_sel

    def _compute_uncertainty(
        self,
        interface: PlanningInterface,
        refiner_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """计算 uncertainty score [B]，范围 [0, 1]。

        信号来源:
        1. plan_confidence 反转 (高 conf → 低 uncertainty)
        2. candidate mode 方差 (高方差 → 高 uncertainty)
        3. residual norm (大残差 → 高 uncertainty)
        """
        device = interface.scene_token.device
        batch_size = interface.scene_token.shape[0]

        scores = []
        weights = []

        # 1. Confidence 反转
        if interface.plan_confidence is not None:
            conf = interface.plan_confidence
            if conf.dim() == 2:
                conf = conf.squeeze(-1)
            u_conf = (1.0 - conf).clamp(0.0, 1.0)
            scores.append(u_conf)
            weights.append(self.cfg.w_confidence)

        # 2. Candidate mode 方差
        if interface.candidate_plans is not None and interface.candidate_plans.shape[1] > 1:
            candidates = interface.candidate_plans  # [B, M, T, 2]
            # cumsum 将增量转为绝对坐标（如果 candidate 是增量）
            abs_plans = candidates.cumsum(dim=2)
            mode_var = abs_plans.var(dim=1).mean(dim=(-2, -1))  # [B]
            u_mode = (1.0 - torch.exp(-mode_var)).clamp(0.0, 1.0)
            scores.append(u_mode)
            weights.append(self.cfg.w_mode_variance)

        # 3. Residual 不确定性
        residual_norm = refiner_outputs.get('residual_norm')
        if residual_norm is not None:
            u_residual = (1.0 - torch.exp(-residual_norm)).clamp(0.0, 1.0)
            scores.append(u_residual)
            weights.append(self.cfg.w_residual_var)

        if not scores:
            return torch.zeros(batch_size, device=device)

        # 归一化权重并加权求和
        total_w = sum(weights)
        uncertainty = torch.zeros(batch_size, device=device)
        for s, w in zip(scores, weights):
            uncertainty = uncertainty + (w / total_w) * s

        return uncertainty.clamp(0.0, 1.0)

    def _compute_support(
        self,
        interface: PlanningInterface,
        refiner_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """计算 support score [B]，范围 [0, 1]。

        信号来源:
        1. residual norm 衰减 (小残差 → 高支持)
        2. 单步最大位移 (无突变 → 高支持)
        """
        device = interface.scene_token.device
        batch_size = interface.scene_token.shape[0]
        alpha = self.cfg.support_alpha

        residual_norm = refiner_outputs.get('residual_norm')
        residual = refiner_outputs.get('residual')

        if residual_norm is None:
            return torch.ones(batch_size, device=device)

        # 1. 整体残差大小衰减
        s_norm = torch.exp(-alpha * residual_norm)  # (0, 1]

        # 2. 单步最大位移衰减
        if residual is not None:
            max_disp = residual.norm(dim=-1).max(dim=-1).values  # [B]
            s_disp = torch.exp(-alpha * max_disp)
        else:
            s_disp = s_norm

        support = 0.6 * s_norm + 0.4 * s_disp

        # 硬上限: residual_norm 超过阈值直接归零
        over_limit = residual_norm > self.cfg.max_residual_norm
        support = support.masked_fill(over_limit, 0.0)

        return support.clamp(0.0, 1.0)

    def _compute_drift(
        self,
        interface: PlanningInterface,
        refiner_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """计算 drift score [B]，范围 [0, 1]。

        信号来源:
        1. comfort 恶化程度 (精炼后 comfort penalty 增大)
        2. curvature 突变 (相邻速度方向变化过大)
        3. residual 大小占比
        """
        device = interface.scene_token.device
        batch_size = interface.scene_token.shape[0]

        refined_plan = refiner_outputs.get('refined_plan')
        reference_plan = interface.reference_plan
        residual_norm = refiner_outputs.get('residual_norm')

        if refined_plan is None:
            return torch.zeros(batch_size, device=device)

        # 1. Comfort 恶化
        comfort_ref = comfort_penalty(reference_plan, dt=0.5)
        comfort_refined = comfort_penalty(refined_plan.detach(), dt=0.5)
        d_comfort = (comfort_refined - comfort_ref).clamp(min=0.0)
        d_comfort_norm = (1.0 - torch.exp(-d_comfort)).clamp(0.0, 1.0)

        # 2. Curvature 突变
        d_curvature = self._compute_curvature_score(refined_plan.detach())

        # 3. Residual 大小占比
        if residual_norm is not None:
            d_mag = (residual_norm / self.cfg.max_residual_norm).clamp(0.0, 1.0)
        else:
            d_mag = torch.zeros(batch_size, device=device)

        drift = (
            self.cfg.w_comfort * d_comfort_norm
            + self.cfg.w_curvature * d_curvature
            + self.cfg.w_residual_mag * d_mag
        )

        return drift.clamp(0.0, 1.0)

    @staticmethod
    def _compute_curvature_score(plan: torch.Tensor, dt: float = 0.5) -> torch.Tensor:
        """计算轨迹的最大 curvature 突变分数 [B]，范围 [0, 1]。

        Args:
            plan: [B, T, 2] 轨迹
            dt: 时间步间隔

        Returns:
            [B] curvature 分数
        """
        if plan.shape[1] < 3:
            return torch.zeros(plan.shape[0], device=plan.device)

        # 速度向量 [B, T-1, 2]
        velocity = torch.diff(plan, dim=1) / dt

        # 相邻速度向量的夹角
        v1 = velocity[:, :-1]  # [B, T-2, 2]
        v2 = velocity[:, 1:]   # [B, T-2, 2]

        # cross product (标量) 和 dot product
        cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]  # [B, T-2]
        dot = (v1 * v2).sum(dim=-1)  # [B, T-2]

        # atan2 计算转角，加 eps 防止数值问题
        angles = torch.atan2(cross, dot + 1e-8)  # [B, T-2]

        # 取最大绝对转角，归一化到 [0, 1]
        max_angle = angles.abs().max(dim=-1).values  # [B]
        curvature_score = (max_angle / math.pi).clamp(0.0, 1.0)

        return curvature_score
