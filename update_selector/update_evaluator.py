"""
UpdateEvaluator — Learned Update Selector / Update Evaluator

核心思想：
不是"筛样本分类器"，而是判断"在这个 state 下，这个 correction action 作为一次 RL update，
值不值得被强化"。

输入：(s, a, ref, corrected)
输出：pred_gain, pred_risk（多头回归）

与旧 ReliabilityNet 的区别：
- 旧 ReliabilityNet：判断"修正是否值得应用"（后验裁决）
- 新 UpdateEvaluator：判断"这个 update 是否值得被 RL 强化"（服务于 policy learning）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from E2E_RL.planning_interface.interface import PlanningInterface


@dataclass
class UpdateEvaluatorConfig:
    """UpdateEvaluator 配置。"""
    scene_dim: int = 256
    plan_len: int = 12
    hidden_dim: int = 256
    dropout: float = 0.1

    # 风险分项权重（用于组合 pred_risk）
    lambda_collision: float = 2.0
    lambda_offroad: float = 1.0
    lambda_comfort: float = 0.5
    lambda_drift: float = 1.0

    # Risk 归一化因子（训练前估计）
    collision_norm: float = 1.0
    offroad_norm: float = 1.0
    comfort_norm: float = 1.0
    drift_norm: float = 1.0


class UpdateEvaluator(nn.Module):
    """Learnable Update Evaluator。

    多头回归架构：
    输入: (scene_token, reference_plan, correction, structured_stats)
    输出:
        - pred_gain: 预测的 ΔR
        - pred_collision: 预测的碰撞风险增量
        - pred_offroad: 预测的离道风险增量
        - pred_comfort: 预测的舒适度恶化
        - pred_drift: 预测的漂移

    最终组合:
        pred_risk = λ1 * pred_collision + λ2 * pred_offroad + λ3 * pred_comfort + λ4 * pred_drift
    """

    def __init__(self, config: UpdateEvaluatorConfig):
        super().__init__()
        self.cfg = config
        self.plan_len = config.plan_len
        self.action_dim = config.plan_len * 2

        # ---- 特征编码器 ----
        # Scene encoder
        self.scene_encoder = nn.Sequential(
            nn.Linear(config.scene_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        # Reference plan encoder (输入维度为 T*2)
        self.ref_encoder = nn.Sequential(
            nn.Linear(self.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        # Correction encoder (输入维度为 T*2)
        self.corr_encoder = nn.Sequential(
            nn.Linear(self.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        # Structured stats encoder（可选特征）
        self.stats_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dim // 4),  # 8 个统计量
            nn.ReLU(inplace=True),
        )

        # Confidence encoder
        self.conf_encoder = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 8),
            nn.ReLU(inplace=True),
        )

        # ---- 共享融合层 ----
        fusion_dim = config.hidden_dim * 3 + config.hidden_dim // 4 + config.hidden_dim // 8
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ---- 多头输出 ----
        # Gain head
        self.gain_head = nn.Linear(config.hidden_dim, 1)

        # Risk 分项 heads
        self.collision_head = nn.Linear(config.hidden_dim, 1)
        self.offroad_head = nn.Linear(config.hidden_dim, 1)
        self.comfort_head = nn.Linear(config.hidden_dim, 1)
        self.drift_head = nn.Linear(config.hidden_dim, 1)

        # ---- Risk 组合权重（可学习） ----
        # 初始为配置中的值
        self.register_buffer(
            'lambda_collision', torch.tensor(config.lambda_collision))
        self.register_buffer(
            'lambda_offroad', torch.tensor(config.lambda_offroad))
        self.register_buffer(
            'lambda_comfort', torch.tensor(config.lambda_comfort))
        self.register_buffer(
            'lambda_drift', torch.tensor(config.lambda_drift))

        # 归一化因子
        self.register_buffer(
            'collision_norm', torch.tensor(config.collision_norm))
        self.register_buffer(
            'offroad_norm', torch.tensor(config.offroad_norm))
        self.register_buffer(
            'comfort_norm', torch.tensor(config.comfort_norm))
        self.register_buffer(
            'drift_norm', torch.tensor(config.drift_norm))

    def encode_structured_stats(
        self,
        residual_norm: torch.Tensor,
        max_step_disp: torch.Tensor,
        curvature_change: torch.Tensor,
        jerk_change: torch.Tensor,
        total_disp: torch.Tensor,
        speed_max: torch.Tensor,
        support_score: torch.Tensor,
        drift_score: torch.Tensor,
    ) -> torch.Tensor:
        """编码结构化统计量。

        Args:
            residual_norm: [B] 残差范数
            max_step_disp: [B] 最大单步位移
            curvature_change: [B] 曲率变化
            jerk_change: [B] 加速度变化
            total_disp: [B] 总位移
            speed_max: [B] 最大速度
            support_score: [B] 支持分数
            drift_score: [B] 漂移分数

        Returns:
            [B, H//4] 编码后的统计特征
        """
        stats = torch.stack([
            residual_norm,
            max_step_disp,
            curvature_change,
            jerk_change,
            total_disp,
            speed_max,
            support_score,
            drift_score,
        ], dim=-1)  # [B, 8]

        return self.stats_encoder(stats)

    def forward(
        self,
        scene_token: torch.Tensor,
        reference_plan: torch.Tensor,
        correction: torch.Tensor,
        plan_confidence: Optional[torch.Tensor] = None,
        structured_stats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            scene_token: [B, D] 场景特征
            reference_plan: [B, T, 2] 或 [B, T*2] 参考轨迹
            correction: [B, T, 2] 或 [B, T*2] 修正量
            plan_confidence: [B, 1] 置信度（可选）
            structured_stats: [B, 8] 结构化统计量（可选）

        Returns:
            dict with:
                - pred_gain: [B, 1] 预测的 ΔR
                - pred_risk: [B, 1] 预测的综合风险
                - pred_collision: [B, 1] 预测的碰撞增量
                - pred_offroad: [B, 1] 预测的离道增量
                - pred_comfort: [B, 1] 预测的舒适度恶化
                - pred_drift: [B, 1] 预测的漂移
        """
        # 确保输入形状正确（支持 [B, T, 2] 或 [B, T*2]）
        if reference_plan.dim() == 3:  # [B, T, 2] -> [B, T*2]
            reference_plan = reference_plan.flatten(start_dim=1)
        if correction.dim() == 3:  # [B, T, 2] -> [B, T*2]
            correction = correction.flatten(start_dim=1)

        # 编码
        scene_feat = self.scene_encoder(scene_token)  # [B, H]
        ref_feat = self.ref_encoder(reference_plan)   # [B, H]
        corr_feat = self.corr_encoder(correction)     # [B, H]

        # 可选特征
        stats_feat = torch.zeros_like(scene_feat[:, :scene_feat.shape[1] // 4])
        if structured_stats is not None:
            if structured_stats.dim() == 1:
                structured_stats = structured_stats.unsqueeze(-1)
            stats_feat = self.stats_encoder(structured_stats)

        conf_feat = torch.zeros_like(scene_feat[:, :scene_feat.shape[1] // 8])
        if plan_confidence is not None:
            if plan_confidence.dim() == 1:
                plan_confidence = plan_confidence.unsqueeze(-1)
            conf_feat = self.conf_encoder(plan_confidence)

        # 融合
        combined = torch.cat([scene_feat, ref_feat, corr_feat, stats_feat, conf_feat], dim=-1)
        hidden = self.fusion(combined)  # [B, H]

        # 多头输出
        pred_gain = self.gain_head(hidden)  # [B, 1]
        pred_collision = self.collision_head(hidden)  # [B, 1]
        pred_offroad = self.offroad_head(hidden)  # [B, 1]
        pred_comfort = self.comfort_head(hidden)  # [B, 1]
        pred_drift = self.drift_head(hidden)  # [B, 1]

        # 归一化 + 组合 risk
        pred_collision_norm = pred_collision / self.collision_norm.clamp(min=1e-6)
        pred_offroad_norm = pred_offroad / self.offroad_norm.clamp(min=1e-6)
        pred_comfort_norm = pred_comfort / self.comfort_norm.clamp(min=1e-6)
        pred_drift_norm = pred_drift / self.drift_norm.clamp(min=1e-6)

        pred_risk = (
            self.lambda_collision * pred_collision_norm
            + self.lambda_offroad * pred_offroad_norm
            + self.lambda_comfort * pred_comfort_norm
            + self.lambda_drift * pred_drift_norm
        )

        return {
            'pred_gain': pred_gain,
            'pred_risk': pred_risk,
            'pred_collision': pred_collision,
            'pred_offroad': pred_offroad,
            'pred_comfort': pred_comfort,
            'pred_drift': pred_drift,
        }

    def evaluate(
        self,
        interface: PlanningInterface,
        correction: torch.Tensor,
        structured_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """从 PlanningInterface 评估。

        Args:
            interface: PlanningInterface
            correction: [B, T, 2] 修正量
            structured_stats: 可选的结构化统计量字典

        Returns:
            评估结果字典
        """
        if structured_stats is None:
            structured_stats_tensor = None
        else:
            structured_stats_tensor = torch.stack([
                structured_stats['residual_norm'],
                structured_stats['max_step_disp'],
                structured_stats['curvature_change'],
                structured_stats['jerk_change'],
                structured_stats['total_disp'],
                structured_stats['speed_max'],
                structured_stats['support_score'],
                structured_stats['drift_score'],
            ], dim=-1)

        return self.forward(
            scene_token=interface.scene_token,
            reference_plan=interface.reference_plan,
            correction=correction,
            plan_confidence=interface.plan_confidence,
            structured_stats=structured_stats_tensor,
        )

    def compute_filter_mask(
        self,
        advantages: torch.Tensor,
        tau_gain: float = 0.0,
        tau_risk: float = 0.5,
    ) -> torch.Tensor:
        """使用评估结果生成过滤掩码。

        这个方法在离线评估时使用。

        Args:
            advantages: [B] 真实 advantage
            tau_gain: Gain 阈值
            tau_risk: Risk 阈值

        Returns:
            [B] bool mask，True = 保留，False = 过滤
        """
        # 使用 pred_gain 和 pred_risk
        # 但这里需要先调用 forward 获取预测

        # 返回 None，表示需要外部传入预测结果
        return None


class LearnedUpdateGate:
    """ Learned Update Gate - 使用 UpdateEvaluator 替代规则 STAPO Gate。

    工作流程：
    1. 只审查会被强化的 update（正 advantage）
    2. Selector 预测 pred_gain 和 pred_risk
    3. 组合过滤：硬底线 + Learned Harmful 判断
    """

    def __init__(
        self,
        evaluator: UpdateEvaluator,
        tau_gain: float = 0.0,
        tau_risk: float = 0.5,
        advantage_threshold: float = 0.0,
    ):
        self.evaluator = evaluator
        self.tau_gain = tau_gain
        self.tau_risk = tau_risk
        self.advantage_threshold = advantage_threshold

    @torch.no_grad()
    def predict(
        self,
        interface: PlanningInterface,
        correction: torch.Tensor,
        structured_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """预测 gain 和 risk。

        Returns:
            dict with pred_gain, pred_risk, is_harmful, should_keep
        """
        # 确保 correction 形状正确
        if correction.dim() == 3:  # [B, T, 2] -> [B, T*2]
            correction_flat = correction.flatten(start_dim=1)
        else:
            correction_flat = correction

        result = self.evaluator.evaluate(interface, correction_flat, structured_stats)

        pred_gain = result['pred_gain'].squeeze(-1)  # [B]
        pred_risk = result['pred_risk'].squeeze(-1)  # [B]

        # Harmful 判断：gain 低 或 risk 高
        is_harmful = (pred_gain < self.tau_gain) | (pred_risk > self.tau_risk)

        return {
            'pred_gain': pred_gain,
            'pred_risk': pred_risk,
            'is_harmful': is_harmful,
        }

    def compute_mask(
        self,
        advantages: torch.Tensor,
        interface: PlanningInterface,
        correction: torch.Tensor,
        structured_stats: Optional[Dict[str, torch.Tensor]] = None,
        safety_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算最终过滤掩码。

        逻辑：
        - 负 advantage: 不过度屏蔽，让 policy 学会"别这么做"
        - 正 advantage: 进入 harmful 审查

        Args:
            advantages: [B] 真实 advantage
            interface: PlanningInterface
            correction: [B, T, 2] 修正量
            structured_stats: 结构化统计量
            safety_mask: [B] SafetyGuard 输出

        Returns:
            (mask, diagnostics)
        """
        B = advantages.shape[0]

        # 1. 判断是否是正 advantage
        is_positive = advantages > self.advantage_threshold

        # 2. Selector 预测
        pred = self.predict(interface, correction, structured_stats)
        is_harmful = pred['is_harmful']

        # 3. 组合过滤
        # 正 advantage ∧ harmful → 过滤
        # 负 advantage → 保留（让 policy 学会别这么做）
        should_filter = is_positive & is_harmful

        # 4. 应用 safety mask
        mask = ~should_filter
        if safety_mask is not None:
            mask = mask & safety_mask

        # 诊断
        diagnostics = {
            'n_total': B,
            'n_positive_adv': is_positive.sum().item(),
            'n_harmful': is_harmful.sum().item(),
            'n_filtered': should_filter.sum().item(),
            'n_safety_violations': (~safety_mask).sum().item() if safety_mask is not None else 0,
            'retention_ratio': mask.float().mean().item(),
            'pred_gain_mean': pred['pred_gain'].mean().item(),
            'pred_risk_mean': pred['pred_risk'].mean().item(),
        }

        return mask, diagnostics
