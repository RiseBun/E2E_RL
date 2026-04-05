"""Planner-agnostic 残差精炼网络。

消费完整的 PlanningInterface:
- scene_token: 场景语义紧凑表示
- reference_plan: 原始规划器输出的参考轨迹
- plan_confidence: 规划置信度 / 不确定性信号
- safety_features: 安全相关特征（碰撞风险、离道风险等）

所有可选字段在缺失时自动用零向量填充，保持向后兼容。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from E2E_RL.planning_interface.interface import PlanningInterface


class InterfaceRefiner(nn.Module):
    """基于完整 PlanningInterface 的残差精炼网络。

    架构:
        scene_token      → scene_proj   → [B, H]
        reference_plan   → plan_proj    → [B, H]
        plan_confidence  → conf_proj    → [B, H_s]
        safety_features  → safety_proj  → [B, H_s]
                                   ↓ concat
                              fusion MLP
                                   ↓
                            residual_head → [B, T, 2]
                            score_head    → [B, 1]

    Args:
        scene_dim: scene_token 维度
        plan_len: reference_plan 展平维度 (T * 2)
        hidden_dim: 主隐层维度
        conf_dim: confidence 输入维度（默认 1）
        safety_dim: safety_features 拼接后的维度（默认 0 = 自动推断）
        dropout: dropout 比例
        output_norm: 是否对 residual 施加 tanh 约束
    """

    def __init__(
        self,
        scene_dim: int,
        plan_len: int,
        hidden_dim: int = 256,
        conf_dim: int = 1,
        safety_dim: int = 0,
        dropout: float = 0.1,
        output_norm: bool = False,
    ):
        super().__init__()
        self.plan_len = plan_len
        self.hidden_dim = hidden_dim
        self.conf_dim = conf_dim
        self.output_norm = output_norm

        # ---- 主特征编码 ----
        self.scene_proj = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.plan_proj = nn.Sequential(
            nn.Linear(plan_len, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- 辅助特征编码 (confidence + safety) ----
        # 用小维度编码，避免主通道被辅助信号淹没
        aux_hidden = max(hidden_dim // 4, 16)
        self.conf_proj = nn.Sequential(
            nn.Linear(conf_dim, aux_hidden),
            nn.ReLU(inplace=True),
        )

        # safety_dim=0 表示未指定，使用默认大小
        # 实际输入维度在 forward 时动态适配
        self._safety_dim = safety_dim
        if safety_dim > 0:
            self.safety_proj = nn.Sequential(
                nn.Linear(safety_dim, aux_hidden),
                nn.ReLU(inplace=True),
            )
        else:
            # 延迟初始化：第一次 forward 时根据实际维度创建
            self.safety_proj = None

        # ---- 融合层 ----
        # fusion 输入 = scene(H) + plan(H) + conf(aux_H) + safety(aux_H)
        fusion_in = hidden_dim * 2 + aux_hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ---- 输出头 ----
        self.residual_head = nn.Linear(hidden_dim, plan_len)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._aux_hidden = aux_hidden

    def _encode_safety(self, interface: PlanningInterface, batch_size: int) -> torch.Tensor:
        """将 safety_features dict 编码为固定维度向量。"""
        device = interface.scene_token.device

        if interface.safety_features is None or len(interface.safety_features) == 0:
            return torch.zeros(batch_size, self._aux_hidden, device=device)

        # 将所有 safety 值展平拼接为 [B, total_dim]
        parts = []
        for key in sorted(interface.safety_features.keys()):
            val = interface.safety_features[key]
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            parts.append(val.reshape(batch_size, -1))
        safety_flat = torch.cat(parts, dim=-1)  # [B, total_safety_dim]

        # 延迟初始化 safety_proj
        if self.safety_proj is None or self._safety_dim != safety_flat.shape[-1]:
            self._safety_dim = safety_flat.shape[-1]
            self.safety_proj = nn.Sequential(
                nn.Linear(self._safety_dim, self._aux_hidden),
                nn.ReLU(inplace=True),
            ).to(device)

        return self.safety_proj(safety_flat)

    def _encode_confidence(self, interface: PlanningInterface, batch_size: int) -> torch.Tensor:
        """将 plan_confidence 编码为固定维度向量。"""
        device = interface.scene_token.device

        if interface.plan_confidence is None:
            return torch.zeros(batch_size, self._aux_hidden, device=device)

        conf = interface.plan_confidence
        if conf.dim() == 1:
            conf = conf.unsqueeze(-1)
        # 若 conf 维度和 conf_dim 不一致，做平均压缩
        if conf.shape[-1] != self.conf_dim:
            conf = conf.mean(dim=-1, keepdim=True)

        return self.conf_proj(conf)

    def forward(
        self,
        interface: PlanningInterface,
        plan_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """精炼参考轨迹。

        Args:
            interface: 完整 PlanningInterface。
            plan_mask: 有效步掩码 [B, T]，可选。

        Returns:
            dict: residual, refined_plan, residual_norm, refine_score
        """
        scene_token = interface.scene_token
        reference_plan = interface.reference_plan

        if scene_token.dim() != 2:
            scene_token = scene_token.flatten(1)

        batch_size, time_steps, coord_dim = reference_plan.shape
        plan_flat = reference_plan.reshape(batch_size, time_steps * coord_dim)

        # ---- 编码各通道 ----
        scene_feat = self.scene_proj(scene_token)       # [B, H]
        plan_feat = self.plan_proj(plan_flat)            # [B, H]
        conf_feat = self._encode_confidence(interface, batch_size)  # [B, H_s]
        safety_feat = self._encode_safety(interface, batch_size)    # [B, H_s]

        # ---- 融合 ----
        fused = torch.cat([scene_feat, plan_feat, conf_feat, safety_feat], dim=-1)
        fused = self.fusion(fused)  # [B, H]

        # ---- 残差输出 ----
        residual_flat = self.residual_head(fused)
        if self.output_norm:
            residual_flat = torch.tanh(residual_flat) * 1.0
        residual = residual_flat.view(batch_size, time_steps, coord_dim)

        refined_plan = reference_plan + residual
        if plan_mask is not None:
            mask = plan_mask.unsqueeze(-1).float()
            residual = residual * mask
            refined_plan = reference_plan * (1.0 - mask) + refined_plan * mask

        residual_norm = torch.norm(residual, dim=-1, p=2).mean(dim=-1)
        refine_score = self.score_head(fused).squeeze(-1)

        return {
            'residual': residual,
            'refined_plan': refined_plan,
            'residual_norm': residual_norm,
            'refine_score': refine_score,
        }
