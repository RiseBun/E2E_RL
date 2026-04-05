"""可靠性评估模型 (工程版): 预测残差更新的可信度 (Gain & Risk).

基于 STAPO 思想和工程实践，采用双头回归架构。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class ReliabilityNet(nn.Module):
    """可训练的更新可靠性评估网络 (双头回归版).

    输入:
        - scene_token: [B, D] 场景特征
        - reference_plan: [B, T, 2] 参考轨迹
        - proposed_residual: [B, T, 2] 预测残差
        - plan_confidence: [B, 1] 原始模型置信度 (如果有)
        - safety_features: [B, K] 预计算的安全特征 (如碰撞距离等)
        - heuristic_scores: [B, 3] 现有的规则评分 (uncertainty, support, drift)

    输出:
        - pred_gain: [B, 1] 预测的 ΔR (Reward 提升)
        - pred_risk: [B, 1] 预测的综合 Risk (连续代价)
    """

    def __init__(
        self,
        scene_dim: int,
        plan_len: int,
        safety_feat_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1. 结构化特征拼接
        # scene(D) + ref_plan(T*2) + residual(T*2) + confidence(1) + safety(K) + heuristics(3)
        input_dim = scene_dim + (plan_len * 2) * 2 + 1 + safety_feat_dim + 3
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 2. Gain Head (回归 ΔR)
        self.gain_head = nn.Linear(hidden_dim, 1)

        # 3. Risk Heads (分项回归): collision/offroad/comfort/drift
        self.risk_collision_head = nn.Linear(hidden_dim, 1)
        self.risk_offroad_head = nn.Linear(hidden_dim, 1)
        self.risk_comfort_head = nn.Linear(hidden_dim, 1)
        self.risk_drift_head = nn.Linear(hidden_dim, 1)

        # 4. 综合风险层（用于兼容旧逻辑）
        self.risk_aggregator = nn.Linear(4, 1, bias=False)
        # 默认初始化为简单加权 (可在训练中调节)
        with torch.no_grad():
            self.risk_aggregator.weight.copy_(torch.tensor([[2.0, 1.0, 0.5, 1.0]]))

    def forward(
        self,
        scene_token: torch.Tensor,
        reference_plan: torch.Tensor,
        proposed_residual: torch.Tensor,
        plan_confidence: Optional[torch.Tensor] = None,
        safety_features: Optional[torch.Tensor] = None,
        heuristic_scores: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """预测 Gain 和 Risk."""
        B = scene_token.shape[0]
        
        # 准备输入张量
        ref_flat = reference_plan.flatten(1)
        res_flat = proposed_residual.flatten(1)
        
        conf = plan_confidence if plan_confidence is not None else torch.zeros((B, 1), device=scene_token.device)
        safety = safety_features if safety_features is not None else torch.zeros((B, 0), device=scene_token.device)
        heuristics = heuristic_scores if heuristic_scores is not None else torch.zeros((B, 3), device=scene_token.device)
        
        x = torch.cat([
            scene_token, 
            ref_flat, 
            res_flat, 
            conf, 
            safety, 
            heuristics
        ], dim=-1)
        
        feat = self.encoder(x)
        
        pred_gain = self.gain_head(feat)

        pred_risk_collision = self.risk_collision_head(feat)
        pred_risk_offroad = self.risk_offroad_head(feat)
        pred_risk_comfort = self.risk_comfort_head(feat)
        pred_risk_drift = self.risk_drift_head(feat)

        pred_risk_components = torch.cat([
            pred_risk_collision,
            pred_risk_offroad,
            pred_risk_comfort,
            pred_risk_drift,
        ], dim=-1)

        pred_risk_total = self.risk_aggregator(pred_risk_components)

        return pred_gain, pred_risk_total, pred_risk_components
