"""
Value Critic - 价值网络

用于 Actor-Critic 架构，评估状态或状态-动作对的价值。
这可以用于：
1. Advantage 计算
2. 指导策略更新
3. 离线 RL 中的 GAE 计算
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional


class StateValueCritic(nn.Module):
    """
    状态价值 Critic：评估 V(s)
    
    输入：场景状态
    输出：状态价值 V(s)
    """
    
    def __init__(
        self,
        scene_dim: int = 256,
        plan_len: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.scene_dim = scene_dim
        self.plan_len = plan_len
        self.action_dim = plan_len * 2
        
        # 特征编码器
        self.scene_proj = nn.Linear(scene_dim, hidden_dim)
        self.plan_proj = nn.Linear(self.action_dim, hidden_dim)
        
        # 融合网络
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        scene_token: torch.Tensor,
        reference_plan: torch.Tensor,
        correction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            scene_token: [B, D] 场景特征
            reference_plan: [B, T, 2] 参考轨迹
            correction: [B, T, 2] 修正量（可选）
        
        Returns:
            value: [B, 1] 状态价值
        """
        scene_feat = self.scene_proj(scene_token)
        plan_feat = self.plan_proj(reference_plan.flatten(1))
        
        if correction is not None:
            corr_feat = self.plan_proj(correction.flatten(1))
            combined = torch.cat([scene_feat, plan_feat, corr_feat], dim=-1)
        else:
            combined = torch.cat([scene_feat, plan_feat], dim=-1)
        
        value = self.network(combined)
        return value  # [B, 1]


class QValueCritic(nn.Module):
    """
    动作价值 Critic：评估 Q(s, a)
    
    输入：场景状态 + 动作
    输出：动作价值 Q(s, a)
    """
    
    def __init__(
        self,
        scene_dim: int = 256,
        plan_len: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.scene_dim = scene_dim
        self.plan_len = plan_len
        self.action_dim = plan_len * 2
        
        # 特征编码器
        self.scene_proj = nn.Linear(scene_dim, hidden_dim)
        self.action_proj = nn.Linear(self.action_dim, hidden_dim)
        
        # Q 网络
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        scene_token: torch.Tensor,
        correction: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            scene_token: [B, D] 场景特征
            correction: [B, T, 2] 修正量
        
        Returns:
            q_value: [B, 1] 动作价值
        """
        scene_feat = self.scene_proj(scene_token)
        action_feat = self.action_proj(correction.flatten(1))
        
        combined = torch.cat([scene_feat, action_feat], dim=-1)
        q_value = self.network(combined)
        return q_value  # [B, 1]


class DualCritic(nn.Module):
    """
    双重 Critic（用于减少价值过估计）
    
    维护两个独立的 Q 网络，取较小的 Q 值
    """
    
    def __init__(
        self,
        scene_dim: int = 256,
        plan_len: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.critic1 = QValueCritic(scene_dim, plan_len, hidden_dim, dropout)
        self.critic2 = QValueCritic(scene_dim, plan_len, hidden_dim, dropout)
        
    def forward(self, scene_token: torch.Tensor, correction: torch.Tensor):
        """返回两个 Q 值"""
        q1 = self.critic1(scene_token, correction)
        q2 = self.critic2(scene_token, correction)
        return q1, q2
    
    def get_min_q(self, scene_token: torch.Tensor, correction: torch.Tensor):
        """返回较小的 Q 值（用于减少过估计）"""
        q1, q2 = self.forward(scene_token, correction)
        return torch.min(q1, q2)
