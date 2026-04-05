"""
Correction Actor - 真正的策略网络

不同于 Refiner 的确定性残差输出，Correction Actor 输出修正的分布。
支持两种模式：
1. Stochastic: 输出均值和方差，采样得到修正
2. Deterministic: 输出修正的均值（可用于推理）

这才是真正的 RL Policy。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class GaussianCorrectionActor(nn.Module):
    """
    高斯策略 Actor：输出修正的高斯分布参数
    
    不同于监督学习的确定性输出，这里输出的是分布参数：
    - mean: 修正的均值
    - log_std: 修正的标准差（对数形式）
    
    这样可以通过采样得到多样化的修正策略。
    """
    
    def __init__(
        self,
        scene_dim: int = 256,
        plan_len: int = 12,
        hidden_dim: int = 256,
        action_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            scene_dim: 场景特征维度
            plan_len: 轨迹长度（时间步）
            hidden_dim: 隐藏层维度
            action_dim: 动作维度，默认 2*plan_len (dx, dy for each timestep)
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.scene_dim = scene_dim
        self.plan_len = plan_len
        self.action_dim = action_dim or plan_len * 2  # dx, dy for each timestep
        
        # 特征编码器
        self.scene_proj = nn.Linear(scene_dim, hidden_dim)
        self.plan_proj = nn.Linear(self.action_dim, hidden_dim)
        self.conf_proj = nn.Linear(1, hidden_dim // 4)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 策略头：输出均值和对数标准差
        self.mean_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
        
        # 可学习的初始对数标准差
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))
        
    def forward(
        self,
        scene_token: torch.Tensor,
        reference_plan: torch.Tensor,
        plan_confidence: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            scene_token: [B, D] 场景特征
            reference_plan: [B, T, 2] 参考轨迹
            plan_confidence: [B, 1] 置信度
            deterministic: 若为 True，返回均值（用于推理）
        
        Returns:
            dict with:
                - mean: [B, action_dim] 修正均值
                - std: [B, action_dim] 修正标准差
                - action: [B, T, 2] 采样的修正 (deterministic=True 时等于 mean)
                - log_prob: [B] 动作的对数概率
        """
        B = scene_token.shape[0]
        
        # 编码特征
        scene_feat = self.scene_proj(scene_token)
        
        plan_flat = reference_plan.flatten(1)  # [B, T*2]
        plan_feat = self.plan_proj(plan_flat)
        
        if plan_confidence is None:
            plan_confidence = torch.ones(B, 1, device=scene_token.device)
        conf_feat = self.conf_proj(plan_confidence)
        
        # 融合
        combined = torch.cat([scene_feat, plan_feat, conf_feat], dim=-1)
        hidden = self.fusion(combined)
        
        # 策略输出
        mean = self.mean_head(hidden)
        
        # 标准差：可学习参数 + 线性组合
        log_std = self.log_std_head(hidden) + self.log_std
        
        # 限制范围，防止数值爆炸
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)
        
        # 输出
        if deterministic:
            # 推理模式：直接用均值
            action = mean
            log_prob = None
            entropy = None
        else:
            # 训练模式：采样
            action = mean + std * torch.randn_like(mean)
            log_prob = self._compute_log_prob(action, mean, std)
            entropy = self._compute_entropy(std)

        # reshape to [B, T, 2]
        action_2d = action.reshape(B, self.plan_len, 2)

        return {
            'mean': mean,
            'std': std,
            'action': action_2d,  # [B, T, 2]
            'log_prob': log_prob,  # [B]
            'entropy': entropy,    # [B]
        }
    
    def _compute_log_prob(
        self,
        action: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """计算高斯动作的对数概率"""
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var 
            + torch.log(var) 
            + torch.log(2 * torch.tensor(torch.pi))
        ).sum(dim=-1)
        return log_prob

    def _compute_entropy(self, std: torch.Tensor) -> torch.Tensor:
        """计算高斯分布的熵"""
        return 0.5 * (
            1 + torch.log(2 * torch.tensor(torch.pi, device=std.device)) + 2 * torch.log(std.clamp(min=1e-8))
        ).sum(dim=-1)

    def get_action(
        self,
        scene_token: torch.Tensor,
        reference_plan: torch.Tensor,
        plan_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        获取修正动作（确定性，用于推理）
        
        Returns: [B, T, 2] 修正量
        """
        output = self.forward(
            scene_token=scene_token,
            reference_plan=reference_plan,
            plan_confidence=plan_confidence,
            deterministic=True,
        )
        return output['action']

    def evaluate_action(
        self,
        interface: PlanningInterface,
        correction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """评估给定动作的 log_prob 和 entropy。

        Args:
            interface: PlanningInterface
            correction: [B, T, 2] 要评估的动作

        Returns:
            dict with:
                - log_prob: [B] 对数概率
                - entropy: [B] 熵
        """
        B = interface.scene_token.shape[0]

        # 获取高斯分布参数
        scene_feat = self.scene_proj(interface.scene_token)
        plan_flat = interface.reference_plan.flatten(1)
        plan_feat = self.plan_proj(plan_flat)

        if interface.plan_confidence is None:
            plan_confidence = torch.ones(B, 1, device=interface.scene_token.device)
        else:
            plan_confidence = interface.plan_confidence
        conf_feat = self.conf_proj(plan_confidence)

        combined = torch.cat([scene_feat, plan_feat, conf_feat], dim=-1)
        hidden = self.fusion(combined)

        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden) + self.log_std
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)

        # 展平
        mean_flat = mean.flatten(start_dim=1)  # [B, T*2]
        correction_flat = correction.flatten(start_dim=1)  # [B, T*2]
        std_flat = std.flatten(start_dim=1)  # [B, T*2]

        # 计算 log_prob
        log_prob = self._compute_log_prob(correction_flat, mean_flat, std_flat)

        # 计算 entropy
        entropy = self._compute_entropy(std_flat)

        return {
            'log_prob': log_prob,
            'entropy': entropy,
        }


class DeterministicCorrectionActor(nn.Module):
    """
    确定性 Actor（简化版，用于对比）
    
    这个和原来的 Refiner 类似，但明确标注为 Policy 而非监督网络。
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
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 动作头
        self.action_head = nn.Linear(hidden_dim, self.action_dim)
        
    def forward(
        self,
        scene_token: torch.Tensor,
        reference_plan: torch.Tensor,
        plan_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（确定性）
        
        Returns:
            dict with:
                - action: [B, T, 2] 修正量
        """
        # 编码特征
        scene_feat = self.scene_proj(scene_token)
        plan_feat = self.plan_proj(reference_plan.flatten(1))
        
        # 融合
        hidden = self.fusion(torch.cat([scene_feat, plan_feat], dim=-1))
        
        # 输出
        action = self.action_head(hidden).reshape(-1, self.plan_len, 2)
        
        return {'action': action}
    
    def get_action(
        self,
        scene_token: torch.Tensor,
        reference_plan: torch.Tensor,
        plan_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取修正动作（确定性，用于推理）"""
        output = self.forward(
            scene_token=scene_token,
            reference_plan=reference_plan,
            plan_confidence=plan_confidence,
        )
        return output['action']
