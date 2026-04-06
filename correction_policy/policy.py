"""
CorrectionPolicy — 统一策略接口

封装 Actor，提供训练和推理的统一 API：
- sample(): 训练时采样 correction + log_prob + entropy
- evaluate(): 给定 correction 计算 log_prob + entropy
- act(): 推理时确定性输出
- get_corrected_plan(): 推理时直接输出修正后的轨迹
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.correction_policy.actor import GaussianCorrectionActor


class CorrectionPolicy(nn.Module):
    """统一策略接口，封装 GaussianCorrectionActor。

    提供三种运行模式：
    1. sample()   - 训练：采样 + log_prob + entropy
    2. evaluate() - 训练/评估：给定 action 计算 log_prob + entropy
    3. act()      - 推理：确定性输出

    架构设计原则：
    - 所有方法接受 PlanningInterface 作为输入，保持接口统一
    - 内部维护 actor，不直接暴露给外部
    - 提供 get_corrected_plan() 一次性返回修正后轨迹
    """

    def __init__(
        self,
        scene_dim: int = 256,
        plan_len: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        action_scale: float = 1.0,
        actor_type: str = 'gaussian',
    ):
        """初始化 CorrectionPolicy。

        Args:
            scene_dim: 场景特征维度
            plan_len: 轨迹时间步数
            hidden_dim: 隐藏层维度
            dropout: Dropout 比例
            log_std_min: 对数标准差下限
            log_std_max: 对数标准差上限
            action_scale: 修正量缩放因子
            actor_type: 'gaussian' 或 'deterministic'
        """
        super().__init__()

        self.scene_dim = scene_dim
        self.plan_len = plan_len
        self.hidden_dim = hidden_dim
        self.actor_type = actor_type

        # 构建 Actor
        if actor_type == 'gaussian':
            self.actor = GaussianCorrectionActor(
                scene_dim=scene_dim,
                plan_len=plan_len,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}")

    def sample(self, interface: PlanningInterface) -> Dict[str, torch.Tensor]:
        """训练时采样 correction。

        从高斯策略采样一个 correction，计算对应的 log_prob 和 entropy。

        Args:
            interface: PlanningInterface 输入

        Returns:
            dict with:
                - correction: [B, T, 2] 采样的修正量
                - corrected_plan: [B, T, 2] ref + correction
                - log_prob: [B] 对数概率
                - entropy: [B] 熵
                - mean: [B, T, 2] 修正均值
                - std: [B, T, 2] 修正标准差
        """
        output = self.actor(
            scene_token=interface.scene_token,
            reference_plan=interface.reference_plan,
            plan_confidence=interface.plan_confidence,
            deterministic=False,
        )

        correction = output['action']
        corrected_plan = interface.reference_plan + correction

        return {
            'correction': correction,
            'corrected_plan': corrected_plan,
            'log_prob': output['log_prob'],
            'entropy': output['entropy'],
            'mean': output['action'],  # 采样时会偏离 mean，需要从 output 取
        }

    def sample_with_stats(self, interface: PlanningInterface) -> Dict[str, torch.Tensor]:
        """采样并返回完整的统计信息（用于调试）。"""
        output = self.actor(
            scene_token=interface.scene_token,
            reference_plan=interface.reference_plan,
            plan_confidence=interface.plan_confidence,
            deterministic=False,
        )

        correction = output['action']
        corrected_plan = interface.reference_plan + correction

        # 从 mean 的 reshape 版本
        B = interface.scene_token.shape[0]
        mean_2d = output['mean'].reshape(B, self.plan_len, 2)

        return {
            'correction': correction,
            'corrected_plan': corrected_plan,
            'log_prob': output['log_prob'],
            'entropy': output['entropy'],
            'mean': mean_2d,
            'std': output['std'].reshape(B, self.plan_len, 2),
        }

    def evaluate(
        self,
        interface: PlanningInterface,
        correction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """给定 correction 计算 log_prob 和 entropy。

        用于：
        1. Behavioral Cloning 阶段评估 GT correction 的概率
        2. PPO 等算法的 clipped surrogate loss 计算

        Args:
            interface: PlanningInterface
            correction: [B, T, 2] 要评估的动作

        Returns:
            dict with:
                - log_prob: [B] 对数概率
                - entropy: [B] 熵
        """
        eval_result = self.actor.evaluate_action(interface, correction)
        return {
            'log_prob': eval_result['log_prob'],
            'entropy': eval_result['entropy'],
        }

    def act(self, interface: PlanningInterface) -> torch.Tensor:
        """推理时获取确定性 correction。

        直接返回高斯策略的均值（不加随机噪声）。

        Args:
            interface: PlanningInterface

        Returns:
            [B, T, 2] 修正量
        """
        output = self.actor(
            scene_token=interface.scene_token,
            reference_plan=interface.reference_plan,
            plan_confidence=interface.plan_confidence,
            deterministic=True,
        )
        return output['action']

    def get_corrected_plan(self, interface: PlanningInterface) -> torch.Tensor:
        """推理时直接获取修正后的轨迹。

        一次性完成：act() + 加到 reference_plan

        Args:
            interface: PlanningInterface

        Returns:
            [B, T, 2] 修正后的轨迹
        """
        correction = self.act(interface)
        return interface.reference_plan + correction

    def get_statistics(self, interface: PlanningInterface) -> Dict[str, float]:
        """获取策略统计信息（用于监控）。

        Returns:
            dict with:
                - mean_abs_correction: 平均修正幅度
                - mean_std: 平均标准差
                - mean_entropy: 平均熵
        """
        output = self.actor(
            scene_token=interface.scene_token,
            reference_plan=interface.reference_plan,
            plan_confidence=interface.plan_confidence,
            deterministic=True,
        )
        mean_abs = output['action'].abs().mean().item()
        std_mean = output['std'].mean().item()

        # 计算熵
        log_std = torch.log(output['std'])
        var = output['std'] ** 2
        import math
        entropy = 0.5 * (torch.log(2 * math.pi * math.e * var)).sum(dim=-1).mean().item()

        return {
            'mean_abs_correction': mean_abs,
            'mean_std': std_mean,
            'mean_entropy': entropy,
        }

    def forward(self, interface: PlanningInterface) -> Dict[str, torch.Tensor]:
        """前向传播：直接返回 act() 的结果。

        兼容 nn.Module 的标准调用方式。
        """
        return {'correction': self.act(interface)}
