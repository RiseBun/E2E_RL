"""
Safety Guard — 硬性物理约束检查

在训练过程中，对每条采样的 correction 进行硬性物理约束检查。
违规的样本直接被 mask 掉（不参与梯度更新），不会进入 STAPO gate。

与旧 HUF 的区别：
- HUF（已废弃）：推理时裁决是否接受修正
- Safety Guard：训练时硬约束，防止危险修正进入梯度

检查项目：
1. 残差范数：||residual||_2 不能过大
2. 单步位移：相邻时间步的 correction 差值不能过大
3. 速度上限：修正后的速度不能超过物理限制
4. 修正后轨迹不能与参考偏离太远（总位移限制）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class SafetyGuardConfig:
    """Safety Guard 配置。"""
    enabled: bool = True

    # 残差范数约束
    max_residual_norm: float = 5.0  # L2 范数阈值

    # 单步位移约束
    max_step_disp: float = 2.0  # 相邻步的最大位移

    # 速度约束
    max_speed: float = 15.0  # 最大速度 m/s（修正后的瞬时速度）
    dt: float = 0.5  # 时间步间隔

    # 修正后轨迹与参考轨迹的总位移约束
    max_total_disp: float = 10.0  # 修正后轨迹与参考轨迹的终点最大距离

    # 舒适度约束（可选）
    max_acceleration: float = 5.0  # 最大加速度 m/s²

    def __post_init__(self):
        assert self.max_residual_norm > 0, "max_residual_norm must be positive"
        assert self.max_step_disp > 0, "max_step_disp must be positive"


class SafetyGuard:
    """硬性物理约束检查器。

    对 correction 进行一系列物理可行性检查，
    返回 [B] bool mask，True = 安全（参与训练），False = 违规（静音）。

    设计原则：
    - 不需要任何可学习参数
    - 完全基于几何/物理约束
    - 作为 STAPO gate 的前置过滤器
    """

    def __init__(self, config: Optional[SafetyGuardConfig] = None, **kwargs):
        """初始化 SafetyGuard。

        Args:
            config: SafetyGuardConfig 配置对象
            **kwargs: 如果没有传 config，可以直接传参
        """
        if config is None:
            config = SafetyGuardConfig(**kwargs)
        self.cfg = config

    def check(
        self,
        correction: torch.Tensor,
        reference_plan: torch.Tensor,
        corrected_plan: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """检查 correction 是否满足物理约束。

        Args:
            correction: [B, T, 2] 修正量
            reference_plan: [B, T, 2] 参考轨迹
            corrected_plan: [B, T, 2] 修正后轨迹（可选，不传则自动计算）
            dt: 时间步间隔（覆盖 config.dt）

        Returns:
            [B] bool mask，True = 安全，False = 违规
        """
        if not self.cfg.enabled:
            # 未启用时全部通过
            return torch.ones(correction.shape[0], dtype=torch.bool, device=correction.device)

        dt = dt if dt is not None else self.cfg.dt
        device = correction.device

        # 初始化 mask（全 True）
        batch_size = correction.shape[0]
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # 1. 残差范数检查
        mask = mask & self._check_residual_norm(correction)

        # 2. 单步位移检查
        mask = mask & self._check_step_displacement(correction)

        # 3. 速度约束检查
        if corrected_plan is None:
            corrected_plan = reference_plan + correction
        mask = mask & self._check_speed(corrected_plan, dt)

        # 4. 总位移约束检查（修正后轨迹与参考轨迹的终点距离）
        mask = mask & self._check_total_displacement(reference_plan, corrected_plan)

        return mask

    def _check_residual_norm(self, correction: torch.Tensor) -> torch.Tensor:
        """检查残差 L2 范数。

        约束：||correction||_2 < max_residual_norm
        """
        # 每条轨迹的 L2 范数：[B]
        residual_norm = torch.norm(correction, dim=-1).norm(dim=-1)
        return residual_norm < self.cfg.max_residual_norm

    def _check_step_displacement(self, correction: torch.Tensor) -> torch.Tensor:
        """检查相邻时间步的位移。

        约束：||correction[t] - correction[t-1]||_2 < max_step_disp
        """
        # 相邻步的差值：[B, T-1, 2]
        step_diff = torch.diff(correction, dim=1)
        # 最大步位移：[B]
        max_step = torch.norm(step_diff, dim=-1).max(dim=-1).values
        return max_step < self.cfg.max_step_disp

    def _check_speed(self, plan: torch.Tensor, dt: float) -> torch.Tensor:
        """检查速度上限。

        约束：speed < max_speed

        注意：这里检查的是修正后轨迹的速度，而非修正量本身的速度。
        """
        if plan.shape[1] < 2:
            return torch.ones(plan.shape[0], dtype=torch.bool, device=plan.device)

        # 速度：[B, T-1, 2]
        velocity = torch.diff(plan, dim=1) / dt
        # 速度幅值：[B, T-1]
        speed = torch.norm(velocity, dim=-1)
        # 最大速度：[B]
        max_speed = speed.max(dim=-1).values
        return max_speed < self.cfg.max_speed

    def _check_total_displacement(
        self,
        reference_plan: torch.Tensor,
        corrected_plan: torch.Tensor,
    ) -> torch.Tensor:
        """检查修正后轨迹与参考轨迹的终点距离。

        约束：||corrected_plan[-1] - reference_plan[-1]||_2 < max_total_disp
        """
        end_disp = torch.norm(
            corrected_plan[:, -1] - reference_plan[:, -1],
            dim=-1
        )
        return end_disp < self.cfg.max_total_disp

    def get_violation_info(
        self,
        correction: torch.Tensor,
        reference_plan: torch.Tensor,
        corrected_plan: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """获取违规详情（用于调试和诊断）。

        Returns:
            dict with:
                - mask: [B] bool mask
                - residual_norm: [B] 每条轨迹的残差范数
                - max_step_disp: [B] 最大单步位移
                - max_speed: [B] 最大速度
                - total_disp: [B] 总位移
                - n_violations: 违规数量
        """
        dt = dt if dt is not None else self.cfg.dt
        if corrected_plan is None:
            corrected_plan = reference_plan + correction

        residual_norm = torch.norm(correction, dim=-1).norm(dim=-1)

        if correction.shape[1] >= 2:
            step_diff = torch.diff(correction, dim=1)
            max_step_disp = torch.norm(step_diff, dim=-1).max(dim=-1).values
        else:
            max_step_disp = torch.zeros_like(residual_norm)

        if corrected_plan.shape[1] >= 2:
            velocity = torch.diff(corrected_plan, dim=1) / dt
            max_speed = torch.norm(velocity, dim=-1).max(dim=-1).values
        else:
            max_speed = torch.zeros_like(residual_norm)

        total_disp = torch.norm(
            corrected_plan[:, -1] - reference_plan[:, -1],
            dim=-1
        )

        mask = self.check(correction, reference_plan, corrected_plan, dt)
        n_violations = (~mask).sum().item()

        return {
            'mask': mask,
            'residual_norm': residual_norm,
            'max_step_disp': max_step_disp,
            'max_speed': max_speed,
            'total_disp': total_disp,
            'n_violations': n_violations,
        }
