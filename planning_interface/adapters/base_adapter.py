"""规划接口适配器的抽象基类。

适配器负责将特定规划器 (VAD / UniAD / ...) 的原始输出映射到
统一的 PlanningInterface，是架构中唯一与规划器耦合的组件。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from E2E_RL.planning_interface.interface import PlanningInterface


class BasePlanningAdapter(ABC):
    """所有 Planner → PlanningInterface 适配器的抽象接口。

    子类需要实现四个提取方法，分别负责提取:
    - scene_token: 场景语义紧凑表示
    - reference_plan: 原始规划器输出的参考轨迹
    - plan_confidence: 规划置信度 / 不确定性信号
    - safety_features: 安全相关的紧凑特征
    """

    @abstractmethod
    def extract_scene_token(
        self,
        planner_outputs: Dict[str, Any],
    ) -> torch.Tensor:
        """从规划器输出中提取场景语义 token。

        Returns:
            shape [B, D] 的紧凑场景表示
        """

    @abstractmethod
    def extract_reference_plan(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """提取参考轨迹和候选轨迹。

        Returns:
            (reference_plan, candidate_plans)
            - reference_plan: [B, T, 2] ego-centric 绝对坐标
            - candidate_plans: [B, M, T, 2] 或 None
        """

    @abstractmethod
    def extract_plan_confidence(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """提取规划置信度 / 不确定性信号。

        Returns:
            [B, 1] 或 [B, M] 的置信度张量，或 None
        """

    @abstractmethod
    def extract_safety_features(
        self,
        planner_outputs: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """提取安全相关特征（碰撞风险、离道风险等）。

        Returns:
            字典，包含各类安全信号张量
        """

    def extract(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
        hard_case_score: Optional[torch.Tensor] = None,
    ) -> PlanningInterface:
        """完整的提取流程：将规划器输出转为 PlanningInterface。

        这是适配器的主入口，子类通常不需要重写此方法。
        """
        scene_token = self.extract_scene_token(planner_outputs)
        reference_plan, candidate_plans = self.extract_reference_plan(
            planner_outputs, ego_fut_cmd
        )
        plan_confidence = self.extract_plan_confidence(
            planner_outputs, ego_fut_cmd
        )
        safety_features = self.extract_safety_features(planner_outputs)

        if hard_case_score is None:
            hard_case_score = torch.zeros(
                (reference_plan.shape[0], 1),
                device=reference_plan.device,
                dtype=reference_plan.dtype,
            )

        return PlanningInterface(
            scene_token=scene_token,
            reference_plan=reference_plan,
            candidate_plans=candidate_plans,
            plan_confidence=plan_confidence,
            safety_features=safety_features,
            hard_case_score=hard_case_score,
            metadata={
                'adapter': self.__class__.__name__,
            },
        )
