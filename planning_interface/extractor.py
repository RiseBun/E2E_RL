"""统一规划接口提取器。

PlanningInterfaceExtractor 作为高层提取管道:
1. 接收规划器原始输出
2. 委托给具体的 adapter 完成提取
3. 可选的后处理（shape 归一化、调试日志等）

设计规则: 提取器本身不包含任何规划器特定逻辑。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from .adapters.base_adapter import BasePlanningAdapter
from .adapters.vad_adapter import VADPlanningAdapter
from .adapters.diffusiondrive_adapter import DiffusionDrivePlanningAdapter
from .interface import PlanningInterface

logger = logging.getLogger(__name__)


class PlanningInterfaceExtractor:
    """从规划器输出中提取统一规划接口。

    该类是 adapter 层的薄封装，提供:
    - 根据 adapter_type 自动选择适配器
    - shape 验证和调试日志
    - 向后兼容的 API

    Args:
        adapter: 具体的适配器实例（如 VADPlanningAdapter）
        debug: 是否打印调试信息
    """

    def __init__(
        self,
        adapter: Optional[BasePlanningAdapter] = None,
        debug: bool = False,
    ):
        if adapter is None:
            # 默认使用 VAD 适配器
            adapter = VADPlanningAdapter()
        self.adapter = adapter
        self.debug = debug

    @classmethod
    def from_config(
        cls,
        adapter_type: str = 'vad',
        scene_pool: str = 'mean',
        debug: bool = False,
        **adapter_kwargs: Any,
    ) -> 'PlanningInterfaceExtractor':
        """根据配置创建提取器。

        Args:
            adapter_type: 适配器类型 ('vad', 'uniad', ...)
            scene_pool: BEV 特征池化方式
            debug: 调试模式
            **adapter_kwargs: 传递给适配器的额外参数
        """
        if adapter_type == 'vad':
            adapter = VADPlanningAdapter(
                scene_pool=scene_pool,
                **adapter_kwargs,
            )
        elif adapter_type == 'diffusiondrive':
            adapter = DiffusionDrivePlanningAdapter(
                scene_pool=scene_pool,
                **adapter_kwargs,
            )
        else:
            raise ValueError(
                f'未知的 adapter_type: {adapter_type}。'
                f'当前支持: vad, diffusiondrive'
            )
        return cls(adapter=adapter, debug=debug)

    def extract(
        self,
        planner_outputs: Dict[str, torch.Tensor],
        img_metas: Optional[List[Dict[str, Any]]] = None,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        hard_case_score: Optional[torch.Tensor] = None,
    ) -> PlanningInterface:
        """从规划器输出中提取 PlanningInterface。

        Args:
            planner_outputs: 规划器的 forward 输出字典
            img_metas: 图像元数据（保留用于兼容）
            ego_fut_cmd: 自车命令 [B, M] 或 [B]
            hard_case_score: 外部提供的 hard-case 分数 [B, 1]

        Returns:
            PlanningInterface 实例
        """
        interface = self.adapter.extract(
            planner_outputs=planner_outputs,
            ego_fut_cmd=ego_fut_cmd,
            hard_case_score=hard_case_score,
        )

        if self.debug:
            logger.info(
                f'PlanningInterface 提取完成:\n{interface.describe()}'
            )

        return interface
