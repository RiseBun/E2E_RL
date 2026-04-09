"""VADv2 专用适配器：将 VADv2 forward 输出映射到 PlanningInterface。

VADv2 是 VAD 的改进版本（ICLR 2026），引入概率规划。
输出格式与 VAD 基本相同，但可能包含额外的不确定性建模字段。

VADv2 输出字典的关键字段:
- bev_embed: [B, bev_h*bev_w, embed_dims]  (200*200=40000, 256)
- ego_fut_preds: [B, ego_fut_mode, fut_ts, 2]  (B, 3, 6, 2) 位移增量
- all_cls_scores: [num_dec, B, num_query, num_cls]  检测分类分数
- all_bbox_preds: [num_dec, B, num_query, 10]  检测框
- all_traj_preds: [num_dec, B, A, fut_mode*fut_ts*2]  Agent 轨迹
- all_traj_cls_scores: [num_dec, B, A, fut_mode]  Agent 轨迹模式分数
- map_all_cls_scores: [num_dec, B, map_num_vec, 3]  地图分类分数
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base_adapter import BasePlanningAdapter


class VADv2PlanningAdapter(BasePlanningAdapter):
    """VADv2 → PlanningInterface 适配器。

    VADv2 与 VAD 的输出格式基本相同，因此直接复用 VADPlanningAdapter 的逻辑。
    如果 VADv2 有额外的字段（如概率分布参数），可以在这里扩展。

    Args:
        scene_pool: BEV 特征池化方式，'mean' / 'max' / 'grid' / 'ego_local'
        ego_fut_mode: 自车规划模式数，默认 3
        fut_ts: 未来时间步长数，默认 6
        grid_size: grid 池化的分块数（每个方向），默认 4
        ego_local_k: ego_local 池化的邻域大小，默认 16
    """

    def __init__(
        self,
        scene_pool: str = 'mean',
        ego_fut_mode: int = 3,
        fut_ts: int = 6,
        grid_size: int = 4,
        ego_local_k: int = 16,
    ):
        # 直接复用 VAD 的逻辑
        from .vad_adapter import VADPlanningAdapter
        
        self.vad_adapter = VADPlanningAdapter(
            scene_pool=scene_pool,
            ego_fut_mode=ego_fut_mode,
            fut_ts=fut_ts,
            grid_size=grid_size,
            ego_local_k=ego_local_k,
        )
        self.scene_pool = scene_pool
        self.ego_fut_mode = ego_fut_mode
        self.fut_ts = fut_ts
        self.grid_size = grid_size
        self.ego_local_k = ego_local_k

    def extract_scene_token(self, planner_outputs: Dict[str, Any]) -> torch.Tensor:
        """从 BEV 特征池化得到场景 token。"""
        return self.vad_adapter.extract_scene_token(planner_outputs)

    def extract_reference_plan(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """提取参考轨迹。"""
        return self.vad_adapter.extract_reference_plan(planner_outputs, ego_fut_cmd)

    def extract_plan_confidence(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """提取规划置信度。"""
        return self.vad_adapter.extract_plan_confidence(planner_outputs, ego_fut_cmd)

    def extract_safety_features(
        self,
        planner_outputs: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """提取安全相关特征。"""
        return self.vad_adapter.extract_safety_features(planner_outputs)
