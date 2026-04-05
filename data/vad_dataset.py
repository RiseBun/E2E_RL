"""VAD 数据集和数据加载器，用于训练 Refiner。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from E2E_RL.planning_interface.adapters.vad_adapter import VADPlanningAdapter
from E2E_RL.planning_interface.interface import PlanningInterface


class VADDataset(Dataset):
    """VAD 推理数据数据集。

    从 manifest.json 和 .pt 文件加载数据，使用 VAD adapter 转换为 PlanningInterface。
    """

    def __init__(
        self,
        data_dir: str,
        manifest_file: str = 'manifest.json',
        adapter_config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / manifest_file

        # 加载 manifest
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.samples = self.manifest['samples']
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # 初始化 adapter
        adapter_config = adapter_config or {}
        self.adapter = VADPlanningAdapter(**adapter_config)

        # 缓存 GT 轨迹（如果有的话）
        self.gt_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取一个样本。

        Returns:
            {
                'interface': PlanningInterface,
                'gt_plan': torch.Tensor,  # [T, 2] 绝对坐标 GT 轨迹
                'plan_mask': torch.Tensor,  # [T] 有效步掩码
                'agent_positions': Optional[torch.Tensor],  # [A, 2]
                'agent_future_trajs': Optional[torch.Tensor],  # [A, T, 2]
                'lane_boundaries': Optional[torch.Tensor],  # [N, P, 2]
            }
        """
        sample_info = self.samples[idx]
        file_path = self.data_dir / sample_info['file']

        # 加载数据
        data = torch.load(file_path, map_location='cpu')

        # 使用 adapter 转换为 PlanningInterface
        interface = self._create_planning_interface(data)

        # 提取 GT 轨迹（从 metric_results 或其他地方）
        gt_plan = self._extract_gt_plan(data)

        # 提取其他信息用于奖励计算
        agent_positions = self._extract_agent_positions(data)
        agent_future_trajs = self._extract_agent_future_trajs(data)
        lane_boundaries = self._extract_lane_boundaries(data)

        # 计划掩码（假设所有步都有效）
        plan_mask = torch.ones(gt_plan.shape[0], dtype=torch.bool)

        return {
            'interface': interface,
            'gt_plan': gt_plan,
            'plan_mask': plan_mask,
            'agent_positions': agent_positions,
            'agent_future_trajs': agent_future_trajs,
            'lane_boundaries': lane_boundaries,
        }

    def _create_planning_interface(self, data: Dict[str, Any]) -> PlanningInterface:
        """使用 adapter 创建 PlanningInterface。"""
        # 提取 scene_token
        scene_token = self.adapter.extract_scene_token(data)

        # 提取 reference_plan
        reference_plan, plan_confidence = self.adapter.extract_reference_plan(data)

        # 提取 candidate_plans (如果有)
        candidate_plans, candidate_scores = self.adapter.extract_candidate_plans(data)

        # 提取 safety_features
        safety_features = self.adapter.extract_safety_features(data)

        # 构建 interface
        interface = PlanningInterface(
            scene_token=scene_token,
            reference_plan=reference_plan,
            plan_uncertainty=None,  # 可以后续计算
            safety_features=safety_features,
            plan_confidence=plan_confidence,
            candidate_plans=candidate_plans,
            candidate_scores=candidate_scores,
        )

        return interface

    def _extract_gt_plan(self, data: Dict[str, Any]) -> torch.Tensor:
        """提取 GT 轨迹。

        从 metric_results 中提取 GT 轨迹，如果没有则使用参考轨迹作为近似。
        """
        if 'metric_results' in data and 'gt_trajectory' in data['metric_results']:
            gt_traj = torch.tensor(data['metric_results']['gt_trajectory'])
            if gt_traj.dim() == 2 and gt_traj.shape[1] == 2:
                return gt_traj

        # 回退：使用 ego_fut_trajs 作为近似 GT
        if 'ego_fut_trajs' in data:
            return data['ego_fut_trajs']

        # 最后回退：使用参考轨迹
        ref_plan, _ = self.adapter.extract_reference_plan(data)
        return ref_plan

    def _extract_agent_positions(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """提取其他 agent 当前位置。"""
        # 从 all_bbox_preds 或类似字段提取
        # 这里简化处理，返回 None
        return None

    def _extract_agent_future_trajs(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """提取其他 agent 未来轨迹。"""
        # 从 all_traj_preds 或类似字段提取
        # 这里简化处理，返回 None
        return None

    def _extract_lane_boundaries(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """提取车道边界。"""
        # 从地图相关字段提取
        # 这里简化处理，返回 None
        return None


def create_vad_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    adapter_config: Optional[Dict[str, Any]] = None,
) -> torch.utils.data.DataLoader:
    """创建 VAD 数据加载器。"""
    dataset = VADDataset(
        data_dir=data_dir,
        max_samples=max_samples,
        adapter_config=adapter_config,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=vad_collate_fn,
    )

    return dataloader


def vad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """VAD 数据批处理函数。"""
    # 提取各个字段
    interfaces = [item['interface'] for item in batch]
    gt_plans = [item['gt_plan'] for item in batch]
    plan_masks = [item['plan_mask'] for item in batch]
    agent_positions = [item['agent_positions'] for item in batch]
    agent_future_trajs = [item['agent_future_trajs'] for item in batch]
    lane_boundaries = [item['lane_boundaries'] for item in batch]

    # 批处理 PlanningInterface
    batched_interface = PlanningInterface.collate(interfaces)

    # 批处理其他张量
    gt_plans = torch.stack(gt_plans)
    plan_masks = torch.stack(plan_masks)

    # 对于可选字段，如果都是 None 则保持 None
    if all(x is None for x in agent_positions):
        agent_positions = None
    else:
        agent_positions = torch.stack([x for x in agent_positions if x is not None])

    if all(x is None for x in agent_future_trajs):
        agent_future_trajs = None
    else:
        agent_future_trajs = torch.stack([x for x in agent_future_trajs if x is not None])

    if all(x is None for x in lane_boundaries):
        lane_boundaries = None
    else:
        lane_boundaries = torch.stack([x for x in lane_boundaries if x is not None])

    return {
        'interface': batched_interface,
        'gt_plan': gt_plans,
        'plan_mask': plan_masks,
        'agent_positions': agent_positions,
        'agent_future_trajs': agent_future_trajs,
        'lane_boundaries': lane_boundaries,
    }