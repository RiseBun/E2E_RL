#!/usr/bin/env python
"""收集 ReliabilityNet 训练数据。

从训练好的 refiner 生成 residual update 样本，记录:
- scene_token, reference_plan, residual
- delta_safe_reward (分项)
- heuristic scores
- candidate augmentations (zero, random, GT-directed, safety-biased)
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.data.dataloader import build_vad_dataloader
from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.reward_proxy import compute_refinement_reward
from E2E_RL.update_filter.scorer import UpdateReliabilityScorer
from E2E_RL.update_filter.config import HUFConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_refiner_checkpoint(refiner: InterfaceRefiner, path: str, device: torch.device) -> None:
    """加载 refiner 检查点。"""
    ckpt = torch.load(path, map_location=device)
    refiner.load_state_dict(ckpt['refiner_state_dict'])
    logger.info(f'Refiner checkpoint loaded from {path}')


def generate_candidate_residuals(
    reference_plan: torch.Tensor,
    residual: torch.Tensor,
    gt_plan: torch.Tensor,
    num_random: int = 2,
) -> List[Dict[str, torch.Tensor]]:
    """生成单样本候选 residual 用于数据增强。

    Args:
        reference_plan: [T, 2]
        residual: [T, 2]
        gt_plan: [T, 2]
        num_random: 随机候选数量

    Returns:
        List of candidate dicts, each containing 'residual' and 'type'
    """
    device = reference_plan.device
    candidates = []

    # 1. Zero residual (baseline)
    zero_residual = torch.zeros_like(residual)
    candidates.append({'residual': zero_residual, 'type': 'zero'})

    # 2. Current refiner output
    candidates.append({'residual': residual, 'type': 'current'})

    # 3. Bounded random residuals
    for _ in range(num_random):
        random_residual = torch.randn_like(residual) * 0.5
        random_residual = torch.clamp(random_residual, -2.0, 2.0)
        candidates.append({'residual': random_residual, 'type': 'random'})

    # 4. GT-directed residuals (towards ground truth)
    gt_residual = gt_plan - reference_plan
    gt_residual = torch.clamp(gt_residual, -3.0, 3.0)
    candidates.append({'residual': gt_residual, 'type': 'gt_directed'})

    # 5. Safety-biased residuals (small conservative adjustments)
    safety_residual = torch.randn_like(residual) * 0.1
    safety_residual = torch.clamp(safety_residual, -0.5, 0.5)
    candidates.append({'residual': safety_residual, 'type': 'safety_biased'})

    return candidates


def collect_batch_data(
    interface: PlanningInterface,
    refiner_outputs: Dict[str, torch.Tensor],
    gt_plan: torch.Tensor,
    reward_config: Dict[str, float],
    update_scorer: UpdateReliabilityScorer,
    agent_positions: torch.Tensor = None,
    agent_future_trajs: torch.Tensor = None,
    lane_boundaries: torch.Tensor = None,
    plan_mask: torch.Tensor = None,
) -> List[Dict[str, Any]]:
    """收集一个 batch 的 scorer 训练数据。"""
    batch_data = []
    B = interface.reference_plan.shape[0]

    for i in range(B):
        sample_scene = interface.scene_token[i].unsqueeze(0)
        sample_ref_plan = interface.reference_plan[i].unsqueeze(0)
        sample_candidate_plans = interface.candidate_plans[i].unsqueeze(0) if interface.candidate_plans is not None else None
        sample_plan_confidence = interface.plan_confidence[i].unsqueeze(0) if interface.plan_confidence is not None else None
        sample_safety_features = {k: v[i].unsqueeze(0) for k, v in interface.safety_features.items()} if interface.safety_features is not None else None
        sample_hard_case_score = interface.hard_case_score[i].unsqueeze(0) if interface.hard_case_score is not None else None

        sample_interface = PlanningInterface(
            scene_token=sample_scene,
            reference_plan=sample_ref_plan,
            candidate_plans=sample_candidate_plans,
            plan_confidence=sample_plan_confidence,
            safety_features=sample_safety_features,
            hard_case_score=sample_hard_case_score,
            metadata=interface.metadata if interface.metadata else {},
        )

        sample_residual = refiner_outputs['residual'][i].unsqueeze(0)
        sample_gt_plan = gt_plan[i].unsqueeze(0)
        sample_mask = plan_mask[i].unsqueeze(0) if plan_mask is not None else None

        # 生成单样本候选 residual
        candidates = generate_candidate_residuals(
            sample_interface.reference_plan.squeeze(0),
            sample_residual.squeeze(0),
            sample_gt_plan.squeeze(0),
            num_random=2,
        )

        # 参考轨迹奖励
        ref_reward_info = compute_refinement_reward(
            refined_plan=sample_interface.reference_plan,
            gt_plan=sample_gt_plan,
            mask=sample_mask,
            agent_positions=agent_positions[i].unsqueeze(0) if agent_positions is not None else None,
            agent_future_trajs=agent_future_trajs[i].unsqueeze(0) if agent_future_trajs is not None else None,
            lane_boundaries=lane_boundaries[i].unsqueeze(0) if lane_boundaries is not None else None,
            **reward_config,
        )

        # squeeze 单样本结果
        ref_reward_info = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in ref_reward_info.items()}

        for candidate in candidates:
            residual = candidate['residual']
            residual_type = candidate['type']
            refined_plan = sample_interface.reference_plan + residual.unsqueeze(0)

            reward_info = compute_refinement_reward(
                refined_plan=refined_plan.detach(),
                gt_plan=sample_gt_plan,
                mask=sample_mask,
                agent_positions=agent_positions[i].unsqueeze(0) if agent_positions is not None else None,
                agent_future_trajs=agent_future_trajs[i].unsqueeze(0) if agent_future_trajs is not None else None,
                lane_boundaries=lane_boundaries[i].unsqueeze(0) if lane_boundaries is not None else None,
                **reward_config,
            )
            reward_info = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in reward_info.items()}

            heuristic_dict = update_scorer.score_batch(
                sample_interface,
                {
                    'residual': residual.unsqueeze(0),
                    'residual_norm': torch.norm(residual, dim=-1).max(dim=-1).values.unsqueeze(-1),
                    'refined_plan': refined_plan,
                }
            )

            delta_progress = (
                reward_info.get('progress_reward', reward_info['total_reward']) -
                ref_reward_info.get('progress_reward', ref_reward_info['total_reward'])
            )
            delta_collision = (
                reward_info.get('collision_penalty', torch.tensor(0.0, device=refined_plan.device)) -
                ref_reward_info.get('collision_penalty', torch.tensor(0.0, device=refined_plan.device))
            ).clamp(min=0)
            delta_offroad = (
                reward_info.get('offroad_penalty', torch.tensor(0.0, device=refined_plan.device)) -
                ref_reward_info.get('offroad_penalty', torch.tensor(0.0, device=refined_plan.device))
            ).clamp(min=0)
            delta_comfort = (
                reward_info.get('comfort_penalty', torch.tensor(0.0, device=refined_plan.device)) -
                ref_reward_info.get('comfort_penalty', torch.tensor(0.0, device=refined_plan.device))
            ).clamp(min=0)
            delta_drift = delta_comfort

            delta_safe_reward = (
                delta_progress
                - 2.0 * delta_collision
                - 1.0 * delta_offroad
                - 0.5 * delta_comfort
                - 1.0 * delta_drift
            )

            data_point = {
                'scene_token': sample_interface.scene_token.squeeze(0).cpu(),
                'reference_plan': sample_interface.reference_plan.squeeze(0).cpu(),
                'residual': residual.cpu(),
                'residual_type': residual_type,
                'plan_confidence': sample_interface.plan_confidence.squeeze(0).cpu() if sample_interface.plan_confidence is not None else None,
                'safety_features': {k: v.squeeze(0).cpu() for k, v in sample_interface.safety_features.items()} if sample_interface.safety_features is not None else None,
                'heuristic_scores': torch.stack([
                    heuristic_dict['uncertainty_score'].view(-1),
                    heuristic_dict['support_score'].view(-1),
                    heuristic_dict['drift_score'].view(-1)
                ], dim=-1).squeeze(0).cpu(),
                'delta_safe_reward': delta_safe_reward.cpu(),
                'delta_progress': delta_progress.cpu(),
                'delta_collision': delta_collision.cpu(),
                'delta_offroad': delta_offroad.cpu(),
                'delta_comfort': delta_comfort.cpu(),
                'delta_drift': delta_drift.cpu(),
                'total_reward': reward_info['total_reward'].cpu(),
                'ref_total_reward': ref_reward_info['total_reward'].cpu(),
            }

            batch_data.append(data_point)

    return batch_data


def main():
    parser = argparse.ArgumentParser(description='收集 ReliabilityNet 训练数据')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--refiner_checkpoint', type=str, required=True, help='Refiner 检查点路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出数据文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--max_samples', type=int, default=10000, help='最大收集样本数')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 构建 refiner
    model_cfg = cfg['model']
    refiner = InterfaceRefiner(
        scene_dim=model_cfg['scene_dim'],
        plan_len=model_cfg['plan_len'],
        hidden_dim=model_cfg['hidden_dim'],
        dropout=model_cfg.get('dropout', 0.1),
        output_norm=model_cfg.get('output_norm', False),
    ).to(device)
    load_refiner_checkpoint(refiner, args.refiner_checkpoint, device)
    refiner.eval()

    # 构建 scorer
    huf_cfg = cfg.get('huf', {})
    update_scorer = UpdateReliabilityScorer(config=HUFConfig(**huf_cfg))

    reward_cfg = cfg['training'].get('reward', {})

    # 构建 DataLoader
    data_cfg = cfg.get('data', {})
    data_dir = data_cfg.get('data_dir', 'E2E_RL/data/vad_dumps')
    dataloader = build_vad_dataloader(
        data_dir=data_dir,
        batch_size=data_cfg.get('batch_size', 4),
        num_workers=data_cfg.get('num_workers', 0),
        shuffle=False,
    )
    logger.info(f'开始收集 scorer 训练数据，data_dir={data_dir}，batch_size={data_cfg.get("batch_size", 4)}')

    all_data = []
    collected_samples = 0

    for batch in dataloader:
        if collected_samples >= args.max_samples:
            break

        interface = batch['interface'].to(device)
        gt_plan = batch['gt_plan'].to(device)
        plan_mask = batch.get('plan_mask')
        if plan_mask is not None:
            plan_mask = plan_mask.to(device)

        with torch.no_grad():
            refiner_outputs = refiner(interface, plan_mask)

        batch_data = collect_batch_data(
            interface,
            refiner_outputs,
            gt_plan,
            reward_cfg,
            update_scorer,
            agent_positions=None,
            agent_future_trajs=None,
            lane_boundaries=None,
            plan_mask=plan_mask,
        )

        if collected_samples + len(batch_data) > args.max_samples:
            batch_data = batch_data[: args.max_samples - collected_samples]

        all_data.extend(batch_data)
        collected_samples += len(batch_data)
        logger.info(f'已收集 {collected_samples} 个样本')

        if collected_samples >= args.max_samples:
            break

    # 保存数据
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(all_data, f)
    logger.info(f'数据已保存到 {args.output_file}，共 {len(all_data)} 个样本')

    # 打印数据统计
    types = {}
    for item in all_data:
        t = item['residual_type']
        types[t] = types.get(t, 0) + 1
    logger.info(f'样本类型分布: {types}')


if __name__ == '__main__':
    main()