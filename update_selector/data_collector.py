"""
UpdateEvaluatorDataCollector — 训练数据收集器

核心思想：
训练 UpdateEvaluator 需要大量 (s, a, ref, corrected) → (gain, risk) 的样本。
这些样本通过在训练数据上 rollout 不同类型的 candidate corrections 自动生成。

训练标签自动从 reward_proxy 生成：
- gain_label = R(corrected) - R(reference)
- collision_label = collision(corrected) - collision(reference)
- offroad_label = offroad(corrected) - offroad(reference)
- comfort_label = comfort(corrected) - comfort(reference)
- drift_label = drift(corrected, reference)

采样策略（已调整）：
- zero: 10%
- deterministic mean: 15%
- policy_sample: 25%
- gt_directed: 30%
- bounded_random: 10%
- safety_biased: 10%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.refinement.reward_proxy import compute_refinement_reward
from E2E_RL.update_selector.candidate_generator import (
    CandidateCorrector,
    CandidateStats,
    compute_structured_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorDataSample:
    """UpdateEvaluator 的单个训练样本。"""
    scene_token: torch.Tensor  # [D]
    reference_plan: torch.Tensor  # [T, 2]
    correction: torch.Tensor  # [T, 2]
    corrected_plan: torch.Tensor  # [T, 2]
    correction_type: str  # 候选类型名称

    # Labels
    gain: torch.Tensor  # scalar
    collision_delta: torch.Tensor  # scalar
    offroad_delta: torch.Tensor  # scalar
    comfort_delta: torch.Tensor  # scalar
    drift: torch.Tensor  # scalar

    # Structured stats
    residual_norm: torch.Tensor
    max_step_disp: torch.Tensor
    curvature_change: torch.Tensor
    jerk_change: torch.Tensor
    total_disp: torch.Tensor
    speed_max: torch.Tensor
    support_score: torch.Tensor
    drift_score: torch.Tensor

    # Optional
    plan_confidence: Optional[torch.Tensor] = None


class UpdateEvaluatorDataset(Dataset):
    """UpdateEvaluator 训练数据集。

    存储预先收集的 (s, a, ref, corrected) → labels 样本。
    """

    def __init__(self):
        self.samples: List[EvaluatorDataSample] = []

    def add_sample(self, sample: EvaluatorDataSample):
        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            'scene_token': s.scene_token,
            'reference_plan': s.reference_plan.flatten(),  # [T*2]
            'correction': s.correction.flatten(),          # [T*2]
            'corrected_plan': s.corrected_plan,
            'gain': s.gain,
            'collision_delta': s.collision_delta,
            'offroad_delta': s.offroad_delta,
            'comfort_delta': s.comfort_delta,
            'drift': s.drift,
            'residual_norm': s.residual_norm,
            'max_step_disp': s.max_step_disp,
            'curvature_change': s.curvature_change,
            'jerk_change': s.jerk_change,
            'total_disp': s.total_disp,
            'speed_max': s.speed_max,
            'support_score': s.support_score,
            'drift_score': s.drift_score,
            'plan_confidence': s.plan_confidence,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        # 需要 flatten 的键
        flatten_keys = ['reference_plan', 'correction', 'corrected_plan']

        keys = batch[0].keys()
        result = {}
        for key in keys:
            tensors = [item[key] for item in batch]
            if all(t is None for t in tensors):
                result[key] = None
            elif key in flatten_keys:
                # flatten 后 stack: [B, T, 2] -> [B, T*2]
                stacked = torch.stack([t.flatten() for t in tensors if t is not None])
                result[key] = stacked
            else:
                result[key] = torch.stack([t for t in tensors if t is not None])
        return result


class UpdateEvaluatorDataCollector:
    """数据收集器：在训练数据上收集 UpdateEvaluator 的训练样本。

    对每个数据样本，生成多种类型的 candidate corrections，
    并计算对应的 reward labels。

    支持：
    - 加权采样（推荐）：使用 generate_weighted() 确保分布平衡
    - 统计收集：自动收集每种候选类型的质量统计
    """

    def __init__(
        self,
        base_dataloader: DataLoader,
        candidate_generator: CandidateCorrector,
        reward_config: Optional[Dict[str, Any]] = None,
        device: torch.device = torch.device('cpu'),
        use_weighted: bool = True,
        n_samples_per_batch: int = 6,
        collect_stats: bool = True,
    ):
        self.base_dataloader = base_dataloader
        self.candidate_generator = candidate_generator
        self.reward_config = reward_config or {}
        self.device = device
        self.use_weighted = use_weighted
        self.n_samples_per_batch = n_samples_per_batch
        self.collect_stats = collect_stats
        self.candidate_stats = CandidateStats()

    def collect(
        self,
        n_batches: Optional[int] = None,
    ) -> UpdateEvaluatorDataset:
        """收集训练数据。

        Args:
            n_batches: 收集多少个 batch 的数据（None 表示全部）

        Returns:
            UpdateEvaluatorDataset
        """
        dataset = UpdateEvaluatorDataset()
        self.candidate_stats.reset()
        n_collected = 0
        batch_count = 0

        logger.info(f"开始收集数据，n_batches={n_batches}")

        for batch in self.base_dataloader:
            if n_batches is not None and batch_count >= n_batches:
                break

            batch_count += 1
            if batch_count % 10 == 0:
                logger.info(f"数据收集进度: {batch_count}/{n_batches} batches")

            interface = batch['interface'].to(self.device)
            gt_plan = batch['gt_plan'].to(self.device)
            plan_mask = batch.get('plan_mask')
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device)

            # 生成候选修正（使用加权采样）
            if self.use_weighted:
                all_corrections, types = self.candidate_generator.generate_weighted(
                    interface, gt_plan, n_samples=self.n_samples_per_batch
                )
            else:
                all_corrections, types = self.candidate_generator.generate_all_types(
                    interface, gt_plan
                )

            B, N_corr, T, _ = all_corrections.shape

            # 对每个候选计算 labels
            for b in range(B):
                for n in range(N_corr):
                    correction = all_corrections[b, n]  # [T, 2]
                    corr_type = types[n] if n < len(types) else 'unknown'
                    corrected_plan = interface.reference_plan[b] + correction

                    # 计算 structured stats
                    stats = compute_structured_stats(
                        correction.unsqueeze(0),
                        interface.reference_plan[b:b+1],
                        corrected_plan.unsqueeze(0),
                        dt=self.reward_config.get('dt', 0.5),
                    )

                    # 计算 reward labels
                    reward_info = compute_refinement_reward(
                        refined_plan=corrected_plan.unsqueeze(0),
                        gt_plan=gt_plan[b:b+1],
                        mask=plan_mask[b:b+1] if plan_mask is not None else None,
                        **self.reward_config,
                    )

                    # 参考轨迹的 reward
                    with torch.no_grad():
                        ref_reward_info = compute_refinement_reward(
                            refined_plan=interface.reference_plan[b:b+1],
                            gt_plan=gt_plan[b:b+1],
                            mask=plan_mask[b:b+1] if plan_mask is not None else None,
                            **self.reward_config,
                        )

                    # Labels
                    gain = reward_info['total_reward'] - ref_reward_info['total_reward']
                    collision_delta = (
                        reward_info['collision_penalty'] - ref_reward_info['collision_penalty']
                    ).clamp(min=0)
                    offroad_delta = (
                        reward_info['offroad_penalty'] - ref_reward_info['offroad_penalty']
                    ).clamp(min=0)
                    comfort_delta = (
                        reward_info['comfort_penalty'] - ref_reward_info['comfort_penalty']
                    ).clamp(min=0)

                    # Drift label
                    drift = stats['drift_score']

                    # 收集统计
                    if self.collect_stats:
                        self.candidate_stats.add_sample(corr_type, gain.item())

                    # 构建样本
                    sample = EvaluatorDataSample(
                        scene_token=interface.scene_token[b],
                        reference_plan=interface.reference_plan[b],  # [T, 2]
                        correction=correction,                       # [T, 2]
                        corrected_plan=corrected_plan,
                        correction_type=corr_type,                   # 类型名称
                        gain=gain.squeeze(),
                        collision_delta=collision_delta.squeeze(),
                        offroad_delta=offroad_delta.squeeze(),
                        comfort_delta=comfort_delta.squeeze(),
                        drift=drift.squeeze(),
                        residual_norm=stats['residual_norm'].squeeze(),
                        max_step_disp=stats['max_step_disp'].squeeze(),
                        curvature_change=stats['curvature_change'].squeeze(),
                        jerk_change=stats['jerk_change'].squeeze(),
                        total_disp=stats['total_disp'].squeeze(),
                        speed_max=stats['speed_max'].squeeze(),
                        support_score=stats['support_score'].squeeze(),
                        drift_score=stats['drift_score'].squeeze(),
                        plan_confidence=(
                            interface.plan_confidence[b] if interface.plan_confidence is not None else None
                        ),
                    )

                    dataset.add_sample(sample)
                    n_collected += 1

            batch_count += 1
            if batch_count % 10 == 0:
                logger.info(f'Collected {n_collected} samples from {batch_count} batches')

        logger.info(f'Data collection complete: {n_collected} total samples')

        # 打印统计报告
        if self.collect_stats:
            self.candidate_stats.print_report()

        return dataset

    def get_stats(self) -> CandidateStats:
        """获取候选质量统计。"""
        return self.candidate_stats

    def collect_and_save(
        self,
        output_path: str,
        n_batches: Optional[int] = None,
    ):
        """收集数据并保存到文件。"""
        dataset = self.collect(n_batches)

        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        samples_list = []
        for s in dataset.samples:
            samples_list.append({
                'scene_token': s.scene_token.numpy(),
                'reference_plan': s.reference_plan.numpy(),
                'correction': s.correction.numpy(),
                'corrected_plan': s.corrected_plan.numpy(),
                'gain': s.gain.item(),
                'collision_delta': s.collision_delta.item(),
                'offroad_delta': s.offroad_delta.item(),
                'comfort_delta': s.comfort_delta.item(),
                'drift': s.drift.item(),
                'residual_norm': s.residual_norm.item(),
                'max_step_disp': s.max_step_disp.item(),
                'curvature_change': s.curvature_change.item(),
                'jerk_change': s.jerk_change.item(),
                'total_disp': s.total_disp.item(),
                'speed_max': s.speed_max.item(),
                'support_score': s.support_score.item(),
                'drift_score': s.drift_score.item(),
            })

        torch.save({'samples': samples_list}, output_path)
        logger.info(f'Dataset saved to {output_path} ({len(samples_list)} samples)')
