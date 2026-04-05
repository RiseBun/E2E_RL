"""Refiner 训练器：支持监督预热和奖励加权精炼两阶段训练。

训练策略:
- Stage 1: 监督预热 - L1/SmoothL1 + ADE/FDE + residual 正则化
- Stage 2: 奖励加权精炼 - 结合 reward proxy 的加权损失
- Stage 3 (可选): Hard-case 过采样微调
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.losses import (
    compute_per_sample_reward_weighted_error,
    reward_weighted_refinement_loss,
    supervised_refinement_loss,
)
from E2E_RL.refinement.reward_proxy import compute_refinement_reward
from E2E_RL.hard_case.mining import HardCaseMiner
from E2E_RL.update_filter.scorer import UpdateReliabilityScorer

logger = logging.getLogger(__name__)


class InterfaceRefinerTrainer:
    """InterfaceRefiner 的两阶段训练器。

    Args:
        refiner: InterfaceRefiner 模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        device: 训练设备
        residual_reg_weight: residual 正则化权重
        reward_config: 奖励计算的参数字典
        hard_case_miner: hard case 挖掘器（可选）
        grad_clip: 梯度裁剪阈值
        update_filter: HarmfulUpdateFilter 实例（可选）
        update_scorer: UpdateReliabilityScorer 实例（可选）
    """

    def __init__(
        self,
        refiner: InterfaceRefiner,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device('cuda'),
        residual_reg_weight: float = 0.01,
        reward_config: Optional[Dict[str, float]] = None,
        hard_case_miner: Optional[HardCaseMiner] = None,
        grad_clip: float = 1.0,
        update_filter: Optional[Any] = None,
        update_scorer: Optional[UpdateReliabilityScorer] = None,
        scorer_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.refiner = refiner.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.residual_reg_weight = residual_reg_weight
        self.reward_config = reward_config or {}
        self.hard_case_miner = hard_case_miner
        self.grad_clip = grad_clip
        self.update_filter = update_filter
        self.update_scorer = update_scorer
        self.scorer_optimizer = scorer_optimizer

    # ------------------------------------------------------------------
    # Scorer 训练逻辑 (回归版)
    # ------------------------------------------------------------------
    def _train_scorer_step(
        self,
        interface: PlanningInterface,
        refiner_outputs: Dict[str, torch.Tensor],
        reward_info: Dict[str, torch.Tensor],
        ref_reward_info: Dict[str, torch.Tensor],
        heuristic_scores: torch.Tensor,
    ) -> float:
        """训练 ReliabilityNet 预测 Gain 和 Risk。"""
        if self.scorer_optimizer is None or self.update_scorer.model is None:
            return 0.0

        # 1. 构造 Gain Label: ΔR
        gain_label = (reward_info['total_reward'] - ref_reward_info['total_reward']).unsqueeze(-1)
        
        # 2. 构造 Risk Label: 分项风险回归 (collision/offroad/comfort/drift)
        delta_collision = (reward_info.get('collision_penalty', 0) - ref_reward_info.get('collision_penalty', 0)).clamp(min=0)
        delta_offroad = (reward_info.get('offroad_penalty', 0) - ref_reward_info.get('offroad_penalty', 0)).clamp(min=0)
        delta_comfort = (reward_info.get('comfort_penalty', 0) - ref_reward_info.get('comfort_penalty', 0)).clamp(min=0)
        # drift 近似用 comfort 差异替代（可扩展为更精细指标）
        delta_drift = delta_comfort

        risk_label_components = torch.stack([
            delta_collision,
            delta_offroad,
            delta_comfort,
            delta_drift,
        ], dim=-1)

        # 预测并计算损失 (MSE 回归)
        self.update_scorer.model.train()
        pred_gain, pred_risk_total, pred_risk_components = self.update_scorer.model(
            interface.scene_token,
            interface.reference_plan,
            refiner_outputs['residual'],
            plan_confidence=interface.plan_confidence,
            heuristic_scores=heuristic_scores
        )
        
        loss_gain = nn.functional.mse_loss(pred_gain, gain_label)
        loss_risk_total = nn.functional.mse_loss(pred_risk_total, (delta_collision + delta_offroad + delta_comfort + delta_drift).unsqueeze(-1))
        loss_risk_components = nn.functional.mse_loss(pred_risk_components, risk_label_components)
        loss = loss_gain + loss_risk_total + loss_risk_components
        
        self.scorer_optimizer.zero_grad()
        loss.backward()
        self.scorer_optimizer.step()
        
        return loss.item()

    # ------------------------------------------------------------------
    # Stage 1: 监督预热
    # ------------------------------------------------------------------
    def train_supervised_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """执行一个监督预热 epoch。

        每个 batch 应提供:
        - 'interface': PlanningInterface
        - 'gt_plan': [B, T, 2] GT 轨迹（绝对坐标）
        - 'plan_mask': [B, T] 有效步掩码（可选）

        Returns:
            epoch 级别的平均损失字典
        """
        self.refiner.train()
        epoch_metrics: Dict[str, float] = {
            'loss_total': 0.0,
            'loss_traj': 0.0,
            'loss_residual_reg': 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            interface: PlanningInterface = batch['interface'].to(self.device)
            gt_plan = batch['gt_plan'].to(self.device)
            plan_mask = batch.get('plan_mask')
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device)

            # 前向传播
            outputs = self.refiner(interface, plan_mask)
            refined_plan = outputs['refined_plan']
            residual_norm = outputs['residual_norm']

            # 监督损失
            loss_traj = supervised_refinement_loss(
                refined_plan, gt_plan, plan_mask
            )
            # residual 正则化
            loss_reg = residual_norm.mean() * self.residual_reg_weight

            loss_total = loss_traj + loss_reg

            # 反向传播
            self.optimizer.zero_grad()
            loss_total.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.refiner.parameters(), self.grad_clip
                )
            self.optimizer.step()

            epoch_metrics['loss_total'] += loss_total.item()
            epoch_metrics['loss_traj'] += loss_traj.item()
            epoch_metrics['loss_residual_reg'] += loss_reg.item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        # 计算平均值
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        logger.info(
            f'[Supervised Epoch {epoch}] '
            f'loss={epoch_metrics["loss_total"]:.4f} '
            f'traj={epoch_metrics["loss_traj"]:.4f} '
            f'reg={epoch_metrics["loss_residual_reg"]:.6f}'
        )
        return epoch_metrics

    # ------------------------------------------------------------------
    # Stage 2: 奖励加权精炼
    # ------------------------------------------------------------------
    def train_reward_weighted_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """执行一个奖励加权精炼 epoch。

        每个 batch 应额外提供:
        - 'agent_positions': [B, A, 2]（可选）
        - 'agent_future_trajs': [B, A, T, 2]（可选）
        - 'lane_boundaries': [B, N, P, 2]（可选）

        Returns:
            epoch 级别的平均损失和奖励字典
        """
        self.refiner.train()
        epoch_metrics: Dict[str, float] = {
            'loss_total': 0.0,
            'loss_reward_weighted': 0.0,
            'loss_residual_reg': 0.0,
            'mean_reward': 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            interface: PlanningInterface = batch['interface'].to(self.device)
            gt_plan = batch['gt_plan'].to(self.device)
            plan_mask = batch.get('plan_mask')
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device)

            # 可选的安全信息
            agent_positions = batch.get('agent_positions')
            agent_future_trajs = batch.get('agent_future_trajs')
            lane_boundaries = batch.get('lane_boundaries')
            if agent_positions is not None:
                agent_positions = agent_positions.to(self.device)
            if agent_future_trajs is not None:
                agent_future_trajs = agent_future_trajs.to(self.device)
            if lane_boundaries is not None:
                lane_boundaries = lane_boundaries.to(self.device)

            # 前向传播
            outputs = self.refiner(interface, plan_mask)
            refined_plan = outputs['refined_plan']
            residual_norm = outputs['residual_norm']

            # 计算奖励
            reward_info = compute_refinement_reward(
                refined_plan=refined_plan.detach(),
                gt_plan=gt_plan,
                mask=plan_mask,
                agent_positions=agent_positions,
                agent_future_trajs=agent_future_trajs,
                lane_boundaries=lane_boundaries,
                **self.reward_config,
            )
            total_reward = reward_info['total_reward']  # [B]

            # 计算参考轨迹的奖励作为 baseline (用于计算 Advantage 和训练 Scorer)
            with torch.no_grad():
                ref_reward_info = compute_refinement_reward(
                    refined_plan=interface.reference_plan,
                    gt_plan=gt_plan,
                    mask=plan_mask,
                    agent_positions=agent_positions,
                    agent_future_trajs=agent_future_trajs,
                    lane_boundaries=lane_boundaries,
                    **self.reward_config,
                )
                ref_reward = ref_reward_info['total_reward']

            # --- STAPO 启发: 训练可靠性评分器 (Learned Scorer - 回归版) ---
            if self.update_scorer is not None:
                # 获取启发式评分作为模型输入
                with torch.no_grad():
                    heuristic_dict = self.update_scorer.score_batch(interface, outputs)
                    h_scores = torch.stack([
                        heuristic_dict['uncertainty_score'], 
                        heuristic_dict['support_score'], 
                        heuristic_dict['drift_score']
                    ], dim=-1)

                scorer_loss = self._train_scorer_step(
                    interface, outputs, reward_info, ref_reward_info, h_scores
                )
                epoch_metrics['loss_scorer'] = epoch_metrics.get('loss_scorer', 0.0) + scorer_loss

            # --- STAPO 启发: 应用有害更新过滤 (Advantage Gating + Hard Guards) ---
            if self.update_filter is not None and self.update_scorer is not None:
                scores = self.update_scorer.score_batch(interface, outputs)
                loss_rw, huf_diag = self.update_filter.apply_filter(
                    per_sample_loss=compute_per_sample_reward_weighted_error(
                        refined_plan, gt_plan, total_reward, plan_mask
                    ),
                    scores=scores,
                    interface=interface,
                    refiner_outputs=outputs,
                    reward=total_reward,
                    ref_reward=ref_reward,
                    reward_info=reward_info,
                    ref_reward_info=ref_reward_info,
                )
            else:
                # 原始奖励加权损失
                loss_rw = reward_weighted_refinement_loss(
                    refined_plan, gt_plan, total_reward, plan_mask
                )

            loss_reg = residual_norm.mean() * self.residual_reg_weight
            loss_total = loss_rw + loss_reg

            self.optimizer.zero_grad()
            loss_total.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.refiner.parameters(), self.grad_clip
                )
            self.optimizer.step()

            epoch_metrics['loss_total'] += loss_total.item()
            epoch_metrics['loss_reward_weighted'] += loss_rw.item()
            epoch_metrics['loss_residual_reg'] += loss_reg.item()
            epoch_metrics['mean_reward'] += total_reward.mean().item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        logger.info(
            f'[Reward Epoch {epoch}] '
            f'loss={epoch_metrics["loss_total"]:.4f} '
            f'rw_loss={epoch_metrics["loss_reward_weighted"]:.4f} '
            f'reward={epoch_metrics["mean_reward"]:.4f}'
        )
        return epoch_metrics

    # ------------------------------------------------------------------
    # 通用工具方法
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str, epoch: int, extra: Optional[Dict] = None) -> None:
        """保存训练检查点。"""
        ckpt = {
            'epoch': epoch,
            'refiner_state_dict': self.refiner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        logger.info(f'Checkpoint 已保存到 {path}')

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载训练检查点。"""
        ckpt = torch.load(path, map_location=self.device)
        self.refiner.load_state_dict(ckpt['refiner_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(f'Checkpoint 已从 {path} 加载 (epoch={ckpt.get("epoch", "?")})')
        return ckpt

    # ------------------------------------------------------------------
    # Stage 2+: 带有害更新过滤的奖励加权精炼
    # ------------------------------------------------------------------
    def train_filtered_reward_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """执行一个带 HUF 过滤的奖励加权精炼 epoch。

        需要在构造时传入 update_filter 和 update_scorer。
        流程: forward → reward → scorer → filter → backward。

        Returns:
            epoch 级别的平均损失、奖励和 HUF 诊断字典
        """
        if self.update_filter is None or self.update_scorer is None:
            raise RuntimeError(
                'train_filtered_reward_epoch 需要 update_filter 和 update_scorer，'
                '请在构造 Trainer 时传入'
            )

        self.refiner.train()
        epoch_metrics: Dict[str, float] = {
            'loss_total': 0.0,
            'loss_filtered': 0.0,
            'loss_residual_reg': 0.0,
            'mean_reward': 0.0,
            'retention_ratio': 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            interface: PlanningInterface = batch['interface'].to(self.device)
            gt_plan = batch['gt_plan'].to(self.device)
            plan_mask = batch.get('plan_mask')
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device)

            agent_positions = batch.get('agent_positions')
            agent_future_trajs = batch.get('agent_future_trajs')
            lane_boundaries = batch.get('lane_boundaries')
            if agent_positions is not None:
                agent_positions = agent_positions.to(self.device)
            if agent_future_trajs is not None:
                agent_future_trajs = agent_future_trajs.to(self.device)
            if lane_boundaries is not None:
                lane_boundaries = lane_boundaries.to(self.device)

            # 前向传播
            outputs = self.refiner(interface, plan_mask)
            refined_plan = outputs['refined_plan']
            residual_norm = outputs['residual_norm']

            # 计算奖励
            reward_info = compute_refinement_reward(
                refined_plan=refined_plan.detach(),
                gt_plan=gt_plan,
                mask=plan_mask,
                agent_positions=agent_positions,
                agent_future_trajs=agent_future_trajs,
                lane_boundaries=lane_boundaries,
                **self.reward_config,
            )
            total_reward = reward_info['total_reward']

            with torch.no_grad():
                ref_reward_info = compute_refinement_reward(
                    refined_plan=interface.reference_plan,
                    gt_plan=gt_plan,
                    mask=plan_mask,
                    agent_positions=agent_positions,
                    agent_future_trajs=agent_future_trajs,
                    lane_boundaries=lane_boundaries,
                    **self.reward_config,
                )
                ref_reward = ref_reward_info['total_reward']

            # 计算 per-sample 加权误差
            per_sample_loss = compute_per_sample_reward_weighted_error(
                refined_plan, gt_plan, total_reward, plan_mask
            )

            # HUF: 评分 + 过滤（传入 reward 以便优势门控与 STAPO）
            scores = self.update_scorer.score_batch(interface, outputs)
            filtered_loss, diag = self.update_filter.apply_filter(
                per_sample_loss,
                scores,
                interface,
                outputs,
                reward=total_reward,
                ref_reward=ref_reward,
                reward_info=reward_info,
                ref_reward_info=ref_reward_info,
            )

            loss_reg = residual_norm.mean() * self.residual_reg_weight
            loss_total = filtered_loss + loss_reg

            self.optimizer.zero_grad()
            loss_total.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.refiner.parameters(), self.grad_clip
                )
            self.optimizer.step()

            epoch_metrics['loss_total'] += loss_total.item()
            epoch_metrics['loss_filtered'] += filtered_loss.item()
            epoch_metrics['loss_residual_reg'] += loss_reg.item()
            epoch_metrics['mean_reward'] += total_reward.mean().item()
            epoch_metrics['retention_ratio'] += diag.get('retention_ratio', 1.0)
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        logger.info(
            f'[Filtered Reward Epoch {epoch}] '
            f'loss={epoch_metrics["loss_total"]:.4f} '
            f'filtered={epoch_metrics["loss_filtered"]:.4f} '
            f'reward={epoch_metrics["mean_reward"]:.4f} '
            f'retention={epoch_metrics["retention_ratio"]:.2%}'
        )
        return epoch_metrics
