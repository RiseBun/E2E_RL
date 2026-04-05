"""
UpdateEvaluatorTrainer — 多头回归训练器

训练目标：
L_total = α * L_gain + β * L_risk

其中：
- L_gain = SmoothL1(pred_gain, y_gain)
- L_risk = Σ_k SmoothL1(pred_risk_k, y_risk_k)

关键设计：
- 不做二分类，做连续回归
- 保留分项 risk 标签，便于 debug
- 可学习 Risk 组合权重
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader

from E2E_RL.update_selector.update_evaluator import UpdateEvaluator, UpdateEvaluatorConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorTrainingConfig:
    """Evaluator 训练配置。"""
    # 损失权重
    alpha_gain: float = 1.0  # Gain loss 权重
    beta_risk: float = 1.0  # Risk loss 权重

    # 优化器
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # 梯度裁剪
    grad_clip: float = 1.0

    # 训练 epochs
    epochs: int = 10

    # 评估频率
    eval_every: int = 5

    # 风险归一化因子（从数据估计）
    collision_norm: float = 1.0
    offroad_norm: float = 1.0
    comfort_norm: float = 1.0
    drift_norm: float = 1.0

    # 过滤阈值（用于评估时的排序质量）
    gain_threshold: float = 0.0  # 正 gain 判定阈值


class UpdateEvaluatorTrainer:
    """UpdateEvaluator 训练器。"""

    def __init__(
        self,
        evaluator: UpdateEvaluator,
        config: EvaluatorTrainingConfig,
        device: torch.device = torch.device('cpu'),
    ):
        self.evaluator = evaluator.to(device)
        self.cfg = config
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            evaluator.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

        # 损失函数
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """计算损失。

        Args:
            batch: 包含输入和标签的字典

        Returns:
            dict with:
                - total_loss
                - loss_gain
                - loss_collision
                - loss_offroad
                - loss_comfort
                - loss_drift
        """
        # 准备输入
        scene_token = batch['scene_token'].to(self.device)
        reference_plan = batch['reference_plan'].to(self.device)  # [B, T*2]
        correction = batch['correction'].to(self.device)          # [B, T*2]
        plan_confidence = batch.get('plan_confidence')
        if plan_confidence is not None:
            plan_confidence = plan_confidence.to(self.device)

        # 结构化统计量
        structured_stats = torch.stack([
            batch['residual_norm'],
            batch['max_step_disp'],
            batch['curvature_change'],
            batch['jerk_change'],
            batch['total_disp'],
            batch['speed_max'],
            batch['support_score'],
            batch['drift_score'],
        ], dim=-1).to(self.device)

        # 前向传播
        output = self.evaluator(
            scene_token=scene_token,
            reference_plan=reference_plan,
            correction=correction,
            plan_confidence=plan_confidence,
            structured_stats=structured_stats,
        )

        # Labels
        y_gain = batch['gain'].to(self.device)
        y_collision = batch['collision_delta'].to(self.device)
        y_offroad = batch['offroad_delta'].to(self.device)
        y_comfort = batch['comfort_delta'].to(self.device)
        y_drift = batch['drift'].to(self.device)

        # 损失
        loss_gain = self.loss_fn(output['pred_gain'].squeeze(-1), y_gain).mean()
        loss_collision = self.loss_fn(output['pred_collision'].squeeze(-1), y_collision).mean()
        loss_offroad = self.loss_fn(output['pred_offroad'].squeeze(-1), y_offroad).mean()
        loss_comfort = self.loss_fn(output['pred_comfort'].squeeze(-1), y_comfort).mean()
        loss_drift = self.loss_fn(output['pred_drift'].squeeze(-1), y_drift).mean()

        # 总损失
        loss_risk = loss_collision + loss_offroad + loss_comfort + loss_drift
        total_loss = self.cfg.alpha_gain * loss_gain + self.cfg.beta_risk * loss_risk

        return {
            'total_loss': total_loss,
            'loss_gain': loss_gain,
            'loss_collision': loss_collision,
            'loss_offroad': loss_offroad,
            'loss_comfort': loss_comfort,
            'loss_drift': loss_drift,
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """训练一个 epoch。"""
        self.evaluator.train()
        epoch_losses = {
            'total_loss': 0.0,
            'loss_gain': 0.0,
            'loss_collision': 0.0,
            'loss_offroad': 0.0,
            'loss_comfort': 0.0,
            'loss_drift': 0.0,
        }
        n_batches = 0

        for batch in dataloader:
            losses = self.compute_loss(batch)

            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            nn.utils.clip_grad_norm_(self.evaluator.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            n_batches += 1

        # 平均
        for key in epoch_losses:
            epoch_losses[key] /= max(n_batches, 1)

        self.scheduler.step()

        logger.info(
            f'[Evaluator Epoch {epoch}] '
            f'total={epoch_losses["total_loss"]:.4f} '
            f'gain={epoch_losses["loss_gain"]:.4f} '
            f'collision={epoch_losses["loss_collision"]:.4f} '
            f'offroad={epoch_losses["loss_offroad"]:.4f} '
            f'comfort={epoch_losses["loss_comfort"]:.4f} '
            f'drift={epoch_losses["loss_drift"]:.4f}'
        )

        return epoch_losses

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """评估。"""
        self.evaluator.eval()
        total_loss = 0.0
        total_gain_loss = 0.0
        total_risk_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            losses = self.compute_loss(batch)
            total_loss += losses['total_loss'].item()
            total_gain_loss += losses['loss_gain'].item()
            total_risk_loss += (
                losses['loss_collision'].item()
                + losses['loss_offroad'].item()
                + losses['loss_comfort'].item()
                + losses['loss_drift'].item()
            )
            n_batches += 1

        n_batches = max(n_batches, 1)

        return {
            'total_loss': total_loss / n_batches,
            'loss_gain': total_gain_loss / n_batches,
            'loss_risk': total_risk_loss / n_batches,
        }

    @torch.no_grad()
    def evaluate_with_ranking_metrics(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """评估并计算排序质量指标。

        计算以下指标：
        - Spearman gain: pred_gain 与 y_gain 的相关性
        - Kendall tau: pred_gain 与 y_gain 的 Kendall tau
        - Spearman risk: pred_total_risk 与真实 risk 的相关性
        - Retained/Filtered gain 均值: 根据阈值划分 retained vs filtered
        - Retained/Filtered risk 均值

        Returns:
            dict 包含所有排序质量指标
        """
        self.evaluator.eval()

        # 收集所有数据用于排序分析
        all_pred_gain = []
        all_pred_risk = []
        all_y_gain = []
        all_y_risk = []

        all_collision = []
        all_offroad = []
        all_comfort = []
        all_drift = []

        for batch in dataloader:
            # 准备输入
            scene_token = batch['scene_token'].to(self.device)
            reference_plan = batch['reference_plan'].to(self.device)
            correction = batch['correction'].to(self.device)
            plan_confidence = batch.get('plan_confidence')
            if plan_confidence is not None:
                plan_confidence = plan_confidence.to(self.device)

            structured_stats = torch.stack([
                batch['residual_norm'],
                batch['max_step_disp'],
                batch['curvature_change'],
                batch['jerk_change'],
                batch['total_disp'],
                batch['speed_max'],
                batch['support_score'],
                batch['drift_score'],
            ], dim=-1).to(self.device)

            # 前向传播
            output = self.evaluator(
                scene_token=scene_token,
                reference_plan=reference_plan,
                correction=correction,
                plan_confidence=plan_confidence,
                structured_stats=structured_stats,
            )

            # Labels
            y_gain = batch['gain'].cpu().numpy()
            y_collision = batch['collision_delta'].cpu().numpy()
            y_offroad = batch['offroad_delta'].cpu().numpy()
            y_comfort = batch['comfort_delta'].cpu().numpy()
            y_drift = batch['drift'].cpu().numpy()

            # 计算预测的 total risk
            pred_gain = output['pred_gain'].squeeze(-1).cpu().numpy()
            pred_collision = output['pred_collision'].squeeze(-1).cpu().numpy()
            pred_offroad = output['pred_offroad'].squeeze(-1).cpu().numpy()
            pred_comfort = output['pred_comfort'].squeeze(-1).cpu().numpy()
            pred_drift = output['pred_drift'].squeeze(-1).cpu().numpy()

            pred_total_risk = pred_collision + pred_offroad + pred_comfort + pred_drift
            y_total_risk = y_collision + y_offroad + y_comfort + y_drift

            all_pred_gain.extend(pred_gain.tolist())
            all_pred_risk.extend(pred_total_risk.tolist())
            all_y_gain.extend(y_gain.tolist())
            all_y_risk.extend(y_total_risk.tolist())

            all_collision.extend(y_collision.tolist())
            all_offroad.extend(y_offroad.tolist())
            all_comfort.extend(y_comfort.tolist())
            all_drift.extend(y_drift.tolist())

        # 转换为 numpy 数组
        pred_gain_arr = np.array(all_pred_gain)
        pred_risk_arr = np.array(all_pred_risk)
        y_gain_arr = np.array(all_y_gain)
        y_risk_arr = np.array(all_y_risk)

        # 1. 计算相关性指标
        # Spearman gain
        if len(np.unique(pred_gain_arr)) > 1 and len(np.unique(y_gain_arr)) > 1:
            spearman_gain, _ = scipy_stats.spearmanr(pred_gain_arr, y_gain_arr)
        else:
            spearman_gain = 0.0

        # Kendall tau
        if len(np.unique(pred_gain_arr)) > 1 and len(np.unique(y_gain_arr)) > 1:
            kendall_gain, _ = scipy_stats.kendalltau(pred_gain_arr, y_gain_arr)
        else:
            kendall_gain = 0.0

        # Spearman risk
        if len(np.unique(pred_risk_arr)) > 1 and len(np.unique(y_risk_arr)) > 1:
            spearman_risk, _ = scipy_stats.spearmanr(pred_risk_arr, y_risk_arr)
        else:
            spearman_risk = 0.0

        # 2. 计算 retained vs filtered 统计
        # 使用 pred_gain > threshold 作为 retained 条件
        threshold = self.cfg.gain_threshold
        retained_mask = pred_gain_arr > threshold
        filtered_mask = ~retained_mask

        retained_gain_mean = y_gain_arr[retained_mask].mean() if retained_mask.sum() > 0 else 0.0
        filtered_gain_mean = y_gain_arr[filtered_mask].mean() if filtered_mask.sum() > 0 else 0.0

        retained_risk_mean = y_risk_arr[retained_mask].mean() if retained_mask.sum() > 0 else 0.0
        filtered_risk_mean = y_risk_arr[filtered_mask].mean() if filtered_mask.sum() > 0 else 0.0

        # 3. 计算 top-k hit rate
        k = min(10, len(y_gain_arr))
        top_k_indices = np.argsort(pred_gain_arr)[-k:]
        top_k_gains = y_gain_arr[top_k_indices]
        top_k_hit_rate = (top_k_gains > 0).sum() / k if k > 0 else 0.0

        return {
            # 相关性指标
            'spearman_gain': spearman_gain if not np.isnan(spearman_gain) else 0.0,
            'kendall_gain': kendall_gain if not np.isnan(kendall_gain) else 0.0,
            'spearman_risk': spearman_risk if not np.isnan(spearman_risk) else 0.0,
            # Top-k 指标
            'top_k_hit_rate': top_k_hit_rate,
            'top_k': k,
            # Retained vs Filtered 指标
            'retained_gain_mean': retained_gain_mean,
            'filtered_gain_mean': filtered_gain_mean,
            'retained_risk_mean': retained_risk_mean,
            'filtered_risk_mean': filtered_risk_mean,
            'gain_diff': retained_gain_mean - filtered_gain_mean,  # 越大越好
            'risk_diff': filtered_risk_mean - retained_risk_mean,   # 越大越好
            # 样本数
            'n_samples': len(pred_gain_arr),
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        """完整训练流程。

        每次评估时输出：
        - 标准 loss
        - 排序质量指标（Spearman, Kendall, Top-k）
        - 过滤质量指标（Retained vs Filtered 的 gain/risk）
        """
        metrics = {'train': [], 'val': []}

        for epoch in range(self.cfg.epochs):
            train_metrics = self.train_epoch(train_dataloader, epoch)
            metrics['train'].append(train_metrics)

            if val_dataloader is not None and epoch % self.cfg.eval_every == 0:
                # 基础 loss
                val_loss_metrics = self.evaluate(val_dataloader)

                # 排序质量指标
                ranking_metrics = self.evaluate_with_ranking_metrics(val_dataloader)

                # 合并所有指标
                val_metrics = {**val_loss_metrics, **ranking_metrics}
                metrics['val'].append(val_metrics)

                # 打印详细日志
                logger.info(
                    f'[Evaluator Val Epoch {epoch}] '
                    f'loss_total={val_metrics["total_loss"]:.4f} '
                    f'loss_gain={val_metrics["loss_gain"]:.4f}'
                )
                logger.info(
                    f'[Evaluator Ranking] '
                    f'spearman_gain={val_metrics["spearman_gain"]:.3f} '
                    f'kendall={val_metrics["kendall_gain"]:.3f} '
                    f'spearman_risk={val_metrics["spearman_risk"]:.3f}'
                )
                logger.info(
                    f'[Evaluator Filtering] '
                    f'retained_gain={val_metrics["retained_gain_mean"]:.3f} '
                    f'filtered_gain={val_metrics["filtered_gain_mean"]:.3f} '
                    f'gain_diff={val_metrics["gain_diff"]:.3f} '
                    f'retained_risk={val_metrics["retained_risk_mean"]:.3f} '
                    f'filtered_risk={val_metrics["filtered_risk_mean"]:.3f}'
                )
                logger.info(
                    f'[Evaluator Top-k] '
                    f'top_{val_metrics["top_k"]}_hit_rate={val_metrics["top_k_hit_rate"]:.1%}'
                )

            # 保存检查点
            if output_dir is not None and epoch % self.cfg.eval_every == 0:
                self.save_checkpoint(
                    Path(output_dir) / f'evaluator_epoch_{epoch}.pth',
                    epoch,
                )

        return metrics

    def save_checkpoint(self, path: Path, epoch: int):
        """保存检查点。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'evaluator_state_dict': self.evaluator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        logger.info(f'Evaluator checkpoint saved to {path}')

    def load_checkpoint(self, path: Path) -> int:
        """加载检查点。"""
        ckpt = torch.load(path, map_location=self.device)
        self.evaluator.load_state_dict(ckpt['evaluator_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(f'Evaluator checkpoint loaded from {path} (epoch={ckpt["epoch"]})')
        return ckpt['epoch']
