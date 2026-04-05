"""
CorrectionPolicyTrainer — 两阶段训练器

训练流程：
- Stage 1: Behavioral Cloning 预热
  - 用 GT correction 做监督学习
  - Loss = -log π(gt_correction | state)
  - 目的：让 Policy 参数初始化到一个合理的起点

- Stage 2: Policy Gradient + STAPO Gate
  - Policy 采样 correction
  - 计算 Advantage = safe_reward(corrected) - safe_reward(ref)
  - Loss = STAPO(filtered(-A * log π(a|s))) - entropy_coef * H
  - 目的：让 Policy 通过 RL 信号学会产生有益的修正
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.correction_policy.losses import (
    behavioral_cloning_loss,
    policy_gradient_loss,
    compute_advantage,
    entropy_bonus_loss,
)
from E2E_RL.update_selector.safety_guard import SafetyGuard, SafetyGuardConfig
from E2E_RL.update_selector.stapo_gate import STAPOGate, STAPOGateConfig
from E2E_RL.update_selector.update_evaluator import LearnedUpdateGate

logger = logging.getLogger(__name__)


class CorrectionPolicyTrainer:
    """CorrectionPolicy 两阶段训练器。

    三层防御级联（按顺序执行）：
    1. SafetyGuard: 硬物理底线（只检查，不过滤）
    2. STAPOGate: 规则兜底，过滤明显的 spurious update
    3. LearnedUpdateGate: 高级学习判断，精细化过滤

    Args:
        policy: CorrectionPolicy 模型
        optimizer: PyTorch 优化器
        scheduler: 学习率调度器（可选）
        device: 训练设备
        reward_config: 奖励计算的参数字典
        safety_guard: SafetyGuard 实例（硬底线）
        stapo_gate: STAPOGate 实例（规则基线）
        learned_gate: LearnedUpdateGate 实例（高级判断）
        entropy_coef: 熵正则化系数
        grad_clip: 梯度裁剪阈值
    """

    def __init__(
        self,
        policy: CorrectionPolicy,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device('cuda'),
        reward_config: Optional[Dict[str, Any]] = None,
        safety_guard: Optional[SafetyGuard] = None,
        stapo_gate: Optional[STAPOGate] = None,
        learned_gate: Optional[LearnedUpdateGate] = None,
        entropy_coef: float = 0.01,
        grad_clip: float = 1.0,
    ):
        self.policy = policy.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.reward_config = reward_config or {}
        
        # 三层防御层级
        self.safety_guard = safety_guard or SafetyGuard(SafetyGuardConfig(enabled=False))
        self.stapo_gate = stapo_gate or STAPOGate(STAPOGateConfig(enabled=False))
        self.learned_gate = learned_gate
        
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

    # =======================================================================
    # Stage 1: Behavioral Cloning
    # =======================================================================

    def train_bc_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """执行一个 Behavioral Cloning epoch。

        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch 编号

        Returns:
            epoch 级别的平均指标字典
        """
        self.policy.train()
        epoch_metrics: Dict[str, float] = {
            'loss_total': 0.0,
            'mean_log_prob': 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            interface = batch['interface'].to(self.device)
            gt_plan = batch['gt_plan'].to(self.device)
            plan_mask = batch.get('plan_mask')
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device)

            # 计算 GT correction
            gt_correction = gt_plan - interface.reference_plan

            # BC loss
            loss = behavioral_cloning_loss(
                policy=self.policy,
                interface=interface,
                gt_correction=gt_correction,
                mask=plan_mask,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self._clip_grad()
            self.optimizer.step()

            # 记录
            with torch.no_grad():
                eval_result = self.policy.evaluate(interface, gt_correction)
                mean_log_prob = eval_result['log_prob'].mean().item()

            epoch_metrics['loss_total'] += loss.item()
            epoch_metrics['mean_log_prob'] += mean_log_prob
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        # 计算平均值
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        logger.info(
            f'[BC Epoch {epoch}] '
            f'loss={epoch_metrics["loss_total"]:.4f} '
            f'log_prob={epoch_metrics["mean_log_prob"]:.4f}'
        )
        return epoch_metrics

    # =======================================================================
    # Stage 2: Policy Gradient + STAPO Gate
    # =======================================================================

    def train_rl_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """执行一个 Policy Gradient + 三层防御 epoch。

        三层防御级联（按顺序执行）：
        1. SafetyGuard: 硬物理底线（只检查，不过滤）
        2. STAPOGate: 规则兜底，过滤明显的 spurious update
        3. LearnedUpdateGate: 高级学习判断，精细化过滤

        训练流程：
        1. Policy 采样 correction
        2. 计算 Advantage
        3. 计算 per-sample PG loss
        4. Safety Guard 检查（只记录，不过滤）
        5. STAPO Gate 过滤（规则基线）
        6. Learned Gate 过滤（高级判断）
        7. Entropy bonus
        8. Backward + step

        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch 编号

        Returns:
            epoch 级别的平均指标字典
        """
        self.policy.train()
        epoch_metrics: Dict[str, float] = {
            'loss_total': 0.0,
            'loss_pg': 0.0,
            'loss_entropy': 0.0,
            'mean_advantage': 0.0,
            'retention_ratio': 0.0,
            'stapo_retention': 0.0,
            'learned_retention': 0.0,
            'safety_pass_ratio': 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            interface = batch['interface'].to(self.device)
            gt_plan = batch['gt_plan'].to(self.device)
            plan_mask = batch.get('plan_mask')
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device)

            # ---- 1. Policy 采样 ----
            sample_result = self.policy.sample(interface)
            correction = sample_result['correction']
            corrected_plan = sample_result['corrected_plan']
            log_prob = sample_result['log_prob']
            entropy = sample_result['entropy']

            # ---- 2. 计算 Advantage ----
            advantage = compute_advantage(
                corrected_plan=corrected_plan,
                reference_plan=interface.reference_plan,
                gt_plan=gt_plan,
                mask=plan_mask,
                reward_config=self.reward_config,
            )

            # ---- 3. Per-sample PG loss ----
            per_sample_pg = policy_gradient_loss(log_prob, advantage)

            # ---- 4. Safety Guard 检查（硬底线，只记录） ----
            safety_mask = self.safety_guard.check(
                correction=correction,
                reference_plan=interface.reference_plan,
            )

            # ---- 5. STAPO Gate 过滤（规则基线） ----
            stapo_mask = self.stapo_gate.compute_mask(
                advantages=advantage,
                action_log_probs=log_prob,
                entropies=entropy,
            )

            # 合并 safety 和 stapo mask（必须同时通过）
            combined_mask = safety_mask & stapo_mask

            # ---- 6. Learned Gate 过滤（高级判断） ----
            if self.learned_gate is not None:
                # 计算结构化统计量
                from E2E_RL.update_selector.candidate_generator import compute_structured_stats
                structured_stats = compute_structured_stats(
                    correction,
                    interface.reference_plan,
                    corrected_plan,
                    dt=self.reward_config.get('dt', 0.5),
                )
                
                # Learned Gate 判断
                learned_mask, learned_diag = self.learned_gate.compute_mask(
                    advantages=advantage,
                    interface=interface,
                    correction=correction,
                    structured_stats=structured_stats,
                    safety_mask=combined_mask,
                )
                
                # 最终 mask = Safety & STAPO & Learned
                final_mask = combined_mask & learned_mask
                
                # 记录 Learned Gate 统计
                epoch_metrics['learned_retention'] += learned_diag.get('retention_ratio', 1.0)
            else:
                final_mask = combined_mask
                learned_diag = {}

            # ---- 7. 应用 mask 到 loss ----
            # 被过滤的样本 loss 设为 0（不参与训练）
            masked_loss = per_sample_pg * final_mask.float()
            filtered_loss = masked_loss.sum() / (final_mask.sum() + 1e-8)

            # ---- 8. Entropy bonus ----
            entropy_loss = entropy_bonus_loss(entropy) * self.entropy_coef

            # ---- 9. Total loss ----
            total_loss = filtered_loss + entropy_loss

            # ---- Backward ----
            self.optimizer.zero_grad()
            total_loss.backward()
            self._clip_grad()
            self.optimizer.step()

            # ---- 记录 ----
            epoch_metrics['loss_total'] += total_loss.item()
            epoch_metrics['loss_pg'] += filtered_loss.item()
            epoch_metrics['loss_entropy'] += entropy_loss.item()
            epoch_metrics['mean_advantage'] += advantage.mean().item()
            epoch_metrics['retention_ratio'] += final_mask.float().mean().item()
            epoch_metrics['stapo_retention'] += combined_mask.float().mean().item()
            epoch_metrics['safety_pass_ratio'] += safety_mask.float().mean().item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        # 计算平均值
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        logger.info(
            f'[RL Epoch {epoch}] '
            f'loss={epoch_metrics["loss_total"]:.4f} '
            f'pg={epoch_metrics["loss_pg"]:.4f} '
            f'entropy={epoch_metrics["loss_entropy"]:.4f} '
            f'adv={epoch_metrics["mean_advantage"]:.4f} '
            f'retent={epoch_metrics["retention_ratio"]:.2%} '
            f'(safety={epoch_metrics["safety_pass_ratio"]:.2%}'
            f', stapo={epoch_metrics["stapo_retention"]:.2%}'
            + (f', learned={epoch_metrics["learned_retention"]:.2%}' if self.learned_gate else '')
            + ')'
        )
        return epoch_metrics

    # =======================================================================
    # 评估模式
    # =======================================================================

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估当前 policy。

        Args:
            dataloader: 数据加载器

        Returns:
            评估指标字典
        """
        self.policy.eval()

        total_reward_diff = 0.0
        total_correction_magnitude = 0.0
        total_entropy = 0.0
        num_samples = 0

        for batch in dataloader:
            interface = batch['interface'].to(self.device)
            gt_plan = batch['gt_plan'].to(self.device)
            plan_mask = batch.get('plan_mask')
            if plan_mask is not None:
                plan_mask = plan_mask.to(self.device)

            # 确定性 correction
            correction = self.policy.act(interface)
            corrected_plan = interface.reference_plan + correction

            # 计算 Advantage
            advantage = compute_advantage(
                corrected_plan=corrected_plan,
                reference_plan=interface.reference_plan,
                gt_plan=gt_plan,
                mask=plan_mask,
                reward_config=self.reward_config,
            )

            # 计算修正幅度
            corr_mag = torch.norm(correction, dim=-1).mean().item()

            # 计算熵
            sample_result = self.policy.sample(interface)
            ent = sample_result['entropy'].mean().item()

            total_reward_diff += advantage.sum().item()
            total_correction_magnitude += corr_mag
            total_entropy += ent
            num_samples += interface.scene_token.shape[0]

        if num_samples == 0:
            return {}

        return {
            'mean_advantage': total_reward_diff / num_samples,
            'mean_correction_magnitude': total_correction_magnitude / len(dataloader),
            'mean_entropy': total_entropy / len(dataloader),
        }

    # =======================================================================
    # 辅助方法
    # =======================================================================

    def _clip_grad(self):
        """梯度裁剪。"""
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存训练检查点。

        Args:
            path: 保存路径
            epoch: 当前 epoch
            extra: 额外信息
        """
        ckpt = {
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        if extra:
            ckpt.update(extra)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)
        logger.info(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载训练检查点。

        Args:
            path: 检查点路径

        Returns:
            检查点字典
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(f'Checkpoint loaded from {path} (epoch={ckpt.get("epoch", "?")})')
        return ckpt

    # =======================================================================
    # 便捷训练方法
    # =======================================================================

    def train(
        self,
        dataloader: DataLoader,
        bc_epochs: int = 5,
        rl_epochs: int = 20,
        output_dir: Optional[str] = None,
        save_every: int = 5,
    ) -> Dict[str, List[Dict]]:
        """完整两阶段训练流程。

        Args:
            dataloader: 数据加载器
            bc_epochs: BC 预热 epoch 数
            rl_epochs: RL 训练 epoch 数
            output_dir: 输出目录
            save_every: 每隔多少 epoch 保存一次

        Returns:
            dict with 'bc_metrics' and 'rl_metrics'
        """
        all_metrics = {
            'bc_metrics': [],
            'rl_metrics': [],
        }

        # ---- Stage 1: BC 预热 ----
        if bc_epochs > 0:
            logger.info('=' * 50)
            logger.info('Stage 1: Behavioral Cloning')
            logger.info('=' * 50)
            for epoch in range(bc_epochs):
                metrics = self.train_bc_epoch(dataloader, epoch)
                all_metrics['bc_metrics'].append(metrics)

                if output_dir and epoch % save_every == 0:
                    self.save_checkpoint(
                        os.path.join(output_dir, f'bc_epoch_{epoch}.pth'),
                        epoch,
                    )

        # ---- Stage 2: RL 训练 ----
        if rl_epochs > 0:
            logger.info('=' * 50)
            logger.info('Stage 2: Policy Gradient + STAPO Gate')
            logger.info('=' * 50)
            for epoch in range(rl_epochs):
                metrics = self.train_rl_epoch(dataloader, epoch)
                all_metrics['rl_metrics'].append(metrics)

                if output_dir and epoch % save_every == 0:
                    self.save_checkpoint(
                        os.path.join(output_dir, f'rl_epoch_{epoch}.pth'),
                        bc_epochs + epoch,
                    )

        return all_metrics
