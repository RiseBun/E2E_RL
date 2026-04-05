"""
OfflineEvaluator — 离线排序验证器

在将 UpdateEvaluator 接进 RL 之前，必须先做离线排序验证。

验证指标：
1. Gain 排序能力：pred_gain 高的，真实 ΔR 是否也高
   - Spearman 相关系数
   - Kendall tau
   - Top-k hit rate

2. Risk 排序能力：pred_risk 高的，真实 risk 增量是否也高

3. 强化命中率：
   keep = (pred_gain > τ_gain) ∧ (pred_risk < τ_risk)
   被 keep 的 updates，真实平均 ΔR 是否更高，真实平均 risk 是否更低
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from E2E_RL.update_selector.update_evaluator import UpdateEvaluator


class OfflineEvaluator:
    """离线排序验证器。"""

    def __init__(
        self,
        evaluator: UpdateEvaluator,
        device: torch.device = torch.device('cpu'),
    ):
        self.evaluator = evaluator.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate_ranking(
        self,
        dataloader: DataLoader,
        k_values: List[int] = [1, 3, 5],
    ) -> Dict[str, float]:
        """评估排序能力。

        Args:
            dataloader: UpdateEvaluatorDataset 的 DataLoader
            k_values: Top-k 的 k 值列表

        Returns:
            排序指标字典
        """
        self.evaluator.eval()

        # 收集所有预测和标签
        all_pred_gain = []
        all_true_gain = []
        all_pred_risk = []
        all_true_risk = []

        for batch in dataloader:
            scene_token = batch['scene_token'].to(self.device)
            reference_plan = batch['reference_plan'].to(self.device)  # [B, T*2]
            correction = batch['correction'].to(self.device)          # [B, T*2]
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

            output = self.evaluator(
                scene_token=scene_token,
                reference_plan=reference_plan,
                correction=correction,
                plan_confidence=plan_confidence,
                structured_stats=structured_stats,
            )

            all_pred_gain.append(output['pred_gain'].squeeze(-1).cpu())
            all_true_gain.append(batch['gain'].cpu())

            # True risk = collision + offroad + comfort + drift
            true_risk = (
                batch['collision_delta']
                + batch['offroad_delta']
                + batch['comfort_delta']
                + batch['drift']
            )
            all_pred_risk.append(output['pred_risk'].squeeze(-1).cpu())
            all_true_risk.append(true_risk.cpu())

        all_pred_gain = torch.cat(all_pred_gain)
        all_true_gain = torch.cat(all_true_gain)
        all_pred_risk = torch.cat(all_pred_risk)
        all_true_risk = torch.cat(all_true_risk)

        results = {}

        # ---- 1. Gain 排序能力 ----
        # Spearman 相关系数
        spearman_gain = self._spearman_corr(all_pred_gain, all_true_gain)
        results['spearman_gain'] = spearman_gain

        # Kendall tau
        kendall_gain = self._kendall_tau(all_pred_gain, all_true_gain)
        results['kendall_gain'] = kendall_gain

        # Top-k hit rate
        for k in k_values:
            hit_rate = self._topk_hit_rate(
                all_pred_gain, all_true_gain, k
            )
            results[f'gain_top{k}_hit_rate'] = hit_rate

        # ---- 2. Risk 排序能力 ----
        spearman_risk = self._spearman_corr(all_pred_risk, all_true_risk)
        results['spearman_risk'] = spearman_risk

        kendall_risk = self._kendall_tau(all_pred_risk, all_true_risk)
        results['kendall_risk'] = kendall_risk

        # ---- 3. 强化命中率 ----
        # 测试不同的阈值组合
        for tau_gain in [0.0, 0.1, 0.2]:
            for tau_risk in [0.5, 1.0, 2.0]:
                keep_mask = (all_pred_gain > tau_gain) & (all_pred_risk < tau_risk)

                if keep_mask.sum() > 0:
                    kept_true_gain = all_true_gain[keep_mask].mean().item()
                    kept_true_risk = all_true_risk[keep_mask].mean().item()
                    kept_ratio = keep_mask.float().mean().item()
                else:
                    kept_true_gain = 0.0
                    kept_true_risk = 0.0
                    kept_ratio = 0.0

                results[f'keep_tg{tau_gain}_tr{tau_risk}_ratio'] = kept_ratio
                results[f'keep_tg{tau_gain}_tr{tau_risk}_gain'] = kept_true_gain
                results[f'keep_tg{tau_gain}_tr{tau_risk}_risk'] = kept_true_risk

        return results

    @torch.no_grad()
    def evaluate_filtering(
        self,
        dataloader: DataLoader,
        tau_gain: float = 0.0,
        tau_risk: float = 0.5,
    ) -> Dict[str, float]:
        """评估过滤效果。

        比较被 keep 的 updates vs 被 filter 的 updates 的真实指标。

        Returns:
            过滤效果字典
        """
        self.evaluator.eval()

        kept_gains = []
        kept_risks = []
        filtered_gains = []
        filtered_risks = []

        for batch in dataloader:
            scene_token = batch['scene_token'].to(self.device)
            reference_plan = batch['reference_plan'].to(self.device)  # [B, T*2]
            correction = batch['correction'].to(self.device)          # [B, T*2]
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

            output = self.evaluator(
                scene_token=scene_token,
                reference_plan=reference_plan,
                correction=correction,
                plan_confidence=plan_confidence,
                structured_stats=structured_stats,
            )

            pred_gain = output['pred_gain'].squeeze(-1)
            pred_risk = output['pred_risk'].squeeze(-1)

            true_gain = batch['gain']
            true_risk = batch['collision_delta'] + batch['offroad_delta'] + batch['comfort_delta'] + batch['drift']

            # Keep = pred_gain > tau_gain AND pred_risk < tau_risk
            keep_mask = (pred_gain > tau_gain) & (pred_risk < tau_risk)

            kept_gains.extend(true_gain[keep_mask].tolist())
            kept_risks.extend(true_risk[keep_mask].tolist())
            filtered_gains.extend(true_gain[~keep_mask].tolist())
            filtered_risks.extend(true_risk[~keep_mask].tolist())

        results = {
            'n_kept': len(kept_gains),
            'n_filtered': len(filtered_gains),
            'kept_mean_gain': sum(kept_gains) / max(len(kept_gains), 1),
            'kept_mean_risk': sum(kept_risks) / max(len(kept_risks), 1),
            'filtered_mean_gain': sum(filtered_gains) / max(len(filtered_gains), 1),
            'filtered_mean_risk': sum(filtered_risks) / max(len(filtered_risks), 1),
            'gain_improvement': (
                sum(kept_gains) / max(len(kept_gains), 1)
                - sum(filtered_gains) / max(len(filtered_gains), 1)
            ),
            'risk_reduction': (
                sum(filtered_risks) / max(len(filtered_risks), 1)
                - sum(kept_risks) / max(len(kept_risks), 1)
            ),
        }

        return results

    # ---- 辅助方法 ----

    @staticmethod
    def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        """计算 Spearman 相关系数。"""
        n = len(x)
        if n < 2:
            return 0.0

        # 排序
        x_sorted, x_rank = torch.sort(torch.argsort(x))
        y_sorted, y_rank = torch.sort(torch.argsort(y))

        # 计算等级差
        d = x_rank.float() - y_rank.float()

        # Spearman = 1 - 6 * Σd² / (n³ - n)
        spearman = 1 - 6 * (d.float() ** 2).sum() / (n ** 3 - n)
        return spearman.item()

    @staticmethod
    def _kendall_tau(x: torch.Tensor, y: torch.Tensor) -> float:
        """计算 Kendall tau 相关系数。"""
        n = len(x)
        if n < 2:
            return 0.0

        # 计算 concordant 和 discordant 对
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                x_diff = (x[i] - x[j]).item()
                y_diff = (y[i] - y[j]).item()

                if x_diff * y_diff > 0:
                    concordant += 1
                elif x_diff * y_diff < 0:
                    discordant += 1

        total_pairs = n * (n - 1) / 2
        tau = (concordant - discordant) / total_pairs
        return tau

    @staticmethod
    def _topk_hit_rate(
        pred: torch.Tensor,
        true: torch.Tensor,
        k: int,
    ) -> float:
        """计算 Top-k hit rate。

        pred 最高的 k 个，对应的 true 是否也高。
        """
        n = len(pred)
        k = min(k, n)

        _, topk_indices = torch.topk(pred, k)
        topk_true = true[topk_indices]

        # Hit = top-k 的 true 均值是否大于全局均值
        global_mean = true.mean()
        hit_rate = (topk_true > global_mean).float().mean().item()

        return hit_rate
