"""
CandidateCorrector — 多样化候选修正生成器

核心思想：
训练 UpdateEvaluator 时，如果只用当前 policy 采样的 correction，会有偏。
需要生成多样化的候选修正，让 selector 学会"比较什么是更值得强化的 update"。

候选类型：
1. zero correction - 不修正
2. policy sample - 当前 policy 采样
3. deterministic mean - policy 均值
4. GT-directed - 朝 GT 方向的小修正

采样策略（已优化）：
- zero: 10%
- deterministic mean: 15%
- policy_sample: 35%
- gt_directed: 40%

已移除：bounded_random 和 safety_biased（产生大量负 gain）

核心原则：100% 的候选来自"有希望变好"的来源（policy/GT/deterministic）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.correction_policy.policy import CorrectionPolicy

# 默认采样比例
# 核心原则：移除 bounded_random 和 safety_biased（产生大量负 gain）
# GT-directed 比例提高，幅度进一步减小以提高正 gain 比例
DEFAULT_SAMPLE_WEIGHTS = {
    'zero': 0.10,
    'deterministic': 0.05,
    'policy_sample': 0.25,
    'gt_directed': 0.60,  # 进一步提高 GT-directed 比例
    # 已移除：bounded_random 和 safety_biased 产生大量负 gain
}


class CandidateStats:
    """候选质量统计收集器。

    每次收集数据时顺手统计每种候选类型的：
    - 数量
    - gain 均值
    - 正 gain 比例
    - risk 均值
    - 被 gate 保留率
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计。"""
        # 按类型存储统计
        # 每个样本: {'gain': float, 'pred_gain': float, 'pred_risk': float, 'is_retained': bool}
        self.samples = defaultdict(list)
        self.total_count = 0

    def add_sample(self, corr_type: str, gain: float, pred_gain: float = None,
                   pred_risk: float = None, is_retained: bool = None):
        """添加一个样本的统计。"""
        self.samples[corr_type].append({
            'gain': gain,
            'pred_gain': pred_gain,
            'pred_risk': pred_risk,
            'is_retained': is_retained,
        })
        self.total_count += 1

    def get_summary(self) -> Dict:
        """获取统计摘要。"""
        summary = {
            'total_count': self.total_count,
            'by_type': {},
            'overall': {},
        }

        all_gains = []
        all_positive_count = 0
        all_retained_count = 0
        all_count = 0

        for corr_type, samples in self.samples.items():
            if not samples:
                continue

            gains = [s['gain'] for s in samples]
            pos_count = sum(1 for g in gains if g > 0)
            retained_count = sum(1 for s in samples if s.get('is_retained', True))
            retained_ratio = retained_count / len(samples)

            summary['by_type'][corr_type] = {
                'count': len(samples),
                'gain_mean': sum(gains) / len(gains),
                'gain_std': (sum((g - sum(gains)/len(gains))**2 for g in gains) / len(gains)) ** 0.5,
                'positive_ratio': pos_count / len(samples),
                'retained_ratio': retained_ratio,
            }

            all_gains.extend(gains)
            all_positive_count += pos_count
            all_retained_count += retained_count
            all_count += len(samples)

        if all_gains:
            summary['overall'] = {
                'count': all_count,
                'gain_mean': sum(all_gains) / len(all_gains),
                'positive_ratio': all_positive_count / all_count,
                'retained_ratio': all_retained_count / all_count if all_count > 0 else 0,
            }

        return summary

    def print_report(self):
        """打印统计报告。"""
        summary = self.get_summary()

        print("=" * 70)
        print("候选质量统计报告")
        print("=" * 70)
        print(f"\n{'类型':<20} {'数量':>6} {'正gain%':>10} {'gain均值':>10} {'gain标准差':>12} {'保留率':>10}")
        print("-" * 70)

        for corr_type in ['zero', 'policy_sample', 'deterministic', 'gt_directed',
                          'bounded_random', 'safety_biased']:
            if corr_type not in summary['by_type']:
                continue
            s = summary['by_type'][corr_type]
            print(f"{corr_type:<20} {s['count']:>6} {s['positive_ratio']:>10.1%} "
                  f"{s['gain_mean']:>10.3f} {s['gain_std']:>12.3f} {s['retained_ratio']:>10.1%}")

        print("-" * 70)
        overall = summary.get('overall', {})
        print(f"{'总体':<20} {overall.get('count', 0):>6} "
              f"{overall.get('positive_ratio', 0):>10.1%} "
              f"{overall.get('gain_mean', 0):>10.3f}")
        print("=" * 70)


class CandidateCorrector:
    """多样化候选修正生成器。

    用于训练 UpdateEvaluator 时生成多样化的 candidate corrections，
    让 selector 学会比较不同类型的 update。

    支持两种模式：
    1. generate_all_types: 生成所有类型（等量）
    2. generate_weighted: 按权重采样（推荐）

    Args:
        policy: CorrectionPolicy（用于采样和均值）
        max_corrections_per_type: 每种类型生成多少个候选
        random_scale: 随机修正的缩放因子
        sample_weights: 采样权重字典
    """

    def __init__(
        self,
        policy: Optional[CorrectionPolicy] = None,
        max_corrections_per_type: int = 1,
        random_scale: float = 2.0,
        gt_directed_scale: float = 0.5,
        sample_weights: Optional[Dict[str, float]] = None,
    ):
        self.policy = policy
        self.max_corrections_per_type = max_corrections_per_type
        self.random_scale = random_scale
        self.gt_directed_scale = gt_directed_scale
        self.sample_weights = sample_weights or DEFAULT_SAMPLE_WEIGHTS
        self.stats = CandidateStats()

    def reset_stats(self):
        """重置统计。"""
        self.stats.reset()

    def get_stats(self) -> CandidateStats:
        """获取统计收集器。"""
        return self.stats

    def generate_all_types(
        self,
        interface: PlanningInterface,
        gt_plan: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """生成所有类型的候选修正。

        注意：这个方法生成等量的各类候选，可能导致分布不平衡。
        推荐使用 generate_weighted() 代替。

        Args:
            interface: PlanningInterface
            gt_plan: [B, T, 2] GT 轨迹（可选，用于 GT-directed）

        Returns:
            dict with:
                - zero: 零修正
                - policy_sample: policy 采样
                - deterministic: policy 均值
                - bounded_random: 有界随机
                - gt_directed: 朝 GT 方向
                - all_corrections: [B, N, T, 2] 所有候选修正堆叠
                - correction_types: list of type names
        """
        B, T, _ = interface.reference_plan.shape
        device = interface.scene_token.device

        corrections = []
        types = []

        # 1. Zero correction
        zero_corr = torch.zeros(B, T, 2, device=device)
        corrections.append(zero_corr)
        types.append('zero')

        # 2. Policy sample
        if self.policy is not None:
            sample_result = self.policy.sample(interface)
            policy_sample = sample_result['correction']
            corrections.append(policy_sample)
            types.append('policy_sample')

        # 3. Deterministic mean
        if self.policy is not None:
            det_corr = self.policy.act(interface)
            corrections.append(det_corr)
            types.append('deterministic')

        # 4. Bounded random (减少数量)
        for _ in range(1):  # 只生成 1 个 bounded_random
            random_corr = (
                torch.randn(B, T, 2, device=device)
                * self.random_scale
            )
            corrections.append(random_corr)
            types.append('bounded_random')

        # 5. GT-directed (增加数量)
        if gt_plan is not None:
            gt_corr = gt_plan - interface.reference_plan
            # 缩放版本
            gt_corr_scaled = gt_corr * self.gt_directed_scale
            corrections.append(gt_corr_scaled)
            types.append('gt_directed')

            # GT-directed with noise
            gt_corr_noisy = gt_corr_scaled + torch.randn_like(gt_corr_scaled) * 0.2
            corrections.append(gt_corr_noisy)
            types.append('gt_directed')

        # 6. Safety-biased (减少数量)
        for _ in range(1):  # 只生成 1 个 safety_biased
            safety_corr = torch.randn(B, T, 2, device=device) * (self.random_scale * 0.5)
            corrections.append(safety_corr)
            types.append('safety_biased')

        # 堆叠所有候选
        all_corrections = torch.stack(corrections, dim=1)  # [B, N_types, T, 2]

        return {
            'zero': zero_corr,
            'policy_sample': policy_sample if self.policy else None,
            'deterministic': det_corr if self.policy else None,
            'all_corrections': all_corrections,
            'correction_types': types,
        }

    def generate_weighted(
        self,
        interface: PlanningInterface,
        gt_plan: Optional[torch.Tensor] = None,
        n_samples: int = 6,
    ) -> Tuple[torch.Tensor, List[str]]:
        """按权重采样生成候选修正（推荐方法）。

        根据 sample_weights 中的比例生成候选，确保分布平衡。
        移除 bounded_random 和 safety_biased，全部使用"有希望变好"的来源。

        Args:
            interface: PlanningInterface
            gt_plan: [B, T, 2] GT 轨迹
            n_samples: 每批生成的候选数量

        Returns:
            (corrections, types)
                corrections: [B, n_samples, T, 2]
                types: list of type names
        """
        B, T, _ = interface.reference_plan.shape
        device = interface.reference_plan.device

        corrections = []
        types = []

        # 计算每种类型的期望数量（按权重比例）
        weights = self.sample_weights.copy()
        total_weight = sum(weights.values())
        expected_counts = {k: n_samples * v / total_weight for k, v in weights.items()}

        # 转换为整数分配 + 剩余分数分配
        int_counts = {k: int(f) for k, f in expected_counts.items()}
        remainder = n_samples - sum(int_counts.values())

        # 按余数排序分配（贪心）
        remainders = {k: expected_counts[k] - int_counts[k] for k in weights}
        sorted_keys = sorted(remainders.keys(), key=lambda k: remainders[k], reverse=True)
        for i, k in enumerate(sorted_keys):
            if i < remainder:
                int_counts[k] += 1

        # 生成每种类型的候选
        # Zero
        for _ in range(int_counts.get('zero', 0)):
            corrections.append(torch.zeros(B, T, 2, device=device))
            types.append('zero')

        # Deterministic
        for _ in range(int_counts.get('deterministic', 0)):
            if self.policy is not None:
                det_corr = self.policy.act(interface)
                corrections.append(det_corr)
                types.append('deterministic')

        # Policy sample
        for _ in range(int_counts.get('policy_sample', 0)):
            if self.policy is not None:
                sample = self.policy.sample(interface)['correction']
                corrections.append(sample)
                types.append('policy_sample')

        # GT-directed（最有希望的来源，生成多个不同幅度的变体）
        gt_count = int_counts.get('gt_directed', 0)
        if gt_plan is not None and gt_count > 0:
            gt_corr = gt_plan - interface.reference_plan

            # 生成不同缩放系数的 GT-directed 修正
            # 原则：更小的缩放系数产生更高的正 gain 比例
            scale_factors = []
            if gt_count >= 1:
                scale_factors.append(0.1)  # 微小幅度（最高正 gain 概率）
            if gt_count >= 2:
                scale_factors.append(0.2)  # 小幅度
            if gt_count >= 3:
                scale_factors.append(0.35)  # 中小幅度
            if gt_count >= 4:
                scale_factors.append(0.5)  # 中等幅度
            # 剩余的用带噪声的变体
            remaining = gt_count - len(scale_factors)

            for scale in scale_factors:
                gt_corr_scaled = gt_corr * scale
                corrections.append(gt_corr_scaled)
                types.append('gt_directed')

            # 添加带小噪声的变体
            for _ in range(remaining):
                scale = scale_factors[-1] if scale_factors else 0.25
                gt_corr_noisy = gt_corr * scale + torch.randn_like(gt_corr) * 0.05
                corrections.append(gt_corr_noisy)
                types.append('gt_directed')

        # 如果还不够 n_samples，用 deterministic 或 zero 填充
        while len(corrections) < n_samples:
            if self.policy is not None:
                det_corr = self.policy.act(interface)
                corrections.append(det_corr)
                types.append('deterministic')
            else:
                # policy 不存在时用 zero 填充
                corrections.append(torch.zeros(B, T, 2, device=device))
                types.append('zero')

        # 限制到 n_samples
        corrections = corrections[:n_samples]
        types = types[:n_samples]

        all_corrections = torch.stack(corrections, dim=1)  # [B, n_samples, T, 2]

        return all_corrections, types

    def generate_batch(
        self,
        interface: PlanningInterface,
        gt_plan: Optional[torch.Tensor] = None,
        n_samples: int = 4,
        use_weighted: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
        """生成一批多样化的候选修正。

        推荐使用 use_weighted=True 来获得更平衡的分布。

        Args:
            interface: PlanningInterface
            gt_plan: [B, T, 2] GT 轨迹
            n_samples: 总共生成多少个候选
            use_weighted: 是否使用加权采样

        Returns:
            (corrections, types)
                corrections: [B, n_samples, T, 2]
                types: list of type names
        """
        if use_weighted:
            return self.generate_weighted(interface, gt_plan, n_samples)

        # 回退到原来的逻辑
        B, T, _ = interface.reference_plan.shape
        device = interface.scene_token.device

        corrections = []
        types = []

        # 固定包含的类型
        # 1. Zero
        corrections.append(torch.zeros(B, T, 2, device=device))
        types.append('zero')

        # 2. Policy sample (如果可用)
        if self.policy is not None and len(corrections) < n_samples:
            sample = self.policy.sample(interface)['correction']
            corrections.append(sample)
            types.append('policy_sample')

        # 3. Policy deterministic
        if self.policy is not None and len(corrections) < n_samples:
            det = self.policy.act(interface)
            corrections.append(det)
            types.append('deterministic')

        # 4. GT-directed (如果有 GT)
        if gt_plan is not None and len(corrections) < n_samples:
            gt_corr = (gt_plan - interface.reference_plan) * self.gt_directed_scale
            corrections.append(gt_corr)
            types.append('gt_directed')

        # 5. 填充随机
        while len(corrections) < n_samples:
            random_corr = torch.randn(B, T, 2, device=device) * self.random_scale
            corrections.append(random_corr)
            types.append('random')

        all_corrections = torch.stack(corrections, dim=1)  # [B, n_samples, T, 2]

        return all_corrections, types


def compute_structured_stats(
    correction: torch.Tensor,
    reference_plan: torch.Tensor,
    corrected_plan: torch.Tensor,
    dt: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """计算结构化统计量。

    这些统计量作为 UpdateEvaluator 的辅助输入。

    Args:
        correction: [B, T, 2] 修正量
        reference_plan: [B, T, 2] 参考轨迹
        corrected_plan: [B, T, 2] 修正后轨迹
        dt: 时间步间隔

    Returns:
        dict with:
            - residual_norm: [B] 残差 L2 范数
            - max_step_disp: [B] 最大单步位移
            - curvature_change: [B] 曲率变化
            - jerk_change: [B] 加速度变化
            - total_disp: [B] 总位移（修正后与参考轨迹终点距离）
            - speed_max: [B] 最大速度
            - support_score: [B] 支持分数（简化版）
            - drift_score: [B] 漂移分数（简化版）
    """
    B, T, _ = correction.shape
    device = correction.device

    # 1. Residual norm
    residual_norm = torch.norm(correction, dim=-1).norm(dim=-1)  # [B]

    # 2. Max step displacement
    if T >= 2:
        step_diff = torch.diff(correction, dim=1)
        max_step_disp = torch.norm(step_diff, dim=-1).max(dim=-1).values  # [B]
    else:
        max_step_disp = torch.zeros(B, device=device)

    # 3. Curvature change
    if T >= 3:
        # 参考轨迹的曲率
        ref_vel = torch.diff(reference_plan, dim=1) / dt
        ref_acc = torch.diff(ref_vel, dim=1) / dt
        ref_curvature = torch.norm(ref_acc, dim=-1).mean(dim=-1)  # [B]

        # 修正后轨迹的曲率
        corr_vel = torch.diff(corrected_plan, dim=1) / dt
        corr_acc = torch.diff(corr_vel, dim=1) / dt
        corr_curvature = torch.norm(corr_acc, dim=-1).mean(dim=-1)  # [B]

        curvature_change = (corr_curvature - ref_curvature).clamp(min=0)
    else:
        curvature_change = torch.zeros(B, device=device)

    # 4. Jerk change
    if T >= 4:
        ref_jerk = torch.diff(torch.diff(torch.diff(reference_plan, dim=1), dim=1), dim=1) / (dt ** 3)
        ref_jerk_mag = torch.norm(ref_jerk, dim=-1).mean(dim=-1)

        corr_jerk = torch.diff(torch.diff(torch.diff(corrected_plan, dim=1), dim=1), dim=1) / (dt ** 3)
        corr_jerk_mag = torch.norm(corr_jerk, dim=-1).mean(dim=-1)

        jerk_change = (corr_jerk_mag - ref_jerk_mag).clamp(min=0)
    else:
        jerk_change = torch.zeros(B, device=device)

    # 5. Total displacement
    total_disp = torch.norm(
        corrected_plan[:, -1] - reference_plan[:, -1],
        dim=-1
    )  # [B]

    # 6. Max speed
    if T >= 2:
        velocity = torch.diff(corrected_plan, dim=1) / dt
        speed = torch.norm(velocity, dim=-1)
        speed_max = speed.max(dim=-1).values  # [B]
    else:
        speed_max = torch.zeros(B, device=device)

    # 7. Support score（简化版：基于 residual norm）
    support_score = torch.exp(-residual_norm / 5.0).clamp(0, 1)  # [B]

    # 8. Drift score（简化版：基于 total displacement）
    drift_score = (1 - torch.exp(-total_disp / 5.0)).clamp(0, 1)  # [B]

    return {
        'residual_norm': residual_norm,
        'max_step_disp': max_step_disp,
        'curvature_change': curvature_change,
        'jerk_change': jerk_change,
        'total_disp': total_disp,
        'speed_max': speed_max,
        'support_score': support_score,
        'drift_score': drift_score,
    }
