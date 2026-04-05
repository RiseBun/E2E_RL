"""Harmful Update Filtering (HUF) 单元测试。

覆盖: HUFConfig, UpdateReliabilityScorer, HarmfulUpdateFilter, 以及集成测试。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.losses import compute_per_sample_reward_weighted_error
from E2E_RL.update_filter.config import HUFConfig
from E2E_RL.update_filter.filter import HarmfulUpdateFilter
from E2E_RL.update_filter.scorer import UpdateReliabilityScorer


# ============================================================
# 辅助函数
# ============================================================

def _make_interface(
    batch_size: int = 8,
    scene_dim: int = 64,
    time_steps: int = 6,
    num_modes: int = 3,
    confidence_value: float = 0.8,
    include_candidates: bool = True,
    include_safety: bool = True,
) -> PlanningInterface:
    """构造 mock PlanningInterface。"""
    scene_token = torch.randn(batch_size, scene_dim)
    reference_plan = torch.cumsum(torch.randn(batch_size, time_steps, 2) * 0.5, dim=1)
    plan_confidence = torch.full((batch_size, 1), confidence_value)

    candidate_plans = None
    if include_candidates:
        candidate_plans = torch.randn(batch_size, num_modes, time_steps, 2) * 0.3

    safety_features = None
    if include_safety:
        safety_features = {
            'safety_plan_mode_variance': torch.rand(batch_size, time_steps),
            'safety_object_density': torch.rand(batch_size, 1),
        }

    return PlanningInterface(
        scene_token=scene_token,
        reference_plan=reference_plan,
        candidate_plans=candidate_plans,
        plan_confidence=plan_confidence,
        safety_features=safety_features,
    )


def _make_refiner_outputs(
    interface: PlanningInterface,
    residual_scale: float = 0.1,
) -> dict:
    """构造 mock refiner 输出。"""
    B, T, _ = interface.reference_plan.shape
    residual = torch.randn(B, T, 2) * residual_scale
    refined_plan = interface.reference_plan + residual
    residual_norm = residual.norm(dim=-1).mean(dim=-1)  # [B]
    refine_score = torch.sigmoid(torch.randn(B))

    return {
        'residual': residual,
        'refined_plan': refined_plan,
        'residual_norm': residual_norm,
        'refine_score': refine_score,
    }


# ============================================================
# 1. HUFConfig 测试
# ============================================================

class TestHUFConfig:
    def test_defaults(self):
        cfg = HUFConfig()
        assert cfg.mode == 'hard'
        assert cfg.enabled is True
        assert cfg.tau_uncertainty == 0.7
        assert cfg.tau_support == 0.3
        assert cfg.tau_drift == 0.8
        assert cfg.min_retention_ratio == 0.3

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            HUFConfig(mode='invalid')

    def test_invalid_retention(self):
        with pytest.raises(ValueError, match="min_retention_ratio"):
            HUFConfig(min_retention_ratio=0.0)


# ============================================================
# 2. UpdateReliabilityScorer 测试
# ============================================================

class TestScorer:
    def test_uncertainty_high_conf(self):
        """高 confidence → 低 uncertainty。"""
        cfg = HUFConfig()
        scorer = UpdateReliabilityScorer(cfg)

        interface = _make_interface(confidence_value=0.95)
        outputs = _make_refiner_outputs(interface, residual_scale=0.05)

        scores = scorer.score_batch(interface, outputs)
        uncertainty = scores['uncertainty_score']

        assert uncertainty.shape == (8,)
        assert (uncertainty >= 0).all()
        assert (uncertainty <= 1).all()
        # 高 confidence + 小 residual → uncertainty 应较低
        assert uncertainty.mean().item() < 0.5

    def test_uncertainty_low_conf(self):
        """低 confidence → 高 uncertainty。"""
        cfg = HUFConfig()
        scorer = UpdateReliabilityScorer(cfg)

        interface = _make_interface(confidence_value=0.1)
        outputs = _make_refiner_outputs(interface, residual_scale=2.0)

        scores = scorer.score_batch(interface, outputs)
        uncertainty = scores['uncertainty_score']

        # 低 confidence + 大 residual → uncertainty 应较高
        assert uncertainty.mean().item() > 0.3

    def test_uncertainty_no_candidates(self):
        """candidate_plans=None 时不报错，权重自动重分配。"""
        cfg = HUFConfig()
        scorer = UpdateReliabilityScorer(cfg)

        interface = _make_interface(include_candidates=False)
        outputs = _make_refiner_outputs(interface)

        scores = scorer.score_batch(interface, outputs)
        assert 'uncertainty_score' in scores
        assert scores['uncertainty_score'].shape == (8,)

    def test_support_small_residual(self):
        """小残差 → 高 support。"""
        cfg = HUFConfig()
        scorer = UpdateReliabilityScorer(cfg)

        interface = _make_interface()
        outputs = _make_refiner_outputs(interface, residual_scale=0.01)

        scores = scorer.score_batch(interface, outputs)
        support = scores['support_score']

        assert support.shape == (8,)
        assert (support >= 0).all()
        assert (support <= 1).all()
        assert support.mean().item() > 0.8

    def test_support_large_residual(self):
        """大残差 → 低 support。"""
        cfg = HUFConfig()
        scorer = UpdateReliabilityScorer(cfg)

        interface = _make_interface()
        outputs = _make_refiner_outputs(interface, residual_scale=10.0)

        scores = scorer.score_batch(interface, outputs)
        support = scores['support_score']

        assert support.mean().item() < 0.3

    def test_drift_smooth_correction(self):
        """平滑修正 → 低 drift。"""
        cfg = HUFConfig()
        scorer = UpdateReliabilityScorer(cfg)

        interface = _make_interface()
        # 极小的残差 → 精炼后轨迹几乎不变
        outputs = _make_refiner_outputs(interface, residual_scale=0.001)

        scores = scorer.score_batch(interface, outputs)
        drift = scores['drift_score']

        assert drift.shape == (8,)
        assert drift.mean().item() < 0.3

    def test_drift_jittery_correction(self):
        """抖动修正 → 高 drift。"""
        cfg = HUFConfig()
        scorer = UpdateReliabilityScorer(cfg)

        interface = _make_interface()
        B, T = 8, 6
        # 构造高抖动的残差：交替正负大幅偏移
        jittery_residual = torch.zeros(B, T, 2)
        for t in range(T):
            sign = 1.0 if t % 2 == 0 else -1.0
            jittery_residual[:, t] = sign * 3.0

        outputs = {
            'residual': jittery_residual,
            'refined_plan': interface.reference_plan + jittery_residual,
            'residual_norm': jittery_residual.norm(dim=-1).mean(dim=-1),
            'refine_score': torch.sigmoid(torch.randn(B)),
        }

        scores = scorer.score_batch(interface, outputs)
        drift = scores['drift_score']

        # 高抖动 → drift 应较高
        assert drift.mean().item() > 0.3


# ============================================================
# 3. HarmfulUpdateFilter 测试
# ============================================================

class TestFilter:
    def test_hard_mask_basic(self):
        """hard mask 在新接口下可运行且返回 bool mask。"""
        cfg = HUFConfig(max_residual_norm=1e9, max_step_disp=1e9)
        huf = HarmfulUpdateFilter(cfg)

        interface = _make_interface(batch_size=8, scene_dim=16, time_steps=6)
        outputs = _make_refiner_outputs(interface, residual_scale=0.01)
        scores = {
            'uncertainty_score': torch.rand(8),
            'support_score': torch.rand(8),
            'drift_score': torch.rand(8),
            'pred_gain': torch.full((8,), 1e6),
            'pred_risk': torch.zeros(8),
        }

        reward = torch.ones(8)
        ref_reward = torch.zeros(8)
        mask = huf.compute_mask(scores, interface, outputs, reward=reward, ref_reward=ref_reward)
        assert mask.dtype == torch.bool
        assert mask.shape == (8,)

    def test_min_retention(self):
        """极端情况至少保留 30%。"""
        cfg = HUFConfig(min_retention_ratio=0.3, max_residual_norm=1e9, max_step_disp=1e9)
        huf = HarmfulUpdateFilter(cfg)

        B, D, T = 20, 16, 6
        interface = _make_interface(batch_size=B, scene_dim=D, time_steps=T)
        outputs = _make_refiner_outputs(interface, residual_scale=0.01)
        scores = {
            'uncertainty_score': torch.rand(B) * 0.5 + 0.5,  # 全部 > 0.1
            'support_score': torch.rand(B) * 0.5,            # 全部 < 0.99
            'drift_score': torch.rand(B) * 0.5 + 0.5,       # 全部 > 0.01
            'pred_gain': torch.zeros(B),
            'pred_risk': torch.ones(B),
        }

        reward = torch.ones(B)
        ref_reward = torch.zeros(B)
        mask = huf.compute_mask(scores, interface, outputs, reward=reward, ref_reward=ref_reward)
        retention = mask.float().mean().item()
        assert retention >= 0.3 - 1e-6

    def test_soft_weight_range(self):
        """soft weight 在 (0, 1] 范围内。"""
        cfg = HUFConfig(mode='soft')
        huf = HarmfulUpdateFilter(cfg)

        scores = {
            'uncertainty_score': torch.rand(16),
            'support_score': torch.rand(16),
            'drift_score': torch.rand(16),
        }

        weight = huf.compute_weight(scores)
        assert weight.shape == (16,)
        assert (weight > 0).all()
        assert (weight <= 1.0 + 1e-6).all()

    def test_apply_loss_value(self):
        """filtered loss 数值正确性验证。"""
        cfg = HUFConfig(mode='hard', tau_gain=0.5, tau_risk=1e9, max_residual_norm=1e9, max_step_disp=1e9)
        huf = HarmfulUpdateFilter(cfg)

        per_sample_loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        interface = _make_interface(batch_size=4, scene_dim=16, time_steps=6)
        outputs = _make_refiner_outputs(interface, residual_scale=0.01)
        scores = {
            'uncertainty_score': torch.zeros(4),
            'support_score': torch.ones(4),
            'drift_score': torch.zeros(4),
            'pred_gain': torch.tensor([1.0, 1.0, 0.0, 0.0]),
            'pred_risk': torch.zeros(4),
        }
        reward = torch.ones(4)
        ref_reward = torch.zeros(4)

        filtered_loss, diag = huf.apply_filter(
            per_sample_loss=per_sample_loss,
            scores=scores,
            interface=interface,
            refiner_outputs=outputs,
            reward=reward,
            ref_reward=ref_reward,
        )

        # 前两个被保留: (1.0 + 2.0) / 2 = 1.5
        assert abs(filtered_loss.item() - 1.5) < 1e-5
        assert diag['retention_ratio'] == 0.5

    def test_diagnostics_keys(self):
        """diagnostics dict 包含所有期望的 key。"""
        cfg = HUFConfig(mode='hard')
        huf = HarmfulUpdateFilter(cfg)

        interface = _make_interface(batch_size=8, scene_dim=16, time_steps=6)
        outputs = _make_refiner_outputs(interface, residual_scale=0.01)
        scores = {
            'uncertainty_score': torch.rand(8),
            'support_score': torch.rand(8),
            'drift_score': torch.rand(8),
            'pred_gain': torch.full((8,), 1e6),
            'pred_risk': torch.zeros(8),
        }
        per_sample_loss = torch.rand(8)
        reward = torch.ones(8)
        ref_reward = torch.zeros(8)

        _, diag = huf.apply_filter(
            per_sample_loss=per_sample_loss,
            scores=scores,
            interface=interface,
            refiner_outputs=outputs,
            reward=reward,
            ref_reward=ref_reward,
        )

        assert 'huf_enabled' in diag
        assert 'retention_ratio' in diag
        assert 'n_kept' in diag
        assert 'filter_by_uncertainty' in diag
        assert 'filter_by_support' in diag
        assert 'filter_by_drift' in diag

    def test_stapo_masks_spurious_positive_adv(self):
        """STAPO: 正优势 + 低执行模态概率 + 低熵 → 样本被静音。"""
        cfg = HUFConfig(
            mode='hard',
            stapo_enabled=True,
            stapo_tau_pi=0.4,
            stapo_tau_entropy=0.55,
            stapo_softmax_temp=0.5,
            tau_gain=-1e9,
            tau_risk=1e9,
        )
        huf = HarmfulUpdateFilter(cfg)

        b, m, t = 4, 3, 6
        scene_token = torch.randn(b, 16)
        ref = torch.randn(b, t, 2) * 0.3
        cand = torch.zeros(b, m, t, 2)
        d0 = torch.zeros(b, t, 2)
        d0[:, 0, :] = ref[:, 0, :]
        d0[:, 1:, :] = ref[:, 1:, :] - ref[:, :-1, :]
        cand[:, 0] = d0
        cand[:, 1] = torch.randn(b, t, 2) * 2.0
        cand[:, 2] = torch.randn(b, t, 2) * 2.0
        iface = PlanningInterface(
            scene_token=scene_token,
            reference_plan=ref,
            candidate_plans=cand,
            plan_confidence=torch.ones(b, 1) * 0.9,
            metadata={'executed_mode_index': 1},
        )
        outputs = _make_refiner_outputs(iface, residual_scale=0.01)
        scorer = UpdateReliabilityScorer(cfg)
        scores = scorer.score_batch(iface, outputs)

        reward = torch.ones(b)
        ref_reward = torch.zeros(b)

        mask = huf.compute_mask(
            scores, iface, outputs, reward=reward, ref_reward=ref_reward
        )
        assert not mask.all(), '至少应有一部分样本被 STAPO 静音'


# ============================================================
# 4. 端到端集成测试
# ============================================================

class TestEndToEnd:
    def test_pipeline_backward(self):
        """scorer → filter → loss 完整流水线可 backward。"""
        cfg = HUFConfig(mode='hard')
        scorer = UpdateReliabilityScorer(cfg)
        huf = HarmfulUpdateFilter(cfg)

        B, D, T = 8, 64, 6

        interface = _make_interface(batch_size=B, scene_dim=D, time_steps=T)

        refiner = InterfaceRefiner(scene_dim=D, plan_len=T * 2, hidden_dim=64)
        outputs = refiner(interface)

        # per_sample_loss 需要保留梯度
        gt_plan = interface.reference_plan + torch.randn(B, T, 2) * 0.1
        error = torch.abs(outputs['refined_plan'] - gt_plan)
        per_sample_loss = error.mean(dim=-1).mean(dim=-1)  # [B]

        scores = scorer.score_batch(interface, outputs)
        reward = torch.ones(B)
        ref_reward = torch.zeros(B)
        filtered_loss, diag = huf.apply_filter(
            per_sample_loss=per_sample_loss,
            scores=scores,
            interface=interface,
            refiner_outputs=outputs,
            reward=reward,
            ref_reward=ref_reward,
        )

        # 必须是标量且可 backward
        assert filtered_loss.dim() == 0
        filtered_loss.backward()

        # 检查梯度存在
        for p in refiner.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_per_sample_reward_weighted_error(self):
        """compute_per_sample_reward_weighted_error 数值正确。"""
        B, T = 4, 6
        refined = torch.randn(B, T, 2)
        gt = torch.randn(B, T, 2)
        reward = torch.tensor([0.0, 0.5, 1.0, 0.3])

        per_sample = compute_per_sample_reward_weighted_error(refined, gt, reward)

        assert per_sample.shape == (B,)
        assert (per_sample >= 0).all()
        # 高 reward 的样本权重应更低
        # reward=1.0 (idx=2) 归一化后 weight=0 → per_sample[2] ≈ 0
        assert per_sample[2].item() < per_sample[0].item()
