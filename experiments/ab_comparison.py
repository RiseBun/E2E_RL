"""A/B 对比: old refiner vs full-interface refiner。

Version A: mean pooled scene token + refiner 只用 scene + plan
Version B: grid pooled scene token + refiner 用 scene + plan + confidence + safety

验证目标:
  接口不仅"结构成立"，而且"功能可用"——
  full-interface refiner 比 partial-consumer refiner 更好。

使用:
    python -m E2E_RL.experiments.ab_comparison
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.planning_interface.adapters.vad_adapter import VADPlanningAdapter
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.losses import supervised_refinement_loss
from E2E_RL.refinement.reward_proxy import compute_refinement_reward
from E2E_RL.evaluators.eval_refined import evaluate_refined_plans

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ---- 数据生成 ----

BATCH = 64
BEV_SIDE = 10            # 10x10=100 tokens（调试用）
BEV_TOKENS = BEV_SIDE ** 2
EMBED = 256
FUT_TS = 6
EGO_MODES = 3


def make_vad_outputs(
    batch_size: int = BATCH,
    difficulty_correlated: bool = True,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """生成模拟 VAD 输出。

    Args:
        difficulty_correlated: 是否让 mode 方差和实际 error 相关。

    Returns:
        (vad_outs, gt_plan, ego_fut_cmd)
    """
    difficulties = torch.rand(batch_size)

    ego_preds_list = []
    gt_list = []
    for d in difficulties:
        spread = d.item() * 1.5 if difficulty_correlated else 0.3
        noise = d.item() * 0.4 if difficulty_correlated else 0.2
        base = torch.tensor([[0.5, 0.0]] * FUT_TS)
        modes = torch.stack([base + torch.randn_like(base) * spread for _ in range(EGO_MODES)])
        ego_preds_list.append(modes)
        gt_list.append(base.cumsum(dim=0) + torch.randn_like(base) * noise)

    vad_outs = {
        'bev_embed': torch.randn(batch_size, BEV_TOKENS, EMBED),
        'ego_fut_preds': torch.stack(ego_preds_list),
        'all_cls_scores': torch.randn(6, batch_size, 50, 10),
        'map_all_cls_scores': torch.randn(6, batch_size, 20, 3),
    }

    gt = torch.stack(gt_list)
    cmd = torch.zeros(batch_size, EGO_MODES)
    cmd[:, 0] = 1.0

    return vad_outs, gt, cmd


def train_and_eval(
    refiner: InterfaceRefiner,
    interface: PlanningInterface,
    gt_plan: torch.Tensor,
    num_steps: int = 200,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """短训练 + 评估。"""
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)
    refiner.train()

    for step in range(num_steps):
        outputs = refiner(interface)
        loss = supervised_refinement_loss(outputs['refined_plan'], gt_plan)
        loss = loss + outputs['residual_norm'].mean() * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估
    refiner.eval()
    with torch.no_grad():
        outputs = refiner(interface)
        refined = outputs['refined_plan']
        baseline = interface.reference_plan

        results = evaluate_refined_plans(baseline, refined, gt_plan)

        reward_info = compute_refinement_reward(refined, gt_plan)
        results['mean_reward'] = reward_info['total_reward'].mean().item()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=200)
    parser.add_argument('--num_trials', type=int, default=3)
    args = parser.parse_args()

    logger.info('=' * 70)
    logger.info('  A/B 对比: old refiner vs full-interface refiner')
    logger.info('=' * 70)

    configs = {
        'Version A (mean + partial)': {
            'scene_pool': 'mean',
            'description': 'mean pooled scene, refiner 只用 scene+plan (confidence/safety 不提供)',
            'strip_aux': True,  # 去掉 confidence/safety 让旧行为
        },
        'Version B (grid + full)': {
            'scene_pool': 'grid',
            'description': 'grid pooled scene, refiner 消费完整接口',
            'strip_aux': False,
        },
    }

    # 收集多次试验
    all_results: Dict[str, Dict[str, list]] = {}

    for trial in range(args.num_trials):
        torch.manual_seed(trial * 42)
        vad_outs, gt, cmd = make_vad_outputs()

        for version_name, cfg in configs.items():
            adapter = VADPlanningAdapter(
                scene_pool=cfg['scene_pool'],
                grid_size=4,
            )
            interface = adapter.extract(vad_outs, ego_fut_cmd=cmd)

            if cfg['strip_aux']:
                # Version A: 模拟旧行为，去掉 confidence 和 safety
                interface = PlanningInterface(
                    scene_token=interface.scene_token,
                    reference_plan=interface.reference_plan,
                )

            scene_dim = interface.scene_token.shape[-1]
            plan_len = FUT_TS * 2

            refiner = InterfaceRefiner(
                scene_dim=scene_dim,
                plan_len=plan_len,
                hidden_dim=128,
            )

            results = train_and_eval(
                refiner, interface, gt,
                num_steps=args.num_steps,
            )

            if version_name not in all_results:
                all_results[version_name] = {k: [] for k in results}
            for k, v in results.items():
                all_results[version_name][k].append(v)

    # ---- 输出 ----
    logger.info('')
    metrics_to_show = [
        ('refined_ade', 'ADE'),
        ('refined_fde', 'FDE'),
        ('refined_l2', 'L2'),
        ('baseline_ade', 'base_ADE'),
        ('improvement_ade_pct', 'improv%'),
        ('mean_reward', 'reward'),
    ]

    header = f'{"Version":<35s}'
    for _, label in metrics_to_show:
        header += f'{label:>10s}'
    logger.info(header)
    logger.info('-' * 70)

    for version_name in configs:
        metrics = all_results[version_name]
        row = f'{version_name:<35s}'
        for key, _ in metrics_to_show:
            vals = metrics.get(key, [0])
            avg = sum(vals) / len(vals)
            row += f'{avg:>10.4f}'
        logger.info(row)

    # ---- 结论 ----
    a_ade = sum(all_results['Version A (mean + partial)']['refined_ade']) / args.num_trials
    b_ade = sum(all_results['Version B (grid + full)']['refined_ade']) / args.num_trials

    logger.info('')
    logger.info('=' * 70)
    if b_ade < a_ade:
        improvement = (a_ade - b_ade) / a_ade * 100
        logger.info(f'  Version B 优于 Version A:  ADE 改善 {improvement:.1f}%')
        logger.info('  结论: 完整接口消费 + grid scene token 提供了有效增益')
        logger.info('  → 接口不仅结构成立，而且功能可用')
    else:
        logger.info('  Version A 与 Version B 无显著差异')
        logger.info('  需要进一步调查 confidence/safety 是否被有效利用')
    logger.info('=' * 70)
    logger.info('')


if __name__ == '__main__':
    main()
