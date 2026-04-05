"""实验 1: 接口字段语义消融。

验证问题: 接口中的各个字段是否有独立的语义贡献？
方法: 对 PlanningInterface 的字段进行 zero-out 消融，
      观察 refiner 输出的变化幅度。

如果某个字段被置零后 refiner 输出几乎不变，
说明该字段在当前设计下没有提供有效信号。

使用方式:
    python -m E2E_RL.experiments.ablation_interface_fields \
        [--use_real_data]  # 默认用合成数据
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.losses import supervised_refinement_loss

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ---- 消融配置 ----

ABLATION_CONFIGS = {
    'full': {
        'description': '完整接口（baseline）',
        'zero_fields': [],
    },
    'no_scene_token': {
        'description': '去掉 scene_token（置零）',
        'zero_fields': ['scene_token'],
    },
    'no_confidence': {
        'description': '去掉 plan_confidence（置零）',
        'zero_fields': ['plan_confidence'],
    },
    'no_safety': {
        'description': '去掉 safety_features（清空）',
        'zero_fields': ['safety_features'],
    },
    'plan_only': {
        'description': '仅保留 reference_plan',
        'zero_fields': ['scene_token', 'plan_confidence', 'safety_features'],
    },
}


def ablate_interface(
    interface: PlanningInterface,
    zero_fields: List[str],
) -> PlanningInterface:
    """对接口字段进行消融（置零或清空）。"""
    kwargs = {
        'scene_token': interface.scene_token.clone(),
        'reference_plan': interface.reference_plan.clone(),
        'candidate_plans': interface.candidate_plans,
        'plan_confidence': interface.plan_confidence.clone() if interface.plan_confidence is not None else None,
        'safety_features': dict(interface.safety_features) if interface.safety_features else None,
        'hard_case_score': interface.hard_case_score,
        'metadata': interface.metadata,
    }

    for field in zero_fields:
        if field == 'scene_token':
            kwargs['scene_token'] = torch.zeros_like(kwargs['scene_token'])
        elif field == 'plan_confidence':
            kwargs['plan_confidence'] = None
        elif field == 'safety_features':
            kwargs['safety_features'] = None

    return PlanningInterface(**kwargs)


def make_synthetic_data(
    batch_size: int = 32,
    scene_dim: int = 256,
    fut_ts: int = 6,
    device: str = 'cpu',
) -> Tuple[PlanningInterface, torch.Tensor]:
    """生成合成数据用于消融实验。"""
    interface = PlanningInterface(
        scene_token=torch.randn(batch_size, scene_dim, device=device),
        reference_plan=torch.randn(batch_size, fut_ts, 2, device=device) * 0.5,
        candidate_plans=torch.randn(batch_size, 3, fut_ts, 2, device=device) * 0.5,
        plan_confidence=torch.rand(batch_size, 1, device=device),
        safety_features={'plan_mode_variance': torch.rand(batch_size, fut_ts, device=device)},
        hard_case_score=torch.rand(batch_size, 1, device=device),
    )
    # GT: 向前直行
    t = torch.arange(1, fut_ts + 1, dtype=torch.float32, device=device)
    gt = torch.stack([t * 0.5, torch.zeros_like(t)], dim=-1)
    gt = gt.unsqueeze(0).expand(batch_size, -1, -1).clone()
    return interface, gt


def run_ablation(
    refiner: InterfaceRefiner,
    interface: PlanningInterface,
    gt_plan: torch.Tensor,
    configs: Dict = ABLATION_CONFIGS,
) -> Dict[str, Dict[str, float]]:
    """执行消融实验。"""
    refiner.eval()
    results = {}

    with torch.no_grad():
        for name, cfg in configs.items():
            ablated = ablate_interface(interface, cfg['zero_fields'])
            outputs = refiner(ablated)
            refined = outputs['refined_plan']
            residual = outputs['residual']

            loss = supervised_refinement_loss(refined, gt_plan).item()
            residual_mag = residual.norm(dim=-1).mean().item()

            # 和 full 版本的输出差异
            if name == 'full':
                full_refined = refined.clone()

            output_diff = (refined - full_refined).norm(dim=-1).mean().item() if name != 'full' else 0.0

            results[name] = {
                'loss': loss,
                'residual_magnitude': residual_mag,
                'output_diff_vs_full': output_diff,
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='接口字段语义消融实验')
    parser.add_argument('--scene_dim', type=int, default=256)
    parser.add_argument('--fut_ts', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_trials', type=int, default=5,
                        help='多次随机初始化取平均')
    args = parser.parse_args()

    plan_len = args.fut_ts * 2

    logger.info('=' * 65)
    logger.info('  实验 1: 接口字段语义消融')
    logger.info('=' * 65)

    # 多次试验取平均
    all_results: Dict[str, Dict[str, List[float]]] = {}

    for trial in range(args.num_trials):
        torch.manual_seed(trial * 42)

        refiner = InterfaceRefiner(
            scene_dim=args.scene_dim,
            plan_len=plan_len,
            hidden_dim=args.hidden_dim,
        )
        interface, gt = make_synthetic_data(
            args.batch_size, args.scene_dim, args.fut_ts,
        )

        results = run_ablation(refiner, interface, gt)

        for name, metrics in results.items():
            if name not in all_results:
                all_results[name] = {k: [] for k in metrics}
            for k, v in metrics.items():
                all_results[name][k].append(v)

    # 汇总输出
    logger.info('')
    logger.info(f'{"配置":<25s} {"loss":>10s} {"residual":>10s} {"diff_vs_full":>14s}')
    logger.info('-' * 65)

    for name, cfg in ABLATION_CONFIGS.items():
        metrics = all_results[name]
        avg_loss = sum(metrics['loss']) / len(metrics['loss'])
        avg_res = sum(metrics['residual_magnitude']) / len(metrics['residual_magnitude'])
        avg_diff = sum(metrics['output_diff_vs_full']) / len(metrics['output_diff_vs_full'])

        marker = ''
        if name != 'full' and avg_diff < 0.01:
            marker = '  <-- 几乎无变化，该字段可能无效'

        logger.info(
            f'{cfg["description"]:<25s} {avg_loss:>10.4f} {avg_res:>10.4f} {avg_diff:>14.4f}{marker}'
        )

    logger.info('')
    logger.info('解读:')
    logger.info('  - diff_vs_full: 消融后 refined_plan 与 full 版本的 L2 差异')
    logger.info('  - 差异越大 → 该字段对 refiner 输出影响越大 → 语义贡献越强')
    logger.info('  - 差异接近 0 → 该字段当前未被有效利用')
    logger.info('')


if __name__ == '__main__':
    main()
