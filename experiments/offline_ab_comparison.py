"""离线 A/B 对比（支持真实 VAD dump 数据）。

在真实 VAD 推理输出上对比:
  Version A: mean pooled scene_token + refiner 只用 scene + plan
  Version B: grid pooled scene_token + refiner 消费完整接口

验证目标:
  在真实 40000 BEV token 数据上，grid pooling 的优势是否更显著。

使用:
    python -m E2E_RL.experiments.offline_ab_comparison \
        --dump_dir E2E_RL/data/vad_dumps \
        --num_steps 300 \
        --num_trials 3

    # 可选: 限制样本数以快速调试
    --max_samples 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.experiments.load_dump import load_all_samples, batch_from_samples
from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.losses import supervised_refinement_loss
from E2E_RL.evaluators.eval_refined import evaluate_refined_plans

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def build_interface_from_batch(
    batch: Dict[str, torch.Tensor],
    strip_aux: bool = False,
) -> PlanningInterface:
    """从 batch_from_samples 的输出构建 PlanningInterface。

    Args:
        batch: batch_from_samples 返回的 dict
        strip_aux: 是否去掉 confidence 和 safety（模拟旧行为）
    """
    scene_token = batch['scene_token']
    reference_plan = batch['reference_plan']

    if strip_aux:
        return PlanningInterface(
            scene_token=scene_token,
            reference_plan=reference_plan,
        )

    # 完整接口
    confidence = batch.get('plan_confidence')
    safety = {}
    for k, v in batch.items():
        if k.startswith('safety_'):
            safety[k] = v
    safety_features = safety if safety else None

    return PlanningInterface(
        scene_token=scene_token,
        reference_plan=reference_plan,
        plan_confidence=confidence,
        safety_features=safety_features,
    )


def train_refiner(
    refiner: InterfaceRefiner,
    interface: PlanningInterface,
    gt_plan: torch.Tensor,
    num_steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> List[float]:
    """训练 refiner（支持 mini-batch）。

    Returns:
        训练过程的 loss 列表
    """
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)
    refiner.train()

    n = interface.scene_token.shape[0]
    losses = []

    for step in range(num_steps):
        # mini-batch 采样
        if n > batch_size:
            idx = torch.randperm(n)[:batch_size]
            mini_interface = PlanningInterface(
                scene_token=interface.scene_token[idx],
                reference_plan=interface.reference_plan[idx],
                plan_confidence=interface.plan_confidence[idx] if interface.plan_confidence is not None else None,
                safety_features={k: v[idx] for k, v in interface.safety_features.items()} if interface.safety_features else None,
            )
            mini_gt = gt_plan[idx]
        else:
            mini_interface = interface
            mini_gt = gt_plan

        outputs = refiner(mini_interface)
        loss = supervised_refinement_loss(outputs['refined_plan'], mini_gt)
        loss = loss + outputs['residual_norm'].mean() * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def evaluate_refiner(
    refiner: InterfaceRefiner,
    interface: PlanningInterface,
    gt_plan: torch.Tensor,
) -> Dict[str, float]:
    """评估 refiner。"""
    refiner.eval()
    with torch.no_grad():
        outputs = refiner(interface)
        refined = outputs['refined_plan']
        baseline = interface.reference_plan
        results = evaluate_refined_plans(baseline, refined, gt_plan)
    return results


def run_offline_ab_comparison(
    dump_dir: str,
    max_samples: Optional[int] = None,
    num_steps: int = 300,
    num_trials: int = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
):
    """执行离线 A/B 对比。"""

    logger.info('=' * 70)
    logger.info('  [真实数据] A/B 对比: mean+partial vs grid+full')
    logger.info('=' * 70)
    logger.info(f'  数据源: {dump_dir}')

    # 加载数据
    samples = load_all_samples(dump_dir, max_samples=max_samples)
    if len(samples) == 0:
        logger.info('无样本可分析')
        return

    configs = {
        'A (mean+partial)': {
            'pool_mode': 'mean',
            'strip_aux': True,
        },
        'B (grid+full)': {
            'pool_mode': 'grid',
            'strip_aux': False,
        },
        'C (ego_local+full)': {
            'pool_mode': 'ego_local',
            'strip_aux': False,
        },
    }

    all_results: Dict[str, Dict[str, list]] = {}

    for trial in range(num_trials):
        torch.manual_seed(trial * 42)
        logger.info(f'\n  --- Trial {trial + 1}/{num_trials} ---')

        for version_name, cfg in configs.items():
            batch = batch_from_samples(samples, pool_mode=cfg['pool_mode'])
            gt_plan = batch.get('ego_fut_trajs')
            if gt_plan is None:
                logger.warning(f'{version_name}: ego_fut_trajs 不存在，跳过')
                continue

            interface = build_interface_from_batch(batch, strip_aux=cfg['strip_aux'])

            scene_dim = interface.scene_token.shape[-1]
            plan_len = interface.reference_plan.shape[1] * 2  # T * 2 (x, y)

            refiner = InterfaceRefiner(
                scene_dim=scene_dim,
                plan_len=plan_len,
                hidden_dim=128,
            )

            # 训练
            losses = train_refiner(
                refiner, interface, gt_plan,
                num_steps=num_steps, lr=lr, batch_size=batch_size,
            )
            logger.info(
                f'  {version_name}: 训练完成, '
                f'final_loss={losses[-1]:.4f}, '
                f'scene_dim={scene_dim}'
            )

            # 评估
            results = evaluate_refiner(refiner, interface, gt_plan)

            if version_name not in all_results:
                all_results[version_name] = {k: [] for k in results}
            for k, v in results.items():
                all_results[version_name][k].append(v)

    # ======== 汇总输出 ========
    logger.info('')
    logger.info('=' * 70)
    logger.info('  汇总结果 (平均 over trials)')
    logger.info('=' * 70)

    metrics_to_show = [
        ('refined_ade', 'ADE'),
        ('refined_fde', 'FDE'),
        ('baseline_ade', 'base_ADE'),
        ('improvement_ade_pct', 'improv%'),
    ]

    header = f'  {"Version":<25s}'
    for _, label in metrics_to_show:
        header += f'{label:>12s}'
    logger.info(header)
    logger.info('  ' + '-' * 70)

    for version_name in configs:
        if version_name not in all_results:
            continue
        metrics = all_results[version_name]
        row = f'  {version_name:<25s}'
        for key, _ in metrics_to_show:
            vals = metrics.get(key, [0])
            avg = sum(vals) / len(vals)
            row += f'{avg:>12.4f}'
        logger.info(row)

    # ======== 关键对比 ========
    logger.info('')
    logger.info('=' * 70)
    logger.info('  关键对比')
    logger.info('=' * 70)

    if 'A (mean+partial)' in all_results and 'B (grid+full)' in all_results:
        a_ade = sum(all_results['A (mean+partial)']['refined_ade']) / num_trials
        b_ade = sum(all_results['B (grid+full)']['refined_ade']) / num_trials

        if b_ade < a_ade:
            improvement = (a_ade - b_ade) / a_ade * 100
            logger.info(f'  B vs A: ADE 改善 {improvement:.1f}% ({a_ade:.4f} → {b_ade:.4f})')
        else:
            degradation = (b_ade - a_ade) / a_ade * 100
            logger.info(f'  B vs A: ADE 退化 {degradation:.1f}% ({a_ade:.4f} → {b_ade:.4f})')

    if 'C (ego_local+full)' in all_results:
        c_ade = sum(all_results['C (ego_local+full)']['refined_ade']) / num_trials
        logger.info(f'  C (ego_local): ADE = {c_ade:.4f}')

    # ======== low-confidence 子集 A/B ========
    logger.info('')
    logger.info('=' * 70)
    logger.info('  Low-confidence 子集 A/B 对比')
    logger.info('=' * 70)

    for pool_mode in ('mean', 'grid'):
        batch = batch_from_samples(samples, pool_mode=pool_mode)
        conf = batch.get('plan_confidence')
        gt_plan = batch.get('ego_fut_trajs')
        ref_plan = batch.get('reference_plan')

        if conf is None or gt_plan is None or ref_plan is None:
            logger.info(f'  {pool_mode}: 数据不完整，跳过')
            continue

        if conf.dim() == 2:
            conf = conf.squeeze(-1)

        k = max(1, int(len(conf) * 0.2))
        low_idx = conf.argsort()[:k]

        # baseline ADE on low-conf subset
        low_ade = torch.norm(ref_plan[low_idx] - gt_plan[low_idx], dim=-1).mean().item()
        logger.info(f'  {pool_mode} pool, low-conf subset ({k} samples): baseline ADE = {low_ade:.4f}')

    logger.info('=' * 70)
    logger.info('')


def main():
    parser = argparse.ArgumentParser(
        description='离线 A/B 对比（真实 VAD 数据）'
    )
    parser.add_argument('--dump_dir', type=str, required=True,
                        help='dump_vad_inference.py 的输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大加载样本数')
    parser.add_argument('--num_steps', type=int, default=300,
                        help='训练步数')
    parser.add_argument('--num_trials', type=int, default=3,
                        help='重复试验次数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Mini-batch 大小')
    args = parser.parse_args()

    run_offline_ab_comparison(
        dump_dir=args.dump_dir,
        max_samples=args.max_samples,
        num_steps=args.num_steps,
        num_trials=args.num_trials,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
