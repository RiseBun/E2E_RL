"""离线 HUF 对比实验（支持真实 VAD dump 数据）。

在真实 VAD 推理输出上对比:
  Baseline:          VAD 原始输出
  Refiner:           InterfaceRefiner (无过滤)
  Refiner + HUF_hard: + hard mask 过滤
  Refiner + HUF_soft: + soft weight 过滤

验证目标:
  有害更新过滤是否能提升 RL 训练稳定性和规划质量。

使用:
    python -m E2E_RL.experiments.offline_huf_experiment \
        --dump_dir E2E_RL/data/vad_dumps \
        --num_steps 300 --num_trials 1 \
        --pool_mode grid
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.experiments.load_dump import batch_from_samples, load_all_samples
from E2E_RL.experiments.offline_ab_comparison import (
    build_interface_from_batch,
    evaluate_refiner,
)
from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.losses import (
    compute_per_sample_reward_weighted_error,
    supervised_refinement_loss,
)
from E2E_RL.refinement.reward_proxy import compute_refinement_reward
from E2E_RL.evaluators.eval_refined import evaluate_refined_plans
from E2E_RL.update_filter.config import HUFConfig
from E2E_RL.update_filter.filter import HarmfulUpdateFilter
from E2E_RL.update_filter.scorer import UpdateReliabilityScorer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def train_refiner_no_filter(
    refiner: InterfaceRefiner,
    interface: PlanningInterface,
    gt_plan: torch.Tensor,
    num_steps: int = 300,
    warmup_ratio: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Dict[str, List[float]]:
    """训练 refiner（无 HUF），两阶段：supervised + reward-weighted。

    Returns:
        训练历史 dict
    """
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)
    refiner.train()

    n = interface.scene_token.shape[0]
    warmup_steps = int(num_steps * warmup_ratio)
    history: Dict[str, List[float]] = {'losses': [], 'loss_vars': []}

    for step in range(num_steps):
        # mini-batch 采样
        if n > batch_size:
            idx = torch.randperm(n)[:batch_size]
            mini_interface = PlanningInterface(
                scene_token=interface.scene_token[idx],
                reference_plan=interface.reference_plan[idx],
                candidate_plans=interface.candidate_plans[idx] if interface.candidate_plans is not None else None,
                plan_confidence=interface.plan_confidence[idx] if interface.plan_confidence is not None else None,
                safety_features={k: v[idx] for k, v in interface.safety_features.items()} if interface.safety_features else None,
            )
            mini_gt = gt_plan[idx]
        else:
            mini_interface = interface
            mini_gt = gt_plan

        outputs = refiner(mini_interface)
        refined_plan = outputs['refined_plan']

        if step < warmup_steps:
            # Stage 1: supervised
            loss = supervised_refinement_loss(refined_plan, mini_gt)
        else:
            # Stage 2: reward-weighted
            reward_info = compute_refinement_reward(
                refined_plan=refined_plan.detach(),
                gt_plan=mini_gt,
            )
            per_sample = compute_per_sample_reward_weighted_error(
                refined_plan, mini_gt, reward_info['total_reward']
            )
            loss = per_sample.mean()

        loss = loss + outputs['residual_norm'].mean() * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history['losses'].append(loss.item())

    return history


def train_refiner_with_huf(
    refiner: InterfaceRefiner,
    interface: PlanningInterface,
    gt_plan: torch.Tensor,
    huf_config: HUFConfig,
    num_steps: int = 300,
    warmup_ratio: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Dict[str, List[float]]:
    """训练 refiner + HUF 过滤，两阶段：supervised + filtered reward。

    Returns:
        训练历史 dict
    """
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)
    scorer = UpdateReliabilityScorer(huf_config)
    huf = HarmfulUpdateFilter(huf_config)
    refiner.train()

    n = interface.scene_token.shape[0]
    warmup_steps = int(num_steps * warmup_ratio)

    history: Dict[str, List[float]] = {
        'losses': [],
        'retention_ratios': [],
        'mean_uncertainty_kept': [],
        'mean_support_kept': [],
        'mean_drift_kept': [],
    }

    for step in range(num_steps):
        # mini-batch 采样
        if n > batch_size:
            idx = torch.randperm(n)[:batch_size]
            mini_interface = PlanningInterface(
                scene_token=interface.scene_token[idx],
                reference_plan=interface.reference_plan[idx],
                candidate_plans=interface.candidate_plans[idx] if interface.candidate_plans is not None else None,
                plan_confidence=interface.plan_confidence[idx] if interface.plan_confidence is not None else None,
                safety_features={k: v[idx] for k, v in interface.safety_features.items()} if interface.safety_features else None,
            )
            mini_gt = gt_plan[idx]
        else:
            mini_interface = interface
            mini_gt = gt_plan

        outputs = refiner(mini_interface)
        refined_plan = outputs['refined_plan']

        if step < warmup_steps:
            # Stage 1: supervised (无过滤)
            loss = supervised_refinement_loss(refined_plan, mini_gt)
            loss = loss + outputs['residual_norm'].mean() * 0.01

            history['losses'].append(loss.item())
            history['retention_ratios'].append(1.0)
            history['mean_uncertainty_kept'].append(0.0)
            history['mean_support_kept'].append(1.0)
            history['mean_drift_kept'].append(0.0)
        else:
            # Stage 2: reward-weighted + HUF
            reward_info = compute_refinement_reward(
                refined_plan=refined_plan.detach(),
                gt_plan=mini_gt,
            )
            ref_reward_info = compute_refinement_reward(
                refined_plan=mini_interface.reference_plan.detach(),
                gt_plan=mini_gt,
            )
            per_sample_loss = compute_per_sample_reward_weighted_error(
                refined_plan, mini_gt, reward_info['total_reward']
            )

            # HUF 评分和过滤
            scores = scorer.score_batch(mini_interface, outputs)
            filtered_loss, diag = huf.apply_filter(
                per_sample_loss=per_sample_loss,
                scores=scores,
                interface=mini_interface,
                refiner_outputs=outputs,
                reward=reward_info['total_reward'],
                ref_reward=ref_reward_info['total_reward'],
            )

            loss = filtered_loss + outputs['residual_norm'].mean() * 0.01

            history['losses'].append(loss.item())
            history['retention_ratios'].append(diag.get('retention_ratio', 1.0))
            history['mean_uncertainty_kept'].append(
                diag.get('mean_uncertainty_kept', 0.0)
            )
            history['mean_support_kept'].append(
                diag.get('mean_support_kept', 1.0)
            )
            history['mean_drift_kept'].append(
                diag.get('mean_drift_kept', 0.0)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return history


def run_offline_huf_experiment(
    dump_dir: str,
    max_samples: Optional[int] = None,
    num_steps: int = 300,
    num_trials: int = 1,
    lr: float = 1e-3,
    batch_size: int = 64,
    pool_mode: str = 'grid',
    huf_modes: Optional[List[str]] = None,
) -> None:
    """执行离线 HUF 对比实验。"""

    if huf_modes is None:
        huf_modes = ['hard', 'soft']

    logger.info('=' * 70)
    logger.info('  [离线 HUF 实验] Harmful Update Filtering 对比')
    logger.info('=' * 70)
    logger.info(f'  数据源: {dump_dir}')
    logger.info(f'  池化模式: {pool_mode}')
    logger.info(f'  训练步数: {num_steps}, trials: {num_trials}')

    # 加载数据
    samples = load_all_samples(dump_dir, max_samples=max_samples)
    if len(samples) == 0:
        logger.info('无样本可分析')
        return

    batch = batch_from_samples(samples, pool_mode=pool_mode)
    gt_plan = batch.get('ego_fut_trajs')
    if gt_plan is None:
        logger.error('ego_fut_trajs 不存在')
        return

    # 构建 interface（填充 candidate_plans）
    interface = build_interface_from_batch(batch, strip_aux=False)
    ego_fut_preds = batch.get('ego_fut_preds')
    if ego_fut_preds is not None:
        interface.candidate_plans = ego_fut_preds

    scene_dim = interface.scene_token.shape[-1]
    plan_len = interface.reference_plan.shape[1] * 2

    logger.info(f'  样本数: {gt_plan.shape[0]}, scene_dim: {scene_dim}')

    # ======== 配置对比版本 ========
    configs: Dict[str, Optional[HUFConfig]] = {
        'Refiner (无过滤)': None,
    }
    for mode in huf_modes:
        configs[f'Refiner + HUF_{mode}'] = HUFConfig(mode=mode)

    all_results: Dict[str, Dict[str, list]] = {}
    all_histories: Dict[str, Dict[str, List[float]]] = {}

    for trial in range(num_trials):
        torch.manual_seed(trial * 42)
        logger.info(f'\n  --- Trial {trial + 1}/{num_trials} ---')

        for version_name, huf_cfg in configs.items():
            # 创建新的 refiner
            refiner = InterfaceRefiner(
                scene_dim=scene_dim,
                plan_len=plan_len,
                hidden_dim=128,
            )

            if huf_cfg is None:
                # 无过滤版本
                history = train_refiner_no_filter(
                    refiner, interface, gt_plan,
                    num_steps=num_steps, lr=lr, batch_size=batch_size,
                )
            else:
                # HUF 过滤版本
                history = train_refiner_with_huf(
                    refiner, interface, gt_plan, huf_cfg,
                    num_steps=num_steps, lr=lr, batch_size=batch_size,
                )

            final_loss = history['losses'][-1] if history['losses'] else float('nan')
            logger.info(f'  {version_name}: 训练完成, final_loss={final_loss:.4f}')

            # 评估
            results = evaluate_refiner(refiner, interface, gt_plan)

            if version_name not in all_results:
                all_results[version_name] = {k: [] for k in results}
            for k, v in results.items():
                all_results[version_name][k].append(v)

            if version_name not in all_histories:
                all_histories[version_name] = history
            else:
                for k in history:
                    all_histories[version_name].setdefault(k, []).extend(history[k])

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

    header = f'  {"Version":<30s}'
    for _, label in metrics_to_show:
        header += f'{label:>12s}'
    logger.info(header)
    logger.info('  ' + '-' * 78)

    for version_name in configs:
        if version_name not in all_results:
            continue
        metrics = all_results[version_name]
        row = f'  {version_name:<30s}'
        for key, _ in metrics_to_show:
            vals = metrics.get(key, [0])
            avg = sum(vals) / len(vals)
            row += f'{avg:>12.4f}'
        logger.info(row)

    # ======== HUF 诊断信息 ========
    logger.info('')
    logger.info('=' * 70)
    logger.info('  HUF 诊断信息')
    logger.info('=' * 70)

    for version_name, history in all_histories.items():
        if 'retention_ratios' not in history:
            continue

        ratios = history['retention_ratios']
        # 只看 Stage 2 部分（warmup 之后）
        warmup_end = int(len(ratios) * 0.3)
        stage2_ratios = ratios[warmup_end:]

        if not stage2_ratios:
            continue

        avg_retention = sum(stage2_ratios) / len(stage2_ratios)

        # Stage 2 的 loss 方差
        stage2_losses = history['losses'][warmup_end:]
        if len(stage2_losses) > 1:
            loss_t = torch.tensor(stage2_losses)
            loss_var = loss_t.var().item()
        else:
            loss_var = 0.0

        logger.info(f'  {version_name}:')
        logger.info(f'    平均保留比例 (Stage 2): {avg_retention:.2%}')
        logger.info(f'    Stage 2 loss 方差: {loss_var:.6f}')

        # 被保留/过滤样本的平均分数
        stage2_unc = history.get('mean_uncertainty_kept', [])[warmup_end:]
        stage2_sup = history.get('mean_support_kept', [])[warmup_end:]
        stage2_dri = history.get('mean_drift_kept', [])[warmup_end:]

        if stage2_unc:
            logger.info(
                f'    保留样本平均 uncertainty: {sum(stage2_unc)/len(stage2_unc):.4f}'
            )
        if stage2_sup:
            logger.info(
                f'    保留样本平均 support: {sum(stage2_sup)/len(stage2_sup):.4f}'
            )
        if stage2_dri:
            logger.info(
                f'    保留样本平均 drift: {sum(stage2_dri)/len(stage2_dri):.4f}'
            )

    # ======== 无过滤版本的 loss 方差作为对照 ========
    if 'Refiner (无过滤)' in all_histories:
        no_filter_losses = all_histories['Refiner (无过滤)']['losses']
        warmup_end = int(len(no_filter_losses) * 0.3)
        stage2_losses = no_filter_losses[warmup_end:]
        if len(stage2_losses) > 1:
            loss_var = torch.tensor(stage2_losses).var().item()
            logger.info(f'  Refiner (无过滤):')
            logger.info(f'    Stage 2 loss 方差: {loss_var:.6f}')

    # ======== 关键对比 ========
    logger.info('')
    logger.info('=' * 70)
    logger.info('  关键对比')
    logger.info('=' * 70)

    base_name = 'Refiner (无过滤)'
    if base_name in all_results:
        base_ade = sum(all_results[base_name]['refined_ade']) / num_trials

        for version_name in configs:
            if version_name == base_name or version_name not in all_results:
                continue
            ver_ade = sum(all_results[version_name]['refined_ade']) / num_trials
            if ver_ade < base_ade:
                improvement = (base_ade - ver_ade) / base_ade * 100
                logger.info(
                    f'  {version_name} vs 无过滤: '
                    f'ADE 改善 {improvement:.1f}% ({base_ade:.4f} -> {ver_ade:.4f})'
                )
            else:
                degradation = (ver_ade - base_ade) / base_ade * 100
                logger.info(
                    f'  {version_name} vs 无过滤: '
                    f'ADE 退化 {degradation:.1f}% ({base_ade:.4f} -> {ver_ade:.4f})'
                )

    logger.info('=' * 70)
    logger.info('')


def main():
    parser = argparse.ArgumentParser(
        description='离线 HUF 对比实验（真实 VAD 数据）'
    )
    parser.add_argument('--dump_dir', type=str, required=True,
                        help='dump_vad_inference.py 的输出目录')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pool_mode', type=str, default='grid',
                        choices=['mean', 'grid', 'ego_local'])
    parser.add_argument('--huf_mode', type=str, default='hard,soft',
                        help='逗号分隔的 HUF 模式列表')
    args = parser.parse_args()

    huf_modes = [m.strip() for m in args.huf_mode.split(',')]

    run_offline_huf_experiment(
        dump_dir=args.dump_dir,
        max_samples=args.max_samples,
        num_steps=args.num_steps,
        num_trials=args.num_trials,
        lr=args.lr,
        batch_size=args.batch_size,
        pool_mode=args.pool_mode,
        huf_modes=huf_modes,
    )


if __name__ == '__main__':
    main()
