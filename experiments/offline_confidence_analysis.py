"""离线 confidence 有效性分析（支持真实 VAD dump 数据）。

在真实 VAD 推理输出上验证:
1. plan_confidence 值分布是否有区分度
2. confidence 分桶 vs planning error (ADE) 的相关性
3. low-confidence 子集的 error 是否显著更高
4. Pearson 相关系数

使用:
    # 真实数据（从 dump_vad_inference.py 导出的目录）
    python -m E2E_RL.experiments.offline_confidence_analysis \
        --dump_dir E2E_RL/data/vad_dumps

    # 可选参数
    --max_samples 500      # 限制样本数
    --num_buckets 5        # 分桶数
    --pool_mode mean       # 使用哪种池化方式的 interface
    --low_conf_pct 0.2     # low-confidence 子集百分比
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.experiments.load_dump import load_all_samples, batch_from_samples

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_per_sample_ade(
    reference_plan: torch.Tensor,
    gt_plan: torch.Tensor,
) -> torch.Tensor:
    """计算逐样本 ADE: mean L2 over timesteps。

    Args:
        reference_plan: [N, T, 2]
        gt_plan: [N, T, 2]

    Returns:
        [N] 的 ADE
    """
    return torch.norm(reference_plan - gt_plan, dim=-1).mean(dim=-1)


def compute_per_sample_fde(
    reference_plan: torch.Tensor,
    gt_plan: torch.Tensor,
) -> torch.Tensor:
    """计算逐样本 FDE: 最后一个时间步的 L2。"""
    return torch.norm(reference_plan[:, -1] - gt_plan[:, -1], dim=-1)


def bucket_analysis(
    values: torch.Tensor,
    metric: torch.Tensor,
    num_buckets: int = 5,
) -> List[Dict[str, float]]:
    """将 values 分桶，统计每桶的 metric。"""
    sorted_idx = values.argsort()
    bucket_size = len(values) // num_buckets
    results = []

    for i in range(num_buckets):
        start = i * bucket_size
        end = start + bucket_size if i < num_buckets - 1 else len(values)
        idx = sorted_idx[start:end]

        results.append({
            'conf_min': values[idx].min().item(),
            'conf_max': values[idx].max().item(),
            'conf_mean': values[idx].mean().item(),
            'metric_mean': metric[idx].mean().item(),
            'metric_std': metric[idx].std().item(),
            'count': len(idx),
        })

    return results


def run_offline_confidence_analysis(
    dump_dir: str,
    max_samples: Optional[int] = None,
    num_buckets: int = 5,
    pool_mode: str = 'mean',
    low_conf_pct: float = 0.2,
):
    """执行离线 confidence 分析。"""

    # ---- 加载数据 ----
    samples = load_all_samples(dump_dir, max_samples=max_samples)
    if len(samples) == 0:
        logger.info('无样本可分析')
        return

    batch = batch_from_samples(samples, pool_mode=pool_mode)

    confidence = batch.get('plan_confidence')
    reference_plan = batch.get('reference_plan')
    gt_plan = batch.get('ego_fut_trajs')

    if confidence is None:
        logger.info('plan_confidence 不存在，无法分析')
        return
    if reference_plan is None or gt_plan is None:
        logger.info('reference_plan 或 ego_fut_trajs 不存在，无法分析')
        return

    # squeeze confidence: [N, 1] → [N]
    if confidence.dim() == 2:
        confidence = confidence.squeeze(-1)

    n_samples = len(confidence)
    ade = compute_per_sample_ade(reference_plan, gt_plan)
    fde = compute_per_sample_fde(reference_plan, gt_plan)

    # ======== 1. 基本统计 ========
    logger.info('')
    logger.info('=' * 65)
    logger.info('  [真实数据] plan_confidence 基本统计')
    logger.info('=' * 65)
    logger.info(f'  数据源:    {dump_dir}')
    logger.info(f'  池化方式:  {pool_mode}')
    logger.info(f'  样本数:    {n_samples}')
    logger.info(f'  conf 均值:  {confidence.mean().item():.6f}')
    logger.info(f'  conf 标准差: {confidence.std().item():.6f}')
    logger.info(f'  conf 最小:  {confidence.min().item():.6f}')
    logger.info(f'  conf 最大:  {confidence.max().item():.6f}')
    logger.info(f'  conf 中位:  {confidence.median().item():.6f}')
    logger.info(f'  ADE 均值:   {ade.mean().item():.4f}')
    logger.info(f'  FDE 均值:   {fde.mean().item():.4f}')

    # 方差过小警告
    if confidence.std().item() < 1e-4:
        logger.info('')
        logger.info('  !! WARNING: confidence 方差极小 (<1e-4)')
        logger.info('  在真实数据上，mode 间方差可能接近常数')
        logger.info('  当前 confidence 定义 (exp(-var)) 可能退化为伪特征')
        logger.info('  建议: 检查 ego_fut_preds 各模式是否足够多样化')

    # ======== 2. Confidence 分桶 vs ADE ========
    logger.info('')
    logger.info('=' * 65)
    logger.info('  Confidence 分桶 vs Planning Error (ADE)')
    logger.info('=' * 65)
    header = f'  {"Bucket":>6s}  {"conf_range":>20s}  {"conf_mean":>10s}  {"ADE_mean":>10s}  {"ADE_std":>8s}  {"N":>5s}'
    logger.info(header)
    logger.info('  ' + '-' * 63)

    buckets = bucket_analysis(confidence, ade, num_buckets)
    for i, b in enumerate(buckets):
        conf_range = f'[{b["conf_min"]:.4f}, {b["conf_max"]:.4f}]'
        logger.info(
            f'  {i+1:>6d}  {conf_range:>20s}  {b["conf_mean"]:>10.4f}  '
            f'{b["metric_mean"]:>10.4f}  {b["metric_std"]:>8.4f}  {b["count"]:>5d}'
        )

    # ======== 3. Confidence 分桶 vs FDE ========
    logger.info('')
    logger.info('=' * 65)
    logger.info('  Confidence 分桶 vs FDE')
    logger.info('=' * 65)
    header = f'  {"Bucket":>6s}  {"conf_range":>20s}  {"conf_mean":>10s}  {"FDE_mean":>10s}  {"FDE_std":>8s}'
    logger.info(header)
    logger.info('  ' + '-' * 58)

    fde_buckets = bucket_analysis(confidence, fde, num_buckets)
    for i, b in enumerate(fde_buckets):
        conf_range = f'[{b["conf_min"]:.4f}, {b["conf_max"]:.4f}]'
        logger.info(
            f'  {i+1:>6d}  {conf_range:>20s}  {b["conf_mean"]:>10.4f}  '
            f'{b["metric_mean"]:>10.4f}  {b["metric_std"]:>8.4f}'
        )

    # ======== 4. Pearson 相关系数 ========
    logger.info('')
    logger.info('=' * 65)
    logger.info('  Confidence vs Error 相关性')
    logger.info('=' * 65)

    conf_np = confidence.numpy()
    ade_np = ade.numpy()
    fde_np = fde.numpy()

    pearson_ade = np.corrcoef(conf_np, ade_np)[0, 1]
    pearson_fde = np.corrcoef(conf_np, fde_np)[0, 1]

    logger.info(f'  Pearson(confidence, ADE): {pearson_ade:.4f}')
    logger.info(f'  Pearson(confidence, FDE): {pearson_fde:.4f}')

    if abs(pearson_ade) > 0.3:
        direction = '负相关 (符合预期)' if pearson_ade < 0 else '正相关 (不符合预期!)'
        logger.info(f'  ADE 解读: 强相关 — {direction}')
    elif abs(pearson_ade) > 0.1:
        logger.info(f'  ADE 解读: 弱相关，confidence 有一定信号但不够强')
    else:
        logger.info(f'  ADE 解读: 基本不相关，confidence 对真实数据可能无效')

    # ======== 5. Low-confidence 子集分析 ========
    logger.info('')
    logger.info('=' * 65)
    logger.info(f'  Low-confidence 子集分析 (bottom {low_conf_pct*100:.0f}%)')
    logger.info('=' * 65)

    k = max(1, int(n_samples * low_conf_pct))
    low_idx = confidence.argsort()[:k]
    high_idx = confidence.argsort()[-k:]

    logger.info(f'  子集大小: {k}')
    logger.info(f'  Low-conf  ADE: {ade[low_idx].mean().item():.4f} (std {ade[low_idx].std().item():.4f})')
    logger.info(f'  High-conf ADE: {ade[high_idx].mean().item():.4f} (std {ade[high_idx].std().item():.4f})')
    logger.info(f'  Low-conf  FDE: {fde[low_idx].mean().item():.4f}')
    logger.info(f'  High-conf FDE: {fde[high_idx].mean().item():.4f}')

    ade_ratio = ade[low_idx].mean().item() / max(ade[high_idx].mean().item(), 1e-8)
    logger.info(f'  Low/High ADE 比值: {ade_ratio:.2f}x')

    if ade_ratio > 1.3:
        logger.info('  → low-confidence 子集明显更难，confidence 是有效的难度指标')
    elif ade_ratio > 1.1:
        logger.info('  → low-confidence 子集略难，confidence 有一定区分能力')
    else:
        logger.info('  → low-confidence 与 high-confidence 无显著差异')

    # ======== 6. 总结判定 ========
    logger.info('')
    logger.info('=' * 65)
    logger.info('  判定结论')
    logger.info('=' * 65)

    is_discriminative = confidence.std().item() > 0.01
    is_correlated = abs(pearson_ade) > 0.15
    is_subset_valid = ade_ratio > 1.2

    verdict_count = sum([is_discriminative, is_correlated, is_subset_valid])
    if verdict_count >= 2:
        logger.info('  plan_confidence: 在真实数据上是有效信号')
        logger.info(f'  - 区分度: {"PASS" if is_discriminative else "FAIL"} (std={confidence.std().item():.4f})')
        logger.info(f'  - 相关性: {"PASS" if is_correlated else "FAIL"} (r={pearson_ade:.4f})')
        logger.info(f'  - 子集验证: {"PASS" if is_subset_valid else "FAIL"} (ratio={ade_ratio:.2f})')
    elif verdict_count == 1:
        logger.info('  plan_confidence: 在真实数据上信号较弱，需要改进')
    else:
        logger.info('  plan_confidence: 在真实数据上无效')
        logger.info('  需要重新设计 confidence 代理信号')
    logger.info('=' * 65)
    logger.info('')


def main():
    parser = argparse.ArgumentParser(
        description='离线 confidence 有效性分析（真实 VAD 数据）'
    )
    parser.add_argument('--dump_dir', type=str, required=True,
                        help='dump_vad_inference.py 的输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大加载样本数')
    parser.add_argument('--num_buckets', type=int, default=5,
                        help='分桶数')
    parser.add_argument('--pool_mode', type=str, default='mean',
                        choices=['mean', 'grid', 'ego_local'],
                        help='使用哪种池化方式的 interface')
    parser.add_argument('--low_conf_pct', type=float, default=0.2,
                        help='low-confidence 子集百分比')
    args = parser.parse_args()

    run_offline_confidence_analysis(
        dump_dir=args.dump_dir,
        max_samples=args.max_samples,
        num_buckets=args.num_buckets,
        pool_mode=args.pool_mode,
        low_conf_pct=args.low_conf_pct,
    )


if __name__ == '__main__':
    main()
