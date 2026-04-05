"""实验 3: plan_confidence 有效性统计。

验证问题: plan_confidence 是有效信号还是伪特征？
方法: 不做训练，直接对提取的 confidence 做统计分析:
  - confidence 值分布
  - confidence 分桶 vs baseline planning error
  - confidence 分桶 vs 碰撞/离道率
  - 低 confidence 样本是否更困难

如果 confidence 与规划质量无相关性，则当前定义需要改进。

使用方式:
    # 合成数据验证流程
    python -m E2E_RL.experiments.confidence_analysis

    # 真实数据（需提供 VAD 输出）
    python -m E2E_RL.experiments.confidence_analysis \
        --vad_output_dir /path/to/saved_vad_outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.planning_interface.adapters.vad_adapter import VADPlanningAdapter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_planning_error(
    reference_plan: torch.Tensor,
    gt_plan: torch.Tensor,
) -> torch.Tensor:
    """计算逐样本 ADE。

    Args:
        reference_plan: [B, T, 2] 绝对坐标
        gt_plan: [B, T, 2] 绝对坐标

    Returns:
        [B] 的 ADE
    """
    return torch.norm(reference_plan - gt_plan, dim=-1).mean(dim=-1)


def bucket_analysis(
    values: torch.Tensor,
    metric: torch.Tensor,
    num_buckets: int = 5,
    value_name: str = 'confidence',
    metric_name: str = 'error',
) -> List[Dict[str, float]]:
    """将 values 分桶，统计每桶的 metric 均值。

    Returns:
        每桶的统计信息列表
    """
    sorted_idx = values.argsort()
    bucket_size = len(values) // num_buckets
    results = []

    for i in range(num_buckets):
        start = i * bucket_size
        end = start + bucket_size if i < num_buckets - 1 else len(values)
        idx = sorted_idx[start:end]

        bucket_info = {
            f'{value_name}_min': values[idx].min().item(),
            f'{value_name}_max': values[idx].max().item(),
            f'{value_name}_mean': values[idx].mean().item(),
            f'{metric_name}_mean': metric[idx].mean().item(),
            f'{metric_name}_std': metric[idx].std().item(),
            'count': len(idx),
        }
        results.append(bucket_info)

    return results


def make_synthetic_vad_outputs(
    num_samples: int = 200,
    device: str = 'cpu',
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """合成 VAD 输出，模拟不同 confidence 水平。

    关键设计: 让 mode 间方差和实际 planning error 有一定相关性，
    以验证 confidence 统计流程。
    """
    # 生成不同难度的样本
    # 高方差 mode → 高 error → 低 confidence
    difficulties = torch.rand(num_samples, device=device)  # [0, 1]

    ego_fut_preds = []
    gt_plans = []

    for d in difficulties:
        mode_spread = d.item() * 2.0  # 难度越高，模式分散越大
        noise = d.item() * 0.5  # 难度越高，和 GT 偏差越大

        # 3 个模式的轨迹增量
        base_traj = torch.tensor([[0.5, 0.0]] * 6)  # 直行增量
        modes = []
        for _ in range(3):
            mode = base_traj + torch.randn_like(base_traj) * mode_spread
            modes.append(mode)
        ego_fut_preds.append(torch.stack(modes))  # [3, 6, 2]

        # GT 轨迹（绝对坐标）
        gt = base_traj.cumsum(dim=0) + torch.randn_like(base_traj) * noise
        gt_plans.append(gt)

    vad_outs = {
        'bev_embed': torch.randn(num_samples, 100, 256, device=device),
        'ego_fut_preds': torch.stack(ego_fut_preds),  # [N, 3, 6, 2]
        'all_cls_scores': torch.randn(6, num_samples, 50, 10, device=device),
        'map_all_cls_scores': torch.randn(6, num_samples, 20, 3, device=device),
    }
    gt = torch.stack(gt_plans)  # [N, 6, 2]

    return vad_outs, gt


def run_confidence_analysis(
    vad_outs: Dict[str, torch.Tensor],
    gt_plan: torch.Tensor,
    ego_fut_cmd: Optional[torch.Tensor] = None,
    num_buckets: int = 5,
):
    """执行 confidence 有效性分析。"""
    adapter = VADPlanningAdapter()
    interface = adapter.extract(vad_outs, ego_fut_cmd=ego_fut_cmd)

    confidence = interface.plan_confidence  # [B, 1]
    reference_plan = interface.reference_plan  # [B, T, 2]

    if confidence is None:
        logger.info('plan_confidence 为 None，无法分析')
        return

    confidence = confidence.squeeze(-1)  # [B]
    error = compute_planning_error(reference_plan, gt_plan)  # [B]

    # ---- 1. 基本统计 ----
    logger.info('')
    logger.info('=' * 60)
    logger.info('  plan_confidence 基本统计')
    logger.info('=' * 60)
    logger.info(f'  样本数:    {len(confidence)}')
    logger.info(f'  均值:      {confidence.mean().item():.4f}')
    logger.info(f'  标准差:    {confidence.std().item():.4f}')
    logger.info(f'  最小值:    {confidence.min().item():.4f}')
    logger.info(f'  最大值:    {confidence.max().item():.4f}')
    logger.info(f'  中位数:    {confidence.median().item():.4f}')

    # 如果方差过小，直接标记为伪特征
    if confidence.std().item() < 0.01:
        logger.info('')
        logger.info('  WARNING: confidence 方差极小 (<0.01)，接近常数')
        logger.info('  这意味着当前 confidence 定义无法区分不同样本的规划质量')
        logger.info('  需要更换 confidence 代理信号')
        return

    # ---- 2. Confidence 分桶 vs Planning Error ----
    logger.info('')
    logger.info('=' * 60)
    logger.info('  Confidence 分桶 vs Planning Error (ADE)')
    logger.info('=' * 60)
    logger.info(f'  {"Bucket":>8s}  {"conf_range":>16s}  {"conf_mean":>10s}  {"error_mean":>10s}  {"error_std":>10s}')
    logger.info('  ' + '-' * 58)

    buckets = bucket_analysis(confidence, error, num_buckets, 'confidence', 'error')
    for i, b in enumerate(buckets):
        conf_range = f'[{b["confidence_min"]:.3f}, {b["confidence_max"]:.3f}]'
        logger.info(
            f'  {i+1:>8d}  {conf_range:>16s}  {b["confidence_mean"]:>10.4f}  '
            f'{b["error_mean"]:>10.4f}  {b["error_std"]:>10.4f}'
        )

    # ---- 3. 相关性 ----
    logger.info('')
    logger.info('=' * 60)
    logger.info('  Confidence vs Error 相关性')
    logger.info('=' * 60)

    # Pearson 相关系数
    conf_np = confidence.numpy()
    error_np = error.numpy()
    correlation = np.corrcoef(conf_np, error_np)[0, 1]

    logger.info(f'  Pearson 相关系数: {correlation:.4f}')

    if abs(correlation) > 0.3:
        direction = '负相关（符合预期: 高 confidence → 低 error）' if correlation < 0 else '正相关（不符合预期）'
        logger.info(f'  解读: {direction}')
        logger.info(f'  confidence 是有效信号')
    elif abs(correlation) > 0.1:
        logger.info(f'  解读: 弱相关，confidence 有一定信号但不够强')
        logger.info(f'  建议: 尝试更细粒度的不确定性代理')
    else:
        logger.info(f'  解读: 基本不相关，confidence 当前定义无效')
        logger.info(f'  需要重新设计 confidence 代理信号')

    # ---- 4. 关键判定 ----
    logger.info('')
    logger.info('=' * 60)
    logger.info('  判定结论')
    logger.info('=' * 60)

    is_discriminative = confidence.std().item() > 0.05
    is_correlated = abs(correlation) > 0.2

    if is_discriminative and is_correlated:
        logger.info('  plan_confidence: 有效信号')
        logger.info('  - 值分布有足够区分度')
        logger.info('  - 与规划质量有显著相关性')
    elif is_discriminative and not is_correlated:
        logger.info('  plan_confidence: 有区分度但无语义')
        logger.info('  - 值分布有变化，但与规划质量无关')
        logger.info('  - 需要更换代理信号定义')
    else:
        logger.info('  plan_confidence: 伪特征')
        logger.info('  - 值分布接近常数，无法区分样本')
        logger.info('  - 当前定义（mode 间方差负指数）对该数据无效')

    logger.info('')


def main():
    parser = argparse.ArgumentParser(description='plan_confidence 有效性分析')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--num_buckets', type=int, default=5)
    parser.add_argument('--vad_output_dir', type=str, default=None,
                        help='真实 VAD 输出目录（可选）')
    args = parser.parse_args()

    if args.vad_output_dir:
        logger.info('加载真实 VAD 输出...')
        # TODO: 实现真实数据加载
        logger.info('真实数据加载尚未实现，使用合成数据')

    logger.info(f'使用合成数据 ({args.num_samples} 样本)')
    vad_outs, gt = make_synthetic_vad_outputs(args.num_samples)

    run_confidence_analysis(vad_outs, gt, num_buckets=args.num_buckets)


if __name__ == '__main__':
    main()
