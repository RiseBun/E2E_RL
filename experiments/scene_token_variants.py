"""实验 2: scene_token 丰富度对比。

验证问题: 当前 mean pooled BEV → [B, D] 是否过于粗糙？
方法: 对比多种 scene_token 设计，用相同 refiner 进行短训练，
      看 refinement 质量差异。

变体:
  - mean_pool:    mean pooling → [B, D]
  - max_pool:     max pooling → [B, D]
  - mean_max:     mean + max concat → [B, 2D]
  - grid_pool:    BEV 分块池化（保留粗粒度空间信息）→ [B, G*D]
  - ego_local:    ego 周围局部 token → [B, K*D] 或 pooled → [B, D]

使用方式:
    python -m E2E_RL.experiments.scene_token_variants
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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


# ---- Scene Token 提取变体 ----

def scene_token_mean(bev_embed: torch.Tensor) -> torch.Tensor:
    """Mean pooling → [B, D]"""
    return bev_embed.mean(dim=1)


def scene_token_max(bev_embed: torch.Tensor) -> torch.Tensor:
    """Max pooling → [B, D]"""
    return bev_embed.max(dim=1).values


def scene_token_mean_max(bev_embed: torch.Tensor) -> torch.Tensor:
    """Mean + Max concat → [B, 2D]"""
    return torch.cat([bev_embed.mean(dim=1), bev_embed.max(dim=1).values], dim=-1)


def scene_token_grid(
    bev_embed: torch.Tensor,
    grid_size: int = 4,
) -> torch.Tensor:
    """BEV 分块池化，保留粗粒度空间结构。

    将 BEV tokens 重排为 (H, W) 然后分块池化。
    假设 bev_embed: [B, N, D], N = bev_h * bev_w

    Returns:
        [B, grid_size^2 * D]
    """
    batch_size, num_tokens, dim = bev_embed.shape
    side = int(num_tokens ** 0.5)
    if side * side != num_tokens:
        # 不是完美正方形，回退到均匀分块
        chunk_size = num_tokens // (grid_size * grid_size)
        chunks = bev_embed[:, :chunk_size * grid_size * grid_size].reshape(
            batch_size, grid_size * grid_size, chunk_size, dim
        )
        return chunks.mean(dim=2).reshape(batch_size, -1)

    # 正方形 BEV: 按空间分块
    bev_2d = bev_embed.reshape(batch_size, side, side, dim)
    block_h = side // grid_size
    block_w = side // grid_size

    pooled = []
    for i in range(grid_size):
        for j in range(grid_size):
            block = bev_2d[:, i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w, :]
            pooled.append(block.reshape(batch_size, -1, dim).mean(dim=1))

    return torch.cat(pooled, dim=-1)  # [B, grid_size^2 * D]


def scene_token_ego_local(
    bev_embed: torch.Tensor,
    top_k: int = 16,
) -> torch.Tensor:
    """取 ego 附近的 token 子集并池化。

    假设 BEV 中心是 ego 位置。
    [B, N, D] → 取中心附近 top_k 个 token → mean → [B, D]
    """
    batch_size, num_tokens, dim = bev_embed.shape
    side = int(num_tokens ** 0.5)
    if side * side != num_tokens:
        # 回退：取前 top_k 个 token
        return bev_embed[:, :top_k].mean(dim=1)

    center = side // 2
    half_k = int(top_k ** 0.5) // 2

    bev_2d = bev_embed.reshape(batch_size, side, side, dim)
    local = bev_2d[:, center-half_k:center+half_k, center-half_k:center+half_k, :]
    return local.reshape(batch_size, -1, dim).mean(dim=1)


# ---- 实验逻辑 ----

VARIANTS = {
    'mean_pool': {
        'fn': scene_token_mean,
        'output_dim_factor': 1,  # D
        'description': 'Mean pooling → [B, D]',
    },
    'max_pool': {
        'fn': scene_token_max,
        'output_dim_factor': 1,
        'description': 'Max pooling → [B, D]',
    },
    'mean_max': {
        'fn': scene_token_mean_max,
        'output_dim_factor': 2,  # 2D
        'description': 'Mean+Max concat → [B, 2D]',
    },
    'grid_4x4': {
        'fn': lambda bev: scene_token_grid(bev, grid_size=4),
        'output_dim_factor': 16,  # 16D
        'description': 'Grid 4x4 pooling → [B, 16D]',
    },
    'ego_local': {
        'fn': scene_token_ego_local,
        'output_dim_factor': 1,
        'description': 'Ego-local pooling → [B, D]',
    },
}


def run_short_training(
    scene_token: torch.Tensor,
    reference_plan: torch.Tensor,
    gt_plan: torch.Tensor,
    scene_dim: int,
    plan_len: int,
    hidden_dim: int = 128,
    num_steps: int = 50,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """对单个 scene_token 变体进行短训练并评估。"""
    refiner = InterfaceRefiner(
        scene_dim=scene_dim,
        plan_len=plan_len,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)
    refiner.train()

    interface = PlanningInterface(
        scene_token=scene_token,
        reference_plan=reference_plan,
    )

    losses = []
    for step in range(num_steps):
        outputs = refiner(interface)
        loss = supervised_refinement_loss(outputs['refined_plan'], gt_plan)
        loss += outputs['residual_norm'].mean() * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # 评估
    refiner.eval()
    with torch.no_grad():
        outputs = refiner(interface)
        refined = outputs['refined_plan']
        final_loss = supervised_refinement_loss(refined, gt_plan).item()
        ade = torch.norm(refined - gt_plan, dim=-1).mean().item()
        fde = torch.norm(refined[:, -1] - gt_plan[:, -1], dim=-1).mean().item()

    return {
        'init_loss': losses[0],
        'final_loss': final_loss,
        'loss_reduction': (losses[0] - final_loss) / losses[0] * 100,
        'ade': ade,
        'fde': fde,
    }


def main():
    parser = argparse.ArgumentParser(description='Scene Token 丰富度对比')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--bev_tokens', type=int, default=100,
                        help='模拟 BEV token 数量 (10x10 grid)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fut_ts', type=int, default=6)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--num_trials', type=int, default=3)
    args = parser.parse_args()

    plan_len = args.fut_ts * 2

    logger.info('=' * 70)
    logger.info('  实验 2: Scene Token 丰富度对比')
    logger.info('=' * 70)
    logger.info(f'  BEV tokens: {args.bev_tokens}, embed_dim: {args.embed_dim}')
    logger.info(f'  训练步数: {args.num_steps}, 试验次数: {args.num_trials}')

    # GT: 向前直行
    t = torch.arange(1, args.fut_ts + 1, dtype=torch.float32)
    gt_base = torch.stack([t * 0.5, torch.zeros_like(t)], dim=-1)
    gt = gt_base.unsqueeze(0).expand(args.batch_size, -1, -1).clone()

    # 收集结果
    all_results: Dict[str, Dict[str, list]] = {}

    for trial in range(args.num_trials):
        torch.manual_seed(trial * 42)

        bev_embed = torch.randn(args.batch_size, args.bev_tokens, args.embed_dim)
        reference_plan = torch.randn(args.batch_size, args.fut_ts, 2) * 0.3

        for name, variant in VARIANTS.items():
            scene_token = variant['fn'](bev_embed)
            scene_dim = scene_token.shape[-1]

            result = run_short_training(
                scene_token=scene_token.detach(),
                reference_plan=reference_plan.detach(),
                gt_plan=gt.detach(),
                scene_dim=scene_dim,
                plan_len=plan_len,
                num_steps=args.num_steps,
            )

            if name not in all_results:
                all_results[name] = {k: [] for k in result}
            for k, v in result.items():
                all_results[name][k].append(v)

    # 输出结果
    logger.info('')
    header = f'{"变体":<22s} {"scene_dim":>10s} {"final_loss":>10s} {"ADE":>8s} {"FDE":>8s} {"loss_red%":>10s}'
    logger.info(header)
    logger.info('-' * 70)

    for name, variant in VARIANTS.items():
        metrics = all_results[name]
        dim = args.embed_dim * variant['output_dim_factor']
        avg = {k: sum(v)/len(v) for k, v in metrics.items()}

        logger.info(
            f'{variant["description"]:<22s} {dim:>10d} '
            f'{avg["final_loss"]:>10.4f} {avg["ade"]:>8.4f} {avg["fde"]:>8.4f} '
            f'{avg["loss_reduction"]:>10.1f}%'
        )

    logger.info('')
    logger.info('解读:')
    logger.info('  - 如果 mean_pool 和其他变体差异很小 → 当前 token 已经够用')
    logger.info('  - 如果 grid/mean_max 明显更好 → 需要保留更多空间信息')
    logger.info('  - 如果 ego_local 更好 → ego 附近信息比全局信息更重要')
    logger.info('')


if __name__ == '__main__':
    main()
