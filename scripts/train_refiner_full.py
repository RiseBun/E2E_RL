#!/usr/bin/env python
"""完整的 Refiner 训练脚本 - 包含数据加载和 HUF 集成。

用法:
    # 第一阶段基线验证（规则 HUF）
    python -m E2E_RL.scripts.train_refiner_full \
      --config E2E_RL/configs/refiner_debug.yaml \
      --data_dir E2E_RL/data/vad_dumps \
      --stage supervised \
      --epochs 5 \
      --output_dir ./experiments/stage1_rule_huf

    # 第二阶段奖励加权（加入 HUF 过滤）
    python -m E2E_RL.scripts.train_refiner_full \
      --config E2E_RL/configs/refiner_debug.yaml \
      --data_dir E2E_RL/data/vad_dumps \
      --stage reward_weighted \
      --epochs 3 \
      --checkpoint ./experiments/stage1_rule_huf/checkpoint_epoch_5.pth \
      --output_dir ./experiments/stage2_rule_huf
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.trainers.trainer_refiner import InterfaceRefinerTrainer
from E2E_RL.hard_case.mining import HardCaseMiner
from E2E_RL.update_filter.config import HUFConfig
from E2E_RL.update_filter.filter import HarmfulUpdateFilter
from E2E_RL.update_filter.scorer import UpdateReliabilityScorer
from E2E_RL.update_filter.model import ReliabilityNet
from E2E_RL.data.dataloader import build_vad_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_refiner(cfg: Dict[str, Any]) -> InterfaceRefiner:
    """根据配置构建 InterfaceRefiner 模型。"""
    model_cfg = cfg['model']
    return InterfaceRefiner(
        scene_dim=model_cfg['scene_dim'],
        plan_len=model_cfg['plan_len'],
        hidden_dim=model_cfg['hidden_dim'],
        dropout=model_cfg.get('dropout', 0.1),
        output_norm=model_cfg.get('output_norm', False),
    )


def build_optimizer(
    refiner: InterfaceRefiner,
    cfg: Dict[str, Any],
    stage: str = 'supervised',
) -> torch.optim.Optimizer:
    """构建优化器。"""
    train_cfg = cfg['training'][stage]
    return torch.optim.AdamW(
        refiner.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg.get('weight_decay', 1e-4),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    """构建学习率调度器。"""
    sched_cfg = cfg['training'].get('scheduler', {})
    sched_type = sched_cfg.get('type', 'cosine')
    if sched_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get('T_max', 30),
            eta_min=sched_cfg.get('eta_min', 1e-6),
        )
    elif sched_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get('step_size', 10),
            gamma=sched_cfg.get('gamma', 0.1),
        )
    else:
        raise ValueError(f'未知的调度器类型: {sched_type}')


def main():
    parser = argparse.ArgumentParser(description='Refiner 完整训练脚本')
    parser.add_argument(
        '--config', type=str,
        default='E2E_RL/configs/refiner_debug.yaml',
        help='配置文件路径',
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='E2E_RL/data/vad_dumps',
        help='数据目录',
    )
    parser.add_argument(
        '--stage', type=str, choices=['supervised', 'reward_weighted'],
        default='supervised',
        help='训练阶段',
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help='训练 epoch 数',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='加载预训练的 refiner checkpoint',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='输出目录',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='训练设备',
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='批大小',
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get('output_dir', f'./experiments/{args.stage}_baseline')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # ---- 构建模块 ----
    refiner = build_refiner(cfg)
    num_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    logger.info(f'Refiner 参数量: {num_params:,}')

    # 加载预训练 checkpoint（如果有）
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        refiner.load_state_dict(ckpt['refiner_state_dict'])
        logger.info(f'已加载 checkpoint 从 {args.checkpoint}')

    # Hard case miner
    hc_cfg = cfg.get('hard_case', {})
    hard_case_miner = HardCaseMiner(
        error_weight=hc_cfg.get('error_weight', 1.0),
        uncertainty_weight=hc_cfg.get('uncertainty_weight', 0.5),
        collision_weight=hc_cfg.get('collision_weight', 0.5),
        top_ratio=hc_cfg.get('top_ratio', 0.2),
    )

    # HUF 配置
    huf_cfg = cfg.get('huf', {})
    update_filter = HarmfulUpdateFilter(HUFConfig(**huf_cfg)) if huf_cfg.get('enabled', False) else None
    logger.info(f'HUF 已启用: {update_filter is not None}')

    # ReliabilityNet (Learned Scorer)
    reliability_cfg = cfg.get('reliability_net', {})
    reliability_net = None
    scorer_optimizer = None
    if reliability_cfg.get('enabled', False):
        reliability_net = ReliabilityNet(
            scene_dim=reliability_cfg['scene_dim'],
            plan_len=reliability_cfg['plan_len'],
            safety_feat_dim=reliability_cfg.get('safety_feat_dim', 0),
            hidden_dim=reliability_cfg['hidden_dim'],
            dropout=reliability_cfg.get('dropout', 0.1),
        ).to(device)
        scorer_optimizer = torch.optim.AdamW(
            reliability_net.parameters(),
            lr=reliability_cfg['lr'],
            weight_decay=reliability_cfg.get('weight_decay', 1e-4),
        )

    update_scorer = UpdateReliabilityScorer(
        config=HUFConfig(**huf_cfg),
        model=reliability_net,
    ) if huf_cfg.get('enabled', False) else None

    # 数据加载
    logger.info(f'从 {args.data_dir} 加载数据...')
    dataloader = build_vad_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
    )
    logger.info(f'数据加载完毕，共 {len(dataloader)} 个 batch')

    # 优化器和调度器
    optimizer = build_optimizer(refiner, cfg, args.stage)
    scheduler = build_scheduler(optimizer, cfg)
    reward_cfg = cfg['training'].get('reward', {})

    # 构建训练器
    trainer = InterfaceRefinerTrainer(
        refiner=refiner,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        residual_reg_weight=cfg['training'][args.stage].get('residual_reg_weight', 0.01),
        reward_config=reward_cfg,
        hard_case_miner=hard_case_miner,
        grad_clip=cfg['training'][args.stage].get('grad_clip', 1.0),
        update_filter=update_filter,
        update_scorer=update_scorer,
        scorer_optimizer=scorer_optimizer,
    )

    # ---- 训练循环 ----
    logger.info('=' * 60)
    logger.info(f'Stage: {args.stage} | Epochs: {args.epochs}')
    logger.info('=' * 60)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        if args.stage == 'supervised':
            metrics = trainer.train_supervised_epoch(dataloader, epoch=epoch)
        else:
            metrics = trainer.train_reward_weighted_epoch(dataloader, epoch=epoch)

        # 检查是否为最佳 epoch
        current_loss = metrics.get('loss_total', float('inf'))
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            # 保存最佳检查点
            trainer.save_checkpoint(
                os.path.join(output_dir, 'checkpoint_best.pth'),
                epoch=epoch,
                extra={'best_loss': best_loss},
            )

        # 定期保存检查点
        if (epoch + 1) % cfg.get('save_every', 5) == 0 or (epoch + 1) == args.epochs:
            trainer.save_checkpoint(
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch=epoch,
            )

    logger.info('=' * 60)
    logger.info(f'训练完毕！最佳 loss 在 epoch {best_epoch}: {best_loss:.4f}')
    logger.info(f'检查点已保存到 {output_dir}')
    logger.info('=' * 60)

    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    main()
