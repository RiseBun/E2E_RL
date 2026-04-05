#!/usr/bin/env python
"""InterfaceRefiner 端到端训练脚本。

用法:
    python -m E2E_RL.scripts.train_interface_refiner \
        --config E2E_RL/configs/refiner_debug.yaml \
        --output_dir ./experiments/refiner_debug

功能:
    1. 从 YAML 加载配置
    2. 构建 Refiner 模型、优化器、调度器
    3. Stage 1: 监督预热
    4. Stage 2: 奖励加权精炼
    5. 保存检查点和评估结果
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
from E2E_RL.data.vad_dataset import create_vad_dataloader

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
    parser = argparse.ArgumentParser(description='InterfaceRefiner 训练')
    parser.add_argument(
        '--config', type=str,
        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/refiner_debug.yaml',
        help='配置文件路径',
    )
    parser.add_argument(
        '--output_dir', type=str, 
        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/refiner_debug',
        help='输出目录（覆盖配置中的 output_dir）',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='训练设备',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='仅构建模型并打印参数，不实际训练',
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 使用命令行参数或配置中的 output_dir，统一转换为绝对路径
    # 优先使用配置文件中的 output_dir（更明确）
    config_output_dir = cfg.get('output_dir')
    output_dir = args.output_dir if args.output_dir != '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/refiner_debug' else config_output_dir
    output_dir = output_dir or config_output_dir or '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/refiner_scorer_huf'
    if not os.path.isabs(output_dir):
        # 如果是相对路径，基于项目根目录转换
        project_root = '/mnt/cpfs/prediction/lipeinan/RL'
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 构建模型
    refiner = build_refiner(cfg)
    num_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    logger.info(f'Refiner 模型参数量: {num_params:,}')

    if args.dry_run:
        logger.info('Dry run 模式，仅打印模型结构:')
        logger.info(str(refiner))
        return

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

    # ReliabilityNet (Learned Scorer)
    reliability_cfg = cfg.get('reliability_net', {})
    reliability_net = None
    scorer_optimizer = None
    
    # 加载预训练 Scorer 或创建新的
    pretrained_path = reliability_cfg.get('pretrained_path')
    if reliability_cfg.get('enabled', False):
        reliability_net = ReliabilityNet(
            scene_dim=reliability_cfg['scene_dim'],
            plan_len=reliability_cfg['plan_len'],
            safety_feat_dim=reliability_cfg.get('safety_feat_dim', 0),
            hidden_dim=reliability_cfg['hidden_dim'],
            dropout=reliability_cfg.get('dropout', 0.1),
        )
        
        # 如果有预训练权重，加载它
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f'加载预训练 Scorer: {pretrained_path}')
            ckpt = torch.load(pretrained_path, map_location='cpu')
            reliability_net.load_state_dict(ckpt)
            reliability_net.to(device)  # 移动到正确设备
            logger.info('Scorer 权重加载成功')
        else:
            logger.warning(f'预训练 Scorer 路径不存在或未指定: {pretrained_path}')
        
        # 如果要训练 Scorer，创建优化器
        if reliability_cfg.get('trainable', True):
            scorer_optimizer = torch.optim.AdamW(
                reliability_net.parameters(),
                lr=reliability_cfg['lr'],
                weight_decay=reliability_cfg.get('weight_decay', 1e-4),
            )
        else:
            reliability_net.eval()  # 冻结推理模式
            logger.info('Scorer 处于推理模式 (trainable=False)')

    # UpdateReliabilityScorer: 当 Scorer 或 HUF 任一启用时创建
    if reliability_cfg.get('enabled', False) or huf_cfg.get('enabled', False):
        update_scorer = UpdateReliabilityScorer(
            config=HUFConfig(**huf_cfg),
            model=reliability_net,
            model_path=None,  # 已在上面加载
            model_plan_len=reliability_cfg.get('plan_len') if reliability_cfg.get('enabled', False) else None,
        )
        logger.info('UpdateReliabilityScorer 已初始化')
    else:
        update_scorer = None

    # ---- 数据加载器 ----
    data_cfg = cfg.get('data', {})
    data_dir = data_cfg.get('data_dir', 'E2E_RL/data/vad_dumps')
    dataloader = build_vad_dataloader(
        data_dir=data_dir,
        batch_size=data_cfg.get('batch_size', 8),
        num_workers=data_cfg.get('num_workers', 0),
        shuffle=True,
    )
    logger.info(f'创建数据加载器，批大小: {data_cfg.get("batch_size", 8)}，样本数: {len(dataloader.dataset)}')

    # ---- Stage 1: 监督预热 ----
    logger.info('=' * 50)
    logger.info('Stage 1: 监督预热训练')
    logger.info('=' * 50)

    sup_cfg = cfg['training']['supervised']
    optimizer = build_optimizer(refiner, cfg, 'supervised')
    scheduler = build_scheduler(optimizer, cfg)
    reward_cfg = cfg['training'].get('reward', {})

    trainer = InterfaceRefinerTrainer(
        refiner=refiner,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        residual_reg_weight=sup_cfg.get('residual_reg_weight', 0.01),
        reward_config=reward_cfg,
        hard_case_miner=hard_case_miner,
        grad_clip=sup_cfg.get('grad_clip', 1.0),
        update_filter=update_filter,
        update_scorer=update_scorer,
        scorer_optimizer=scorer_optimizer,
    )

    # ---- 执行训练 ----
    if not args.dry_run:
        # Stage 1: 监督预热
        sup_epochs = sup_cfg.get('epochs', 20)
        for epoch in range(sup_epochs):
            metrics = trainer.train_supervised_epoch(dataloader, epoch)
            logger.info(f'Supervised Epoch {epoch}: {metrics}')

        # 保存监督预热检查点
        sup_checkpoint_path = os.path.join(output_dir, 'checkpoint_supervised.pth')
        trainer.save_checkpoint(sup_checkpoint_path, sup_epochs)

        # Stage 2: 奖励加权精炼
        logger.info('=' * 50)
        logger.info('Stage 2: 奖励加权精炼训练')
        logger.info('=' * 50)

        rw_epochs = cfg['training']['reward_weighted'].get('epochs', 10)
        for epoch in range(rw_epochs):
            metrics = trainer.train_reward_weighted_epoch(dataloader, epoch)
            logger.info(f'Reward Epoch {epoch}: {metrics}')

        # 保存最终检查点
        final_checkpoint_path = os.path.join(output_dir, 'checkpoint_final.pth')
        trainer.save_checkpoint(final_checkpoint_path, sup_epochs + rw_epochs)

    # 保存配置
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    logger.info(f'配置已保存到 {config_save_path}')


if __name__ == '__main__':
    main()
