#!/usr/bin/env python
"""
CorrectionPolicy 训练脚本

用法:
    python -m E2E_RL.scripts.train_correction_policy \
        --config E2E_RL/configs/correction_policy.yaml \
        --output_dir ./experiments/correction_policy

功能:
    1. 从 YAML 加载配置
    2. 构建 CorrectionPolicy、优化器、调度器
    3. Stage 1: Behavioral Cloning 预热
    4. Stage 2: Policy Gradient + STAPO Gate
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

from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.rl_trainer.correction_policy_trainer import CorrectionPolicyTrainer
from E2E_RL.update_selector.safety_guard import SafetyGuard, SafetyGuardConfig
from E2E_RL.update_selector.stapo_gate import STAPOGate, STAPOGateConfig
from E2E_RL.update_selector.update_evaluator import LearnedUpdateGate, UpdateEvaluator, UpdateEvaluatorConfig
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


def build_policy(cfg: Dict[str, Any]) -> CorrectionPolicy:
    """根据配置构建 CorrectionPolicy。"""
    model_cfg = cfg['model']
    return CorrectionPolicy(
        scene_dim=model_cfg['scene_dim'],
        plan_len=model_cfg['plan_len'],
        hidden_dim=model_cfg['hidden_dim'],
        dropout=model_cfg.get('dropout', 0.1),
        log_std_min=model_cfg.get('log_std_min', -5.0),
        log_std_max=model_cfg.get('log_std_max', 2.0),
        action_scale=model_cfg.get('action_scale', 1.0),
        actor_type='gaussian',
    )


def build_optimizer(
    policy: CorrectionPolicy,
    cfg: Dict[str, Any],
    stage: str = 'bc',
) -> torch.optim.Optimizer:
    """构建优化器。"""
    train_cfg = cfg['training'][stage]
    return torch.optim.AdamW(
        policy.parameters(),
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
    elif sched_type == 'none':
        return None
    else:
        raise ValueError(f'Unknown scheduler type: {sched_type}')


def build_safety_guard(cfg: Dict[str, Any]) -> SafetyGuard:
    """构建 SafetyGuard。"""
    safety_cfg = cfg.get('safety_guard', {})
    if not safety_cfg.get('enabled', False):
        return SafetyGuard(SafetyGuardConfig(enabled=False))

    return SafetyGuard(SafetyGuardConfig(
        enabled=True,
        max_residual_norm=safety_cfg.get('max_residual_norm', 5.0),
        max_step_disp=safety_cfg.get('max_step_disp', 2.0),
        max_speed=safety_cfg.get('max_speed', 15.0),
        max_total_disp=safety_cfg.get('max_total_disp', 10.0),
        dt=safety_cfg.get('dt', 0.5),
    ))


def build_stapo_gate(cfg: Dict[str, Any]) -> STAPOGate:
    """构建 STAPO Gate。"""
    stapo_cfg = cfg.get('stapo_gate', {})
    if not stapo_cfg.get('enabled', False):
        return STAPOGate(STAPOGateConfig(enabled=False))

    return STAPOGate(STAPOGateConfig(
        enabled=True,
        advantage_threshold=stapo_cfg.get('advantage_threshold', 0.0),
        probability_threshold=stapo_cfg.get('probability_threshold', 0.1),
        entropy_threshold=stapo_cfg.get('entropy_threshold', 0.5),
        entropy_normalization=stapo_cfg.get('entropy_normalization', 'max'),
        min_retention_ratio=stapo_cfg.get('min_retention_ratio', 0.1),
        use_combined_threshold=stapo_cfg.get('use_combined_threshold', True),
    ))


def build_learned_gate(cfg: Dict[str, Any], device: torch.device) -> Optional[LearnedUpdateGate]:
    """构建 LearnedUpdateGate。"""
    learned_cfg = cfg.get('learned_gate', {})
    if not learned_cfg.get('enabled', False):
        logger.info('LearnedUpdateGate: disabled')
        return None

    # 加载 UpdateEvaluator
    evaluator_path = learned_cfg.get('evaluator_path')
    if not evaluator_path or not os.path.exists(evaluator_path):
        logger.warning(f'LearnedUpdateGate: evaluator_path not found: {evaluator_path}')
        return None

    model_cfg = cfg.get('model', {})
    evaluator_config = UpdateEvaluatorConfig(
        scene_dim=model_cfg.get('scene_dim', 256),
        plan_len=model_cfg.get('plan_len', 6),
        hidden_dim=model_cfg.get('hidden_dim', 256),
        dropout=model_cfg.get('dropout', 0.1),
    )
    
    evaluator = UpdateEvaluator(evaluator_config)
    ckpt = torch.load(evaluator_path, map_location=device)
    if 'evaluator_state_dict' in ckpt:
        evaluator.load_state_dict(ckpt['evaluator_state_dict'])
    else:
        evaluator.load_state_dict(ckpt)
    evaluator.to(device)
    evaluator.eval()
    
    learned_gate = LearnedUpdateGate(
        evaluator=evaluator,
        tau_gain=learned_cfg.get('tau_gain', 0.0),
        tau_risk=learned_cfg.get('tau_risk', 0.5),
    )
    
    logger.info(f'LearnedUpdateGate: loaded from {evaluator_path}')
    return learned_gate


def main():
    parser = argparse.ArgumentParser(description='CorrectionPolicy 训练')
    parser.add_argument(
        '--config', type=str,
        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/configs/correction_policy.yaml',
        help='配置文件路径',
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/correction_policy',
        help='输出目录',
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
    output_dir = args.output_dir or cfg.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # ---- 构建模型 ----
    policy = build_policy(cfg)
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f'CorrectionPolicy 参数量: {num_params:,}')

    if args.dry_run:
        logger.info('Dry run 模式，仅打印模型结构:')
        logger.info(str(policy))
        return

    # ---- 构建 SafetyGuard 和 STAPO Gate ----
    safety_guard = build_safety_guard(cfg)
    stapo_gate = build_stapo_gate(cfg)
    learned_gate = build_learned_gate(cfg, device)
    logger.info(f'SafetyGuard: enabled={safety_guard.cfg.enabled}')
    logger.info(f'STAPO Gate: enabled={stapo_gate.cfg.enabled}')
    logger.info(f'LearnedUpdateGate: enabled={learned_gate is not None}')

    # ---- 数据加载器 ----
    data_cfg = cfg.get('data', {})
    data_dir = data_cfg.get('data_dir', '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps')
    dataloader = build_vad_dataloader(
        data_dir=data_dir,
        batch_size=data_cfg.get('batch_size', 8),
        num_workers=data_cfg.get('num_workers', 0),
        shuffle=True,
    )
    logger.info(f'创建数据加载器，批大小: {data_cfg.get("batch_size", 8)}，样本数: {len(dataloader.dataset)}')

    # ---- Stage 1: BC 预热 ----
    bc_cfg = cfg['training']['bc']
    if bc_cfg.get('enabled', True):
        logger.info('=' * 50)
        logger.info('Stage 1: Behavioral Cloning')
        logger.info('=' * 50)

        bc_optimizer = build_optimizer(policy, cfg, 'bc')
        bc_scheduler = build_scheduler(bc_optimizer, cfg)
        reward_cfg = cfg['training'].get('reward', {})

        trainer = CorrectionPolicyTrainer(
            policy=policy,
            optimizer=bc_optimizer,
            scheduler=bc_scheduler,
            device=device,
            reward_config=reward_cfg,
            safety_guard=safety_guard,
            stapo_gate=stapo_gate,
            learned_gate=learned_gate,
            entropy_coef=cfg['training'].get('entropy_coef', 0.01),
            grad_clip=bc_cfg.get('grad_clip', 1.0),
        )

        bc_epochs = bc_cfg.get('epochs', 5)
        for epoch in range(bc_epochs):
            metrics = trainer.train_bc_epoch(dataloader, epoch)
            logger.info(f'BC Epoch {epoch}: {metrics}')

        # 保存 BC 检查点
        bc_checkpoint_path = os.path.join(output_dir, 'checkpoint_bc.pth')
        trainer.save_checkpoint(bc_checkpoint_path, bc_epochs)

    # ---- Stage 2: RL 训练 ----
    rl_cfg = cfg['training']['rl']
    if rl_cfg.get('enabled', True):
        logger.info('=' * 50)
        logger.info('Stage 2: Policy Gradient + STAPO Gate')
        logger.info('=' * 50)

        rl_optimizer = build_optimizer(policy, cfg, 'rl')
        rl_scheduler = build_scheduler(rl_optimizer, cfg)
        reward_cfg = cfg['training'].get('reward', {})

        trainer = CorrectionPolicyTrainer(
            policy=policy,
            optimizer=rl_optimizer,
            scheduler=rl_scheduler,
            device=device,
            reward_config=reward_cfg,
            safety_guard=safety_guard,
            stapo_gate=stapo_gate,
            learned_gate=learned_gate,
            entropy_coef=rl_cfg.get('entropy_coef', 0.01),
            grad_clip=rl_cfg.get('grad_clip', 1.0),
        )

        rl_epochs = rl_cfg.get('epochs', 20)
        save_every = cfg.get('save_every', 5)

        for epoch in range(rl_epochs):
            metrics = trainer.train_rl_epoch(dataloader, epoch)
            logger.info(f'RL Epoch {epoch}: {metrics}')

            if epoch % save_every == 0:
                rl_checkpoint_path = os.path.join(output_dir, f'checkpoint_rl_epoch_{epoch}.pth')
                trainer.save_checkpoint(rl_checkpoint_path, bc_cfg.get('epochs', 0) + epoch)

        # 保存最终检查点
        final_checkpoint_path = os.path.join(output_dir, 'checkpoint_final.pth')
        trainer.save_checkpoint(final_checkpoint_path, bc_cfg.get('epochs', 0) + rl_epochs)

    # ---- 评估 ----
    logger.info('=' * 50)
    logger.info('Evaluation')
    logger.info('=' * 50)

    eval_metrics = trainer.evaluate(dataloader)
    logger.info(f'Eval metrics: {eval_metrics}')

    # ---- 保存配置 ----
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    logger.info(f'配置已保存到 {config_save_path}')

    logger.info('训练完成！')


if __name__ == '__main__':
    main()
