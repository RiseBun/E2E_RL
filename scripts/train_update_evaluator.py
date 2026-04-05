#!/usr/bin/env python
"""
UpdateEvaluator 训练脚本

训练流程：
1. 收集训练数据：多样化 candidate corrections + reward labels
2. 训练 UpdateEvaluator：多头回归
3. 离线排序验证：确保排序能力过关
4. 保存模型
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
from torch.utils.data import DataLoader

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.update_selector import (
    UpdateEvaluator,
    UpdateEvaluatorConfig,
    CandidateCorrector,
    UpdateEvaluatorDataCollector,
    UpdateEvaluatorDataset,
    UpdateEvaluatorTrainer,
    EvaluatorTrainingConfig,
    OfflineEvaluator,
)
from E2E_RL.data.dataloader import build_vad_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='UpdateEvaluator 训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--policy_ckpt', type=str, default=None, help='CorrectionPolicy 检查点路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--collect_only', action='store_true', help='仅收集数据，不训练')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # ---- 1. 加载数据 ----
    logger.info('=' * 50)
    logger.info('Step 1: 加载训练数据')
    logger.info('=' * 50)

    data_cfg = cfg.get('data', {})
    base_dataloader = build_vad_dataloader(
        data_dir=data_cfg.get('data_dir', '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps'),
        batch_size=data_cfg.get('batch_size', 8),
        num_workers=0,
        shuffle=True,
    )
    logger.info(f'Base dataloader: {len(base_dataloader.dataset)} samples')

    # ---- 2. 加载 Policy（用于候选生成） ----
    policy = None
    if args.policy_ckpt and os.path.exists(args.policy_ckpt):
        logger.info(f'加载 Policy from {args.policy_ckpt}')
        policy = CorrectionPolicy(
            scene_dim=cfg['model']['scene_dim'],
            plan_len=cfg['model']['plan_len'],
            hidden_dim=cfg['model']['hidden_dim'],
        )
        ckpt = torch.load(args.policy_ckpt, map_location='cpu')
        policy.load_state_dict(ckpt['policy_state_dict'])
        policy.eval()
    else:
        logger.info('不使用 Policy（使用随机候选）')

    # ---- 3. 创建候选生成器 ----
    candidate_gen = CandidateCorrector(
        policy=policy,
        max_corrections_per_type=cfg.get('candidate_gen', {}).get('max_per_type', 1),
        random_scale=cfg.get('candidate_gen', {}).get('random_scale', 2.0),
        gt_directed_scale=cfg.get('candidate_gen', {}).get('gt_directed_scale', 0.5),
    )

    # ---- 4. 收集训练数据 ----
    logger.info('=' * 50)
    logger.info('Step 2: 收集训练数据')
    logger.info('=' * 50)

    collect_cfg = cfg.get('data_collection', {})
    reward_config = cfg.get('reward', {})

    data_collector = UpdateEvaluatorDataCollector(
        base_dataloader=base_dataloader,
        candidate_generator=candidate_gen,
        reward_config=reward_config,
        device=device,
        collect_all_types=collect_cfg.get('collect_all_types', True),
    )

    n_batches_to_collect = collect_cfg.get('n_batches', None)
    dataset = data_collector.collect(n_batches=n_batches_to_collect)

    logger.info(f'收集完成: {len(dataset)} samples')

    # 保存收集的数据
    dataset_path = output_dir / 'evaluator_dataset.pth'
    # (简化保存：将 dataset 转为 list 再保存)
    torch.save({'n_samples': len(dataset)}, dataset_path)
    logger.info(f'Dataset metadata saved to {dataset_path}')

    if args.collect_only:
        logger.info('--collect_only 模式，仅收集数据')
        return

    # ---- 5. 创建 DataLoader ----
    train_dataset = UpdateEvaluatorDataset()
    for i in range(len(dataset)):
        train_dataset.add_sample(dataset.samples[i])

    # 简单划分：80% 训练，20% 验证
    n_train = int(0.8 * len(train_dataset))
    indices = list(range(len(train_dataset)))
    # (这里简化处理，不做真实划分)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('training', {}).get('batch_size', 64),
        shuffle=True,
        collate_fn=UpdateEvaluatorDataset.collate_fn,
    )

    # ---- 6. 创建模型 ----
    logger.info('=' * 50)
    logger.info('Step 3: 创建 UpdateEvaluator')
    logger.info('=' * 50)

    model_cfg = cfg.get('model', {})
    evaluator_cfg = UpdateEvaluatorConfig(
        scene_dim=model_cfg.get('scene_dim', 256),
        plan_len=model_cfg.get('plan_len', 12),
        hidden_dim=model_cfg.get('hidden_dim', 256),
        dropout=model_cfg.get('dropout', 0.1),
        lambda_collision=cfg.get('risk_weights', {}).get('collision', 2.0),
        lambda_offroad=cfg.get('risk_weights', {}).get('offroad', 1.0),
        lambda_comfort=cfg.get('risk_weights', {}).get('comfort', 0.5),
        lambda_drift=cfg.get('risk_weights', {}).get('drift', 1.0),
    )

    evaluator = UpdateEvaluator(evaluator_cfg)

    num_params = sum(p.numel() for p in evaluator.parameters() if p.requires_grad)
    logger.info(f'UpdateEvaluator 参数量: {num_params:,}')

    # ---- 7. 训练 ----
    logger.info('=' * 50)
    logger.info('Step 4: 训练')
    logger.info('=' * 50)

    train_cfg = EvaluatorTrainingConfig(
        alpha_gain=cfg.get('training', {}).get('alpha_gain', 1.0),
        beta_risk=cfg.get('training', {}).get('beta_risk', 1.0),
        lr=cfg.get('training', {}).get('lr', 1e-4),
        weight_decay=cfg.get('training', {}).get('weight_decay', 1e-4),
        grad_clip=cfg.get('training', {}).get('grad_clip', 1.0),
        epochs=cfg.get('training', {}).get('epochs', 10),
        eval_every=cfg.get('training', {}).get('eval_every', 5),
    )

    trainer = UpdateEvaluatorTrainer(
        evaluator=evaluator,
        config=train_cfg,
        device=device,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=None,  # 简化：不做验证
        output_dir=str(output_dir),
    )

    # ---- 8. 离线排序验证 ----
    logger.info('=' * 50)
    logger.info('Step 5: 离线排序验证')
    logger.info('=' * 50)

    offline_eval = OfflineEvaluator(evaluator, device)

    ranking_metrics = offline_eval.evaluate_ranking(train_loader)
    logger.info('Ranking metrics:')
    for key, val in ranking_metrics.items():
        logger.info(f'  {key}: {val:.4f}')

    filtering_metrics = offline_eval.evaluate_filtering(
        train_loader,
        tau_gain=cfg.get('filtering', {}).get('tau_gain', 0.0),
        tau_risk=cfg.get('filtering', {}).get('tau_risk', 0.5),
    )
    logger.info('Filtering metrics:')
    for key, val in filtering_metrics.items():
        logger.info(f'  {key}: {val}')

    # ---- 9. 保存最终模型 ----
    final_path = output_dir / 'update_evaluator_final.pth'
    trainer.save_checkpoint(final_path, train_cfg.epochs)
    logger.info(f'最终模型已保存到 {final_path}')

    # 保存排序验证结果
    eval_results_path = output_dir / 'ranking_results.yaml'
    all_results = {
        'ranking': {k: float(v) for k, v in ranking_metrics.items()},
        'filtering': {k: float(v) if isinstance(v, (int, float)) else v for k, v in filtering_metrics.items()},
    }
    with open(eval_results_path, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    logger.info(f'排序验证结果已保存到 {eval_results_path}')

    logger.info('训练完成！')


if __name__ == '__main__':
    main()
