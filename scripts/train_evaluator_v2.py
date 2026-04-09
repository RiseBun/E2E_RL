"""
训练 UpdateEvaluator v2

使用改进的加权采样策略训练 UpdateEvaluator。

改进：
1. 使用加权采样（移除 bounded_random/safety_biased）
2. 记录排序质量指标（Spearman, Kendall, Top-k）
3. 记录过滤质量指标（Retained vs Filtered）

用法:
    # 使用默认配置
    python scripts/train_evaluator_v2.py
    
    # 使用命令行参数覆盖
    python scripts/train_evaluator_v2.py \
        --data_dir data/vad_dumps_full \
        --output_dir experiments/update_evaluator \
        --epochs 50 \
        --batch_size 16
"""

import argparse
import torch
import yaml
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from E2E_RL.data.dataloader import build_vad_dataloader
from E2E_RL.update_selector import (
    UpdateEvaluator,
    UpdateEvaluatorConfig,
    CandidateCorrector,
    UpdateEvaluatorDataCollector,
    UpdateEvaluatorDataset,
    UpdateEvaluatorTrainer,
    EvaluatorTrainingConfig,
)
from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.update_selector.candidate_generator import DEFAULT_SAMPLE_WEIGHTS

# 默认配置
DEFAULT_CONFIG = {
    'data': {
        'data_dir': 'data/vad_dumps_full',
        'batch_size': 16,
        'val_split': 0.2,
    },
    'candidate_gen': {
        'use_weighted': True,
        'sample_weights': DEFAULT_SAMPLE_WEIGHTS,
        'n_samples': 6,
    },
    'data_collection': {
        'n_batches': 200,
    },
    'model': {
        'scene_dim': 256,
        'plan_len': 6,
        'hidden_dim': 256,
        'dropout': 0.1,
    },
    'risk_weights': {
        'collision': 2.0,
        'offroad': 1.0,
        'comfort': 0.5,
        'drift': 1.0,
    },
    'training': {
        'batch_size': 128,
        'alpha_gain': 1.0,
        'beta_risk': 1.0,
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'epochs': 50,
        'eval_every': 5,
        'gain_threshold': 0.0,
    },
    'reward': {
        'dt': 0.5,
        'w_progress': 1.0,
        'w_collision': 0.5,
        'w_offroad': 0.3,
        'w_comfort': 0.1,
    },
    'output_dir': 'experiments/update_evaluator',
}


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='训练 UpdateEvaluator v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/train_evaluator_v2.py
  
  # 覆盖关键参数
  python scripts/train_evaluator_v2.py \\
      --data_dir data/sparsedrive_dumps_full \\
      --output_dir experiments/sparsedrive_evaluator \\
      --epochs 100 \\
      --lr 1e-4
        """,
    )
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录 (默认: data/vad_dumps_full)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='数据加载 batch size (默认: 16)')
    parser.add_argument('--val_split', type=float, default=None,
                        help='验证集比例 (默认: 0.2)')
    
    # 模型参数
    parser.add_argument('--scene_dim', type=int, default=None,
                        help='场景特征维度 (默认: 256)')
    parser.add_argument('--plan_len', type=int, default=None,
                        help='轨迹长度 (默认: 6)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='隐藏层维度 (默认: 256)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='权重衰减 (默认: 1e-4)')
    parser.add_argument('--train_batch_size', type=int, default=None,
                        help='训练 batch size (默认: 128)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: experiments/update_evaluator)')
    
    # 其他参数
    parser.add_argument('--n_samples', type=int, default=None,
                        help='候选样本数 (默认: 6)')
    parser.add_argument('--n_batches', type=int, default=None,
                        help='数据收集 batch 数 (默认: 200)')
    
    return parser.parse_args()


def merge_config(args):
    """合并默认配置和命令行参数。"""
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    
    # 数据参数
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.val_split is not None:
        config['data']['val_split'] = args.val_split
    
    # 模型参数
    if args.scene_dim is not None:
        config['model']['scene_dim'] = args.scene_dim
    if args.plan_len is not None:
        config['model']['plan_len'] = args.plan_len
    if args.hidden_dim is not None:
        config['model']['hidden_dim'] = args.hidden_dim
    
    # 训练参数
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
    if args.train_batch_size is not None:
        config['training']['batch_size'] = args.train_batch_size
    
    # 输出参数
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    
    # 其他参数
    if args.n_samples is not None:
        config['candidate_gen']['n_samples'] = args.n_samples
    if args.n_batches is not None:
        config['data_collection']['n_batches'] = args.n_batches
    
    return config


def main():
    import logging
    import json
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    
    # 解析命令行参数
    args = parse_args()
    config = merge_config(args)
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("UpdateEvaluator v2 训练")
    logger.info("=" * 60)
    logger.info("配置:")
    logger.info(json.dumps(config, indent=2, default=str))
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 打印采样配置
    logger.info("=" * 50)
    logger.info("采样策略配置")
    logger.info("=" * 50)
    for k, v in DEFAULT_SAMPLE_WEIGHTS.items():
        logger.info(f"  {k}: {v:.1%}")
    logger.info("=" * 50)

    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"配置已保存到: {output_dir / 'config.json'}")

    # 1. 加载数据
    logger.info("Step 1: 加载数据")
    full_dataloader = build_vad_dataloader(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=0,
        shuffle=True,
    )
    logger.info(f"总数据量: {len(full_dataloader.dataset)} samples")

    # 划分 train/val
    n_samples = len(full_dataloader.dataset)
    n_val = int(n_samples * config['data']['val_split'])
    val_indices = list(range(n_val))
    train_indices = list(range(n_val, n_samples))
    logger.info(f"训练集: {len(train_indices)} samples, 验证集: {len(val_indices)} samples")

    # 创建子集 dataloader
    train_dataset = torch.utils.data.Subset(full_dataloader.dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataloader.dataset, val_indices)

    train_base_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=full_dataloader.collate_fn,
    )
    val_base_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=full_dataloader.collate_fn,
    )

    # 2. 加载 Policy
    policy = None
    policy_ckpt = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/correction_policy/policy_final.pth'
    if Path(policy_ckpt).exists():
        logger.info(f"加载 Policy from {policy_ckpt}")
        policy = CorrectionPolicy(
            scene_dim=config['model']['scene_dim'],
            plan_len=config['model']['plan_len'],
            hidden_dim=config['model']['hidden_dim'],
        )
        ckpt = torch.load(policy_ckpt, map_location='cpu')
        policy.load_state_dict(ckpt['policy_state_dict'])
        policy.eval()
    else:
        logger.info("Policy 不存在，使用随机候选")

    # 3. 创建候选生成器（使用加权采样）
    candidate_gen = CandidateCorrector(
        policy=policy,
        sample_weights=config['candidate_gen']['sample_weights'],
        gt_directed_scale=0.5,
    )

    # 4. 收集训练数据
    logger.info("=" * 50)
    logger.info("Step 2: 收集训练数据")
    logger.info("=" * 50)

    train_collector = UpdateEvaluatorDataCollector(
        base_dataloader=train_base_loader,
        candidate_generator=candidate_gen,
        reward_config=config['reward'],
        device=device,
        use_weighted=config['candidate_gen']['use_weighted'],
        n_samples_per_batch=config['candidate_gen']['n_samples'],
    )

    train_dataset_full = train_collector.collect(
        n_batches=config['data_collection']['n_batches']
    )
    logger.info(f"训练数据收集完成: {len(train_dataset_full)} samples")

    # 训练集统计
    train_gains = [s.gain.item() for s in train_dataset_full.samples]
    train_pos_ratio = sum(1 for g in train_gains if g > 0) / len(train_gains)
    logger.info(f"训练集正 gain 比例: {train_pos_ratio:.2%}")
    logger.info(f"训练集 gain 均值: {sum(train_gains)/len(train_gains):.3f}")

    # 5. 收集验证数据
    logger.info("收集验证数据...")
    val_collector = UpdateEvaluatorDataCollector(
        base_dataloader=val_base_loader,
        candidate_generator=candidate_gen,
        reward_config=config['reward'],
        device=device,
        use_weighted=config['candidate_gen']['use_weighted'],
        n_samples_per_batch=config['candidate_gen']['n_samples'],
    )

    val_dataset_full = val_collector.collect(
        n_batches=len(val_base_loader)
    )
    logger.info(f"验证数据收集完成: {len(val_dataset_full)} samples")

    # 验证集统计
    val_gains = [s.gain.item() for s in val_dataset_full.samples]
    val_pos_ratio = sum(1 for g in val_gains if g > 0) / len(val_gains)
    logger.info(f"验证集正 gain 比例: {val_pos_ratio:.2%}")
    logger.info(f"验证集 gain 均值: {sum(val_gains)/len(val_gains):.3f}")

    # 6. 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset_full,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=UpdateEvaluatorDataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset_full,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=UpdateEvaluatorDataset.collate_fn,
    )

    # 7. 创建模型
    logger.info("=" * 50)
    logger.info("Step 3: 创建 UpdateEvaluator")
    logger.info("=" * 50)

    evaluator_cfg = UpdateEvaluatorConfig(
        scene_dim=config['model']['scene_dim'],
        plan_len=config['model']['plan_len'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        lambda_collision=config['risk_weights']['collision'],
        lambda_offroad=config['risk_weights']['offroad'],
        lambda_comfort=config['risk_weights']['comfort'],
        lambda_drift=config['risk_weights']['drift'],
    )

    evaluator = UpdateEvaluator(evaluator_cfg)
    num_params = sum(p.numel() for p in evaluator.parameters() if p.requires_grad)
    logger.info(f"UpdateEvaluator 参数量: {num_params:,}")

    # 8. 训练
    logger.info("=" * 50)
    logger.info("Step 4: 训练（记录排序/过滤指标）")
    logger.info("=" * 50)
    logger.info("评估指标说明:")
    logger.info("  - spearman_gain: pred_gain 与 y_gain 的 Spearman 相关性（越高越好）")
    logger.info("  - kendall_gain: pred_gain 与 y_gain 的 Kendall tau（越高越好）")
    logger.info("  - retained_gain > filtered_gain: 保留的样本 gain 应更高")
    logger.info("  - retained_risk < filtered_risk: 保留的样本 risk 应更低")
    logger.info("=" * 50)

    train_cfg = EvaluatorTrainingConfig(
        alpha_gain=config['training']['alpha_gain'],
        beta_risk=config['training']['beta_risk'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        grad_clip=config['training']['grad_clip'],
        epochs=config['training']['epochs'],
        eval_every=config['training']['eval_every'],
        gain_threshold=config['training']['gain_threshold'],
    )

    trainer = UpdateEvaluatorTrainer(
        evaluator=evaluator,
        config=train_cfg,
        device=device,
    )

    # 训练时传入 val_dataloader 以启用排序指标计算
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=str(config['output_dir']),
    )

    # 9. 保存最终模型
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / 'update_evaluator_final.pth'
    trainer.save_checkpoint(final_path, train_cfg.epochs)
    logger.info(f"最终模型已保存到 {final_path}")

    # 10. 保存配置
    config_to_save = CONFIG.copy()
    config_to_save['candidate_gen']['sample_weights'] = dict(DEFAULT_SAMPLE_WEIGHTS)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False)

    logger.info("训练完成！")


if __name__ == '__main__':
    main()
