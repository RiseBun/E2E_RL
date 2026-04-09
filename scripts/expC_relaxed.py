"""
实验 C: SafetyGuard + LearnedUpdateGate (放宽 SafetyGuard)

策略（保守接入）：
- SafetyGuard: 保留（放宽约束）
- STAPOGate: 禁用（让 LearnedUpdateGate 单独工作）
- LearnedUpdateGate: 主判断

用法:
    # 使用默认配置
    python scripts/expC_relaxed.py
    
    # 使用命令行参数覆盖
    python scripts/expC_relaxed.py \
        --data_dir data/vad_dumps_full \
        --evaluator_ckpt experiments/update_evaluator/evaluator_final.pth \
        --output_dir experiments/expC \
        --rl_epochs 50
"""

import argparse
import torch
import sys
import json
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from E2E_RL.data.dataloader import build_vad_dataloader
from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.rl_trainer.correction_policy_trainer import CorrectionPolicyTrainer
from E2E_RL.update_selector.safety_guard import SafetyGuard, SafetyGuardConfig
from E2E_RL.update_selector.stapo_gate import STAPOGate, STAPOGateConfig
from E2E_RL.update_selector.update_evaluator import UpdateEvaluator, UpdateEvaluatorConfig, LearnedUpdateGate

# 默认配置
DEFAULT_CONFIG = {
    'data': {
        'data_dir': 'data/vad_dumps_full',
        'batch_size': 16,
    },
    'model': {
        'scene_dim': 256,
        'plan_len': 6,
        'hidden_dim': 256,
    },
    'training': {
        'bc_epochs': 3,
        'rl_epochs': 15,
        'lr': 3e-4,
        'grad_clip': 1.0,
        'entropy_coef': 0.01,
    },
    'reward': {
        'dt': 0.5,
        'w_progress': 1.0,
        'w_collision': 0.5,
        'w_offroad': 0.3,
        'w_comfort': 0.01,
        'fde_scale': 5.0,
    },
    'evaluator_ckpt': 'experiments/update_evaluator/evaluator_final.pth',
    'output_dir': 'experiments/expC_learned_gate',
}


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='实验 C: SafetyGuard + LearnedUpdateGate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/expC_relaxed.py
  
  # 覆盖关键参数
  python scripts/expC_relaxed.py \\
      --data_dir data/sparsedrive_dumps_full \\
      --evaluator_ckpt experiments/sparsedrive_evaluator/evaluator_final.pth \\
      --output_dir experiments/sparsedrive_expC \\
      --rl_epochs 50
        """,
    )
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录 (默认: data/vad_dumps_full)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (默认: 16)')
    
    # 模型参数
    parser.add_argument('--scene_dim', type=int, default=None,
                        help='场景特征维度 (默认: 256)')
    parser.add_argument('--plan_len', type=int, default=None,
                        help='轨迹长度 (默认: 6)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='隐藏层维度 (默认: 256)')
    
    # 训练参数
    parser.add_argument('--bc_epochs', type=int, default=None,
                        help='Behavioral Cloning 预热轮数 (默认: 3)')
    parser.add_argument('--rl_epochs', type=int, default=None,
                        help='Reinforcement Learning 轮数 (默认: 15)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认: 3e-4)')
    
    # 评估器参数
    parser.add_argument('--evaluator_ckpt', type=str, default=None,
                        help='UpdateEvaluator 检查点路径')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: experiments/expC_learned_gate)')
    
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
    
    # 模型参数
    if args.scene_dim is not None:
        config['model']['scene_dim'] = args.scene_dim
    if args.plan_len is not None:
        config['model']['plan_len'] = args.plan_len
    if args.hidden_dim is not None:
        config['model']['hidden_dim'] = args.hidden_dim
    
    # 训练参数
    if args.bc_epochs is not None:
        config['training']['bc_epochs'] = args.bc_epochs
    if args.rl_epochs is not None:
        config['training']['rl_epochs'] = args.rl_epochs
    if args.lr is not None:
        config['training']['lr'] = args.lr
    
    # 评估器参数
    if args.evaluator_ckpt is not None:
        config['evaluator_ckpt'] = args.evaluator_ckpt
    
    # 输出参数
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    
    return config


def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    logger = logging.getLogger(__name__)
    
    # 解析命令行参数
    args = parse_args()
    config = merge_config(args)
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("实验 C: SafetyGuard + LearnedUpdateGate")
    logger.info("=" * 60)
    logger.info("配置:")
    logger.info(json.dumps(config, indent=2, default=str))
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"配置已保存到: {output_dir / 'config.json'}")

    logger.info("=" * 60)
    logger.info("Step 1: 加载数据")
    logger.info("=" * 60)

    dataloader = build_vad_dataloader(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=0,
        shuffle=True,
    )
    logger.info(f"数据量: {len(dataloader.dataset)} samples")

    logger.info("=" * 60)
    logger.info("Step 2: 创建 Policy")
    logger.info("=" * 60)

    policy = CorrectionPolicy(
        scene_dim=config['model']['scene_dim'],
        plan_len=config['model']['plan_len'],
        hidden_dim=config['model']['hidden_dim'],
    )
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Policy 参数量: {num_params:,}")

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config['training']['lr'],
        weight_decay=1e-4,
    )

    # SafetyGuard: 放宽约束
    safety_guard = SafetyGuard(
        SafetyGuardConfig(
            enabled=True,
            max_residual_norm=50.0,
            max_step_disp=20.0,
            max_speed=50.0,
            max_total_disp=100.0,
            dt=0.5,
        )
    )

    # STAPOGate: 禁用，让 LearnedUpdateGate 单独工作
    stapo_gate = STAPOGate(STAPOGateConfig(enabled=False))

    # LearnedUpdateGate: 加载预训练的 Evaluator
    evaluator_ckpt = config['evaluator_ckpt']
    if Path(evaluator_ckpt).exists():
        logger.info(f"加载 Evaluator from {evaluator_ckpt}")
        evaluator_cfg = UpdateEvaluatorConfig(
            scene_dim=config['model']['scene_dim'],
            plan_len=config['model']['plan_len'],
            hidden_dim=config['model']['hidden_dim'],
        )
        evaluator = UpdateEvaluator(evaluator_cfg).to(device)
        ckpt = torch.load(evaluator_ckpt, map_location=device, weights_only=False)
        if 'evaluator_state_dict' in ckpt:
            evaluator.load_state_dict(ckpt['evaluator_state_dict'])
        else:
            evaluator.load_state_dict(ckpt)
        evaluator.eval()

        learned_gate = LearnedUpdateGate(
            evaluator=evaluator,
            tau_gain=0.0,
            tau_risk=0.3,
            advantage_threshold=0.0,
        )
    else:
        logger.warning(f"Evaluator checkpoint 不存在: {evaluator_ckpt}")
        learned_gate = None

    logger.info("=" * 60)
    logger.info("防御层级配置 (实验 C: SafetyGuard + LearnedUpdateGate)")
    logger.info("  1. SafetyGuard: ✓ 启用（放宽约束）")
    logger.info("  2. STAPOGate: ✗ 禁用")
    logger.info(f"  3. LearnedUpdateGate: {'✓ 启用' if learned_gate else '✗ 禁用'}")
    logger.info("=" * 60)

    trainer = CorrectionPolicyTrainer(
        policy=policy,
        optimizer=optimizer,
        device=device,
        reward_config=config['reward'],
        safety_guard=safety_guard,
        stapo_gate=stapo_gate,
        learned_gate=learned_gate,
        entropy_coef=config['training']['entropy_coef'],
        grad_clip=config['training']['grad_clip'],
    )

    logger.info("=" * 60)
    logger.info("Step 3: 开始训练")
    logger.info("=" * 60)

    metrics = trainer.train(
        dataloader=dataloader,
        bc_epochs=config['training']['bc_epochs'],
        rl_epochs=config['training']['rl_epochs'],
        output_dir=str(output_dir),
        save_every=5,
    )

    final_path = output_dir / 'policy_final.pth'
    trainer.save_checkpoint(
        str(final_path),
        config['training']['bc_epochs'] + config['training']['rl_epochs'],
    )
    logger.info(f"最终模型已保存到 {final_path}")

    metrics_serializable = {
        'bc_metrics': [
            {k: float(v) if isinstance(v, (int, float)) else v
             for k, v in m.items()}
            for m in metrics.get('bc_metrics', [])
        ],
        'rl_metrics': [
            {k: float(v) if isinstance(v, (int, float)) else v
             for k, v in m.items()}
            for m in metrics.get('rl_metrics', [])
        ],
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    logger.info("=" * 60)
    logger.info("训练结果")
    logger.info("=" * 60)

    if metrics['rl_metrics']:
        for i, m in enumerate(metrics['rl_metrics']):
            logger.info(
                f"RL Epoch {i}: "
                f"loss={m['loss_total']:.4f}, "
                f"adv={m['mean_advantage']:.4f}, "
                f"retention={m['retention_ratio']:.2%}, "
                f"learned_ret={m.get('learned_retention', 0):.2%}, "
                f"retained_adv={m.get('retained_advantage_mean', 0):.4f}, "
                f"filtered_adv={m.get('filtered_advantage_mean', 0):.4f}"
            )

        final_rl = metrics['rl_metrics'][-1]
        logger.info(f"\n最终 retention ratio: {final_rl['retention_ratio']:.2%}")

        # 检查 retained vs filtered
        if 'retained_advantage_mean' in final_rl and 'filtered_advantage_mean' in final_rl:
            logger.info(f"Retained advantage: {final_rl['retained_advantage_mean']:.4f}")
            logger.info(f"Filtered advantage: {final_rl['filtered_advantage_mean']:.4f}")
            if final_rl['retained_advantage_mean'] > final_rl['filtered_advantage_mean']:
                logger.info("✓ Learned gate 在挑选更高 advantage 的样本！")

    logger.info("实验 C 完成！")


if __name__ == '__main__':
    main()
