"""
实验 B: SafetyGuard + STAPOGate (规则基线 - 放宽 SafetyGuard)

策略：
- SafetyGuard: 启用（放宽约束）
- STAPOGate: 启用（规则过滤）
- LearnedUpdateGate: 禁用
"""

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

CONFIG = {
    'data': {
        'data_dir': '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps_full',
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
        'w_comfort': 0.01,  # 减小 comfort 权重
        'fde_scale': 5.0,   # 调整 FDE 归一化因子
    },
    'output_dir': '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/ab_comparison_v2/expB_stapo_gate',
}


def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Step 1: 加载数据")
    logger.info("=" * 60)

    dataloader = build_vad_dataloader(
        data_dir=CONFIG['data']['data_dir'],
        batch_size=CONFIG['data']['batch_size'],
        num_workers=0,
        shuffle=True,
    )
    logger.info(f"数据量: {len(dataloader.dataset)} samples")

    logger.info("=" * 60)
    logger.info("Step 2: 创建 Policy")
    logger.info("=" * 60)

    policy = CorrectionPolicy(
        scene_dim=CONFIG['model']['scene_dim'],
        plan_len=CONFIG['model']['plan_len'],
        hidden_dim=CONFIG['model']['hidden_dim'],
    )
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Policy 参数量: {num_params:,}")

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=CONFIG['training']['lr'],
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

    # STAPOGate: 启用（标准规则配置）
    stapo_gate = STAPOGate(
        STAPOGateConfig(
            enabled=True,
            advantage_threshold=0.0,
            probability_threshold=0.1,
            entropy_threshold=0.5,
            min_retention_ratio=0.3,
            use_combined_threshold=True,
        )
    )

    learned_gate = None

    logger.info("=" * 60)
    logger.info("防御层级配置 (实验 B: SafetyGuard + STAPOGate)")
    logger.info("  1. SafetyGuard: ✓ 启用（放宽约束）")
    logger.info("  2. STAPOGate: ✓ 启用（规则过滤）")
    logger.info("  3. LearnedUpdateGate: ✗ 禁用")
    logger.info("=" * 60)

    trainer = CorrectionPolicyTrainer(
        policy=policy,
        optimizer=optimizer,
        device=device,
        reward_config=CONFIG['reward'],
        safety_guard=safety_guard,
        stapo_gate=stapo_gate,
        learned_gate=learned_gate,
        entropy_coef=CONFIG['training']['entropy_coef'],
        grad_clip=CONFIG['training']['grad_clip'],
    )

    logger.info("=" * 60)
    logger.info("Step 3: 开始训练")
    logger.info("=" * 60)

    metrics = trainer.train(
        dataloader=dataloader,
        bc_epochs=CONFIG['training']['bc_epochs'],
        rl_epochs=CONFIG['training']['rl_epochs'],
        output_dir=str(output_dir),
        save_every=5,
    )

    final_path = output_dir / 'policy_final.pth'
    trainer.save_checkpoint(
        str(final_path),
        CONFIG['training']['bc_epochs'] + CONFIG['training']['rl_epochs'],
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
                f"stapo_retention={m.get('stapo_retention', 0):.2%}, "
                f"retained_adv={m.get('retained_advantage_mean', 0):.4f}"
            )

        final_rl = metrics['rl_metrics'][-1]
        logger.info(f"\n最终 retention ratio: {final_rl['retention_ratio']:.2%}")

    logger.info("实验 B 完成！")


if __name__ == '__main__':
    main()
