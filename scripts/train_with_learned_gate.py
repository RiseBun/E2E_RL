"""
保守接入 LearnedUpdateGate

策略：
- SafetyGuard: 保留（硬底线）
- STAPOGate: 只做日志/诊断，不作为主要过滤
- LearnedUpdateGate: 主要过滤层

目标：验证 LearnedUpdateGate 是否比规则 gate 更会挑"对 RL 有益的 update"
"""

import torch
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from E2E_RL.data.dataloader import build_vad_dataloader
from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.correction_policy.losses import compute_advantage
from E2E_RL.rl_trainer.correction_policy_trainer import CorrectionPolicyTrainer
from E2E_RL.update_selector.safety_guard import SafetyGuard, SafetyGuardConfig
from E2E_RL.update_selector.stapo_gate import STAPOGate, STAPOGateConfig
from E2E_RL.update_selector.update_evaluator import UpdateEvaluator, UpdateEvaluatorConfig, LearnedUpdateGate

# 配置
CONFIG = {
    'data': {
        'data_dir': '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps',
        'batch_size': 16,
    },
    'model': {
        'scene_dim': 256,
        'plan_len': 6,
        'hidden_dim': 256,
    },
    'training': {
        'bc_epochs': 5,
        'rl_epochs': 20,
        'lr': 3e-4,
        'grad_clip': 1.0,
        'entropy_coef': 0.01,
    },
    'reward': {
        'dt': 0.5,
        'w_progress': 1.0,
        'w_collision': 0.5,
        'w_offroad': 0.3,
        'w_comfort': 0.1,
    },
    'learned_gate': {
        'tau_gain': 0.0,     # gain 阈值
        'tau_risk': 0.5,     # risk 阈值
        'advantage_threshold': 0.0,
    },
    'output_dir': '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/correction_policy_with_learned_gate',
}


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 1. 加载数据
    logger.info("=" * 50)
    logger.info("Step 1: 加载数据")
    logger.info("=" * 50)

    dataloader = build_vad_dataloader(
        data_dir=CONFIG['data']['data_dir'],
        batch_size=CONFIG['data']['batch_size'],
        num_workers=0,
        shuffle=True,
    )
    logger.info(f"数据量: {len(dataloader.dataset)} samples")

    # 2. 创建 Policy
    logger.info("=" * 50)
    logger.info("Step 2: 创建 Policy")
    logger.info("=" * 50)

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

    # 3. 创建三层防御
    logger.info("=" * 50)
    logger.info("Step 3: 创建三层防御（保守接入）")
    logger.info("=" * 50)

    # SafetyGuard: 暂时禁用，只用 LearnedUpdateGate（保守策略）
    safety_guard = SafetyGuard(SafetyGuardConfig(enabled=False))
    logger.info("✗ SafetyGuard: 禁用（只用于验证）")

    # STAPOGate: 只做日志/诊断，不作为主要过滤
    stapo_gate = STAPOGate(STAPOGateConfig(enabled=False))  # 禁用，只记录
    logger.info("✗ STAPOGate: 禁用（只做日志）")

    # LearnedUpdateGate: 主要过滤层
    evaluator_ckpt = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/update_evaluator_v3/update_evaluator_final.pth'
    if Path(evaluator_ckpt).exists():
        logger.info(f"加载 Evaluator from {evaluator_ckpt}")

        evaluator_cfg = UpdateEvaluatorConfig(
            scene_dim=CONFIG['model']['scene_dim'],
            plan_len=CONFIG['model']['plan_len'],
            hidden_dim=CONFIG['model']['hidden_dim'],
        )
        evaluator = UpdateEvaluator(evaluator_cfg).to(device)
        ckpt = torch.load(evaluator_ckpt, map_location=device)
        evaluator.load_state_dict(ckpt['evaluator_state_dict'])
        evaluator.eval()

        learned_gate = LearnedUpdateGate(
            evaluator=evaluator,
            tau_gain=CONFIG['learned_gate']['tau_gain'],
            tau_risk=CONFIG['learned_gate']['tau_risk'],
            advantage_threshold=CONFIG['learned_gate']['advantage_threshold'],
        )
        logger.info(f"✓ LearnedUpdateGate: 启用（主要过滤层）")
    else:
        logger.warning(f"Evaluator checkpoint 不存在: {evaluator_ckpt}")
        logger.warning("使用 STAPOGate 作为后备（保守策略）")
        learned_gate = None
        stapo_gate = STAPOGate(STAPOGateConfig(enabled=True))

    # 4. 创建 Trainer
    logger.info("=" * 50)
    logger.info("Step 4: 创建 Trainer（保守接入）")
    logger.info("=" * 50)

    trainer = CorrectionPolicyTrainer(
        policy=policy,
        optimizer=optimizer,
        device=device,
        reward_config=CONFIG['reward'],
        safety_guard=safety_guard,
        stapo_gate=stapo_gate,  # 只做日志
        learned_gate=learned_gate,  # 主要过滤
        entropy_coef=CONFIG['training']['entropy_coef'],
        grad_clip=CONFIG['training']['grad_clip'],
    )

    # 5. 训练
    logger.info("=" * 50)
    logger.info("Step 5: 训练（保守策略）")
    logger.info("=" * 50)
    logger.info("防御层级配置:")
    logger.info("  1. SafetyGuard: ✓ 启用（硬底线）")
    logger.info("  2. STAPOGate: ✗ 禁用（只做日志）")
    logger.info(f"  3. LearnedUpdateGate: {'✓ 启用' if learned_gate else '✗ 禁用'}")
    logger.info("=" * 50)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = trainer.train(
        dataloader=dataloader,
        bc_epochs=CONFIG['training']['bc_epochs'],
        rl_epochs=CONFIG['training']['rl_epochs'],
        output_dir=str(output_dir),
        save_every=5,
    )

    # 6. 保存最终模型
    final_path = output_dir / 'policy_final.pth'
    trainer.save_checkpoint(str(final_path), CONFIG['training']['bc_epochs'] + CONFIG['training']['rl_epochs'])
    logger.info(f"最终模型已保存到 {final_path}")

    # 7. 分析训练结果
    logger.info("=" * 50)
    logger.info("Step 6: 训练结果分析")
    logger.info("=" * 50)

    if metrics['rl_metrics']:
        final_rl = metrics['rl_metrics'][-1]
        logger.info(f"最终 RL retention ratio: {final_rl['retention_ratio']:.2%}")
        if 'learned_retention' in final_rl:
            logger.info(f"Learned gate retention: {final_rl['retention_ratio']:.2%}")

    logger.info("训练完成！")


if __name__ == '__main__':
    main()
