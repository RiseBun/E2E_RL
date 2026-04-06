"""
保守接入 LearnedUpdateGate - 实验 C: SafetyGuard + LearnedUpdateGate

策略（保守接入）：
- SafetyGuard: 保留（硬底线，确保物理安全）
- STAPOGate: 弱兜底（宽松阈值，防止完全失控）
- LearnedUpdateGate: 主判断（核心过滤层）

目标：验证 LearnedUpdateGate 是否比规则 STAPO 更能提升 policy
"""

import torch
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from E2E_RL.data.dataloader import build_vad_dataloader
from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.correction_policy.losses import compute_advantage
from E2E_RL.rl_trainer.correction_policy_trainer import CorrectionPolicyTrainer
from E2E_RL.update_selector.safety_guard import SafetyGuard, SafetyGuardConfig
from E2E_RL.update_selector.stapo_gate import STAPOGate, STAPOGateConfig
from E2E_RL.update_selector.update_evaluator import UpdateEvaluator, UpdateEvaluatorConfig, LearnedUpdateGate

# =============================================================================
# 实验配置 - 保守接入
# =============================================================================
CONFIG = {
    # 数据配置 - 使用 5k 增强数据集
    'data': {
        'data_dir': '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps_full',  # 5000 样本
        'batch_size': 16,
    },
    # 模型配置
    'model': {
        'scene_dim': 256,
        'plan_len': 6,  # 必须与数据集一致！
        'hidden_dim': 256,
    },
    # 训练配置 - 短程训练用于快速验证
    'training': {
        'bc_epochs': 3,        # BC 预热 3 epochs
        'rl_epochs': 15,       # RL 训练 15 epochs（快速验证）
        'lr': 3e-4,
        'grad_clip': 1.0,
        'entropy_coef': 0.01,
    },
    # 奖励配置
    'reward': {
        'dt': 0.5,
        'w_progress': 1.0,
        'w_collision': 0.5,
        'w_offroad': 0.3,
        'w_comfort': 0.1,
    },
    # SafetyGuard - 硬底线配置
    'safety_guard': {
        'enabled': True,
        'max_residual_norm': 5.0,
        'max_step_disp': 2.0,
        'max_speed': 15.0,
    },
    # STAPOGate - 弱兜底配置（宽松阈值）
    'stapo_gate': {
        'enabled': True,           # 启用但用宽松阈值
        'advantage_threshold': 0.0,
        'probability_threshold': 0.05,   # 更宽松
        'entropy_threshold': 0.3,         # 更宽松
        'min_retention_ratio': 0.5,      # 最低保留 50%
    },
    # LearnedUpdateGate - 主判断配置
    'learned_gate': {
        'enabled': True,
        'tau_gain': 0.0,          # 适度过滤低 gain
        'tau_risk': 0.3,          # 过滤高 risk（保守）
        'advantage_threshold': 0.0,
    },
    # Evaluator checkpoint
    'evaluator_ckpt': '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth',
    # 输出目录
    'output_dir': '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/ab_comparison/expC_learned_gate',
}


def build_safety_guard(cfg: dict) -> SafetyGuard:
    """构建 SafetyGuard"""
    guard_cfg = cfg.get('safety_guard', {})
    return SafetyGuard(
        SafetyGuardConfig(
            enabled=guard_cfg.get('enabled', True),
            max_residual_norm=guard_cfg.get('max_residual_norm', 5.0),
            max_step_disp=guard_cfg.get('max_step_disp', 2.0),
            max_speed=guard_cfg.get('max_speed', 15.0),
            max_total_disp=guard_cfg.get('max_total_disp', 10.0),
            dt=cfg['reward'].get('dt', 0.5),
        )
    )


def build_stapo_gate(cfg: dict) -> STAPOGate:
    """构建 STAPOGate（弱兜底）"""
    gate_cfg = cfg.get('stapo_gate', {})
    return STAPOGate(
        STAPOGateConfig(
            enabled=gate_cfg.get('enabled', True),
            advantage_threshold=gate_cfg.get('advantage_threshold', 0.0),
            probability_threshold=gate_cfg.get('probability_threshold', 0.1),
            entropy_threshold=gate_cfg.get('entropy_threshold', 0.5),
            min_retention_ratio=gate_cfg.get('min_retention_ratio', 0.5),
            use_combined_threshold=True,
        )
    )


def build_learned_gate(cfg: dict, device: torch.device) -> tuple:
    """构建 LearnedUpdateGate"""
    if not cfg.get('learned_gate', {}).get('enabled', False):
        return None

    evaluator_ckpt = cfg.get('evaluator_ckpt')
    if not evaluator_ckpt or not Path(evaluator_ckpt).exists():
        raise FileNotFoundError(f"Evaluator checkpoint 不存在: {evaluator_ckpt}")

    evaluator_cfg = UpdateEvaluatorConfig(
        scene_dim=cfg['model']['scene_dim'],
        plan_len=cfg['model']['plan_len'],
        hidden_dim=cfg['model']['hidden_dim'],
    )
    evaluator = UpdateEvaluator(evaluator_cfg).to(device)
    ckpt = torch.load(evaluator_ckpt, map_location=device, weights_only=False)
    if 'evaluator_state_dict' in ckpt:
        evaluator.load_state_dict(ckpt['evaluator_state_dict'])
    else:
        evaluator.load_state_dict(ckpt)
    evaluator.eval()

    return LearnedUpdateGate(
        evaluator=evaluator,
        tau_gain=cfg['learned_gate'].get('tau_gain', 0.0),
        tau_risk=cfg['learned_gate'].get('tau_risk', 0.5),
        advantage_threshold=cfg['learned_gate'].get('advantage_threshold', 0.0),
    )


def main():
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存实验配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)

    # =================================================================
    # Step 1: 加载数据
    # =================================================================
    logger.info("=" * 60)
    logger.info("Step 1: 加载数据")
    logger.info("=" * 60)

    dataloader = build_vad_dataloader(
        data_dir=CONFIG['data']['data_dir'],
        batch_size=CONFIG['data']['batch_size'],
        num_workers=0,
        shuffle=True,
    )
    logger.info(
        f"数据量: {len(dataloader.dataset)} samples, "
        f"batch_size: {CONFIG['data']['batch_size']}"
    )

    # =================================================================
    # Step 2: 创建 Policy
    # =================================================================
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

    # =================================================================
    # Step 3: 构建三层防御（保守接入）
    # =================================================================
    logger.info("=" * 60)
    logger.info("Step 3: 构建三层防御（保守接入）")
    logger.info("=" * 60)

    # SafetyGuard: 硬底线
    safety_guard = build_safety_guard(CONFIG)
    logger.info(
        f"[C] SafetyGuard: {'启用' if safety_guard.cfg.enabled else '禁用'} "
        f"(max_residual={CONFIG['safety_guard'].get('max_residual_norm', 5.0)})"
    )

    # STAPOGate: 弱兜底
    stapo_gate = build_stapo_gate(CONFIG)
    logger.info(
        f"[C] STAPOGate: {'启用 (弱兜底)' if stapo_gate.cfg.enabled else '禁用'} "
        f"(prob_th={CONFIG['stapo_gate'].get('probability_threshold', 0.1)}, "
        f"entropy_th={CONFIG['stapo_gate'].get('entropy_threshold', 0.5)})"
    )

    # LearnedUpdateGate: 主判断
    learned_gate = build_learned_gate(CONFIG, device)
    if learned_gate:
        logger.info(
            f"[C] LearnedUpdateGate: 启用 (主判断层) "
            f"(tau_gain={CONFIG['learned_gate']['tau_gain']}, "
            f"tau_risk={CONFIG['learned_gate']['tau_risk']})"
        )
    else:
        logger.warning("[C] LearnedUpdateGate: 禁用")

    logger.info("=" * 60)
    logger.info("防御层级配置 (实验 C: SafetyGuard + LearnedUpdateGate):")
    logger.info("  1. SafetyGuard: ✓ 启用（硬底线，物理安全）")
    logger.info("  2. STAPOGate: ✓ 启用（弱兜底）")
    logger.info(f"  3. LearnedUpdateGate: {'✓ 启用 (主判断)' if learned_gate else '✗ 禁用'}")
    logger.info("=" * 60)

    # =================================================================
    # Step 4: 创建 Trainer
    # =================================================================
    logger.info("=" * 60)
    logger.info("Step 4: 创建 Trainer")
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

    # =================================================================
    # Step 5: 训练
    # =================================================================
    logger.info("=" * 60)
    logger.info("Step 5: 开始训练")
    logger.info("=" * 60)

    metrics = trainer.train(
        dataloader=dataloader,
        bc_epochs=CONFIG['training']['bc_epochs'],
        rl_epochs=CONFIG['training']['rl_epochs'],
        output_dir=str(output_dir),
        save_every=5,
    )

    # =================================================================
    # Step 6: 保存最终模型
    # =================================================================
    final_path = output_dir / 'policy_final.pth'
    trainer.save_checkpoint(
        str(final_path),
        CONFIG['training']['bc_epochs'] + CONFIG['training']['rl_epochs'],
    )
    logger.info(f"最终模型已保存到 {final_path}")

    # =================================================================
    # Step 7: 保存训练指标
    # =================================================================
    logger.info("=" * 60)
    logger.info("Step 7: 保存训练指标")
    logger.info("=" * 60)

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

    # =================================================================
    # Step 8: 分析结果
    # =================================================================
    logger.info("=" * 60)
    logger.info("Step 8: 训练结果分析")
    logger.info("=" * 60)

    if metrics['rl_metrics']:
        logger.info("\n--- RL 训练曲线 ---")
        for i, m in enumerate(metrics['rl_metrics']):
            logger.info(
                f"Epoch {i}: "
                f"loss={m['loss_total']:.4f}, "
                f"adv={m['mean_advantage']:.4f}, "
                f"retention={m['retention_ratio']:.2%}, "
                f"entropy_loss={m['loss_entropy']:.4f}"
            )

        final_rl = metrics['rl_metrics'][-1]
        logger.info(f"\n最终 retention ratio: {final_rl['retention_ratio']:.2%}")

        if 'learned_retention' in final_rl:
            logger.info(
                f"Learned gate retention: {final_rl['learned_retention']:.2%}"
            )

        # 检查 retention ratio
        retention = final_rl['retention_ratio']
        if 0.6 <= retention <= 0.9:
            logger.info(
                f"✓ Retention ratio 在理想范围 [60%-90%]: {retention:.2%}"
            )
        elif retention < 0.6:
            logger.warning(
                f"⚠ Retention ratio 过低 [{retention:.2%}]，gate 可能太狠"
            )
        else:
            logger.warning(
                f"⚠ Retention ratio 过高 [{retention:.2%}]，gate 可能没起作用"
            )

    logger.info("\n训练完成！")
    return metrics


if __name__ == '__main__':
    main()
