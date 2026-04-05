#!/usr/bin/env python
"""
三层防御验证脚本

用法:
    # 快速验证（只跑层 1）
    python scripts/validate_defense_layers.py --mode quick

    # 完整验证（跑全部 4 层）
    python scripts/validate_defense_layers.py --mode full

    # 单层验证
    python scripts/validate_defense_layers.py --mode layer1
    python scripts/validate_defense_layers.py --mode layer2
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

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.update_selector import (
    UpdateEvaluator,
    UpdateEvaluatorConfig,
    SafetyGuard,
    SafetyGuardConfig,
    STAPOGate,
    STAPOGateConfig,
    LearnedUpdateGate,
    DefenseLayerValidator,
    ValidationConfig,
    UpdateEvaluatorDataCollector,
    UpdateEvaluatorDataset,
)
from E2E_RL.data.dataloader import build_vad_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_evaluator(ckpt_path: str, device: torch.device) -> UpdateEvaluator:
    """加载训练好的 UpdateEvaluator。"""
    config = UpdateEvaluatorConfig(plan_len=6, hidden_dim=256)
    evaluator = UpdateEvaluator(config)
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        evaluator.load_state_dict(ckpt['evaluator_state_dict'])
        logger.info(f'从 {ckpt_path} 加载模型')
    else:
        logger.warning(f'{ckpt_path} 不存在，使用随机初始化模型')
    
    return evaluator.to(device)


def build_gates(evaluator: UpdateEvaluator, device: torch.device):
    """构建三层防御门控。"""
    safety_gate = SafetyGuard(SafetyGuardConfig(enabled=True))
    stapo_gate = STAPOGate(STAPOGateConfig(enabled=True))
    learned_gate = LearnedUpdateGate(
        evaluator=evaluator,
        tau_gain=0.0,
        tau_risk=0.5,
        advantage_threshold=0.0,
    )
    return safety_gate, stapo_gate, learned_gate


def gate_fn(batch: Dict, safety_gate, stapo_gate, learned_gate, device):
    """组合门控函数。"""
    advantages = batch['gain'].float().to(device)
    B = len(advantages)

    # 层 1: Safety Guard
    safety_mask = torch.ones(B, dtype=torch.bool, device=device)

    # 层 2: STAPO Gate
    stapo_mask = stapo_gate.compute_mask(
        advantages=advantages,
        action_log_probs=torch.zeros_like(advantages),
    )

    # 层 3: Learned Gate
    scene_token = batch['scene_token'].to(device)
    reference_plan = batch['reference_plan'].to(device)
    correction = batch['correction'].to(device)

    # 构造简单的 interface
    class SimpleInterface:
        def __init__(self, scene_token, reference_plan):
            self.scene_token = scene_token
            self.reference_plan = reference_plan
            self.plan_confidence = None

    interface = SimpleInterface(scene_token, reference_plan)

    learned_mask, diag = learned_gate.compute_mask(
        advantages=advantages,
        interface=interface,
        correction=correction,
        safety_mask=safety_mask & stapo_mask,
    )

    final_mask = safety_mask & stapo_mask & learned_mask
    return final_mask, diag


def dummy_train_fn(config: Dict, epochs: int) -> Dict:
    """模拟训练函数（实际使用时替换为真实训练）。"""
    return {
        'entropy_std': 0.1,
        'mean_advantage': 0.5,
        'final_loss': 0.2,
    }


def dummy_eval_fn(training_result: Dict) -> Dict:
    """模拟评估函数（实际使用时替换为真实评估）。"""
    return {
        'ade': 1.2,
        'fde': 2.3,
        'collision': 0.05,
        'offroad': 0.02,
        'comfort': 0.1,
        'overall_score': 0.85,
    }


def main():
    parser = argparse.ArgumentParser(description='三层防御验证')
    parser.add_argument('--mode', type=str, default='quick', 
                        choices=['quick', 'full', 'layer1', 'layer2', 'layer3', 'layer4'],
                        help='验证模式')
    parser.add_argument('--evaluator_ckpt', type=str, 
                        default='/tmp/evaluator_test/update_evaluator_final.pth',
                        help='UpdateEvaluator 检查点路径')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/validation',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据
    logger.info('加载数据...')
    base_dataloader = build_vad_dataloader(
        data_dir=args.data_dir,
        batch_size=32,
        num_workers=0,
        shuffle=False,
    )
    logger.info(f'数据加载完成: {len(base_dataloader.dataset)} samples')

    # 2. 收集训练数据（用于层 1/2 验证）
    logger.info('收集训练数据...')
    from E2E_RL.update_selector.candidate_generator import CandidateCorrector
    from E2E_RL.update_selector.update_evaluator import UpdateEvaluator, UpdateEvaluatorConfig

    candidate_gen = CandidateCorrector(policy=None, max_corrections_per_type=1)
    data_collector = UpdateEvaluatorDataCollector(
        base_dataloader=base_dataloader,
        candidate_generator=candidate_gen,
        reward_config={'dt': 0.5},
        device=device,
        collect_all_types=True,
    )
    dataset = data_collector.collect(n_batches=10)
    logger.info(f'收集完成: {len(dataset)} samples')

    # 创建 DataLoader
    eval_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=UpdateEvaluatorDataset.collate_fn,
    )

    # 3. 加载模型
    logger.info('加载 UpdateEvaluator...')
    evaluator = load_evaluator(args.evaluator_ckpt, device)

    # 4. 构建门控
    safety_gate, stapo_gate, learned_gate = build_gates(evaluator, device)

    # 5. 创建验证器
    validator = DefenseLayerValidator(device=device)

    # 6. 执行验证
    if args.mode == 'quick' or args.mode == 'layer1':
        # 层 1: 标签层验证
        logger.info('=' * 60)
        logger.info('层 1: 标签层验证')
        logger.info('=' * 60)
        result = validator.validate_layer1_label_quality(
            eval_dataloader, evaluator, n_batches=50
        )
        
        logger.info('\n[层 1 结果]')
        logger.info(f"  Spearman (gain): {result['spearman_gain']:.4f}")
        logger.info(f"  Kendall (gain): {result['kendall_gain']:.4f}")
        logger.info(f"  Spearman (total risk): {result['spearman_total_risk']:.4f}")
        logger.info(f"  Top-3 hit rate: {result['topk_hit_rates'][3]:.2%}")
        logger.info(f"  综合评分: {result['overall_score']:.4f}")
        
        pass_offline = result['spearman_gain'] > 0.7 and result['spearman_total_risk'] > 0.5
        logger.info(f"\n判定: {'✓ PASS' if pass_offline else '✗ FAIL'}")
        
        # 保存结果
        output_path = os.path.join(args.output_dir, 'layer1_result.yaml')
        with open(output_path, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        logger.info(f'结果已保存到 {output_path}')

    elif args.mode == 'full' or args.mode == 'layer2':
        # 层 2: 分布层验证
        logger.info('=' * 60)
        logger.info('层 2: 分布层验证')
        logger.info('=' * 60)
        
        result_l2 = validator.validate_layer2_distribution(
            eval_dataloader,
            lambda b: gate_fn(b, safety_gate, stapo_gate, learned_gate, device),
            n_batches=50,
        )
        
        logger.info('\n[层 2 结果]')
        logger.info(f"  保留样本数: {result_l2['n_retained']}")
        logger.info(f"  过滤样本数: {result_l2['n_filtered']}")
        logger.info(f"  Gain 差异: {result_l2['gain_difference']:.4f}")
        logger.info(f"  过滤质量: {result_l2['filter_quality']}")
        logger.info(f"  Spurious 比例: retained={result_l2['retained_spurious_ratio']:.2%}, filtered={result_l2['filtered_spurious_ratio']:.2%}")
        
        # 保存结果
        output_path = os.path.join(args.output_dir, 'layer2_result.yaml')
        with open(output_path, 'w') as f:
            yaml.dump(result_l2, f, default_flow_style=False)
        logger.info(f'结果已保存到 {output_path}')

    elif args.mode == 'layer3':
        # 层 3: 反事实训练验证
        logger.info('=' * 60)
        logger.info('层 3: 反事实训练验证')
        logger.info('=' * 60)
        
        configs = [
            {'name': 'A_no_filter', 'filter': 'none'},
            {'name': 'B_stapo', 'filter': 'stapo'},
            {'name': 'C_learned', 'filter': 'learned'},
            {'name': 'D_cascade', 'filter': 'cascade'},
        ]
        
        result_l3 = validator.validate_layer3_ablation(
            dummy_train_fn, dummy_eval_fn,
            configs=configs, short_epochs=3, long_epochs=5,
        )
        
        logger.info('\n[层 3 结果]')
        for name, metrics in result_l3['comparison_summary'].get('all_metrics', {}).items():
            logger.info(f"  {name}: ADE={metrics['final_ade']:.3f}, collision={metrics['final_collision']:.3f}")
        
        # 保存结果
        output_path = os.path.join(args.output_dir, 'layer3_result.yaml')
        with open(output_path, 'w') as f:
            yaml.dump(result_l3, f, default_flow_style=False)
        logger.info(f'结果已保存到 {output_path}')

    elif args.mode == 'layer4':
        # 层 4: 最终策略验证
        logger.info('=' * 60)
        logger.info('层 4: 最终策略验证')
        logger.info('=' * 60)
        logger.info('需要先完成层 3 对照实验')

    elif args.mode == 'full':
        # 完整 4 层验证
        logger.info('=' * 60)
        logger.info('完整 4 层验证')
        logger.info('=' * 60)
        
        report = validator.run_full_validation(
            eval_dataloader,
            evaluator,
            lambda b: gate_fn(b, safety_gate, stapo_gate, learned_gate, device),
            dummy_train_fn,
            dummy_eval_fn,
        )
        
        logger.info('\n' + '=' * 60)
        logger.info('最终判定')
        logger.info('=' * 60)
        logger.info(f"  验证结果: {report['verdict']}")
        for reason in report['reasons']:
            logger.info(f"  - {reason}")
        
        # 保存完整报告
        output_path = os.path.join(args.output_dir, 'full_validation_report.yaml')
        validator.save_report(report, output_path)
        logger.info(f'完整报告已保存到 {output_path}')

    logger.info('\n验证完成！')


if __name__ == '__main__':
    main()
