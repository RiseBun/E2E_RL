#!/usr/bin/env python
"""
三层防御诊断实验

实验 1: 只测 LearnedUpdateGate（关掉 STAPO）
实验 2: 统计不同 candidate 来源的 gain 分布
实验 3: 正 gain 样本被 STAPO 过滤的比例
"""

from __future__ import annotations

import torch
import sys
from pathlib import Path
from collections import defaultdict

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from E2E_RL.data.dataloader import build_vad_dataloader
from E2E_RL.update_selector import (
    UpdateEvaluator,
    UpdateEvaluatorConfig,
    STAPOGate,
    STAPOGateConfig,
    LearnedUpdateGate,
    CandidateCorrector,
    UpdateEvaluatorDataCollector,
    UpdateEvaluatorDataset,
)

# Correction types (from candidate_generator.py)
CORRECTION_TYPES = ['zero', 'policy_sample', 'deterministic', 'bounded_random', 'gt_directed', 'safety_biased']


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("=" * 60)
    logger.info("三层防御诊断实验")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("\n[Step 1] 加载数据...")
    base_dataloader = build_vad_dataloader(
        data_dir='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps',
        batch_size=32,
        num_workers=0,
    )
    logger.info(f"  数据集大小: {len(base_dataloader.dataset)}")

    # 2. 收集数据（记录每个 correction 的类型）
    logger.info("\n[Step 2] 收集数据并记录 correction 类型...")
    
    candidate_gen = CandidateCorrector(policy=None, max_corrections_per_type=1)
    data_collector = UpdateEvaluatorDataCollector(
        base_dataloader=base_dataloader,
        candidate_generator=candidate_gen,
        reward_config={'dt': 0.5},
        device=device,
        collect_all_types=True,
    )
    
    # 修改收集逻辑，记录 correction 类型
    from E2E_RL.update_selector.candidate_generator import compute_structured_stats
    from E2E_RL.refinement.reward_proxy import compute_refinement_reward

    dataset_samples = []  # (sample, correction_type)
    
    for batch in base_dataloader:
        interface = batch['interface'].to(device)
        gt_plan = batch['gt_plan'].to(device)
        plan_mask = batch.get('plan_mask')
        if plan_mask is not None:
            plan_mask = plan_mask.to(device)

        # 生成候选修正
        cand_result = candidate_gen.generate_all_types(interface, gt_plan)
        all_corrections = cand_result['all_corrections']
        types = cand_result['correction_types']

        B, N_corr, T, _ = all_corrections.shape

        for b in range(B):
            for n in range(N_corr):
                correction = all_corrections[b, n]
                corrected_plan = interface.reference_plan[b] + correction
                corr_type = types[n]

                # 计算 reward
                reward_info = compute_refinement_reward(
                    refined_plan=corrected_plan.unsqueeze(0),
                    gt_plan=gt_plan[b:b+1],
                    mask=plan_mask[b:b+1] if plan_mask is not None else None,
                    dt=0.5,
                )
                ref_reward_info = compute_refinement_reward(
                    refined_plan=interface.reference_plan[b:b+1],
                    gt_plan=gt_plan[b:b+1],
                    mask=plan_mask[b:b+1] if plan_mask is not None else None,
                    dt=0.5,
                )
                gain = reward_info['total_reward'] - ref_reward_info['total_reward']

                dataset_samples.append({
                    'gain': gain.item(),
                    'correction_type': corr_type,
                    'correction': correction,
                })

        if len(dataset_samples) >= 2000:  # 限制样本数
            break

    logger.info(f"  收集样本数: {len(dataset_samples)}")

    # =========================================================================
    # 实验 2: 统计不同 candidate 来源的 gain 分布
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("实验 2: 不同 correction 类型的 gain 分布")
    logger.info("=" * 60)

    type_stats = defaultdict(lambda: {'gains': [], 'count': 0})
    for sample in dataset_samples:
        t = sample['correction_type']
        type_stats[t]['gains'].append(sample['gain'])
        type_stats[t]['count'] += 1

    logger.info(f"\n{'类型':<20} {'数量':>6} {'正gain%':>10} {'均值':>10} {'标准差':>10}")
    logger.info("-" * 60)

    for t in CORRECTION_TYPES:
        stats = type_stats[t]
        if stats['count'] == 0:
            continue
        gains = stats['gains']
        pos_ratio = sum(1 for g in gains if g > 0) / len(gains)
        mean = sum(gains) / len(gains)
        std = (sum((g - mean)**2 for g in gains) / len(gains)) ** 0.5
        logger.info(f"{t:<20} {stats['count']:>6} {pos_ratio:>10.1%} {mean:>10.3f} {std:>10.3f}")

    # 汇总
    all_gains = [s['gain'] for s in dataset_samples]
    pos_ratio = sum(1 for g in all_gains if g > 0) / len(all_gains)
    logger.info("-" * 60)
    logger.info(f"{'总体':<20} {len(all_gains):>6} {pos_ratio:>10.1%} {sum(all_gains)/len(all_gains):>10.3f}")

    # 找出哪个类型把负 gain 比例拉高
    worst_type = min(type_stats.items(), key=lambda x: sum(x[1]['gains'])/len(x[1]['gains']) if x[1]['count'] > 0 else 0)
    logger.info(f"\n⚠️  最差类型: {worst_type[0]}, 均值={sum(worst_type[1]['gains'])/len(worst_type[1]['gains']):.3f}")

    # =========================================================================
    # 实验 3: 正 gain 样本被 STAPO 过滤的比例
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("实验 3: STAPO Gate 对正 gain 样本的影响")
    logger.info("=" * 60)

    stapo_gate = STAPOGate(STAPOGateConfig(enabled=True))

    # 统计正 gain 样本被 STAPO 过滤的情况
    positive_samples = [s for s in dataset_samples if s['gain'] > 0]
    positive_filtered = 0
    positive_retained = 0

    for sample in positive_samples:
        correction = sample['correction']
        
        # 模拟 policy 行为：计算 log_prob 和 entropy
        # 这里用 correction 的 magnitude 近似
        corr_norm = torch.norm(correction)
        log_prob = torch.log(corr_norm.clamp(min=1e-6))  # 近似
        entropy = 0.5  # 假设中等熵
        
        # STAPO 判断
        mask = stapo_gate.compute_mask(
            advantages=torch.tensor([sample['gain']]),
            action_log_probs=torch.tensor([log_prob.item()]),
        )
        
        if mask[0]:
            positive_retained += 1
        else:
            positive_filtered += 1

    logger.info(f"\n正 gain 样本数: {len(positive_samples)}")
    logger.info(f"  被 STAPO 保留: {positive_retained} ({positive_retained/len(positive_samples):.1%})")
    logger.info(f"  被 STAPO 过滤: {positive_filtered} ({positive_filtered/len(positive_samples):.1%})")

    if positive_filtered > positive_retained:
        logger.info("\n⚠️  警告: STAPO 过滤了大部分正 gain 样本！")
        logger.info("     问题可能在 STAPO 阈值，不在 evaluator 训练。")
    else:
        logger.info("\n✓  STAPO 保留了大部分正 gain 样本")

    # =========================================================================
    # 实验 1: 只测 LearnedUpdateGate（关掉 STAPO）
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("实验 1: 只用 LearnedUpdateGate（关掉 STAPO）")
    logger.info("=" * 60)

    # 加载 evaluator
    evaluator_ckpt = '/tmp/evaluator_test/update_evaluator_final.pth'
    if Path(evaluator_ckpt).exists():
        config = UpdateEvaluatorConfig(plan_len=6, hidden_dim=256)
        evaluator = UpdateEvaluator(config)
        ckpt = torch.load(evaluator_ckpt, map_location='cpu')
        evaluator.load_state_dict(ckpt['evaluator_state_dict'])
        evaluator.eval()
        logger.info(f"  加载模型: {evaluator_ckpt}")
    else:
        logger.info("  模型不存在，跳过 Learned Gate 测试")
        return

    learned_gate = LearnedUpdateGate(
        evaluator=evaluator,
        tau_gain=0.0,
        tau_risk=0.5,
        advantage_threshold=0.0,
    )

    # 测试 learned gate
    learned_retained_gains = []
    learned_filtered_gains = []

    for sample in dataset_samples[:500]:  # 限制数量
        correction = sample['correction'].unsqueeze(0).to(device)
        
        # 构造 interface
        class FakeInterface:
            def __init__(self):
                self.scene_token = torch.randn(1, 256).to(device)
                self.reference_plan = torch.randn(1, 6, 2).to(device)
                self.plan_confidence = None

        interface = FakeInterface()
        
        with torch.no_grad():
            result = learned_gate.predict(interface, correction)
            pred_gain = result['pred_gain'][0].item()
            is_harmful = result['is_harmful'][0].item()

        if is_harmful:
            learned_filtered_gains.append(sample['gain'])
        else:
            learned_retained_gains.append(sample['gain'])

    logger.info(f"\nLearned Gate 结果 (关掉 STAPO):")
    logger.info(f"  保留样本 gain 均值: {sum(learned_retained_gains)/len(learned_retained_gains):.3f}" if learned_retained_gains else "  无保留样本")
    logger.info(f"  过滤样本 gain 均值: {sum(learned_filtered_gains)/len(learned_filtered_gains):.3f}" if learned_filtered_gains else "  无过滤样本")
    
    if learned_retained_gains and learned_filtered_gains:
        diff = sum(learned_retained_gains)/len(learned_retained_gains) - sum(learned_filtered_gains)/len(learned_filtered_gains)
        if diff > 0:
            logger.info(f"\n✓  Learned Gate 工作正常: 保留的 gain 比过滤的高 {diff:.3f}")
        else:
            logger.info(f"\n⚠️  Learned Gate 也筛反了: 保留的 gain 比过滤的低 {-diff:.3f}")

    # =========================================================================
    # 总结
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("诊断总结")
    logger.info("=" * 60)
    
    if worst_type[0] in ['random', 'bounded_random']:
        logger.info("⚠️  主要问题: 随机扰动占主导，负 gain 比例高")
        logger.info("   建议: 增加 policy sample / GT-directed 候选比例")
    
    if positive_filtered > positive_retained:
        logger.info("⚠️  主要问题: STAPO Gate 误杀了正 gain 样本")
        logger.info("   建议: 调高 probability_threshold 或 entropy_threshold")
    
    if learned_retained_gains and learned_filtered_gains and diff < 0:
        logger.info("⚠️  主要问题: Learned Gate 也筛反了")
        logger.info("   建议: 需要重新训练，但先解决候选分布问题")


if __name__ == '__main__':
    main()
