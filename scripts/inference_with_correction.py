#!/usr/bin/env python
"""
在线推理脚本 - 使用训练好的 CorrectionPolicy 进行轨迹修正

功能：
    1. 加载训练好的 CorrectionPolicy 模型
    2. 加载 UpdateEvaluator（用于 LearnedUpdateGate）
    3. 构建三层防御体系（SafetyGuard + STAPOGate + LearnedUpdateGate）
    4. 对输入的 PlanningInterface 进行在线修正

用法：
    # 单样本推理
    python scripts/inference_with_correction.py \
        --checkpoint experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth \
        --evaluator experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth \
        --data_dir data/vad_dumps \
        --scene_token "scene_001"

    # 批量推理
    python scripts/inference_with_correction.py \
        --checkpoint experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth \
        --evaluator experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth \
        --data_dir data/vad_dumps \
        --batch_size 8 \
        --output_dir outputs/inference_results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# 添加项目根目录
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from E2E_RL.correction_policy.policy import CorrectionPolicy
from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.update_selector.safety_guard import SafetyGuard, SafetyGuardConfig
from E2E_RL.update_selector.stapo_gate import STAPOGate, STAPOGateConfig
from E2E_RL.update_selector.update_evaluator import (
    UpdateEvaluator,
    UpdateEvaluatorConfig,
    LearnedUpdateGate,
)
from E2E_RL.data.dataloader import build_planner_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# 防御体系配置（与训练时保持一致）
# =============================================================================
SAFETY_GUARD_CONFIG = SafetyGuardConfig(
    enabled=True,
    max_residual_norm=5.0,
    max_step_disp=2.0,
    max_speed=15.0,
    max_total_disp=10.0,
    dt=0.5,
)

STAPO_GATE_CONFIG = STAPOGateConfig(
    enabled=True,
    advantage_threshold=0.0,
    probability_threshold=0.05,
    entropy_threshold=0.3,
    min_retention_ratio=0.5,
    use_combined_threshold=True,
)

LEARNED_GATE_CONFIG = {
    'tau_gain': 0.0,
    'tau_risk': 0.3,
    'advantage_threshold': 0.0,
}


# =============================================================================
# 模型加载
# =============================================================================

def load_policy(
    checkpoint_path: str,
    device: torch.device,
) -> CorrectionPolicy:
    """加载训练好的 CorrectionPolicy"""
    logger.info(f"加载 Policy: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从 checkpoint 推断模型配置
    state_dict = ckpt['policy_state_dict'] if 'policy_state_dict' in ckpt else ckpt

    # 从权重形状推断维度
    scene_dim = state_dict['actor.scene_proj.weight'].shape[1]
    hidden_dim = state_dict['actor.scene_proj.weight'].shape[0]
    plan_len = state_dict['actor.plan_proj.weight'].shape[1] // 2

    policy = CorrectionPolicy(
        scene_dim=scene_dim,
        plan_len=plan_len,
        hidden_dim=hidden_dim,
    )
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    logger.info(f"  Scene dim: {scene_dim}, Plan len: {plan_len}, Hidden dim: {hidden_dim}")
    logger.info(f"  参数量: {sum(p.numel() for p in policy.parameters()):,}")

    return policy


def load_evaluator(
    checkpoint_path: str,
    device: torch.device,
) -> UpdateEvaluator:
    """加载 UpdateEvaluator（用于 LearnedUpdateGate）"""
    logger.info(f"加载 Evaluator: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt.get('evaluator_state_dict', ckpt)

    # 从权重推断维度 - Evaluator 使用 scene_encoder, ref_encoder
    # 尝试多种可能的 key 名称
    if 'scene_encoder.0.weight' in state_dict:
        scene_dim = state_dict['scene_encoder.0.weight'].shape[1]
        hidden_dim = state_dict['scene_encoder.0.weight'].shape[0]
        # ref_encoder.0.weight shape [256, plan_len*2]
        plan_len = state_dict['ref_encoder.0.weight'].shape[1] // 2
    elif 'scene_proj.weight' in state_dict:
        scene_dim = state_dict['scene_proj.weight'].shape[1]
        hidden_dim = state_dict['scene_proj.weight'].shape[0]
        plan_len = state_dict['plan_proj.weight'].shape[1] // 2
    else:
        # 遍历找第一个匹配
        for k, v in state_dict.items():
            if k.endswith('.weight') and v.dim() == 2 and v.shape[0] == v.shape[1]:
                hidden_dim = v.shape[0]
                scene_dim = v.shape[1]
                break
        plan_len = 6  # 默认值

    evaluator = UpdateEvaluator(
        UpdateEvaluatorConfig(
            scene_dim=scene_dim,
            plan_len=plan_len,
            hidden_dim=hidden_dim,
        )
    )
    evaluator.load_state_dict(state_dict)
    evaluator.to(device)
    evaluator.eval()

    logger.info(f"  Scene dim: {scene_dim}, Plan len: {plan_len}, Hidden dim: {hidden_dim}")

    return evaluator


def build_defense_system(
    evaluator: Optional[UpdateEvaluator],
    device: torch.device,
) -> tuple:
    """构建三层防御体系"""
    # SafetyGuard: 硬底线
    safety_guard = SafetyGuard(SAFETY_GUARD_CONFIG)
    logger.info(f"SafetyGuard: {'启用' if safety_guard.cfg.enabled else '禁用'}")

    # STAPOGate: 弱兜底
    stapo_gate = STAPOGate(STAPO_GATE_CONFIG)
    logger.info(f"STAPOGate: {'启用' if stapo_gate.cfg.enabled else '禁用'}")

    # LearnedUpdateGate: 主判断
    learned_gate = None
    if evaluator is not None:
        learned_gate = LearnedUpdateGate(
            evaluator=evaluator,
            **LEARNED_GATE_CONFIG,
        )
        logger.info("LearnedUpdateGate: 启用（主判断）")
    else:
        logger.warning("LearnedUpdateGate: 禁用（未提供 Evaluator）")

    return safety_guard, stapo_gate, learned_gate


# =============================================================================
# 核心推理逻辑
# =============================================================================

class CorrectionInference:
    """在线修正推理器"""

    def __init__(
        self,
        policy: CorrectionPolicy,
        safety_guard: SafetyGuard,
        stapo_gate: STAPOGate,
        learned_gate: Optional[LearnedUpdateGate],
        device: torch.device,
    ):
        self.policy = policy
        self.safety_guard = safety_guard
        self.stapo_gate = stapo_gate
        self.learned_gate = learned_gate
        self.device = device

    def inference_single(
        self,
        interface: PlanningInterface,
        verbose: bool = True,
    ) -> Dict:
        """单样本推理

        Args:
            interface: PlanningInterface 输入
            verbose: 是否打印详细信息

        Returns:
            dict: {
                'reference_plan': 原始参考轨迹 [T, 2],
                'corrected_plan': 修正后轨迹 [T, 2],
                'correction': 修正量 [T, 2],
                'updated': 是否进行了更新,
                'update_stage': 更新阶段（None/refused/accepted）,
                'stats': 策略统计信息,
            }
        """
        interface = interface.to(self.device)

        # 1. Policy 前向传播
        correction = self.policy.act(interface)  # [1, T, 2]

        # 2. 尝试修正
        ref_plan = interface.reference_plan  # [1, T, 2]
        candidate_plan = ref_plan + correction

        # 3. 三层防御检查
        update_decision, update_info = self._defense_check(
            ref_plan, candidate_plan, correction, interface
        )

        # 4. 获取最终轨迹
        if update_decision:
            final_plan = candidate_plan
        else:
            final_plan = ref_plan

        # 5. 计算统计信息
        stats = self.policy.get_statistics(interface)

        result = {
            'reference_plan': ref_plan[0].detach().cpu().numpy(),
            'corrected_plan': final_plan[0].detach().cpu().numpy(),
            'correction': correction[0].detach().cpu().numpy(),
            'updated': update_decision,
            'update_stage': update_info.get('stage', 'none'),
            'stats': stats,
        }

        if verbose:
            self._print_result(result, update_info)

        return result

    def _defense_check(
        self,
        ref_plan: torch.Tensor,
        candidate_plan: torch.Tensor,
        correction: torch.Tensor,
        interface: PlanningInterface,
    ) -> tuple:
        """三层防御检查"""
        update_info = {'stage': 'refused', 'reason': '多层防御拒绝'}

        # Stage 1: SafetyGuard 硬底线
        # SafetyGuard.check(correction, reference_plan, corrected_plan)
        safety_ok_tensor = self.safety_guard.check(
            correction, ref_plan, candidate_plan
        )
        safety_ok = safety_ok_tensor.item()  # 转为 Python bool
        if not safety_ok:
            update_info['stage'] = 'safety_guard'
            update_info['reason'] = 'SafetyGuard 物理约束检查失败'
            return False, update_info

        # Stage 2: LearnedUpdateGate 主判断
        if self.learned_gate is not None:
            pred_result = self.learned_gate.predict(interface, correction)
            if pred_result['is_harmful'].item():
                update_info['stage'] = 'learned_gate'
                update_info['reason'] = (
                    f"LearnedGate 判断有害: "
                    f"gain={pred_result['pred_gain'].item():.3f}, "
                    f"risk={pred_result['pred_risk'].item():.3f}"
                )
                return False, update_info

        # 全部通过
        update_info['stage'] = 'accepted'
        update_info['reason'] = '所有防御层通过'
        return True, update_info

    def inference_batch(
        self,
        interfaces: List[PlanningInterface],
    ) -> List[Dict]:
        """批量推理"""
        results = []
        for interface in interfaces:
            result = self.inference_single(interface, verbose=False)
            results.append(result)
        return results

    def _print_result(self, result: Dict, update_info: Dict):
        """打印推理结果"""
        print("\n" + "=" * 60)
        print("推理结果")
        print("=" * 60)

        print(f"更新决策: {'✓ 接受修正' if result['updated'] else '✗ 拒绝修正'}")
        print(f"拒绝阶段: {update_info.get('stage', 'none')}")
        print(f"拒绝原因: {update_info.get('reason', 'N/A')}")

        print(f"\n参考轨迹 (首尾各3点):")
        ref = result['reference_plan']
        print(f"  前3点: {ref[:3].round(3)}")
        print(f"  后3点: {ref[-3:].round(3)}")

        if result['updated']:
            print(f"\n修正轨迹 (首尾各3点):")
            corr = result['corrected_plan']
            print(f"  前3点: {corr[:3].round(3)}")
            print(f"  后3点: {corr[-3:].round(3)}")

            print(f"\n修正量 (统计):")
            correction = result['correction']
            print(f"  均值: {correction.mean(axis=0).round(3)}")
            print(f"  标准差: {correction.std(axis=0).round(3)}")
            print(f"  最大偏移: {correction.max():.3f}m")

        print(f"\n策略统计: {result['stats']}")
        print("=" * 60)


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='在线推理 - CorrectionPolicy')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='CorrectionPolicy checkpoint 路径',
    )
    parser.add_argument(
        '--evaluator',
        type=str,
        default=None,
        help='UpdateEvaluator checkpoint 路径（可选，用于 LearnedUpdateGate）',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='数据目录（包含 dump 数据）',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='批大小',
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最多处理样本数（None=全部）',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（None=不保存）',
    )
    parser.add_argument(
        '--scene_token',
        type=str,
        default=None,
        help='指定场景 token（None=使用第一个）',
    )
    parser.add_argument(
        '--disable_learned_gate',
        action='store_true',
        help='禁用 LearnedUpdateGate',
    )
    parser.add_argument(
        '--disable_stapo_gate',
        action='store_true',
        help='禁用 STAPOGate',
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载模型
    policy = load_policy(args.checkpoint, device)

    evaluator = None
    if not args.disable_learned_gate and args.evaluator:
        evaluator = load_evaluator(args.evaluator, device)

    # 构建防御系统
    safety_guard, stapo_gate, learned_gate = build_defense_system(evaluator, device)

    if args.disable_stapo_gate:
        stapo_gate = STAPOGate(STAPOGateConfig(enabled=False))
        logger.info("STAPOGate: 已禁用")

    if args.disable_learned_gate:
        learned_gate = None
        logger.info("LearnedUpdateGate: 已禁用")

    # 创建推理器
    inference = CorrectionInference(
        policy=policy,
        safety_guard=safety_guard,
        stapo_gate=stapo_gate,
        learned_gate=learned_gate,
        device=device,
    )

    # 加载数据
    logger.info(f"加载数据: {args.data_dir}")
    dataloader = build_planner_dataloader(
        data_dir=args.data_dir,
        adapter_type='vad',
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )
    logger.info(f"数据集大小: {len(dataloader.dataset)}")

    # 收集结果
    all_results = []
    update_count = 0
    refused_reasons = {}

    for batch_idx, batch in enumerate(dataloader):
        if args.max_samples and batch_idx >= args.max_samples:
            break

        interface = batch['interface']
        scene_token = interface.scene_token[0]

        # 打印进度
        logger.info(f"[{batch_idx + 1}/{len(dataloader)}] 场景: {scene_token}")

        # 单样本推理
        result = inference.inference_single(interface, verbose=True)

        all_results.append({
            'scene_token': scene_token,
            **result,
        })

        if result['updated']:
            update_count += 1
        else:
            reason = result['update_stage']
            refused_reasons[reason] = refused_reasons.get(reason, 0) + 1

    # 统计结果
    total = len(all_results)
    logger.info("\n" + "=" * 60)
    logger.info("批量推理统计")
    logger.info("=" * 60)
    logger.info(f"总样本数: {total}")
    logger.info(f"接受修正: {update_count} ({100 * update_count / total:.1f}%)")
    logger.info(f"拒绝修正: {total - update_count} ({100 * (total - update_count) / total:.1f}%)")

    if refused_reasons:
        logger.info("\n拒绝原因分布:")
        for reason, count in refused_reasons.items():
            logger.info(f"  {reason}: {count} ({100 * count / total:.1f}%)")

    # 保存结果
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存为 JSON
        results_for_json = []
        for r in all_results:
            results_for_json.append({
                'scene_token': r['scene_token'],
                'updated': r['updated'],
                'update_stage': r['update_stage'],
                'reference_plan': r['reference_plan'].tolist(),
                'corrected_plan': r['corrected_plan'].tolist(),
            })

        with open(output_path / 'inference_results.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)

        logger.info(f"结果已保存到: {output_path / 'inference_results.json'}")

    logger.info("\n推理完成！")
    return all_results


if __name__ == '__main__':
    main()
