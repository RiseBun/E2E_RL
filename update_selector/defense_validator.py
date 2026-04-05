"""
DefenseLayerValidator — 三层防御效果验证器

4 层验证框架：

层 1: 标签层验证
- Spearman / Kendall tau / Top-k hit rate
- 分项 risk 验证（collision/offroad/comfort/drift）

层 2: 分布层验证
- retained vs filtered 在 gain/risk/entropy/prob 上的分布差异
- 检测 spurious 模式

层 3: 反事实训练验证（A/B/C/D 对照实验）
- A: 不过滤
- B: SafetyGuard + STAPOGate
- C: SafetyGuard + LearnedUpdateGate
- D: SafetyGuard + STAPOGate + LearnedUpdateGate

层 4: 最终策略验证
- ADE/FDE/collision/offroad 等最终指标

核心原则：
不是验证"谁分高"，而是验证"谁训练后更强"。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """验证配置。"""
    # 层 1: 标签层
    n_batches_label: int = 100

    # 层 2: 分布层
    n_batches_dist: int = 50

    # 层 3: 训练对照
    ablation_configs: List[Dict] = field(default_factory=lambda: [
        {'name': 'A_no_filter', 'filter': 'none'},
        {'name': 'B_stapo', 'filter': 'stapo'},
        {'name': 'C_learned', 'filter': 'learned'},
        {'name': 'D_cascade', 'filter': 'cascade'},
    ])
    short_train_epochs: int = 5
    long_train_epochs: int = 20


class DefenseLayerValidator:
    """三层防御效果验证器。

    实现 4 层验证框架，核心原则：
    不是验证"谁分高"，而是验证"谁训练后更强"。
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        device: torch.device = torch.device('cpu'),
    ):
        self.cfg = config or ValidationConfig()
        self.device = device

    # ===================================================================
    # 层 1: 标签层验证
    # ===================================================================

    def validate_layer1_label_quality(
        self,
        dataloader: DataLoader,
        evaluator,  # UpdateEvaluator
        n_batches: int = 100,
    ) -> Dict[str, Any]:
        """验证标签层排序能力。

        核心指标：
        - Spearman / Kendall tau（gain）
        - Top-k hit rate（gain）
        - 分项 risk 排序（collision/offroad/comfort/drift）
        """
        result = {
            'spearman_gain': 0.0,
            'kendall_gain': 0.0,
            'topk_hit_rates': {},
            'spearman_collision': 0.0,
            'spearman_offroad': 0.0,
            'spearman_comfort': 0.0,
            'spearman_drift': 0.0,
            'spearman_total_risk': 0.0,
            'overall_score': 0.0,
        }

        all_pred_gain, all_true_gain = [], []
        all_pred_collision, all_true_collision = [], []
        all_pred_offroad, all_true_offroad = [], []
        all_pred_comfort, all_true_comfort = [], []
        all_pred_drift, all_true_drift = [], []

        evaluator.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break

                # 准备输入
                scene_token = batch['scene_token'].to(self.device)
                reference_plan = batch['reference_plan'].to(self.device)
                correction = batch['correction'].to(self.device)

                # 预测
                output = evaluator(
                    scene_token=scene_token,
                    reference_plan=reference_plan,
                    correction=correction,
                )

                pred_gain = output['pred_gain'].cpu().squeeze(-1)
                all_pred_gain.append(pred_gain)
                all_true_gain.append(batch['gain'].float())

                # Risk 分项
                all_pred_collision.append(output.get('pred_collision', torch.zeros_like(pred_gain)).cpu().squeeze(-1))
                all_true_collision.append(batch['collision_delta'].float())

                all_pred_offroad.append(output.get('pred_offroad', torch.zeros_like(pred_gain)).cpu().squeeze(-1))
                all_true_offroad.append(batch['offroad_delta'].float())

                all_pred_comfort.append(output.get('pred_comfort', torch.zeros_like(pred_gain)).cpu().squeeze(-1))
                all_true_comfort.append(batch['comfort_delta'].float())

                all_pred_drift.append(output.get('pred_drift', torch.zeros_like(pred_gain)).cpu().squeeze(-1))
                all_true_drift.append(batch['drift'].float())

        # 合并
        all_pred_gain = torch.cat(all_pred_gain)
        all_true_gain = torch.cat(all_true_gain)
        all_pred_collision = torch.cat(all_pred_collision)
        all_true_collision = torch.cat(all_true_collision)
        all_pred_offroad = torch.cat(all_pred_offroad)
        all_true_offroad = torch.cat(all_true_offroad)
        all_pred_comfort = torch.cat(all_pred_comfort)
        all_true_comfort = torch.cat(all_true_comfort)
        all_pred_drift = torch.cat(all_pred_drift)
        all_true_drift = torch.cat(all_true_drift)

        # 统一移到 CPU 进行计算
        all_pred_gain = all_pred_gain.cpu()
        all_true_gain = all_true_gain.cpu()
        all_pred_collision = all_pred_collision.cpu()
        all_true_collision = all_true_collision.cpu()
        all_pred_offroad = all_pred_offroad.cpu()
        all_true_offroad = all_true_offroad.cpu()
        all_pred_comfort = all_pred_comfort.cpu()
        all_true_comfort = all_true_comfort.cpu()
        all_pred_drift = all_pred_drift.cpu()
        all_true_drift = all_true_drift.cpu()

        # Gain 排序
        result['spearman_gain'] = self._spearman(all_pred_gain, all_true_gain)
        result['kendall_gain'] = self._kendall(all_pred_gain, all_true_gain)

        for k in [1, 3, 5]:
            result['topk_hit_rates'][k] = self._topk_hit_rate(all_pred_gain, all_true_gain, k)

        # Risk 排序
        result['spearman_collision'] = self._spearman(all_pred_collision, all_true_collision)
        result['spearman_offroad'] = self._spearman(all_pred_offroad, all_true_offroad)
        result['spearman_comfort'] = self._spearman(all_pred_comfort, all_true_comfort)
        result['spearman_drift'] = self._spearman(all_pred_drift, all_true_drift)

        # Total risk
        all_true_total_risk = all_true_collision + all_true_offroad + all_true_comfort + all_true_drift
        all_pred_total_risk = all_pred_collision + all_pred_offroad + all_pred_comfort + all_pred_drift
        result['spearman_total_risk'] = self._spearman(all_pred_total_risk, all_true_total_risk)

        # 综合评分
        result['overall_score'] = (
            result['spearman_gain'] * 0.4 +
            result['spearman_total_risk'] * 0.3 +
            result['kendall_gain'] * 0.3
        )

        logger.info(f"[Layer 1] Gain: spearman={result['spearman_gain']:.3f}, kendall={result['kendall_gain']:.3f}")
        logger.info(f"[Layer 1] Risk: total={result['spearman_total_risk']:.3f}")
        logger.info(f"[Layer 1] Top-k hit rates: {result['topk_hit_rates']}")
        logger.info(f"[Layer 1] Overall score: {result['overall_score']:.3f}")

        return result

    # ===================================================================
    # 层 2: 分布层验证
    # ===================================================================

    def validate_layer2_distribution(
        self,
        dataloader: DataLoader,
        gate_fn: Callable,
        n_batches: int = 50,
    ) -> Dict[str, Any]:
        """验证分布层差异。

        比较 retained vs filtered 在：
        - gain/risk 分布
        - spurious 模式检测
        """
        result = {}

        retained_gains, filtered_gains = [], []
        retained_risks, filtered_risks = [], []
        retained_samples, filtered_samples = [], []

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            advantages = batch['gain'].float()
            mask, diag = gate_fn(batch)

            for j in range(len(advantages)):
                if mask[j] if hasattr(mask, '__getitem__') else mask:
                    retained_gains.append(advantages[j].item())
                    retained_samples.append({
                        'gain': advantages[j].item(),
                        'log_prob': 0,
                        'entropy': 1,
                    })
                else:
                    filtered_gains.append(advantages[j].item())
                    filtered_samples.append({
                        'gain': advantages[j].item(),
                        'log_prob': 0,
                        'entropy': 1,
                    })

        # Gain 分布
        result['n_retained'] = len(retained_gains)
        result['n_filtered'] = len(filtered_gains)
        result['retained_gain_mean'] = np.mean(retained_gains) if retained_gains else 0
        result['retained_gain_std'] = np.std(retained_gains) if retained_gains else 0
        result['filtered_gain_mean'] = np.mean(filtered_gains) if filtered_gains else 0
        result['filtered_gain_std'] = np.std(filtered_gains) if filtered_gains else 0
        result['gain_difference'] = result['retained_gain_mean'] - result['filtered_gain_mean']

        # 分布重叠度
        result['gain_overlap_ratio'] = self._compute_overlap(retained_gains, filtered_gains)

        # Spurious 模式检测
        retained_spurious = sum(
            1 for s in retained_samples
            if s['gain'] > 0 and s['log_prob'] < -2.3 and s['entropy'] < 0.5
        )
        filtered_spurious = sum(
            1 for s in filtered_samples
            if s['gain'] > 0 and s['log_prob'] < -2.3 and s['entropy'] < 0.5
        )

        result['retained_spurious_ratio'] = retained_spurious / len(retained_samples) if retained_samples else 0
        result['filtered_spurious_ratio'] = filtered_spurious / len(filtered_samples) if filtered_samples else 0

        # 判断过滤质量
        reasons = []
        if result['filtered_spurious_ratio'] > result['retained_spurious_ratio']:
            result['spurious_pattern_quality'] = 'good'
            reasons.append("过滤样本中 spurious 比例更高")
        else:
            result['spurious_pattern_quality'] = 'concerning'
            reasons.append("过滤样本中 spurious 比例未更高")

        if result['gain_difference'] > 0:
            reasons.append(f"保留样本 gain 平均高 {result['gain_difference']:.3f}")
        else:
            reasons.append(f"保留样本 gain 平均低 {-result['gain_difference']:.3f}")

        result['filter_quality'] = 'good' if (
            result['spurious_pattern_quality'] == 'good' and
            result['filtered_spurious_ratio'] > 0.1
        ) else 'concerning'

        result['quality_reasons'] = reasons

        logger.info(f"[Layer 2] Gain diff: retained={result['retained_gain_mean']:.3f}, filtered={result['filtered_gain_mean']:.3f}")
        logger.info(f"[Layer 2] Spurious: retained={result['retained_spurious_ratio']:.2%}, filtered={result['filtered_spurious_ratio']:.2%}")
        logger.info(f"[Layer 2] Filter quality: {result['filter_quality']}")

        return result

    # ===================================================================
    # 层 3: 反事实训练验证
    # ===================================================================

    def validate_layer3_ablation(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        configs: Optional[List[Dict]] = None,
        short_epochs: int = 5,
        long_epochs: int = 20,
    ) -> Dict[str, Any]:
        """执行 A/B/C/D 对照实验。

        Args:
            train_fn: 训练函数，接受 (config, epochs) 返回训练结果
            eval_fn: 评估函数，接受训练结果返回最终指标
            configs: 配置列表
            short_epochs: 短程训练 epoch 数
            long_epochs: 长程训练 epoch 数
        """
        configs = configs or self.cfg.ablation_configs
        result = {'configs_results': {}, 'comparison_summary': {}}
        all_metrics = {}

        for cfg in configs:
            name = cfg['name']
            logger.info(f"[Layer 3] Running {name}...")

            # 短程训练
            short_metrics = train_fn(cfg, short_epochs)

            # 长程训练
            long_metrics = train_fn(cfg, long_epochs)

            # 最终评估
            final_metrics = eval_fn(long_metrics)

            all_metrics[name] = {
                'short': short_metrics,
                'long': long_metrics,
                'final': final_metrics,
                'config': cfg,
            }

            logger.info(f"[Layer 3] {name} final: {final_metrics}")

        result['configs_results'] = all_metrics

        # 对比总结
        result['comparison_summary'] = self._summarize_ablation(all_metrics)

        return result

    def _summarize_ablation(self, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """总结对照实验结果。"""
        summary = {}

        for name, data in metrics.items():
            final = data.get('final', {})
            short = data.get('short', {})

            summary[name] = {
                'final_ade': final.get('ade', 0),
                'final_collision': final.get('collision', 0),
                'training_stability': short.get('entropy_std', 0),
                'final_performance': final.get('overall_score', 0),
            }

        if not summary:
            return {}

        best_by_ade = min(summary.items(), key=lambda x: x[1]['final_ade'])
        best_by_collision = min(summary.items(), key=lambda x: x[1]['final_collision'])
        best_by_stability = min(summary.items(), key=lambda x: x[1]['training_stability'])

        return {
            'best_by_ade': {'name': best_by_ade[0], 'value': best_by_ade[1]['final_ade']},
            'best_by_collision': {'name': best_by_collision[0], 'value': best_by_collision[1]['final_collision']},
            'best_by_stability': {'name': best_by_stability[0], 'value': best_by_stability[1]['training_stability']},
            'all_metrics': summary,
        }

    # ===================================================================
    # 层 4: 最终策略验证
    # ===================================================================

    def validate_layer4_final_policy(
        self,
        layer3_results: Dict,
    ) -> Dict[str, Any]:
        """验证最终策略性能。"""
        result = {
            'final_metrics': {},
            'improvement_summary': {},
        }

        configs = layer3_results.get('configs_results', {})
        summary = layer3_results.get('comparison_summary', {})

        # 提取最终指标
        for name, data in configs.items():
            result['final_metrics'][name] = data.get('final', {})

        # 改进总结
        if 'D_cascade' in configs and 'A_no_filter' in configs:
            cascade = configs['D_cascade']['final']
            no_filter = configs['A_no_filter']['final']

            result['improvement_summary'] = {
                'ade_improvement': no_filter.get('ade', 0) - cascade.get('ade', 0),
                'collision_improvement': no_filter.get('collision', 0) - cascade.get('collision', 0),
                'offroad_improvement': no_filter.get('offroad', 0) - cascade.get('offroad', 0),
                'overall_improvement': cascade.get('overall_score', 0) - no_filter.get('overall_score', 0),
            }

        return result

    # ===================================================================
    # 完整验证流程
    # ===================================================================

    def run_full_validation(
        self,
        dataloader: DataLoader,
        evaluator,
        gate_fn: Callable,
        train_fn: Callable,
        eval_fn: Callable,
    ) -> Dict[str, Any]:
        """执行完整 4 层验证。"""
        report = {
            'layer1': {},
            'layer2': {},
            'layer3': {},
            'layer4': {},
            'pass_offline': False,
            'pass_online_stability': False,
            'pass_online_performance': False,
            'overall_pass': False,
            'verdict': 'unknown',
            'reasons': [],
        }

        logger.info("=" * 60)
        logger.info("开始 4 层验证")
        logger.info("=" * 60)

        # 层 1
        logger.info("\n[Layer 1] 标签层验证...")
        report['layer1'] = self.validate_layer1_label_quality(dataloader, evaluator)
        report['pass_offline'] = (
            report['layer1']['spearman_gain'] > 0.7 and
            report['layer1']['spearman_total_risk'] > 0.5
        )

        # 层 2
        logger.info("\n[Layer 2] 分布层验证...")
        report['layer2'] = self.validate_layer2_distribution(dataloader, gate_fn)

        # 层 3
        logger.info("\n[Layer 3] 反事实训练验证...")
        report['layer3'] = self.validate_layer3_ablation(train_fn, eval_fn)

        # 检查稳定性
        summary = report['layer3'].get('comparison_summary', {})
        if 'all_metrics' in summary:
            all_metrics = summary['all_metrics']
            if 'D_cascade' in all_metrics and 'A_no_filter' in all_metrics:
                cascade_stability = all_metrics['D_cascade']['training_stability']
                no_filter_stability = all_metrics['A_no_filter']['training_stability']
                report['pass_online_stability'] = cascade_stability <= no_filter_stability

        # 层 4
        logger.info("\n[Layer 4] 最终策略验证...")
        report['layer4'] = self.validate_layer4_final_policy(report['layer3'])

        # 检查性能
        improvement = report['layer4'].get('improvement_summary', {})
        if improvement:
            report['pass_online_performance'] = (
                improvement.get('ade_improvement', 0) > 0 or
                improvement.get('collision_improvement', 0) > 0
            )

        # 综合判定
        report['overall_pass'] = (
            report['pass_offline'] and
            report['pass_online_stability'] and
            report['pass_online_performance']
        )

        if report['overall_pass']:
            report['verdict'] = "PASS"
            report['reasons'].append("离线排序能力达标")
            report['reasons'].append("训练稳定性提升")
            report['reasons'].append("最终策略性能提升")
        else:
            report['verdict'] = "FAIL"
            if not report['pass_offline']:
                report['reasons'].append("离线排序能力不达标")
            if not report['pass_online_stability']:
                report['reasons'].append("训练稳定性未提升")
            if not report['pass_online_performance']:
                report['reasons'].append("最终策略性能未提升")

        logger.info("\n" + "=" * 60)
        logger.info(f"验证结果: {report['verdict']}")
        for reason in report['reasons']:
            logger.info(f"  - {reason}")
        logger.info("=" * 60)

        return report

    def save_report(self, report: Dict, path: str):
        """保存验证报告。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"验证报告已保存到 {path}")

    # ===================================================================
    # 辅助方法
    # ===================================================================

    @staticmethod
    def _spearman(pred: torch.Tensor, true: torch.Tensor) -> float:
        """计算 Spearman 相关系数。"""
        n = len(pred)
        if n < 2:
            return 0.0

        _, indices = torch.sort(pred)
        pred_rank = torch.zeros_like(pred).scatter_(0, indices, torch.arange(n, device=pred.device).float())

        _, indices = torch.sort(true)
        true_rank = torch.zeros_like(true).scatter_(0, indices, torch.arange(n, device=true.device).float())

        d = pred_rank - true_rank
        spearman = 1 - 6 * (d.float() ** 2).sum() / (n ** 3 - n)
        return spearman.item()

    @staticmethod
    def _kendall(pred: torch.Tensor, true: torch.Tensor) -> float:
        """计算 Kendall tau 相关系数。"""
        n = len(pred)
        if n < 2:
            return 0.0

        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                pred_diff = (pred[i] - pred[j]).item()
                true_diff = (true[i] - true[j]).item()

                if pred_diff * true_diff > 0:
                    concordant += 1
                elif pred_diff * true_diff < 0:
                    discordant += 1

        total_pairs = n * (n - 1) / 2
        tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0
        return tau

    @staticmethod
    def _topk_hit_rate(pred: torch.Tensor, true: torch.Tensor, k: int) -> float:
        """计算 Top-k hit rate。"""
        n = len(pred)
        k = min(k, n)

        _, topk_indices = torch.topk(pred, k)
        topk_true = true[topk_indices]

        global_mean = true.mean()
        hit_rate = (topk_true > global_mean).float().mean().item()
        return hit_rate

    @staticmethod
    def _compute_overlap(list1: List[float], list2: List[float]) -> float:
        """计算两个分布的重叠比例。"""
        if not list1 or not list2:
            return 0.0

        min_val = min(min(list1), min(list2))
        max_val = max(max(list1), max(list2))

        bins = np.linspace(min_val, max_val, 20)
        hist1, _ = np.histogram(list1, bins=bins)
        hist2, _ = np.histogram(list2, bins=bins)

        hist1 = hist1 / max(hist1.sum(), 1)
        hist2 = hist2 / max(hist2.sum(), 1)

        overlap = np.sum(np.minimum(hist1, hist2))
        return float(overlap)


# =============================================================================
# 便捷函数
# =============================================================================

def create_validation_pipeline(
    evaluator,
    stapo_gate,
    learned_gate,
    safety_guard,
    device: torch.device = torch.device('cpu'),
) -> Tuple:
    """创建验证 pipeline。"""
    validator = DefenseLayerValidator(device)

    def gate_fn(batch):
        """组合的门控函数。"""
        advantages = batch['gain'].float()
        B = len(advantages)

        # 1. Safety Guard
        safety_mask = torch.ones(B, dtype=torch.bool, device=advantages.device)

        # 2. STAPO Gate
        stapo_mask = stapo_gate.compute_mask(
            advantages=advantages,
            action_log_probs=torch.zeros_like(advantages),
        )

        # 3. Learned Gate
        if learned_gate is not None:
            scene_token = batch['scene_token'].to(device)
            reference_plan = batch['reference_plan'].to(device)
            correction = batch['correction'].to(device)

            learned_mask, diag = learned_gate.compute_mask(
                advantages=advantages,
                interface=type('Interface', (), {
                    'scene_token': scene_token,
                    'reference_plan': reference_plan,
                })(),
                correction=correction,
                safety_mask=safety_mask & stapo_mask,
            )
            final_mask = safety_mask & stapo_mask & learned_mask
        else:
            final_mask = safety_mask & stapo_mask
            diag = {'retention_ratio': final_mask.float().mean().item()}

        return final_mask, diag

    return validator, gate_fn


def run_quick_validation(
    dataloader: DataLoader,
    evaluator,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """快速验证：只跑层 1（离线排序验证）。

    适用于快速检查训练好的 UpdateEvaluator 是否合格。
    """
    validator = DefenseLayerValidator(device)

    result = validator.validate_layer1_label_quality(dataloader, evaluator)

    # 简单判定
    pass_offline = result['spearman_gain'] > 0.7 and result['spearman_total_risk'] > 0.5

    result['pass_offline'] = pass_offline
    result['verdict'] = "PASS" if pass_offline else "FAIL"

    if pass_offline:
        logger.info("✓ 离线排序验证通过")
    else:
        logger.warning("✗ 离线排序验证未通过")

    return result
