"""Hard case mining 模块：基于规则的困难场景识别和采样。

困难场景代理信号:
- 高规划误差 (ADE/FDE)
- 高碰撞风险
- 高不确定性（模式方差大）
- 长尾场景标签（如有）
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from E2E_RL.planning_interface.interface import PlanningInterface


class HardCaseMiner:
    """基于规则的困难场景挖掘器。

    根据多种信号对样本进行 hard-case 评分，
    支持子集筛选和过采样。

    Args:
        error_weight: 规划误差信号权重
        uncertainty_weight: 不确定性信号权重
        collision_weight: 碰撞风险信号权重
        top_ratio: 被标记为 hard case 的样本比例
    """

    def __init__(
        self,
        error_weight: float = 1.0,
        uncertainty_weight: float = 0.5,
        collision_weight: float = 0.5,
        top_ratio: float = 0.2,
    ):
        self.error_weight = error_weight
        self.uncertainty_weight = uncertainty_weight
        self.collision_weight = collision_weight
        self.top_ratio = top_ratio

    def score_batch(
        self,
        interface: PlanningInterface,
        gt_plan: torch.Tensor,
        refined_plan: Optional[torch.Tensor] = None,
        reward_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """计算 batch 中每个样本的 hard-case 分数。

        分数越高表示越困难。

        Args:
            interface: 统一规划接口
            gt_plan: [B, T, 2] GT 轨迹（绝对坐标）
            refined_plan: [B, T, 2] 精炼后轨迹（可选，用于计算误差）
            reward_info: 奖励信息字典（含 collision_penalty 等）

        Returns:
            [B] 的 hard-case 分数
        """
        device = gt_plan.device
        batch_size = gt_plan.shape[0]
        scores = torch.zeros(batch_size, device=device)

        # 1. 规划误差信号
        plan = refined_plan if refined_plan is not None else interface.reference_plan
        error = torch.norm(plan - gt_plan, dim=-1).mean(dim=-1)  # [B] ADE
        scores = scores + self.error_weight * error

        # 2. 不确定性信号
        if interface.plan_confidence is not None:
            confidence = interface.plan_confidence
            if confidence.dim() == 2:
                confidence = confidence.mean(dim=-1)  # [B]
            # 置信度越低 → 越困难
            uncertainty = 1.0 - confidence.clamp(0, 1)
            scores = scores + self.uncertainty_weight * uncertainty

        # 3. 碰撞风险信号
        if reward_info is not None and 'collision_penalty' in reward_info:
            col = reward_info['collision_penalty']  # [B]
            scores = scores + self.collision_weight * col

        return scores

    def select_hard_cases(
        self,
        scores: torch.Tensor,
        top_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """选择 hard case 的索引。

        Args:
            scores: [N] 样本分数
            top_ratio: 选取比例，默认使用初始化时的 top_ratio

        Returns:
            被选中样本的索引 tensor
        """
        ratio = top_ratio if top_ratio is not None else self.top_ratio
        k = max(1, int(scores.shape[0] * ratio))
        _, indices = torch.topk(scores, k, largest=True)
        return indices.sort().values

    def get_oversampling_weights(
        self,
        scores: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """计算用于过采样的权重。

        分数越高的样本被采样的概率越大。

        Args:
            scores: [N] 样本分数
            temperature: 温度参数，越低则分布越尖锐

        Returns:
            [N] 的采样权重（已归一化）
        """
        logits = scores / max(temperature, 1e-6)
        weights = torch.softmax(logits, dim=0)
        return weights

    def build_hard_subset_indices(
        self,
        all_scores: List[torch.Tensor],
        top_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """从多个 batch 的分数中构建 hard case 子集索引。

        Args:
            all_scores: 每个 batch 的分数列表
            top_ratio: 选取比例

        Returns:
            全局偏移后的 hard case 索引
        """
        ratio = top_ratio if top_ratio is not None else self.top_ratio
        concatenated = torch.cat(all_scores, dim=0)
        k = max(1, int(concatenated.shape[0] * ratio))
        _, indices = torch.topk(concatenated, k, largest=True)
        return indices.sort().values
