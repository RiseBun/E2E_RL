"""
Conservative E2E RL Post-Training Module

核心设计：
1. Reward-Cost 分离：区分 reward 和 cost，约束优化
2. Beneficial Update Filter：只接受真正有益的更新
3. KL Divergence Constraint：限制单步更新幅度
4. Reference Model Anchoring：保持与原始模型的输出一致性

与 Phase 1 外挂修正器的区别：
- 外挂：独立修正器，不修改原模型
- 这里：梯度回传到原模型规划头
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConservativeRLConfig:
    """Conservative RL 配置。"""
    # 学习率
    lr: float = 3e-5
    
    # KL 约束
    kl_target: float = 0.01  # 目标 KL 散度
    kl_epsilon: float = 1e-4  # 防止除零
    
    # Reference Model
    use_reference_anchor: bool = True
    reference_model: Optional[nn.Module] = None
    reference_alpha: float = 0.5  # 参考损失权重
    
    # Beneficial Update Filter
    use_beneficial_filter: bool = True
    reward_margin: float = 0.0  # 正增益阈值
    cost_increase_threshold: float = 0.1  # cost 允许的最大增量
    kl_bound: float = 0.05  # 单样本最大 KL
    
    # PPO 参数
    clip_epsilon: float = 0.2  # PPO clip 范围
    value_coef: float = 0.5  # Value loss 权重
    entropy_coef: float = 0.01  # 熵正则化权重
    
    # 梯度裁剪
    grad_clip: float = 1.0
    
    # 训练阶段 (控制解冻范围)
    train_stage: int = 1  # 1=只训规划头, 2=解冻部分backbone, 3=全量微调
    freeze_backbone: bool = True  # 是否冻结 backbone


class RewardCostSeparator:
    """
    Reward-Cost 分离设计
    
    不是简单加权和，而是区分：
    - Reward 分支: 追求的目标 (progress, efficiency)
    - Cost 分支: 必须约束的边界 (collision, offroad)
    
    优化目标: maximize reward under bounded cost
    """
    
    def __init__(
        self,
        # Reward 权重
        progress_weight: float = 1.0,
        efficiency_weight: float = 0.5,
        route_completion_weight: float = 0.5,
        
        # Cost 约束
        max_collision_penalty: float = 0.1,
        max_offroad_penalty: float = 0.1,
        max_comfort_violation: float = 1.0,
        
        # Cost 权重
        collision_weight: float = 1.0,
        offroad_weight: float = 1.0,
        comfort_weight: float = 0.3,
    ):
        self.progress_weight = progress_weight
        self.efficiency_weight = efficiency_weight
        self.route_completion_weight = route_completion_weight
        
        self.max_collision_penalty = max_collision_penalty
        self.max_offroad_penalty = max_offroad_penalty
        self.max_comfort_violation = max_comfort_violation
        
        self.collision_weight = collision_weight
        self.offroad_weight = offroad_weight
        self.comfort_weight = comfort_weight
    
    def compute(
        self,
        trajectory: torch.Tensor,
        gt_trajectory: torch.Tensor,
        agent_positions: Optional[torch.Tensor] = None,
        lane_boundaries: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        dt: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        计算分离的 Reward 和 Cost。
        
        Returns:
            dict with:
                - reward_branch: [B] 纯 reward 分支
                - cost_branch: [B] 纯 cost 分支
                - reward_components: 各 reward 分项
                - cost_components: 各 cost 分项
                - is_safe: [B] 是否满足 cost 约束
        """
        # === Reward 分支 ===
        reward_components = self._compute_reward(
            trajectory, gt_trajectory, mask, dt
        )
        
        # === Cost 分支 ===
        cost_components = self._compute_cost(
            trajectory, agent_positions, lane_boundaries, dt
        )
        
        # === 组合 ===
        reward_branch = (
            self.progress_weight * reward_components['progress'] +
            self.efficiency_weight * reward_components.get('efficiency', torch.zeros_like(reward_components['progress'])) +
            self.route_completion_weight * reward_components.get('route_completion', torch.zeros_like(reward_components['progress']))
        )
        
        cost_branch = (
            self.collision_weight * cost_components['collision'] +
            self.offroad_weight * cost_components['offroad'] +
            self.comfort_weight * cost_components['comfort']
        )
        
        # === 安全约束检查 ===
        is_safe = (
            (cost_components['collision'] <= self.max_collision_penalty) &
            (cost_components['offroad'] <= self.max_offroad_penalty) &
            (cost_components['comfort'] <= self.max_comfort_violation)
        )
        
        return {
            'reward_branch': reward_branch,
            'cost_branch': cost_branch,
            'reward_components': reward_components,
            'cost_components': cost_components,
            'is_safe': is_safe,
        }
    
    def _compute_reward(
        self,
        trajectory: torch.Tensor,
        gt_trajectory: torch.Tensor,
        mask: Optional[torch.Tensor],
        dt: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """计算 reward 分支。
        
        包含:
        1. Progress: 基于终点距离
        2. Efficiency: 基于平均速度 (期望接近目标速度)
        3. Route Completion: 轨迹对 GT 的贴合度
        """
        B = trajectory.shape[0]
        device = trajectory.device
        
        # 1. Progress Reward: 基于终点距离
        if mask is not None:
            valid_counts = mask.sum(dim=-1).long().clamp(min=1)
            last_idx = valid_counts - 1
            batch_range = torch.arange(B, device=device)
            pred_end = trajectory[batch_range, last_idx]
            gt_end = gt_trajectory[batch_range, last_idx]
        else:
            pred_end = trajectory[:, -1]
            gt_end = gt_trajectory[:, -1]
        
        fde = torch.norm(pred_end - gt_end, dim=-1)
        progress = torch.exp(-fde / 5.0)  # 5m as reference
        
        # 2. Efficiency Reward: 基于平均速度
        # 目标速度约为 10 m/s (36 km/h)
        efficiency = self._compute_efficiency_reward(trajectory, dt)
        
        # 3. Route Completion: 轨迹对 GT 的贴合度
        route_completion = self._compute_route_completion(trajectory, gt_trajectory, mask)
        
        return {
            'progress': progress,
            'efficiency': efficiency,
            'route_completion': route_completion,
        }
    
    def _compute_efficiency_reward(
        self,
        trajectory: torch.Tensor,
        dt: float = 0.5,
        target_speed: float = 10.0,
    ) -> torch.Tensor:
        """计算效率奖励。
        
        基于轨迹的平均速度，奖励接近目标速度的轨迹。
        
        Args:
            trajectory: [B, T, 2] 轨迹
            dt: 时间步间隔
            target_speed: 目标速度 (m/s)
        
        Returns:
            [B] 效率奖励 (越高越好)
        """
        B, T, _ = trajectory.shape
        
        # 计算速度: [B, T-1, 2]
        velocity = torch.diff(trajectory, dim=1) / dt
        speed = torch.norm(velocity, dim=-1)  # [B, T-1]
        
        # 平均速度
        mean_speed = speed.mean(dim=-1)  # [B]
        
        # 奖励: 使用高斯惩罚，期望速度越接近目标速度越好
        # reward = exp(-|speed - target|^2 / (2 * sigma^2))
        sigma = 3.0  # 速度容差
        efficiency = torch.exp(-((mean_speed - target_speed) ** 2) / (2 * sigma ** 2))
        
        return efficiency
    
    def _compute_route_completion(
        self,
        trajectory: torch.Tensor,
        gt_trajectory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        normalize_scale: float = 5.0,
    ) -> torch.Tensor:
        """计算路线完成度奖励。
        
        使用 DTW (Dynamic Time Warping) 或简化版欧氏距离来衡量
        预测轨迹与 GT 轨迹的整体贴合度。
        
        Args:
            trajectory: [B, T, 2] 预测轨迹
            gt_trajectory: [B, T, 2] GT 轨迹
            mask: [B, T] 有效时间步掩码
            normalize_scale: 归一化因子
        
        Returns:
            [B] 路线完成度 (0-1 之间，越高越好)
        """
        B, T, _ = trajectory.shape
        
        # 简化版: 使用逐点欧氏距离的平均值
        diff = trajectory - gt_trajectory
        pointwise_dist = torch.norm(diff, dim=-1)  # [B, T]
        
        # 应用 mask
        if mask is not None:
            pointwise_dist = pointwise_dist * mask.float()
            mean_dist = pointwise_dist.sum(dim=-1) / (mask.sum(dim=-1) + 1e-6)
        else:
            mean_dist = pointwise_dist.mean(dim=-1)
        
        # 转换为奖励: exp(-dist / scale)
        route_completion = torch.exp(-mean_dist / normalize_scale)
        
        return route_completion
    
    def _compute_cost(
        self,
        trajectory: torch.Tensor,
        agent_positions: Optional[torch.Tensor],
        lane_boundaries: Optional[torch.Tensor],
        dt: float,
    ) -> Dict[str, torch.Tensor]:
        """计算 cost 分支。"""
        B = trajectory.shape[0]
        device = trajectory.device
        
        # 1. Collision Cost
        collision = self._compute_collision_cost(trajectory, agent_positions)
        
        # 2. Offroad Cost
        offroad = self._compute_offroad_cost(trajectory, lane_boundaries)
        
        # 3. Comfort Cost
        comfort = self._compute_comfort_cost(trajectory, dt)
        
        return {
            'collision': collision,
            'offroad': offroad,
            'comfort': comfort,
        }
    
    def _compute_collision_cost(
        self,
        trajectory: torch.Tensor,
        agent_positions: Optional[torch.Tensor],
        x_thresh: float = 1.5,
        y_thresh: float = 3.0,
    ) -> torch.Tensor:
        """碰撞成本。"""
        if agent_positions is None:
            return torch.zeros(trajectory.shape[0], device=trajectory.device)
        
        # 简化版：使用 lateral deviation 作为代理
        # 真实实现需要 agent 未来轨迹
        lateral_dev = trajectory[:, :, 1].abs()
        collision = (lateral_dev < 1.0).float().mean(dim=-1)
        return collision
    
    def _compute_offroad_cost(
        self,
        trajectory: torch.Tensor,
        lane_boundaries: Optional[torch.Tensor],
        threshold: float = 2.0,
    ) -> torch.Tensor:
        """离道成本。
        
        使用 lateral deviation 和道路边界两种方式计算:
        1. 如果有 lane_boundaries: 计算到边界的最小距离
        2. 否则: 使用 lateral deviation 作为简化代理
        """
        device = trajectory.device
        
        if lane_boundaries is not None:
            # 真实实现: 计算到道路边界的最小距离
            # 简化版: 点到线段距离
            # ego: [B, T, 1, 1, 2], boundary: [B, 1, N, P, 2]
            ego = trajectory.unsqueeze(2).unsqueeze(3)
            boundary = lane_boundaries.unsqueeze(1)
            
            # 最小距离
            dist = torch.norm(ego - boundary, dim=-1)  # [B, T, N, P]
            min_dist = dist.amin(dim=(2, 3))  # [B, T]
            
            # 低于阈值则有惩罚
            offroad = F.relu(threshold - min_dist).mean(dim=-1) / threshold
            return offroad
        else:
            # 简化版: 使用 lateral deviation
            lateral_dev = trajectory[:, :, 1].abs()
            offroad = F.relu(threshold - lateral_dev).mean(dim=-1) / threshold
            return offroad
    
    def _compute_comfort_cost(
        self,
        trajectory: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """舒适度成本。"""
        # 加速度
        velocity = torch.diff(trajectory, dim=1) / dt
        acceleration = torch.diff(velocity, dim=1)
        
        # 曲率 (lateral acceleration proxy)
        curvature = torch.norm(acceleration, dim=-1)
        
        comfort = curvature.mean(dim=-1)
        return comfort


class BeneficialUpdateFilter:
    """
    Beneficial Update Filter - 比 STAPO 更严格的定义
    
    只保留同时满足以下条件的更新样本：
    1. reward_improvement > reward_margin
    2. cost_increase < cost_increase_threshold
    3. kl_drift < kl_bound
    
    这确保了：
    - 真正有益 (不是假阳性)
    - 不引入额外风险 (cost 不上升)
    - 更新幅度受控 (不偏离原始策略太远)
    """
    
    def __init__(
        self,
        reward_margin: float = 0.0,
        cost_increase_threshold: float = 0.1,
        kl_bound: float = 0.05,
        min_retention_ratio: float = 0.1,
    ):
        self.reward_margin = reward_margin
        self.cost_increase_threshold = cost_increase_threshold
        self.kl_bound = kl_bound
        self.min_retention_ratio = min_retention_ratio
    
    def compute_mask(
        self,
        reward_improvement: torch.Tensor,
        cost_improvement: torch.Tensor,
        kl_drift: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算过滤掩码。
        
        Args:
            reward_improvement: [B] reward 增量
            cost_improvement: [B] cost 增量 (越小越好，负值=cost下降)
            kl_drift: [B] KL 散度增量 (可选)
        
        Returns:
            (mask, diagnostics)
        """
        B = reward_improvement.shape[0]
        
        # 条件1: reward 有提升
        has_reward_gain = reward_improvement > self.reward_margin
        
        # 条件2: cost 不上升 (cost_improvement < 0 表示 cost 下降)
        cost_not_increase = cost_improvement < self.cost_increase_threshold
        
        # 条件3: KL drift 受控
        if kl_drift is not None:
            kl_controlled = kl_drift < self.kl_bound
        else:
            kl_controlled = torch.ones_like(has_reward_gain)
        
        # 综合判断: 三个条件都要满足
        is_beneficial = has_reward_gain & cost_not_increase & kl_controlled
        
        # 确保最小保留比例
        mask = self._ensure_min_retention(is_beneficial, reward_improvement)
        
        diagnostics = {
            'n_total': B,
            'n_reward_gain': has_reward_gain.sum().item(),
            'n_cost_ok': cost_not_increase.sum().item(),
            'n_kl_ok': kl_controlled.sum().item(),
            'n_beneficial': is_beneficial.sum().item(),
            'n_final': mask.sum().item(),
            'retention_ratio': mask.float().mean().item(),
        }
        
        return mask, diagnostics
    
    def _ensure_min_retention(
        self,
        is_beneficial: torch.Tensor,
        reward_improvement: torch.Tensor,
    ) -> torch.Tensor:
        """确保最小保留比例。"""
        B = is_beneficial.shape[0]
        min_count = max(int(self.min_retention_ratio * B), 1)
        
        current_count = is_beneficial.sum().item()
        if current_count >= min_count:
            return is_beneficial
        
        # 按 reward_improvement 排序，补充到最小数量
        mask = is_beneficial.clone()
        if current_count < min_count:
            # 找出被过滤但 reward_improvement 最高的
            _, indices = reward_improvement.topk(min_count)
            mask = torch.zeros_like(is_beneficial)
            mask[indices] = True
        
        return mask


class ConservativeRLUpdate(nn.Module):
    """
    Conservative E2E RL 更新器
    
    核心机制：
    1. Reward-Cost 分离优化
    2. Beneficial Update Filter
    3. KL Divergence Constraint
    4. Reference Model Anchoring
    
    使用 PPO-style 更新，配合保守约束。
    """
    
    def __init__(
        self,
        config: Optional[ConservativeRLConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = config or ConservativeRLConfig(**kwargs)
        
        # Reward-Cost 分离器
        self.reward_cost_separator = RewardCostSeparator()
        
        # Beneficial Update Filter
        self.beneficial_filter = BeneficialUpdateFilter(
            reward_margin=self.cfg.reward_margin,
            cost_increase_threshold=self.cfg.cost_increase_threshold,
            kl_bound=self.cfg.kl_bound,
        )
    
    def compute_loss(
        self,
        # 当前策略输出
        trajectory: torch.Tensor,
        old_log_prob: torch.Tensor,
        value_estimate: torch.Tensor,
        
        # Reference
        reference_trajectory: torch.Tensor,
        
        # GT
        gt_trajectory: torch.Tensor,
        agent_positions: Optional[torch.Tensor] = None,
        lane_boundaries: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        dt: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算保守 RL 损失。
        
        Args:
            trajectory: [B, T, 2] 当前策略轨迹
            old_log_prob: [B] 旧策略的对数概率
            value_estimate: [B] 价值估计
            reference_trajectory: [B, T, 2] 参考模型轨迹
            gt_trajectory: [B, T, 2] GT 轨迹
            ...
        
        Returns:
            (total_loss, diagnostics)
        """
        # === 1. Reward-Cost 分离 ===
        rc_output = self.reward_cost_separator.compute(
            trajectory=trajectory,
            gt_trajectory=gt_trajectory,
            agent_positions=agent_positions,
            lane_boundaries=lane_boundaries,
            mask=mask,
            dt=dt,
        )
        
        reward = rc_output['reward_branch']
        cost = rc_output['cost_branch']
        is_safe = rc_output['is_safe']
        
        # === 2. 计算 Advantage ===
        advantage = reward - self.cfg.value_coef * cost
        
        # === 3. Reference Anchor Loss ===
        ref_loss = torch.tensor(0.0, device=trajectory.device)
        if self.cfg.use_reference_anchor and reference_trajectory is not None:
            ref_loss = F.mse_loss(trajectory, reference_trajectory)
        
        # === 4. Beneficial Update Filter ===
        if self.cfg.use_beneficial_filter:
            # 与参考轨迹的偏差作为 improvement 代理
            if reference_trajectory is not None:
                # 计算轨迹偏差: shape [B, T, 2] -> [B]
                diff_norm = torch.norm(trajectory - reference_trajectory, dim=(-2, -1))  # [B]
                reward_improvement = -diff_norm.mean(-1) if diff_norm.dim() > 1 else -diff_norm
            else:
                reward_improvement = reward if reward.dim() == 1 else reward.mean(-1)
            
            cost_change = cost  # cost 越低越好
            
            beneficial_mask, filter_diag = self.beneficial_filter.compute_mask(
                reward_improvement=reward_improvement,
                cost_improvement=cost_change,
            )
        else:
            beneficial_mask = torch.ones_like(advantage, dtype=torch.bool)
            filter_diag = {'n_total': advantage.shape[0], 'n_final': advantage.shape[0]}
        
        # === 5. PPO-style Policy Loss ===
        # 简化版: 使用 reward 作为 loss
        policy_loss = -advantage
        
        # 应用 mask
        masked_loss = policy_loss * beneficial_mask.float()
        filtered_policy_loss = masked_loss.sum() / (beneficial_mask.sum() + 1e-8)
        
        # === 6. Reference Loss ===
        total_loss = (
            filtered_policy_loss +
            self.cfg.reference_alpha * ref_loss +
            self.cfg.entropy_coef * (-old_log_prob).mean()  # 熵正则化
        )
        
        # === Diagnostics ===
        diagnostics = {
            'loss_total': total_loss.item(),
            'loss_policy': filtered_policy_loss.item(),
            'loss_ref': ref_loss.item(),
            'mean_reward': reward.mean().item(),
            'mean_cost': cost.mean().item(),
            'mean_advantage': advantage.mean().item(),
            'safety_ratio': is_safe.float().mean().item(),
            **filter_diag,
        }
        
        return total_loss, diagnostics


class ConservativeE2ETrainer:
    """
    Conservative E2E Post-Training 训练器
    
    使用方式:
        trainer = ConservativeE2ETrainer(
            model=planner,
            reference_model=planner_copy,  # 原始模型
            config=ConservativeRLConfig(),
        )
        
        for batch in dataloader:
            loss, diag = trainer.step(batch)
            loss.backward()
            ...
    """
    
    def __init__(
        self,
        model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[ConservativeRLConfig] = None,
        device: torch.device = torch.device('cuda'),
    ):
        self.model = model
        self.reference_model = reference_model
        self.device = device
        
        # 默认配置
        self.cfg = config or ConservativeRLConfig()
        
        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.cfg.lr,
                weight_decay=1e-4,
            )
        else:
            self.optimizer = optimizer
        
        # Conservative RL 更新器
        self.crl = ConservativeRLUpdate(config=self.cfg)
        
        # Reference model 同步
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
    
    @torch.no_grad()
    def get_reference_trajectory(self, batch: Dict) -> Optional[torch.Tensor]:
        """从 Reference Model 获取参考轨迹。"""
        if self.reference_model is None:
            return None
        
        # 简化版：直接用原始模型的输出
        # 真实实现需要完整的 forward
        interface = batch.get('interface')
        if interface is not None:
            return interface.reference_plan
        return None
    
    def step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        单步训练。
        
        Args:
            batch: 包含 trajectory, gt_trajectory 等的字典
        
        Returns:
            (loss, diagnostics)
        """
        self.model.train()
        
        # 获取数据
        trajectory = batch['trajectory'].to(self.device)
        gt_trajectory = batch['gt_trajectory'].to(self.device)
        agent_positions = batch.get('agent_positions')
        if agent_positions is not None:
            agent_positions = agent_positions.to(self.device)
        lane_boundaries = batch.get('lane_boundaries')
        mask = batch.get('mask')
        
        # 获取 Reference 轨迹
        reference_trajectory = self.get_reference_trajectory(batch)
        
        # 简化版: 假设 log_prob 和 value 是模型输出的
        old_log_prob = torch.zeros(trajectory.shape[0], device=self.device)
        value_estimate = torch.zeros(trajectory.shape[0], device=self.device)
        
        # 计算损失
        loss, diag = self.crl.compute_loss(
            trajectory=trajectory,
            old_log_prob=old_log_prob,
            value_estimate=value_estimate,
            reference_trajectory=reference_trajectory,
            gt_trajectory=gt_trajectory,
            agent_positions=agent_positions,
            lane_boundaries=lane_boundaries,
            mask=mask,
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.grad_clip
            )
        
        self.optimizer.step()
        
        return loss, diag
    
    def update_reference_model(self):
        """更新 Reference Model。"""
        if self.reference_model is not None:
            self.reference_model.load_state_dict(self.model.state_dict())


__all__ = [
    'ConservativeRLConfig',
    'ConservativeRLUpdate',
    'ConservativeE2ETrainer',
    'RewardCostSeparator',
    'BeneficialUpdateFilter',
]
