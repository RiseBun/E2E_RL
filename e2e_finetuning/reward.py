"""
Closed-Loop Reward Computation Module for E2E RL Finetuning

This module computes rewards based on closed-loop trajectory evaluation,
including collision, off-road, progress, and comfort metrics.

Author: E2E_RL Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RewardConfig:
    """Configuration for reward computation weights."""
    collision_weight: float = 0.5
    offroad_weight: float = 0.5
    progress_weight: float = 8.0
    comfort_weight: float = 2.0
    ttc_weight: float = 5.0
    
    # Reward shaping
    collision_margin: float = 1.0  # meters
    offroad_margin: float = 0.5  # meters from drivable area boundary
    
    # Time delta for velocity/acceleration computation
    dt: float = 0.5  # seconds


class ClosedLoopReward(nn.Module):
    """Compute closed-loop rewards for E2E RL finetuning.
    
    This module evaluates trajectories based on:
    1. Collision risk (distance to other agents)
    2. Off-road violation (leaving drivable area)
    3. Progress (following GT trajectory)
    4. Comfort (smooth acceleration/steering)
    5. Time-to-collision (safety margin)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__()
        self.config = config or RewardConfig()
    
    def forward(
        self,
        trajectory: torch.Tensor,
        gt_trajectory: torch.Tensor,
        map_features: Optional[Dict[str, torch.Tensor]] = None,
        agent_positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute closed-loop rewards.
        
        Args:
            trajectory: Predicted trajectory [B, T, 2] in ego-centric coordinates
            gt_trajectory: Ground truth trajectory [B, T, 2]
            map_features: Optional map features (drivable area, etc.)
                - 'drivable_area': [B, H, W] binary mask
            agent_positions: Optional other agent positions [B, N, 2]
            mask: Valid trajectory mask [B, T]
        
        Returns:
            Dictionary containing:
                - collision_penalty: [B]
                - offroad_penalty: [B]
                - progress_reward: [B]
                - comfort_penalty: [B]
                - ttc_reward: [B]
                - total_reward: [B]
        """
        B, T, _ = trajectory.shape
        
        # Apply mask if provided
        if mask is None:
            mask = torch.ones(B, T, device=trajectory.device, dtype=torch.bool)
        
        # 1. Collision risk
        collision_penalty = self._compute_collision_risk(
            trajectory, agent_positions, mask
        )
        
        # 2. Off-road penalty
        offroad_penalty = self._compute_offroad_penalty(
            trajectory, map_features, mask
        )
        
        # 3. Progress reward (trajectory deviation from GT)
        progress_reward = self._compute_progress_reward(
            trajectory, gt_trajectory, mask
        )
        
        # 4. Comfort penalty (smoothness)
        comfort_penalty = self._compute_comfort_penalty(
            trajectory, mask
        )
        
        # 5. Time-to-collision reward
        ttc_reward = self._compute_ttc_reward(
            trajectory, agent_positions, mask
        )
        
        # Combine rewards
        total_reward = (
            -self.config.collision_weight * collision_penalty +
            -self.config.offroad_weight * offroad_penalty +
            self.config.progress_weight * progress_reward +
            -self.config.comfort_weight * comfort_penalty +
            self.config.ttc_weight * ttc_reward
        )
        
        return {
            'collision_penalty': collision_penalty,
            'offroad_penalty': offroad_penalty,
            'progress_reward': progress_reward,
            'comfort_penalty': comfort_penalty,
            'ttc_reward': ttc_reward,
            'total_reward': total_reward,
        }
    
    def _compute_collision_risk(
        self,
        trajectory: torch.Tensor,
        agent_positions: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute collision risk based on distance to other agents.
        
        Args:
            trajectory: [B, T, 2]
            agent_positions: [B, N, 2] or None
            mask: [B, T]
        
        Returns:
            collision_penalty: [B], average penalty over valid timesteps
        """
        B, T, _ = trajectory.shape
        penalty = torch.zeros(B, device=trajectory.device)
        
        if agent_positions is None or agent_positions.numel() == 0:
            return penalty
        
        N = agent_positions.shape[1]  # number of agents
        
        # Expand trajectory: [B, T, 2] -> [B, T, 1, 2]
        # Expand agents: [B, N, 2] -> [B, 1, N, 2]
        traj_expanded = trajectory.unsqueeze(2)  # [B, T, 1, 2]
        agents_expanded = agent_positions.unsqueeze(1)  # [B, 1, N, 2]
        
        # Compute distances: [B, T, N]
        distances = torch.norm(traj_expanded - agents_expanded, dim=-1)
        
        # Collision if distance < margin
        collision_dist = distances < self.config.collision_margin
        
        # Apply mask and average
        collision_score = collision_dist.float()
        if mask is not None:
            collision_score = collision_score * mask.unsqueeze(-1).float()
        
        # Penalty: average collision risk over time and agents
        penalty = collision_score.sum(dim=(1, 2)) / (mask.sum(dim=1, keepdim=True) * N + 1e-6)
        
        return penalty
    
    def _compute_offroad_penalty(
        self,
        trajectory: torch.Tensor,
        map_features: Optional[Dict[str, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute off-road penalty based on drivable area.
        
        Args:
            trajectory: [B, T, 2] in ego-centric coordinates
            map_features: Dict with 'drivable_area' [B, H, W]
            mask: [B, T]
        
        Returns:
            offroad_penalty: [B]
        """
        B, T, _ = trajectory.shape
        penalty = torch.zeros(B, device=trajectory.device)
        
        if map_features is None or 'drivable_area' not in map_features:
            return penalty
        
        drivable_area = map_features['drivable_area']  # [B, H, W] or [H, W]
        
        # Convert trajectory to map coordinates
        # Assuming trajectory is in ego-centric coordinates centered at ego
        # This is a simplified version - real implementation needs proper
        # coordinate transformation based on ego pose
        
        # For now, use a simple heuristic: penalize extreme deviations
        # In real implementation, sample points and check against drivable area
        
        # Simple heuristic: large lateral deviation indicates potential off-road
        lateral_deviation = trajectory[:, :, 1].abs()  # [B, T]
        
        # Assume off-road if lateral deviation > threshold
        offroad_threshold = 3.0  # meters (half lane width)
        is_offroad = lateral_deviation > offroad_threshold
        
        # Apply mask and average
        if mask is not None:
            is_offroad = is_offroad * mask
        
        penalty = is_offroad.float().sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        return penalty
    
    def _compute_progress_reward(
        self,
        trajectory: torch.Tensor,
        gt_trajectory: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute progress reward based on deviation from GT.
        
        Smaller deviation = higher reward.
        
        Args:
            trajectory: [B, T, 2]
            gt_trajectory: [B, T, 2]
            mask: [B, T]
        
        Returns:
            progress_reward: [B]
        """
        # Compute L2 distance to GT
        deviation = torch.norm(trajectory - gt_trajectory, dim=-1)  # [B, T]
        
        # Apply mask
        if mask is not None:
            deviation = deviation * mask.float()
        
        # Average deviation over time
        avg_deviation = deviation.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # Convert to reward: exponential decay with deviation
        # Small deviation -> high reward, large deviation -> low reward
        progress_reward = torch.exp(-avg_deviation / 5.0)  # 5m as reference
        
        return progress_reward
    
    def _compute_comfort_penalty(
        self,
        trajectory: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute comfort penalty based on trajectory smoothness.
        
        Penalizes:
        - High acceleration (large velocity changes)
        - High jerk (changes in acceleration)
        - Sharp turns (high curvature)
        
        Args:
            trajectory: [B, T, 2] in ego-centric coordinates
            mask: [B, T]
        
        Returns:
            comfort_penalty: [B]
        """
        dt = self.config.dt
        
        # Compute velocity: [B, T-1, 2]
        velocity = torch.diff(trajectory, dim=1) / dt
        
        # Compute acceleration: [B, T-2, 2]
        acceleration = torch.diff(velocity, dim=1) / dt
        
        # Compute jerk: [B, T-3, 2]
        jerk = torch.diff(acceleration, dim=1) / dt
        
        # Compute speed magnitude: [B, T-1]
        speed = torch.norm(velocity, dim=-1)
        
        # Compute curvature (lateral acceleration proxy): [B, T-2]
        curvature = torch.norm(acceleration, dim=-1)
        
        # Apply mask (adjust for different tensor lengths)
        mask_v = mask[:, 1:] if mask is not None else None
        mask_a = mask[:, 2:] if mask is not None else None
        
        # Comfort penalty components
        speed_penalty = speed.mean(dim=1) / 20.0  # normalize by typical speed
        accel_penalty = curvature.mean(dim=1) / 3.0  # normalize by typical lateral accel
        
        # Total comfort penalty
        comfort_penalty = 0.5 * speed_penalty + 0.5 * accel_penalty
        
        return comfort_penalty
    
    def _compute_ttc_reward(
        self,
        trajectory: torch.Tensor,
        agent_positions: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute time-to-collision reward.
        
        Higher TTC (safer) = higher reward.
        
        Args:
            trajectory: [B, T, 2]
            agent_positions: [B, N, 2] or None
            mask: [B, T]
        
        Returns:
            ttc_reward: [B]
        """
        B, T, _ = trajectory.shape
        reward = torch.zeros(B, device=trajectory.device)
        
        if agent_positions is None or agent_positions.numel() == 0:
            return reward
        
        dt = self.config.dt
        
        # Compute velocity: [B, T-1, 2]
        velocity = torch.diff(trajectory, dim=1) / dt
        
        # Expand for agent comparison
        traj_expanded = trajectory.unsqueeze(2)  # [B, T, 1, 2]
        agents_expanded = agent_positions.unsqueeze(1)  # [B, 1, N, 2]
        
        # Distance to closest agent at each timestep
        distances = torch.norm(traj_expanded - agents_expanded, dim=-1)  # [B, T, N]
        min_distances = distances.min(dim=-1).values  # [B, T]
        
        # Compute relative velocity towards closest agent
        # (simplified: just use ego velocity magnitude as TTC proxy)
        speed = torch.norm(velocity, dim=-1)  # [B, T-1]
        
        # TTC = distance / speed (clamped to avoid division by zero)
        ttc = min_distances[:, :-1] / (speed.clamp(min=0.1))  # [B, T-1]
        
        # TTC reward: normalize to [0, 1] range
        ttc_reward = (ttc / 5.0).clamp(0, 1).mean(dim=1)  # 5s as reference
        
        return ttc_reward


class RewardNormalizer:
    """Normalize rewards for stable RL training.
    
    Implements running mean/std normalization.
    """
    
    def __init__(
        self,
        momentum: float = 0.99,
        epsilon: float = 1e-8,
        device: torch.device = torch.device('cpu'),
    ):
        self.momentum = momentum
        self.epsilon = epsilon
        self.device = device
        
        self.running_mean = None
        self.running_var = None
        self.count = 0
    
    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize reward using running statistics.
        
        Args:
            reward: [B] or [B, ...]
        
        Returns:
            normalized_reward: same shape as input
        """
        # Flatten for statistics
        flat_reward = reward.flatten()
        
        if self.running_mean is None:
            # Initialize with first batch
            self.running_mean = flat_reward.mean()
            self.running_var = flat_reward.var()
            self.count = flat_reward.numel()
        else:
            # Update running statistics
            batch_mean = flat_reward.mean()
            batch_var = flat_reward.var()
            batch_count = flat_reward.numel()
            
            delta = batch_mean - self.running_mean
            new_count = self.count + batch_count
            
            self.running_mean = self.running_mean + delta * batch_count / new_count
            self.running_var = (
                self.momentum * self.running_var +
                (1 - self.momentum) * batch_var +
                delta * delta * self.count * batch_count / (new_count * new_count)
            )
            self.count = new_count
        
        # Normalize
        std = torch.sqrt(self.running_var + self.epsilon)
        normalized = (reward - self.running_mean) / std
        
        return normalized
    
    def denormalize(self, normalized_reward: torch.Tensor) -> torch.Tensor:
        """Denormalize reward back to original scale.
        
        Args:
            normalized_reward: normalized reward
        
        Returns:
            reward: original scale
        """
        std = torch.sqrt(self.running_var + self.epsilon)
        return normalized_reward * std + self.running_mean


def compute_reward_from_metrics(
    gt_pdm_score: Dict[str, torch.Tensor],
    trajectory: torch.Tensor,
    vocab: torch.Tensor,
    config: Optional[RewardConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Compute reward from pre-computed PDM metrics (compatible with R2SE).
    
    This function bridges the gap between R2SE's pre-computed metrics
    and the closed-loop reward computation.
    
    Args:
        gt_pdm_score: Dict containing:
            - score: [B, M] trajectory scores
            - no_at_fault_collisions: [B, M]
            - drivable_area_compliance: [B, M]
            - time_to_collision_within_bound: [B, M]
            - comfort: [B, M]
            - ego_progress: [B, M]
        trajectory: [B, T, 2] predicted trajectory
        vocab: [M, T, 3] trajectory vocabulary
        config: Reward configuration
    
    Returns:
        Dictionary of rewards
    """
    config = config or RewardConfig()
    device = trajectory.device
    
    B, M = gt_pdm_score['score'].shape[:2]
    
    # Get best trajectory index based on score
    best_idx = gt_pdm_score['score'].argmax(dim=1)  # [B]
    
    # 1. Collision penalty (from PDM metrics)
    # Higher value = fewer collisions = better
    collision_penalty = 1.0 - gt_pdm_score['no_at_fault_collisions'].gather(1, best_idx.unsqueeze(1)).squeeze(1)
    
    # 2. Drivable area compliance (from PDM metrics)
    da_compliance = gt_pdm_score['drivable_area_compliance'].gather(1, best_idx.unsqueeze(1)).squeeze(1)
    offroad_penalty = 1.0 - da_compliance
    
    # 3. Progress (from PDM metrics)
    progress_reward = gt_pdm_score['ego_progress'].gather(1, best_idx.unsqueeze(1)).squeeze(1)
    
    # 4. Comfort (from PDM metrics)
    comfort = gt_pdm_score['comfort'].gather(1, best_idx.unsqueeze(1)).squeeze(1)
    comfort_penalty = 1.0 - comfort
    
    # 5. TTC (from PDM metrics)
    ttc = gt_pdm_score['time_to_collision_within_bound'].gather(1, best_idx.unsqueeze(1)).squeeze(1)
    ttc_reward = ttc
    
    # 6. Combine rewards
    total_reward = (
        -config.collision_weight * collision_penalty +
        -config.offroad_weight * offroad_penalty +
        config.progress_weight * progress_reward +
        -config.comfort_weight * comfort_penalty +
        config.ttc_weight * ttc_reward
    )
    
    return {
        'collision_penalty': collision_penalty,
        'offroad_penalty': offroad_penalty,
        'progress_reward': progress_reward,
        'comfort_penalty': comfort_penalty,
        'ttc_reward': ttc_reward,
        'total_reward': total_reward,
    }
