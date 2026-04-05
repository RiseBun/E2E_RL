"""
STAPO Gate - STAPO 风格的门控机制

核心思想：在 Policy Gradient Loss 内部，筛选出"虚假有益更新"并静音

STAPO 识别的虚假有益更新特征：
- 正 advantage（看起来是有益的）
- 低概率（被执行的概率低）
- 低熵（模型对这个动作的确信度高）

这类更新被静音后重归一化 loss，实现"只学习真正值得学的更新"。

这才是正确的 STAPO 应用方式：在 loss 内部筛选，而非外挂一个裁判器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class STAPOGateConfig:
    """STAPO Gate 配置。"""
    enabled: bool = True
    advantage_threshold: float = 0.0
    probability_threshold: float = 0.1
    entropy_threshold: float = 0.5
    min_retention_ratio: float = 0.1
    use_combined_threshold: bool = True


class STAPOGate:
    """
    STAPO 门控机制
    
    在计算 policy gradient loss 时，找出虚假有益更新并静音。
    
    不同于之前的 HUF：
    - 之前：在推理时裁决是否接受修正（后验）
    - 现在：在训练时静音虚假有益更新（同步于 loss 计算）
    """
    
    def __init__(
        self,
        config: Optional[STAPOGateConfig] = None,
        **kwargs,
    ):
        """初始化 STAPO Gate。

        Args:
            config: STAPOGateConfig 配置对象
            **kwargs: 如果没有传 config，可以直接传参
        """
        if config is None:
            config = STAPOGateConfig(**kwargs)
        self.cfg = config
        self.stapo_threshold = 0.5  # 保留兼容性
        self.min_retention_ratio = config.min_retention_ratio
        self.advantage_threshold = config.advantage_threshold
        self.probability_threshold = config.probability_threshold
        self.entropy_threshold = config.entropy_threshold
        self.use_combined_threshold = config.use_combined_threshold
    
    def compute_mask(
        self,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_probs: Optional[torch.Tensor] = None,
        entropies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算静音掩码
        
        Args:
            advantages: [B] or [B, ...] advantage 值
            action_log_probs: [B] or [B, ...] 动作对数概率
            action_probs: [B] or [B, ...] 动作概率（可选）
            entropies: [B] or [B, ...] 熵（可选）
        
        Returns:
            mask: [B] 布尔掩码，True 表示保留，False 表示静音
        """
        # 确保形状一致
        shape = advantages.shape
        flat_adv = advantages.flatten()
        
        # 转换为概率（如果只有 log_prob）
        if action_probs is None:
            action_probs = action_log_probs.exp()
        
        # 熵
        if entropies is None:
            entropies = -(action_log_probs * action_probs).sum(dim=-1) if action_log_probs.dim() > 1 else -action_log_probs * action_probs
        
        # STAPO 条件检测
        # 虚假有益更新 = 正 advantage + 低概率 + 低熵
        is_positive_adv = flat_adv > self.advantage_threshold
        is_low_prob = action_probs < self.probability_threshold
        is_low_entropy = entropies < self.entropy_threshold if entropies is not None else torch.zeros_like(is_positive_adv)
        
        # 综合判断
        if self.use_combined_threshold:
            is_spurious = is_positive_adv & is_low_prob & is_low_entropy
        else:
            # 简化模式
            is_spurious = is_low_prob
        
        # 创建掩码：虚假更新静音，其余保留
        mask = ~is_spurious
        
        # 确保最小保留比例
        if mask.sum() < self.min_retention_ratio * mask.numel():
            # 按 advantage 排序，保留最高的
            _, indices = flat_adv.topk(int(self.min_retention_ratio * mask.numel()))
            mask = torch.zeros_like(mask)
            mask[indices] = True
        
        return mask.view(shape)
    
    def filter_loss(
        self,
        policy_loss: torch.Tensor,
        advantages: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_probs: Optional[torch.Tensor] = None,
        entropies: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        过滤 loss 并重归一化
        
        Args:
            policy_loss: [B] 每个样本的 policy loss
            advantages: [B] advantage 值
            action_log_probs: [B] 动作对数概率
            action_probs: [B] 动作概率
            entropies: [B] 熵
        
        Returns:
            filtered_loss: 重归一化后的 loss
            diagnostics: 诊断信息
        """
        mask = self.compute_mask(
            advantages=advantages,
            action_log_probs=action_log_probs,
            action_probs=action_probs,
            entropies=entropies,
        )
        
        # 应用掩码
        masked_loss = policy_loss * mask.float()
        
        # 重归一化
        n_active = mask.sum()
        if n_active > 0:
            filtered_loss = (masked_loss.sum() / n_active) * policy_loss.numel()
        else:
            filtered_loss = policy_loss.mean()
        
        # 诊断信息
        diagnostics = {
            'n_total': policy_loss.numel(),
            'n_active': n_active.item(),
            'n_masked': (mask == False).sum().item(),
            'retention_ratio': n_active.item() / policy_loss.numel(),
        }
        
        return filtered_loss, diagnostics


class AdvantageThresholdGate:
    """
    简化的 Advantage 门控
    
    只保留正 advantage 的更新
    """
    
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
    
    def filter(
        self,
        losses: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        过滤 loss
        
        Args:
            losses: [B] 每个样本的 loss
            advantages: [B] advantage 值
        
        Returns:
            filtered_loss: 过滤后的 loss
            diagnostics: 诊断信息
        """
        mask = advantages > self.threshold
        
        if mask.sum() > 0:
            filtered_loss = (losses * mask.float()).sum() / mask.sum()
        else:
            filtered_loss = losses.mean()
        
        diagnostics = {
            'n_total': losses.numel(),
            'n_active': mask.sum().item(),
            'mean_advantage': advantages.mean().item(),
            'positive_advantage_ratio': (advantages > 0).float().mean().item(),
        }
        
        return filtered_loss, diagnostics
