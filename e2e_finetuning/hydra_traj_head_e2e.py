"""
HydraTrajHeadE2E - 支持端到端 RL 微调的增强规划头

核心增强：
1. LoRA Adapter: 只训练少量参数，保持主体不变
2. Value Head: 估计状态价值，用于 advantage 计算
3. PlanningInterface 输出: 与 E2E_RL 统一接口兼容
4. Log Prob 输出: 支持 PPO-style 更新

使用方式:
    # 原规划头
    head = HydraTrajEnsHead(...)
    
    # 包装为 E2E 版本
    e2e_head = HydraTrajHeadE2E(
        base_head=head,
        enable_lora=True,
        lora_rank=16,
        enable_value_head=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 PlanningInterface
try:
    from E2E_RL.planning_interface.interface import PlanningInterface
except ImportError:
    PlanningInterface = None


@dataclass
class LoRAConfig:
    """LoRA 配置。"""
    enabled: bool = True
    rank: int = 16  # LoRA rank
    alpha: float = 1.0  # scaling factor
    dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("plan_reg_branch", "plan_cls_branch")


class LoRALinear(nn.Module):
    """
    LoRA Linear Layer
    
    将原始 Linear 层分解为:
    W = W_base + alpha * (W_a @ W_b)
    
    其中 W_a, W_b 是可学习的低秩矩阵。
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # 冻结原始权重
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # LoRA 参数
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        # 原始输出
        base_output = self.base_layer(x)
        
        # LoRA 增量
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        lora_output = self.lora_dropout(lora_output)
        
        return base_output + self.alpha * lora_output
    
    def merge_weights(self):
        """合并 LoRA 权重到 base_layer。"""
        W_delta = self.alpha * (self.lora_B @ self.lora_A)
        W_new = self.base_layer.weight + W_delta
        self.base_layer.weight = nn.Parameter(W_new)
        # 清零 LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros_like(self.lora_A))
        self.lora_B = nn.Parameter(torch.zeros_like(self.lora_B))


class ValueHead(nn.Module):
    """
    Value Head - 估计状态价值 V(s)
    
    用于 advantage 计算: A(s, a) = Q(s, a) - V(s)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出状态价值。"""
        return self.net(x)


class HydraTrajHeadE2E(nn.Module):
    """
    支持端到端 RL 微调的增强规划头
    
    包装原始 HydraTrajEnsHead，添加:
    1. LoRA adapters (可选)
    2. Value head
    3. 统一 PlanningInterface 输出
    
    使用方式:
        # 包装
        e2e_head = HydraTrajHeadE2E(base_head=vad_head)
        
        # 前向传播 (与原 head 相同接口)
        outputs = e2e_head(bev_embed, ...)
        
        # 额外输出
        interface = e2e_head.extract_planning_interface(outputs)
        value = e2e_head.estimate_value(interface)
    """
    
    def __init__(
        self,
        base_head: nn.Module,
        lora_config: Optional[LoRAConfig] = None,
        enable_value_head: bool = True,
        scene_dim: int = 256,
        plan_len: int = 6,
    ):
        super().__init__()
        self.base_head = base_head
        self.lora_config = lora_config or LoRAConfig()
        self.enable_value_head = enable_value_head
        self.scene_dim = scene_dim
        self.plan_len = plan_len
        
        # 保存原始 head 作为子模块 (自动管理参数)
        self.add_module('base_head', base_head)
        
        # LoRA adapters
        self.lora_modules = nn.ModuleDict()
        if self.lora_config.enabled:
            self._setup_lora()
        
        # Value head
        if self.enable_value_head:
            self.value_head = ValueHead(
                input_dim=scene_dim + plan_len * 2,  # scene + trajectory
                hidden_dim=128,
            )
        
        # 原始参数标记为冻结 (除非 LoRA 启用)
        self._freeze_base_params()
    
    def _freeze_base_params(self):
        """冻结基础参数。"""
        if self.lora_config.enabled:
            # LoRA 模式下冻结原始参数
            for name, param in self.base_head.named_parameters():
                # 跳过 LoRA 模块 (在 self.lora_modules 中)
                if not any(lora_name in name for lora_name in self.lora_modules):
                    param.requires_grad = False
        else:
            # 非 LoRA 模式: 所有参数可训练
            for param in self.base_head.parameters():
                param.requires_grad = True
    
    def _setup_lora(self):
        """为指定模块添加 LoRA。"""
        target_modules = self.lora_config.target_modules
        
        for name, module in self.base_head.named_modules():
            # 检查是否在目标模块中
            if not any(target in name for target in target_modules):
                continue
            
            # 只处理 Linear 层
            if not isinstance(module, nn.Linear):
                continue
            
            # 跳过 LoRA 模块本身
            if 'lora' in name:
                continue
            
            # 创建 LoRA wrapper
            lora_name = name.replace('.', '_')
            self.lora_modules[lora_name] = LoRALinear(
                base_layer=module,
                rank=self.lora_config.rank,
                alpha=self.lora_config.alpha,
                dropout=self.lora_config.dropout,
            )
            
            # 替换原始模块
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = self.base_head.get_submodule(parent_name) if parent_name else self.base_head
            setattr(parent, child_name, self.lora_modules[lora_name])
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """前向传播 - 直接透传到 base_head。"""
        return self.base_head(*args, **kwargs)
    
    def extract_planning_interface(
        self,
        outputs: Dict[str, torch.Tensor],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> Optional[PlanningInterface]:
        """
        从规划头输出提取 PlanningInterface。
        
        Args:
            outputs: base_head 的输出字典
            ego_fut_cmd: 驾驶命令 (可选)
        
        Returns:
            PlanningInterface 或 None
        """
        if PlanningInterface is None:
            return None
        
        # 从 outputs 中提取 scene_token 和 reference_plan
        # 具体实现取决于 base_head 的输出格式
        
        # 尝试常见字段名
        scene_token = outputs.get('scene_token') or outputs.get('bev_embed')
        reference_plan = outputs.get('ego_fut_preds') or outputs.get('trajectory')
        
        if scene_token is None or reference_plan is None:
            return None
        
        # 处理维度
        if scene_token.dim() == 3:  # [B, N, D] -> [B, D]
            scene_token = scene_token.mean(dim=1)
        
        if reference_plan.dim() == 4:  # [B, M, T, 2] -> [B, T, 2]
            # 根据 command 选择模式
            if ego_fut_cmd is not None:
                batch_idx = torch.arange(reference_plan.shape[0], device=reference_plan.device)
                reference_plan = reference_plan[batch_idx, ego_fut_cmd.long()]
            else:
                reference_plan = reference_plan[:, 0]  # 默认选择第一个模式
        
        # cumsum 转为绝对坐标 (如果是位移增量)
        if reference_plan.abs().max() < 10:  # 简单判断是否为增量
            reference_plan = torch.cumsum(reference_plan, dim=1)
        
        return PlanningInterface(
            scene_token=scene_token,
            reference_plan=reference_plan,
            plan_confidence=outputs.get('plan_confidence'),
            candidate_plans=outputs.get('ego_fut_preds'),
        )
    
    def estimate_value(
        self,
        interface: PlanningInterface,
        correction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        估计状态价值 V(s)。
        
        Args:
            interface: PlanningInterface
            correction: 可选的修正量 (用于计算 corrected trajectory)
        
        Returns:
            [B] 状态价值估计
        """
        if not self.enable_value_head:
            return torch.zeros(interface.scene_token.shape[0])
        
        # 构建输入
        scene_feat = interface.scene_token
        
        # 如果有 correction，计算 corrected trajectory
        if correction is not None:
            traj = interface.reference_plan + correction
        else:
            traj = interface.reference_plan
        
        # 展平 trajectory
        traj_flat = traj.flatten(1)  # [B, T*2]
        
        # 拼接
        value_input = torch.cat([scene_feat, traj_flat], dim=-1)
        
        return self.value_head(value_input).squeeze(-1)
    
    def estimate_value_from_trajectory(
        self,
        scene_token: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        直接从 trajectory 估计价值。
        
        Args:
            scene_token: [B, D] 场景特征
            trajectory: [B, T, 2] 轨迹
        
        Returns:
            [B] 状态价值估计
        """
        if not self.enable_value_head:
            return torch.zeros(scene_token.shape[0])
        
        traj_flat = trajectory.flatten(1)
        value_input = torch.cat([scene_token, traj_flat], dim=-1)
        
        return self.value_head(value_input).squeeze(-1)
    
    def compute_log_prob(
        self,
        interface: PlanningInterface,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 trajectory 的对数概率 (用于 PPO)。
        
        这里简化处理，实际实现取决于你的策略分布假设。
        
        Args:
            interface: PlanningInterface
            trajectory: [B, T, 2] 要评估的轨迹
        
        Returns:
            [B] 对数概率
        """
        # 简化为与 reference_plan 的相似度
        ref = interface.reference_plan
        
        # 计算距离
        diff = trajectory - ref
        log_prob = -0.5 * (diff ** 2).sum(dim=(-2, -1))
        
        return log_prob
    
    def get_trainable_parameters(self) -> list:
        """获取可训练参数列表。"""
        trainable = []
        
        # LoRA 参数
        for module in self.lora_modules.values():
            trainable.extend(module.parameters())
        
        # Value head
        if self.enable_value_head:
            trainable.extend(self.value_head.parameters())
        
        return trainable
    
    def merge_lora_weights(self):
        """合并 LoRA 权重到 base_layer。"""
        if not self.lora_config.enabled:
            return
        
        for module in self.lora_modules.values():
            module.merge_weights()
        
        # 解冻所有参数
        for param in self.base_head.parameters():
            param.requires_grad = True


class E2EFinetuningWrapper:
    """
    E2E 微调包装器
    
    包装任意规划头，添加 RL 微调支持。
    
    使用方式:
        # 创建 wrapper
        wrapper = E2EFinetuningWrapper(
            model=planner,
            planning_head=planner.planning_head,
            enable_lora=True,
        )
        
        # 前向传播
        outputs = wrapper(inputs)
        
        # RL 训练
        loss, interface, value = wrapper.compute_rl_loss(outputs, gt)
    """
    
    def __init__(
        self,
        model: nn.Module,
        planning_head: nn.Module,
        lora_config: Optional[LoRAConfig] = None,
        enable_value_head: bool = True,
        scene_dim: int = 256,
        plan_len: int = 6,
    ):
        self.model = model
        self.planning_head = planning_head
        
        # 包装规划头
        self.e2e_head = HydraTrajHeadE2E(
            base_head=planning_head,
            lora_config=lora_config,
            enable_value_head=enable_value_head,
            scene_dim=scene_dim,
            plan_len=plan_len,
        )
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """前向传播。"""
        # 让 model 处理输入
        outputs = self.model(*args, **kwargs)
        return outputs
    
    def extract_interface(self, outputs: Dict) -> PlanningInterface:
        """提取 PlanningInterface。"""
        return self.e2e_head.extract_planning_interface(outputs)
    
    def estimate_value(self, interface: PlanningInterface) -> torch.Tensor:
        """估计状态价值。"""
        return self.e2e_head.estimate_value(interface)
    
    def compute_advantages(
        self,
        trajectory: torch.Tensor,
        gt_trajectory: torch.Tensor,
        value_estimate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 advantages。
        
        使用 reward - value 作为 advantage 的简化版。
        
        Returns:
            (rewards, advantages)
        """
        # 简化的 reward: 基于 FDE
        fde = torch.norm(trajectory[:, -1] - gt_trajectory[:, -1], dim=-1)
        rewards = torch.exp(-fde / 5.0)
        
        # Advantage = reward - value
        advantages = rewards - value_estimate
        
        return rewards, advantages


__all__ = [
    'LoRAConfig',
    'LoRALinear',
    'ValueHead',
    'HydraTrajHeadE2E',
    'E2EFinetuningWrapper',
]
