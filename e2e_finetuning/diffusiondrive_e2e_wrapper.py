"""
DiffusionDrive E2E 微调包装器

将 DiffusionDrive 的 DiffMotionPlanningRefinementModule 包装为支持 Phase 2 端到端微调的版本。

核心增强:
1. 为 DiffMotionPlanningRefinementModule 添加 LoRA
2. 添加 Value Head
3. 输出 PlanningInterface
"""

from __future__ import annotations

import sys
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

# 添加项目路径
PROJECT_ROOT = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from planning_interface.interface import PlanningInterface
except ImportError:
    PlanningInterface = None

try:
    from e2e_finetuning.hydra_traj_head_e2e import LoRALinear, ValueHead
except ImportError:
    LoRALinear = ValueHead = None


@dataclass
class DiffusionDriveE2EConfig:
    """DiffusionDrive E2E 配置。"""
    lora_enabled: bool = True
    lora_rank: int = 16
    lora_alpha: float = 1.0
    lora_dropout: float = 0.1
    
    enable_value_head: bool = True
    value_hidden_dim: int = 128
    
    ego_fut_ts: int = 8
    ego_fut_mode: int = 20
    embed_dims: int = 256


class DiffusionDriveHeadE2E(nn.Module):
    """
    DiffusionDrive 规划头 E2E 包装器
    
    DiffMotionPlanningRefinementModule 结构:
        traj_feature [B, ego_fut_mode, D] → plan_reg [B, ego_fut_mode, ego_fut_ts, 3]
                                           → plan_cls [B, ego_fut_mode]
    
    E2E 结构:
        traj_feature → LoRA(plan_reg_branch) → Δplan_reg
                        ↓
                  Value Head → V(s)
    """
    
    def __init__(
        self,
        base_module: nn.Module,
        config: Optional[DiffusionDriveE2EConfig] = None,
    ):
        super().__init__()
        self.base_module = base_module
        self.cfg = config or DiffusionDriveE2EConfig()
        
        # 获取配置
        self.ego_fut_ts = getattr(base_module, 'ego_fut_ts', 8)
        self.ego_fut_mode = getattr(base_module, 'ego_fut_mode', 20)
        self.embed_dims = getattr(base_module, 'embed_dims', 256)
        
        # 冻结原始 plan_reg_branch
        if hasattr(base_module, 'plan_reg_branch'):
            self.base_plan_reg_branch = base_module.plan_reg_branch
            for param in self.base_plan_reg_branch.parameters():
                param.requires_grad = False
        else:
            self.base_plan_reg_branch = None
        
        # LoRA for plan_reg_branch
        self.lora_plan_reg = None
        if self.cfg.lora_enabled and self.base_plan_reg_branch is not None:
            self._setup_lora()
        
        # Value Head
        self.value_head = None
        if self.cfg.enable_value_head:
            # 输入: traj_feature 或 ego query
            self.value_head = nn.Sequential(
                nn.Linear(self.embed_dims, self.cfg.value_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.value_hidden_dim, 1),
            )
    
    def _setup_lora(self):
        """为 plan_reg_branch 设置 LoRA。"""
        if self.base_plan_reg_branch is None:
            return
        
        # 找到 plan_reg_branch 的最后一个 Linear
        modules = list(self.base_plan_reg_branch.modules())
        output_linear = None
        for m in modules:
            if isinstance(m, nn.Linear):
                output_linear = m
        
        if output_linear is not None:
            self.lora_plan_reg = LoRALinear(
                base_layer=output_linear,
                rank=self.cfg.lora_rank,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
            )
    
    def forward(
        self,
        traj_feature: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        
        Args:
            traj_feature: [B, ego_fut_mode, D] 轨迹特征
        
        Returns:
            dict with:
                - plan_reg: [B, ego_fut_mode, ego_fut_ts, 3] 轨迹参数
                - plan_cls: [B, ego_fut_mode] 分类分数
                - value: [B] 状态价值估计
        """
        bs, ego_fut_mode, _ = traj_feature.shape
        
        # 1. LoRA 增量
        lora_delta = None
        if self.lora_plan_reg is not None:
            # 展平输入
            traj_flat = traj_feature.view(bs, ego_fut_mode, -1).mean(dim=1)  # [B, D]
            
            # LoRA 输出
            lora_out = self.lora_plan_reg(traj_flat.unsqueeze(1))  # [B, 1, ego_fut_ts*3]
            lora_delta = lora_out.view(bs, 1, self.ego_fut_ts, 3)  # [B, 1, T, 3]
        
        # 2. 原始预测
        plan_reg, plan_cls = self.base_module(traj_feature)
        
        # 3. 添加 LoRA 增量
        if lora_delta is not None:
            plan_reg = plan_reg + lora_delta
        
        # 4. Value
        value = None
        if self.value_head is not None:
            traj_mean = traj_feature.mean(dim=1)  # [B, D]
            value = self.value_head(traj_mean).squeeze(-1)  # [B]
        
        return {
            'plan_reg': plan_reg,  # [B, M, T, 3] (x, y, heading)
            'plan_cls': plan_cls,  # [B, M]
            'value': value,
        }
    
    def extract_planning_interface(
        self,
        outputs: Dict[str, torch.Tensor],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> Optional[PlanningInterface]:
        """
        从 DiffusionDrive 输出提取 PlanningInterface。
        
        DiffusionDrive 输出格式:
            plan_reg: [B, M, T, 3] - (x, y, heading) 绝对坐标
            plan_cls: [B, M] - 模式分数
        
        Returns:
            PlanningInterface
        """
        if PlanningInterface is None:
            return None
        
        plan_reg = outputs.get('plan_reg')  # [B, M, T, 3]
        plan_cls = outputs.get('plan_cls')  # [B, M]
        
        if plan_reg is None:
            return None
        
        B = plan_reg.shape[0]
        
        # 1. Scene Token: 使用 plan_cls 权重平均 plan_reg
        if plan_cls is not None:
            # Softmax 权重
            weights = torch.softmax(plan_cls, dim=-1)  # [B, M]
            # 加权平均作为 scene token
            scene_token = (plan_reg[:, :, :, :2].mean(dim=-1) * weights.unsqueeze(-1)).sum(dim=1)  # [B, 2]
            scene_token = torch.cat([scene_token, torch.zeros(B, self.embed_dims - 2, device=scene_token.device)], dim=-1)
        else:
            scene_token = plan_reg[:, 0, :, :2].mean(dim=1)  # [B, 2]
            scene_token = torch.cat([scene_token, torch.zeros(B, self.embed_dims - 2, device=scene_token.device)], dim=-1)
        
        # 2. Reference Plan: 使用最高分模式的轨迹
        if plan_cls is not None:
            best_mode = plan_cls.argmax(dim=-1)  # [B]
            batch_idx = torch.arange(B, device=plan_reg.device)
            reference_plan = plan_reg[batch_idx, best_mode, :, :2]  # [B, T, 2]
        else:
            reference_plan = plan_reg[:, 0, :, :2]  # [B, T, 2]
        
        # 3. Candidate Plans
        candidate_plans = plan_reg[:, :, :, :2]  # [B, M, T, 2]
        
        # 4. Plan Confidence
        plan_confidence = None
        if plan_cls is not None:
            plan_confidence = torch.softmax(plan_cls, dim=-1).max(dim=-1).values.unsqueeze(-1)  # [B, 1]
        
        return PlanningInterface(
            scene_token=scene_token,
            reference_plan=reference_plan,
            candidate_plans=candidate_plans,
            plan_confidence=plan_confidence,
            metadata={'source': 'DiffusionDriveHeadE2E'},
        )
    
    def get_trainable_parameters(self) -> list:
        """获取可训练参数。"""
        params = []
        if self.lora_plan_reg is not None:
            params.extend(self.lora_plan_reg.parameters())
        if self.value_head is not None:
            params.extend(self.value_head.parameters())
        return params


def wrap_diffusiondrive_head(
    dd_module: nn.Module,
    lora_rank: int = 16,
    enable_value_head: bool = True,
) -> DiffusionDriveHeadE2E:
    """便捷函数: 包装 DiffusionDrive 规划头。"""
    config = DiffusionDriveE2EConfig(
        lora_enabled=True,
        lora_rank=lora_rank,
        enable_value_head=enable_value_head,
    )
    return DiffusionDriveHeadE2E(dd_module, config)


__all__ = [
    'DiffusionDriveE2EConfig',
    'DiffusionDriveHeadE2E',
    'wrap_diffusiondrive_head',
]
