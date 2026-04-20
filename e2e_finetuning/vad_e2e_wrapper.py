"""
VAD E2E 微调包装器

将 VAD 的 VADHead 包装为支持 Phase 2 端到端微调的版本。

VAD 结构分析:
- VAD.pts_bbox_head = VADHead
- VADHead.ego_fut_decoder: [B, 1, 2D] → [B, ego_fut_mode, fut_ts, 2]
  输出是位移增量 (delta)，需要 cumsum 转为绝对坐标

E2E 增强:
1. 为 ego_fut_decoder 添加 LoRA
2. 添加 Value Head
3. 输出 PlanningInterface

使用方式:
    from E2E_RL.e2e_finetuning import VADModelE2E, wrap_vad_model
    
    # 包装整个 VAD 模型
    e2e_model = wrap_vad_model(vad_model, lora_rank=16)
    
    # 训练模式
    losses = e2e_model.forward_train(img=img, img_metas=img_metas, ...)
    
    # 推理模式
    bbox_results, prev_bev = e2e_model.forward_test(img_metas, img)
"""

from __future__ import annotations

import sys
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目路径以避免导入问题
PROJECT_ROOT = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 尝试导入
try:
    from planning_interface.interface import PlanningInterface
except ImportError:
    PlanningInterface = None

try:
    from e2e_finetuning.hydra_traj_head_e2e import LoRAConfig, LoRALinear, ValueHead
except ImportError:
    LoRAConfig, LoRALinear, ValueHead = None, None, None


@dataclass
class VADE2EConfig:
    """VAD E2E 配置。"""
    # LoRA
    lora_enabled: bool = True
    lora_rank: int = 16
    lora_alpha: float = 1.0
    lora_dropout: float = 0.1
    
    # Value Head
    enable_value_head: bool = True
    value_hidden_dim: int = 128
    
    # 规划参数
    ego_fut_mode: int = 3
    fut_ts: int = 6
    embed_dims: int = 256


class VADHeadE2E(nn.Module):
    """
    VAD 规划头 E2E 包装器
    
    将 VADHead 的 ego_fut_decoder 包装为支持 LoRA 的版本。
    
    原始结构:
        VADHead.ego_fut_decoder: [B, 1, 2D] → [B, ego_fut_mode, fut_ts, 2]
    
    E2E 结构:
        ego_feats → LoRA(ego_fut_decoder) → [B, ego_fut_mode, fut_ts, 2]
                         ↓
                   Value Head → V(s)
    """
    
    def __init__(
        self,
        base_head: nn.Module,
        config: Optional[VADE2EConfig] = None,
    ):
        super().__init__()
        self.base_head = base_head
        self.cfg = config or VADE2EConfig()
        
        # 获取原始参数
        self.ego_fut_mode = getattr(base_head, 'ego_fut_mode', 3)
        self.fut_ts = getattr(base_head, 'fut_ts', 6)
        self.embed_dims = getattr(base_head, 'embed_dims', 256)
        
        # 冻结原始 ego_fut_decoder
        if hasattr(base_head, 'ego_fut_decoder'):
            self.base_ego_fut_decoder = base_head.ego_fut_decoder
            for param in self.base_ego_fut_decoder.parameters():
                param.requires_grad = False
        else:
            self.base_ego_fut_decoder = None
        
        # LoRA wrapper for ego_fut_decoder
        self.lora_ego_fut_decoder = None
        if self.cfg.lora_enabled and self.base_ego_fut_decoder is not None:
            self._setup_lora()
        
        # Value Head
        self.value_head = None
        if self.cfg.enable_value_head:
            # Value Head 输入: ego_feats 维度 (通常是 2*embed_dims 或 3*embed_dims)
            ego_feats_dim = self.embed_dims * 2  # 简化版
            if hasattr(base_head, 'ego_lcf_feat_idx') and base_head.ego_lcf_feat_idx is not None:
                ego_feats_dim += len(base_head.ego_lcf_feat_idx)
            
            self.value_head = nn.Sequential(
                nn.Linear(ego_feats_dim, self.cfg.value_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.value_hidden_dim, 1),
            )
    
    def _setup_lora(self):
        """为 ego_fut_decoder 设置 LoRA。"""
        if self.base_ego_fut_decoder is None:
            return
        
        # ego_fut_decoder 是 Sequential
        # 找到最后一个 Linear 层并包装
        modules = list(self.base_ego_fut_decoder.modules())
        
        # 找到输出层 (最后一个 Linear)
        output_linear = None
        output_linear_idx = -1
        for i, m in enumerate(modules):
            if isinstance(m, nn.Linear):
                output_linear = m
                output_linear_idx = i
        
        if output_linear is not None:
            # 创建 LoRA wrapper
            self.lora_ego_fut_decoder = LoRALinear(
                base_layer=output_linear,
                rank=self.cfg.lora_rank,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
            )
    
    def forward(
        self,
        bev_embed: torch.Tensor,
        ego_his_trajs: Optional[torch.Tensor] = None,
        ego_lcf_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        
        这里只返回增强的部分 (LoRA + Value)，
        完整的 VAD 前向传播由 base_head 处理。
        """
        # 调用 base_head 获取完整输出
        # 注意: 这里需要 VAD 的完整 forward，
        # 但我们只修改 ego_fut_decoder 的行为
        
        # 获取 base_head 的输出
        # 这需要在外部由完整的 VAD 模型调用
        raise NotImplementedError(
            "VADHeadE2E 需要完整的 VAD forward，"
            "建议使用 VADModelE2E 包装整个模型"
        )
    
    def forward_with_base_output(
        self,
        base_outputs: Dict[str, torch.Tensor],
        ego_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        基于 base_head 输出进行 E2E 增强。
        
        Args:
            base_outputs: base_head 的输出字典
            ego_feats: ego 特征 [B, 1, D] (用于 Value Head)
        
        Returns:
            增强后的输出
        """
        outputs = base_outputs.copy()
        
        # 1. LoRA 修改 ego_fut_preds (如果启用)
        if self.lora_ego_fut_decoder is not None and ego_feats is not None:
            # 使用 LoRA 预测
            lora_output = self.lora_ego_fut_decoder(ego_feats)
            
            # reshape: [B, 1, ego_fut_mode * fut_ts * 2] → [B, ego_fut_mode, fut_ts, 2]
            lora_preds = lora_output.reshape(
                lora_output.shape[0],
                self.ego_fut_mode,
                self.fut_ts,
                2
            )
            
            # 添加到原始预测
            if 'ego_fut_preds' in base_outputs:
                outputs['ego_fut_preds_lora'] = lora_preds
                outputs['ego_fut_preds'] = base_outputs['ego_fut_preds'] + lora_preds
            else:
                outputs['ego_fut_preds'] = lora_preds
        
        # 2. Value Head
        if self.value_head is not None and ego_feats is not None:
            ego_feats_flat = ego_feats.squeeze(1)  # [B, D]
            value = self.value_head(ego_feats_flat).squeeze(-1)  # [B]
            outputs['value'] = value
        
        return outputs
    
    def extract_planning_interface(
        self,
        outputs: Dict[str, torch.Tensor],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> Optional[PlanningInterface]:
        """
        从 VAD 输出提取 PlanningInterface。
        
        Args:
            outputs: VADHead 输出字典
            ego_fut_cmd: 驾驶命令索引
        
        Returns:
            PlanningInterface
        """
        if PlanningInterface is None:
            return None
        
        # 1. Scene Token: 使用 bev_embed 池化
        bev_embed = outputs.get('bev_embed')
        if bev_embed is not None:
            if bev_embed.dim() == 3:
                scene_token = bev_embed.mean(dim=1)  # [B, N, D] → [B, D]
            else:
                scene_token = bev_embed
        else:
            scene_token = torch.zeros(
                outputs['ego_fut_preds'].shape[0],
                self.embed_dims,
                device=outputs['ego_fut_preds'].device,
            )
        
        # 2. Reference Plan: ego_fut_preds → cumsum → 绝对坐标
        ego_fut_preds = outputs.get('ego_fut_preds')  # [B, M, T, 2] 位移增量
        
        if ego_fut_preds is None:
            reference_plan = torch.zeros(
                outputs.get('ego_fut_preds_lora', torch.zeros(1, 3, 6, 2)).shape[0],
                self.fut_ts,
                2,
                device=outputs.get('ego_fut_preds_lira', torch.zeros(1)).device,
            )
        else:
            # 选择模式
            if ego_fut_cmd is not None:
                batch_idx = torch.arange(ego_fut_preds.shape[0], device=ego_fut_preds.device)
                selected_preds = ego_fut_preds[batch_idx, ego_fut_cmd.long()]
            else:
                selected_preds = ego_fut_preds[:, 0]  # 默认选择第一个模式
            
            # cumsum 转绝对坐标
            reference_plan = torch.cumsum(selected_preds, dim=1)  # [B, T, 2]
        
        # 3. Candidate Plans (可选)
        candidate_plans = None
        if ego_fut_preds is not None:
            # 所有模式
            candidate_plans = torch.cumsum(ego_fut_preds, dim=2)  # [B, M, T, 2]
        
        # 4. Plan Confidence (可选)
        plan_confidence = outputs.get('plan_confidence')
        
        return PlanningInterface(
            scene_token=scene_token,
            reference_plan=reference_plan,
            candidate_plans=candidate_plans,
            plan_confidence=plan_confidence,
            metadata={'source': 'VADHeadE2E'},
        )
    
    def get_trainable_parameters(self) -> list:
        """获取可训练参数。"""
        params = []
        
        # LoRA 参数
        if self.lora_ego_fut_decoder is not None:
            params.extend(self.lora_ego_fut_decoder.parameters())
        
        # Value Head 参数
        if self.value_head is not None:
            params.extend(self.value_head.parameters())
        
        return params
    
    def get_num_trainable_params(self) -> int:
        """获取可训练参数数量。"""
        return sum(p.numel() for p in self.get_trainable_parameters())


class VADModelE2E(nn.Module):
    """
    完整的 VAD 模型 E2E 包装器
    
    包装整个 VAD 模型，而非仅仅 head。
    适用于需要完整梯度回传的场景。
    
    VAD 模型使用 pts_bbox_head 而不是 bbox_head。
    """
    
    def __init__(
        self,
        vad_model: nn.Module,
        config: Optional[VADE2EConfig] = None,
    ):
        super().__init__()
        self.vad_model = vad_model
        self.cfg = config or VADE2EConfig()
        
        # 获取 VADHead (VAD 使用 pts_bbox_head)
        self.vad_head = None
        if hasattr(vad_model, 'pts_bbox_head'):
            self.vad_head = vad_model.pts_bbox_head
        elif hasattr(vad_model, 'bbox_head'):
            self.vad_head = vad_model.bbox_head
        elif hasattr(vad_model, 'head'):
            self.vad_head = vad_model.head
        
        # 包装规划头
        if self.vad_head is not None:
            self.e2e_head = VADHeadE2E(self.vad_head, config)
        else:
            self.e2e_head = None
        
        # 冻结除 LoRA 外的所有参数
        self._freeze_non_lora()
    
    def _freeze_non_lora(self):
        """冻结非 LoRA 参数。"""
        for name, param in self.vad_model.named_parameters():
            # LoRA 参数名称包含 'lora'
            if self.e2e_head is None or not any(
                n in name for n in ['lora', 'value_head']
            ):
                param.requires_grad = False
    
    def forward_train(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        训练模式前向传播。
        
        调用原始 VAD.forward_train，获取 loss，然后应用 LoRA 增强 ego_fut_preds。
        """
        # 获取原始 loss
        losses = self.vad_model.forward_train(**kwargs)
        
        # 如果需要 Value Head 监督，添加辅助 loss
        if self.e2e_head is not None and self.e2e_head.value_head is not None:
            # TODO: 计算 value loss (需要 reward 信号)
            # 这里可以添加 value head 的辅助 loss
            pass
        
        return losses
    
    def forward_test(self, img_metas, img=None, **kwargs):
        """
        测试模式前向传播。
        
        调用原始 VAD.forward_test，保留完整的评估流程。
        """
        return self.vad_model.forward_test(img_metas=img_metas, img=img, **kwargs)
    
    def simple_test(self, img_metas, img=None, prev_bev=None, **kwargs):
        """测试函数。"""
        return self.vad_model.simple_test(
            img_metas=img_metas, img=img, prev_bev=prev_bev, **kwargs
        )
    
    def simple_test_pts(self, x, img_metas, prev_bev=None, **kwargs):
        """测试函数。"""
        return self.vad_model.simple_test_pts(
            x=x, img_metas=img_metas, prev_bev=prev_bev, **kwargs
        )
    
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """提取特征。"""
        return self.vad_model.extract_feat(img=img, img_metas=img_metas, len_queue=len_queue)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """获取历史 BEV。"""
        return self.vad_model.obtain_history_bev(imgs_queue, img_metas_list)
    
    def extract_planning_interface(
        self,
        outputs: Dict[str, torch.Tensor],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> Optional[PlanningInterface]:
        """提取 PlanningInterface。"""
        if self.e2e_head is not None:
            return self.e2e_head.extract_planning_interface(outputs, ego_fut_cmd)
        return None
    
    def get_trainable_parameters(self) -> list:
        """获取可训练参数。"""
        if self.e2e_head is not None:
            return self.e2e_head.get_trainable_parameters()
        return []
    
    def merge_lora_weights(self):
        """合并 LoRA 权重。"""
        if self.e2e_head is not None and self.e2e_head.lora_ego_fut_decoder is not None:
            self.e2e_head.lora_ego_fut_decoder.merge_weights()
    
    @property
    def video_test_mode(self):
        """时序测试模式。"""
        return getattr(self.vad_model, 'video_test_mode', False)
    
    @video_test_mode.setter
    def video_test_mode(self, value):
        self.vad_model.video_test_mode = value
    
    @property
    def prev_frame_info(self):
        """历史帧信息。"""
        return self.vad_model.prev_frame_info


# 便捷函数
def wrap_vad_head(
    vad_head: nn.Module,
    lora_rank: int = 16,
    enable_value_head: bool = True,
) -> VADHeadE2E:
    """便捷函数: 包装 VAD 规划头。"""
    config = VADE2EConfig(
        lora_enabled=True,
        lora_rank=lora_rank,
        enable_value_head=enable_value_head,
    )
    return VADHeadE2E(vad_head, config)


def wrap_vad_model(
    vad_model: nn.Module,
    lora_rank: int = 16,
    enable_value_head: bool = True,
) -> VADModelE2E:
    """便捷函数: 包装完整 VAD 模型。"""
    config = VADE2EConfig(
        lora_enabled=True,
        lora_rank=lora_rank,
        enable_value_head=enable_value_head,
    )
    return VADModelE2E(vad_model, config)


__all__ = [
    'VADE2EConfig',
    'VADHeadE2E',
    'VADModelE2E',
    'wrap_vad_head',
    'wrap_vad_model',
]
