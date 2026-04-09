"""SparseDriveV2 专用适配器：将 SparseDriveV2 forward 输出映射到 PlanningInterface。

SparseDriveV2 输出字典的关键字段（从 motion_plan_head）:
- motion_output: dict，包含 agent 运动预测
  - classification: [num_dec, B, A, fut_mode]  Agent 轨迹模式分数
  - prediction: [num_dec, B, A, fut_mode, fut_ts, 2]  Agent 轨迹
- planning_output: dict，包含自车规划
  - classification: [num_dec, B, M]  规划模式分数
  - prediction: [num_dec, B, M, T, 2]  规划轨迹（位移增量）
  - status: [num_dec, B, 1]  规划状态

注意:
- ego_fut_mode = 6（SparseDriveV2 有 6 个规划模式）
- ego_fut_ts = 6（6 个时间步）
- 轨迹格式：ego-centric 位移增量（需要 cumsum 转绝对坐标）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base_adapter import BasePlanningAdapter


class SparseDriveV2PlanningAdapter(BasePlanningAdapter):
    """SparseDriveV2 → PlanningInterface 适配器。

    Args:
        scene_pool: 场景特征池化方式，'mean' / 'instance' / 'det_weighted'
        ego_fut_mode: 自车规划模式数，默认 6（SparseDriveV2）
        fut_ts: 未来时间步长数，默认 6
    """

    def __init__(
        self,
        scene_pool: str = 'mean',
        ego_fut_mode: int = 6,
        fut_ts: int = 6,
    ):
        self.scene_pool = scene_pool
        self.ego_fut_mode = ego_fut_mode
        self.fut_ts = fut_ts

    # ------------------------------------------------------------------
    # scene_token
    # ------------------------------------------------------------------
    def extract_scene_token(
        self,
        planner_outputs: Dict[str, Any],
    ) -> torch.Tensor:
        """从 SparseDriveV2 输出中提取场景 token。

        SparseDriveV2 使用 sparse instance 而非密集 BEV，有多种提取方式:
        
        方案 1（推荐）: 从 dump 数据中的 interface_* 字典提取（如果已预计算）
        方案 2: 从 instance 特征池化（需要 motion_output 中有 instance_feat）
        方案 3: 从检测输出中加权池化（使用分类分数）
        方案 4: 使用默认随机特征（调试用）

        Returns:
            scene_token: [B, D] 或 [D]
        """
        # 1. 从 dump 数据中的 interface_* 字典提取
        for key in ('interface_mean', 'interface_grid', 'interface_ego_local'):
            if key in planner_outputs and planner_outputs[key] is not None:
                interface_dict = planner_outputs[key]
                if isinstance(interface_dict, dict):
                    for token_key in ('scene_token', 'token'):
                        if token_key in interface_dict:
                            token = interface_dict[token_key]
                            if isinstance(token, torch.Tensor):
                                return token

        # 2. 从 instance 特征池化（如果 motion_output 中有）
        if 'motion_output' in planner_outputs and planner_outputs['motion_output'] is not None:
            motion_out = planner_outputs['motion_output']
            
            # 尝试从 instance_feat 或 query_feat 中提取
            for feat_key in ('instance_feat', 'query_feat', 'feat'):
                if feat_key in motion_out:
                    feat = motion_out[feat_key]
                    if isinstance(feat, torch.Tensor):
                        # [B, N, D] -> [B, D]
                        if feat.dim() == 3:
                            return feat.mean(dim=1)
                        elif feat.dim() == 2:
                            return feat

        # 3. 从检测输出中加权池化
        if 'det_output' in planner_outputs and planner_outputs['det_output'] is not None:
            det_out = planner_outputs['det_output']
            
            # 如果有 classification 和 instance feature
            if 'classification' in det_out and 'instance_feat' in det_out:
                cls_scores = det_out['classification']
                instance_feat = det_out['instance_feat']
                
                if isinstance(cls_scores, torch.Tensor) and isinstance(instance_feat, torch.Tensor):
                    # 使用分类分数加权
                    if cls_scores.dim() >= 2:
                        weights = cls_scores.max(dim=-1).values.softmax(dim=-1)  # [B, N]
                        if weights.dim() == 2 and instance_feat.dim() == 3:
                            return (weights.unsqueeze(-1) * instance_feat).sum(dim=1)  # [B, D]

        # 4. 使用规划输出中的特征（备选）
        if 'planning_output' in planner_outputs and planner_outputs['planning_output'] is not None:
            plan_out = planner_outputs['planning_output']
            if 'status' in plan_out:
                status = plan_out['status']
                if isinstance(status, torch.Tensor):
                    # [B, 1] -> 扩展为 [B, D]
                    if status.dim() == 2:
                        return status.expand(-1, 256)  # 默认 256 维

        # 5. 默认：返回随机特征（仅用于调试）
        logger.warning('无法从 SparseDriveV2 输出中提取 scene_token，使用随机特征')
        return torch.randn(256)

    # ------------------------------------------------------------------
    # reference_plan
    # ------------------------------------------------------------------
    def extract_reference_plan(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """提取参考轨迹。

        SparseDriveV2 planning_output 格式:
        - prediction: [num_dec, B, M, T, 2] 或 [B, M, T, 2] 位移增量
        - classification: [num_dec, B, M] 或 [B, M] 模式分数

        根据 ego_fut_cmd 选择对应模式并 cumsum 转为绝对坐标。

        Returns:
            (reference_plan, candidate_plans)
            - reference_plan: [B, T, 2] ego-centric 绝对坐标
            - candidate_plans: [B, M, T, 2] 所有候选轨迹
        """
        # 从 planning_output 中提取
        if 'planning_output' in planner_outputs and planner_outputs['planning_output'] is not None:
            plan_out = planner_outputs['planning_output']
            
            if 'prediction' not in plan_out:
                raise KeyError('SparseDriveV2 planning_output 中缺少 prediction')
            
            preds = plan_out['prediction']
            
            # 处理多解码器输出：取最后一层
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            
            # 判断是否有 batch 维度
            # [M, T, 2] -> dump 数据，无 batch
            # [B, M, T, 2] -> 在线推理，有 batch
            has_batch = preds.dim() == 4
            
            if not has_batch:
                # dump 数据格式 [M, T, 2] -> 添加 batch 维度 -> [1, M, T, 2]
                preds = preds.unsqueeze(0)
            
            candidate_plans = preds  # [B, M, T, 2]
            
            # 根据 command 选择模式
            selected_idx = self._resolve_command_index(
                ego_fut_cmd, candidate_plans.shape[0], candidate_plans.shape[1]
            )
            
            # 取出对应模式的位移增量
            batch_idx = torch.arange(
                candidate_plans.shape[0], device=selected_idx.device
            )
            reference_deltas = candidate_plans[batch_idx, selected_idx]  # [B, T, 2]
            
            # cumsum 转为 ego-centric 绝对坐标
            reference_plan = torch.cumsum(reference_deltas, dim=-2)  # [B, T, 2]
            
            return reference_plan, candidate_plans

        # 也支持从顶层字段提取（dump 数据格式）
        if 'ego_fut_preds' in planner_outputs:
            ego_fut_preds = planner_outputs['ego_fut_preds']
            
            # 判断是否有 batch 维度
            has_batch = ego_fut_preds.dim() == 4
            
            if not has_batch:
                # dump 数据格式 [M, T, 2] -> 添加 batch 维度
                ego_fut_preds = ego_fut_preds.unsqueeze(0)
            
            candidate_plans = ego_fut_preds  # [B, M, T, 2]
            
            # 根据 command 选择模式
            selected_idx = self._resolve_command_index(
                ego_fut_cmd, candidate_plans.shape[0], candidate_plans.shape[1]
            )
            
            batch_idx = torch.arange(
                candidate_plans.shape[0], device=selected_idx.device
            )
            reference_deltas = candidate_plans[batch_idx, selected_idx]  # [B, T, 2]
            
            # cumsum 转为 ego-centric 绝对坐标
            reference_plan = torch.cumsum(reference_deltas, dim=-2)  # [B, T, 2]
            
            return reference_plan, candidate_plans

        raise KeyError('无法从 SparseDriveV2 输出中提取 reference_plan')

    # ------------------------------------------------------------------
    # plan_confidence
    # ------------------------------------------------------------------
    def extract_plan_confidence(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """提取规划置信度。

        使用 planning classification scores 计算置信度:
        - 选择模式的分类概率 → 置信度
        """
        # 从 planning_output 中提取
        if 'planning_output' in planner_outputs and planner_outputs['planning_output'] is not None:
            plan_out = planner_outputs['planning_output']
            
            if 'classification' not in plan_out:
                return None
            
            cls_scores = plan_out['classification']
            
            # 处理多解码器输出：取最后一层
            if isinstance(cls_scores, (list, tuple)):
                cls_scores = cls_scores[-1]
            
            # [B, M] -> 对 M 维度 softmax -> 取最大值
            if cls_scores.dim() == 2:
                probs = torch.softmax(cls_scores, dim=-1)  # [B, M]
                confidence = probs.max(dim=-1, keepdim=True).values  # [B, 1]
                return confidence
            elif cls_scores.dim() == 1:
                # dump 数据格式 [M]
                probs = torch.softmax(cls_scores, dim=-1)
                confidence = probs.max(dim=-1, keepdim=True).values.unsqueeze(0)  # [1, 1]
                return confidence

        # 也支持从顶层字段提取
        if 'planning_cls_scores' in planner_outputs:
            cls_scores = planner_outputs['planning_cls_scores']
            
            if cls_scores.dim() == 1:
                # [M] -> 添加 batch 维度
                cls_scores = cls_scores.unsqueeze(0)
            
            if cls_scores.dim() == 2:
                probs = torch.softmax(cls_scores, dim=-1)
                confidence = probs.max(dim=-1, keepdim=True).values
                return confidence

        # 如果没有分类分数，使用模式间方差作为不确定性
        if 'ego_fut_preds' in planner_outputs and planner_outputs['ego_fut_preds'] is not None:
            preds = planner_outputs['ego_fut_preds']
            
            has_batch = preds.dim() == 4
            if not has_batch:
                preds = preds.unsqueeze(0)
            
            if preds.shape[1] > 1:
                # [B, M, T, 2] → 模式间方差 → [B]
                mode_variance = preds.var(dim=1).mean(dim=(-2, -1))
                confidence = torch.exp(-mode_variance).unsqueeze(-1)
                return confidence

        return None

    # ------------------------------------------------------------------
    # safety_features
    # ------------------------------------------------------------------
    def extract_safety_features(
        self,
        planner_outputs: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """提取安全相关特征。

        从 SparseDriveV2 输出中收集:
        - plan_mode_variance: 规划模式间方差
        - object_density: 检测物体密度代理
        """
        safety: Dict[str, torch.Tensor] = {}

        # 1. 规划模式间方差作为不确定性安全信号
        if 'ego_fut_preds' in planner_outputs and planner_outputs['ego_fut_preds'] is not None:
            preds = planner_outputs['ego_fut_preds']
            
            has_batch = preds.dim() == 4
            if not has_batch:
                preds = preds.unsqueeze(0)
            
            if preds.dim() == 4 and preds.shape[1] > 1:
                # [B, M, T, 2] → [B, T] 每步方差
                safety['plan_mode_variance'] = preds.var(dim=1).mean(dim=-1)

        # 2. 检测物体置信度摘要（如果有 det_output）
        if 'det_output' in planner_outputs and planner_outputs['det_output'] is not None:
            det_out = planner_outputs['det_output']
            
            if 'classification' in det_out:
                cls_scores = det_out['classification']
                
                # 处理多解码器输出
                if isinstance(cls_scores, (list, tuple)):
                    cls_scores = cls_scores[-1]
                
                # [B, N, C] → object density
                if cls_scores.dim() >= 3:
                    obj_prob = torch.sigmoid(cls_scores).max(dim=-1).values
                    safety['object_density'] = obj_prob.mean(dim=-1, keepdim=True)

        # 3. 规划分类分数摘要
        if 'planning_cls_scores' in planner_outputs:
            cls_scores = planner_outputs['planning_cls_scores']
            
            if cls_scores.dim() == 1:
                cls_scores = cls_scores.unsqueeze(0)
            
            if cls_scores.dim() == 2:
                plan_confidence = torch.softmax(cls_scores, dim=-1).max(dim=-1).values
                safety['planning_confidence'] = plan_confidence

        return safety if safety else None

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_command_index(
        ego_fut_cmd: Optional[torch.Tensor],
        batch_size: int,
        num_modes: int,
    ) -> torch.Tensor:
        """将 ego_fut_cmd 转化为模式索引。

        ego_fut_cmd 可能是:
        - None → 默认选择模式 0
        - [B] 整数索引
        - [B, M] one-hot 或浮点权重
        """
        if ego_fut_cmd is None:
            return torch.zeros((batch_size,), dtype=torch.long)

        cmd = ego_fut_cmd
        # 压缩多余维度
        while cmd.dim() > 2 and cmd.shape[0] == 1:
            cmd = cmd.squeeze(0)
        
        if cmd.dim() == 2:
            return cmd.float().argmax(dim=-1)
        if cmd.dim() == 1:
            return cmd.long()
        
        raise ValueError(f'不支持的 ego_fut_cmd 形状: {tuple(cmd.shape)}')
