"""VAD 专用适配器：将 VAD forward 输出映射到 PlanningInterface。

VAD 输出字典的关键字段:
- bev_embed: [B, bev_h*bev_w, embed_dims]  (200*200=40000, 256)
- ego_fut_preds: [B, ego_fut_mode, fut_ts, 2]  (B, 3, 6, 2) 位移增量
- all_cls_scores: [num_dec, B, num_query, num_cls]  检测分类分数
- all_bbox_preds: [num_dec, B, num_query, 10]  检测框
- all_traj_preds: [num_dec, B, A, fut_mode*fut_ts*2]  Agent 轨迹
- all_traj_cls_scores: [num_dec, B, A, fut_mode]  Agent 轨迹模式分数
- map_all_cls_scores: [num_dec, B, map_num_vec, 3]  地图分类分数
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base_adapter import BasePlanningAdapter


class VADPlanningAdapter(BasePlanningAdapter):
    """VAD → PlanningInterface 适配器。

    Args:
        scene_pool: BEV 特征池化方式，'mean' / 'max' / 'grid' / 'ego_local'
        ego_fut_mode: 自车规划模式数，默认 3
        fut_ts: 未来时间步长数，默认 6
        grid_size: grid 池化的分块数（每个方向），默认 4
        ego_local_k: ego_local 池化的邻域大小，默认 16
    """

    def __init__(
        self,
        scene_pool: str = 'mean',
        ego_fut_mode: int = 3,
        fut_ts: int = 6,
        grid_size: int = 4,
        ego_local_k: int = 16,
    ):
        self.scene_pool = scene_pool
        self.ego_fut_mode = ego_fut_mode
        self.fut_ts = fut_ts
        self.grid_size = grid_size
        self.ego_local_k = ego_local_k

    # ------------------------------------------------------------------
    # scene_token
    # ------------------------------------------------------------------
    def extract_scene_token(
        self,
        planner_outputs: Dict[str, Any],
    ) -> torch.Tensor:
        """从 BEV 特征池化得到场景 token。

        支持的 scene_pool 方式:
        - 'mean':      全局均值池化 → [B, D]
        - 'max':       全局最大池化 → [B, D]
        - 'grid':      空间分块池化 → [B, grid_size^2 * D]，保留粗粒度空间
        - 'ego_local': ego 附近局部 token 均值池化 → [B, D]

        也支持从 dump 数据中提取（interface_mean['token'] 等）。
        """
        # 1. 从 BEV embedding 池化
        if 'bev_embed' in planner_outputs and planner_outputs['bev_embed'] is not None:
            bev = planner_outputs['bev_embed']
            if bev.dim() == 3:
                if bev.shape[0] > bev.shape[1]:
                    bev = bev.permute(1, 0, 2)
            elif bev.dim() == 4:
                bev = bev.flatten(2).permute(0, 2, 1)
            else:
                raise ValueError(f'不支持的 bev_embed 形状: {tuple(bev.shape)}')

            if self.scene_pool == 'mean':
                return bev.mean(dim=1)
            elif self.scene_pool == 'max':
                return bev.max(dim=1).values
            elif self.scene_pool == 'grid':
                return self._grid_pool(bev)
            elif self.scene_pool == 'ego_local':
                return self._ego_local_pool(bev)
            else:
                raise ValueError(f'未知的池化方式: {self.scene_pool}')

        # 2. 从 dump 数据中的 interface_* 字典提取
        for key in ('interface_mean', 'interface_grid', 'interface_ego_local'):
            if key in planner_outputs and planner_outputs[key] is not None:
                interface_dict = planner_outputs[key]
                if isinstance(interface_dict, dict):
                    # dump 数据中可能叫 'scene_token' 或 'token'
                    for token_key in ('scene_token', 'token'):
                        if token_key in interface_dict:
                            token = interface_dict[token_key]
                            if isinstance(token, torch.Tensor):
                                return token

        # 3. 使用 ego 级别特征
        for key in ('ego_feats', 'ego_agent_feat', 'ego_map_feat'):
            if key in planner_outputs and planner_outputs[key] is not None:
                feat = planner_outputs[key]
                if feat.dim() == 3:
                    return feat.squeeze(1)
                return feat

        raise KeyError('无法从 VAD 输出中提取 scene_token')

    # ------------------------------------------------------------------
    # reference_plan
    # ------------------------------------------------------------------
    def extract_reference_plan(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """提取参考轨迹。

        VAD 输出 ego_fut_preds 支持两种格式：
        - 在线推理: [B, M, T, 2] 位移增量
        - dump 数据: [M, T, 2] 位移增量（无 batch 维度）

        根据 ego_fut_cmd 选择对应模式并 cumsum 转为绝对坐标。

        Returns:
            (reference_plan, candidate_plans)
            - reference_plan: [B, T, 2] ego-centric 绝对坐标（或 [T, 2] 如果无 batch）
            - candidate_plans: [B, M, T, 2] 或 [M, T, 2] 位移增量
        """
        if 'ego_fut_preds' not in planner_outputs:
            raise KeyError('VAD 输出中缺少 ego_fut_preds')

        ego_fut_preds = planner_outputs['ego_fut_preds']

        # 判断是否有 batch 维度
        # [M, T, 2] -> dump 数据，无 batch
        # [B, M, T, 2] -> 在线推理，有 batch
        has_batch = ego_fut_preds.dim() == 4

        if not has_batch:
            # dump 数据格式 [M, T, 2] -> 添加 batch 维度 -> [1, M, T, 2]
            ego_fut_preds = ego_fut_preds.unsqueeze(0)

        candidate_plans = ego_fut_preds  # [B, M, T, 2]

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

    # ------------------------------------------------------------------
    # plan_confidence
    # ------------------------------------------------------------------
    def extract_plan_confidence(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """提取规划置信度。

        VAD 没有显式的 ego 规划置信度输出。
        使用候选模式间的方差作为不确定性代理:
        方差越大 → 不确定性越高 → 置信度越低。
        """
        if 'ego_fut_preds' in planner_outputs and planner_outputs['ego_fut_preds'] is not None:
            preds = planner_outputs['ego_fut_preds']
            if preds.dim() == 4 and preds.shape[1] > 1:
                # [B, M, T, 2] → 模式间方差 → [B, 1]
                mode_variance = preds.var(dim=1).mean(dim=(-2, -1))  # [B]
                # 方差越大，置信度越低；用负指数映射到 (0, 1]
                confidence = torch.exp(-mode_variance).unsqueeze(-1)  # [B, 1]
                return confidence

        # 回退：均匀置信度
        ego_fut_preds = planner_outputs.get('ego_fut_preds')
        if ego_fut_preds is not None:
            batch_size = ego_fut_preds.shape[0]
            return torch.ones(
                (batch_size, 1),
                device=ego_fut_preds.device,
                dtype=ego_fut_preds.dtype,
            )
        return None

    # ------------------------------------------------------------------
    # safety_features
    # ------------------------------------------------------------------
    def extract_safety_features(
        self,
        planner_outputs: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """提取安全相关特征。

        从 VAD 输出中收集:
        - plan_mode_variance: 规划模式间方差 [B, T]
        - object_density: 检测物体密度代理 [B, 1]
        """
        safety: Dict[str, torch.Tensor] = {}

        # 1. 规划模式间方差作为不确定性安全信号
        if 'ego_fut_preds' in planner_outputs and planner_outputs['ego_fut_preds'] is not None:
            preds = planner_outputs['ego_fut_preds']
            if preds.dim() == 4 and preds.shape[1] > 1:
                # [B, M, T, 2] → [B, T] 每步方差
                safety['plan_mode_variance'] = preds.var(dim=1).mean(dim=-1)

        # 2. 检测物体置信度摘要
        if 'all_cls_scores' in planner_outputs and planner_outputs['all_cls_scores'] is not None:
            cls_scores = planner_outputs['all_cls_scores']
            # 取最后一层解码器: [B, num_query, num_cls]
            if cls_scores.dim() >= 3:
                last_layer = cls_scores[-1] if cls_scores.dim() == 4 else cls_scores
                # sigmoid 概率的均值作为 object density
                obj_prob = torch.sigmoid(last_layer).max(dim=-1).values  # [B, Q]
                safety['object_density'] = obj_prob.mean(dim=-1, keepdim=True)  # [B, 1]

        # 3. 地图检测分数摘要
        if 'map_all_cls_scores' in planner_outputs and planner_outputs['map_all_cls_scores'] is not None:
            map_scores = planner_outputs['map_all_cls_scores']
            if map_scores.dim() >= 3:
                last_layer = map_scores[-1] if map_scores.dim() == 4 else map_scores
                map_prob = torch.sigmoid(last_layer).max(dim=-1).values
                safety['map_density'] = map_prob.mean(dim=-1, keepdim=True)

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
        - [B, M, 1] 包裹维度
        """
        if ego_fut_cmd is None:
            return torch.zeros((batch_size,), dtype=torch.long)

        cmd = ego_fut_cmd
        # 压缩 DataContainer 嵌套产生的多余维度，如 (1,1,1,3) → (1,3)
        while cmd.dim() > 2 and cmd.shape[0] == 1:
            cmd = cmd.squeeze(0)
        if cmd.dim() > 2:
            cmd = cmd.view(cmd.shape[0], -1)
        if cmd.dim() == 3 and cmd.shape[-1] == 1:
            cmd = cmd.squeeze(-1)
        if cmd.dim() == 2:
            return cmd.float().argmax(dim=-1)
        if cmd.dim() == 1:
            return cmd.long()
        raise ValueError(f'不支持的 ego_fut_cmd 形状: {tuple(cmd.shape)}')

    # ------------------------------------------------------------------
    # scene_token 池化辅助方法
    # ------------------------------------------------------------------
    def _grid_pool(self, bev: torch.Tensor) -> torch.Tensor:
        """空间分块池化，保留粗粒度空间结构。

        Args:
            bev: [B, N, D]

        Returns:
            [B, grid_size^2 * D]
        """
        batch_size, num_tokens, dim = bev.shape
        g = self.grid_size

        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            bev_2d = bev.reshape(batch_size, side, side, dim)
            bh, bw = side // g, side // g
            pooled = []
            for i in range(g):
                for j in range(g):
                    block = bev_2d[:, i*bh:(i+1)*bh, j*bw:(j+1)*bw, :]
                    pooled.append(block.reshape(batch_size, -1, dim).mean(dim=1))
            return torch.cat(pooled, dim=-1)
        else:
            chunk_size = num_tokens // (g * g)
            usable = chunk_size * g * g
            chunks = bev[:, :usable].reshape(batch_size, g * g, chunk_size, dim)
            return chunks.mean(dim=2).reshape(batch_size, -1)

    def _ego_local_pool(self, bev: torch.Tensor) -> torch.Tensor:
        """取 ego 附近的 BEV token 并池化（假设 BEV 中心 = ego）。

        Args:
            bev: [B, N, D]

        Returns:
            [B, D]
        """
        batch_size, num_tokens, dim = bev.shape
        side = int(num_tokens ** 0.5)

        if side * side == num_tokens:
            center = side // 2
            half_k = max(int(self.ego_local_k ** 0.5) // 2, 1)
            bev_2d = bev.reshape(batch_size, side, side, dim)
            r0 = max(center - half_k, 0)
            r1 = min(center + half_k, side)
            c0 = max(center - half_k, 0)
            c1 = min(center + half_k, side)
            local = bev_2d[:, r0:r1, c0:c1, :]
            return local.reshape(batch_size, -1, dim).mean(dim=1)
        else:
            mid = num_tokens // 2
            half = self.ego_local_k // 2
            return bev[:, max(mid-half, 0):min(mid+half, num_tokens)].mean(dim=1)
