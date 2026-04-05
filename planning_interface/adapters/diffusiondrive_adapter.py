"""DiffusionDrive 适配器：将 DiffusionDrive 原始输出映射到 PlanningInterface。

不修改 DiffusionDrive 模型代码，仅消费其标准 forward 输出。

DiffusionDrive 输出字典的关键字段:
- trajectory: [B, T, 3]  最优轨迹，绝对坐标 (x, y, heading)，T=8
- agent_states: [B, A, 5]  检测到的车辆状态 (x, y, heading, length, width)
- agent_labels: [B, A]  车辆有效性分数 (logits)
- bev_semantic_map: [B, C, H, W]  BEV 语义分割图 (C=7, H=128, W=256)

可选字段（如果模型暴露了多模态输出）:
- all_poses_reg: [B, M, T, 3]  全部候选轨迹
- all_poses_cls: [B, M]  候选轨迹分类分数
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base_adapter import BasePlanningAdapter


class DiffusionDrivePlanningAdapter(BasePlanningAdapter):
    """DiffusionDrive -> PlanningInterface 适配器。

    核心设计原则：不修改 DiffusionDrive 任何代码，仅消费其标准输出。

    Args:
        scene_pool: BEV 语义图池化方式，'mean' / 'max' / 'grid' / 'flatten'
        fut_ts: 未来时间步长数（DiffusionDrive 默认 8）
        grid_size: grid 池化的分块数（每个方向），默认 4
        bev_semantic_classes: BEV 语义类别数，默认 7
    """

    def __init__(
        self,
        scene_pool: str = 'mean',
        fut_ts: int = 8,
        grid_size: int = 4,
        bev_semantic_classes: int = 7,
    ):
        self.scene_pool = scene_pool
        self.fut_ts = fut_ts
        self.grid_size = grid_size
        self.bev_semantic_classes = bev_semantic_classes

    # ------------------------------------------------------------------
    # scene_token
    # ------------------------------------------------------------------
    def extract_scene_token(
        self,
        planner_outputs: Dict[str, Any],
    ) -> torch.Tensor:
        """从 BEV 语义图池化得到场景 token。

        DiffusionDrive 没有显式的 BEV embedding，
        但其 bev_semantic_map [B, C, H, W] 包含丰富的场景结构信息。
        我们对其进行池化得到紧凑的场景表示。

        支持的 scene_pool 方式:
        - 'mean':    全局均值池化 → [B, C]
        - 'max':     全局最大池化 → [B, C]
        - 'grid':    空间分块池化 → [B, grid_size^2 * C]
        - 'flatten': 展平后全局均值 → [B, C * H_pool * W_pool]
        """
        if 'bev_semantic_map' in planner_outputs and planner_outputs['bev_semantic_map'] is not None:
            bev_map = planner_outputs['bev_semantic_map']  # [B, C, H, W]
            return self._pool_bev_semantic(bev_map)

        # 回退：如果有内部 BEV 特征（模型可选暴露）
        if 'bev_feature' in planner_outputs and planner_outputs['bev_feature'] is not None:
            bev = planner_outputs['bev_feature']
            if bev.dim() == 4:
                # [B, C, H, W] → [B, N, C]
                bev = bev.flatten(2).permute(0, 2, 1)
            if bev.dim() == 3:
                return bev.mean(dim=1)
            return bev

        # 最终回退：从轨迹和检测信息构建伪 scene_token
        return self._build_fallback_scene_token(planner_outputs)

    # ------------------------------------------------------------------
    # reference_plan
    # ------------------------------------------------------------------
    def extract_reference_plan(
        self,
        planner_outputs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """提取参考轨迹。

        DiffusionDrive 输出 trajectory 为 [B, T, 3] 的绝对坐标 (x, y, heading)。
        统一接口需要 [B, T, 2] 的 (x, y) 格式。

        DiffusionDrive 内部已完成模式选择（20 模式中取最优），
        因此 reference_plan 就是最终轨迹。

        Returns:
            (reference_plan, candidate_plans)
            - reference_plan: [B, T, 2] ego-centric 绝对坐标
            - candidate_plans: [B, M, T, 2] 或 None（DiffusionDrive 标准输出无此项）
        """
        if 'trajectory' not in planner_outputs:
            raise KeyError('DiffusionDrive 输出中缺少 trajectory')

        trajectory = planner_outputs['trajectory']  # [B, T, 3]

        # 取 (x, y)，去掉 heading
        reference_plan = trajectory[..., :2]  # [B, T, 2]

        # 候选轨迹：如果模型额外暴露了多模态输出
        candidate_plans = None
        if 'all_poses_reg' in planner_outputs and planner_outputs['all_poses_reg'] is not None:
            all_poses = planner_outputs['all_poses_reg']  # [B, M, T, 3]
            candidate_plans = all_poses[..., :2]  # [B, M, T, 2]

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

        DiffusionDrive 标准输出不包含显式置信度。
        策略:
        1. 如果有多模态分类分数 (all_poses_cls)，用 softmax 提取最优模式置信度
        2. 如果有检测信息，用检测覆盖度作为场景复杂度代理
        3. 回退为均匀置信度
        """
        # 策略 1：多模态分类分数（模型可选暴露）
        if 'all_poses_cls' in planner_outputs and planner_outputs['all_poses_cls'] is not None:
            cls_scores = planner_outputs['all_poses_cls']  # [B, M]
            probs = torch.softmax(cls_scores, dim=-1)
            # 最优模式的概率作为置信度
            confidence = probs.max(dim=-1).values.unsqueeze(-1)  # [B, 1]
            return confidence

        # 策略 2：用检测密度作为场景复杂度代理
        # 高检测密度 → 复杂场景 → 低置信度
        if 'agent_labels' in planner_outputs and planner_outputs['agent_labels'] is not None:
            labels = planner_outputs['agent_labels']  # [B, A]
            # sigmoid 概率，有效检测比例越高表示场景越复杂
            det_ratio = torch.sigmoid(labels).mean(dim=-1, keepdim=True)  # [B, 1]
            # 反转：复杂场景 → 低置信度
            confidence = 1.0 - det_ratio * 0.5  # [B, 1]，范围 [0.5, 1.0]
            return confidence

        # 回退：均匀置信度
        trajectory = planner_outputs.get('trajectory')
        if trajectory is not None:
            batch_size = trajectory.shape[0]
            return torch.ones(
                (batch_size, 1),
                device=trajectory.device,
                dtype=trajectory.dtype,
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

        从 DiffusionDrive 输出中收集:
        - object_density: 检测到的有效物体密度 [B, 1]
        - trajectory_heading_var: 轨迹 heading 变化率 [B, 1]
        - bev_road_coverage: BEV 语义图中道路覆盖比例 [B, 1]
        """
        safety: Dict[str, torch.Tensor] = {}

        # 1. 检测物体密度
        if 'agent_labels' in planner_outputs and planner_outputs['agent_labels'] is not None:
            labels = planner_outputs['agent_labels']  # [B, A]
            obj_prob = torch.sigmoid(labels)  # [B, A]
            safety['object_density'] = obj_prob.mean(dim=-1, keepdim=True)  # [B, 1]

        # 2. 轨迹 heading 变化率作为安全信号
        if 'trajectory' in planner_outputs and planner_outputs['trajectory'] is not None:
            traj = planner_outputs['trajectory']  # [B, T, 3]
            if traj.shape[-1] >= 3:
                heading = traj[..., 2]  # [B, T]
                # 相邻步之间的 heading 变化
                heading_diff = torch.diff(heading, dim=-1).abs()  # [B, T-1]
                safety['heading_change_rate'] = heading_diff.mean(
                    dim=-1, keepdim=True
                )  # [B, 1]

        # 3. BEV 语义图中的道路/障碍物比例
        if 'bev_semantic_map' in planner_outputs and planner_outputs['bev_semantic_map'] is not None:
            bev_map = planner_outputs['bev_semantic_map']  # [B, C, H, W]
            # softmax 后取各类别概率
            bev_probs = torch.softmax(bev_map, dim=1)  # [B, C, H, W]
            # 类别 0 通常是道路（按 DiffusionDrive 定义）
            if bev_probs.shape[1] > 1:
                road_coverage = bev_probs[:, 1].mean(dim=(-2, -1)).unsqueeze(-1)  # [B, 1]
                safety['road_coverage'] = road_coverage
            # 障碍物/车辆密度（类别 4=static_objects, 5=vehicles）
            if bev_probs.shape[1] > 5:
                obstacle_density = bev_probs[:, 4:6].sum(dim=1).mean(
                    dim=(-1, -2)
                ).unsqueeze(-1)  # [B, 1]
                safety['obstacle_density'] = obstacle_density

        return safety if safety else None

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _pool_bev_semantic(self, bev_map: torch.Tensor) -> torch.Tensor:
        """对 BEV 语义图进行池化，得到场景 token。

        Args:
            bev_map: [B, C, H, W] BEV 语义分割 logits

        Returns:
            场景 token 张量
        """
        batch_size, channels, height, width = bev_map.shape

        if self.scene_pool == 'mean':
            # 全局均值池化 → [B, C]
            return bev_map.mean(dim=(-2, -1))

        elif self.scene_pool == 'max':
            # 全局最大池化 → [B, C]
            return bev_map.amax(dim=(-2, -1))

        elif self.scene_pool == 'grid':
            # 分块池化，保留粗粒度空间结构 → [B, grid_size^2 * C]
            g = self.grid_size
            bh = height // g
            bw = width // g
            pooled = []
            for i in range(g):
                for j in range(g):
                    block = bev_map[
                        :, :,
                        i * bh:(i + 1) * bh,
                        j * bw:(j + 1) * bw,
                    ]  # [B, C, bh, bw]
                    pooled.append(block.mean(dim=(-2, -1)))  # [B, C]
            return torch.cat(pooled, dim=-1)  # [B, g^2 * C]

        elif self.scene_pool == 'flatten':
            # 先下采样再展平 → [B, C * 4 * 8] (适中维度)
            import torch.nn.functional as F
            down = F.adaptive_avg_pool2d(bev_map, (4, 8))  # [B, C, 4, 8]
            return down.flatten(1)  # [B, C * 32]

        else:
            raise ValueError(f'未知的池化方式: {self.scene_pool}')

    def _build_fallback_scene_token(
        self, planner_outputs: Dict[str, Any],
    ) -> torch.Tensor:
        """当 BEV 特征不可用时，从轨迹和检测构建伪 scene_token。

        拼接: trajectory_flat + agent_summary → 线性映射到固定维度。
        """
        parts = []

        if 'trajectory' in planner_outputs:
            traj = planner_outputs['trajectory']  # [B, T, 3]
            parts.append(traj.flatten(1))  # [B, T*3]

        if 'agent_states' in planner_outputs and 'agent_labels' in planner_outputs:
            states = planner_outputs['agent_states']  # [B, A, 5]
            labels = planner_outputs['agent_labels']  # [B, A]
            # 按置信度加权的 agent 状态摘要
            weights = torch.sigmoid(labels).unsqueeze(-1)  # [B, A, 1]
            weighted_summary = (states * weights).mean(dim=1)  # [B, 5]
            parts.append(weighted_summary)

        if not parts:
            raise KeyError('无法从 DiffusionDrive 输出中提取任何 scene_token')

        return torch.cat(parts, dim=-1)  # [B, D_fallback]
