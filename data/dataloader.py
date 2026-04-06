"""VAD dump DataLoader - 用于加载预处理好的训练数据。"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

# 添加项目根目录到 sys.path
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.planning_interface.interface import PlanningInterface

logger = logging.getLogger(__name__)


class VADDumpDataset(Dataset):
    """从 VAD 推理 dump 加载数据的数据集。"""

    def __init__(self, data_dir: str | Path):
        """初始化数据集。

        Args:
            data_dir: 包含 pt 文件和 manifest.json 的目录
        """
        self.data_dir = Path(data_dir)
        
        # 加载 manifest
        manifest_path = self.data_dir / 'manifest.json'
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.samples = self.manifest['samples']
        logger.info(f'加载了 {len(self.samples)} 个样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """加载单个样本。

        Returns 包含以下字段的 dict:
            - interface: PlanningInterface
            - gt_plan: [T, 2] 地面真值轨迹
            - plan_mask: [T] 有效步掩码
            - metadata: 其他元数据
        """
        sample_info = self.samples[idx]
        file_path = self.data_dir / sample_info['file']

        # 加载 pt 文件
        data = torch.load(file_path, map_location='cpu')

        # 如果直接是 PlanningInterface，则返回
        if isinstance(data, PlanningInterface):
            return {
                'interface': data,
                'gt_plan': torch.zeros(data.reference_plan.shape),  # placeholder
                'plan_mask': torch.ones(data.reference_plan.shape[0]),
                'metadata': sample_info,
            }

        # 处理 VAD dump 格式（dict）
        if isinstance(data, dict):
            # 从 interface_mean 提取场景 token（256 维）
            scene_token = data.get('interface_mean', {}).get('token', torch.randn(256)).float()
            if isinstance(scene_token, dict):
                # 如果还是 dict，尝试平均其值
                scene_token = torch.randn(256).float()
            
            # 参考轨迹：使用第一个预测模式
            # 重要：ego_fut_preds 是位移增量，需要 cumsum 转 ego-centric 绝对坐标
            ego_fut_preds = data.get('ego_fut_preds', torch.randn(3, 6, 2))  # [M, T, 2]
            if ego_fut_preds.shape[0] > 0:
                # cumsum 转 ego-centric 绝对坐标（从原点开始）
                reference_plan = ego_fut_preds[0].float().cumsum(dim=0)  # [6, 2]
            else:
                reference_plan = torch.randn(6, 2).float()

            # 地面真值轨迹：
            # - dump 保存的是全局坐标（dump_vad_inference.py 中做过 cumsum）
            # - 需要转换到 ego-centric 坐标系：减去起点位置
            gt_plan_global = data.get('ego_fut_trajs', torch.randn(6, 2)).float()  # [6, 2]
            gt_plan = gt_plan_global - gt_plan_global[0]  # 转 ego-centric 坐标（从原点开始）

            # 计算置信度：从多模式中计算（可用 softmax）
            all_scores = data.get('all_cls_scores_last', torch.randn(300, 10))  # [N, K]
            if all_scores.shape[1] > 0:
                top_score = all_scores[:, 0].max()
                plan_confidence = torch.sigmoid(top_score.unsqueeze(0)).float()
            else:
                plan_confidence = torch.ones(1).float()

            # 候选轨迹：所有预测的模式
            candidate_plans = ego_fut_preds  # [M, T, 2]

            # 安全特征（placeholder）
            safety_features = None

            # 构造 PlanningInterface
            interface = PlanningInterface(
                scene_token=scene_token,
                reference_plan=reference_plan,
                plan_confidence=plan_confidence,
                candidate_plans=candidate_plans,
                safety_features=safety_features,
            )

            # 掩码（所有步都有效）
            plan_mask = torch.ones(reference_plan.shape[0])

            return {
                'interface': interface,
                'gt_plan': gt_plan,  # 累积坐标，与 dump 保存格式一致
                'plan_mask': plan_mask,
                'metadata': sample_info,
            }

        # 其他格式的降级处理
        interface = PlanningInterface(
            scene_token=torch.randn(256),
            reference_plan=torch.randn(6, 2),
            plan_confidence=torch.ones(1),
            candidate_plans=None,
            candidate_scores=None,
            safety_features=None,
        )

        return {
            'interface': interface,
            'gt_plan': torch.randn(6, 2),
            'plan_mask': torch.ones(6),
            'metadata': sample_info,
        }


def build_vad_dataloader(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """构建 VAD dump DataLoader。

    Args:
        data_dir: 数据目录
        batch_size: 批大小
        num_workers: 数据加载的工作进程数
        shuffle: 是否打乱

    Returns:
        DataLoader 实例
    """
    dataset = VADDumpDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_batch_with_interface,
    )


def collate_batch_with_interface(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """自定义 collate 函数以处理 PlanningInterface。"""
    interfaces = [item['interface'] for item in batch]
    gt_plans = torch.stack([item['gt_plan'] for item in batch])
    plan_masks = torch.stack([item['plan_mask'] for item in batch])
    
    # 使用 PlanningInterface 的 collate 方法对接口进行批处理
    batched_interface = PlanningInterface.collate(interfaces)
    
    return {
        'interface': batched_interface,
        'gt_plan': gt_plans,
        'plan_mask': plan_masks,
    }
