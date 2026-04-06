"""通用 Planner DataLoader - 支持所有规划模型的统一数据加载。

核心设计原则：
1. Dataset 只负责加载原始数据，不做坐标系转换
2. Adapter 负责坐标系转换和数据格式化
3. 新模型只需要创建新的 Adapter，不需要修改 dataloader

使用方式：
    from E2E_RL.data.dataloader import build_planner_dataloader

    # VAD
    loader = build_planner_dataloader('data/vad_dumps', adapter_type='vad')

    # DiffusionDrive
    loader = build_planner_dataloader('data/diffusion_dumps', adapter_type='diffusiondrive')
"""

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


# =============================================================================
# Adapter 注册表
# =============================================================================
def _get_adapter_class(adapter_type: str):
    """根据 adapter_type 获取对应的 Adapter 类。"""
    from E2E_RL.planning_interface.adapters.vad_adapter import VADPlanningAdapter
    from E2E_RL.planning_interface.adapters.diffusiondrive_adapter import (
        DiffusionDrivePlanningAdapter,
    )

    adapter_map = {
        'vad': VADPlanningAdapter,
        'diffusiondrive': DiffusionDrivePlanningAdapter,
        'diffusion_drive': DiffusionDrivePlanningAdapter,
    }

    if adapter_type not in adapter_map:
        raise ValueError(
            f'未知的 adapter_type: {adapter_type}，可用: {list(adapter_map.keys())}'
        )

    return adapter_map[adapter_type]


# =============================================================================
# 通用 Dataset
# =============================================================================
class PlannerDumpDataset(Dataset):
    """通用规划器 dump 数据集。

    职责：
    1. 加载原始数据文件
    2. 返回原始数据和元信息

    注意：
    - Dataset 不做坐标系转换，转换由 Adapter 处理
    - GT 坐标系由 dump 时的设置决定，Adapter 负责转换

    Args:
        data_dir: 包含 pt 文件和 manifest.json 的目录
        adapter_type: 适配器类型 ('vad', 'diffusiondrive', ...)
        gt_in_ego_frame: GT 是否已在 ego-centric 坐标系
            - True: GT 直接使用（如 DiffusionDrive 的 dump）
            - False: GT 在全局坐标系，需要转换为 ego-centric
        default_plan_len: 默认轨迹长度
    """

    def __init__(
        self,
        data_dir: str | Path,
        adapter_type: str = 'vad',
        gt_in_ego_frame: bool = False,
        default_plan_len: int = 6,
    ):
        self.data_dir = Path(data_dir)
        self.adapter_type = adapter_type
        self.gt_in_ego_frame = gt_in_ego_frame
        self.default_plan_len = default_plan_len

        # 加载 manifest
        manifest_path = self.data_dir / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'manifest.json 不存在: {manifest_path}')

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # manifest 格式兼容
        if isinstance(self.manifest, dict):
            self.samples = self.manifest.get('samples', [])
        else:
            self.samples = self.manifest

        logger.info(
            f'加载了 {len(self.samples)} 个样本 (adapter={adapter_type}, '
            f'gt_in_ego_frame={gt_in_ego_frame})'
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """加载单个样本。

        Returns:
            dict:
                - planner_outputs: 规划器原始输出（传给 Adapter）
                - gt_plan: GT 轨迹
                - metadata: 元信息
        """
        sample_info = self.samples[idx]
        file_path = self.data_dir / sample_info['file']

        # 加载原始数据
        data = torch.load(file_path, map_location='cpu')

        # 如果直接是 PlanningInterface，直接返回
        if isinstance(data, PlanningInterface):
            return {
                'planner_outputs': data,
                'gt_plan': data.reference_plan,  # placeholder
                'metadata': sample_info,
            }

        # 提取 GT 轨迹
        gt_plan = self._extract_gt_plan(data)

        return {
            'planner_outputs': data,
            'gt_plan': gt_plan,
            'metadata': sample_info,
        }

    def _extract_gt_plan(self, data: Dict) -> torch.Tensor:
        """从原始数据中提取 GT 轨迹。

        不同模型的 dump 数据可能有不同的字段名。
        子类可以重写此方法。
        """
        # 通用字段名尝试顺序
        gt_field_names = ['ego_fut_trajs', 'gt_trajectory', 'gt_plan', 'future_trajectory']

        for field_name in gt_field_names:
            if field_name in data:
                gt = data[field_name]
                if isinstance(gt, torch.Tensor):
                    return gt.float()

        # 默认值
        logger.warning(f'未找到 GT 轨迹字段，使用随机值')
        return torch.randn(self.default_plan_len, 2).float()


# =============================================================================
# Collate 函数
# =============================================================================
def collate_with_adapter(
    batch: List[Dict[str, Any]],
    adapter,
    gt_in_ego_frame: bool = False,
) -> Dict[str, Any]:
    """使用 Adapter 构建 PlanningInterface 的 collate 函数。

    Args:
        batch: Dataset 返回的原始数据批次
        adapter: Adapter 实例，负责坐标系转换
        gt_in_ego_frame: GT 是否已在 ego-centric 坐标系

    Returns:
        dict:
            - interface: PlanningInterface (batched)
            - gt_plan: GT 轨迹 [B, T, 2] (已转为 ego-centric)
            - metadata: 元信息
    """
    batch_size = len(batch)

    # 收集数据
    planner_outputs_list = [item['planner_outputs'] for item in batch]
    gt_plans_raw = [item['gt_plan'] for item in batch]
    metadata_list = [item['metadata'] for item in batch]

    # 判断是 PlanningInterface 还是 dict
    if isinstance(planner_outputs_list[0], PlanningInterface):
        # 已经是 PlanningInterface，直接 collate
        interfaces = planner_outputs_list
    else:
        # 需要用 Adapter 转换
        interfaces = _build_interfaces_from_dicts(planner_outputs_list, adapter)

    # 处理 GT 轨迹
    gt_plans = _process_gt_plans(
        gt_plans_raw, gt_in_ego_frame, interfaces[0].reference_plan.shape
    )

    # Collate PlanningInterface
    batched_interface = PlanningInterface.collate(interfaces)

    # 掩码（所有步都有效）
    plan_mask = torch.ones(batch_size, batched_interface.reference_plan.shape[1])

    return {
        'interface': batched_interface,
        'gt_plan': gt_plans,
        'plan_mask': plan_mask,
        'metadata': metadata_list,
    }


def _build_interfaces_from_dicts(
    outputs_list: List[Dict],
    adapter,
) -> List[PlanningInterface]:
    """从原始输出字典列表构建 PlanningInterface 列表。"""
    interfaces = []

    for outputs in outputs_list:
        try:
            interface = _dict_to_interface(outputs, adapter)
            interfaces.append(interface)
        except Exception as e:
            logger.warning(f'转换 PlanningInterface 失败: {e}，使用默认值')
            # 返回默认 interface
            interfaces.append(
                PlanningInterface(
                    scene_token=torch.randn(256),
                    reference_plan=torch.randn(6, 2),
                    plan_confidence=torch.ones(1),
                    candidate_plans=None,
                    safety_features=None,
                )
            )

    return interfaces


def _dict_to_interface(outputs: Dict, adapter) -> PlanningInterface:
    """将规划器输出字典转换为 PlanningInterface。"""
    # 提取各字段
    scene_token = adapter.extract_scene_token(outputs)
    reference_plan, candidate_plans = adapter.extract_reference_plan(outputs)
    plan_confidence = adapter.extract_plan_confidence(outputs)
    safety_features = adapter.extract_safety_features(outputs)

    # squeeze batch dim（如果有）
    if scene_token.dim() > 1 and scene_token.shape[0] == 1:
        scene_token = scene_token[0]
    if reference_plan.dim() > 2 and reference_plan.shape[0] == 1:
        reference_plan = reference_plan[0]
    if plan_confidence is not None and plan_confidence.shape[0] == 1:
        plan_confidence = plan_confidence[0]

    return PlanningInterface(
        scene_token=scene_token,
        reference_plan=reference_plan,
        plan_confidence=plan_confidence,
        candidate_plans=candidate_plans,
        safety_features=safety_features,
    )


def _process_gt_plans(
    gt_plans_raw: List[torch.Tensor],
    gt_in_ego_frame: bool,
    ref_shape: tuple,
) -> torch.Tensor:
    """处理 GT 轨迹，统一为 ego-centric 坐标系。

    Args:
        gt_plans_raw: 原始 GT 轨迹列表
        gt_in_ego_frame: GT 是否已在 ego-centric 坐标系
        ref_shape: 参考轨迹形状，用于验证

    Returns:
        GT 轨迹 [B, T, 2] (ego-centric)
    """
    gt_plans = []

    for gt in gt_plans_raw:
        # 确保 [T, 2]
        if gt.dim() == 1:
            gt = gt.reshape(-1, 2)
        elif gt.dim() == 3:
            gt = gt[0]

        # 坐标系转换
        if not gt_in_ego_frame:
            # GT 在全局坐标系，转换为 ego-centric
            gt = gt - gt[0]

        gt_plans.append(gt)

    # Stack 并 padding 到统一长度
    max_len = ref_shape[0]
    padded = []
    for gt in gt_plans:
        if gt.shape[0] < max_len:
            # Padding
            pad_len = max_len - gt.shape[0]
            gt = torch.cat([gt, torch.zeros(pad_len, 2)], dim=0)
        elif gt.shape[0] > max_len:
            # Truncate
            gt = gt[:max_len]
        padded.append(gt)

    return torch.stack(padded).float()


# =============================================================================
# 工厂函数
# =============================================================================
def build_planner_dataloader(
    data_dir: str | Path,
    adapter_type: str = 'vad',
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    gt_in_ego_frame: Optional[bool] = None,
    **adapter_kwargs,
) -> DataLoader:
    """构建通用规划器 DataLoader。

    Args:
        data_dir: 数据目录（包含 manifest.json 和 pt 文件）
        adapter_type: 适配器类型
            - 'vad': VAD 适配器
            - 'diffusiondrive': DiffusionDrive 适配器
        batch_size: 批大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        gt_in_ego_frame: GT 是否已在 ego-centric 坐标系
            - None: 自动推断（VAD=False, DiffusionDrive=True）
        **adapter_kwargs: 传递给 Adapter 的额外参数

    Returns:
        DataLoader 实例

    Example:
        # VAD 数据
        loader = build_planner_dataloader('data/vad_dumps', adapter_type='vad')

        # DiffusionDrive 数据
        loader = build_planner_dataloader('data/diffusion_dumps', adapter_type='diffusiondrive')
    """
    # 自动推断 gt_in_ego_frame
    if gt_in_ego_frame is None:
        gt_in_ego_frame = adapter_type in ['diffusiondrive', 'diffusion_drive']

    # 获取 Adapter 类并实例化
    adapter_class = _get_adapter_class(adapter_type)
    adapter = adapter_class(**adapter_kwargs)

    # 创建 Dataset
    dataset = PlannerDumpDataset(
        data_dir=data_dir,
        adapter_type=adapter_type,
        gt_in_ego_frame=gt_in_ego_frame,
    )

    # 创建 collate 函数（绑定 adapter）
    def collate_fn(batch):
        return collate_with_adapter(batch, adapter, gt_in_ego_frame)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


# =============================================================================
# 兼容旧接口
# =============================================================================
def build_vad_dataloader(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """构建 VAD dump DataLoader（向后兼容）。

    内部使用 build_planner_dataloader 实现。
    """
    return build_planner_dataloader(
        data_dir=data_dir,
        adapter_type='vad',
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        gt_in_ego_frame=False,  # VAD GT 在全局坐标系
    )


# =============================================================================
# VAD 专用 Dataset（向后兼容）
# =============================================================================
class VADDumpDataset(PlannerDumpDataset):
    """VAD dump 数据集（向后兼容）。

    已知 VAD dump 数据的特性：
    - GT (ego_fut_trajs): 全局坐标系，已 cumsum
    - ego_fut_preds: ego-centric 位移增量

    使用 build_vad_dataloader() 更简洁。
    """

    def __init__(self, data_dir: str | Path):
        super().__init__(
            data_dir=data_dir,
            adapter_type='vad',
            gt_in_ego_frame=False,
            default_plan_len=6,
        )


# =============================================================================
# DiffusionDrive 专用 Dataset
# =============================================================================
class DiffusionDriveDumpDataset(PlannerDumpDataset):
    """DiffusionDrive dump 数据集。

    已知 DiffusionDrive dump 数据的特性：
    - GT: ego-centric 绝对坐标
    - trajectory: ego-centric 绝对坐标
    """

    def __init__(self, data_dir: str | Path):
        super().__init__(
            data_dir=data_dir,
            adapter_type='diffusiondrive',
            gt_in_ego_frame=True,  # DiffusionDrive GT 已在 ego-centric
            default_plan_len=8,
        )
