"""共享的 dump 数据加载工具。

从 dump_vad_inference.py 导出的 .pt 文件中批量加载数据，
供 offline_confidence_analysis.py 和 offline_ab_comparison.py 使用。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def load_manifest(dump_dir: str) -> Dict[str, Any]:
    """加载 manifest.json。"""
    manifest_path = Path(dump_dir) / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'manifest.json 不存在: {manifest_path}')
    with open(manifest_path) as f:
        return json.load(f)


def load_sample(dump_dir: str, idx: int) -> Dict[str, Any]:
    """加载单个样本。"""
    path = Path(dump_dir) / f'{idx:06d}.pt'
    if not path.exists():
        raise FileNotFoundError(f'样本文件不存在: {path}')
    return torch.load(path, map_location='cpu')


def load_all_samples(
    dump_dir: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """加载所有样本。

    Args:
        dump_dir: dump 目录路径
        max_samples: 最大加载样本数（None = 全部）

    Returns:
        样本列表
    """
    manifest = load_manifest(dump_dir)
    samples_info = manifest['samples']
    total = len(samples_info)
    if max_samples is not None:
        total = min(total, max_samples)

    logger.info(f'加载 {total}/{len(samples_info)} 个样本...')

    samples = []
    for i in range(total):
        try:
            sample = load_sample(dump_dir, i)
            samples.append(sample)
        except FileNotFoundError:
            logger.warning(f'样本 {i} 文件缺失，跳过')
    logger.info(f'成功加载 {len(samples)} 个样本')
    return samples


def batch_from_samples(
    samples: List[Dict[str, Any]],
    pool_mode: str = 'mean',
) -> Dict[str, torch.Tensor]:
    """将样本列表组合为 batch tensor。

    Args:
        samples: load_all_samples 返回的样本列表
        pool_mode: 使用哪种池化方式的 interface ('mean' / 'grid' / 'ego_local')

    Returns:
        包含以下键的 dict:
        - ego_fut_preds: [N, M, T, 2]
        - ego_fut_trajs: [N, T, 2] （GT，已 cumsum）
        - ego_fut_cmd: [N, M]
        - ego_fut_cmd_idx: [N]
        - scene_token: [N, D]
        - reference_plan: [N, T, 2]
        - plan_confidence: [N, 1]
        - safety_*: 各安全特征
        - metric_results: list[dict]
    """
    interface_key = f'interface_{pool_mode}'

    # 收集各字段
    ego_fut_preds_list = []
    ego_fut_trajs_list = []
    ego_fut_cmd_list = []
    cmd_idx_list = []
    scene_token_list = []
    reference_plan_list = []
    confidence_list = []
    safety_keys: Optional[List[str]] = None
    safety_lists: Dict[str, list] = {}
    metrics_list = []

    for s in samples:
        # 原始 VAD 输出
        if 'ego_fut_preds' in s:
            ego_fut_preds_list.append(s['ego_fut_preds'])
        if 'ego_fut_trajs' in s:
            ego_fut_trajs_list.append(s['ego_fut_trajs'])
        if 'ego_fut_cmd' in s:
            ego_fut_cmd_list.append(s['ego_fut_cmd'])
        cmd_idx_list.append(s.get('ego_fut_cmd_idx', 0))

        # PlanningInterface
        iface = s.get(interface_key, {})
        if 'scene_token' in iface:
            scene_token_list.append(iface['scene_token'])
        if 'reference_plan' in iface:
            reference_plan_list.append(iface['reference_plan'])
        if 'plan_confidence' in iface:
            confidence_list.append(iface['plan_confidence'])

        # 安全特征
        if safety_keys is None:
            safety_keys = [k for k in iface if k.startswith('safety_')]
            for sk in safety_keys:
                safety_lists[sk] = []
        for sk in (safety_keys or []):
            if sk in iface:
                safety_lists[sk].append(iface[sk])

        # 指标
        if 'metric_results' in s:
            metrics_list.append(s['metric_results'])

    result: Dict[str, Any] = {}

    def _stack(lst, name):
        if lst:
            try:
                return torch.stack(lst)
            except RuntimeError:
                logger.warning(f'{name}: shape 不一致，无法 stack')
        return None

    result['ego_fut_preds'] = _stack(ego_fut_preds_list, 'ego_fut_preds')
    result['ego_fut_trajs'] = _stack(ego_fut_trajs_list, 'ego_fut_trajs')
    result['ego_fut_cmd'] = _stack(ego_fut_cmd_list, 'ego_fut_cmd')
    result['ego_fut_cmd_idx'] = torch.tensor(cmd_idx_list, dtype=torch.long)
    result['scene_token'] = _stack(scene_token_list, 'scene_token')
    result['reference_plan'] = _stack(reference_plan_list, 'reference_plan')
    result['plan_confidence'] = _stack(confidence_list, 'plan_confidence')
    for sk, sl in safety_lists.items():
        result[sk] = _stack(sl, sk)
    result['metric_results'] = metrics_list

    # 过滤 None
    result = {k: v for k, v in result.items() if v is not None}

    return result
