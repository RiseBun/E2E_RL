#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.planning_interface.adapters.diffusiondrive_adapter import (
    DiffusionDrivePlanningAdapter,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _load_raw_paths(input_dir: Path, max_samples: Optional[int]) -> List[Path]:
    paths = sorted(input_dir.glob('*.pt'))
    if max_samples is not None:
        paths = paths[:max_samples]
    return paths


def _extract_planner_outputs(raw: Dict[str, Any]) -> Dict[str, Any]:
    if 'planner_outputs' in raw and isinstance(raw['planner_outputs'], dict):
        return raw['planner_outputs']
    if 'outputs' in raw and isinstance(raw['outputs'], dict):
        return raw['outputs']
    if 'trajectory' in raw:
        return raw
    raise KeyError('无法在输入样本中找到 planner_outputs/outputs/trajectory 字段')


def _extract_gt_plan(raw: Dict[str, Any]) -> torch.Tensor:
    for k in ('gt_plan', 'ego_fut_trajs', 'gt_traj', 'gt_trajectory'):
        if k in raw:
            gt = raw[k]
            if isinstance(gt, torch.Tensor):
                break
    else:
        raise KeyError('无法在输入样本中找到 gt_plan/ego_fut_trajs/gt_traj/gt_trajectory 字段')

    if gt.dim() == 2 and gt.shape[-1] == 2:
        return gt
    if gt.dim() == 3 and gt.shape[-1] == 2:
        return gt[0]
    raise ValueError(f'GT shape 不符合预期，期望 [T,2] 或 [1,T,2]，得到 {tuple(gt.shape)}')


def _interface_to_cpu_dict(interface: Any) -> Dict[str, torch.Tensor]:
    def _maybe_squeeze_first_dim(x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 1 and x.shape[0] == 1:
            return x.squeeze(0)
        return x

    result: Dict[str, torch.Tensor] = {
        'scene_token': _maybe_squeeze_first_dim(interface.scene_token.detach().cpu()),
        'reference_plan': _maybe_squeeze_first_dim(interface.reference_plan.detach().cpu()),
    }
    if getattr(interface, 'candidate_plans', None) is not None:
        result['candidate_plans'] = _maybe_squeeze_first_dim(
            interface.candidate_plans.detach().cpu()
        )
    if getattr(interface, 'plan_confidence', None) is not None:
        result['plan_confidence'] = _maybe_squeeze_first_dim(
            interface.plan_confidence.detach().cpu()
        )
    safety = getattr(interface, 'safety_features', None)
    if isinstance(safety, dict):
        for k, v in safety.items():
            result[f'safety_{k}'] = _maybe_squeeze_first_dim(v.detach().cpu())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='将 DiffusionDrive 离线输出转换为 E2E_RL dump 格式')
    parser.add_argument('--input_dir', type=str, required=True, help='原始 DiffusionDrive 输出目录（.pt 文件）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出 dump 目录（含 manifest.json + 000000.pt...）')
    parser.add_argument('--pool_mode', type=str, default='grid', choices=['mean', 'grid', 'ego_local'])
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _load_raw_paths(input_dir, args.max_samples)
    if not paths:
        raise FileNotFoundError(f'input_dir 下未找到 .pt 文件: {input_dir}')

    adapter = DiffusionDrivePlanningAdapter(
        scene_pool=args.pool_mode,
        grid_size=args.grid_size,
    )

    manifest: Dict[str, Any] = {
        'format': 'E2E_RL_dump_v1',
        'planner': 'DiffusionDrive',
        'pool_mode': args.pool_mode,
        'samples': [],
    }

    for i, p in enumerate(paths):
        raw = torch.load(p, map_location='cpu')
        planner_outputs = _extract_planner_outputs(raw)
        gt_plan = _extract_gt_plan(raw)

        interface = adapter.extract(planner_outputs, ego_fut_cmd=None)

        save_dict: Dict[str, Any] = {
            'sample_idx': i,
            'source_path': str(p),
            'ego_fut_trajs': gt_plan.detach().cpu(),
            f'interface_{args.pool_mode}': _interface_to_cpu_dict(interface),
            'planner_outputs': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                               for k, v in planner_outputs.items()},
        }

        torch.save(save_dict, output_dir / f'{i:06d}.pt')
        manifest['samples'].append({'idx': i, 'file': f'{i:06d}.pt'})

    with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f'转换完成: {len(paths)} samples -> {output_dir}')


if __name__ == '__main__':
    main()
