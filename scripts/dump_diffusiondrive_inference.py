"""导出 DiffusionDrive (NAVSIM) 推理输出到磁盘。

用途:
    逐场景运行 DiffusionDrive 的前向推理，导出原始 planner_outputs 与 GT。
    输出可直接喂给 scripts/convert_diffusiondrive_dump.py 转成 E2E_RL 标准格式。

使用方式:
    python scripts/dump_diffusiondrive_inference.py \
        --agent_config /path/to/diffusiondrive_agent.yaml \
        --checkpoint /path/to/checkpoint.pth \
        --data_path /path/to/navsim_logs/val \
        --sensor_path /path/to/sensor_blobs/val \
        --output_dir data/diffusiondrive_raw \
        --max_samples 100 \
        --device cuda

输出结构:
    output_dir/
    ├── manifest.json
    ├── 000000.pt
    ├── 000001.pt
    └── ...

每个 .pt 文件包含:
    {
        'sample_idx': int,
        'token': str,
        'scene_token': str,
        'planner_outputs': {
            'trajectory': [T, 3],
            'bev_semantic_map': [C, H, W],
            'agent_states': [A, 5],
            'agent_labels': [A],
            ...
        },
        'gt_plan': [T, 2],
        'ego_fut_cmd': [M],  # one-hot 或命令向量
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def _to_tensor_cpu(x: Any) -> Any:
    """将张量/嵌套结构安全转换到 CPU，便于序列化保存。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_tensor_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_tensor_cpu(v) for v in x]
        return type(x)(t) if isinstance(x, tuple) else t
    return x


def _build_agent(
    diffusiondrive_root: Path,
    agent_config_path: str,
    checkpoint_path: str,
    device: str,
):
    """从 Hydra yaml 实例化 DiffusionDrive agent。"""
    if str(diffusiondrive_root) not in sys.path:
        sys.path.insert(0, str(diffusiondrive_root))

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(agent_config_path)
    cfg.checkpoint_path = checkpoint_path
    agent = instantiate(cfg)
    agent.initialize()
    agent.to(torch.device(device))
    agent.eval()
    return agent


def _build_scene_loader(
    diffusiondrive_root: Path,
    data_path: str,
    sensor_path: str,
    sensor_config,
):
    """构建 NAVSIM SceneLoader。"""
    if str(diffusiondrive_root) not in sys.path:
        sys.path.insert(0, str(diffusiondrive_root))

    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter

    scene_filter = SceneFilter()  # 默认与官方脚本一致: 4 history + 10 future
    loader = SceneLoader(
        data_path=Path(data_path),
        sensor_blobs_path=Path(sensor_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    return loader


def _extract_gt_and_meta_from_frames(
    frames: List[Dict[str, Any]],
    num_history_frames: int,
    num_future_poses: int,
) -> tuple[torch.Tensor, str]:
    """从原始 frame 列表提取 GT 轨迹和 scene_token（避免依赖 map API）。"""
    from pyquaternion import Quaternion
    import numpy as np
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    current_idx = num_history_frames - 1
    end_idx = min(current_idx + num_future_poses, len(frames) - 1)

    global_poses = []
    for idx in range(current_idx, end_idx + 1):
        tr = frames[idx]['ego2global_translation']
        quat = Quaternion(*frames[idx]['ego2global_rotation'])
        global_poses.append([tr[0], tr[1], quat.yaw_pitch_roll[0]])

    origin = StateSE2(*global_poses[0])
    future_local = convert_absolute_to_relative_se2_array(
        origin,
        np.array(global_poses[1:], dtype=np.float64),
    )
    gt_plan = torch.from_numpy(future_local[..., :2]).float()

    scene_token = frames[current_idx].get('scene_token', '')
    return gt_plan, scene_token


def run_inference_and_dump(
    agent: torch.nn.Module,
    scene_loader,
    output_dir: Path,
    max_samples: Optional[int] = None,
    device: str = 'cuda',
):
    """运行推理并保存输出。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    all_tokens: List[str] = scene_loader.tokens
    if max_samples is not None:
        all_tokens = all_tokens[:max_samples]

    manifest: Dict[str, Any] = {
        'total_samples': 0,
        'skipped_samples': 0,
        'scene_filter': str(scene_loader._scene_filter),
        'samples': [],
    }

    feature_builders = agent.get_feature_builders()

    skipped_count = 0
    with torch.no_grad():
        for token in all_tokens:
            start_time = time.time()

            # 跳过缺少 sensor blobs 的场景
            try:
                agent_input = scene_loader.get_agent_input_from_token(token)
                frames = scene_loader.scene_frames_dicts[token]
            except (FileNotFoundError, KeyError) as e:
                skipped_count += 1
                if skipped_count % 100 == 0:
                    logger.warning(f'跳过 {skipped_count} 个缺少 sensor blobs 的场景')
                continue

            # 手动构建 feature 并前向，拿到完整 planner_outputs 字典
            features: Dict[str, torch.Tensor] = {}
            for builder in feature_builders:
                features.update(builder.compute_features(agent_input))
            features = {
                k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }

            outputs = agent.forward(features)
            planner_outputs = _to_tensor_cpu(outputs)

            # GT 从原始日志提取，不依赖 map API
            gt_plan, scene_token = _extract_gt_and_meta_from_frames(
                frames=frames,
                num_history_frames=scene_loader._scene_filter.num_history_frames,
                num_future_poses=agent._config.trajectory_sampling.num_poses,
            )

            # 驾驶命令（与特征构建一致，取最后一帧）
            ego_fut_cmd = torch.tensor(
                agent_input.ego_statuses[-1].driving_command,
                dtype=torch.float32,
            )

            save_dict: Dict[str, Any] = {
                'sample_idx': sample_count,
                'token': token,
                'scene_token': scene_token,
                'planner_outputs': planner_outputs,
                'gt_plan': gt_plan,
                'ego_fut_cmd': ego_fut_cmd,
            }

            output_file = output_dir / f'{sample_count:06d}.pt'
            torch.save(save_dict, output_file)

            manifest['samples'].append({
                'idx': sample_count,
                'file': output_file.name,
                'token': token,
                'scene_token': scene_token,
            })

            elapsed = time.time() - start_time
            sample_count += 1

            if sample_count % 10 == 0:
                logger.info(
                    f'[{sample_count}/{len(all_tokens)}] '
                    f'Saved {output_file.name} ({elapsed:.2f}s)'
                )

    manifest['total_samples'] = sample_count
    manifest['skipped_samples'] = skipped_count
    with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f'✓ 推理完成！共保存 {sample_count} 个样本，跳过 {skipped_count} 个')
    logger.info(f'输出目录: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='导出 DiffusionDrive (NAVSIM) 推理输出')
    parser.add_argument('--agent_config', type=str, required=True, help='DiffusionDrive agent YAML 路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='DiffusionDrive 权重路径')
    parser.add_argument('--data_path', type=str, required=True, help='NAVSIM logs 路径（如 navsim_logs/val）')
    parser.add_argument('--sensor_path', type=str, required=True, help='NAVSIM sensor_blobs 路径（如 sensor_blobs/val）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数（用于调试）')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备')
    parser.add_argument(
        '--diffusiondrive_root',
        type=str,
        default=None,
        help='DiffusionDrive 项目根目录（默认自动推断为 ../DiffusionDrive）',
    )
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA 不可用，自动切换到 CPU')
        args.device = 'cpu'

    if args.diffusiondrive_root is None:
        # .../RL/E2E_RL/scripts -> .../RL/DiffusionDrive
        diffusiondrive_root = Path(__file__).resolve().parents[2] / 'DiffusionDrive'
    else:
        diffusiondrive_root = Path(args.diffusiondrive_root)
    if not diffusiondrive_root.exists():
        raise FileNotFoundError(f'DiffusionDrive 根目录不存在: {diffusiondrive_root}')

    logger.info('='*60)
    logger.info('DiffusionDrive (NAVSIM) 数据导出工具')
    logger.info('='*60)

    # Step 1: 加载模型
    logger.info('Step 1: 加载 DiffusionDrive 模型...')
    agent = _build_agent(
        diffusiondrive_root=diffusiondrive_root,
        agent_config_path=args.agent_config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    logger.info('✓ 模型加载完成')

    # Step 2: 构建 scene loader
    logger.info('Step 2: 构建 NAVSIM SceneLoader...')
    scene_loader = _build_scene_loader(
        diffusiondrive_root=diffusiondrive_root,
        data_path=args.data_path,
        sensor_path=args.sensor_path,
        sensor_config=agent.get_sensor_config(),
    )
    logger.info(f'✓ SceneLoader 构建完成，共 {len(scene_loader)} 个场景')

    # Step 3: 运行推理并保存
    logger.info('Step 3: 运行推理并导出...')
    output_dir = Path(args.output_dir)
    run_inference_and_dump(
        agent=agent,
        scene_loader=scene_loader,
        output_dir=output_dir,
        max_samples=args.max_samples,
        device=args.device,
    )

    logger.info('='*60)
    logger.info('导出完成！')
    logger.info(f'输出目录: {output_dir}')
    logger.info('下一步: 运行 convert_diffusiondrive_dump.py 转换为标准格式')
    logger.info('='*60)


if __name__ == '__main__':
    main()
