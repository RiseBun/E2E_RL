"""导出 SparseDrive 推理输出 + GT 到磁盘。

用途:
    将 SparseDrive 模型在 val set 上的推理结果逐帧保存，供后续离线分析使用。

实现方式:
    1. 复用 SparseDrive tools/test.py 的模型/数据加载逻辑
    2. 在 motion_plan_head 上注册 forward hook 捕获原始输出
    3. 逐帧保存 .pt 文件，同时生成 manifest.json 索引

使用方式:
    python scripts/dump_sparsedrive_inference.py \
        --config projects/SparseDrive/projects/configs/sparsedrive_small_stage2.py \
        --checkpoint /path/to/sparsedrive_stage2.pth \
        --output_dir data/sparsedrive_dumps \
        --data_root /path/to/nuscenes/data/ \
        --max_samples 100  # 调试用，省略则跑全集

输出结构:
    output_dir/
    ├── manifest.json          # 样本索引
    ├── 000000.pt              # 第 0 帧
    ├── 000001.pt              # 第 1 帧
    └── ...

每个 .pt 文件包含:
    {
        'sample_idx': int,
        'scene_token': str,
        # --- 原始 SparseDrive 输出（关键字段） ---
        'ego_fut_preds': [M, T, 2],        # 位移增量（M=6, T=6）
        'planning_cls_scores': [M],        # 规划模式分数
        # --- GT ---
        'ego_fut_trajs': [T, 2],            # GT 轨迹（位移增量）
        'ego_fut_masks': [T],               # 有效 mask
        'ego_fut_cmd': [M],                 # 命令（one-hot）
        'ego_fut_cmd_idx': int,             # 命令索引
        # --- 元信息 ---
        'timestamp': int,
        'sample_token': str,
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def _unwrap_dc(val):
    """解包 mmcv DataContainer，返回底层 tensor 或原始值。"""
    if val is None:
        return None
    # 单个 DataContainer
    if hasattr(val, 'data'):
        inner = val.data
        if isinstance(inner, list) and len(inner) > 0:
            return inner[0]
        return inner
    # list[DataContainer] — collate 后常见格式
    if isinstance(val, (list, tuple)) and len(val) > 0 and hasattr(val[0], 'data'):
        inner = val[0].data
        if isinstance(inner, list) and len(inner) > 0:
            return inner[0]
        return inner
    return val


# ======================================================================
# Forward hook：捕获 motion_plan_head 的原始输出
# ======================================================================
class SparseDriveOutsCapture:
    """通过 forward hook 捕获 SparseDrive MotionPlanningHead 的 forward 返回值。"""

    def __init__(self):
        self.captured_motion_outs: Optional[Dict[str, Any]] = None
        self.captured_planning_outs: Optional[Dict[str, Any]] = None

    def hook_fn(self, module, input, output):
        """forward hook 回调。
        
        SparseDriveHead.forward 返回 (det_output, map_output, motion_output, planning_output)
        我们只关心 motion_output 和 planning_output。
        """
        if isinstance(output, (tuple, list)) and len(output) >= 4:
            self.captured_motion_outs = output[2] if output[2] is not None else None
            self.captured_planning_outs = output[3] if output[3] is not None else None
        else:
            logger.warning(f'Unexpected output format: {type(output)}')

    def reset(self):
        self.captured_motion_outs = None
        self.captured_planning_outs = None


# ======================================================================
# 单帧保存逻辑
# ======================================================================
def save_sample(
    output_dir: Path,
    idx: int,
    motion_outs: Dict[str, Any],
    planning_outs: Dict[str, Any],
    img_metas: List[Dict[str, Any]],
    ego_fut_trajs: torch.Tensor,
    ego_fut_cmd: torch.Tensor,
    ego_fut_masks: Optional[torch.Tensor],
):
    """保存单帧推理结果。"""
    save_dict: Dict[str, Any] = {
        'sample_idx': idx,
    }

    # ---- 元信息 ----
    if img_metas and len(img_metas) > 0:
        meta = img_metas[0] if isinstance(img_metas[0], dict) else {}
        save_dict['scene_token'] = meta.get('scene_token', '')
        save_dict['sample_token'] = meta.get('sample_idx', '')
        save_dict['timestamp'] = meta.get('timestamp', 0)

    # ---- 原始 SparseDrive 输出（关键字段，squeeze batch=1） ----
    # planning output 中的 prediction 是位移增量
    if planning_outs is not None and 'prediction' in planning_outs:
        # prediction 是多解码器输出，取最后一层
        preds = planning_outs['prediction']
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]  # 取最后一层解码器
        
        # [B, M, T, 2] -> [M, T, 2] (squeeze batch)
        if preds.dim() == 4:
            preds = preds[0]  # [M, T, 2]
        save_dict['ego_fut_preds'] = preds.detach().cpu()

    # planning classification scores
    if planning_outs is not None and 'classification' in planning_outs:
        cls_scores = planning_outs['classification']
        if isinstance(cls_scores, (list, tuple)):
            cls_scores = cls_scores[-1]  # 取最后一层
        
        # [B, M] -> [M]
        if cls_scores.dim() == 2:
            cls_scores = cls_scores[0]
        save_dict['planning_cls_scores'] = cls_scores.detach().cpu()

    # ---- GT 数据 ----
    # ego_fut_trajs: 原始为 [B, T*2] 或 [B, T, 2]
    gt_trajs = ego_fut_trajs
    if gt_trajs is not None:
        gt_trajs = gt_trajs.detach().cpu()
        # 如果 shape 是 [B, T*2]，reshape 为 [T, 2]
        if gt_trajs.dim() == 2:
            gt_trajs = gt_trajs[0]  # [T*2]
            if gt_trajs.dim() == 1:
                gt_trajs = gt_trajs.reshape(-1, 2)  # [T, 2]
        elif gt_trajs.dim() == 3:
            gt_trajs = gt_trajs[0]  # [T, 2]
        save_dict['ego_fut_trajs'] = gt_trajs

    # ego_fut_cmd
    cmd = ego_fut_cmd
    if cmd is not None:
        cmd = cmd.detach().cpu()
        # 通常 [B, M]
        if cmd.dim() == 2:
            cmd_vec = cmd[0]  # [M]
        else:
            cmd_vec = cmd
        save_dict['ego_fut_cmd'] = cmd_vec
        # 命令索引
        cmd_idx = torch.nonzero(cmd_vec.float())
        save_dict['ego_fut_cmd_idx'] = cmd_idx[0, 0].item() if len(cmd_idx) > 0 else 0

    # ego_fut_masks
    if ego_fut_masks is not None:
        masks = ego_fut_masks.detach().cpu()
        if masks.dim() >= 2:
            masks = masks[0]
        save_dict['ego_fut_masks'] = masks

    # ---- 保存 ----
    save_path = output_dir / f'{idx:06d}.pt'
    torch.save(save_dict, save_path)


# ======================================================================
# 主流程
# ======================================================================
def build_model_and_dataloader(cfg_path: str, ckpt_path: str, data_root: Optional[str] = None):
    """复用 SparseDrive tools/test.py 的模型和数据加载逻辑。"""
    import importlib

    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model

    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet.datasets import replace_ImageToTensor
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(cfg_path)
    
    # 允许命令行覆盖 data_root
    if data_root is not None:
        logger.info(f'使用指定的数据目录: {data_root}')
        cfg.data_root = data_root
        # 更新 dataset 配置中的 data_root
        if hasattr(cfg, 'data'):
            for split in ['train', 'val', 'test']:
                if hasattr(cfg.data, split) and isinstance(cfg.data[split], dict):
                    cfg.data[split]['data_root'] = data_root
            # 更新 ann_file 路径
            for split in ['val', 'test']:
                if hasattr(cfg.data, split) and isinstance(cfg.data[split], dict):
                    if 'ann_file' in cfg.data[split]:
                        cfg.data[split]['ann_file'] = data_root + cfg.data[split]['ann_file'].split('/')[-1]

    # 加载插件
    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_path = '.'.join(_module_dir.split('/'))
            importlib.import_module(_module_path)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pop('samples_per_gpu', 1)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = 1

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    return model, data_loader, dataset


def run_dump(
    cfg_path: str,
    ckpt_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    data_root: Optional[str] = None,
):
    """执行推理 dump。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f'配置文件: {cfg_path}')
    logger.info(f'检查点:   {ckpt_path}')
    if data_root is not None:
        logger.info(f'数据目录: {data_root}')
    logger.info(f'输出目录: {output_path}')

    # 构建模型和数据加载器
    model, data_loader, dataset = build_model_and_dataloader(cfg_path, ckpt_path, data_root)
    logger.info(f'数据集大小: {len(dataset)}')

    total = min(len(dataset), max_samples) if max_samples else len(dataset)
    logger.info(f'将导出 {total} 个样本')

    # 注册 forward hook 捕获 SparseDriveHead 原始输出
    capture = SparseDriveOutsCapture()
    # MMDataParallel 包裹下，真正的模型在 model.module
    actual_model = model.module
    hook_handle = actual_model.head.register_forward_hook(capture.hook_fn)
    logger.info('已注册 head forward hook')

    # 推理循环
    manifest = []
    t_start = time.time()

    for i, data in enumerate(data_loader):
        if i >= total:
            break

        capture.reset()

        # data 是 mmcv 的 collate 输出，包含 img_metas, img, GT 等
        with torch.no_grad():
            # 调用模型的 forward_test
            result = model(return_loss=False, rescale=True, **data)

        # 检查 hook 是否捕获成功
        motion_outs = capture.captured_motion_outs
        planning_outs = capture.captured_planning_outs
        
        if motion_outs is None or planning_outs is None:
            logger.warning(f'样本 {i}: hook 未捕获到输出，跳过')
            continue

        # 提取 img_metas
        img_metas = data.get('img_metas', [None])
        if isinstance(img_metas, list) and len(img_metas) > 0:
            # DataContainer 格式
            if hasattr(img_metas[0], 'data'):
                img_metas = img_metas[0].data[0]
            elif isinstance(img_metas[0], list):
                img_metas = img_metas[0]

        # 提取 GT 数据（处理 mmcv DataContainer 格式）
        ego_fut_trajs = _unwrap_dc(data.get('gt_ego_fut_trajs', None))
        ego_fut_cmd = _unwrap_dc(data.get('gt_ego_fut_cmd', None))
        ego_fut_masks = _unwrap_dc(data.get('gt_ego_fut_masks', None))

        # 保存
        save_sample(
            output_dir=output_path,
            idx=i,
            motion_outs=motion_outs,
            planning_outs=planning_outs,
            img_metas=img_metas if isinstance(img_metas, list) else [img_metas],
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_fut_masks=ego_fut_masks,
        )

        # manifest 条目
        entry = {'idx': i, 'file': f'{i:06d}.pt'}
        if isinstance(img_metas, list) and len(img_metas) > 0:
            meta = img_metas[0] if isinstance(img_metas[0], dict) else {}
            entry['scene_token'] = meta.get('scene_token', '')

        manifest.append(entry)

        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            eta = (total - i - 1) / fps if fps > 0 else 0
            logger.info(
                f'[{i+1}/{total}] {fps:.1f} samples/s, '
                f'ETA: {eta:.0f}s'
            )

    hook_handle.remove()

    # 保存 manifest
    manifest_path = output_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'total_samples': len(manifest),
            'config': cfg_path,
            'checkpoint': ckpt_path,
            'samples': manifest,
        }, f, indent=2)

    elapsed_total = time.time() - t_start
    logger.info(f'导出完成: {len(manifest)} 个样本, 耗时 {elapsed_total:.1f}s')
    logger.info(f'输出目录: {output_path}')
    logger.info(f'Manifest: {manifest_path}')


def main():
    parser = argparse.ArgumentParser(
        description='导出 SparseDrive 推理输出 + GT'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='SparseDrive 配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='SparseDrive 检查点路径')
    parser.add_argument('--output_dir', type=str,
                        default='data/sparsedrive_dumps',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大导出样本数（调试用）')
    parser.add_argument('--data_root', type=str, default=None,
                        help='nuScenes 数据根目录（覆盖配置文件中的 data_root）')
    args = parser.parse_args()

    run_dump(
        cfg_path=args.config,
        ckpt_path=args.checkpoint,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        data_root=args.data_root,
    )


if __name__ == '__main__':
    main()
