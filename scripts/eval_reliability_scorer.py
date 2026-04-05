#!/usr/bin/env python3
"""评估已训练的 ReliabilityNet（筛选器 / Learned Scorer）。

前提：已用 collect_scorer_data.py 生成 pickle，或用 train_scorer.py 的训练集。

用法示例:
    python -m E2E_RL.scripts.eval_reliability_scorer \\
        --config E2E_RL/configs/refiner_debug.yaml \\
        --data_file /path/to/scorer_data.pkl \\
        --checkpoint ./experiments/scorer_training/best_model.pth

若无 checkpoint 或 data_file，脚本会提示；指标与 train_scorer 验证阶段一致
（Gain/Risk 的 Spearman、Top-5 近似匹配）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import yaml

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.scripts.train_scorer import ScorerDataset, validate_epoch
from E2E_RL.update_filter.model import ReliabilityNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='评估 ReliabilityNet')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    reliability_cfg = cfg.get('reliability_net', {})
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = ScorerDataset(args.data_file)
    if len(dataset) == 0:
        logger.error('数据为空')
        sys.exit(1)

    sample = dataset[0]
    plan_len = sample['reference_plan'].shape[0]
    scene_dim = sample['scene_token'].shape[-1]
    if reliability_cfg.get('plan_len', plan_len) != plan_len:
        logger.warning('配置 plan_len 与数据不一致，使用数据维度 %d', plan_len)
    if reliability_cfg.get('scene_dim', scene_dim) != scene_dim:
        logger.warning('配置 scene_dim 与数据不一致，使用数据维度 %d', scene_dim)

    model = ReliabilityNet(
        scene_dim=scene_dim,
        plan_len=plan_len,
        safety_feat_dim=reliability_cfg.get('safety_feat_dim', 0),
        hidden_dim=reliability_cfg.get('hidden_dim', 128),
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    logger.info('已加载权重: %s', args.checkpoint)

    def collate_fn(batch):
        out = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if all(v is None for v in values):
                out[key] = None
            else:
                out[key] = default_collate(values)
        return out

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    metrics = validate_epoch(model, loader, device)
    logger.info('--- ReliabilityNet 评估结果 ---')
    for k, v in metrics.items():
        logger.info('  %s: %.4f', k, v)


if __name__ == '__main__':
    main()
