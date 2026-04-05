#!/usr/bin/env python
"""精炼轨迹评估脚本。

用法:
    python -m E2E_RL.scripts.eval_refined \
        --checkpoint ./experiments/refiner_debug/best.pth \
        --config E2E_RL/configs/refiner_debug.yaml

功能:
    1. 加载预训练 refiner 检查点
    2. 在验证集上运行 baseline vs refined 对比评估
    3. 输出 ADE/FDE/L2/碰撞率等指标
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.evaluators.eval_refined import evaluate_refined_plans
from E2E_RL.hard_case.mining import HardCaseMiner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='精炼轨迹评估')
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Refiner 检查点路径',
    )
    parser.add_argument(
        '--config', type=str,
        default='E2E_RL/configs/refiner_debug.yaml',
        help='配置文件路径',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='评估设备',
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 构建模型并加载权重
    model_cfg = cfg['model']
    refiner = InterfaceRefiner(
        scene_dim=model_cfg['scene_dim'],
        plan_len=model_cfg['plan_len'],
        hidden_dim=model_cfg['hidden_dim'],
        dropout=model_cfg.get('dropout', 0.1),
        output_norm=model_cfg.get('output_norm', False),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    refiner.load_state_dict(ckpt['refiner_state_dict'])
    refiner.eval()
    logger.info(f'已加载检查点: {args.checkpoint} (epoch={ckpt.get("epoch", "?")})')

    # 注意: 实际使用时需替换为真实的验证 DataLoader
    logger.info(
        '请提供真实的验证 DataLoader 以运行评估。\n'
        '评估函数可通过以下方式调用:\n'
        '  from E2E_RL.evaluators.eval_refined import evaluate_refined_plans\n'
        '  results = evaluate_refined_plans(\n'
        '      baseline_plan, refined_plan, gt_plan, mask, ...)\n'
    )


if __name__ == '__main__':
    main()
