#!/usr/bin/env python
"""训练 ReliabilityNet (Learned Scorer)。

使用收集的数据进行离线训练和排序验证。
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import yaml

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.update_filter.model import ReliabilityNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


class ScorerDataset(Dataset):
    """ReliabilityNet 训练数据集。"""

    def __init__(self, data_file: str):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f'加载了 {len(self.data)} 个训练样本')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'scene_token': item['scene_token'],
            'reference_plan': item['reference_plan'],
            'residual': item['residual'],
            'plan_confidence': item['plan_confidence'],
            'safety_features': item['safety_features'],
            'heuristic_scores': item['heuristic_scores'],
            'delta_safe_reward': item['delta_safe_reward'],
            'delta_progress': item['delta_progress'],
            'delta_collision': item['delta_collision'],
            'delta_offroad': item['delta_offroad'],
            'delta_comfort': item['delta_comfort'],
            'delta_drift': item['delta_drift'],
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_epoch(
    model: ReliabilityNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        scene_token = batch['scene_token'].to(device)
        reference_plan = batch['reference_plan'].to(device)
        residual = batch['residual'].to(device)
        plan_confidence = batch['plan_confidence'].to(device) if batch['plan_confidence'] is not None else None
        safety_features = batch['safety_features'].to(device) if batch['safety_features'] is not None else None
        heuristic_scores = batch['heuristic_scores'].to(device)
        if heuristic_scores.ndim == 3 and heuristic_scores.shape[1] == 1:
            heuristic_scores = heuristic_scores.squeeze(1)

        # Labels
        gain_label = batch['delta_safe_reward'].unsqueeze(-1).to(device)
        risk_components_label = torch.stack([
            batch['delta_collision'],
            batch['delta_offroad'],
            batch['delta_comfort'],
            batch['delta_drift'],
        ], dim=-1).to(device)

        # Forward
        pred_gain, pred_risk_total, pred_risk_components = model(
            scene_token, reference_plan, residual,
            plan_confidence, safety_features, heuristic_scores
        )

        # Loss
        loss_gain = nn.functional.mse_loss(pred_gain, gain_label)
        loss_risk_total = nn.functional.mse_loss(pred_risk_total, risk_components_label.sum(dim=-1, keepdim=True))
        loss_risk_components = nn.functional.mse_loss(pred_risk_components, risk_components_label)
        loss = loss_gain + loss_risk_total + loss_risk_components

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(
    model: ReliabilityNet,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """验证一个 epoch，返回排序指标。"""
    model.eval()
    all_pred_gain = []
    all_true_gain = []
    all_pred_risk = []
    all_true_risk = []

    with torch.no_grad():
        for batch in dataloader:
            scene_token = batch['scene_token'].to(device)
            reference_plan = batch['reference_plan'].to(device)
            residual = batch['residual'].to(device)
            plan_confidence = batch['plan_confidence'].to(device) if batch['plan_confidence'] is not None else None
            safety_features = batch['safety_features'].to(device) if batch['safety_features'] is not None else None
            heuristic_scores = batch['heuristic_scores'].to(device)
            if heuristic_scores.ndim == 3 and heuristic_scores.shape[1] == 1:
                heuristic_scores = heuristic_scores.squeeze(1)

            pred_gain, pred_risk_total, _ = model(
                scene_token, reference_plan, residual,
                plan_confidence, safety_features, heuristic_scores
            )

            true_gain = batch['delta_safe_reward'].to(device)
            true_risk = (batch['delta_collision'] + batch['delta_offroad'] + 
                        batch['delta_comfort'] + batch['delta_drift']).to(device)

            all_pred_gain.append(pred_gain.squeeze(-1))
            all_true_gain.append(true_gain)
            all_pred_risk.append(pred_risk_total.squeeze(-1))
            all_true_risk.append(true_risk)

    # 拼接所有批次
    pred_gain = torch.cat(all_pred_gain)
    true_gain = torch.cat(all_true_gain)
    pred_risk = torch.cat(all_pred_risk)
    true_risk = torch.cat(all_true_risk)

    # 计算排序相关性 (Spearman)
    def spearman_corr(pred, true):
        pred_rank = pred.argsort().argsort().float()
        true_rank = true.argsort().argsort().float()
        return torch.corrcoef(torch.stack([pred_rank, true_rank]))[0, 1].item()

    gain_corr = spearman_corr(pred_gain, true_gain)
    risk_corr = spearman_corr(pred_risk, true_risk)

    # 计算准确率 (Top-K 匹配)
    def top_k_accuracy(pred, true, k=5):
        _, pred_topk = pred.topk(k, largest=True)
        _, true_topk = true.topk(k, largest=True)
        matches = 0
        for i in range(len(pred_topk)):
            if pred_topk[i] in true_topk[i]:
                matches += 1
        return matches / len(pred_topk)

    gain_top5_acc = top_k_accuracy(pred_gain, true_gain, k=5)
    risk_top5_acc = top_k_accuracy(pred_risk, true_risk, k=5)

    return {
        'gain_spearman_corr': gain_corr,
        'risk_spearman_corr': risk_corr,
        'gain_top5_acc': gain_top5_acc,
        'risk_top5_acc': risk_top5_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='训练 ReliabilityNet')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_file', type=str, required=True, help='训练数据文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--epochs', type=int, default=50, help='训练 epoch 数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    args = parser.parse_args()

    cfg = load_config(args.config)
    reliability_cfg = cfg.get('reliability_net', {})
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 数据集
    dataset = ScorerDataset(args.data_file)
    actual_plan_len = dataset[0]['reference_plan'].shape[0]
    if reliability_cfg.get('plan_len', actual_plan_len) != actual_plan_len:
        logger.warning(
            '配置 plan_len=%d 与数据实际 plan_len=%d 不匹配，强制使用实际值',
            reliability_cfg.get('plan_len', actual_plan_len),
            actual_plan_len,
        )
        reliability_cfg['plan_len'] = actual_plan_len

    if reliability_cfg.get('scene_dim') != dataset[0]['scene_token'].shape[-1]:
        logger.warning(
            '配置 scene_dim=%d 与数据实际 scene_dim=%d 不匹配，强制使用实际值',
            reliability_cfg.get('scene_dim'),
            dataset[0]['scene_token'].shape[-1],
        )
        reliability_cfg['scene_dim'] = dataset[0]['scene_token'].shape[-1]

    # 构建模型
    model = ReliabilityNet(
        scene_dim=reliability_cfg['scene_dim'],
        plan_len=reliability_cfg['plan_len'],
        safety_feat_dim=reliability_cfg.get('safety_feat_dim', 0),
        hidden_dim=reliability_cfg['hidden_dim'],
        dropout=reliability_cfg.get('dropout', 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=reliability_cfg['lr'],
        weight_decay=reliability_cfg.get('weight_decay', 1e-4),
    )

    def collate_fn(batch):
        batch_collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if all(v is None for v in values):
                batch_collated[key] = None
            else:
                batch_collated[key] = default_collate(values)
        return batch_collated

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 训练循环
    best_corr = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(model, dataloader, optimizer, device)

        # 验证
        val_metrics = validate_epoch(model, dataloader, device)

        logger.info(
            f'Epoch {epoch+1}/{args.epochs} | '
            f'Loss: {train_loss:.4f} | '
            f'Gain Corr: {val_metrics["gain_spearman_corr"]:.4f} | '
            f'Risk Corr: {val_metrics["risk_spearman_corr"]:.4f} | '
            f'Gain Top5: {val_metrics["gain_top5_acc"]:.4f} | '
            f'Risk Top5: {val_metrics["risk_top5_acc"]:.4f}'
        )

        # 保存最佳模型
        avg_corr = (val_metrics['gain_spearman_corr'] + val_metrics['risk_spearman_corr']) / 2
        if avg_corr > best_corr:
            best_corr = avg_corr
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f'保存最佳模型，相关性: {best_corr:.4f}')

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    logger.info('训练完成！')


if __name__ == '__main__':
    main()