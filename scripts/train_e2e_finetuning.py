"""
Phase 2: Conservative E2E Post-Training 训练脚本

这个脚本演示如何使用 Phase 2 的核心组件进行端到端微调。

核心流程:
1. 加载预训练的 E2E 模型 (Frozen)
2. 包装为 E2E 版本 (添加 LoRA + Value Head)
3. 使用 Conservative RL 进行训练
4. 梯度回传到规划头

使用方式:
    # 单模型训练
    python scripts/train_e2e_finetuning.py \
        --model_type vad \
        --checkpoint /path/to/vad.pth \
        --output_dir experiments/e2e_finetuned

    # 多模型训练
    python scripts/train_e2e_finetuning.py \
        --model_type diffusiondrive \
        --checkpoint /path/to/dd.pth \
        --lora_rank 16 \
        --epochs 50
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目路径
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from E2E_RL.e2e_finetuning.conservative_rl import (
    ConservativeRLConfig,
    ConservativeRLUpdate,
    ConservativeE2ETrainer,
    RewardCostSeparator,
    BeneficialUpdateFilter,
)
from E2E_RL.e2e_finetuning.hydra_traj_head_e2e import (
    LoRAConfig,
    HydraTrajHeadE2E,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# 默认配置
DEFAULT_CONFIG = {
    'model': {
        'model_type': 'vad',  # vad, diffusiondrive, sparsedrive
        'checkpoint': None,
        'scene_dim': 256,
        'plan_len': 6,
    },
    'lora': {
        'enabled': True,
        'rank': 16,
        'alpha': 1.0,
        'dropout': 0.1,
    },
    'value_head': {
        'enabled': True,
        'hidden_dim': 128,
    },
    'training': {
        'epochs': 50,
        'batch_size': 16,
        'lr': 3e-5,
        'grad_clip': 1.0,
        'val_every': 5,
    },
    'conservative_rl': {
        'kl_target': 0.01,
        'use_reference_anchor': True,
        'reference_alpha': 0.5,
        'use_beneficial_filter': True,
        'reward_margin': 0.0,
        'cost_increase_threshold': 0.1,
        'kl_bound': 0.05,
    },
    'reward_cost': {
        'progress_weight': 1.0,
        'efficiency_weight': 0.5,
        'collision_weight': 1.0,
        'offroad_weight': 1.0,
        'comfort_weight': 0.3,
        'max_collision_penalty': 0.1,
        'max_offroad_penalty': 0.1,
        'max_comfort_violation': 1.0,
    },
    'data': {
        'data_dir': 'data/vad_dumps_full',
        'num_workers': 0,
    },
    'output_dir': 'experiments/e2e_finetuned',
}


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='Conservative E2E Post-Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='vad',
                        choices=['vad', 'vadv2', 'diffusiondrive', 'sparsedrive'],
                        help='模型类型')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径')
    parser.add_argument('--scene_dim', type=int, default=256,
                        help='场景特征维度')
    parser.add_argument('--plan_len', type=int, default=6,
                        help='轨迹长度')
    
    # LoRA 参数
    parser.add_argument('--no_lora', action='store_true',
                        help='禁用 LoRA (全量微调)')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank')
    
    # 训练参数
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='学习率')
    
    return parser.parse_args()


def merge_config(args):
    """合并配置。"""
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    
    if args.model_type:
        config['model']['model_type'] = args.model_type
    if args.checkpoint:
        config['model']['checkpoint'] = args.checkpoint
    if args.scene_dim:
        config['model']['scene_dim'] = args.scene_dim
    if args.plan_len:
        config['model']['plan_len'] = args.plan_len
    
    if args.no_lora:
        config['lora']['enabled'] = False
    
    config['lora']['rank'] = args.lora_rank
    
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    
    return config


class SimulatedPlanner(nn.Module):
    """
    模拟的规划器 (用于演示)
    
    实际使用时替换为真实模型。
    """
    
    def __init__(
        self,
        scene_dim: int = 256,
        plan_len: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.scene_dim = scene_dim
        self.plan_len = plan_len
        
        # 场景编码器
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 规划头
        self.planning_head = nn.Sequential(
            nn.Linear(hidden_dim + plan_len * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, plan_len * 2),  # [T, 2] 轨迹
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(
        self,
        scene_token: torch.Tensor,
        reference_plan: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。"""
        # 编码场景
        scene_feat = self.scene_encoder(scene_token)
        
        # 生成轨迹 (如果未提供)
        if reference_plan is None:
            traj_input = torch.randn(
                scene_token.shape[0], self.plan_len * 2,
                device=scene_token.device
            )
        else:
            traj_input = reference_plan.flatten(1)
        
        # 融合
        fused = torch.cat([scene_feat, traj_input], dim=-1)
        
        # 预测轨迹偏移
        traj_delta = self.planning_head(fused)
        trajectory = traj_delta.reshape(-1, self.plan_len, 2)
        
        # 如果有参考轨迹，加上去
        if reference_plan is not None:
            trajectory = trajectory + reference_plan
        
        # Value
        value = self.value_head(scene_feat).squeeze(-1)
        
        return {
            'trajectory': trajectory,
            'scene_token': scene_feat,
            'value': value,
        }


class E2EFinetuningDataModule:
    """数据模块。"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config['data']['data_dir']
    
    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """获取数据加载器。"""
        from E2E_RL.data.dataloader import build_planner_dataloader
        
        model_type = self.config['model']['model_type']
        adapter_type = model_type.lower()  # vad, diffusiondrive, etc.
        
        try:
            loader = build_planner_dataloader(
                data_dir=self.data_dir,
                adapter_type=adapter_type,
                batch_size=batch_size,
                num_workers=0,
                shuffle=shuffle,
            )
            return loader
        except Exception as e:
            logger.warning(f"无法加载真实数据: {e}")
            logger.info("使用模拟数据演示")
            return self._get_simulated_dataloader(batch_size, shuffle)
    
    def _get_simulated_dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        """生成模拟数据用于演示。"""
        from torch.utils.data import TensorDataset, DataLoader
        
        B = 1000  # 模拟 1000 个样本
        scene_dim = self.config['model']['scene_dim']
        plan_len = self.config['model']['plan_len']
        
        # 随机场景特征
        scene_tokens = torch.randn(B, scene_dim)
        
        # 随机参考轨迹
        reference_plans = torch.randn(B, plan_len, 2)
        
        # GT 轨迹 (在参考轨迹基础上加一些偏移)
        gt_plans = reference_plans + torch.randn(B, plan_len, 2) * 0.5
        
        dataset = TensorDataset(scene_tokens, reference_plans, gt_plans)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch):
        """Collate 函数。"""
        scene_tokens, reference_plans, gt_plans = zip(*batch)
        
        return {
            'scene_token': torch.stack(scene_tokens),
            'reference_plan': torch.stack(reference_plans),
            'gt_trajectory': torch.stack(gt_plans),
            'trajectory': torch.stack(reference_plans),  # 初始输出
        }


def create_e2e_model(config: Dict, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    创建 E2E 微调模型。
    
    Returns:
        (model, reference_model)
    """
    model_type = config['model']['model_type']
    scene_dim = config['model']['scene_dim']
    plan_len = config['model']['plan_len']
    
    logger.info(f"创建 {model_type} E2E 模型...")
    
    # 模拟创建模型 (实际实现需要加载真实模型)
    base_model = SimulatedPlanner(
        scene_dim=scene_dim,
        plan_len=plan_len,
    )
    
    # 加载预训练权重
    checkpoint = config['model']['checkpoint']
    if checkpoint and Path(checkpoint).exists():
        logger.info(f"加载预训练权重: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        base_model.load_state_dict(ckpt, strict=False)
    
    # 包装为 E2E 版本
    e2e_model = HydraTrajHeadE2E(
        base_head=base_model.planning_head,
        lora_config=LoRAConfig(
            enabled=config['lora']['enabled'],
            rank=config['lora']['rank'],
            alpha=config['lora']['alpha'],
        ) if config['lora']['enabled'] else None,
        enable_value_head=config['value_head']['enabled'],
        scene_dim=scene_dim,
        plan_len=plan_len,
    )
    
    # 创建 Reference Model (原始模型副本)
    reference_model = copy.deepcopy(base_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    return base_model.to(device), reference_model.to(device)


def train_epoch(
    trainer: ConservativeE2ETrainer,
    dataloader: DataLoader,
    epoch: int,
    device: torch.device,
) -> Dict:
    """训练一个 epoch。"""
    trainer.model.train()
    
    epoch_metrics = {
        'loss_total': 0.0,
        'loss_policy': 0.0,
        'loss_ref': 0.0,
        'mean_reward': 0.0,
        'mean_cost': 0.0,
        'mean_advantage': 0.0,
        'safety_ratio': 0.0,
    }
    num_batches = 0
    
    for batch in dataloader:
        # 从 batch 中提取数据 (根据 dataloader 实际返回的键)
        interface = batch.get('interface')  # PlanningInterface
        gt_plan = batch.get('gt_plan')     # [B, T, 2] GT 轨迹
        
        if interface is not None:
            scene_token = interface.scene_token.to(device) if hasattr(interface, 'scene_token') else None
            reference_plan = interface.reference_plan.to(device) if hasattr(interface, 'reference_plan') else None
        else:
            scene_token = None
            reference_plan = None
        
        gt_trajectory = gt_plan.to(device) if gt_plan is not None else None
        
        if scene_token is None or reference_plan is None or gt_trajectory is None:
            logger.warning(f"跳过 batch: interface={interface is not None}, gt_plan={gt_plan is not None}")
            continue
        
        # 前向传播
        outputs = trainer.model(scene_token, reference_plan)
        trajectory = outputs.get('trajectory', reference_plan)  # fallback to reference
        
        # 构建 batch dict
        batch_dict = {
            'trajectory': trajectory,
            'gt_trajectory': gt_trajectory,
            'interface': type('Interface', (), {
                'scene_token': outputs['scene_token'],
                'reference_plan': reference_plan,
            })(),
        }
        
        # 训练 step
        loss, diag = trainer.step(batch_dict)
        
        # 记录
        for key in epoch_metrics:
            if key in diag:
                epoch_metrics[key] += diag[key]
        num_batches += 1
    
    # 平均
    for key in epoch_metrics:
        epoch_metrics[key] /= max(num_batches, 1)
    
    logger.info(
        f"[Epoch {epoch}] "
        f"loss={epoch_metrics['loss_total']:.4f} "
        f"reward={epoch_metrics['mean_reward']:.4f} "
        f"cost={epoch_metrics['mean_cost']:.4f} "
        f"safety={epoch_metrics['safety_ratio']:.2%}"
    )
    
    return epoch_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict:
    """验证。"""
    model.eval()
    
    total_reward = 0.0
    total_cost = 0.0
    total_fde = 0.0
    num_samples = 0
    
    rc_separator = RewardCostSeparator()
    
    for batch in dataloader:
        # 从 batch 中提取数据
        interface = batch.get('interface')
        gt_plan = batch.get('gt_plan')
        
        if interface is not None:
            scene_token = interface.scene_token.to(device) if hasattr(interface, 'scene_token') else None
            reference_plan = interface.reference_plan.to(device) if hasattr(interface, 'reference_plan') else None
        else:
            scene_token = batch.get('scene_token', torch.zeros(1, 256)).to(device)
            reference_plan = batch.get('reference_plan', torch.zeros(1, 6, 2)).to(device)
        
        gt_trajectory = gt_plan.to(device) if gt_plan is not None else None
        
        if scene_token is None:
            continue
        
        # 前向传播
        outputs = model(scene_token, reference_plan)
        trajectory = outputs.get('trajectory', reference_plan)
        
        # 计算指标
        rc_output = rc_separator.compute(
            trajectory=trajectory,
            gt_trajectory=gt_trajectory,
        )
        
        # FDE
        fde = torch.norm(trajectory[:, -1] - gt_trajectory[:, -1], dim=-1)
        
        total_reward += rc_output['reward_branch'].sum().item()
        total_cost += rc_output['cost_branch'].sum().item()
        total_fde += fde.sum().item()
        num_samples += scene_token.shape[0]
    
    return {
        'mean_reward': total_reward / num_samples,
        'mean_cost': total_cost / num_samples,
        'mean_fde': total_fde / num_samples,
    }


def main():
    """主函数。"""
    # 解析参数
    args = parse_args()
    config = merge_config(args)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("Conservative E2E Post-Training")
    logger.info("=" * 60)
    logger.info(json.dumps(config, indent=2, default=str))
    logger.info("=" * 60)
    
    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # 数据模块
    data_module = E2EFinetuningDataModule(config)
    train_loader = data_module.get_dataloader(
        batch_size=config['training']['batch_size'],
        shuffle=True,
    )
    val_loader = data_module.get_dataloader(
        batch_size=config['training']['batch_size'],
        shuffle=False,
    )
    
    # 创建模型
    model, reference_model = create_e2e_model(config, device)
    
    # 创建训练器
    crl_config = ConservativeRLConfig(
        lr=config['training']['lr'],
        kl_target=config['conservative_rl']['kl_target'],
        use_reference_anchor=config['conservative_rl']['use_reference_anchor'],
        reference_alpha=config['conservative_rl']['reference_alpha'],
        use_beneficial_filter=config['conservative_rl']['use_beneficial_filter'],
        reward_margin=config['conservative_rl']['reward_margin'],
        cost_increase_threshold=config['conservative_rl']['cost_increase_threshold'],
        kl_bound=config['conservative_rl']['kl_bound'],
        grad_clip=config['training']['grad_clip'],
    )
    
    trainer = ConservativeE2ETrainer(
        model=model,
        reference_model=reference_model,
        config=crl_config,
        device=device,
    )
    
    # 训练循环
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)
    
    all_metrics = []
    
    for epoch in range(config['training']['epochs']):
        # 训练
        train_metrics = train_epoch(trainer, train_loader, epoch, device)
        all_metrics.append(train_metrics)
        
        # 验证
        if (epoch + 1) % config['training']['val_every'] == 0:
            val_metrics = validate(model, val_loader, device)
            logger.info(
                f"[Val] reward={val_metrics['mean_reward']:.4f} "
                f"cost={val_metrics['mean_cost']:.4f} "
                f"FDE={val_metrics['mean_fde']:.2f}m"
            )
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
            }, ckpt_path)
            logger.info(f"保存检查点: {ckpt_path}")
    
    # 最终保存
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': all_metrics,
    }, final_path)
    logger.info(f"最终模型已保存: {final_path}")
    
    # 保存 metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    logger.info("训练完成!")


if __name__ == '__main__':
    main()
