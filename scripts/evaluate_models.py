"""评估不同实验的 Refiner 模型性能。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/mnt/cpfs/prediction/lipeinan/RL')

from E2E_RL.data.dataloader import build_vad_dataloader
from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.reward_proxy import compute_refinement_reward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_refiner(checkpoint_path: str, device: torch.device) -> InterfaceRefiner:
    """加载 Refiner 模型。"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 从 checkpoint 推断模型配置
    state_dict = ckpt['refiner_state_dict']
    
    # 根据权重形状推断维度
    scene_dim = state_dict['scene_proj.weight'].shape[1]
    plan_len = state_dict['plan_proj.weight'].shape[1] // 2  # [hidden, T*2]
    hidden_dim = state_dict['scene_proj.weight'].shape[0]
    
    refiner = InterfaceRefiner(
        scene_dim=scene_dim,
        plan_len=plan_len,
        hidden_dim=hidden_dim,
    )
    refiner.load_state_dict(state_dict)
    refiner.to(device)
    refiner.eval()
    
    logger.info(f"加载模型: {checkpoint_path}")
    logger.info(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    logger.info(f"  参数量: {sum(p.numel() for p in refiner.parameters()):,}")
    
    return refiner


def evaluate_model(
    refiner: InterfaceRefiner,
    dataloader,
    device: torch.device,
    reward_config: Dict,
    max_batches: int = None,
) -> Dict[str, float]:
    """评估单个模型。"""
    refiner.eval()
    
    metrics = {
        'loss_traj': 0.0,
        'mean_reward': 0.0,
        'reward_progress': 0.0,
        'reward_collision': 0.0,
        'reward_offroad': 0.0,
        'reward_comfort': 0.0,
        'num_samples': 0,
    }
    
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            interface = batch['interface']
            gt_plan = batch['gt_plan'].to(device)
            plan_mask = batch['plan_mask'].to(device)
            
            # 前向传播
            refined_plan = refiner(interface, plan_mask)
            
            # 计算轨迹损失
            loss_traj = torch.nn.functional.l1_loss(
                refined_plan[plan_mask.bool()], 
                gt_plan[plan_mask.bool()]
            ).item()
            
            # 计算奖励
            reward_info = compute_refinement_reward(
                refined_plan=refined_plan,
                gt_plan=gt_plan,
                mask=plan_mask,
                **reward_config,
            )
            
            # 累积指标
            metrics['loss_traj'] += loss_traj
            metrics['mean_reward'] += reward_info['total_reward'].mean().item()
            metrics['reward_progress'] += reward_info.get('progress_reward', torch.tensor(0.0)).mean().item()
            metrics['reward_collision'] += reward_info.get('collision_penalty', torch.tensor(0.0)).mean().item()
            metrics['reward_offroad'] += reward_info.get('offroad_penalty', torch.tensor(0.0)).mean().item()
            metrics['reward_comfort'] += reward_info.get('comfort_penalty', torch.tensor(0.0)).mean().item()
            metrics['num_samples'] += len(refined_plan)
            
            num_batches += 1
    
    # 平均化
    for key in ['loss_traj', 'mean_reward', 'reward_progress', 
                'reward_collision', 'reward_offroad', 'reward_comfort']:
        metrics[key] /= num_batches
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='评估 Refiner 模型')
    parser.add_argument(
        '--experiments_dir', 
        type=str,
        default='/mnt/cpfs/prediction/lipeinan/RL/experiments',
        help='实验目录'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps',
        help='数据目录'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='批大小'
    )
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='最大评估批次数（None=全部）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备'
    )
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    dataloader = build_vad_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )
    logger.info(f"数据集大小: {len(dataloader.dataset)}")
    
    # 奖励配置
    reward_config = {
        'agent_positions': None,
        'agent_velocities': None,
        'lane_boundaries': None,
        'dt': 0.5,
        'w_progress': 1.0,
        'w_collision': 0.5,
        'w_offroad': 0.3,
        'w_comfort': 0.1,
    }
    
    # 扫描实验目录
    experiments_dir = Path(args.experiments_dir)
    experiments = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
    
    results = {}
    
    for exp_dir in experiments:
        exp_name = exp_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"评估实验: {exp_name}")
        logger.info(f"{'='*60}")
        
        # 查找 checkpoint
        final_ckpt = exp_dir / 'checkpoint_final.pth'
        supervised_ckpt = exp_dir / 'checkpoint_supervised.pth'
        
        if not final_ckpt.exists():
            logger.warning(f"  跳过: 未找到 checkpoint_final.pth")
            continue
        
        # 加载并评估最终模型
        try:
            refiner = load_refiner(str(final_ckpt), device)
            metrics = evaluate_model(
                refiner, 
                dataloader, 
                device, 
                reward_config,
                max_batches=args.max_batches,
            )
            results[exp_name] = metrics
            
            logger.info(f"\n结果:")
            logger.info(f"  Trajectory Loss: {metrics['loss_traj']:.4f}")
            logger.info(f"  Mean Reward:     {metrics['mean_reward']:.4f}")
            logger.info(f"    - Progress:    {metrics['reward_progress']:.4f}")
            logger.info(f"    - Collision:   {metrics['reward_collision']:.4f}")
            logger.info(f"    - Offroad:     {metrics['reward_offroad']:.4f}")
            logger.info(f"    - Comfort:     {metrics['reward_comfort']:.4f}")
            
        except Exception as e:
            logger.error(f"  评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印对比表格
    logger.info(f"\n\n{'='*60}")
    logger.info("实验对比总结")
    logger.info(f"{'='*60}")
    
    if results:
        # 表头
        print(f"\n{'Experiment':<25} {'Loss':>8} {'Reward':>8} {'Progress':>10} {'Collision':>10} {'Offroad':>8} {'Comfort':>8}")
        print("-" * 85)
        
        for exp_name, metrics in sorted(results.items()):
            print(
                f"{exp_name:<25} "
                f"{metrics['loss_traj']:>8.4f} "
                f"{metrics['mean_reward']:>8.4f} "
                f"{metrics['reward_progress']:>10.4f} "
                f"{metrics['reward_collision']:>10.4f} "
                f"{metrics['reward_offroad']:>8.4f} "
                f"{metrics['reward_comfort']:>8.4f}"
            )
        
        # 找出最佳模型
        best_by_reward = max(results.items(), key=lambda x: x[1]['mean_reward'])
        best_by_loss = min(results.items(), key=lambda x: x[1]['loss_traj'])
        
        print(f"\n✅ Best by Reward: {best_by_reward[0]} (Reward={best_by_reward[1]['mean_reward']:.4f})")
        print(f"✅ Best by Loss:   {best_by_loss[0]} (Loss={best_by_loss[1]['loss_traj']:.4f})")
    else:
        logger.warning("没有成功评估的实验")


if __name__ == '__main__':
    main()
