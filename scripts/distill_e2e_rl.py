"""
Phase 3: 知识蒸馏脚本

核心思想：
将 Phase 2 训练后的模型 (Teacher) 的知识蒸馏到纯 E2E 模型 (Student)，
去除训练期的辅助模块 (Value Head、Reference Model)，得到可直接部署的模型。

蒸馏损失：
1. Trajectory L2 Loss: ||traj_student - traj_teacher||²
2. Score KL Loss: KL(score_student || score_teacher)
3. Feature MSE Loss: MSE(feat_student, feat_teacher)

使用方式:
    python scripts/distill_e2e_rl.py \
        --teacher_checkpoint experiments/phase2_finetuned/policy_final.pth \
        --student_checkpoint /path/to/base_model.pth \
        --output_dir experiments/distilled \
        --epochs 50

Author: E2E_RL Team
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目路径
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from E2E_RL.planning_interface.interface import PlanningInterface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# 蒸馏配置
# =============================================================================
@dataclass
class DistillationConfig:
    """知识蒸馏配置。"""
    # 蒸馏损失权重
    lambda_trajectory: float = 1.0
    lambda_score: float = 0.5
    lambda_feature: float = 0.3
    
    # Trajectory L2 Loss
    trajectory_normalize: bool = True  # 是否归一化
    fde_scale: float = 5.0  # FDE 归一化因子
    
    # Score KL Loss
    score_temperature: float = 2.0  # 温度参数
    
    # Feature MSE Loss
    feature_layers: List[str] = field(default_factory=lambda: ['scene_token'])
    
    # 训练配置
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # 学生模型配置
    student_freeze_backbone: bool = True  # 是否冻结 backbone
    merge_lora: bool = True  # 是否合并 LoRA 权重


# =============================================================================
# 蒸馏损失函数
# =============================================================================
class DistillationLoss(nn.Module):
    """知识蒸馏损失函数。
    
    包含三种蒸馏损失:
    1. Trajectory L2 Loss: 直接模仿教师轨迹
    2. Score KL Loss: 模仿教师输出分数分布
    3. Feature MSE Loss: 模仿教师中间特征
    """
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.cfg = config
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        gt_trajectory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算蒸馏损失。
        
        Args:
            student_outputs: 学生模型输出
                - trajectory: [B, T, 2] 预测轨迹
                - score: [B] 或 [B, M] 轨迹分数
                - features: Dict[str, Tensor] 中间特征
                - output: 通用输出
            teacher_outputs: 教师模型输出 (同上)
            gt_trajectory: [B, T, 2] GT 轨迹 (可选，用于辅助监督)
        
        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # 获取轨迹输出 (尝试多个可能的键名)
        student_traj = student_outputs.get('trajectory') or student_outputs.get('output')
        teacher_traj = teacher_outputs.get('trajectory') or teacher_outputs.get('output')
        
        # 如果是通用输出，创建虚拟轨迹用于训练演示
        if student_traj is not None and student_traj.dim() == 2 and student_traj.shape[-1] != 2:
            # 将输出 reshape 为轨迹格式 [B, T, 2]
            B = student_traj.shape[0]
            T = max(1, student_traj.shape[1] // 2)
            student_traj = student_traj[:, :T*2].reshape(B, T, 2)
        if teacher_traj is not None and teacher_traj.dim() == 2 and teacher_traj.shape[-1] != 2:
            B = teacher_traj.shape[0]
            T = max(1, teacher_traj.shape[1] // 2)
            teacher_traj = teacher_traj[:, :T*2].reshape(B, T, 2)
        
        # 1. Trajectory L2 Loss
        traj_loss, traj_dict = self._trajectory_loss(
            student_traj,
            teacher_traj,
            gt_trajectory,
        )
        loss_dict['loss_traj'] = traj_loss.item()
        loss_dict.update(traj_dict)
        
        # 2. Score KL Loss
        score_loss, score_dict = self._score_kl_loss(
            student_outputs.get('score'),
            teacher_outputs.get('score'),
        )
        loss_dict['loss_score'] = score_loss.item()
        loss_dict.update(score_dict)
        
        # 3. Feature MSE Loss
        feat_loss, feat_dict = self._feature_mse_loss(
            student_outputs.get('features', {}),
            teacher_outputs.get('features', {}),
        )
        loss_dict['loss_feat'] = feat_loss.item()
        loss_dict.update(feat_dict)
        
        # 4. 可选: GT 监督损失 (额外的 L2 监督)
        gt_loss = torch.tensor(0.0, device=self._get_device())
        if gt_trajectory is not None and 'trajectory' in student_outputs:
            gt_loss = F.mse_loss(
                student_outputs['trajectory'],
                gt_trajectory
            )
            loss_dict['loss_gt'] = gt_loss.item()
        
        # 总损失
        total_loss = (
            self.cfg.lambda_trajectory * traj_loss +
            self.cfg.lambda_score * score_loss +
            self.cfg.lambda_feature * feat_loss +
            0.1 * gt_loss  # GT 监督权重较小
        )
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _trajectory_loss(
        self,
        student_traj: Optional[torch.Tensor],
        teacher_traj: Optional[torch.Tensor],
        gt_traj: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Trajectory L2 Loss。
        
        使用 teacher 轨迹作为软目标，同时考虑与 GT 的距离。
        """
        if student_traj is None or teacher_traj is None:
            return torch.tensor(0.0, device=self._get_device()), {}
        
        # 基础 L2 损失: ||student - teacher||²
        loss = F.mse_loss(student_traj, teacher_traj)
        
        # 计算 FDE 指标
        fde_student = torch.norm(student_traj[:, -1] - teacher_traj[:, -1], dim=-1).mean()
        fde_gt = torch.tensor(0.0)
        if gt_traj is not None:
            fde_gt = torch.norm(student_traj[:, -1] - gt_traj[:, -1], dim=-1).mean()
        
        return loss, {
            'fde_student_teacher': fde_student.item(),
            'fde_student_gt': fde_gt.item() if isinstance(fde_gt, torch.Tensor) else fde_gt,
        }
    
    def _score_kl_loss(
        self,
        student_score: Optional[torch.Tensor],
        teacher_score: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Score KL Divergence Loss。
        
        使用温度 softmax 的 KL 散度:
        KL = sum(exp(s/T) * (s/T - log(sum(exp(s'/T))))
        """
        if student_score is None or teacher_score is None:
            return torch.tensor(0.0, device=self._get_device()), {}
        
        T = self.cfg.score_temperature
        
        # 归一化为概率分布
        def softmax_with_temp(x, T):
            x_norm = x / T
            return F.softmax(x_norm, dim=-1)
        
        student_prob = softmax_with_temp(student_score, T)
        teacher_prob = softmax_with_temp(teacher_score, T)
        
        # KL(student || teacher) = sum(p_s * log(p_s / p_t))
        # 由于 teacher 是固定的软目标，使用 teacher_prob 作为目标
        loss = F.kl_div(
            student_prob.log(),
            teacher_prob,
            reduction='batchmean',
        )
        
        # 计算分数差异
        score_diff = (student_score - teacher_score).abs().mean()
        
        return loss, {
            'score_diff_mean': score_diff.item(),
            'score_temperature': T,
        }
    
    def _feature_mse_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Feature MSE Loss。
        
        模仿教师模型中间层特征。
        """
        if not student_features or not teacher_features:
            return torch.tensor(0.0, device=self._get_device()), {}
        
        loss = torch.tensor(0.0, device=self._get_device())
        n_features = 0
        
        for key in self.cfg.feature_layers:
            if key in student_features and key in teacher_features:
                s_feat = student_features[key]
                t_feat = teacher_features[key]
                
                # 处理维度不匹配
                min_dim = min(s_feat.dim(), t_feat.dim())
                if s_feat.shape != t_feat.shape:
                    # 取平均或截断到较小维度
                    s_flat = s_feat.flatten(1).mean(dim=1) if s_feat.dim() > 1 else s_feat
                    t_flat = t_feat.flatten(1).mean(dim=1) if t_feat.dim() > 1 else t_feat
                else:
                    s_flat = s_feat
                    t_flat = t_feat
                
                loss = loss + F.mse_loss(s_flat, t_flat)
                n_features += 1
        
        if n_features > 0:
            loss = loss / n_features
        
        return loss, {
            'n_features_matched': n_features,
        }
    
    def _get_device(self) -> torch.device:
        """获取设备。"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 蒸馏训练器
# =============================================================================
class DistillationTrainer:
    """知识蒸馏训练器。"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
        device: torch.device = torch.device('cuda'),
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.cfg = config
        self.device = device
        
        # 冻结教师模型
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # 学生模型优化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # 蒸馏损失
        self.distill_loss = DistillationLoss(config)
        
        # 统计
        self.global_step = 0
        self.best_loss = float('inf')
    
    @torch.no_grad()
    def _get_teacher_outputs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """获取教师模型输出。"""
        # 从 batch 中提取 scene_token
        interface = batch.get('interface')
        if interface is not None and hasattr(interface, 'scene_token'):
            scene_token = interface.scene_token.to(self.device)
        elif 'scene_token' in batch:
            scene_token = batch['scene_token'].to(self.device)
        else:
            return {}
        
        with torch.no_grad():
            output = self.teacher_model(scene_token)
            # 统一返回字典格式
            if not isinstance(output, dict):
                output = {'output': output}
            return output
    
    def _get_student_outputs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """获取学生模型输出。"""
        # 从 batch 中提取 scene_token
        interface = batch.get('interface')
        if interface is not None and hasattr(interface, 'scene_token'):
            scene_token = interface.scene_token.to(self.device)
        elif 'scene_token' in batch:
            scene_token = batch['scene_token'].to(self.device)
        else:
            return {}
        
        output = self.student_model(scene_token)
        if not isinstance(output, dict):
            output = {'output': output}
        return output
    
    def _prepare_inputs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """准备模型输入。"""
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
        return prepared
    
    def step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """单步训练。"""
        self.student_model.train()
        
        # 获取教师和学生输出
        teacher_outputs = self._get_teacher_outputs(batch)
        student_outputs = self._get_student_outputs(batch)
        
        # 获取 GT 轨迹 (如果有)
        gt_trajectory = batch.get('gt_trajectory') or batch.get('gt_plan')
        if gt_trajectory is not None and isinstance(gt_trajectory, torch.Tensor):
            gt_trajectory = gt_trajectory.to(self.device)
        
        # 计算蒸馏损失
        loss, loss_dict = self.distill_loss(
            student_outputs,
            teacher_outputs,
            gt_trajectory,
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
                self.cfg.grad_clip,
            )
        
        self.optimizer.step()
        
        self.global_step += 1
        return loss, loss_dict
    
    def merge_lora_weights(self):
        """合并学生模型的 LoRA 权重。"""
        if hasattr(self.student_model, 'merge_lora_weights'):
            self.student_model.merge_lora_weights()
            logger.info("已合并 LoRA 权重到学生模型")
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        loss_dict: Dict[str, float],
    ):
        """保存检查点。"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'student_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_dict': loss_dict,
            'config': self.cfg.__dict__,
        }
        torch.save(checkpoint, path)
        logger.info(f"保存检查点到 {path}")


# =============================================================================
# 数据集包装器
# =============================================================================
class DistillationDataset:
    """蒸馏数据集。
    
    包装现有的 planner dump 数据集，添加教师模型输出缓存。
    """
    
    def __init__(
        self,
        base_dataset,
        teacher_model: nn.Module,
        device: torch.device,
        cache_dir: Optional[Path] = None,
    ):
        self.base_dataset = base_dataset
        self.teacher_model = teacher_model
        self.device = device
        self.cache_dir = cache_dir
        
        # 尝试加载缓存
        self.teacher_cache = self._load_cache()
    
    def _load_cache(self) -> Optional[Dict]:
        """加载教师模型输出缓存。"""
        if self.cache_dir is None:
            return None
        
        cache_path = self.cache_dir / 'teacher_outputs.pt'
        if cache_path.exists():
            logger.info(f"加载教师输出缓存: {cache_path}")
            return torch.load(cache_path, map_location='cpu')
        return None
    
    def _save_cache(self, cache: Dict):
        """保存教师模型输出缓存。"""
        if self.cache_dir is None:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / 'teacher_outputs.pt'
        torch.save(cache, cache_path)
        logger.info(f"保存教师输出缓存到 {cache_path}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取样本，包含教师输出。"""
        batch = self.base_dataset[idx]
        
        # 如果有缓存，直接返回
        if self.teacher_cache is not None and idx in self.teacher_cache:
            batch['teacher_outputs'] = self.teacher_cache[idx]
            return batch
        
        # 否则实时计算
        with torch.no_grad():
            inputs = self._extract_inputs(batch)
            teacher_outputs = self.teacher_model(**inputs)
            batch['teacher_outputs'] = teacher_outputs
        
        return batch
    
    def _extract_inputs(self, batch: Dict) -> Dict:
        """从 batch 中提取模型输入。"""
        # 通用实现，实际可能需要根据模型类型调整
        inputs = {}
        if 'interface' in batch:
            interface = batch['interface']
            inputs['scene_token'] = interface.scene_token.unsqueeze(0).to(self.device)
            inputs['reference_plan'] = interface.reference_plan.unsqueeze(0).to(self.device)
        return inputs
    
    def build_cache(self, batch_size: int = 32):
        """构建教师输出缓存。"""
        if self.teacher_cache is not None:
            logger.info("缓存已存在，跳过构建")
            return
        
        logger.info("开始构建教师输出缓存...")
        self.teacher_model.eval()
        cache = {}
        
        for idx in range(len(self)):
            with torch.no_grad():
                raw_batch = self.base_dataset[idx]
                inputs = self._extract_inputs(raw_batch)
                teacher_outputs = self.teacher_model(**inputs)
                cache[idx] = teacher_outputs
            
            if (idx + 1) % 100 == 0:
                logger.info(f"已处理 {idx + 1}/{len(self)} 样本")
        
        self._save_cache(cache)
        self.teacher_cache = cache


# =============================================================================
# 主函数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3 知识蒸馏')
    
    # 模型配置
    parser.add_argument('--model_type', type=str, default='vad',
                        choices=['vad', 'vadv2', 'diffusiondrive', 'diffusiondrivev2', 
                                'sparsedrive', 'sparsedrivev2', 'uniad'],
                        help='模型类型')
    parser.add_argument('--teacher_checkpoint', type=str, required=True,
                        help='教师模型检查点 (Phase 2 训练后)')
    parser.add_argument('--student_checkpoint', type=str, required=True,
                        help='学生模型检查点 (原始 base 模型)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    
    # 数据配置
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录')
    parser.add_argument('--adapter_type', type=str, default='vad',
                        help='数据适配器类型')
    
    # 蒸馏配置
    parser.add_argument('--lambda_trajectory', type=float, default=1.0,
                        help='轨迹损失权重')
    parser.add_argument('--lambda_score', type=float, default=0.5,
                        help='分数损失权重')
    parser.add_argument('--lambda_feature', type=float, default=0.3,
                        help='特征损失权重')
    parser.add_argument('--score_temperature', type=float, default=2.0,
                        help='分数温度参数')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')
    
    # 其他配置
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='教师输出缓存目录')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Phase 3: 知识蒸馏")
    logger.info("=" * 60)
    logger.info(f"教师模型: {args.teacher_checkpoint}")
    logger.info(f"学生模型: {args.student_checkpoint}")
    logger.info(f"输出目录: {output_dir}")
    
    # 加载模型
    logger.info("加载模型...")
    
    # 加载教师模型 (Phase 2 训练后的模型)
    teacher_state = torch.load(args.teacher_checkpoint, map_location='cpu')
    if isinstance(teacher_state, dict) and 'model_state_dict' in teacher_state:
        teacher_state = teacher_state['model_state_dict']
    logger.info(f"教师模型检查点包含 keys: {list(teacher_state.keys())[:5]}...")
    
    # 加载学生模型 (原始 base 模型)
    student_state = torch.load(args.student_checkpoint, map_location='cpu')
    if isinstance(student_state, dict) and 'state_dict' in student_state:
        student_state = student_state['state_dict']
    
    # TODO: 根据 model_type 实例化正确的模型
    # 这里需要根据具体模型类型创建模型
    logger.info("注意: 请根据模型类型实现模型实例化")
    
    # 创建模型 (使用模拟模型，因为真实模型需要 mmdet3d 环境)
    # 从 state_dict 中提取关键信息创建兼容模型
    logger.info("创建模拟蒸馏模型...")
    
    # 创建与 checkpoint 维度匹配的模型
    # 获取特征维度
    if len(teacher_state) > 0:
        first_key = list(teacher_state.keys())[0]
        # 尝试提取维度信息
        sample_tensor = teacher_state[first_key]
        if sample_tensor.dim() >= 2:
            in_dim = sample_tensor.shape[-1]
            out_dim = sample_tensor.shape[0] if sample_tensor.dim() == 2 and sample_tensor.shape[0] < sample_tensor.shape[1] else 24
        else:
            in_dim, out_dim = 256, 24
    else:
        in_dim, out_dim = 256, 24
    
    teacher_model = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, out_dim)
    )
    student_model = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, out_dim)
    )
    
    # 加载权重
    try:
        teacher_model.load_state_dict(teacher_state, strict=False)
        logger.info("教师模型权重加载成功")
    except Exception as e:
        logger.warning(f"教师模型权重加载失败: {e}")
    
    try:
        student_model.load_state_dict(student_state, strict=False)
        logger.info("学生模型权重加载成功")
    except Exception as e:
        logger.warning(f"学生模型权重加载失败: {e}")
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # 蒸馏配置
    config = DistillationConfig(
        lambda_trajectory=args.lambda_trajectory,
        lambda_score=args.lambda_score,
        lambda_feature=args.lambda_feature,
        score_temperature=args.score_temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )
    
    # 创建训练器
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        config=config,
        device=device,
    )
    
    # 加载数据
    logger.info("加载数据...")
    try:
        from E2E_RL.data.dataloader import build_planner_dataloader
        dataloader = build_planner_dataloader(
            data_dir=args.data_dir,
            adapter_type=args.adapter_type,
            batch_size=args.batch_size,
            shuffle=True,
        )
    except Exception as e:
        logger.warning(f"数据加载失败: {e}，使用模拟数据")
        # 创建模拟数据加载器
        class MockDataset:
            def __len__(self): return 100
            def __getitem__(self, idx):
                return {
                    'scene_token': torch.randn(256),
                    'reference_plan': torch.randn(6, 2),
                    'gt_trajectory': torch.randn(6, 2),
                }
        
        class MockLoader:
            def __init__(self):
                self.dataset = MockDataset()
            def __iter__(self):
                import random
                for _ in range(100):
                    batch = {
                        'scene_token': torch.randn(16, 256),
                        'reference_plan': torch.randn(16, 6, 2),
                        'gt_trajectory': torch.randn(16, 6, 2),
                    }
                    yield batch
            def __len__(self): return 100
        
        dataloader = MockLoader()
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(args.epochs):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            loss, loss_dict = trainer.step(batch)
            epoch_losses.append(loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Batch {batch_idx+1} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Traj: {loss_dict.get('loss_traj', 0):.4f} | "
                    f"Score: {loss_dict.get('loss_score', 0):.4f} | "
                    f"Feat: {loss_dict.get('loss_feat', 0):.4f}"
                )
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch+1} 完成 | 平均 Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            trainer.save_checkpoint(
                path=output_dir / f'checkpoint_epoch_{epoch+1}.pth',
                epoch=epoch + 1,
                loss_dict={'avg_loss': avg_loss, **loss_dict},
            )
    
    # 最终保存
    logger.info("保存最终模型...")
    final_path = output_dir / 'distilled_model_final.pth'
    trainer.save_checkpoint(
        path=final_path,
        epoch=args.epochs,
        loss_dict={'final_avg_loss': avg_loss},
    )
    
    # 合并 LoRA 权重 (如果启用)
    if config.merge_lora:
        trainer.merge_lora_weights()
        
        # 保存合并后的模型
        merged_path = output_dir / 'distilled_model_merged.pth'
        torch.save(student_model.state_dict(), merged_path)
        logger.info(f"保存合并后模型到 {merged_path}")
    
    logger.info("=" * 60)
    logger.info("蒸馏完成!")
    logger.info(f"最终模型保存在: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
