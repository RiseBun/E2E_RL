#!/usr/bin/env python3
"""
测试 VAD E2E 包装器

验证:
1. VAD 模型加载
2. VADModelE2E 包装
3. LoRA 参数正确添加
4. 前向传播输出
"""

import sys
import os

# 添加项目路径
PROJECT_ROOT = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL'
VAD_ROOT = os.path.join(PROJECT_ROOT, 'projects/VAD')
VAD_PROJECTS = os.path.join(VAD_ROOT, 'projects')

# VAD 项目结构: projects/VAD/projects/mmdet3d_plugin/VAD/
sys.path.insert(0, PROJECT_ROOT)       # 访问 projects.VAD
sys.path.insert(0, VAD_ROOT)            # 访问 VAD 包
sys.path.insert(0, VAD_PROJECTS)        # 访问 projects 包

import torch

def test_vad_model_load():
    """测试 1: VAD 模型加载"""
    print("=" * 60)
    print("测试 1: VAD 模型加载 (checkpoint 分析)")
    print("=" * 60)
    
    # 加载 checkpoint
    checkpoint_path = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/VAD/VAD_base.pth'
    state = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint keys: {list(state.keys())}")
    print(f"Meta info: {state.get('meta', {})}")
    
    # 分析 state_dict
    sd = state['state_dict']
    
    # 查找 VADHead 相关参数
    head_keys = [k for k in sd.keys() if 'pts_bbox_head' in k]
    print(f"\nVADHead 参数 ({len(head_keys)} 个)")
    
    # 查找规划头参数
    ego_fut_keys = [k for k in head_keys if 'ego_fut' in k]
    print(f"\nego_fut_decoder 参数 ({len(ego_fut_keys)} 个):")
    for k in ego_fut_keys:
        print(f"  {k.split('.')[-2]}.{k.split('.')[-1]}: {sd[k].shape}")
    
    print(f"\n总参数量: {sum(v.numel() for v in sd.values()):,}")
    print(f"规划头参数量: {sum(sd[k].numel() for k in ego_fut_keys):,}")
    
    print("\n✓ 模型加载成功")
    return state

def test_vad_e2e_wrapper():
    """测试 2: VAD E2E 包装器"""
    print("\n" + "=" * 60)
    print("测试 2: VAD E2E 包装器")
    print("=" * 60)
    
    # 加载模块
    from e2e_finetuning.vad_e2e_wrapper import (
        VADE2EConfig, VADHeadE2E, VADModelE2E,
        wrap_vad_head, wrap_vad_model
    )
    
    # 测试配置
    config = VADE2EConfig(
        lora_enabled=True,
        lora_rank=16,
        lora_alpha=1.0,
        enable_value_head=True,
        value_hidden_dim=128,
    )
    
    print(f"配置: {config}")
    
    # 创建模拟的 VADHead
    class MockVADHead:
        def __init__(self):
            self.ego_fut_mode = 3
            self.fut_ts = 6
            self.embed_dims = 256
            self.ego_lcf_feat_idx = None
            
            # 模拟 ego_fut_decoder
            self.ego_fut_decoder = torch.nn.Sequential(
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 36),  # 3 * 6 * 2 = 36
            )
    
    vad_head = MockVADHead()
    print(f"\n模拟 VADHead:")
    print(f"  ego_fut_mode: {vad_head.ego_fut_mode}")
    print(f"  fut_ts: {vad_head.fut_ts}")
    print(f"  embed_dims: {vad_head.embed_dims}")
    
    # 测试 VADHeadE2E
    e2e_head = VADHeadE2E(vad_head, config)
    
    print(f"\nVADHeadE2E 创建成功:")
    print(f"  LoRA 启用: {e2e_head.lora_ego_fut_decoder is not None}")
    print(f"  Value Head: {e2e_head.value_head is not None}")
    
    # 测试前向传播
    ego_feats = torch.randn(2, 1, 512)
    base_outputs = {
        'ego_fut_preds': torch.randn(2, 3, 6, 2),
        'bev_embed': torch.randn(2, 256, 50, 50),
    }
    
    enhanced_outputs = e2e_head.forward_with_base_output(base_outputs, ego_feats)
    
    print(f"\n前向传播输出:")
    for k, v in enhanced_outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")
    
    # 测试规划接口提取
    interface = e2e_head.extract_planning_interface(enhanced_outputs)
    if interface is not None:
        print(f"\nPlanningInterface:")
        print(f"  scene_token: {interface.scene_token.shape}")
        print(f"  reference_plan: {interface.reference_plan.shape}")
        print(f"  candidate_plans: {interface.candidate_plans.shape if interface.candidate_plans is not None else None}")
    
    # 测试可训练参数
    trainable_params = e2e_head.get_trainable_parameters()
    print(f"\n可训练参数: {len(trainable_params)} 个")
    print(f"可训练参数量: {sum(p.numel() for p in trainable_params):,}")
    
    print("\n✓ VADHeadE2E 测试通过")
    return True

def test_lora_integration():
    """测试 3: LoRA 集成到 VADHead"""
    print("\n" + "=" * 60)
    print("测试 3: LoRA 集成")
    print("=" * 60)
    
    from e2e_finetuning.hydra_traj_head_e2e import LoRALinear
    import torch.nn as nn
    
    # 创建原始层
    original_linear = nn.Linear(512, 36)
    print(f"原始 Linear: in_features=512, out_features=36")
    print(f"  参数量: {sum(p.numel() for p in original_linear.parameters()):,}")
    
    # 创建 LoRA wrapper
    lora_linear = LoRALinear(
        base_layer=original_linear,
        rank=16,
        alpha=1.0,
        dropout=0.0,
    )
    
    print(f"\nLoRA Linear:")
    print(f"  参数量: {sum(p.numel() for p in lora_linear.parameters()):,}")
    print(f"  LoRA-A: {lora_linear.lora_A.shape if lora_linear.lora_A is not None else None}")
    print(f"  LoRA-B: {lora_linear.lora_B.shape if lora_linear.lora_B is not None else None}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 512)
    output = lora_linear(x)
    print(f"\n输出 shape: {output.shape}")
    
    print("\n✓ LoRA 集成测试通过")

def test_with_real_checkpoint():
    """测试 4: 使用真实 checkpoint"""
    print("\n" + "=" * 60)
    print("测试 4: 使用真实 Checkpoint 分析")
    print("=" * 60)
    
    checkpoint_path = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects/VAD/VAD_base.pth'
    state = torch.load(checkpoint_path, map_location='cpu')
    sd = state['state_dict']
    
    # 分析规划头结构
    ego_fut_decoder_keys = [k for k in sd.keys() if 'ego_fut_decoder' in k]
    print(f"\nego_fut_decoder 参数 ({len(ego_fut_decoder_keys)} 个):")
    for k in ego_fut_decoder_keys:
        print(f"  {k}: {sd[k].shape}")
    
    # 计算参数维度
    # ego_fut_decoder.0: Linear(512, 512)
    # ego_fut_decoder.2: Linear(512, 512)
    # ego_fut_decoder.4: Linear(512, 36) ← 输出层
    
    # 分析 ego_fut_mode * fut_ts * 2
    output_weight = sd['pts_bbox_head.ego_fut_decoder.4.weight']  # [36, 512]
    ego_fut_mode = 3
    fut_ts = 6
    expected_out = ego_fut_mode * fut_ts * 2  # 3 * 6 * 2 = 36
    print(f"\n分析结果:")
    print(f"  输出层维度: {output_weight.shape}")
    print(f"  期望输出: ego_fut_mode({ego_fut_mode}) * fut_ts({fut_ts}) * 2 = {expected_out}")
    
    # 计算 LoRA 参数量
    # LoRA: rank * (in_dim + out_dim) = 16 * (512 + 36) = 8768
    rank = 16
    lora_params = rank * (512 + 36)
    print(f"\nLoRA 参数量 (rank={rank}):")
    print(f"  LoRA-A: {rank} * 512 = {rank * 512:,}")
    print(f"  LoRA-B: {rank} * 36 = {rank * 36:,}")
    print(f"  总计: {lora_params:,}")
    
    print("\n✓ 真实 checkpoint 分析完成")

def main():
    print("VAD E2E 包装器测试")
    print("=" * 60)
    
    try:
        # 测试 1: 模型加载
        test_vad_model_load()
        
        # 测试 2: E2E 包装器
        test_vad_e2e_wrapper()
        
        # 测试 3: LoRA 集成
        test_lora_integration()
        
        # 测试 4: 真实 checkpoint
        test_with_real_checkpoint()
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
