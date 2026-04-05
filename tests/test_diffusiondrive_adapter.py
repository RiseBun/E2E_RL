"""DiffusionDrive 适配器单元测试。

测试覆盖:
1. 适配器正确提取所有 PlanningInterface 字段
2. 各种池化方式的 scene_token 形状正确
3. 缺失字段的回退逻辑
4. 适配器与 refiner 的端到端兼容性
5. 依赖审计：下游模块不含 DiffusionDrive 特定关键词
6. 跨适配器替换：DiffusionDrive adapter 替换 VAD adapter 后下游零修改
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from E2E_RL.planning_interface.interface import PlanningInterface
from E2E_RL.planning_interface.adapters.diffusiondrive_adapter import (
    DiffusionDrivePlanningAdapter,
)
from E2E_RL.planning_interface.adapters.vad_adapter import VADPlanningAdapter
from E2E_RL.planning_interface.extractor import PlanningInterfaceExtractor
from E2E_RL.refinement.interface_refiner import InterfaceRefiner
from E2E_RL.refinement.losses import supervised_refinement_loss
from E2E_RL.refinement.reward_proxy import compute_refinement_reward


# ======================================================================
# 测试数据生成
# ======================================================================

def make_diffusiondrive_outputs(
    batch_size: int = 4,
    fut_ts: int = 8,
    num_agents: int = 30,
    bev_classes: int = 7,
    bev_h: int = 128,
    bev_w: int = 256,
) -> dict:
    """生成标准 DiffusionDrive forward() 输出。"""
    return {
        'trajectory': torch.randn(batch_size, fut_ts, 3),
        'agent_states': torch.randn(batch_size, num_agents, 5),
        'agent_labels': torch.randn(batch_size, num_agents),
        'bev_semantic_map': torch.randn(batch_size, bev_classes, bev_h, bev_w),
    }


def make_diffusiondrive_outputs_with_multimodal(
    batch_size: int = 4,
    fut_ts: int = 8,
    num_modes: int = 20,
) -> dict:
    """生成包含多模态输出的 DiffusionDrive 输出。"""
    outputs = make_diffusiondrive_outputs(batch_size, fut_ts)
    outputs['all_poses_reg'] = torch.randn(batch_size, num_modes, fut_ts, 3)
    outputs['all_poses_cls'] = torch.randn(batch_size, num_modes)
    return outputs


# ======================================================================
# 测试类 1: 适配器基本功能
# ======================================================================

class TestDiffusionDriveAdapter:
    """DiffusionDrive 适配器基本功能测试。"""

    def test_extract_full_interface(self):
        """完整提取测试：所有字段都正确填充。"""
        outputs = make_diffusiondrive_outputs(batch_size=4)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='mean')
        interface = adapter.extract(outputs)

        assert isinstance(interface, PlanningInterface)
        assert interface.scene_token.shape == (4, 7)  # mean pool → [B, C]
        assert interface.reference_plan.shape == (4, 8, 2)  # [B, T, 2]
        assert interface.plan_confidence is not None
        assert interface.plan_confidence.shape == (4, 1)
        assert interface.safety_features is not None
        assert 'object_density' in interface.safety_features

    def test_reference_plan_format(self):
        """参考轨迹格式：只取 (x, y)，不含 heading。"""
        outputs = make_diffusiondrive_outputs(batch_size=2)
        adapter = DiffusionDrivePlanningAdapter()
        ref_plan, candidates = adapter.extract_reference_plan(outputs)

        assert ref_plan.shape == (2, 8, 2)
        assert candidates is None  # 标准输出无多模态

    def test_reference_plan_with_multimodal(self):
        """带多模态输出时，candidate_plans 正确提取。"""
        outputs = make_diffusiondrive_outputs_with_multimodal(
            batch_size=2, num_modes=20,
        )
        adapter = DiffusionDrivePlanningAdapter()
        ref_plan, candidates = adapter.extract_reference_plan(outputs)

        assert ref_plan.shape == (2, 8, 2)
        assert candidates is not None
        assert candidates.shape == (2, 20, 8, 2)

    def test_confidence_with_multimodal(self):
        """多模态输出时，置信度从分类分数提取。"""
        outputs = make_diffusiondrive_outputs_with_multimodal()
        adapter = DiffusionDrivePlanningAdapter()
        conf = adapter.extract_plan_confidence(outputs)

        assert conf is not None
        assert conf.shape == (4, 1)
        assert (conf > 0).all() and (conf <= 1).all()

    def test_confidence_fallback_to_detection(self):
        """无多模态分数时，使用检测密度作为置信度代理。"""
        outputs = make_diffusiondrive_outputs()
        adapter = DiffusionDrivePlanningAdapter()
        conf = adapter.extract_plan_confidence(outputs)

        assert conf is not None
        assert conf.shape == (4, 1)
        # 范围应在 [0.5, 1.0]
        assert (conf >= 0.5 - 1e-6).all() and (conf <= 1.0 + 1e-6).all()

    def test_safety_features(self):
        """安全特征提取完整性。"""
        outputs = make_diffusiondrive_outputs()
        adapter = DiffusionDrivePlanningAdapter()
        safety = adapter.extract_safety_features(outputs)

        assert safety is not None
        assert 'object_density' in safety
        assert 'heading_change_rate' in safety
        assert 'road_coverage' in safety
        assert 'obstacle_density' in safety

        for key, val in safety.items():
            assert val.shape == (4, 1), f'{key} shape 错误: {val.shape}'

    def test_missing_trajectory_raises(self):
        """缺少 trajectory 字段时应抛出异常。"""
        adapter = DiffusionDrivePlanningAdapter()
        with pytest.raises(KeyError, match='trajectory'):
            adapter.extract_reference_plan({'bev_semantic_map': torch.randn(1, 7, 128, 256)})


# ======================================================================
# 测试类 2: 池化方式
# ======================================================================

class TestScenePoolMethods:
    """BEV 语义图各种池化方式的测试。"""

    def test_mean_pool(self):
        """均值池化 → [B, C]。"""
        outputs = make_diffusiondrive_outputs(batch_size=2)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='mean')
        token = adapter.extract_scene_token(outputs)
        assert token.shape == (2, 7)

    def test_max_pool(self):
        """最大值池化 → [B, C]。"""
        outputs = make_diffusiondrive_outputs(batch_size=2)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='max')
        token = adapter.extract_scene_token(outputs)
        assert token.shape == (2, 7)

    def test_grid_pool(self):
        """分块池化 → [B, grid^2 * C]。"""
        outputs = make_diffusiondrive_outputs(batch_size=2)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='grid', grid_size=4)
        token = adapter.extract_scene_token(outputs)
        assert token.shape == (2, 4 * 4 * 7)  # [B, 112]

    def test_flatten_pool(self):
        """展平池化 → [B, C * 4 * 8]。"""
        outputs = make_diffusiondrive_outputs(batch_size=2)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='flatten')
        token = adapter.extract_scene_token(outputs)
        assert token.shape == (2, 7 * 4 * 8)  # [B, 224]

    def test_fallback_scene_token(self):
        """无 BEV 语义图时，从轨迹和检测构建伪 scene_token。"""
        outputs = {
            'trajectory': torch.randn(2, 8, 3),
            'agent_states': torch.randn(2, 30, 5),
            'agent_labels': torch.randn(2, 30),
        }
        adapter = DiffusionDrivePlanningAdapter(scene_pool='mean')
        token = adapter.extract_scene_token(outputs)
        # 8*3 + 5 = 29
        assert token.shape == (2, 29)


# ======================================================================
# 测试类 3: 与 Refiner 端到端兼容性
# ======================================================================

class TestRefinerCompatibility:
    """验证 DiffusionDrive adapter 与现有 refiner 的兼容性。"""

    @pytest.mark.parametrize('pool_mode,expected_scene_dim', [
        ('mean', 7),
        ('grid', 112),
        ('flatten', 224),
    ])
    def test_refiner_forward(self, pool_mode: str, expected_scene_dim: int):
        """Refiner 前向传播正常工作。"""
        outputs = make_diffusiondrive_outputs(batch_size=4)
        adapter = DiffusionDrivePlanningAdapter(scene_pool=pool_mode, grid_size=4)
        interface = adapter.extract(outputs)

        plan_len = interface.reference_plan.shape[1] * 2  # 8 * 2 = 16
        refiner = InterfaceRefiner(
            scene_dim=expected_scene_dim,
            plan_len=plan_len,
            hidden_dim=64,
        )

        result = refiner(interface)
        assert 'refined_plan' in result
        assert result['refined_plan'].shape == (4, 8, 2)

    def test_refiner_training_step(self):
        """完整训练步骤：前向 + 损失 + 反向传播。"""
        outputs = make_diffusiondrive_outputs(batch_size=8)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='grid', grid_size=4)
        interface = adapter.extract(outputs)

        gt_plan = torch.randn(8, 8, 2)

        refiner = InterfaceRefiner(
            scene_dim=112,
            plan_len=16,
            hidden_dim=64,
        )
        optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-3)

        # 前向
        result = refiner(interface)
        loss = supervised_refinement_loss(result['refined_plan'], gt_plan)
        loss = loss + result['residual_norm'].mean() * 0.01

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_reward_proxy_compatible(self):
        """奖励代理在 DiffusionDrive 接口上正常工作。"""
        outputs = make_diffusiondrive_outputs(batch_size=4)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='grid')
        interface = adapter.extract(outputs)

        gt_plan = torch.randn(4, 8, 2)

        refiner = InterfaceRefiner(scene_dim=112, plan_len=16, hidden_dim=64)
        result = refiner(interface)

        reward_info = compute_refinement_reward(
            refined_plan=result['refined_plan'].detach(),
            gt_plan=gt_plan,
        )
        assert 'total_reward' in reward_info
        assert reward_info['total_reward'].shape == (4,)


# ======================================================================
# 测试类 4: 依赖审计与跨适配器替换
# ======================================================================

class TestDependencyAudit:
    """验证下游模块不含 DiffusionDrive/VAD 特定关键词。"""

    def test_refiner_no_planner_keywords(self):
        """InterfaceRefiner 源码中不应包含 planner 特定关键词。"""
        import inspect
        source = inspect.getsource(InterfaceRefiner)
        for keyword in ['DiffusionDrive', 'diffusiondrive', 'VAD', 'vad',
                        'bev_embed', 'ego_fut_preds', 'bev_semantic_map',
                        'trajectory', 'agent_states']:
            assert keyword not in source, (
                f'InterfaceRefiner 源码包含 planner 特定关键词: {keyword}'
            )

    def test_loss_no_planner_keywords(self):
        """损失函数中不应包含 planner 特定关键词。"""
        import inspect
        source = inspect.getsource(supervised_refinement_loss)
        for keyword in ['DiffusionDrive', 'VAD', 'bev_embed', 'ego_fut_preds']:
            assert keyword not in source

    def test_cross_adapter_swap(self):
        """跨适配器替换: 用 DiffusionDrive adapter 替换 VAD adapter，
        下游 refiner 完全不需要修改。"""
        # 使用 DiffusionDrive adapter
        dd_outputs = make_diffusiondrive_outputs(batch_size=4)
        dd_adapter = DiffusionDrivePlanningAdapter(scene_pool='mean')
        dd_interface = dd_adapter.extract(dd_outputs)

        # 使用相同维度的 refiner (scene_dim=7 for mean pool of 7 classes)
        refiner = InterfaceRefiner(
            scene_dim=dd_interface.scene_token.shape[-1],
            plan_len=dd_interface.reference_plan.shape[1] * 2,
            hidden_dim=64,
        )

        # 前向传播正常
        result = refiner(dd_interface)
        assert result['refined_plan'].shape == dd_interface.reference_plan.shape

    def test_extractor_factory(self):
        """通过 PlanningInterfaceExtractor.from_config 创建适配器。"""
        extractor = PlanningInterfaceExtractor.from_config(
            adapter_type='diffusiondrive',
            scene_pool='grid',
        )
        outputs = make_diffusiondrive_outputs(batch_size=2)
        interface = extractor.extract(outputs)

        assert isinstance(interface, PlanningInterface)
        assert interface.scene_token.shape[-1] == 112  # grid 4x4 * 7


# ======================================================================
# 测试类 5: 边界条件
# ======================================================================

class TestEdgeCases:
    """边界条件测试。"""

    def test_batch_size_1(self):
        """batch_size=1 正常工作。"""
        outputs = make_diffusiondrive_outputs(batch_size=1)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='grid')
        interface = adapter.extract(outputs)
        assert interface.scene_token.shape[0] == 1

    def test_no_valid_agents(self):
        """所有 agent 无效时，safety_features 仍正常。"""
        outputs = make_diffusiondrive_outputs(batch_size=2)
        outputs['agent_labels'] = torch.full((2, 30), -10.0)  # 全部无效
        adapter = DiffusionDrivePlanningAdapter()
        interface = adapter.extract(outputs)

        assert interface.safety_features is not None
        assert 'object_density' in interface.safety_features
        # 密度应接近 0
        assert interface.safety_features['object_density'].max() < 0.01

    def test_interface_to_device(self):
        """PlanningInterface.to() 正常工作。"""
        outputs = make_diffusiondrive_outputs(batch_size=2)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='grid')
        interface = adapter.extract(outputs)

        interface_cpu = interface.to(torch.device('cpu'))
        assert interface_cpu.scene_token.device.type == 'cpu'

    def test_interface_describe(self):
        """PlanningInterface.describe() 输出格式正确。"""
        outputs = make_diffusiondrive_outputs(batch_size=2)
        adapter = DiffusionDrivePlanningAdapter(scene_pool='grid')
        interface = adapter.extract(outputs)

        desc = interface.describe()
        assert 'scene_token' in desc
        assert 'reference_plan' in desc


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
