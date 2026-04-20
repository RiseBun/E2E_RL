"""验证导出的 DiffusionDrive 数据文件完整性。

检查所有 .pt 文件是否包含训练所需的完整字段，
并分析数据质量和潜在问题。
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Any

def verify_pt_file(file_path: Path) -> Dict[str, Any]:
    """验证单个 .pt 文件的完整性。"""
    result = {
        'file': file_path.name,
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f'文件加载失败: {e}')
        return result
    
    # 检查必需字段
    required_fields = ['sample_idx', 'token', 'scene_token', 'planner_outputs', 'gt_plan', 'ego_fut_cmd']
    for field in required_fields:
        if field not in data:
            result['valid'] = False
            result['errors'].append(f'缺少必需字段: {field}')
    
    # 检查 planner_outputs 的子字段
    if 'planner_outputs' in data:
        planner_outputs = data['planner_outputs']
        required_planner_fields = ['trajectory', 'bev_semantic_map', 'agent_states', 'agent_labels']
        for field in required_planner_fields:
            if field not in planner_outputs:
                result['warnings'].append(f'planner_outputs 缺少字段: {field}')
        
        # 检查张量形状
        if 'trajectory' in planner_outputs:
            traj = planner_outputs['trajectory']
            if not isinstance(traj, torch.Tensor):
                result['errors'].append('trajectory 不是张量')
            elif traj.shape[-1] != 3:
                result['warnings'].append(f'trajectory 最后一个维度应为 3 (x,y,heading)，实际为 {traj.shape[-1]}')
        
        if 'bev_semantic_map' in planner_outputs:
            bev = planner_outputs['bev_semantic_map']
            if not isinstance(bev, torch.Tensor):
                result['errors'].append('bev_semantic_map 不是张量')
            elif bev.dim() != 4:
                result['warnings'].append(f'bev_semantic_map 应为 4D 张量，实际为 {bev.dim()}D')
    
    # 检查 gt_plan
    if 'gt_plan' in data:
        gt_plan = data['gt_plan']
        if not isinstance(gt_plan, torch.Tensor):
            result['errors'].append('gt_plan 不是张量')
        elif gt_plan.shape[-1] != 2:
            result['warnings'].append(f'gt_plan 最后一个维度应为 2 (x,y)，实际为 {gt_plan.shape[-1]}')
    
    # 检查 ego_fut_cmd
    if 'ego_fut_cmd' in data:
        ego_fut_cmd = data['ego_fut_cmd']
        if not isinstance(ego_fut_cmd, torch.Tensor):
            result['errors'].append('ego_fut_cmd 不是张量')
    
    # 检查 interface_grid（如果存在）
    if 'interface_grid' in data:
        interface = data['interface_grid']
        required_interface_fields = ['scene_token', 'reference_plan']
        for field in required_interface_fields:
            if field not in interface:
                result['warnings'].append(f'interface_grid 缺少字段: {field}')
        
        result['info']['has_interface'] = True
        result['info']['interface_fields'] = list(interface.keys())
    else:
        result['info']['has_interface'] = False
        result['warnings'].append('缺少 interface_grid 字段（如果使用了 --convert 参数）')
    
    # 记录基本信息
    result['info']['sample_idx'] = data.get('sample_idx', 'N/A')
    result['info']['token'] = data.get('token', 'N/A')
    result['info']['scene_token'] = data.get('scene_token', 'N/A')
    
    if 'gt_plan' in data and isinstance(data['gt_plan'], torch.Tensor):
        result['info']['gt_plan_shape'] = list(data['gt_plan'].shape)
    
    if 'planner_outputs' in data and 'trajectory' in data['planner_outputs']:
        traj = data['planner_outputs']['trajectory']
        if isinstance(traj, torch.Tensor):
            result['info']['trajectory_shape'] = list(traj.shape)
    
    return result


def main():
    dump_dir = Path('data/diffusiondrive_dumps')
    
    print('='*70)
    print('DiffusionDrive 导出数据验证报告')
    print('='*70)
    
    # 加载 manifest
    manifest_path = dump_dir / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"\n📊 Manifest 信息:")
        print(f"  总样本数: {manifest['total_samples']}")
        print(f"  跳过样本数: {manifest['skipped_samples']}")
        print(f"  成功率: {manifest['total_samples']/(manifest['total_samples']+manifest['skipped_samples'])*100:.1f}%")
        print(f"  场景过滤器: {manifest['scene_filter']}")
    else:
        print("\n⚠️  未找到 manifest.json")
        return
    
    # 验证所有 .pt 文件
    pt_files = sorted(dump_dir.glob('*.pt'))
    print(f"\n🔍 验证 {len(pt_files)} 个 .pt 文件...")
    
    all_results = []
    valid_count = 0
    error_count = 0
    
    for pt_file in pt_files:
        result = verify_pt_file(pt_file)
        all_results.append(result)
        
        if result['valid']:
            valid_count += 1
        else:
            error_count += 1
            print(f"\n❌ {pt_file.name}:")
            for error in result['errors']:
                print(f"   错误: {error}")
    
    # 打印警告信息
    warning_files = [r for r in all_results if r['warnings']]
    if warning_files:
        print(f"\n⚠️  发现 {len(warning_files)} 个文件有警告:")
        for result in warning_files[:3]:  # 只显示前3个
            print(f"   {result['file']}:")
            for warning in result['warnings']:
                print(f"     - {warning}")
        if len(warning_files) > 3:
            print(f"   ... 还有 {len(warning_files)-3} 个文件")
    
    # 打印总结
    print(f"\n{'='*70}")
    print(f"📈 验证总结:")
    print(f"  ✅ 有效文件: {valid_count}/{len(pt_files)}")
    print(f"  ❌ 错误文件: {error_count}/{len(pt_files)}")
    
    if valid_count == len(pt_files):
        print(f"\n🎉 所有文件都完整可用！可以直接用于训练。")
    else:
        print(f"\n⚠️  部分文件存在问题，请检查上述错误信息。")
    
    # 检查数据一致性
    print(f"\n🔬 数据一致性检查:")
    
    # 检查 trajectory 时间步数
    traj_lengths = []
    gt_lengths = []
    for result in all_results:
        if 'trajectory_shape' in result['info']:
            traj_lengths.append(result['info']['trajectory_shape'][1])
        if 'gt_plan_shape' in result['info']:
            gt_lengths.append(result['info']['gt_plan_shape'][0])
    
    if traj_lengths:
        print(f"  Trajectory 时间步数: {set(traj_lengths)} (应该一致)")
    if gt_lengths:
        print(f"  GT Plan 时间步数: {set(gt_lengths)} (应该一致)")
    
    # 检查 interface_grid 是否存在
    interface_count = sum(1 for r in all_results if r['info'].get('has_interface'))
    print(f"  包含 interface_grid 的文件: {interface_count}/{len(pt_files)}")
    
    print(f"\n{'='*70}")
    
    # 提供使用建议
    print(f"\n💡 使用建议:")
    if valid_count == len(pt_files) and interface_count == len(pt_files):
        print(f"  ✅ 数据完整，可以直接用于 E2E_RL 训练")
        print(f"  ✅ 已包含 interface_grid，无需额外转换")
        print(f"  📝 训练时确保使用相同的 pool_mode='grid' 配置")
    elif valid_count == len(pt_files) and interface_count == 0:
        print(f"  ⚠️  数据完整但缺少 interface_grid")
        print(f"  📝 需要运行 convert_diffusiondrive_dump.py 进行转换")
    else:
        print(f"  ⚠️  部分文件存在问题，建议修复后再使用")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
