"""验证坐标系修复后的数据质量。

检查：
1. GT 和 reference 是否在同一个坐标系
2. GT correction 是否在合理范围内（应该是 0-5m，而非 20m）
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from E2E_RL.data.dataloader import VADDumpDataset


def verify_data_coordinate_system():
    """验证 GT 和 reference 是否在同一个坐标系。"""
    data_dir = Path(__file__).parent.parent / 'data' / 'vad_dumps'

    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return

    dataset = VADDumpDataset(str(data_dir))

    print(f"数据集大小: {len(dataset)}")
    print("=" * 60)

    corrections = []
    ref_fdes = []
    gt_fdes = []

    for i in range(min(10, len(dataset))):
        sample = dataset[i]

        gt_plan = sample['gt_plan']  # [T, 2]
        ref_plan = sample['interface'].reference_plan  # [T, 2]

        # 计算终点距离原点的距离
        gt_end_dist = torch.norm(gt_plan[-1]).item()
        ref_end_dist = torch.norm(ref_plan[-1]).item()

        # 计算 GT correction（FDE）
        correction = torch.norm(gt_plan[-1] - ref_plan[-1]).item()

        # 计算各时刻的平均差距
        mean_correction = torch.norm(gt_plan - ref_plan, dim=1).mean().item()

        corrections.append(correction)
        ref_fdes.append(ref_end_dist)
        gt_fdes.append(gt_end_dist)

        print(f"\n样本 {i}:")
        print(f"  GT 终点: ({gt_plan[-1, 0]:.3f}, {gt_plan[-1, 1]:.3f}), 距原点: {gt_end_dist:.3f}m")
        print(f"  Ref 终点: ({ref_plan[-1, 0]:.3f}, {ref_plan[-1, 1]:.3f}), 距原点: {ref_end_dist:.3f}m")
        print(f"  GT correction (FDE): {correction:.3f}m")
        print(f"  各时刻平均差距: {mean_correction:.3f}m")

    print("\n" + "=" * 60)
    print("统计结果:")
    print(f"  GT correction (FDE) 均值: {sum(corrections)/len(corrections):.3f}m")
    print(f"  GT correction (FDE) 范围: [{min(corrections):.3f}, {max(corrections):.3f}]")
    print(f"  GT 距原点均值: {sum(gt_fdes)/len(gt_fdes):.3f}m")
    print(f"  Ref 距原点均值: {sum(ref_fdes)/len(ref_fdes):.3f}m")

    # 判断是否合理
    avg_correction = sum(corrections) / len(corrections)
    if avg_correction < 10.0:
        print(f"\n✓ 坐标系修复成功！平均 correction ({avg_correction:.3f}m) 在合理范围内")
    elif avg_correction < 20.0:
        print(f"\n⚠ 坐标系可能有部分问题，平均 correction ({avg_correction:.3f}m) 偏高")
    else:
        print(f"\n✗ 坐标系问题仍存在，平均 correction ({avg_correction:.3f}m) 仍然很大")


if __name__ == '__main__':
    verify_data_coordinate_system()
