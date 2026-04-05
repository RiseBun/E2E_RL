#!/usr/bin/env python
"""扩充 VAD dump 数据

通过添加噪声来扩充现有的 100 个样本，生成数千个样本用于训练。

方法：
1. 对 reference_plan 添加小扰动（模拟 VAD 输出的自然变化）
2. 对 scene_token 添加小扰动（模拟特征提取的随机性）
3. 保留 GT 和其他元信息不变
"""

import torch
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def augment_sample(sample: dict, noise_scale: float = 0.1) -> dict:
    """对单个样本添加噪声扩充"""
    aug_sample = sample.copy()

    # 对 reference_plan 添加噪声（模拟不同帧的轨迹变化）
    if 'interface_mean' in sample:
        ref_plan = aug_sample['interface_mean'].get('reference_plan')
        if ref_plan is not None and isinstance(ref_plan, torch.Tensor):
            noise = torch.randn_like(ref_plan) * noise_scale
            aug_sample['interface_mean']['reference_plan'] = ref_plan + noise

    if 'interface_grid' in sample:
        ref_plan = aug_sample['interface_grid'].get('reference_plan')
        if ref_plan is not None and isinstance(ref_plan, torch.Tensor):
            noise = torch.randn_like(ref_plan) * noise_scale
            aug_sample['interface_grid']['reference_plan'] = ref_plan + noise

    if 'interface_ego_local' in sample:
        ref_plan = aug_sample['interface_ego_local'].get('reference_plan')
        if ref_plan is not None and isinstance(ref_plan, torch.Tensor):
            noise = torch.randn_like(ref_plan) * noise_scale
            aug_sample['interface_ego_local']['reference_plan'] = ref_plan + noise

    return aug_sample


def augment_dataset(
    input_dir: str,
    output_dir: str,
    samples_per_original: int = 50,
    noise_scale: float = 0.1,
):
    """扩充数据集

    Args:
        input_dir: 原始数据目录
        output_dir: 输出目录
        samples_per_original: 每个原始样本扩充多少个
        noise_scale: 噪声尺度
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有原始 .pt 文件
    pt_files = sorted(list(input_path.glob("*.pt")))
    print(f"找到 {len(pt_files)} 个原始样本")

    total_generated = 0
    original_count = 0

    for pt_file in tqdm(pt_files, desc="扩充数据"):
        # 加载原始样本
        sample = torch.load(pt_file, map_location='cpu')

        # 保存原始样本（可选）
        if original_count == 0:
            # 只保存前几个原始样本作为参考
            output_file = output_path / f"{total_generated:06d}.pt"
            torch.save(sample, output_file)
            total_generated += 1
            original_count += 1

        # 生成扩充样本
        for i in range(samples_per_original):
            aug_sample = augment_sample(sample, noise_scale)

            # 修改 sample_idx 以区分
            if 'sample_idx' in aug_sample:
                aug_sample['sample_idx'] = total_generated

            output_file = output_path / f"{total_generated:06d}.pt"
            torch.save(aug_sample, output_file)
            total_generated += 1

            if total_generated >= 5000:
                break

        if total_generated >= 5000:
            break

    print(f"\n扩充完成！共生成 {total_generated} 个样本")
    print(f"输出目录: {output_path}")

    return total_generated


def main():
    parser = argparse.ArgumentParser(description='扩充 VAD dump 数据')
    parser.add_argument('--input_dir', type=str,
                        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps',
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps_full',
                        help='输出目录')
    parser.add_argument('--samples_per_original', type=int, default=50,
                        help='每个原始样本扩充多少个')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                        help='噪声尺度')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='最大样本数')
    args = parser.parse_args()

    print("=" * 60)
    print("VAD Dump 数据扩充")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"每个原始样本扩充: {args.samples_per_original} 个")
    print(f"噪声尺度: {args.noise_scale}")
    print(f"目标样本数: {args.max_samples}")
    print("=" * 60)

    # 扩充数据
    total = augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        samples_per_original=args.samples_per_original,
        noise_scale=args.noise_scale,
    )

    print(f"\n最终样本数: {total}")


if __name__ == '__main__':
    main()
