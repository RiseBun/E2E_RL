"""简单诊断：检查 sensor_blobs 数据完整性。

不需要加载模型，只检查文件系统。
"""

import sys
from pathlib import Path
import json
from typing import List, Dict
from collections import Counter

def check_sensor_completeness(
    data_path: str,
    sensor_path: str,
    max_samples: int = 100,
):
    """检查传感器数据的完整性。"""
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter
    
    print("="*70)
    print("传感器数据完整性检查")
    print("="*70)
    
    # 构建 scene loader
    print(f"\n📂 数据路径: {data_path}")
    print(f"📂 传感器路径: {sensor_path}")
    
    scene_filter = SceneFilter()
    loader = SceneLoader(
        data_path=Path(data_path),
        sensor_blobs_path=Path(sensor_path),
        scene_filter=scene_filter,
        sensor_config=None,  # 不需要 sensor_config 来检查 tokens
    )
    
    all_tokens = loader.tokens[:max_samples]
    print(f"\n📊 总场景数: {len(all_tokens)}")
    
    # 检查每个场景
    missing_sensor_scenes = []
    complete_scenes = []
    
    print(f"\n🔍 检查传感器文件...")
    
    for idx, token in enumerate(all_tokens):
        if idx % 20 == 0:
            print(f"  进度: {idx}/{len(all_tokens)}")
        
        try:
            # 尝试获取场景的帧数据
            frames_dict = loader.scene_frames_dicts[token]
            
            # 检查是否有 sensor blobs
            # NAVSIM 的 sensor blobs 通常在 sensor_blobs_path/{token}/cameras/ 下
            sensor_dir = Path(sensor_path) / token / 'cameras'
            
            if not sensor_dir.exists():
                missing_sensor_scenes.append({
                    'token': token,
                    'reason': f'传感器目录不存在: {sensor_dir}',
                })
                continue
            
            # 检查是否有相机文件
            cam_files = list(sensor_dir.glob('*.jpg')) + list(sensor_dir.glob('*.png'))
            if len(cam_files) == 0:
                missing_sensor_scenes.append({
                    'token': token,
                    'reason': f'传感器目录为空: {sensor_dir}',
                })
                continue
            
            complete_scenes.append({
                'token': token,
                'sensor_files': len(cam_files),
            })
            
        except KeyError as e:
            missing_sensor_scenes.append({
                'token': token,
                'reason': f'KeyError: {e}',
            })
        except Exception as e:
            missing_sensor_scenes.append({
                'token': token,
                'reason': f'{type(e).__name__}: {e}',
            })
    
    # 打印结果
    print("\n" + "="*70)
    print("检查结果")
    print("="*70)
    print(f"✅ 完整场景: {len(complete_scenes)}")
    print(f"❌ 缺失传感器: {len(missing_sensor_scenes)}")
    print(f"成功率: {len(complete_scenes)/len(all_tokens)*100:.1f}%")
    
    if missing_sensor_scenes:
        print(f"\n📊 缺失原因统计:")
        reason_counter = Counter()
        for scene in missing_sensor_scenes:
            # 简化原因
            if '传感器目录不存在' in scene['reason']:
                reason_counter['传感器目录不存在'] += 1
            elif '传感器目录为空' in scene['reason']:
                reason_counter['传感器目录为空'] += 1
            else:
                reason_counter['其他错误'] += 1
        
        for reason, count in reason_counter.most_common():
            print(f"  {reason}: {count} ({count/len(missing_sensor_scenes)*100:.1f}%)")
        
        # 显示前 5 个示例
        print(f"\n📝 前 5 个缺失场景示例:")
        for i, scene in enumerate(missing_sensor_scenes[:5]):
            print(f"  [{i+1}] Token: {scene['token']}")
            print(f"      原因: {scene['reason']}")
    
    # 检查 sensor_blobs 目录结构
    print(f"\n📁 Sensor blobs 目录结构示例:")
    sensor_root = Path(sensor_path)
    if sensor_root.exists():
        # 列出前 5 个 token 目录
        token_dirs = [d for d in sensor_root.iterdir() if d.is_dir()][:5]
        print(f"  示例 token 目录 (共 {len(list(sensor_root.iterdir()))} 个):")
        for token_dir in token_dirs:
            cam_dir = token_dir / 'cameras'
            if cam_dir.exists():
                cam_count = len(list(cam_dir.glob('*.jpg'))) + len(list(cam_dir.glob('*.png')))
                print(f"    {token_dir.name}/cameras/ ({cam_count} 个文件)")
            else:
                print(f"    {token_dir.name}/ (无 cameras 目录)")
    
    print("\n" + "="*70)
    
    return complete_scenes, missing_sensor_scenes


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检查传感器数据完整性')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sensor_path', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()
    
    # 添加 DiffusionDrive 到 sys.path
    script_dir = Path(__file__).resolve().parent
    e2e_rl_dir = script_dir.parent
    diffusiondrive_root = e2e_rl_dir / 'projects' / 'DiffusionDrive'
    
    if str(diffusiondrive_root) not in sys.path:
        sys.path.insert(0, str(diffusiondrive_root))
    
    check_sensor_completeness(
        data_path=args.data_path,
        sensor_path=args.sensor_path,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()
