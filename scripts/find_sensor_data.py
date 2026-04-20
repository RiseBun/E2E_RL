"""查找成功导出的样本的传感器数据位置。"""

import json
from pathlib import Path
from PIL import Image

# 读取 manifest
with open('data/diffusiondrive_dumps/manifest.json', 'r') as f:
    manifest = json.load(f)

# 检查第一个样本的传感器数据
first_token = manifest['samples'][0]['token']
print(f"检查 token: {first_token}")

# 可能的传感器路径
possible_paths = [
    Path('projects/DiffusionDrive/navsim_workspace/dataset/sensor_blobs/trainval'),
    Path('/mnt/datasets/e2e-navsim/20260302/home/ma-user/work/yyx/navsim_train/sensor_blobs/trainval'),
]

for sensor_base in possible_paths:
    print(f"\n检查基础路径: {sensor_base}")
    
    # 尝试不同的目录结构
    sensor_dirs = [
        sensor_base / first_token / 'cameras',
        sensor_base / first_token,
    ]
    
    for sensor_dir in sensor_dirs:
        if sensor_dir.exists():
            print(f"  ✅ 存在: {sensor_dir}")
            # 列出文件
            files = list(sensor_dir.rglob('*.jpg')) + list(sensor_dir.rglob('*.png'))
            print(f"     找到 {len(files)} 个图像文件")
            if files:
                print(f"     示例: {files[0].name}")
        else:
            print(f"  ❌ 不存在: {sensor_dir}")

# 检查 pt 文件中的相机数据
import torch
pt_file = Path('data/diffusiondrive_dumps/000000.pt')
if pt_file.exists():
    data = torch.load(pt_file, map_location='cpu', weights_only=False)
    print(f"\n📦 PT 文件内容:")
    print(f"  Keys: {list(data.keys())}")
    if 'planner_outputs' in data:
        print(f"  planner_outputs keys: {list(data['planner_outputs'].keys())}")
