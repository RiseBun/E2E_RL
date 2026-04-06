# Projects 文件夹

用于放置需要集成的 E2E 规划模型项目。

## 使用方法

1. 将模型项目复制或克隆到此目录
2. 安装模型依赖
3. 运行 dump 脚本生成数据

```bash
# 示例：放置 VAD
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL/projects
cp -r /path/to/your/VAD ./VAD
cd VAD && pip install -r requirements.txt
```

## 目录结构

```
projects/
├── VAD/               # VAD 模型
└── DiffusionDrive/    # DiffusionDrive 模型（可选）
```

## 数据生成

在 E2E_RL 根目录运行 dump 脚本：

```bash
cd /mnt/cpfs/prediction/lipeinan/RL/E2E_RL
export PYTHONPATH=/path/to/your/VAD:$PYTHONPATH
python scripts/dump_vad_inference.py \
    --config /path/to/your/VAD/projects/configs/... \
    --checkpoint /path/to/model.pth \
    --output_dir data/vad_dumps
```
