# Projects 文件夹

用于放置需要集成的 E2E 规划模型项目。

## 已集成项目

| 项目 | 框架 | 状态 | 适配器 |
|------|------|------|--------|
| **VAD** | mmdet3d | ✅ 已集成 | `VADPlanningAdapter` |
| **SparseDrive** | mmdet3d | ✅ 已集成 | `SparseDrivePlanningAdapter` |
| **SparseDriveV2** | mmdet3d | ⏳ 待集成 | - |
| **UniAD** | mmdet3d | ⏳ 待集成 | - |
| **DiffusionDrive** | NAVSIM | ✅ 已集成 | `DiffusionDrivePlanningAdapter` |
| **DiffusionDriveV2** | NAVSIM | ⏳ 待集成 | - |

## 使用方法

### 1. 环境配置

每个模型可能需要独立的 conda 环境（如果依赖冲突）：

```bash
# VAD / SparseDrive 系列（mmdet3d）
conda create -n vad python=3.10 -y
conda activate vad
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# DiffusionDrive 系列（NAVSIM）
conda create -n diffusiondrive python=3.10 -y
conda activate diffusiondrive
# 参考 projects/DiffusionDrive/README.md
```

### 2. 数据导出

在 E2E_RL 根目录运行 dump 脚本：

```bash
cd ~/E2E_RL

# VAD
export PYTHONPATH=projects/VAD:$PYTHONPATH
python scripts/dump_vad_inference.py \
    --config projects/VAD/projects/configs/VAD/VAD_base_e2e.py \
    --checkpoint /path/to/vad_epoch_xxx.pth \
    --output_dir data/vad_dumps \
    --max_samples 10

# SparseDrive
export PYTHONPATH=projects/SparseDrive:$PYTHONPATH
python scripts/dump_sparsedrive_inference.py \
    --config projects/SparseDrive/projects/configs/sparsedrive_small_stage2.py \
    --checkpoint /path/to/sparsedrive_stage2.pth \
    --output_dir data/sparsedrive_dumps \
    --max_samples 10

# DiffusionDrive
python scripts/dump_diffusiondrive_inference.py \
    --diffusiondrive_root projects/DiffusionDrive \
    --output_dir data/diffusiondrive_dumps \
    --max_samples 10
```

### 3. 训练和推理

参考 README.md 中的完整流程。

## 目录结构

```
projects/
├── VAD/                     # VAD 模型（已集成）
├── SparseDrive/             # SparseDrive 模型（已集成）
├── SparseDriveV2/           # SparseDriveV2 模型（待集成）
├── UniAD/                   # UniAD 模型（待集成）
├── DiffusionDrive/          # DiffusionDrive 模型（已集成）
├── DiffusionDriveV2/        # DiffusionDriveV2 模型（待集成）
└── README.md                # 本文件
```

## 集成新模型

参考 `docs/SparseDrive集成指南.md` 中的详细步骤：
1. 创建 Dump 脚本
2. 创建 Adapter
3. 注册 Adapter
4. 验证数据加载
5. 训练和推理

## 推荐的新项目

以下是值得集成的最新 E2E 自动驾驶项目：

### 高优先级
- **HybridE2E**: 混合架构 E2E 驾驶（ICRA 2025）
- **TCP**: Trajectory-guided Control Prediction（ECCV 2024）
- **VADv2**: VAD 的改进版本
- **FUMP**: Fully Unified Motion Planning（2025）

### 中优先级
- **OpenEMMA**: 开源多模态 E2E 驾驶（2024）
- **LAW**: LAtent World model（ICLR 2025）
- **VLM-AD**: Vision-Language Model for AD（2025）

### 资源
- [End-to-end-Autonomous-Driving](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving): E2E 驾驶论文集合
- [GE2EAD](https://github.com/AutoLab-SAI-SJTU/GE2EAD): E2E 学习论文收集
