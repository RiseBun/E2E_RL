# SparseDrive 集成指南

本文档说明如何将 SparseDrive 模型集成到 E2E_RL 框架中。

---

## 概述

SparseDrive 与 VAD 类似，都是基于 mmdet3d 的端到端自动驾驶模型，因此集成流程与 VAD 高度相似。

**关键信息**：
- **框架**: mmdet3d + mmcv（与 VAD 相同）
- **轨迹输出**: `ego_fut_mode=6`, `ego_fut_ts=6`（6个模式，6个时间步）
- **坐标系**: ego-centric 位移增量（需要 cumsum 转绝对坐标）
- **场景特征**: 可从 BEV 特征或 instance 特征中提取

---

## 集成步骤

### Step 1: 环境配置

SparseDrive 与 VAD 共享相同的依赖（mmdet3d, mmcv），可以共用同一个 conda 环境。

```bash
# 如果已有 VAD 环境，可以直接使用
conda activate vad

# 否则创建新环境
conda create -n sparsedrive python=3.10 -y
conda activate sparsedrive

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 mmcv
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 安装 SparseDrive 依赖
cd ~/E2E_RL/projects/SparseDrive
pip install -r requirement.txt
```

### Step 2: 创建 Dump 脚本

创建 `E2E_RL/scripts/dump_sparsedrive_inference.py`，参考 `dump_vad_inference.py`。

**关键点**：
- SparseDrive 使用 `tools/test.py` 进行推理
- 规划输出在 `planning_result` 中
- 需要注册 forward hook 捕获 `motion_plan_head` 的输出

**输出格式**（每个 .pt 文件）：
```python
{
    'sample_idx': int,
    'scene_token': str,
    # SparseDrive 原始输出
    'ego_fut_preds': [M, T, 2],        # 位移增量（M=6, T=6）
    'planning_cls_scores': [M],        # 规划模式分数
    # GT 数据
    'ego_fut_trajs': [T, 2],           # GT 轨迹（位移增量）
    'ego_fut_masks': [T],              # 有效 mask
    'ego_fut_cmd': [M],                # 命令（one-hot）
    # 元信息
    'timestamp': int,
    'sample_token': str,
}
```

### Step 3: 创建 SparseDrive Adapter

创建 `E2E_RL/planning_interface/adapters/sparsedrive_adapter.py`。

**与 VAD Adapter 的主要区别**：
1. **场景特征提取**：SparseDrive 使用 sparse instance 而非密集 BEV
   - 可从 `instance_feat` 或 `det_output` 中提取
   - 或使用检测框的 top-k 特征池化

2. **规划输出格式**：与 VAD 类似
   - `ego_fut_preds`: [B, M, T, 2] 或 [M, T, 2]（位移增量）
   - 需要根据 `ego_fut_cmd` 选择模式
   - cumsum 转为绝对坐标

3. **置信度计算**：使用规划分类分数
   - `planning_cls_scores` → sigmoid → 取最大值

**Adapter 实现模板**：
```python
class SparseDrivePlanningAdapter(BasePlanningAdapter):
    def extract_scene_token(self, planner_outputs):
        # 从 instance 特征或检测输出中提取
        # 方案 1: 使用 top-k 检测框特征池化
        # 方案 2: 使用全局均值池化（如果有 bev_embed）
        pass
    
    def extract_reference_plan(self, planner_outputs, ego_fut_cmd=None):
        # 与 VAD 类似：
        # 1. 根据 ego_fut_cmd 选择模式
        # 2. cumsum 转绝对坐标
        pass
    
    def extract_plan_confidence(self, planner_outputs, ego_fut_cmd=None):
        # 使用 planning_cls_scores 计算置信度
        pass
    
    def extract_safety_features(self, planner_outputs):
        # 从检测和地图输出中提取安全特征
        pass
```

### Step 4: 注册 Adapter

在 `E2E_RL/data/dataloader.py` 中注册：

```python
from E2E_RL.planning_interface.adapters.sparsedrive_adapter import (
    SparseDrivePlanningAdapter,
)

adapter_map = {
    'vad': VADPlanningAdapter,
    'diffusiondrive': DiffusionDrivePlanningAdapter,
    'sparsedrive': SparseDrivePlanningAdapter,  # 新增
}
```

### Step 5: 运行 Dump

```bash
cd ~/E2E_RL

# 设置 PYTHONPATH
export PYTHONPATH=projects/SparseDrive:$PYTHONPATH

# 运行 dump（先用 10 个样本测试）
python scripts/dump_sparsedrive_inference.py \
    --config projects/SparseDrive/projects/configs/sparsedrive_small_stage2.py \
    --checkpoint /path/to/sparsedrive_stage2.pth \
    --output_dir data/sparsedrive_dumps \
    --data_root /path/to/nuscenes/data/ \
    --max_samples 10
```

### Step 6: 验证数据加载

```bash
cd ~/E2E_RL

python -c "
from data.dataloader import build_planner_dataloader
loader = build_planner_dataloader('data/sparsedrive_dumps', adapter_type='sparsedrive', batch_size=8)
for batch in loader:
    gt = batch['gt_plan']
    ref = batch['interface'].reference_plan
    print(f'✅ GT终点距原点: {gt[:, -1, :].norm(dim=-1)[:3]}')
    print(f'✅ Ref终点距原点: {ref[:, -1, :].norm(dim=-1)[:3]}')
    print(f'✅ scene_token shape: {batch[\"interface\"].scene_token.shape}')
    break
"
```

**期望输出**（ego-centric 坐标系）：
```
✅ GT终点距原点: tensor([15.2, 18.5, 20.1])
✅ Ref终点距原点: tensor([15.0, 18.3, 19.8])
✅ scene_token shape: torch.Size([8, 256])
```

### Step 7: 数据增强（可选）

```bash
cd ~/E2E_RL

python scripts/augment_vad_data.py \
    --input_dir data/sparsedrive_dumps \
    --output_dir data/sparsedrive_dumps_full \
    --samples_per_original 50 \
    --noise_scale 0.1 \
    --max_samples 5000
```

### Step 8: 训练 UpdateEvaluator

编辑 `scripts/train_evaluator_v2.py` 中的 CONFIG：

```python
CONFIG = {
    'data': {
        'data_dir': 'data/sparsedrive_dumps_full',  # 修改这里
        'batch_size': 16,
        'val_split': 0.2,
    },
    'model': {
        'scene_dim': 256,      # 确认与 SparseDrive 的 scene_token 维度一致
        'plan_len': 6,         # SparseDrive ego_fut_ts=6
        'hidden_dim': 256,
        'dropout': 0.1,
    },
    'output_dir': 'experiments/sparsedrive_evaluator',  # 修改这里
    # ... 其他配置保持不变
}
```

运行训练：
```bash
cd ~/E2E_RL
python scripts/train_evaluator_v2.py
```

### Step 9: 训练 CorrectionPolicy

编辑 `scripts/expC_relaxed.py` 中的 CONFIG：

```python
CONFIG = {
    'data': {
        'data_dir': 'data/sparsedrive_dumps_full',  # 修改这里
        'batch_size': 16,
    },
    'model': {
        'scene_dim': 256,
        'plan_len': 6,
        'hidden_dim': 256,
    },
    'evaluator_ckpt': 'experiments/sparsedrive_evaluator/evaluator_final.pth',
    'output_dir': 'experiments/sparsedrive_policy',  # 修改这里
    # ... 其他配置保持不变
}
```

运行训练：
```bash
cd ~/E2E_RL
python scripts/expC_relaxed.py
```

### Step 10: 推理验证

```bash
cd ~/E2E_RL

python scripts/inference_with_correction.py \
    --checkpoint experiments/sparsedrive_policy/policy_final.pth \
    --evaluator experiments/sparsedrive_evaluator/evaluator_final.pth \
    --data_dir data/sparsedrive_dumps \
    --max_samples 100
```

---

## 关键差异对比

| 特性 | VAD | SparseDrive |
|------|-----|-------------|
| **框架** | mmdet3d | mmdet3d（相同） |
| **规划模式数** | 3 | 6 |
| **时间步长** | 6 | 6 |
| **场景特征** | BEV 特征池化 | Instance 特征池化 |
| **轨迹格式** | 位移增量 | 位移增量（相同） |
| **坐标系** | ego-centric | ego-centric（相同） |
| **Adapter 复杂度** | 中等 | 中等（类似） |

---

## 注意事项

### 1. Scene Token 提取

SparseDrive 不使用密集 BEV 特征，需要采用其他方式提取场景特征：

**方案 A**（推荐）：Instance 特征池化
```python
# 从检测头输出中提取 top-k instance 特征
instance_feats = planner_outputs['instance_feat']  # [B, N, D]
# 取 top-k 或均值池化
scene_token = instance_feats.mean(dim=1)  # [B, D]
```

**方案 B**：检测框特征拼接
```python
# 使用检测框的分类分数加权池化
cls_scores = planner_outputs['det_cls_scores']  # [B, N, C]
weights = cls_scores.max(dim=-1).values.softmax(dim=-1)  # [B, N]
scene_token = (weights.unsqueeze(-1) * instance_feats).sum(dim=1)
```

### 2. 规划模式选择

SparseDrive 有 6 个规划模式（VAD 只有 3 个），但选择逻辑相同：
```python
# 根据 ego_fut_cmd 选择对应模式
cmd_idx = ego_fut_cmd.argmax(dim=-1)  # [B]
reference_plan = ego_fut_preds[batch_idx, cmd_idx]  # [B, T, 2]
```

### 3. 配置文件参数

在训练脚本中确认以下参数：
- `scene_dim`: 必须与 SparseDrive 的 scene_token 维度一致（通常 256）
- `plan_len`: 必须与 `ego_fut_ts` 一致（SparseDrive 为 6）

---

## 故障排查

### Q: dump 时找不到模块

**A**: 检查 PYTHONPATH 是否正确设置：
```bash
export PYTHONPATH=projects/SparseDrive:$PYTHONPATH
```

### Q: scene_token 维度不匹配

**A**: 检查 Adapter 中 extract_scene_token 的输出维度，必须与训练脚本中的 `scene_dim` 一致。

### Q: 轨迹坐标系不正确

**A**: 确认：
1. dump 脚本保存的是位移增量（不是绝对坐标）
2. Adapter 中执行了 cumsum 转换
3. 验证时 GT 和 Ref 终点距离应该在 15-25m 范围内

---

## 下一步

完成 SparseDrive 集成后，你可以：

1. **对比不同模型的性能**：
   - VAD vs SparseDrive vs DiffusionDrive
   - 比较 retained_advantage、retention ratio 等指标

2. **尝试模型融合**：
   - 使用多个模型的 PlanningInterface 进行集成
   - 探索最佳修正策略

3. **优化 Adapter**：
   - 尝试不同的 scene_token 提取方式
   - 优化置信度计算策略

---

## 参考文件

- VAD Dump 脚本: `scripts/dump_vad_inference.py`
- VAD Adapter: `planning_interface/adapters/vad_adapter.py`
- SparseDrive 配置: `projects/SparseDrive/projects/configs/sparsedrive_small_stage2.py`
- SparseDrive 测试: `projects/SparseDrive/tools/test.py`
