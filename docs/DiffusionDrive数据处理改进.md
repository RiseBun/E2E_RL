# DiffusionDrive 数据处理改进

> **改进时间**: 2025-04-09
> **改进内容**: 将两步流程合并为一步

---

## 📊 改进对比

### ❌ 旧版流程（两步）

```bash
# Step 1: 导出原始数据
python scripts/dump_diffusiondrive_inference.py \
    --agent_config /path/to/agent.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/diffusiondrive_raw \
    --max_samples 100

# Step 2: 转换为标准格式
python scripts/convert_diffusiondrive_dump.py \
    --input_dir data/diffusiondrive_raw \
    --output_dir data/diffusiondrive_dumps \
    --pool_mode grid
```

**问题**:
- 需要运行两个脚本
- 中间产生冗余数据（`diffusiondrive_raw/`）
- 容易忘记转换步骤
- 增加磁盘占用

---

### ✅ 新版流程（一步）

```bash
# 一步完成导出+转换
python scripts/dump_diffusiondrive_inference.py \
    --agent_config /path/to/agent.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/diffusiondrive_dumps \
    --max_samples 100 \
    --convert \
    --pool_mode grid
```

**优势**:
- ✅ 只需运行一个脚本
- ✅ 无中间数据，节省磁盘空间
- ✅ 不会忘记转换步骤
- ✅ 更简洁的工作流

---

## 🔧 实现细节

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--convert` | bool | False | 是否直接转换为 E2E_RL 标准格式 |
| `--pool_mode` | str | 'grid' | BEV 特征池化方式 ('mean' / 'grid' / 'ego_local') |
| `--grid_size` | int | 4 | grid 池化的分块数 |

### 工作流程

```
1. 运行 DiffusionDrive 推理
   ↓
2. 获取 planner_outputs 和 GT
   ↓
3. 如果 --convert:
   ├─ 初始化 DiffusionDrivePlanningAdapter
   ├─ 调用 adapter.extract() 提取 interface
   ├─ 转换为 CPU dict
   └─ 添加到 save_dict['interface_grid']
   ↓
4. 保存 .pt 文件（包含 interface 字段）
   ↓
5. 生成 manifest.json
```

### 代码修改

**文件**: `scripts/dump_diffusiondrive_inference.py`

**修改内容**:
1. 导入 `DiffusionDrivePlanningAdapter`
2. `run_inference_and_dump()` 函数新增参数：
   - `convert: bool = False`
   - `pool_mode: str = 'grid'`
   - `grid_size: int = 4`
3. 在保存逻辑中添加转换代码
4. `main()` 函数新增 3 个命令行参数

---

## 📁 输出格式对比

### 旧版输出（未转换）

```python
{
    'sample_idx': 0,
    'token': 'scene_001',
    'scene_token': 'scene_001',
    'planner_outputs': {
        'trajectory': [T, 3],
        'bev_semantic_map': [C, H, W],
        ...
    },
    'gt_plan': [T, 2],
    'ego_fut_cmd': [M],
}
```

### 新版输出（使用 --convert）

```python
{
    'sample_idx': 0,
    'token': 'scene_001',
    'scene_token': 'scene_001',
    'planner_outputs': {
        'trajectory': [T, 3],
        'bev_semantic_map': [C, H, W],
        ...
    },
    'gt_plan': [T, 2],
    'ego_fut_cmd': [M],
    # ✨ 新增 interface 字段
    'interface_grid': {
        'scene_token': [D],
        'reference_plan': [T, 2],
        'candidate_plans': [K, T, 2],  # 可选
        'plan_confidence': [K],         # 可选
        'safety_ttc': [T],              # 可选
        'safety_collision_prob': [T],   # 可选
    }
}
```

---

## 🚀 使用建议

### 推荐用法（一步完成）

```bash
# 导出并转换（推荐）
python scripts/dump_diffusiondrive_inference.py \
    --agent_config projects/DiffusionDrive/navsim_agent/diffusiondrive_agent.yaml \
    --checkpoint /path/to/diffusiondrive.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/diffusiondrive_dumps \
    --max_samples 100 \
    --convert \
    --pool_mode grid
```

### 兼容用法（两步流程）

如果已经有原始 dump 数据，或者需要保留原始数据：

```bash
# Step 1: 仅导出（不转换）
python scripts/dump_diffusiondrive_inference.py \
    --agent_config /path/to/agent.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/diffusiondrive_raw \
    --max_samples 100
    # 注意：不加 --convert

# Step 2: 转换（如果需要）
python scripts/convert_diffusiondrive_dump.py \
    --input_dir data/diffusiondrive_raw \
    --output_dir data/diffusiondrive_dumps \
    --pool_mode grid
```

---

## 📊 性能对比

| 指标 | 旧版（两步） | 新版（一步） | 改进 |
|------|------------|------------|------|
| **脚本数量** | 2 | 1 | -50% |
| **磁盘占用** | 原始+标准 (2x) | 仅标准 (1x) | -50% |
| **操作步骤** | 2 步 | 1 步 | -50% |
| **出错概率** | 中（易忘记转换） | 低 | - |
| **运行时间** | T1 + T2 | T1 + T2 | 相同 |

---

## 🔄 迁移指南

### 如果你之前使用旧版流程

**选项 1: 继续使用已有数据**
```bash
# 已有的 diffusiondrive_raw 数据可以继续使用
python scripts/convert_diffusiondrive_dump.py \
    --input_dir data/diffusiondrive_raw \
    --output_dir data/diffusiondrive_dumps
```

**选项 2: 重新导出（推荐）**
```bash
# 删除旧数据
rm -rf data/diffusiondrive_raw
rm -rf data/diffusiondrive_dumps

# 使用新流程重新导出
python scripts/dump_diffusiondrive_inference.py \
    --agent_config /path/to/agent.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/diffusiondrive_dumps \
    --convert \
    --pool_mode grid
```

---

## 💡 技术细节

### Adapter 复用

转换逻辑复用了 `DiffusionDrivePlanningAdapter`，确保与 VAD、SparseDrive 等模型的接口一致性：

```python
from E2E_RL.planning_interface.adapters.diffusiondrive_adapter import (
    DiffusionDrivePlanningAdapter,
)

adapter = DiffusionDrivePlanningAdapter(
    scene_pool=pool_mode,
    grid_size=grid_size,
)

interface = adapter.extract(planner_outputs, ego_fut_cmd=ego_fut_cmd)
```

### 错误处理

转换失败不会中断整个流程，只会记录警告并跳过该样本的 interface 字段：

```python
try:
    interface = adapter.extract(planner_outputs, ego_fut_cmd=ego_fut_cmd)
    # 转换逻辑...
except Exception as e:
    logger.warning(f'样本 {sample_count} 转换失败: {e}')
    # 继续处理下一个样本
```

---

## 📚 相关文档

- [Scripts脚本完整分析.md](file:///mnt/cpfs/prediction/lipeinan/RL/E2E_RL/docs/Scripts脚本完整分析.md) - 所有脚本的详细说明
- [多模型集成完成报告.md](file:///mnt/cpfs/prediction/lipeinan/RL/E2E_RL/docs/多模型集成完成报告.md) - 模型集成状态

---

**改进时间**: 2025-04-09
**改进作者**: AI Assistant
**状态**: ✅ 已完成并测试
