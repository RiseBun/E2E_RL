# E2E 模型集成计划与进度

> **目标**: 集成所有主流 E2E 自动驾驶模型，建立统一的评估和训练框架

---

## 📊 集成总览

### 当前状态（2025-04-09）

```
已集成: 3/12 (25%)
  ✅ VAD
  ✅ SparseDrive  
  ✅ DiffusionDrive

已下载待集成: 4
  📦 SparseDriveV2
  📦 UniAD
  📦 DiffusionDriveV2
  📦 VADv2

待下载: 5
  ⏳ TCP (找到代码，准备下载)
  ⏳ HybridE2E
  ⏳ FUMP
  ⏳ OpenEMMA
  ⏳ LAW
```

---

## 🎯 集成优先级

### P0 - 立即集成（本周完成）

#### 1. VADv2 ⭐⭐⭐
- **重要性**: VAD 的改进版本，ICLR 2026
- **难度**: ⭐ 低（与 VAD 相同框架）
- **代码**: ✅ projects/VADv2/
- **预计时间**: 2-3 小时
- **收益**: 概率规划，性能提升

#### 2. SparseDriveV2 ⭐⭐⭐
- **重要性**: SparseDrive 升级版，2026 最新
- **难度**: ⭐ 低（与 SparseDrive 相同框架）
- **代码**: ✅ projects/SparseDriveV2/
- **预计时间**: 2-3 小时
- **收益**: 更好的稀疏表示

#### 3. DiffusionDriveV2 ⭐⭐
- **重要性**: DiffusionDrive 改进版
- **难度**: ⭐ 低（与 DiffusionDrive 相同框架）
- **代码**: ✅ projects/DiffusionDriveV2/
- **预计时间**: 2-3 小时
- **收益**: 扩散模型优化

### P1 - 近期集成（本月完成）

#### 4. TCP ⭐⭐
- **重要性**: ECCV 2024，简单但强大的 baseline
- **难度**: ⭐⭐ 中（需要分析框架）
- **代码**: 🔄 正在下载
- **预计时间**: 4-6 小时
- **收益**: 轨迹引导控制预测

#### 5. UniAD ⭐⭐
- **重要性**: CVPR 2023，经典统一架构
- **难度**: ⭐⭐ 中
- **代码**: ✅ projects/UniAD/
- **预计时间**: 4-6 小时
- **收益**: 统一感知预测规划

### P2 - 未来集成（按需）

#### 6-10. 其他模型
- HybridE2E (ICRA 2025)
- FUMP (2025.04)
- OpenEMMA (2024.12)
- LAW (ICLR 2025)
- VLM-AD (2025)

---

## 📋 快速集成方案

### 方案 A: 同类框架复用

**适用于**: VADv2, SparseDriveV2, DiffusionDriveV2

这些模型与已有模型框架相同，只需少量修改：

```bash
# VADv2 集成示例（预计 2 小时）

# 1. 复制 dump 脚本
cp scripts/dump_vad_inference.py scripts/dump_vadv2_inference.py
# 修改: 配置文件路径、hook 位置（如果有变化）

# 2. 复制 adapter
cp planning_interface/adapters/vad_adapter.py planning_interface/adapters/vadv2_adapter.py
# 修改: 类名、特定字段提取

# 3. 注册
# 编辑 data/dataloader.py
from E2E_RL.planning_interface.adapters.vadv2_adapter import VADv2PlanningAdapter
adapter_map['vadv2'] = VADv2PlanningAdapter

# 4. 测试
export PYTHONPATH=projects/VADv2:$PYTHONPATH
python scripts/dump_vadv2_inference.py --max_samples 10
```

**工作量估算**:
- Dump 脚本: 30 分钟
- Adapter: 1 小时
- 测试调试: 30 分钟
- **总计**: 2 小时

### 方案 B: 新框架集成

**适用于**: TCP, HybridE2E, FUMP

需要完整分析模型输出格式：

```bash
# TCP 集成示例（预计 4-6 小时）

# 1. 分析代码（1-2 小时）
# - 查看 TCP 的 forward 函数
# - 确定规划输出格式
# - 确认坐标系

# 2. 创建 dump 脚本（1-2 小时）
# 从零开始或参考最相似的模型

# 3. 创建 adapter（1 小时）
# 根据输出格式实现 4 个核心方法

# 4. 测试调试（1-2 小时）
```

**工作量估算**:
- 代码分析: 1-2 小时
- Dump 脚本: 1-2 小时
- Adapter: 1 小时
- 测试调试: 1-2 小时
- **总计**: 4-6 小时

---

## 🚀 执行计划

### 第一阶段: VAD 系列增强（本周）

**目标**: 完成 VADv2 集成

**任务**:
- [ ] 分析 VADv2 与 VAD 的差异
- [ ] 创建 `dump_vadv2_inference.py`
- [ ] 创建 `vadv2_adapter.py`
- [ ] 注册 adapter
- [ ] 测试 10 个样本
- [ ] 更新文档

**预计完成**: 2025-04-10

### 第二阶段: SparseDrive 系列增强（本周）

**目标**: 完成 SparseDriveV2 集成

**任务**:
- [ ] 分析 SparseDriveV2 的改进
- [ ] 创建 `dump_sparsedrivev2_inference.py`
- [ ] 创建 `sparsedrivev2_adapter.py`
- [ ] 注册 adapter
- [ ] 测试验证

**预计完成**: 2025-04-11

### 第三阶段: DiffusionDrive 系列（下周）

**目标**: 完成 DiffusionDriveV2 集成

**任务**:
- [ ] 分析 DiffusionDriveV2 变化
- [ ] 创建 dump 脚本
- [ ] 创建/更新 adapter
- [ ] 测试验证

**预计完成**: 2025-04-15

### 第四阶段: 新模型集成（本月）

**目标**: 完成 TCP 和 UniAD 集成

**任务**:
- [ ] TCP 完整集成
- [ ] UniAD 完整集成
- [ ] 对比实验

**预计完成**: 2025-04-30

---

## 📈 收益评估

### 科学价值

| 模型 | 独特性 | 对比价值 | 论文价值 |
|------|--------|---------|---------|
| VAD | 矢量化 | ⭐⭐⭐ | ⭐⭐⭐ |
| VADv2 | 概率规划 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| SparseDrive | 稀疏表示 | ⭐⭐⭐ | ⭐⭐⭐ |
| SparseDriveV2 | 改进稀疏 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| DiffusionDrive | 扩散模型 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| TCP | 轨迹引导 | ⭐⭐⭐ | ⭐⭐⭐ |
| UniAD | 统一架构 | ⭐⭐⭐ | ⭐⭐⭐ |

### 工程价值

- **统一框架**: 所有模型通过 PlanningInterface 解耦
- **公平对比**: 相同的评估指标和训练流程
- **快速实验**: 新模型只需实现 Adapter
- **论文支撑**: 多模型对比实验

---

## 🛠️ 工具和资源

### 模板文件

- **Dump 脚本模板**: `scripts/dump_vad_inference.py`
- **Adapter 模板**: `planning_interface/adapters/vad_adapter.py`
- **集成指南**: `docs/SparseDrive集成指南.md`
- **状态跟踪**: `docs/E2E模型集成状态.md`

### 参考资源

- [End-to-end-Autonomous-Driving](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving): 270+ 论文
- [GE2EAD](https://github.com/AutoLab-SAI-SJTU/GE2EAD): E2E 论文收集
- [VADv2 项目页](https://hgao-cv.github.io/VADv2/): 演示和论文

---

## 📝 注意事项

### 依赖管理

```bash
# mmdet3d 系列（VAD, SparseDrive, UniAD）
conda create -n mmdet3d python=3.10
pip install torch==2.0.0 mmcv==2.0.0

# NAVSIM 系列（DiffusionDrive）
conda create -n navsim python=3.10
# 参考 NAVSIM 文档

# 其他
# 根据每个模型的要求单独配置
```

### 数据管理

```
data/
├── vad_dumps/
├── vad_dumps_full/
├── sparsedrive_dumps/
├── sparsedrive_dumps_full/
├── vadv2_dumps/
├── sparsedrivev2_dumps/
├── diffusiondrive_dumps/
└── diffusiondrivev2_dumps/
```

### 实验管理

```
experiments/
├── vad/
│   ├── evaluator/
│   └── policy/
├── sparsedrive/
│   ├── evaluator/
│   └── policy/
├── vadv2/
├── sparsedrivev2/
└── diffusiondrive/
```

---

## 🎯 成功标准

### 技术标准

- [ ] 所有模型都能成功 dump 数据
- [ ] 所有 adapter 都能正确转换数据
- [ ] 数据加载验证通过（坐标系正确）
- [ ] 训练脚本能正常运行
- [ ] 推理脚本能正常输出

### 文档标准

- [ ] 每个模型都有集成指南
- [ ] README 包含所有模型的运行示例
- [ ] 状态文档保持更新
- [ ] 常见问题有解答

### 实验标准

- [ ] 至少 3 个模型的对比实验
- [ ] 统一的评估指标
- [ ] 可复现的训练配置

---

**创建时间**: 2025-04-09
**最后更新**: 2025-04-09
**下次更新**: 2025-04-10（完成 VADv2 集成后）
