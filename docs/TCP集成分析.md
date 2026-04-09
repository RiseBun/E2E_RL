# TCP 集成分析

> **TCP**: Trajectory-guided Control Prediction for End-to-end Autonomous Driving (ECCV 2024)

---

## 框架分析

TCP 使用的框架与现有项目**完全不同**：

| 特性 | TCP | 现有项目（VAD/SparseDrive） |
|------|-----|---------------------------|
| **模拟器** | CARLA | nuScenes |
| **框架** | 自训练框架 | mmdet3d |
| **数据格式** | CARLA leaderboard | nuScenes format |
| **传感器** | RGB + 深度 + 语义分割 | 多摄像头 |
| **输出** | 控制信号（油门、刹车、转向） | 轨迹坐标 |

---

## 集成挑战

### 1. 数据格式不兼容

TCP 在 CARLA 中运行，输出是控制信号而非轨迹坐标：
```python
# TCP 输出
control = {
    'steer': float,    # 转向 [-1, 1]
    'throttle': float, # 油门 [0, 1]
    'brake': float,    # 刹车 [0, 1]
}

# PlanningInterface 期望
interface = {
    'reference_plan': [T, 2],  # 轨迹坐标 (x, y)
    'scene_token': [D],
}
```

### 2. 需要 CARLA 环境

运行 TCP 需要：
- CARLA 模拟器
- CARLA leaderboard
- Scenario Runner
- Roach 训练框架

### 3. 无法直接复用 Dump 脚本

现有的 dump 脚本依赖 mmdet3d 的数据加载逻辑，无法用于 TCP。

---

## 集成方案

### 方案 A: 完整集成（推荐，2-3天）

1. **设置 CARLA 环境**
2. **创建 TCP 数据收集脚本**
   - 在 CARLA 中运行 TCP 推理
   - 记录轨迹坐标（而非控制信号）
   - 转换为 PlanningInterface 格式

3. **创建 TCP Adapter**
   - 从记录的轨迹中提取 reference_plan
   - 使用图像特征作为 scene_token

4. **验证和测试**

### 方案 B: 延迟集成（当前推荐）

由于 TCP 与现有框架差异较大，建议：

1. **先完成 mmdet3d 系列模型的集成**
   - VAD ✅
   - VADv2 ✅
   - SparseDrive ✅
   - SparseDriveV2 ✅
   - UniAD ⏳

2. **收集足够的实验数据**

3. **后续再集成 TCP**
   - 作为 CARLA 环境的对比
   - 需要额外的实验设置

---

## 当前状态

⏳ **延迟集成**

**原因**:
1. 框架不兼容（CARLA vs nuScenes）
2. 输出格式不同（控制信号 vs 轨迹坐标）
3. 需要额外的环境配置
4. 集成工作量大（2-3天）

**建议**:
先完成 UniAD 集成（mmdet3d 框架，预计4-6小时），然后再考虑 TCP。

---

## 参考资源

- **TCP 论文**: https://arxiv.org/abs/2206.08129
- **TCP 代码**: projects/TCP/
- **CARLA**: https://carla.org/
- **CARLA Leaderboard**: https://leaderboard.carla.org/

---

**创建时间**: 2025-04-09
**状态**: 延迟集成
