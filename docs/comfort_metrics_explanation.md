# 舒适度指标说明

## 概述

舒适度评估用于量化规划轨迹的乘坐舒适性，主要关注加速度和加加速度(Jerk)的变化。

## 核心指标

### 加速度违规率 (Acceleration Violation Rate)

指轨迹中加速度超出舒适阈值的时间步占比。

- **阈值**：通常设定为 3.0 m/s²（参考值）
- **计算方式**：超出阈值的时间步数 / 总时间步数
- **目标**：越低越好

### 减速度违规率 (Deceleration Violation Rate)

指轨迹中减速度（负加速度）超出舒适阈值的时间步占比。

- **阈值**：通常设定为 3.0 m/s²
- **计算方式**：超出阈值的时间步数 / 总时间步数
- **目标**：越低越好

### Jerk RMS (加加速度均方根)

Jerk 是加速度的时间导数，衡量加速度变化的平滑程度。

$$Jerk_{RMS} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} ||\frac{d a}{dt}||^2}$$

- **计算**：对加速度序列求差分，再计算 RMS
- **目标**：越低说明轨迹越平滑

## PDMS 评分中的舒适度

PDMS (Planning Decision Making Score) 评分将舒适度作为评估指标之一：

| 指标 | 计算方式 |
|------|----------|
| 加速度违规 | 超出 3.0 m/s² 的时间步占比 |
| 减速度违规 | 超出 3.0 m/s² 的时间步占比 |
| Jerk RMS | 加加速度的均方根值 |

## 在 Reward 中的使用

在 `e2e_finetuning/reward.py` 中，舒适度惩罚通过以下方式计算：

```python
def _compute_comfort_penalty(self, trajectory, mask):
    # 计算速度
    velocity = diff(trajectory, dim=1) / dt
    
    # 计算加速度
    acceleration = diff(velocity, dim=1)
    
    # 计算 jerk (加速度变化率)
    jerk = diff(acceleration, dim=1) / dt
    
    # 曲率作为 lateral acceleration 代理
    curvature = norm(acceleration, dim=-1)
    
    comfort_penalty = 0.5 * speed_penalty + 0.5 * accel_penalty
```

## 配置参数

在 `RewardConfig` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `comfort_weight` | 2.0 | 舒适度惩罚权重 |
| `dt` | 0.5 | 时间步间隔 (秒) |

## 优化建议

1. **限制加速度变化率**：使用 jerk 约束平滑轨迹
2. **软约束边界**：使用指数惩罚而非硬阈值
3. **分方向处理**：纵向/横向加速度可能有不同的舒适度阈值