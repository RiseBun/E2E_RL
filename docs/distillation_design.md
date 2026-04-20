# Phase 3: 知识蒸馏设计

## 核心思想

Phase 2 训练后的模型带有 Value Head、Reference Model 等训练期辅助模块，Phase 3 将知识蒸馏回纯 E2E 模型，得到可直接部署的版本。

## 蒸馏目标

- 去除训练期辅助模块 (Value Head、Reference Model、Reward 模块)
- 保留有益修正能力到 student 模型
- 得到单模型部署版本，无外挂依赖

## 蒸馏架构

```
┌───────────────────┐                        ┌───────────────────┐
│ Teacher (Phase2)  │    ┌───────────┐      │ Student (Base)    │
│ - Value Head      │ →  │ 蒸馏损失   │ →    │ - 无外挂模块      │
│ - Ref Model       │    │ - Traj L2 │      │ - 纯E2E模型       │
│ - Reward模块      │    │ - Score KL│      │                   │
└───────────────────┘    │ - Feat MSE│      └───────────────────┘
                         └───────────┘
```

## 蒸馏损失

### 1. Trajectory L2 Loss

```python
traj_loss = MSE(traj_student, traj_teacher)
```

直接模仿教师轨迹，保留修正能力。

### 2. Score KL Loss

```python
# 温度 softmax
student_prob = softmax(student_score / T)
teacher_prob = softmax(teacher_score / T)
score_loss = KL(student_prob || teacher_prob)
```

模仿教师输出分数分布，包含不确定性信息。

### 3. Feature MSE Loss

```python
feat_loss = MSE(feat_student, feat_teacher)
```

模仿教师中间层特征，传递场景理解能力。

### 总损失

```python
loss_total = (
    λ_trajectory * traj_loss +
    λ_score * score_loss +
    λ_feature * feat_loss
)
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_trajectory` | 1.0 | 轨迹损失权重 |
| `lambda_score` | 0.5 | 分数损失权重 |
| `lambda_feature` | 0.3 | 特征损失权重 |
| `score_temperature` | 2.0 | KL 散度温度参数 |
| `student_freeze_backbone` | True | 是否冻结 backbone |
| `merge_lora` | True | 是否合并 LoRA 权重 |

## 训练流程

```python
for epoch in range(epochs):
    for batch in dataloader:
        # 1. 教师前向 (freeze)
        with torch.no_grad():
            teacher_outputs = teacher_model(batch)
        
        # 2. 学生前向
        student_outputs = student_model(batch)
        
        # 3. 计算蒸馏损失
        loss, metrics = distill_loss(
            student_outputs, teacher_outputs, gt_trajectory
        )
        
        # 4. 反向更新学生
        loss.backward()
        optimizer.step()
    
    # 保存检查点
    if (epoch + 1) % 10 == 0:
        save_checkpoint(epoch, loss)
    
    # 合并 LoRA (可选)
    if merge_lora and (epoch + 1) == epochs:
        student_model.merge_lora_weights()
```

## 部署版本生成

蒸馏完成后，产出可部署的纯模型：

1. **合并 LoRA 权重**：将 LoRA 权重合并回 base layer
2. **移除辅助模块**：删除 Value Head、Reference Model
3. **导出模型**：保存为部署格式

## 与 Phase 1/2 的关系

| 阶段 | 产物 | 用途 |
|------|------|------|
| Phase 1 | 有益修正策略、训练样本 | 发现改进机会 |
| Phase 2 | 带辅助模块的增强模型 | 闭环训练内化能力 |
| Phase 3 | 纯 E2E 部署模型 | 最终部署 |

## 脚本使用

```bash
python scripts/distill_e2e_rl.py \
    --teacher_checkpoint experiments/phase2_finetuned/policy_final.pth \
    --student_checkpoint /path/to/base_model.pth \
    --data_dir data/vad_dumps \
    --output_dir experiments/distilled \
    --epochs 50 \
    --batch_size 16 \
    --lambda_trajectory 1.0 \
    --lambda_score 0.5 \
    --lambda_feature 0.3
```

## 关键实现

见 `scripts/distill_e2e_rl.py`，包含：
- `DistillationLoss`: 三重蒸馏损失
- `DistillationTrainer`: 蒸馏训练器
- `DistillationDataset`: 数据集包装