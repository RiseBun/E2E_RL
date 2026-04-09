# Scripts 文件夹完整分析

> **项目路径**: `/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/scripts`
> **脚本总数**: 30 个
> **更新时间**: 2025-04-09

---

## 📊 脚本分类总览

| 类别 | 数量 | 说明 |
|------|------|------|
| 📥 **数据导出** | 7 | 从不同 E2E 模型导出推理数据 |
| 🔧 **数据处理** | 3 | 数据增强、格式转换、验证 |
| 🎓 **模型训练** | 6 | 训练 UpdateEvaluator、CorrectionPolicy 等 |
| 🧪 **A/B 实验** | 5 | 对比不同防御层配置 |
| 📈 **分析评估** | 6 | 模型性能分析、可视化 |
| 🚀 **推理部署** | 2 | 在线推理、轨迹修正 |
| 🛡️ **防御验证** | 3 | 三层防御体系验证 |
| 📦 **工具脚本** | 1 | 数据集解压 |

---

## 1️⃣ 数据导出脚本（7个）

### 📌 dump_vad_inference.py
**功能**: 导出 VAD 模型推理数据  
**输入**: VAD 检查点 + nuScenes 数据集  
**输出**: `data/vad_dumps/` 目录（.pt 文件 + manifest.json）  
**用法**:
```bash
python scripts/dump_vad_inference.py \
    --config projects/VAD/projects/configs/VAD/VAD_base_e2e.py \
    --checkpoint /path/to/vad_epoch_xxx.pth \
    --output_dir data/vad_dumps \
    --data_root /path/to/nuscenes/ \
    --max_samples 100
```
**关键特性**:
- 通过 forward hook 捕获 VAD 输出
- 多种池化方式预计算 scene_token（mean/grid/ego_local）
- 保存 GT 轨迹和元数据
- 生成 manifest.json 索引

---

### 📌 dump_vadv2_inference.py
**功能**: 导出 VADv2 模型推理数据（ICLR 2026，概率规划）  
**输入**: VADv2 检查点 + nuScenes 数据集  
**输出**: `data/vadv2_dumps/`  
**用法**: 同 VAD，修改配置路径  
**差异**: 与 VAD 框架相同，直接复用代码

---

### 📌 dump_sparsedrive_inference.py
**功能**: 导出 SparseDrive 模型推理数据（ECCV 2024，稀疏表示）  
**输入**: SparseDrive 检查点 + nuScenes 数据集  
**输出**: `data/sparsedrive_dumps/`  
**用法**:
```bash
python scripts/dump_sparsedrive_inference.py \
    --config projects/SparseDrive/projects/configs/sparsedrive_small_stage2.py \
    --checkpoint /path/to/sparsedrive.pth \
    --output_dir data/sparsedrive_dumps \
    --data_root /path/to/nuscenes/
```
**关键特性**:
- Hook 位置：`model.head`（返回 4 个输出）
- 从 instance 特征提取 scene_token
- 支持 6 个 ego_fut_mode

---

### 📌 dump_sparsedrivev2_inference.py
**功能**: 导出 SparseDriveV2 模型推理数据  
**输出**: `data/sparsedrivev2_dumps/`  
**说明**: 与 SparseDrive 框架相同，直接复用

---

### 📌 dump_diffusiondrive_inference.py
**功能**: 导出 DiffusionDrive 模型推理数据（扩散模型）  
**框架**: NAVSIM（与 mmdet3d 不同）  
**输入**: DiffusionDrive 检查点 + NAVSIM 数据集  
**输出**: `data/diffusiondrive_dumps/`  
**用法**:
```bash
# 方式 1: 仅导出原始数据（两步流程）
python scripts/dump_diffusiondrive_inference.py \
    --agent_config /path/to/diffusiondrive_agent.yaml \
    --checkpoint /path/to/diffusiondrive.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/diffusiondrive_raw \
    --max_samples 100

# 方式 2: 导出 + 转换（一步完成，推荐）✨
python scripts/dump_diffusiondrive_inference.py \
    --agent_config /path/to/diffusiondrive_agent.yaml \
    --checkpoint /path/to/diffusiondrive.pth \
    --data_path /path/to/navsim_logs/val \
    --sensor_path /path/to/sensor_blobs/val \
    --output_dir data/diffusiondrive_dumps \
    --max_samples 100 \
    --convert \
    --pool_mode grid
```
**关键特性**:
- 使用 NAVSIM 框架专用导出流程
- 从扩散模型采样中获取轨迹
- **新增**: 支持 `--convert` 参数，一步完成导出+转换
- **新增**: 支持 `--pool_mode` 选择 BEV 池化方式
- **改进**: 不再需要单独运行 `convert_diffusiondrive_dump.py`

---

### 📌 dump_diffusiondrivev2_inference.py
**功能**: 导出 DiffusionDriveV2 模型推理数据  
**输出**: `data/diffusiondrivev2_dumps/`  
**说明**: 与 DiffusionDrive 框架相同，直接复用

---

### 📌 dump_uniad_inference.py
**功能**: 导出 UniAD 模型推理数据（CVPR 2023，统一架构）  
**框架**: mmdet3d  
**输出**: `data/uniad_dumps/`  
**用法**: 同 VAD，修改配置路径  
**说明**: UniAD 与 VAD 框架相同，直接复用

---

## 2️⃣ 数据处理脚本（3个）

### 📌 augment_vad_data.py
**功能**: 数据增强（对 reference_plan 和 scene_token 添加噪声）  
**输入**: 已有的 dump 数据目录  
**输出**: 扩充后的数据目录（含 manifest.json）  
**用法**:
```bash
python scripts/augment_vad_data.py \
    --input_dir data/vad_dumps \
    --output_dir data/vad_dumps_augmented \
    --samples_per_original 50 \
    --noise_scale 0.1
```
**关键特性**:
- 对 reference_plan 添加高斯噪声（模拟 VAD 输出的自然变化）
- 对 scene_token 添加高斯噪声（模拟特征提取的随机性）
- 保留 GT 不变
- 生成 manifest.json 索引
- 可从 100 个样本扩充到 5000 个

---

### 📌 convert_diffusiondrive_dump.py
**功能**: 将 DiffusionDrive 的原始 dump 转换为 PlanningInterface 格式  
**输入**: DiffusionDrive 原始输出  
**输出**: 符合 PlanningInterface 标准的数据  
**用法**:
```bash
python scripts/convert_diffusiondrive_dump.py \
    --input_dir data/diffusiondrive_raw \
    --output_dir data/diffusiondrive_dumps
```
**说明**: 
- ⚠️ **旧版脚本**，已不推荐使用
- ✅ **推荐使用**: `dump_diffusiondrive_inference.py --convert`（一步完成）
- 保留原因：兼容已有的原始 dump 数据

---

### 📌 verify_coordinates.py
**功能**: 验证坐标系修复后的数据质量  
**检查项**:
1. GT 和 reference 是否在同一坐标系
2. GT correction 是否在合理范围（0-5m，而非 20m）
3. 轨迹终点距离原点是否在 15-25m  
**用法**:
```bash
python scripts/verify_coordinates.py
```
**输出**: 坐标系验证报告

---

## 3️⃣ 模型训练脚本（6个）

### 📌 train_update_evaluator.py
**功能**: 训练 UpdateEvaluator（评分器）  
**训练流程**:
1. 收集训练数据：多样化 candidate corrections + reward labels
2. 训练 UpdateEvaluator：多头回归
3. 离线排序验证：确保排序能力过关
4. 保存模型  
**用法**:
```bash
python scripts/train_update_evaluator.py \
    --data_dir data/vad_dumps \
    --output_dir experiments/update_evaluator
```
**输出**: `experiments/update_evaluator/update_evaluator_final.pth`

---

### 📌 train_evaluator_v2.py
**功能**: 训练 UpdateEvaluator v2（改进版）  
**改进**:
1. 使用加权采样（移除 bounded_random/safety_biased）
2. 记录排序质量指标（Spearman, Kendall, Top-k）
3. 记录过滤质量指标（Retained vs Filtered）  
**用法**: 同 v1  
**说明**: 推荐使用此版本

---

### 📌 train_correction_policy.py
**功能**: 训练 CorrectionPolicy（修正策略）  
**训练阶段**:
- **Stage 1**: Behavioral Cloning 预热
- **Stage 2**: Policy Gradient + STAPO Gate  
**用法**:
```bash
python scripts/train_correction_policy.py \
    --config configs/correction_policy.yaml \
    --output_dir experiments/correction_policy
```
**输出**: `experiments/correction_policy/policy_final.pth`

---

### 📌 train_refiner_full.py
**功能**: 完整的 Refiner 训练脚本（旧版，包含 HUF 集成）  
**训练阶段**:
- **Stage 1**: 基线验证（规则 HUF）
- **Stage 2**: 奖励加权（加入 HUF 过滤）  
**用法**:
```bash
# 第一阶段
python scripts/train_refiner_full.py \
    --config configs/refiner_debug.yaml \
    --data_dir data/vad_dumps \
    --stage supervised \
    --epochs 5

# 第二阶段
python scripts/train_refiner_full.py \
    --stage reward_weighted \
    --checkpoint ./experiments/stage1_rule_huf/checkpoint_epoch_5.pth
```
**说明**: 这是旧版 Refiner，已被 CorrectionPolicy 替代

---

### 📌 train_with_learned_gate.py
**功能**: 训练 CorrectionPolicy + LearnedUpdateGate（实验 C）  
**防御策略**:
- SafetyGuard: 保留（硬底线）
- STAPOGate: 弱兜底（宽松阈值）
- LearnedUpdateGate: 主判断  
**用法**:
```bash
python scripts/train_with_learned_gate.py \
    --data_dir data/vad_dumps \
    --evaluator_checkpoint experiments/update_evaluator/update_evaluator_final.pth \
    --output_dir experiments/expC_learned_gate
```
**目标**: 验证 LearnedUpdateGate 是否比规则 STAPO 更能提升 policy

---

### 📌 expA_safety_guard_only.py
**功能**: 实验 A 基线训练（仅 SafetyGuard）  
**防御配置**:
- SafetyGuard: ✅ 启用
- STAPOGate: ❌ 禁用
- LearnedUpdateGate: ❌ 禁用  
**用途**: 基线对照，只用物理约束

---

## 4️⃣ A/B 实验脚本（5个）

### 📌 expA_relaxed.py
**功能**: 实验 A（放宽约束）- SafetyGuard Only  
**策略**:
- SafetyGuard: ✅ 启用（放宽约束，让 RL 能学到东西）
- STAPOGate: ❌ 禁用
- LearnedUpdateGate: ❌ 禁用  
**目标**: 基线对照，只用物理约束（放宽版）  
**数据**: `data/vad_dumps_full`

---

### 📌 expB_relaxed.py
**功能**: 实验 B（放宽约束）- SafetyGuard + STAPOGate  
**策略**:
- SafetyGuard: ✅ 启用（放宽约束）
- STAPOGate: ✅ 启用（规则过滤）
- LearnedUpdateGate: ❌ 禁用  
**目标**: 验证规则过滤层的效果  
**数据**: `data/vad_dumps_full`

---

### 📌 expC_relaxed.py
**功能**: 实验 C（放宽约束）- SafetyGuard + LearnedUpdateGate  
**策略**（保守接入）:
- SafetyGuard: ✅ 保留（放宽约束）
- STAPOGate: ❌ 禁用（让 LearnedUpdateGate 单独工作）
- LearnedUpdateGate: ✅ 主判断  
**目标**: 验证 LearnedUpdateGate 是否比规则 STAPO 更能提升 policy  
**数据**: `data/vad_dumps_full`

---

### 📌 expA_safety_guard_only.py
**功能**: 实验 A - SafetyGuard Only（标准约束）  
**与 expA_relaxed 的区别**: 使用标准 SafetyGuard 约束（更严格）  
**用途**: 对比放宽约束 vs 标准约束的效果

---

### 📌 expB_stapo_gate.py
**功能**: 实验 B - SafetyGuard + STAPOGate（标准约束）  
**与 expB_relaxed 的区别**: 使用标准 SafetyGuard 约束  
**用途**: 验证规则过滤层的效果

---

## 5️⃣ 分析评估脚本（6个）

### 📌 analyze_evaluator_effectiveness.py
**功能**: 可视化 UpdateEvaluator 的筛选效果  
**分析内容**:
1. 排序质量（Spearman, Kendall Tau）
2. 过滤效果（Retained vs Filtered 的 gain 分布）
3. Top-k 命中率  
**用法**:
```bash
python scripts/analyze_evaluator_effectiveness.py \
    --evaluator_checkpoint experiments/update_evaluator/update_evaluator_final.pth \
    --data_dir data/vad_dumps
```
**输出**: 可视化图表

---

### 📌 analyze_gradient_quality.py
**功能**: 分析筛选器对 Policy Gradient 质量的影响  
**分析内容**:
1. 筛选前后梯度的方向差异
2. 梯度幅值对比
3. 梯度信噪比  
**用法**:
```bash
python scripts/analyze_gradient_quality.py \
    --data_dir data/vad_dumps \
    --policy_checkpoint experiments/correction_policy/policy_final.pth
```
**输出**: 梯度质量分析报告

---

### 📌 compare_experiments.py
**功能**: 消融实验对比分析  
**对比实验**:
1. refiner_debug: Refiner + HUF (baseline)
2. refiner_with_scorer: Refiner + HUF (无 Scorer)
3. rule_huf_test: Refiner + HUF (规则调参)
4. refiner_scorer_huf: Refiner + Scorer + HUF (完整三层)  
**用法**:
```bash
python scripts/compare_experiments.py
```
**输出**: 实验对比报告

---

### 📌 evaluate_models.py
**功能**: 评估不同实验的 Refiner 模型性能  
**评估指标**:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- L2 距离
- 碰撞率  
**用法**:
```bash
python scripts/evaluate_models.py \
    --data_dir data/vad_dumps \
    --config experiments/refiner_debug/config.yaml
```
**输出**: 模型性能评估报告

---

### 📌 eval_refined.py
**功能**: 精炼轨迹评估脚本  
**功能**:
1. 加载预训练 refiner 检查点
2. 在验证集上运行 baseline vs refined 对比评估
3. 输出 ADE/FDE/L2/碰撞率等指标  
**用法**:
```bash
python scripts/eval_refined.py \
    --checkpoint experiments/refiner_debug/best.pth \
    --config configs/refiner_debug.yaml
```
**说明**: 旧版评估脚本，已被 evaluate_models.py 替代

---

### 📌 verify_evaluator_rl_effectiveness.py
**功能**: A/B 对比实验 - 验证 LearnedUpdateGate 对 RL 训练的有效性  
**分析内容**:
1. 训练曲线对比（有/无 LearnedUpdateGate）
2. Advantage 分布对比
3. Retention ratio 变化
4. 熵值变化  
**用法**:
```bash
python scripts/verify_evaluator_rl_effectiveness.py \
    --expA_log experiments/expA_relaxed/training_log.json \
    --expC_log experiments/expC_relaxed/training_log.json
```
**输出**: A/B 对比可视化

---

## 6️⃣ 推理部署脚本（2个）

### 📌 inference_with_correction.py
**功能**: 在线推理 - 使用训练好的 CorrectionPolicy 进行轨迹修正  
**功能**:
1. 加载训练好的 CorrectionPolicy 模型
2. 加载 UpdateEvaluator（用于 LearnedUpdateGate）
3. 构建三层防御体系（SafetyGuard + STAPOGate + LearnedUpdateGate）
4. 对输入的 PlanningInterface 进行在线修正  
**用法**:
```bash
# 单样本推理
python scripts/inference_with_correction.py \
    --checkpoint experiments/ab_comparison_v2/expC_learned_gate/policy_final.pth \
    --evaluator experiments/update_evaluator_v4_5k_samples/update_evaluator_final.pth \
    --data_dir data/vad_dumps \
    --scene_token "scene_001"

# 批量推理
python scripts/inference_with_correction.py \
    --checkpoint experiments/correction_policy/policy_final.pth \
    --evaluator experiments/update_evaluator/update_evaluator_final.pth \
    --data_dir data/vad_dumps \
    --batch_size 8 \
    --output_dir outputs/inference_results
```
**输出**: 修正后的轨迹

---

## 7️⃣ 防御验证脚本（3个）

### 📌 validate_defense_layers.py
**功能**: 三层防御验证脚本  
**验证模式**:
- `--mode quick`: 快速验证（只跑层 1）
- `--mode full`: 完整验证（跑全部 4 层）
- `--mode layer1/2/3`: 单层验证  
**验证层**:
1. SafetyGuard（物理约束）
2. STAPOGate（规则过滤）
3. LearnedUpdateGate（学习过滤）
4. 完整三层防御  
**用法**:
```bash
python scripts/validate_defense_layers.py --mode full
```
**输出**: 各层防御效果报告

---

### 📌 diagnose_defense.py
**功能**: 三层防御诊断实验  
**实验内容**:
1. **实验 1**: 只测 LearnedUpdateGate（关掉 STAPO）
2. **实验 2**: 统计不同 candidate 来源的 gain 分布
3. **实验 3**: 正 gain 样本被 STAPO 过滤的比例  
**用法**:
```bash
python scripts/diagnose_defense.py \
    --data_dir data/vad_dumps \
    --evaluator_checkpoint experiments/update_evaluator/update_evaluator_final.pth
```
**输出**: 防御层诊断报告

---

### 📌 verify_coordinates.py
**功能**: 验证坐标系修复后的数据质量  
（已在数据处理部分介绍）

---

## 8️⃣ 工具脚本（1个）

### 📌 extract_navsim_dataset.sh
**功能**: 解压 NAVSIM 数据集  
**输入**: `/mnt/datasets/e2e-navsim/20260302/navsim_train.tar`  
**输出**: `DiffusionDrive/data/` 目录  
**用法**:
```bash
bash scripts/extract_navsim_dataset.sh
```
**解压内容**:
- trainval_navsim_logs/
- trainval_sensor_blobs/
- nuplan-maps-v1.0/

---

## 📋 完整工作流

### 标准训练流程

```bash
# 1. 导出数据
python scripts/dump_vad_inference.py \
    --config projects/VAD/projects/configs/VAD/VAD_base_e2e.py \
    --checkpoint /path/to/vad.pth \
    --output_dir data/vad_dumps \
    --max_samples 100

# 2. 验证数据
python scripts/verify_coordinates.py

# 3. 数据增强（可选）
python scripts/augment_vad_data.py \
    --input_dir data/vad_dumps \
    --output_dir data/vad_dumps_augmented \
    --samples_per_original 50

# 4. 训练 UpdateEvaluator
python scripts/train_evaluator_v2.py \
    --data_dir data/vad_dumps_augmented \
    --output_dir experiments/update_evaluator

# 5. 训练 CorrectionPolicy
python scripts/train_correction_policy.py \
    --config configs/correction_policy.yaml \
    --output_dir experiments/correction_policy

# 6. A/B 实验对比
python scripts/expC_relaxed.py

# 7. 验证防御层
python scripts/validate_defense_layers.py --mode full

# 8. 分析效果
python scripts/analyze_evaluator_effectiveness.py \
    --evaluator_checkpoint experiments/update_evaluator/update_evaluator_final.pth

# 9. 在线推理
python scripts/inference_with_correction.py \
    --checkpoint experiments/correction_policy/policy_final.pth \
    --evaluator experiments/update_evaluator/update_evaluator_final.pth \
    --data_dir data/vad_dumps
```

---

## 🎯 脚本使用建议

### 新手入门（按顺序）

1. **verify_coordinates.py** - 验证数据质量
2. **dump_vad_inference.py** - 导出数据
3. **augment_vad_data.py** - 数据增强
4. **train_evaluator_v2.py** - 训练评分器
5. **train_correction_policy.py** - 训练修正策略
6. **inference_with_correction.py** - 推理验证

### A/B 实验（进阶）

1. **expA_relaxed.py** - 基线（仅 SafetyGuard）
2. **expB_relaxed.py** - 基线 + STAPOGate
3. **expC_relaxed.py** - 完整三层防御
4. **compare_experiments.py** - 对比分析
5. **verify_evaluator_rl_effectiveness.py** - 验证有效性

### 分析诊断（专家）

1. **analyze_evaluator_effectiveness.py** - 分析筛选效果
2. **analyze_gradient_quality.py** - 分析梯度质量
3. **diagnose_defense.py** - 诊断防御层
4. **validate_defense_layers.py** - 验证防御层

---

## 📊 脚本依赖关系

```
数据导出脚本 (7个)
    ↓
verify_coordinates.py (验证)
    ↓
augment_vad_data.py (增强)
    ↓
    ├─→ train_evaluator_v2.py → analyze_evaluator_effectiveness.py
    └─→ train_correction_policy.py
            ↓
            ├─→ expA/B/C_relaxed.py (A/B 实验)
            ├─→ inference_with_correction.py (推理)
            └─→ validate_defense_layers.py (验证)
                    ↓
                    ├─→ diagnose_defense.py (诊断)
                    ├─→ analyze_gradient_quality.py (分析)
                    └─→ compare_experiments.py (对比)
```

---

## 💡 关键说明

### 1. 新旧版本

- **新版（推荐）**: `train_evaluator_v2.py`, `train_correction_policy.py`
- **旧版（已过时）**: `train_refiner_full.py`, `eval_refined.py`

### 2. 数据目录

- `data/vad_dumps/` - VAD 导出数据
- `data/vad_dumps_full/` - VAD 全量数据（5001 个样本）
- `data/vad_dumps_augmented/` - 增强后的数据
- `data/sparsedrive_dumps/` - SparseDrive 导出数据
- `data/diffusiondrive_dumps/` - DiffusionDrive 导出数据

### 3. 实验配置

所有实验脚本使用**硬编码 CONFIG 字典**，不支持命令行参数覆盖。  
如需修改配置，请直接编辑脚本文件中的 CONFIG 字典。

---

**文档生成时间**: 2025-04-09  
**脚本总数**: 30 个  
**分类**: 8 大类
