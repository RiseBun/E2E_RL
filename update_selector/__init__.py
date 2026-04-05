"""
Update Selector 模块 - 服务于 RL 训练的筛选器

核心组件：
- SafetyGuard: 硬性物理约束检查
- STAPOGate: 规则版 STAPO 门控（基线）
- UpdateEvaluator: Learned Update Selector（多头回归）
- CandidateCorrector: 多样化候选修正生成器
- OfflineEvaluator: 离线排序验证器
- DefenseLayerValidator: 三层防御效果验证器

从规则筛选升级到 Learned Selector：
- 规则 STAPO: positive advantage + low prob + low entropy → 静音
- Learned Selector: 预测 pred_gain 和 pred_risk，决定是否值得强化

三层防御级联：
1. SafetyGuard: 硬物理底线
2. STAPOGate: 规则兜底
3. LearnedUpdateGate: 高级学习判断
"""

from .safety_guard import SafetyGuard, SafetyGuardConfig
from .stapo_gate import STAPOGate, STAPOGateConfig, AdvantageThresholdGate
from .update_evaluator import (
    UpdateEvaluator,
    UpdateEvaluatorConfig,
    LearnedUpdateGate,
)
from .candidate_generator import (
    CandidateCorrector,
    CandidateStats,
    compute_structured_stats,
    DEFAULT_SAMPLE_WEIGHTS,
)
from .data_collector import (
    UpdateEvaluatorDataset,
    UpdateEvaluatorDataCollector,
    EvaluatorDataSample,
)
from .evaluator_trainer import (
    UpdateEvaluatorTrainer,
    EvaluatorTrainingConfig,
)
from .offline_evaluator import OfflineEvaluator
from .defense_validator import (
    DefenseLayerValidator,
    ValidationConfig,
    create_validation_pipeline,
    run_quick_validation,
)

__all__ = [
    # Safety Guard
    'SafetyGuard',
    'SafetyGuardConfig',
    # 规则 STAPO Gate
    'STAPOGate',
    'STAPOGateConfig',
    'AdvantageThresholdGate',
    # Learned Update Selector
    'UpdateEvaluator',
    'UpdateEvaluatorConfig',
    'LearnedUpdateGate',
    # 候选生成
    'CandidateCorrector',
    'compute_structured_stats',
    # 数据收集
    'UpdateEvaluatorDataset',
    'UpdateEvaluatorDataCollector',
    'EvaluatorDataSample',
    # 训练
    'UpdateEvaluatorTrainer',
    'EvaluatorTrainingConfig',
    # 离线评估
    'OfflineEvaluator',
    # 防御验证
    'DefenseLayerValidator',
    'ValidationConfig',
    'create_validation_pipeline',
    'run_quick_validation',
]
