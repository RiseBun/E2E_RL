"""Harmful Update Filtering (HUF) 模块。

受 STAPO 启发，识别并抑制有害的残差更新，对剩余有益更新重归一化损失。
从 sample selection 升级到 update selection。
"""

from E2E_RL.update_filter.config import HUFConfig
from E2E_RL.update_filter.filter import HarmfulUpdateFilter
from E2E_RL.update_filter.scorer import UpdateReliabilityScorer

__all__ = ['HUFConfig', 'UpdateReliabilityScorer', 'HarmfulUpdateFilter']
