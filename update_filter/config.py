"""HUF 配置：集中管理所有超参数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HUFConfig:
    """Harmful Update Filtering 配置。

    Attributes:
        mode: 过滤模式，'hard' (二值 mask) 或 'soft' (连续权重)
        enabled: 是否启用 HUF
        tau_uncertainty: uncertainty 阈值，高于此值被抑制
        w_confidence: plan_confidence 反转在 uncertainty 中的权重
        w_mode_variance: candidate mode 方差在 uncertainty 中的权重
        w_residual_var: residual 不确定性在 uncertainty 中的权重
        tau_support: support 阈值，低于此值被抑制
        support_alpha: exp(-alpha * norm) 的衰减速率
        max_residual_norm: residual norm 超过此值直接标记为低支持
        tau_drift: drift 阈值，高于此值被抑制
        w_comfort: comfort 变化在 drift 中的权重
        w_curvature: curvature 突变在 drift 中的权重
        w_residual_mag: residual 大小在 drift 中的权重
        soft_temperature: soft mode 下 sigmoid 的温度参数
        min_retention_ratio: 安全下限，至少保留此比例样本
        w_uncertainty_final: 综合评分中 uncertainty 的权重
        w_support_final: 综合评分中 support 的权重
        w_drift_final: 综合评分中 drift 的权重
    """

    mode: str = 'hard'
    enabled: bool = True

    # ---- Uncertainty 评分参数 ----
    tau_uncertainty: float = 0.7
    w_confidence: float = 0.5
    w_mode_variance: float = 0.3
    w_residual_var: float = 0.2

    # ---- Support 评分参数 ----
    tau_support: float = 0.3
    support_alpha: float = 1.0
    max_residual_norm: float = 5.0

    # ---- Drift 评分参数 ----
    tau_drift: float = 0.8
    w_comfort: float = 0.4
    w_curvature: float = 0.3
    w_residual_mag: float = 0.3

    # ---- Learned Scorer 参数 (回归版) ----
    tau_gain: float = 0.05       # 预测提升低于此值认为“虚假高分”
    tau_risk: float = 0.5        # 预测综合风险高于此值认为“有害动作”
    delta_margin: float = 0.01   # 判断 positive_adv 的阈值 (总 reward 差)

    # ---- 安全增强门控 ----
    delta_safe_margin: float = 0.005
    lambda_collision: float = 2.0
    lambda_offroad: float = 1.0
    lambda_comfort: float = 0.5
    lambda_drift: float = 1.0

    # ---- 硬物理底线 (Hard Guards) ----
    max_step_disp: float = 2.0
    collision_guard: float = 0.1  # 碰撞增加量上限

    # ---- STAPO 式门控 (多模态规划 π/H + 正优势 → 静音虚假更新) ----
    # 与 LLM token 级 STAPO 对应：低选中模态概率 π、低归一化熵 H（分布过尖）、且安全优势为正时掩码
    stapo_enabled: bool = False
    stapo_tau_pi: float = 0.35
    stapo_tau_entropy: float = 0.40
    stapo_softmax_temp: float = 1.0

    # ---- Soft mode ----
    soft_temperature: float = 1.0

    # ---- 安全下限 ----
    min_retention_ratio: float = 0.3

    # ---- 综合评分权重 (用于 min_retention 排序) ----
    w_uncertainty_final: float = 0.4
    w_support_final: float = 0.3
    w_drift_final: float = 0.3

    def __post_init__(self):
        if self.mode not in ('hard', 'soft'):
            raise ValueError(f"mode 必须为 'hard' 或 'soft'，收到: {self.mode}")
        if not 0.0 < self.min_retention_ratio <= 1.0:
            raise ValueError(
                f"min_retention_ratio 必须在 (0, 1] 范围内，收到: {self.min_retention_ratio}"
            )
