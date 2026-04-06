"""分析筛选器对 Policy Gradient 质量的影响"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_gradient_quality(
    retained_updates,
    filtered_updates,
    policy_network,
    state,
):
    """分析筛选前后梯度质量的差异
    
    Args:
        retained_updates: 筛选器保留的修正量 [N_retained, T, 2]
        filtered_updates: 筛选器过滤的修正量 [N_filtered, T, 2]
        policy_network: CorrectionPolicy 网络
        state: 当前状态 (scene_token, reference_plan)
    """
    
    scene_token, reference_plan = state
    
    # 计算保留样本的梯度
    retained_log_probs = []
    retained_advantages = []
    
    for update in retained_updates:
        # 计算 log π(a|s)
        dist = policy_network.get_distribution(scene_token, reference_plan)
        log_prob = dist.log_prob(update)
        
        # 计算 advantage (简化版，实际应从 reward 计算)
        corrected_plan = reference_plan + update
        advantage = compute_advantage(reference_plan, corrected_plan)
        
        retained_log_probs.append(log_prob.detach())
        retained_advantages.append(advantage.detach())
    
    # 计算过滤样本的梯度
    filtered_log_probs = []
    filtered_advantages = []
    
    for update in filtered_updates:
        dist = policy_network.get_distribution(scene_token, reference_plan)
        log_prob = dist.log_prob(update)
        corrected_plan = reference_plan + update
        advantage = compute_advantage(reference_plan, corrected_plan)
        
        filtered_log_probs.append(log_prob.detach())
        filtered_advantages.append(advantage.detach())
    
    # 分析梯度质量
    retained_log_probs = torch.stack(retained_log_probs)
    retained_advantages = torch.stack(retained_advantages)
    filtered_log_probs = torch.stack(filtered_log_probs)
    filtered_advantages = torch.stack(filtered_advantages)
    
    # Policy gradient: ∇log π * A
    retained_gradients = retained_log_probs * retained_advantages
    filtered_gradients = filtered_log_probs * filtered_advantages
    
    # 计算统计量
    results = {
        'retained': {
            'mean_adv': retained_advantages.mean().item(),
            'std_adv': retained_advantages.std().item(),
            'positive_ratio': (retained_advantages > 0).float().mean().item(),
            'mean_gradient': retained_gradients.mean().item(),
            'gradient_snr': abs(retained_gradients.mean().item()) / (retained_gradients.std().item() + 1e-8),
        },
        'filtered': {
            'mean_adv': filtered_advantages.mean().item(),
            'std_adv': filtered_advantages.std().item(),
            'positive_ratio': (filtered_advantages > 0).float().mean().item(),
            'mean_gradient': filtered_gradients.mean().item(),
            'gradient_snr': abs(filtered_gradients.mean().item()) / (filtered_gradients.std().item() + 1e-8),
        }
    }
    
    return results

def compute_advantage(reference_plan, corrected_plan, reward_fn=None):
    """计算 advantage (简化版)"""
    # 实际应该用完整的 reward 函数
    # 这里用简化的 L2 distance to GT 作为 proxy
    
    # 假设我们有 GT (实际应该从数据中获取)
    # 这里只是一个示例
    gt_plan = reference_plan  # 占位符
    
    # 简化的 reward: 离 GT 越近越好
    def compute_reward(plan):
        dist = torch.norm(plan - gt_plan, dim=-1).mean()
        return -dist
    
    reward_ref = compute_reward(reference_plan)
    reward_corr = compute_reward(corrected_plan)
    
    return reward_corr - reward_ref  # advantage

def visualize_gradient_analysis(results, save_path='gradient_analysis.png'):
    """可视化梯度分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Advantage 分布对比
    ax = axes[0, 0]
    categories = ['Retained', 'Filtered']
    mean_advs = [results['retained']['mean_adv'], results['filtered']['mean_adv']]
    std_advs = [results['retained']['std_adv'], results['filtered']['std_adv']]
    
    ax.bar(categories, mean_advs, yerr=std_advs, 
           color=['green', 'red'], alpha=0.7, capsize=5)
    ax.set_ylabel('Mean Advantage', fontsize=12)
    ax.set_title('Advantage Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (m, s) in enumerate(zip(mean_advs, std_advs)):
        ax.text(i, m + s + 0.01, f'{m:.4f}±{s:.4f}', 
                ha='center', va='bottom', fontsize=10)
    
    # 2. Positive Advantage 比例
    ax = axes[0, 1]
    pos_ratios = [results['retained']['positive_ratio'], 
                  results['filtered']['positive_ratio']]
    
    ax.bar(categories, pos_ratios, color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Positive Advantage Ratio', fontsize=12)
    ax.set_title('Positive Advantage Ratio', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    for i, r in enumerate(pos_ratios):
        ax.text(i, r + 0.02, f'{r*100:.1f}%', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 3. Gradient SNR 对比
    ax = axes[1, 0]
    snrs = [results['retained']['gradient_snr'], 
            results['filtered']['gradient_snr']]
    
    ax.bar(categories, snrs, color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Gradient SNR (Signal-to-Noise Ratio)', fontsize=12)
    ax.set_title('Gradient Quality (Higher is Better)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    for i, s in enumerate(snrs):
        ax.text(i, s + 0.1, f'{s:.2f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. 综合评分
    ax = axes[1, 1]
    # 综合评分 = positive_ratio * gradient_snr
    scores = [pos_ratios[0] * snrs[0], pos_ratios[1] * snrs[1]]
    
    ax.bar(categories, scores, color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Gradient Quality Score', fontsize=12)
    ax.set_title('Overall Gradient Quality', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    for i, sc in enumerate(scores):
        ax.text(i, sc + 0.1, f'{sc:.2f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved gradient analysis to {save_path}")
    plt.close()
    
    return results

def main():
    print("=" * 80)
    print("分析筛选器对 Policy Gradient 质量的影响")
    print("=" * 80)
    
    # 这里需要实际的训练数据
    # 示例代码展示分析框架
    
    print("\n理论分析:")
    print("-" * 80)
    
    print("""
Policy Gradient 公式:
    ∇J(θ) = E[∇log π(a|s) * A(s,a)]

筛选器的作用:
    1. 提高 positive advantage 的比例
       - 不使用筛选器: ~26% positive
       - 使用筛选器: ~50%+ positive
    
    2. 提高 Gradient SNR (信噪比)
       - signal = mean(∇log π * A)
       - noise = std(∇log π * A)
       - SNR = |signal| / noise
    
    3. 稳定训练
       - 减少 gradient variance
       - 避免被负 advantage 样本误导

关键证明:
    如果 retained_updates 的 Gradient SNR > filtered_updates 的 Gradient SNR
    → 筛选器提高了梯度质量
    → 对 RL 训练有益
""")
    
    print("=" * 80)
    print("实际验证方法:")
    print("=" * 80)
    print("""
1. 运行 A/B 实验:
   python scripts/expA_relaxed.py --output_dir experiments/baseline
   python scripts/expC_relaxed.py --output_dir experiments/with_gate

2. 对比训练曲线:
   python scripts/verify_evaluator_rl_effectiveness.py

3. 检查日志中的关键指标:
   - retained_adv vs filtered_adv (应该 retained > filtered)
   - 收敛速度 (with_gate 应该更快)
   - 最终性能 (with_gate 应该更好)

4. 统计显著性检验:
   - t-test: p-value < 0.05
   - Cohen's d: 效应量 > 0.5
""")

if __name__ == '__main__':
    main()
