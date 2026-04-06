"""可视化 UpdateEvaluator 的筛选效果"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

def load_evaluator_checkpoint(ckpt_path):
    """加载训练好的 Evaluator"""
    from E2E_RL.update_selector.update_evaluator import UpdateEvaluator
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    config = ckpt.get('config', {})
    
    evaluator = UpdateEvaluator(
        scene_dim=config.get('scene_dim', 256),
        plan_len=config.get('plan_len', 6),
        hidden_dim=config.get('hidden_dim', 256),
    )
    evaluator.load_state_dict(ckpt['model_state_dict'])
    evaluator.eval()
    
    return evaluator

def collect_test_samples(data_dir, n_samples=500):
    """收集测试样本"""
    from pathlib import Path
    
    pt_files = sorted(list(Path(data_dir).glob('*.pt')))[:n_samples]
    samples = []
    
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location='cpu')
        
        # 提取关键字段
        sample = {
            'scene_token': data['interface_ego_local']['scene_token'],
            'reference_plan': data['interface_ego_local']['reference_plan'],
            'gt_plan': data['ego_fut_trajs'],
        }
        samples.append(sample)
    
    return samples

def generate_candidates(sample, n_candidates=10):
    """为单个样本生成多个候选修正"""
    ref_plan = sample['reference_plan']
    gt_plan = sample['gt_plan']
    
    candidates = []
    
    # 1. Zero correction
    candidates.append(torch.zeros_like(ref_plan))
    
    # 2. GT-directed corrections (不同 scale)
    direction = gt_plan - ref_plan
    for scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
        correction = direction * scale
        candidates.append(correction)
    
    # 3. Random perturbations
    for _ in range(n_candidates - 6):
        noise = torch.randn_like(ref_plan) * 0.5
        candidates.append(noise)
    
    return torch.stack(candidates)  # [N, T, 2]

def compute_true_rewards(ref_plan, corrections, reward_fn=None):
    """计算每个修正的真实 reward"""
    # 简化版：使用 L2 distance to GT 作为 proxy reward
    # 实际应该用完整的 reward_proxy
    gt_plan = None  # 需要从 sample 中获取
    
    rewards = []
    for corr in corrections:
        corrected = ref_plan + corr
        # 简化的 reward: 离 GT 越近越好
        if gt_plan is not None:
            dist = torch.norm(corrected - gt_plan, dim=-1).mean()
            reward = -dist.item()
        else:
            reward = 0.0
        rewards.append(reward)
    
    return torch.tensor(rewards)

def visualize_ranking_correlation(pred_gains, true_gains, save_path='ranking_corr.png'):
    """可视化预测 vs 真实 gain 的相关性"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(true_gains, pred_gains, alpha=0.6, s=20)
    
    # 计算相关性
    spearman_corr, _ = spearmanr(true_gains, pred_gains)
    kendall_tau, _ = kendalltau(true_gains, pred_gains)
    
    # 拟合直线
    z = np.polyfit(true_gains, pred_gains, 1)
    p = np.poly1d(z)
    x_line = np.linspace(true_gains.min(), true_gains.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit line')
    
    ax.set_xlabel('True Gain', fontsize=12)
    ax.set_ylabel('Predicted Gain', fontsize=12)
    ax.set_title(f'Ranking Correlation\nSpearman={spearman_corr:.3f}, Kendall={kendall_tau:.3f}', 
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram of gains
    ax = axes[1]
    bins = np.linspace(min(true_gains.min(), pred_gains.min()), 
                       max(true_gains.max(), pred_gains.max()), 30)
    ax.hist(true_gains, bins=bins, alpha=0.5, label='True Gains', color='blue')
    ax.hist(pred_gains, bins=bins, alpha=0.5, label='Predicted Gains', color='orange')
    ax.set_xlabel('Gain Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Gain Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ranking correlation plot to {save_path}")
    plt.close()

def visualize_filtering_effect(pred_gains, true_gains, retention_ratio=0.3, 
                                save_path='filtering_effect.png'):
    """可视化筛选效果"""
    n_total = len(pred_gains)
    n_retained = int(n_total * retention_ratio)
    
    # 按预测 gain 排序
    sorted_indices = torch.argsort(pred_gains, descending=True)
    retained_mask = torch.zeros(n_total, dtype=torch.bool)
    retained_mask[sorted_indices[:n_retained]] = True
    
    retained_true_gains = true_gains[retained_mask]
    filtered_true_gains = true_gains[~retained_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    ax = axes[0]
    data_to_plot = [retained_true_gains.numpy(), filtered_true_gains.numpy()]
    bp = ax.boxplot(data_to_plot, labels=['Retained', 'Filtered'], patch_artist=True)
    
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 添加均值线
    retained_mean = retained_true_gains.mean().item()
    filtered_mean = filtered_true_gains.mean().item()
    ax.axhline(y=retained_mean, color='green', linestyle='--', linewidth=2, 
               label=f'Retained mean={retained_mean:.4f}')
    ax.axhline(y=filtered_mean, color='red', linestyle='--', linewidth=2,
               label=f'Filtered mean={filtered_mean:.4f}')
    
    gain_diff = retained_mean - filtered_mean
    ax.set_title(f'Filtering Effect\nGain Diff = {gain_diff:.4f}', fontsize=14)
    ax.set_ylabel('True Gain', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative gain curve
    ax = axes[1]
    sorted_true_gains = true_gains[sorted_indices]
    cumulative_mean = torch.cumsum(sorted_true_gains, dim=0) / torch.arange(1, n_total + 1)
    
    ax.plot(range(n_total), cumulative_mean.numpy(), linewidth=2, color='blue')
    ax.axvline(x=n_retained, color='red', linestyle='--', linewidth=2, 
               label=f'Retention threshold ({retention_ratio*100:.0f}%)')
    ax.set_xlabel('Number of Samples (sorted by predicted gain)', fontsize=12)
    ax.set_ylabel('Cumulative Mean True Gain', fontsize=12)
    ax.set_title('Cumulative Gain Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved filtering effect plot to {save_path}")
    plt.close()
    
    return gain_diff

def main():
    print("=" * 80)
    print("UpdateEvaluator 有效性验证")
    print("=" * 80)
    
    # 配置
    evaluator_ckpt = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/update_evaluator_v4_5k_samples/evaluator_epoch_30.pth'
    data_dir = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/data/vad_dumps_full'
    output_dir = Path('/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/evaluator_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 加载 Evaluator
    print("\n[Step 1] 加载 UpdateEvaluator...")
    evaluator = load_evaluator_checkpoint(evaluator_ckpt)
    print(f"✓ Evaluator 加载完成")
    
    # Step 2: 收集测试样本
    print("\n[Step 2] 收集测试样本...")
    samples = collect_test_samples(data_dir, n_samples=100)
    print(f"✓ 收集了 {len(samples)} 个样本")
    
    # Step 3: 生成候选并评估
    print("\n[Step 3] 生成候选修正并评估...")
    all_pred_gains = []
    all_true_gains = []
    
    for i, sample in enumerate(samples):
        if i % 20 == 0:
            print(f"  处理样本 {i}/{len(samples)}...")
        
        # 生成候选
        candidates = generate_candidates(sample, n_candidates=10)
        ref_plan = sample['reference_plan'].unsqueeze(0).expand(len(candidates), -1, -1)
        scene_token = sample['scene_token'].unsqueeze(0).expand(len(candidates), -1)
        
        # 预测 gain
        with torch.no_grad():
            outputs = evaluator(scene_token, ref_plan, candidates)
            pred_gains = outputs['pred_gain'].squeeze(-1)
        
        # 计算真实 gain (简化版)
        true_gains = compute_true_rewards(sample['reference_plan'], candidates)
        
        all_pred_gains.append(pred_gains)
        all_true_gains.append(true_gains)
    
    all_pred_gains = torch.cat(all_pred_gains)
    all_true_gains = torch.cat(all_true_gains)
    
    print(f"\n✓ 共生成 {len(all_pred_gains)} 个候选修正")
    
    # Step 4: 计算指标
    print("\n[Step 4] 计算评估指标...")
    spearman_corr, _ = spearmanr(all_true_gains.numpy(), all_pred_gains.numpy())
    kendall_tau, _ = kendalltau(all_true_gains.numpy(), all_pred_gains.numpy())
    
    print(f"  Spearman Correlation: {spearman_corr:.4f}")
    print(f"  Kendall Tau:          {kendall_tau:.4f}")
    
    # Step 5: 可视化
    print("\n[Step 5] 生成可视化...")
    visualize_ranking_correlation(
        all_pred_gains.numpy(), 
        all_true_gains.numpy(),
        save_path=output_dir / 'ranking_correlation.png'
    )
    
    gain_diff = visualize_filtering_effect(
        all_pred_gains,
        all_true_gains,
        retention_ratio=0.3,
        save_path=output_dir / 'filtering_effect.png'
    )
    
    print(f"\n  Gain Difference (retained - filtered): {gain_diff:.4f}")
    
    # Step 6: 总结
    print("\n" + "=" * 80)
    print("验证结果总结")
    print("=" * 80)
    print(f"✓ Spearman Correlation: {spearman_corr:.4f} {'(优秀)' if spearman_corr > 0.7 else '(良好)' if spearman_corr > 0.5 else '(需改进)'}")
    print(f"✓ Kendall Tau:          {kendall_tau:.4f} {'(优秀)' if kendall_tau > 0.5 else '(良好)' if kendall_tau > 0.3 else '(需改进)'}")
    print(f"✓ Gain Difference:      {gain_diff:.4f} {'(有效)' if gain_diff > 0 else '(无效)'}")
    print("=" * 80)
    
    if spearman_corr > 0.5 and gain_diff > 0:
        print("\n🎉 结论: UpdateEvaluator 能有效筛选出高质量的更新！")
    else:
        print("\n⚠️  结论: UpdateEvaluator 效果有限，建议重新训练或调整架构")

if __name__ == '__main__':
    main()
