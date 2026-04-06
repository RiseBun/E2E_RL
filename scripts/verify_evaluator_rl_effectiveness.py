"""A/B 对比实验：验证 LearnedUpdateGate 对 RL 训练的有效性"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from scipy import stats

def load_training_log(log_path):
    """加载训练日志"""
    metrics = {
        'epoch': [],
        'rl_loss': [],
        'retained_adv': [],
        'filtered_adv': [],
        'retention_ratio': [],
        'entropy': [],
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            if '[RL Epoch' in line:
                # 解析日志行
                # 示例: [RL Epoch 0] loss=-41.6687 pg=-41.2702 entropy=-0.3984 adv=-1.3295 retent=50.06%
                parts = line.strip().split()
                epoch = int(parts[2].rstrip(']'))
                
                metrics['epoch'].append(epoch)
                
                # 提取各项指标
                for part in parts:
                    if part.startswith('loss='):
                        metrics['rl_loss'].append(float(part.split('=')[1]))
                    elif part.startswith('entropy='):
                        metrics['entropy'].append(float(part.split('=')[1]))
                    elif part.startswith('adv='):
                        pass  # 这是 overall adv
                    elif part.startswith('retent='):
                        metrics['retention_ratio'].append(float(part.split('=')[1].rstrip('%')))
                
                # 如果有 retained_adv 和 filtered_adv
                if '[Online Stats]' in line:
                    # 解析下一行或同一行
                    pass
    
    return metrics

def compare_training_curves(exp_a_metrics, exp_b_metrics, save_path='training_comparison.png'):
    """对比训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. RL Loss 对比
    ax = axes[0, 0]
    if exp_a_metrics['rl_loss']:
        ax.plot(exp_a_metrics['epoch'], exp_a_metrics['rl_loss'], 
                'r-', linewidth=2, label='Baseline (No Gate)', alpha=0.7)
    if exp_b_metrics['rl_loss']:
        ax.plot(exp_b_metrics['epoch'], exp_b_metrics['rl_loss'], 
                'b-', linewidth=2, label='With LearnedUpdateGate', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('RL Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Advantage 对比
    ax = axes[0, 1]
    if 'retained_adv' in exp_a_metrics and exp_a_metrics['retained_adv']:
        ax.plot(exp_a_metrics['epoch'], exp_a_metrics['retained_adv'], 
                'r-', linewidth=2, label='Baseline', alpha=0.7)
    if 'retained_adv' in exp_b_metrics and exp_b_metrics['retained_adv']:
        ax.plot(exp_b_metrics['epoch'], exp_b_metrics['retained_adv'], 
                'b-', linewidth=2, label='With Gate', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Retained Advantage', fontsize=12)
    ax.set_title('Advantage Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Entropy 对比
    ax = axes[0, 2]
    if exp_a_metrics['entropy']:
        ax.plot(exp_a_metrics['epoch'], exp_a_metrics['entropy'], 
                'r-', linewidth=2, label='Baseline', alpha=0.7)
    if exp_b_metrics['entropy']:
        ax.plot(exp_b_metrics['epoch'], exp_b_metrics['entropy'], 
                'b-', linewidth=2, label='With Gate', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title('Policy Entropy (探索程度)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 收敛速度对比
    ax = axes[1, 0]
    # 计算达到某个 loss 阈值需要的 epoch
    threshold = -30.0  # 示例阈值
    if exp_a_metrics['rl_loss']:
        epochs_to_converge_a = next(
            (e for e, l in zip(exp_a_metrics['epoch'], exp_a_metrics['rl_loss']) if l < threshold),
            len(exp_a_metrics['epoch'])
        )
        ax.bar(['Baseline'], [epochs_to_converge_a], color='red', alpha=0.7, label='No Gate')
    
    if exp_b_metrics['rl_loss']:
        epochs_to_converge_b = next(
            (e for e, l in zip(exp_b_metrics['epoch'], exp_b_metrics['rl_loss']) if l < threshold),
            len(exp_b_metrics['epoch'])
        )
        ax.bar(['With Gate'], [epochs_to_converge_b], color='blue', alpha=0.7, label='With Gate')
    
    ax.set_ylabel('Epochs to Converge', fontsize=12)
    ax.set_title(f'Convergence Speed (loss < {threshold})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Final performance 对比
    ax = axes[1, 1]
    final_metrics = {}
    
    if exp_a_metrics['rl_loss']:
        final_a = np.mean(exp_a_metrics['rl_loss'][-5:])  # 最后 5 个 epoch 的平均
        final_metrics['Baseline'] = final_a
    
    if exp_b_metrics['rl_loss']:
        final_b = np.mean(exp_b_metrics['rl_loss'][-5:])
        final_metrics['With Gate'] = final_b
    
    ax.bar(final_metrics.keys(), final_metrics.values(), 
           color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Final RL Loss (avg last 5 epochs)', fontsize=12)
    ax.set_title('Final Performance', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 6. 训练稳定性对比
    ax = axes[1, 2]
    if len(exp_a_metrics['rl_loss']) > 5:
        std_a = np.std(exp_a_metrics['rl_loss'][-20:])  # 最后 20 个 epoch 的标准差
        ax.bar(['Baseline'], [std_a], color='red', alpha=0.7)
    
    if len(exp_b_metrics['rl_loss']) > 5:
        std_b = np.std(exp_b_metrics['rl_loss'][-20:])
        ax.bar(['With Gate'], [std_b], color='blue', alpha=0.7)
    
    ax.set_ylabel('Loss Std Dev (last 20 epochs)', fontsize=12)
    ax.set_title('Training Stability (lower is better)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training comparison to {save_path}")
    plt.close()
    
    return final_metrics

def statistical_test(exp_a_metrics, exp_b_metrics):
    """统计显著性检验"""
    if not exp_a_metrics['rl_loss'] or not exp_b_metrics['rl_loss']:
        return None
    
    # 使用最后 20 个 epoch 的数据
    last_a = exp_a_metrics['rl_loss'][-20:]
    last_b = exp_b_metrics['rl_loss'][-20:]
    
    # t-test
    t_stat, p_value = stats.ttest_ind(last_a, last_b)
    
    # Cohen's d (效应量)
    pooled_std = np.sqrt((np.std(last_a)**2 + np.std(last_b)**2) / 2)
    cohens_d = (np.mean(last_b) - np.mean(last_a)) / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'mean_a': np.mean(last_a),
        'mean_b': np.mean(last_b),
        'improvement': (np.mean(last_b) - np.mean(last_a)) / abs(np.mean(last_a)) * 100
    }

def analyze_filtering_quality(exp_b_metrics):
    """分析筛选质量"""
    if 'retained_adv' not in exp_b_metrics or 'filtered_adv' not in exp_b_metrics:
        return None
    
    retained = exp_b_metrics['retained_adv']
    filtered = exp_b_metrics['filtered_adv']
    
    # 在整个训练过程中，保留的样本是否 consistently 优于过滤的样本
    diff = [r - f for r, f in zip(retained, filtered)]
    
    return {
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'positive_ratio': np.sum(np.array(diff) > 0) / len(diff),
        'min_diff': np.min(diff),
        'max_diff': np.max(diff)
    }

def main():
    print("=" * 80)
    print("A/B 实验：验证 LearnedUpdateGate 的有效性")
    print("=" * 80)
    
    # 配置
    exp_a_log = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/ablation_baseline/training.log'
    exp_b_log = '/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/ablation_full/training.log'
    output_dir = Path('/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments/ablation_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 加载训练日志
    print("\n[Step 1] 加载训练日志...")
    exp_a_metrics = load_training_log(exp_a_log)
    exp_b_metrics = load_training_log(exp_b_log)
    print(f"✓ 实验 A (Baseline): {len(exp_a_metrics['epoch'])} epochs")
    print(f"✓ 实验 B (With Gate): {len(exp_b_metrics['epoch'])} epochs")
    
    # Step 2: 对比训练曲线
    print("\n[Step 2] 生成对比图表...")
    final_metrics = compare_training_curves(
        exp_a_metrics, 
        exp_b_metrics,
        save_path=output_dir / 'training_comparison.png'
    )
    
    # Step 3: 统计显著性检验
    print("\n[Step 3] 统计显著性检验...")
    test_result = statistical_test(exp_a_metrics, exp_b_metrics)
    
    if test_result:
        print(f"  t-statistic:  {test_result['t_statistic']:.4f}")
        print(f"  p-value:      {test_result['p_value']:.6f}")
        print(f"  significant:  {'✅ Yes (p < 0.05)' if test_result['significant'] else '❌ No'}")
        print(f"  Cohen's d:    {test_result['cohens_d']:.4f}")
        print(f"  improvement:  {test_result['improvement']:.2f}%")
    
    # Step 4: 筛选质量分析
    print("\n[Step 4] 筛选质量分析...")
    filtering_quality = analyze_filtering_quality(exp_b_metrics)
    
    if filtering_quality:
        print(f"  Mean diff (retained - filtered): {filtering_quality['mean_diff']:.4f}")
        print(f"  Positive ratio: {filtering_quality['positive_ratio']*100:.1f}%")
        print(f"  筛选器在 {filtering_quality['positive_ratio']*100:.1f}% 的 epoch 中成功筛选出更好的样本")
    
    # Step 5: 总结
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    evidence_count = 0
    
    if test_result and test_result['significant']:
        print("✅ 证据 1: 性能提升具有统计显著性")
        evidence_count += 1
    else:
        print("❌ 证据 1: 性能提升不显著")
    
    if test_result and test_result['cohens_d'] > 0.5:
        print(f"✅ 证据 2: 效应量大 (Cohen's d = {test_result['cohens_d']:.2f})")
        evidence_count += 1
    elif test_result:
        print(f"⚠️  证据 2: 效应量中等 (Cohen's d = {test_result['cohens_d']:.2f})")
        evidence_count += 0.5
    
    if filtering_quality and filtering_quality['positive_ratio'] > 0.8:
        print(f"✅ 证据 3: 筛选器 consistently 有效 ({filtering_quality['positive_ratio']*100:.1f}%)")
        evidence_count += 1
    elif filtering_quality:
        print(f"⚠️  证据 3: 筛选器部分有效 ({filtering_quality['positive_ratio']*100:.1f}%)")
        evidence_count += 0.5
    
    if test_result and test_result['improvement'] > 2:
        print(f"✅ 证据 4: 性能提升 > 2% ({test_result['improvement']:.2f}%)")
        evidence_count += 1
    
    print(f"\n总证据数: {evidence_count}/4")
    
    if evidence_count >= 3:
        print("\n🎉 强证据: LearnedUpdateGate 显著提升 RL 训练效果！")
    elif evidence_count >= 2:
        print("\n✓ 中等证据: LearnedUpdateGate 对 RL 训练有积极影响")
    else:
        print("\n⚠️  弱证据: 需要更多实验验证")
    
    print("=" * 80)

if __name__ == '__main__':
    main()
