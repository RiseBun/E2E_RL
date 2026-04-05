"""消融实验对比分析。

对比不同配置下的模型性能：
1. refiner_debug: Refiner + HUF (baseline)
2. refiner_with_scorer: Refiner + HUF (命名误导，实际无 Scorer)
3. rule_huf_test: Refiner + HUF (规则调参)
4. refiner_scorer_huf: Refiner + Scorer + HUF (完整三层)
"""

import torch
import yaml
from pathlib import Path

experiments_dir = Path('/mnt/cpfs/prediction/lipeinan/RL/E2E_RL/experiments')

# 定义要对比的实验
experiments = {
    'refiner_debug': {
        'name': 'Refiner + HUF (Baseline)',
        'config': experiments_dir / 'refiner_debug' / 'config.yaml',
        'checkpoint': experiments_dir / 'refiner_debug' / 'checkpoint_final.pth',
    },
    'refiner_with_scorer': {
        'name': 'Refiner + HUF (误命名)',
        'config': experiments_dir / 'refiner_with_scorer' / 'config.yaml',
        'checkpoint': experiments_dir / 'refiner_with_scorer' / 'checkpoint_final.pth',
    },
    'rule_huf_test': {
        'name': 'Refiner + HUF (规则调参)',
        'config': experiments_dir / 'rule_huf_test' / 'config.yaml',
        'checkpoint': experiments_dir / 'rule_huf_test' / 'checkpoint_final.pth',
    },
    'refiner_scorer_huf': {
        'name': 'Refiner + Scorer + HUF (完整三层)',
        'config': experiments_dir / 'refiner_scorer_huf' / 'config.yaml',
        'checkpoint': experiments_dir / 'refiner_scorer_huf' / 'checkpoint_final.pth',
    },
}

print("=" * 80)
print("消融实验对比分析")
print("=" * 80)
print()

results = []

for exp_key, exp_info in experiments.items():
    print(f"\n{'─'*80}")
    print(f"实验: {exp_info['name']}")
    print(f"ID: {exp_key}")
    print(f"{'─'*80}")
    
    # 检查文件是否存在
    config_exists = exp_info['config'].exists()
    ckpt_exists = exp_info['checkpoint'].exists()
    
    if not config_exists or not ckpt_exists:
        print(f"  ❌ 文件缺失:")
        if not config_exists:
            print(f"     Config: {exp_info['config']}")
        if not ckpt_exists:
            print(f"     Checkpoint: {exp_info['checkpoint']}")
        continue
    
    # 加载配置
    with open(exp_info['config'], 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 加载 checkpoint
    try:
        ckpt = torch.load(exp_info['checkpoint'], map_location='cpu')
        epoch = ckpt.get('epoch', 'N/A')
    except Exception as e:
        print(f"  ⚠️  无法加载 checkpoint: {e}")
        epoch = 'Error'
    
    # 提取关键配置
    scorer_enabled = cfg.get('reliability_net', {}).get('enabled', False)
    huf_enabled = cfg.get('huf', {}).get('enabled', False)
    huf_mode = cfg.get('huf', {}).get('mode', 'N/A')
    pretrained_path = cfg.get('reliability_net', {}).get('pretrained_path', None)
    
    # 打印配置信息
    print(f"  配置:")
    print(f"    • Scorer 启用: {'✅' if scorer_enabled else '❌'}")
    if scorer_enabled and pretrained_path:
        print(f"    • Scorer 预训练: {Path(pretrained_path).name}")
    print(f"    • HUF 启用: {'✅' if huf_enabled else '❌'}")
    if huf_enabled:
        print(f"    • HUF 模式: {huf_mode}")
    print(f"    • 训练 Epoch: {epoch}")
    
    results.append({
        'id': exp_key,
        'name': exp_info['name'],
        'scorer': scorer_enabled,
        'huf': huf_enabled,
        'huf_mode': huf_mode if huf_enabled else 'N/A',
        'epoch': epoch,
        'has_ckpt': ckpt_exists,
    })

# 打印对比表格
print(f"\n\n{'='*80}")
print("实验对比总结")
print(f"{'='*80}")
print()

if results:
    # 表头
    print(f"{'实验 ID':<25} {'Scorer':>8} {'HUF':>6} {'HUF模式':>10} {'Epoch':>6} {'状态':>6}")
    print("-" * 70)
    
    for r in results:
        status = "✅" if r['has_ckpt'] else "❌"
        scorer_str = "✅" if r['scorer'] else "❌"
        huf_str = "✅" if r['huf'] else "❌"
        
        print(
            f"{r['id']:<25} "
            f"{scorer_str:>8} "
            f"{huf_str:>6} "
            f"{r['huf_mode']:>10} "
            f"{str(r['epoch']):>6} "
            f"{status:>6}"
        )
    
    print()
    print("关键发现:")
    print("  1. refiner_with_scorer 虽然名字叫 'with_scorer'，但 Scorer 实际未启用")
    print("  2. refiner_scorer_huf 是唯一真正使用 Scorer 的实验")
    print("  3. 所有实验都启用了 HUF（规则筛选）")
    print()
    print("建议下一步:")
    print("  • 运行评估脚本，对比各实验的 Reward 指标")
    print("  • 可视化轨迹修正效果")
    print("  • 分析 Scorer 对最终性能的贡献")
