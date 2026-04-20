"""诊断样本跳过原因。

分析哪些场景被跳过以及跳过的具体原因。
"""

import sys
from pathlib import Path
import logging

# 兼容 diffusers 新版本：旧版 torch 不包含 torch.xpu / torch.mps
# 必须在导入 diffusers 之前设置
import torch

if not hasattr(torch, "xpu"):
    class _TorchXpuCompat:
        @staticmethod
        def empty_cache() -> None:
            return None
        
        @staticmethod
        def is_available() -> bool:
            return False
        
        @staticmethod
        def device_count() -> int:
            return 0
        
        @staticmethod
        def manual_seed(seed: int) -> None:
            torch.manual_seed(seed)
        
        @staticmethod
        def reset_peak_memory_stats(device=None) -> None:
            return None
        
        @staticmethod
        def max_memory_allocated(device=None) -> int:
            return 0
        
        @staticmethod
        def synchronize(device=None) -> None:
            return None
    
    torch.xpu = _TorchXpuCompat()  # type: ignore[attr-defined]

if not hasattr(torch, "mps"):
    class _TorchMpsCompat:
        @staticmethod
        def empty_cache() -> None:
            return None
        
        @staticmethod
        def is_available() -> bool:
            return False
        
        @staticmethod
        def device_count() -> int:
            return 0
        
        @staticmethod
        def manual_seed(seed: int) -> None:
            torch.manual_seed(seed)
        
        @staticmethod
        def reset_peak_memory_stats(device=None) -> None:
            return None
        
        @staticmethod
        def max_memory_allocated(device=None) -> int:
            return 0
        
        @staticmethod
        def synchronize(device=None) -> None:
            return None
    
    torch.mps = _TorchMpsCompat()  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def diagnose_skipped_samples(
    diffusiondrive_root: str,
    data_path: str,
    sensor_path: str,
    agent_config: str,
    checkpoint: str,
    max_samples: int = 100,
):
    """诊断样本跳过的原因。"""
    from pathlib import Path
    import sys
    
    # 添加 DiffusionDrive 到 sys.path，使 navsim 模块可导入
    diffusiondrive_root = Path(diffusiondrive_root)
    if str(diffusiondrive_root) not in sys.path:
        sys.path.insert(0, str(diffusiondrive_root))
        logger.info(f"已添加 DiffusionDrive 到 sys.path: {diffusiondrive_root}")
    
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter
    
    # 加载 agent
    logger.info("加载 DiffusionDrive agent...")
    cfg = OmegaConf.load(agent_config)
    cfg.checkpoint_path = checkpoint
    agent = instantiate(cfg)
    agent.initialize()
    agent.eval()
    
    # 构建 scene loader
    logger.info("构建 SceneLoader...")
    scene_filter = SceneFilter()
    loader = SceneLoader(
        data_path=Path(data_path),
        sensor_blobs_path=Path(sensor_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    
    all_tokens = loader.tokens[:max_samples]
    logger.info(f"总场景数: {len(all_tokens)}")
    
    # 诊断每个场景
    skipped_details = []
    success_details = []
    
    feature_builders = agent.get_feature_builders()
    
    logger.info("\n开始诊断每个场景...")
    
    for idx, token in enumerate(all_tokens):
        if idx % 20 == 0:
            logger.info(f"进度: {idx}/{len(all_tokens)}")
        
        # 检查 1: 是否能获取 agent_input
        try:
            agent_input = loader.get_agent_input_from_token(token)
            frames = loader.scene_frames_dicts[token]
        except FileNotFoundError as e:
            skipped_details.append({
                'token': token,
                'reason': 'FileNotFoundError',
                'detail': str(e)
            })
            continue
        except KeyError as e:
            skipped_details.append({
                'token': token,
                'reason': 'KeyError',
                'detail': str(e)
            })
            continue
        except Exception as e:
            skipped_details.append({
                'token': token,
                'reason': f'Other: {type(e).__name__}',
                'detail': str(e)
            })
            continue
        
        # 检查 2: 是否能构建特征
        try:
            features = {}
            for builder in feature_builders:
                features.update(builder.compute_features(agent_input))
            features = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }
        except Exception as e:
            skipped_details.append({
                'token': token,
                'reason': f'Feature building failed: {type(e).__name__}',
                'detail': str(e)
            })
            continue
        
        # 检查 3: 是否能前向传播
        try:
            with torch.no_grad():
                outputs = agent.forward(features)
        except Exception as e:
            skipped_details.append({
                'token': token,
                'reason': f'Forward failed: {type(e).__name__}',
                'detail': str(e)
            })
            continue
        
        # 成功
        success_details.append({
            'token': token,
            'scene_token': frames[3].get('scene_token', 'N/A'),
        })
    
    # 打印诊断结果
    logger.info("\n" + "="*70)
    logger.info("诊断结果")
    logger.info("="*70)
    logger.info(f"✅ 成功: {len(success_details)}")
    logger.info(f"❌ 跳过: {len(skipped_details)}")
    logger.info(f"成功率: {len(success_details)/len(all_tokens)*100:.1f}%")
    
    # 分析跳过原因
    if skipped_details:
        logger.info("\n📊 跳过原因统计:")
        reason_counts = {}
        for detail in skipped_details:
            reason = detail['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count} ({count/len(skipped_details)*100:.1f}%)")
        
        # 显示前 5 个跳过的详情
        logger.info("\n📝 前 5 个跳过场景的详细信息:")
        for i, detail in enumerate(skipped_details[:5]):
            logger.info(f"  [{i+1}] Token: {detail['token']}")
            logger.info(f"      原因: {detail['reason']}")
            logger.info(f"      详情: {detail['detail'][:100]}")
    
    # 显示成功的场景
    if success_details:
        logger.info("\n✅ 前 5 个成功场景:")
        for i, detail in enumerate(success_details[:5]):
            logger.info(f"  [{i+1}] Token: {detail['token']}, Scene: {detail['scene_token']}")
    
    return skipped_details, success_details


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='诊断样本跳过原因')
    parser.add_argument('--agent_config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sensor_path', type=str, required=True)
    parser.add_argument('--diffusiondrive_root', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()
    
    if args.diffusiondrive_root is None:
        # 尝试多个可能的位置：
        # 1. E2E_RL/projects/DiffusionDrive (当前项目内)
        # 2. RL/DiffusionDrive (兄弟目录)
        # 3. 当前目录的 projects/DiffusionDrive
        script_dir = Path(__file__).resolve().parent
        e2e_rl_dir = script_dir.parent  # E2E_RL/
        rl_dir = e2e_rl_dir.parent  # RL/
        
        options = [
            e2e_rl_dir / 'projects' / 'DiffusionDrive',
            rl_dir / 'DiffusionDrive',
            Path.cwd() / 'projects' / 'DiffusionDrive',
        ]
        
        diffusiondrive_root = None
        for option in options:
            if option.exists():
                diffusiondrive_root = option
                break
        
        if diffusiondrive_root is None:
            raise FileNotFoundError(
                f"DiffusionDrive 未找到！\n"
                f"尝试了以下位置:\n" + 
                "\n".join([f"  - {opt}" for opt in options])
            )
    else:
        diffusiondrive_root = Path(args.diffusiondrive_root)
    
    diagnose_skipped_samples(
        diffusiondrive_root=str(diffusiondrive_root),
        data_path=args.data_path,
        sensor_path=args.sensor_path,
        agent_config=args.agent_config,
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()
