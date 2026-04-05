from __future__ import annotations

from typing import Optional

import torch


def pooled_scene_token(bev_embed: torch.Tensor, method: str = 'mean') -> torch.Tensor:
    """Pool BEV feature tokens into a compact scene token.

    Args:
        bev_embed: [num_tokens, B, D] or [B, D, H, W]
        method: pooling method, currently 'mean' or 'max'.
    Returns:
        Tensor: [B, D]
    """
    if bev_embed.dim() == 3:
        bev_embed = bev_embed.permute(1, 0, 2)
    elif bev_embed.dim() == 4:
        bev_embed = bev_embed.flatten(2).permute(0, 2, 1)
    else:
        raise ValueError(f'Unsupported bev_embed shape {tuple(bev_embed.shape)}')

    if method == 'mean':
        return bev_embed.mean(dim=1)
    if method == 'max':
        return bev_embed.max(dim=1).values
    raise ValueError(f'Unsupported pooling method: {method}')


def ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def canonicalize_ego_fut_preds(ego_fut_preds: torch.Tensor) -> torch.Tensor:
    """Ensure ego_fut_preds has shape [B, M, T, 2]."""
    if ego_fut_preds.dim() == 3:
        ego_fut_preds = ego_fut_preds.unsqueeze(1)
    if ego_fut_preds.dim() != 4:
        raise ValueError(f'Unsupported ego_fut_preds shape {tuple(ego_fut_preds.shape)}')
    return ego_fut_preds
