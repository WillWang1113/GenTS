import torch
from typing import Dict
import math


def cosine_schedule(n_steps: int, s: float = 0.008) -> Dict[str, torch.Tensor]:
    """Cosine schedule for noise schedule.

    Args:
        n_steps (int): total number of steps.
        s (float, optional): tolerance. Defaults to 0.008.

    Returns:
        Dict[str, torch.Tensor]: noise schedule.
    """
    steps = n_steps + 1
    x = torch.linspace(0, n_steps, steps)
    alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.001, 0.999)
    alphas = 1.0 - betas
    alphas_bars = torch.cumprod(alphas, dim=0)
    noise_schedule = dict(
        betas=betas,
        alphas=alphas,
        alpha_bars=alphas_bars,
        beta_bars=None,
    )
    return noise_schedule


def linear_schedule(
    n_steps: int, min_beta: float = 1e-4, max_beta: float = 2e-2
) -> Dict[str, torch.Tensor]:
    """Linear schedule for noise schedule.

    Args:
        n_steps (int): total number of steps.
        min_beta (float, optional): mininum of beta. Defaults to 1e-4.
        max_beta (float, optional): maximum of beta. Defaults to 2e-2.

    Returns:
        Dict[str, torch.Tensor]: noise schedule.
    """
    betas = torch.linspace(min_beta, max_beta, n_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "alpha_bars": alpha_bars.float(),
        "beta_bars": None,
        "alphas": alphas.float(),
        "betas": betas.float(),
    }

