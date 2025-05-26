import math
import torch
import numpy as np
from einops import repeat, rearrange
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR


def linear_warmup_cosine_annealingLR(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    linear_warmup_rate: float = 0.05,
    min_lr: float = 5e-4,
):
    assert linear_warmup_rate > 0.0 and linear_warmup_rate < 1.0, (
        "0 < linear_warmup_rate < 1."
    )

    warmup_steps = int(max_steps * linear_warmup_rate)  # n% of max_steps

    # Define the warmup scheduler
    def warmup_lambda(current_step):
        if current_step >= warmup_steps:
            return 1.0
        return float(current_step) / float(max(1, warmup_steps))

    # Create the warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Create the cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer, max_steps - warmup_steps, eta_min=min_lr
    )

    # Combine the warmup and cosine annealing schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


def time_to_timefreq(x, n_fft: int, C: int, norm: bool = True):
    """
    x: (B, C, L)
    """
    x = rearrange(x, "b c l -> (b c) l")
    x = torch.stft(
        x,
        n_fft,
        normalized=norm,
        return_complex=True,
        window=torch.hann_window(window_length=n_fft, device=x.device),
    )
    x = torch.view_as_real(x)  # (B, N, T, 2); 2: (real, imag)
    x = rearrange(x, "(b c) n t z -> b (c z) n t ", c=C)  # z=2 (real, imag)
    return x.float()  # (B, C, H, W)


def timefreq_to_time(x, n_fft: int, C: int, norm: bool = True):
    x = rearrange(x, "b (c z) n t -> (b c) n t z", c=C).contiguous()
    x = x.contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(
        x,
        n_fft,
        normalized=norm,
        window=torch.hann_window(window_length=n_fft, device=x.device),
    )
    x = rearrange(x, "(b c) l -> b c l", c=C)
    return x.float()


def compute_downsample_rate(input_length: int, n_fft: int, downsampled_width: int):
    return (
        round(input_length / (np.log2(n_fft) - 1) / downsampled_width)
        if input_length >= downsampled_width
        else 1
    )


def zero_pad_high_freq(xf, copy=False):
    """
    xf: (B, C, H, W); H: frequency-axis, W: temporal-axis
    """
    if not copy:
        xf_l = torch.zeros(xf.shape).to(xf.device)
        xf_l[:, :, 0, :] = xf[:, :, 0, :]  # (b c h w)
    else:
        # model input: copy the LF component and paste it to the rest of the frequency bands
        xf_l = xf[:, :, [0], :]  # (b c 1 w)
        xf_l = repeat(xf_l, "b c 1 w -> b c h w", h=xf.shape[2]).float()  # (b c h w)
    return xf_l


def zero_pad_low_freq(xf, copy=False):
    """
    xf: (B, C, H, W); H: frequency-axis, W: temporal-axis
    """
    if not copy:
        xf_h = torch.zeros(xf.shape).to(xf.device)
        xf_h[:, :, 1:, :] = xf[:, :, 1:, :]
    else:
        # model input: copy the first HF component, and paste it to the LF band
        xf_h = xf[:, :, 1:, :]  # (b c h-1 w)
        xf_h = torch.cat((xf_h[:, :, [0], :], xf_h), dim=2).float()  # (b c h w)
    return xf_h


def l2norm(t):
    return F.normalize(t, dim=-1)


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.LayerNorm(dim),
        weight_norm(nn.Linear(dim, inner_dim, bias=False)),
        nn.GELU(),
        weight_norm(nn.LayerNorm(inner_dim)),
        nn.Linear(inner_dim, dim, bias=False),
    )


def calculate_padding(kernel_size, stride, dilation):
    """
    Calculate the padding size for a convolutional layer to achieve 'same' padding.

    Args:
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        dilation (int, optional): Dilation rate. Defaults to 1.

    Returns:
        int: Calculated padding size.
    """
    # Effective kernel size considering dilation
    effective_kernel_size = dilation * (kernel_size - 1) + 1

    # Calculate padding
    padding = math.floor((effective_kernel_size - stride) / 2)

    return padding
