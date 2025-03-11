import math

import torch
from einops import rearrange, repeat
import numpy as np


def dft(x: torch.Tensor) -> torch.Tensor:
    """Compute the DFT of the input time series by keeping only the non-redundant components.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)

    # Compute the FFT until the Nyquist frequency
    dft_full = torch.fft.rfft(x, dim=1, norm="ortho")
    dft_re = torch.real(dft_full)
    dft_im = torch.imag(dft_full)

    # The first harmonic corresponds to the mean, which is always real
    zero_padding = torch.zeros_like(dft_im[:, 0, :], device=x.device)
    assert torch.allclose(dft_im[:, 0, :], zero_padding), (
        f"The first harmonic of a real time series should be real, yet got imaginary part {dft_im[:, 0, :]}."
    )
    dft_im = dft_im[:, 1:]

    # If max_len is even, the last component is always zero
    if max_len % 2 == 0:
        assert torch.allclose(dft_im[:, -1, :], zero_padding), (
            f"Got an even {max_len=}, which should be real at the Nyquist frequency, yet got imaginary part {dft_im[:, -1, :]}."
        )
        dft_im = dft_im[:, :-1]

    # Concatenate real and imaginary parts
    x_tilde = torch.cat((dft_re, dft_im), dim=1)
    assert x_tilde.size() == x.size(), (
        f"The DFT and the input should have the same size. Got {x_tilde.size()} and {x.size()} instead."
    )

    return x_tilde.detach()


def idft(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse DFT of the input DFT that only contains non-redundant components.

    Args:
        x (torch.Tensor): DFT of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: Inverse DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)
    n_real = math.ceil((max_len + 1) / 2)

    # Extract real and imaginary parts
    x_re = x[:, :n_real, :]
    x_im = x[:, n_real:, :]

    # Create imaginary tensor
    zero_padding = torch.zeros(size=(x.size(0), 1, x.size(2)))
    x_im = torch.cat((zero_padding, x_im), dim=1)

    # If number of time steps is even, put the null imaginary part
    if max_len % 2 == 0:
        x_im = torch.cat((x_im, zero_padding), dim=1)

    assert x_im.size() == x_re.size(), (
        f"The real and imaginary parts should have the same shape, got {x_re.size()} and {x_im.size()} instead."
    )

    x_freq = torch.complex(x_re, x_im)

    # Apply IFFT
    x_time = torch.fft.irfft(x_freq, n=max_len, dim=1, norm="ortho")

    assert isinstance(x_time, torch.Tensor)
    assert x_time.size() == x.size(), (
        f"The inverse DFT and the input should have the same size. Got {x_time.size()} and {x.size()} instead."
    )

    return x_time.detach()


def compute_downsample_rate(input_length: int,
                            n_fft: int,
                            downsampled_width: int):
    return round(input_length / (np.log2(n_fft) - 1) / downsampled_width) if input_length >= downsampled_width else 1




def time_to_timefreq(x, n_fft: int, C: int, norm:bool=True):
    """
    x: (B, C, L)
    """
    x = rearrange(x, 'b c l -> (b c) l')
    x = torch.stft(x, n_fft, normalized=norm, return_complex=True, window=torch.hann_window(window_length=n_fft, device=x.device))
    x = torch.view_as_real(x)  # (B, N, T, 2); 2: (real, imag)
    x = rearrange(x, '(b c) n t z -> b (c z) n t ', c=C)  # z=2 (real, imag)
    return x.float()  # (B, C, H, W)


def timefreq_to_time(x, n_fft: int, C: int, norm:bool=True):
    x = rearrange(x, 'b (c z) n t -> (b c) n t z', c=C).contiguous()
    x = x.contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft, normalized=norm, window=torch.hann_window(window_length=n_fft, device=x.device))
    x = rearrange(x, '(b c) l -> b c l', c=C)
    return x.float()



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
        xf_l = repeat(xf_l, 'b c 1 w -> b c h w', h=xf.shape[2]).float()  # (b c h w)
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
        xf_h = torch.cat((xf_h[:,:,[0],:], xf_h), dim=2).float()  # (b c h w)
    return xf_h