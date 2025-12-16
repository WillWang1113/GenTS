from abc import ABC, abstractmethod
import torch
import torchaudio.transforms as T

# from utils.utils_data import MinMaxScaler, MinMaxArgs
import numpy as np
import torch.nn as nn


def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, seq_len):
        # self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, seq_len, delay, embedding):
        super().__init__(seq_len)
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None

    def pad_to_square(self, x, mask=0):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0,
            max_side - rows,
            0,
            max_side - cols,
        )  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode="constant", value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal, pad=True, mask=0):
        device = signal.device
        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if (
            i * self.delay != self.seq_len
            and i * self.delay + self.embedding > self.seq_len
        ):
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            # end = start + (self.embedding - 1) - missing_vals
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image

    def img_to_ts(self, img):
        
        # In the original implementation, the shape is memeorized in self.img_shape during ts_to_img
        # Here, we recompute the shape based on seq_len, delay, and embedding for better modularity
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            # start = i * self.delay
            # end = start + self.embedding
            # x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if (
            i * self.delay != self.seq_len
            and i * self.delay + self.embedding > self.seq_len
        ):
            # start = i * self.delay
            # end = signal[:, start:].permute(0, 2, 1).shape[-1]
            # end = start + (self.embedding - 1) - missing_vals
            # x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1
            
            
        img_non_square = self.unpad(img, (None, None, self.embedding, i))

        batch, channels, rows, cols = img_non_square.shape

        reconstructed_x_time_series = torch.zeros((batch, channels, self.seq_len))

        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed_x_time_series[:, :, start:end] = img_non_square[:, :, :, i]

        ### SPECIAL CASE
        start = (cols - 1) * self.delay
        end = reconstructed_x_time_series[:, :, start:].shape[-1]
        reconstructed_x_time_series[:, :, start:] = img_non_square[:, :, :end, cols - 1]
        reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)

        return reconstructed_x_time_series.cuda()


class STFTEmbedder(TsImgEmbedder):
    """
    STFT transformation
    """

    def __init__(self, seq_len, n_fft, hop_length):
        super().__init__(seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_real, self.max_real, self.min_imag, self.max_imag = (
            None,
            None,
            None,
            None,
        )

    def cache_min_max_params(self, train_data):
        """
        Args:
            train_data: training timeseries dataset. shape: B*L*K
        this function initializes the min and max values for the real and imaginary parts.
        we'll use this function only once, before the training loop starts.
        """
        real, imag = self.stft_transform(train_data)
        # compute and cache min and max values
        real, min_real, max_real = MinMaxScaler(real.numpy(), True)
        imag, min_imag, max_imag = MinMaxScaler(imag.numpy(), True)
        self.min_real, self.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
        self.min_imag, self.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)

    def stft_transform(self, data):
        """
        Args:
            data: time series data. Shape: B*L*K
        Returns:
            real and imaginary parts of the STFT transformation
        """
        data = torch.permute(
            data, (0, 2, 1)
        )  # we permute to match requirements of torchaudio.transforms.Spectrogram
        spec = T.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, center=True, power=None
        ).to(data.device)
        transformed_data = spec(data)
        return transformed_data.real, transformed_data.imag

    def ts_to_img(self, signal):
        assert self.min_real is not None, (
            "use init_norm_args() to compute scaling arguments"
        )
        device = signal.device
        # convert to complex spectrogram
        real, imag = self.stft_transform(signal)
        # MinMax scaling
        real = (
            MinMaxArgs(real, self.min_real.to(device), self.max_real.to(device)) - 0.5
        ) * 2
        imag = (
            MinMaxArgs(imag, self.min_imag.to(device), self.max_imag.to(device)) - 0.5
        ) * 2
        # stack real and imag parts
        stft_out = torch.cat((real, imag), dim=1)
        return stft_out

    def img_to_ts(self, x_image):
        device = x_image.device
        n_fft = self.n_fft
        hop_length, length = self.hop_length, self.seq_len
        min_real, max_real, min_imag, max_imag = (
            self.min_real.to(device),
            self.max_real.to(device),
            self.min_imag.to(device),
            self.max_imag.to(device),
        )
        # -- combine real and imaginary parts --
        split = torch.split(
            x_image, x_image.shape[1] // 2, dim=1
        )  # x_image.shape[1] is twice the size of the original dim

        real, imag = split[0], split[1]
        unnormalized_real = ((real / 2) + 0.5) * (max_real - min_real) + min_real
        unnormalized_imag = ((imag / 2) + 0.5) * (max_imag - min_imag) + min_imag
        unnormalized_stft = torch.complex(unnormalized_real, unnormalized_imag)
        # -- inverse stft --
        ispec = T.InverseSpectrogram(
            n_fft=n_fft, hop_length=hop_length, center=True
        ).to(device)

        x_time_series = ispec(unnormalized_stft, length)

        return torch.permute(x_time_series, (0, 2, 1))  # B*L*K(C)



class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True,warmup=0):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []
        self.warmup = warmup

    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    if self.num_updates > self.warmup:
                        shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                    else:
                        shadow_params[sname].copy_(m_param[key])
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)