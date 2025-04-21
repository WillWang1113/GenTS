import torch
import torch.nn as nn
from torchcde import CubicSpline, cdeint
from torchvision.ops import MLP


####################################
########### Neural DE ##############
####################################


class ContinuousRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(
            self.input_channels + self.hidden_channels, self.input_channels
        )
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer("out_base", out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(
            self.input_channels, self.hidden_channels
        )

    def forward(self, t, z):
        # z is a tensor of shape (..., input_channels + hidden_channels)
        x = z[..., : self.input_channels]
        h = z[..., self.input_channels :]
        # In theory the hidden state must lie in this region. And most of the time it does anyway! Very occasionally
        # it escapes this and breaks everything, though. (Even when using adaptive solvers or small step sizes.) Which
        # is kind of surprising given how similar the GRU-ODE is to a standard negative exponential problem, we'd
        # expect to get absolute stability without too much difficulty. Maybe there's a bug in the implementation
        # somewhere, but not that I've been able to find... (and h does only escape this region quite rarely.)
        h = h.clamp(-1, 1)
        # model_out is a tensor of shape (..., hidden_channels)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels :, 0] = model_out
        return out


class NeuralCDE(torch.nn.Module):
    """A Neural CDE model. Provides a wrapper around the lower-level cdeint function, to get a flexible Neural CDE
    model.

    Specifically, considering the CDE
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where X is determined by the datasets, and given some terminal time t_N, then this model first computes z_{t_N}, then
    performs a linear function on it, and then outputs the result.

    It's known that linear functions on CDEs are universal approximators, so this is a very general type of model.
    """

    def __init__(
        self, func, input_channels, hidden_channels, output_channels, initial=True
    ):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from datasets (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        # import pdb
        # pdb.set_trace()
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

        self.activation_fn = torch.sigmoid

    def extra_repr(self):
        return (
            "input_channels={}, hidden_channels={}, output_channels={}, initial={}"
            "".format(
                self.input_channels,
                self.hidden_channels,
                self.output_channels,
                self.initial,
            )
        )

    def forward(self, times, coeffs, final_index, z0=None, stream=True, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t f``or which there was datasets.
        """
        # Extract the sizes of the batch dimensions from the coefficients
        # coeff, _, _, _ = coeffs
        batch_dims = coeffs.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, (
                "coeff.shape[:-2] must be the same as final_index.shape. "
                "coeff.shape[:-2]={}, final_index.shape={}"
                "".format(batch_dims, final_index.shape)
            )

        cubic_spline = CubicSpline(coeffs, times)
        # cubic_spline = NaturalCubicSpline(times, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(
                    *batch_dims,
                    self.hidden_channels,
                    dtype=coeffs.dtype,
                    device=coeffs.device,
                )
            else:
                z0 = self.initial_network(
                    cubic_spline.evaluate(cubic_spline.interval[0])
                )
                # z0 = self.initial_network(cubic_spline.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            # continuing adventures in ugly hacks
            if isinstance(self.func, ContinuousRNNConverter):
                z0_extra = torch.zeros(
                    *batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device
                )
                z0 = torch.cat([z0_extra, z0], dim=-1)
        # Figure out what times we need to solve for

        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True
            )
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [
                    times[0].unsqueeze(0),
                    times[sorted_final_index],
                    times[-1].unsqueeze(0),
                ]
            )

        # Switch default solver
        if "method" not in kwargs:
            kwargs["method"] = "rk4"
        if kwargs["method"] == "rk4":
            if "options" not in kwargs:
                kwargs["options"] = {}
            options = kwargs["options"]
            if "step_size" not in options and "grid_constructor" not in options:
                time_diffs = times[1:] - times[:-1]
                options["step_size"] = time_diffs.min().item()
        # Actually solve the CDE
        z_t = cdeint(X=cubic_spline, z0=z0, func=self.func, t=t, **kwargs)
        # z_t = cdeint(
        #     dX_dt=cubic_spline.derivative, z0=z0, func=self.func, t=t, **kwargs
        # )

        # Organise the output

        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            # for i in range(len(z_t.shape) - 2, 0, -1):
            #     z_t = z_t.transpose(0, i)
            pass

        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = (
                final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            )
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0).type(torch.int64)
        # z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)
        # Linear map and return
        pred_y = self.linear(z_t)
        pred_y = self.activation_fn(pred_y)
        return pred_y


class FinalTanh(torch.nn.Module):
    def __init__(
        self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers
    ):
        super(FinalTanh, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        )
        self.linear_out = torch.nn.Linear(
            hidden_hidden_channels, input_channels * hidden_channels
        )

    def extra_repr(self):
        return (
            "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}"
            "".format(
                self.input_channels,
                self.hidden_channels,
                self.hidden_hidden_channels,
                self.num_hidden_layers,
            )
        )

    def forward(self, t, z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(
            *z.shape[:-1], self.hidden_channels, self.input_channels
        )
        z = z.tanh()
        return z


class RNNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_size=128,
        out_dim=128,
        num_layers=1,
        dropout=0.0,
        rnn_type="gru",
        **kwargs,
    ):
        super().__init__()
        assert rnn_type in ["gru", "lstm", "rnn"]
        if rnn_type == "gru":
            rnn_class = nn.GRU
        elif rnn_type == "lstm":
            rnn_class = nn.LSTM
        else:
            rnn_class = nn.RNN

        self.gru = rnn_class(
            in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        emb = self.proj(h)
        return emb


########################################
########### Series Decomp ##############
########################################


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)

        res = x - moving_mean
        return res, moving_mean


######################################
########### MLP Enc Dec ##############
######################################


class MLPEncoder(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[64, 128, 256], **kwargs
    ):
        super().__init__()
        # Build Encoder
        self.encoder = MLP(seq_len * seq_dim, hidden_size_list + [latent_dim])

    def forward(self, x):
        x = self.encoder(x.flatten(1))
        return x


class MLPDecoder(nn.Module):
    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim,
        hidden_size_list=[256, 128, 64],
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        # Build Decoder
        self.decoder = MLP(latent_dim, hidden_size_list + [seq_len * seq_dim])

    def forward(self, x):
        x = self.decoder(x)
        return x.reshape(-1, self.seq_len, self.seq_dim)




###############################
########### Norm ##############
###############################


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2




