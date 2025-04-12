import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcde import CubicSpline, cdeint
from torchdiffeq import odeint_adjoint as odeint
from TorchDiffEqPack import odesolve_adjoint_sym12 as odesolve

from src.layers.misc import (
    NONLINEARITIES,
    _flip,
    divergence_approx,
    divergence_bf,
    sample_gaussian_like,
    sample_rademacher_like,
    squeeze,
    unsqueeze,
)


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


class ODEfunc(nn.Module):
    def __init__(
        self,
        diffeq,
        divergence_fn="approximate",
        residual=False,
        rademacher=False,
        div_samples=1,
    ):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher
        self.div_samples = div_samples

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.0))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)
        self._sqjacnorm = None

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        # t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = [sample_rademacher_like(y) for k in range(self.div_samples)]
            else:
                self._e = [sample_gaussian_like(y) for k in range(self.div_samples)]

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence, sqjacnorm = self.divergence_fn(dy, y, e=self._e)
                divergence = divergence.view(batchsize, 1)
            self.sqjacnorm = sqjacnorm
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(
                np.prod(y.shape[1:]), dtype=torch.float32
            ).to(divergence)
        return tuple(
            [dy, -divergence]
            + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]]
        )


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
        self,
        hidden_dims,
        input_shape,
        strides,
        conv,
        layer_type="concat",
        nonlinearity="softplus",
        num_squeeze=0,
        zero_last_weight=True,
    ):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze

        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": IgnoreConv2d,
                "hyper": HyperConv2d,
                "squash": SquashConv2d,
                "concat": ConcatConv2d,
                "concat_v2": ConcatConv2d_v2,
                "concatsquash": ConcatSquashConv2d,
                "blend": BlendConv2d,
                "concatcoord": ConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": IgnoreLinear,
                "hyper": HyperLinear,
                "squash": SquashLinear,
                "concat": ConcatLinear,
                "concat_v2": ConcatLinear_v2,
                "concatsquash": ConcatSquashLinear,
                "blend": BlendLinear,
                "concatcoord": ConcatLinear,
            }[layer_type]

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        self.last_act = NONLINEARITIES[nonlinearity]
        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 0:
                layer_kwargs = {
                    "ksize": 1,
                    "stride": 1,
                    "padding": 0,
                    "transpose": False,
                }
            elif stride == 1:
                layer_kwargs = {
                    "ksize": 3,
                    "stride": 1,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == 2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == -2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": True,
                }
            else:
                raise ValueError("Unsupported stride: {}".format(stride))

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] // 2,
                    hidden_shape[2] // 2,
                )
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] * 2,
                    hidden_shape[2] * 2,
                )

        self.layers = nn.ModuleList(layers)
        if zero_last_weight:
            for name, p in self.layers[-1].named_parameters():
                if "weight" in name:
                    p.data.zero_()
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
    
    def forward(self, t, y):
        dx = y

        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1 or classname.find("Conv") != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[: self.dim_out].view(self.dim_out)
        w = params[self.dim_out :].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        sh = x.shape
        tt = t.expand(sh[0], 1)
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(
            self._hyper_gate(t.view(1, 1))
        ) + self._hyper_bias(t.view(1, 1))


class HyperConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(HyperConv2d, self).__init__()
        assert dim_in % groups == 0 and dim_out % groups == 0, (
            "dim_in and dim_out must both be divisible by groups."
        )
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose

        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d

        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(
            self.dim_in * self.dim_out * self.ksize * self.ksize / self.groups
        )
        if self.transpose:
            weight = params[:weight_size].view(
                self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize
            )
        else:
            weight = params[:weight_size].view(
                self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize
            )
        bias = params[: self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(
            x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
        )


class IgnoreConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(IgnoreConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # self._layer.weight.data.zero_()

    def forward(self, t, x):
        return self._layer(x)


class SquashConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(SquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1))).view(
            1, -1, 1, 1
        )


class ConcatConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # self._layer.weight.data.zero_()

    def forward(self, t, x):
        sh = x.shape
        tt = t.expand(sh[0], 1, *sh[2:])
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatConv2d_v2(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatSquashConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(ConcatSquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))).view(
            1, -1, 1, 1
        ) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatCoordConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(ConcatCoordConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 3,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        b, c, h, w = x.shape
        hh = torch.arange(h).to(x).view(1, 1, h, 1).expand(b, 1, h, w)
        ww = torch.arange(w).to(x).view(1, 1, 1, w).expand(b, 1, h, w)
        tt = t.to(x).view(1, 1, 1, 1).expand(b, 1, h, w)
        x_aug = torch.cat([x, tt, hh, ww], 1)
        return self._layer(x_aug)


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatedLinear, self).__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1
    ):
        super(GatedConv, self).__init__()
        self.layer_f = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
        )
        self.layer_g = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConvTranspose(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
    ):
        super(GatedConvTranspose, self).__init__()
        self.layer_f = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )
        self.layer_g = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class BlendLinear(nn.Module):
    def __init__(self, dim_in, dim_out, layer_type=nn.Linear, **unused_kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = layer_type(dim_in, dim_out)
        self._layer1 = layer_type(dim_in, dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class BlendConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
        **unused_kwargs,
    ):
        super(BlendConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._layer1 = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class RegularizedODEfunc(nn.Module):
    def __init__(self, odefunc, regularization_fns):
        super(RegularizedODEfunc, self).__init__()
        self.odefunc = odefunc
        self.regularization_fns = regularization_fns

    def before_odeint(self, *args, **kwargs):
        self.odefunc.before_odeint(*args, **kwargs)

    def forward(self, t, state):
        with torch.enable_grad():
            x, logp = state[:2]
            x.requires_grad_(True)
            t.requires_grad_(True)
            logp.requires_grad_(True)
            dstate = self.odefunc(t, (x, logp))
            if len(state) > 2:
                dx, dlogp = dstate[:2]
                reg_states = tuple(
                    reg_fn(x, t, logp, dx, dlogp, self.odefunc)
                    for reg_fn in self.regularization_fns
                )
                return dstate + reg_states
            else:
                return dstate

    @property
    def _num_evals(self):
        return self.odefunc._num_evals


class CNF(nn.Module):
    def __init__(
        self,
        odefunc,
        T=1.0,
        train_T=False,
        regularization_fns=None,
        solver="dopri5",
        atol=1e-5,
        rtol=1e-5,
    ):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter(
                "sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T)))
            )
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.test_solver_options = {}

    def forward(
        self, z, logpz=None, reg_states=tuple(), integration_times=None, reverse=False
    ):
        if not len(reg_states) == self.nreg:  # and self.training:
            reg_states = tuple(torch.zeros(z.size(0)).to(z) for i in range(self.nreg))

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor(
                [0.0, self.sqrt_end_time * self.sqrt_end_time]
            ).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        # configure training options
        options = {}
        options.update({"method": self.solver})
        options.update({"h": self.solver_options["step_size"]})
        options.update({"t0": integration_times[0]})
        options.update({"t1": integration_times[1]})
        options.update({"rtol": [self.rtol, self.rtol] + [1e20] * len(reg_states)})
        options.update({"atol": [self.atol, self.atol] + [1e20] * len(reg_states)})
        options.update({"print_neval": False})
        options.update({"neval_max": 1000000})
        options.update({"safety": None})
        options.update({"t_eval": None})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})
        options.update({"print_time": False})

        if self.training:
            if self.solver in [
                "sym12async",
                "adalf",
                "fixedstep_sym12async",
                "fixedstep_adalf",
            ]:
                initial = (z, _logpz) + reg_states
                out = odesolve(self.odefunc, initial, options=options)
                state_t = []
                for _out1, _out2 in zip(initial, out):
                    state_t.append(torch.stack((_out1, _out2), 0))
                state_t = tuple(state_t)
            else:
                state_t = odeint(
                    self.odefunc,
                    (z, _logpz) + reg_states,
                    integration_times.to(z),
                    atol=self.atol + 1e20 * len(reg_states),
                    rtol=self.rtol + 1e20 * len(reg_states),
                    method=self.solver,
                    options=self.solver_options,
                )
        else:
            if self.test_solver in [
                "sym12async",
                "adalf",
                "fixedstep_sym12async",
                "fixedstep_adalf",
            ]:
                initial = (z, _logpz) + reg_states
                out = odesolve(self.odefunc, initial, options=options)
                state_t = []
                for _out1, _out2 in zip(initial, out):
                    state_t.append(torch.stack((_out1, _out2)))
                state_t = tuple(state_t)
            else:
                state_t = odeint(
                    self.odefunc,
                    (z, _logpz),
                    integration_times.to(z),
                    atol=self.test_atol,
                    rtol=self.test_rtol,
                    method=self.test_solver,
                    options=self.test_solver_options,
                )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        reg_states = state_t[2:]

        return z_t, logpz_t, reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()
