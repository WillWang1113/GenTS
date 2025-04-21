"Requirement on pip install TorchDiffEqPack"

import numpy as np
import six
import torch
import torch.nn.functional as F

import torch.nn as nn
import copy
from torchdiffeq import odeint_adjoint as odeint
from TorchDiffEqPack import odesolve_adjoint_sym12 as odesolve

from ._utils import (
    REGULARIZATION_FNS,
    divergence_approx,
    divergence_bf,
    sample_gaussian_like,
    sample_rademacher_like,
    sample_standard_gaussian,
    _flip,
    NONLINEARITIES,
    squeeze,
    unsqueeze,
)
from ._layers import (
    IgnoreConv2d,
    IgnoreLinear,
    HyperConv2d,
    HyperLinear,
    ConcatConv2d,
    ConcatCoordConv2d,
    ConcatLinear,
    ConcatLinear_v2,
    BlendConv2d,
    SquashLinear,
    BlendLinear,
    ConcatConv2d_v2,
    SquashConv2d,
    ConcatSquashConv2d,
    ConcatSquashLinear, RegularizedODEfunc, MovingBatchNorm1d
)


class FullGRUODECell_Autonomous(nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        # self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        # self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        # self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        # self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """
        # xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        for i in inds:
            x, logpx, reg_states = self.chain[i](x, logpx, reg_states, reverse=reverse)
        return x, logpx, reg_states


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


class RecoveryODENetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        gru_input_size,
        x_hidden,
        delta_t,
        last_activation="identity",
        solver="euler",
    ):
        """24 24 6 24 48
        Arguments:
            input_size: input shape
            hidden_size: shape of hidden state of GRUODE and GRU
            output_size: output shape
            gru_input_size: input size of GRU (raw input will pass through x_model which change shape input_size to gru_input_size)
            x_hidden: shape going through x_model
            delta_t: integration time step for fixed integrator
            solver: ['euler','midpoint','dopri5']
        """
        super(RecoveryODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        self.x_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.x_hidden, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_out),
            torch.nn.Linear(self.x_hidden, self.gru_input_size, bias=self.bias),
        )
        self.gru_layer = FullGRUODECell_Autonomous(self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size
        )

        if last_activation == "identity":
            self.last_layer = None
        elif last_activation == "softplus":
            self.last_layer = torch.nn.Softplus()
        elif last_activation == "tanh":
            self.last_layer = torch.nn.Tanh()
        elif last_activation == "sigmoid":
            self.last_layer = torch.nn.Sigmoid()
        self.rec_linear = torch.nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert not self.impute
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t])
            )
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        HH = self.x_model(H)
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to(times.device)
        current_time = times[0, 0] - 1
        # final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            while current_time < (obs_time - 0.001 * self.delta_t):
                if self.solver == "dopri5":
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time - current_time, current_time
                    )
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time
                    )
            current_out, tmp = self.gru_obs(
                torch.reshape(HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])),
                h[None, :, :],
            )
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + current_out.reshape(
                HH.shape[0], HH.shape[-1]
            )
        X_tilde = self.rec_linear(out)
        if self.last_layer is not None:
            X_tilde = self.last_layer(X_tilde)
        return X_tilde


class First_ODENetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        gru_input_size,
        x_hidden,
        delta_t,
        solver="euler",
    ):
        super(First_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        self.x_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.x_hidden, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_out),
            torch.nn.Linear(self.x_hidden, self.gru_input_size, bias=self.bias),
        )
        self.gru_layer = FullGRUODECell_Autonomous(self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size
        )
        # self.rec_linear = torch.nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert not self.impute
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t])
            )
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        HH = self.x_model(H)
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to(times.device)
        current_time = times[0, 0] - 1
        # final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            while current_time < (obs_time - 0.001 * self.delta_t):
                if self.solver == "dopri5":
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time - current_time, current_time
                    )
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time
                    )
            current_out, tmp = self.gru_obs(
                torch.reshape(HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])),
                h[None, :, :],
            )
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + current_out.reshape(
                HH.shape[0], HH.shape[-1]
            )
        # X_tilde = self.rec_linear(out)
        # return X_tilde
        return out


class Mid_ODENetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        gru_input_size,
        x_hidden,
        delta_t,
        solver="euler",
    ):
        super(Mid_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        # self.x_model = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, self.x_hidden, bias = self.bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p = self.drop_out),
        #     torch.nn.Linear(self.x_hidden, self.gru_input_size, bias = self.bias)
        # )
        self.gru_layer = FullGRUODECell_Autonomous(self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size
        )
        # self.rec_linear = torch.nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert not self.impute
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t])
            )
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        # HH = self.x_model(H)
        HH = H
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to(times.device)
        current_time = times[0, 0] - 1
        # final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            while current_time < (obs_time - 0.001 * self.delta_t):
                if self.solver == "dopri5":
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time - current_time, current_time
                    )
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time
                    )
            current_out, tmp = self.gru_obs(
                torch.reshape(HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])),
                h[None, :, :],
            )
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + current_out.reshape(
                HH.shape[0], HH.shape[-1]
            )
        # X_tilde = self.rec_linear(out)
        # return X_tilde
        return out


class Last_ODENetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        gru_input_size,
        x_hidden,
        delta_t,
        last_activation="identity",
        solver="euler",
    ):
        super(Last_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        # self.x_model = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, self.x_hidden, bias = self.bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p = self.drop_out),
        #     torch.nn.Linear(self.x_hidden, self.gru_input_size, bias = self.bias)
        # )
        self.gru_layer = FullGRUODECell_Autonomous(self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size
        )
        if last_activation == "identity":
            self.last_layer = None
        elif last_activation == "softplus":
            self.last_layer = torch.nn.Softplus()
        elif last_activation == "tanh":
            self.last_layer = torch.nn.Tanh()
        elif last_activation == "sigmoid":
            self.last_layer = torch.nn.Sigmoid()
        self.rec_linear = torch.nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert not self.impute
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t])
            )
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        # HH = self.x_model(H)
        HH = H
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to(times.device)
        current_time = times[0, 0] - 1
        # final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            while current_time < (obs_time - 0.001 * self.delta_t):
                if self.solver == "dopri5":
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time - current_time, current_time
                    )
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time
                    )
            current_out, tmp = self.gru_obs(
                torch.reshape(HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])),
                h[None, :, :],
            )
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + current_out.reshape(
                HH.shape[0], HH.shape[-1]
            )
        X_tilde = self.rec_linear(out)
        if self.last_layer is not None:
            X_tilde = self.last_layer(X_tilde)
        return X_tilde


class Multi_Layer_ODENetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        gru_input_size,
        x_hidden,
        delta_t,
        num_layer,
        last_activation="identity",
        solver="euler",
    ):
        super(Multi_Layer_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True
        self.num_layer = num_layer
        self.last_activation = last_activation

        if num_layer == 1:
            self.model = RecoveryODENetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                gru_input_size=gru_input_size,
                x_hidden=x_hidden,
                last_activation=self.last_activation,
                delta_t=delta_t,
                solver=solver,
            )
        elif num_layer == 2:
            self.model = torch.nn.ModuleList(
                [
                    First_ODENetwork(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        gru_input_size=gru_input_size,
                        x_hidden=x_hidden,
                        delta_t=delta_t,
                        solver=solver,
                    ),
                    Last_ODENetwork(
                        input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        gru_input_size=gru_input_size,
                        x_hidden=x_hidden,
                        last_activation=self.last_activation,
                        delta_t=delta_t,
                        solver=solver,
                    ),
                ]
            )
        else:
            self.model = torch.nn.ModuleList()
            for i in range(num_layer):
                if i == 0:
                    self.model.append(
                        First_ODENetwork(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=hidden_size,
                            gru_input_size=gru_input_size,
                            x_hidden=x_hidden,
                            delta_t=delta_t,
                            solver=solver,
                        )
                    )
                elif i == num_layer - 1:
                    self.model.append(
                        Last_ODENetwork(
                            input_size=hidden_size,
                            hidden_size=hidden_size,
                            output_size=output_size,
                            gru_input_size=gru_input_size,
                            x_hidden=x_hidden,
                            last_activation=self.last_activation,
                            delta_t=delta_t,
                            solver=solver,
                        )
                    )
                else:
                    self.model.append(
                        Mid_ODENetwork(
                            input_size=hidden_size,
                            hidden_size=hidden_size,
                            output_size=hidden_size,
                            gru_input_size=gru_input_size,
                            x_hidden=x_hidden,
                            delta_t=delta_t,
                            solver=solver,
                        )
                    )

    def forward(self, H, times):
        if self.num_layer == 1:
            out = self.model(H, times)
        else:
            out = H
            for model in self.model:
                out = model(out, times)
        return out


def build_model_tabular_nonlinear(args, dims, regularization_fns=None):
    # hidden_dims = tuple(map(int, args.dims.split("-")))
    hidden_dims = args.dims

    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims + 1,),
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )
        return cnf

    chain_block = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [
            MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)
        ]
        bn_chain = [MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain_block, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain_block = bn_chain
    model = SequentialFlow(chain_block)

    set_cnf_options(args, model)

    return model


def set_cnf_options(args, model):
    def _set(module):
        if isinstance(module, CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options["step_size"] = args.step_size
            if args.first_step is not None:
                module.solver_options["first_step"] = args.first_step

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ["fixed_adams", "explicit_adams"]:
                module.solver_options["max_order"] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol
            if args.test_step_size is not None:
                module.test_solver_options["step_size"] = args.test_step_size
            if args.test_first_step is not None:
                module.test_solver_options["first_step"] = args.test_first_step

    model.apply(_set)


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def run_latent_ctfp_model5_2(args, aug_model, values, times, device, z=True):
    """
    Functions for running the latent ctfp model
    Parameters:
        args: arguments returned from parse_arguments
        encoder: ode_rnn model as encoder
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables
        evluation (bool): whether to run the latent ctfp model in the evaluation
                          mode. Return IWAE if set to true. Return both IWAE and
                          training loss if set to false
    Returns:
        Return IWAE if evaluation set to true.
        Return both IWAE and training loss if evaluation set to false.
    """
    """
    if evaluation:
        num_iwae_samples = args.niwae_test
        batch_size = args.test_batch_size
    else:
        num_iwae_samples = args.num_iwae_samples
        batch_size = args.batch_size
    data_batches = create_separate_batches(values, times, masks)
    mean_list, stdv_list = [], []
    # item[0] : 1 46 2 item[1] : 46
    # every iter different seq  -> output is same z_mean (1,1,10), z_stdv(1,1,10)
    for item in data_batches:
        z_mean, z_stdv = encoder(item[0], item[1])
        mean_list.append(z_mean)
        stdv_list.append(z_stdv)
    pdb.set_trace()
    means = torch.cat(mean_list, dim=1)
    stdvs = torch.cat(stdv_list, dim=1)
    # Sample latent variables means.shape = 3
    repeat_times = [1] * len(means.shape)
    repeat_times[0] = num_iwae_samples
    means = means.repeat(*repeat_times)
    stdvs = stdvs.repeat(*repeat_times)
    # mean, stdvs : 3 50 10 -> latent 3 50 10 
    """
    if z:
        mu = torch.zeros(1, values.shape[0], values.shape[2]).to(device)
        stdvs = torch.ones(1, values.shape[0], values.shape[2]).to(device)
        latent = sample_standard_gaussian(mu, stdvs)
        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        max_length = times.shape[1]
        # 150 89 10
        latent_sequence = latent_sequence.repeat(1, max_length, 1)
        # aux = torch.cat([latent_sequence, times], dim=2)
        aux = torch.cat([latent_sequence, times], dim=2)
        # aux = latent_sequence
        # 13350, 12
        aux = aux.view(-1, aux.shape[2])
        # 13350, 12 , torch.zeros(aux.shape[0], 1) : 13350, 12  => 13350 12
        # import pdb;pdb.set_trace()
        aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        # aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, : -times.shape[2]]
        aux = aux.view(values.shape[0], -1, values.shape[2])
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89
            aux, _ = log_jaco(aux, reverse=True)
        elif args.activation == "softplus":
            aux = inversoft_jaco(aux, reverse=True)
        elif args.activation == "identity":
            pass
        else:
            raise NotImplementedError
        return aux
    else:
        max_length = times.shape[1]
        time_to_cat = times.repeat(args.num_iwae_samples, 1, 1)
        values = values.repeat(args.num_iwae_samples, 1, 1)
        aux = torch.cat([torch.zeros_like(values), time_to_cat], dim=2)
        aux = aux.view(-1, aux.shape[2])
        aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, -times.shape[2] :]
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        vars = torch.ones_like(stdvs).squeeze(0)
        masks = torch.ones_like(stdvs).squeeze(0)
        if args.activation == "exp":
            # values 150 89 1 => transform_valeus 150 89 1 transform_lodget 150 89
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError
        # transform + aux = aug _values  13350 1 + 13350 11 => 13350 12
        aug_values = transform_values.view(-1, transform_values.shape[2])
        aug_values = torch.cat([aug_values, aux], dim=1)
        # aug_values = aug_values.view(-1, aug_values.shape[2])
        # input : 13350 12, 13350 1 => base_values: 13350 12 flow_lodget 13350 1
        if args.kinetic_energy is None:
            base_values, flow_logdet, _ = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
        else:
            base_values, flow_logdet, reg_states = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
            )
            reg_states = tuple(torch.mean(rs) for rs in reg_states)
        # base_values -> 150 89 1
        base_values = base_values[:, : -times.shape[2]]
        base_values = base_values.view(values.shape[0], -1, args.effective_shape)
        ## flow_logdet and transform_logdet are both of size length*batch_size x length
        # flow_logdet -> 150 89 transform_logdet -> 150 89
        flow_logdet = flow_logdet.sum(-1).view(base_values.shape[0], -1)
        transform_logdet = transform_logdet.view(base_values.shape[0], -1)
        if len(vars.shape) == 2:
            vars_unsqueed = vars.unsqueeze(-1)
        else:
            vars_unsqueed = vars
        ll = compute_ll(
            flow_logdet + transform_logdet,
            base_values,
            vars_unsqueed.repeat(1, 1, 1),
            masks.repeat(1, 1),
        )
        ll = ll.view(
            args.num_iwae_samples, int((base_values.shape[0] / args.num_iwae_samples))
        )
        weights = ll  # + prior_z - posterior_z
        loss = -torch.logsumexp(weights, 0) + np.log(args.num_iwae_samples)
        loss = torch.sum(loss) / (
            int((base_values.shape[0] / args.num_iwae_samples)) * base_values.shape[1]
        )
        loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
            int((base_values.shape[0] / args.num_iwae_samples)) * (base_values.shape[1])
        )
        if args.kinetic_energy is None:
            return loss, loss_training
        else:
            return loss, loss_training, reg_states[0]


def log_jaco(values, reverse=False):
    """
    compute log transformation and log determinant of jacobian

    Parameters:
        values: tensor to be transformed
        reverse (bool): If reverse is False, given z_1 return z_0 = log(z_1) and
                        log det of d z_1/d z_0. If reverse is True, given z_0
                        return z_1 = exp(z_0) and log det of d z_1/d z_0

    Returns:
        transformed tesnors and log determinant of the transformation
    """
    if not reverse:
        log_values = torch.log(values)
        return log_values, torch.sum(log_values, dim=2)
    else:
        return torch.exp(values), torch.sum(values, dim=2)


def inversoft_jaco(values, reverse=False):
    """
    compute softplus  transformation and log determinant of jacobian

    Parameters:
        values: tensor to be transformed
        reverse (bool): If reverse is False, given z_1 return
                        z_0 = inverse_softplus(z_1) and log det of d z_1/d z_0.
                        If reverse is True, given z_0 return z_1 = softplus(z_0)
                        and log det of d z_1/d z_0

    Returns:
        transformed tesnors and log determinant of the transformation
    """
    if not reverse:
        inverse_values = torch.log(1 - torch.exp(-values)) + values
        log_det = torch.sum(
            inverse_values - torch.nn.functional.softplus(inverse_values), dim=2
        )
        return inverse_values, log_det
    else:
        log_det = torch.sum(values - torch.nn.functional.softplus(values), dim=2)
        return torch.nn.functional.softplus(values)


def compute_ll(log_det, base_variables, vars, masks):
    """
    This function computes the log likelihood of observations with respect to base wiener
    process used for latent_CTFP.

    Parameters:
        log_det: log determinant of transformation 2-D vectors of size
                 batch_size x length
        base_variables: Tensor after mapping observations back to the space of
                        base Wiener process. 3-D tensor of size batch_size x
                        length x input_shape
        vars: Difference between consequtive observation time stampes.
              3-D tensor of size batch_size x length x 1
        masks: Binary tensor showing whether a place is actual observation or
               padded dummy variable. 2-D binary vectors of size
               batch_size x length

    Returns:
        the sum of log likelihood of all observations
    """
    # import pdb;pdb.set_trace()
    mean_martingale = base_variables.clone()
    mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
    mean_martingale[:, 0:1] = 0
    normal_distri = torch.distributions.Normal(mean_martingale, torch.sqrt(vars))
    LL = normal_distri.log_prob(base_variables)
    LL = (torch.sum(LL, -1) - log_det) * masks
    return torch.sum(LL, -1)


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
