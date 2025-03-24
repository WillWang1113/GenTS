"Requirement on pip install TorchDiffEqPack"

from argparse import Namespace
from itertools import chain
import numpy as np
import six
import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F

from src.layers.flow import SequentialFlow
from src.layers.misc import REGULARIZATION_FNS, sample_standard_gaussian
from src.layers.mlp import FinalTanh
from src.layers.nde import NeuralCDE, ODEfunc, ODEnet, CNF
from src.layers.norm import MovingBatchNorm1d
from src.layers.rnn import FullGRUODECell_Autonomous, RNNLayer
from src.model.base import BaseModel


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


# def compute_loss(log_det, base_variables, args):
#     """
#     This function computes the loss of observations with respect to base wiener
#     process.

#     Parameters:
#         log_det: log determinant of transformation 1-D vectors of size
#                  batch_size*length
#         base_variables: Tensor after mapping observations back to the space of
#                         base Wiener process. 2-D tensor of size batch_size*length
#                         x input_shape
#         vars: Difference between consequtive observation time stampes.
#               2-D tensor of size batch_size*length x input_shape
#         masks: Binary tensor showing whether a place is actual observation or
#                padded dummy variable. 1-D binary vectors of size
#                batch_size*length

#     Returns:
#         the step-wise mean of observations' negative log likelihood
#     """
#     mean_martingale = base_variables.clone()
#     mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
#     mean_martingale[:, 0:1] = 0
#     mean_martingale = mean_martingale.view(-1, mean_martingale.shape[2])
#     base_variables = base_variables.view(-1, base_variables.shape[2])
#     # import pdb
#     # pdb.set_trace()
#     vars = torch.ones(base_variables.shape[0], base_variables.shape[-1]).to(base_variables)
#     masks = torch.ones(base_variables.shape[0])
#     # non_zero_idx = masks.nonzero()[:, 0]
#     mean_martingale_masked = mean_martingale#[non_zero_idx]
#     vars_masked = vars#[non_zero_idx]
#     log_det_masked = log_det#[non_zero_idx]
#     base_variables_masked = base_variables#[non_zero_idx]
#     #num_samples = non_zero_idx.shape[0]
#     normal_distri = torch.distributions.Normal(
#         mean_martingale_masked, torch.sqrt(vars_masked)
#     )
#     LL = normal_distri.log_prob(base_variables_masked).view(base_variables_masked.shape[0], -1).sum(1, keepdim=True) - log_det_masked.flatten()
#     return -torch.mean(LL)


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


# def create_separate_batches(data, times, masks):
#     """
#     Separate a batch of data with unequal length into smaller batch of size 1
#     the length of each smaller batch is different and contains no padded dummy
#     variables

#     Parameters:
#        data: observations, a 3-D tensor of shape batchsize x max_length x input_size
#        times: observation time stamps, a 2-D tensor of shape batchsize x max_length
#        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
#               position is observation or padded dummy variables

#     Returns:
#         a list of tuples containing the data, time, masks
#     """
#     batch_size = data.shape[0]
#     data_size = data.shape[-1]
#     ## only repeat the last dimension to concatenate with data
#     repeat_times = tuple([1] * (len(data.shape) - 1) + [data_size])
#     separate_batches = []
#     for i in range(batch_size):
#         length = int(torch.sum(masks[i]))
#         data_item = data[i: i + 1, :length]
#         time_item = times[i, :length].squeeze(-1)
#         mask_item = masks[i: i + 1, :length].unsqueeze(-1).repeat(*repeat_times)
#         separate_batches.append((torch.cat([data_item, mask_item], -1), time_item))
#     return separate_batches


class GTGAN(BaseModel):
    def __init__(
        self,
        seq_len,
        seq_dim,
        hidden_size,
        num_layers_r=2,
        num_layers_d=2,
        num_layers_mlp=3,
        x_hidden=48,
        last_activation_r="sigmoid",
        last_activation_d="sigmoid",
        solver="sym12async",
        atol=1e-3,
        rtol=1e-3,
        time_length=1.0,
        train_T=True,
        nonlinearity="softplus",
        step_size=0.1,
        first_step=0.16667,
        divergence_fn="approximate",
        residual=False,
        rademacher=True,
        layer_type="concat",
        reconstruction=0.01,
        kinetic_energy=0.05,
        jacobian_norm2=0.1,
        directional_penalty=0.01,
        total_deriv=None,
        activation="exp",
        num_iwae_samples=1,
        num_blocks=1,
        batch_norm=False,
        bn_lag=0.0,
        cnf_hidden_dims=(32, 64, 64, 32),
        test_solver=None,
        test_atol=0.1,
        test_rtol=0.1,
        test_step_size=None,
        test_first_step=None,
        gamma=1.0,
        log_time=2,
        lr={"ER": 1e-3, "G": 1e-3, "D": 1e-3},
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.args = Namespace(
            effective_shape=hidden_size, dims=cnf_hidden_dims, **self.hparams_initial
        )
        ode_func = FinalTanh(seq_dim, hidden_size, hidden_size, num_layers_mlp)
        self.embedder = NeuralCDE(
            func=ode_func,
            input_channels=seq_dim,
            hidden_channels=hidden_size,
            output_channels=hidden_size,
        )
        self.recovery = Multi_Layer_ODENetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=seq_dim,
            gru_input_size=hidden_size,
            x_hidden=x_hidden,
            num_layer=num_layers_r,
            last_activation=last_activation_r,
            delta_t=0.5,
        )
        regularization_fns, regularization_coeffs = create_regularization_fns(self.args)

        self.generator = build_model_tabular_nonlinear(
            self.args, self.args.effective_shape, regularization_fns=regularization_fns
        )
        self.supervisor = nn.Sequential(
            RNNLayer(hidden_size, hidden_size, hidden_size, num_layers=num_layers_r),
            nn.Sigmoid(),
        )
        self.discriminator = Multi_Layer_ODENetwork(
            input_size=seq_dim,
            hidden_size=hidden_size,
            output_size=1,
            gru_input_size=hidden_size,
            x_hidden=x_hidden,
            last_activation=last_activation_d,
            num_layer=num_layers_d,
            delta_t=0.5,
        )
        self.gamma = gamma

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        # for n, p in self.named_parameters():
        # print(n, p.device)
        # batch = dataset[dataset_size]
        # x = batch['data'].to(device)
        # train_coeffs = batch['inter']#.to(device)
        # original_x = batch['original_data'].to(device)
        assert kwargs.get("t", None) is not None
        assert kwargs.get("t").shape[0] == n_sample
        obs = kwargs.get("t")
        # x = x[:, :, :-1]
        z = torch.randn(n_sample, self.hparams.seq_len, self.args.effective_shape).to(
            self.device
        )
        time = torch.FloatTensor(list(range(self.hparams.seq_len))).to(self.device)

        # final_index = (torch.ones(n_sample) * self.hparams.seq_len - 1).to(self.device)

        ###########################################
        # time = torch.FloatTensor(list(range(24))).cuda()
        times = time
        times = times.unsqueeze(0)
        times = times.unsqueeze(2)
        times = times.repeat(obs.shape[0], 1, 1)
        h_hat = run_latent_ctfp_model5_2(
            self.args, self.generator, z, times, self.device, z=True
        )
        # print(obs.device)
        # print(h_hat.device)
        # print(next(self.recovery.parameters()).device)
        x_hat = self.recovery(h_hat, obs)
        ###########################################
        # h_hat = run_latent_ctfp_model5_2(self.args, self.generator, z, times, self.device, z=True)
        ###################################
        # x_hat = self.recovery(h_hat, obs)
        return x_hat

    def configure_optimizers(self):
        optimizer_er = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.hparams.lr["ER"],
        )
        optimizer_gs = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.lr["G"]
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.lr["D"]
        )
        return [optimizer_er, optimizer_gs, optimizer_d], []

    def training_step(self, batch, batch_idx):
        max_steps = self.trainer.max_epochs
        x = batch["seq"]
        cond = batch.get("c", None)
        if cond is not None:
            x = x.masked_fill(cond.bool(), float("nan"))
        batch_size = x.shape[0]
        t = batch["t"]
        time = torch.arange(x.shape[1]).to(x)
        times = time.unsqueeze(0)
        times = times.unsqueeze(2)
        times = times.repeat(batch_size, 1, 1)
        final_index = (torch.ones(batch_size) * x.shape[1] - 1).to(self.device)

        optimizer_er, optimizer_gs, optimizer_d = self.optimizers()

        if (self.current_epoch >= 0) and (self.current_epoch < int(1 / 2 * max_steps)):
            self.toggle_optimizer(optimizer_er)
            # x = batch['data'].to(device)
            train_coeffs = batch["coeffs"]
            # original_x = batch['original_data'].to(device)
            obs = t
            # x = x[:, :, :-1]
            # time = torch.arange(x.shape[1]).to(x)
            # final_index = (torch.ones(batch_size) * x.shape[1] - 1).to(self.device)
            h = self.embedder(time, train_coeffs, final_index)
            x_tilde = self.recovery(h, obs)
            x_no_nan = x[~torch.isnan(x)]
            x_tilde_no_nan = x_tilde[~torch.isnan(x)]
            loss_e_t0 = self._loss_e_t0(x_tilde_no_nan, x_no_nan)
            loss_e_0 = self._loss_e_0(loss_e_t0)
            optimizer_er.zero_grad()
            self.manual_backward(loss_e_0)
            # loss_e_0.backward()
            optimizer_er.step()
            self.untoggle_optimizer(optimizer_er)
            self.log("loss_e_0", loss_e_0)
        else:
            for _ in range(2):
                # self.generator.train()
                # self.supervisor.train()
                # self.recovery.train()
                # self.toggle_optimizer(optimizer_d)

                # batch = dataset[batch_size]
                # x = batch['data'].to(device)
                train_coeffs = batch["coeffs"]
                # original_x = batch['original_data'].to(device)
                obs = t
                # x = x[:, :, :-1]
                z = torch.randn(batch_size, x.size(1), self.args.effective_shape).to(x)
                # time = torch.FloatTensor(list(range(24))).cuda()
                # final_index = (torch.ones(batch_size) * 23).cuda()
                h = self.embedder(time, train_coeffs, final_index)
                # times = time
                # times = times.unsqueeze(0)
                # times = times.unsqueeze(2)
                # times = times.repeat(obs.shape[0], 1, 1)
                h_hat = run_latent_ctfp_model5_2(
                    self.args, self.generator, z, times, self.device, z=True
                )
                x_real = self.recovery(h, obs)
                x_fake = self.recovery(h_hat, obs)
                y_fake = self.discriminator(x_fake, obs)
                y_real = self.discriminator(x_real, obs)
                loss_d = self._loss_d2(y_real, y_fake)

                if loss_d.item() > 0.15:
                    optimizer_d.zero_grad()
                    self.manual_backward(loss_d)
                    # loss_d.backward()
                    optimizer_d.step()
                    # torch.cuda.empty_cache()
                self.log("loss_d", loss_d)
                # self.untoggle_optimizer(optimizer_d)

                self.toggle_optimizer(optimizer_er)
                #############Recovery######################
                h = self.embedder(time, train_coeffs, final_index)
                x_tilde = self.recovery(h, obs)

                x_no_nan = x[~torch.isnan(x)]
                x_tilde_no_nan = x_tilde[~torch.isnan(x)]
                loss_e_t0 = self._loss_e_t0(x_tilde_no_nan, x_no_nan)

                loss_e_0 = self._loss_e_0(loss_e_t0)
                loss_e = loss_e_0
                optimizer_er.zero_grad()
                self.manual_backward(loss_e)
                # loss_e.backward()
                optimizer_er.step()
                self.log("loss_e", loss_e)

                # torch.cuda.empty_cache()
                self.untoggle_optimizer(optimizer_er)

            self.toggle_optimizer(optimizer_gs)
            if self.global_step % self.args.log_time == 0:
                # batch = dataset[batch_size]
                # x = batch['data'].to(device)
                train_coeffs = batch["coeffs"]  # .to(device)
                # original_x = batch['original_data'].to(device)
                obs = t
                # x = x[:, :, :-1]
                # time = torch.FloatTensor(list(range(24))).cuda()
                # final_index = (torch.ones(batch_size) * 23).cuda()

                h = self.embedder(time, train_coeffs, final_index)
                # times = time
                # times = times.unsqueeze(0)
                # times = times.unsqueeze(2)
                # times = times.repeat(obs.shape[0], 1, 1)
                #################################################
                if self.args.kinetic_energy is None:
                    loss_s, loss = run_latent_ctfp_model5_2(
                        self.args, self.generator, h, times, self.device, z=False
                    )
                    optimizer_gs.zero_grad()
                    self.manual_backward(loss_s)
                    # loss_s.backward()
                else:
                    loss_s, loss, reg_state = run_latent_ctfp_model5_2(
                        self.args, self.generator, h, times, self.device, z=False
                    )
                    optimizer_gs.zero_grad()
                    self.manual_backward(loss_s + reg_state)
                    # (loss_s+reg_state).backward()
                optimizer_gs.step()
                self.log("loss_s", loss_s)

            # batch = dataset[batch_size]
            # x = batch['data'].to(device)
            train_coeffs = batch["coeffs"]  # .to(device)
            # original_x = batch['original_data'].to(device)
            obs = t
            # x = x[:, :, :-1]
            # time = torch.FloatTensor(list(range(24))).cuda()
            # final_index = (torch.ones(batch_size) * 23).cuda()
            z = torch.randn(batch_size, x.size(1), self.args.effective_shape).to(
                self.device
            )
            h = self.embedder(time, train_coeffs, final_index)
            # times = time.unsqueeze(0)
            # times = times.unsqueeze(2)
            # times = times.repeat(obs.shape[0], 1, 1)
            h_hat = run_latent_ctfp_model5_2(
                self.args, self.generator, z, times, self.device, z=True
            )

            x_hat = self.recovery(h_hat, obs)

            x_no_nan = x[~torch.isnan(x)]
            x_hat_no_nan = x_hat[~torch.isnan(x)]

            y_fake = self.discriminator(x_hat, obs)
            loss_g_u = self._loss_g_u(y_fake)
            loss_g_v = self._loss_g_v(x_no_nan, x_hat_no_nan)
            loss_g = self._loss_g3(loss_g_u, loss_g_v)
            optimizer_gs.zero_grad()
            # loss_g.backward()
            self.manual_backward(loss_g)
            optimizer_gs.step()
            self.untoggle_optimizer(optimizer_gs)
            self.log("loss_g", loss_g)

        # return super().training_step(*args, **kwargs)

    def _loss_e_t0(self, x_tilde, x):
        return F.mse_loss(x_tilde, x)

    def _loss_e_0(self, loss_e_t0):
        return torch.sqrt(loss_e_t0) * 10

    def _loss_d2(self, y_real, y_fake):
        loss_d_real = F.binary_cross_entropy_with_logits(
            y_real, torch.ones_like(y_real)
        )
        loss_d_fake = F.binary_cross_entropy_with_logits(
            y_fake, torch.zeros_like(y_fake)
        )
        return loss_d_real + loss_d_fake

    def _loss_g_u(self, y_fake):
        return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

    def _loss_g_u_e(self, y_fake_e):
        return F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    def _loss_g_v(self, x_hat, x):
        loss_g_v1 = torch.mean(
            torch.abs(
                torch.sqrt(torch.var(x_hat, 0) + 1e-6)
                - torch.sqrt(torch.var(x, 0) + 1e-6)
            )
        )
        loss_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
        return loss_g_v1 + loss_g_v2

    def _loss_g(self, loss_g_u, loss_g_u_e, loss_s, loss_g_v):
        return (
            loss_g_u
            + self.gamma * loss_g_u_e
            + 100 * torch.sqrt(loss_s)
            + 100 * loss_g_v
        )

    def _loss_g2(self, loss_g_u, loss_s, loss_g_v):
        return loss_g_u + loss_s + 100 * loss_g_v

    def _loss_g3(self, loss_g_u, loss_g_v):
        return loss_g_u + 100 * loss_g_v
