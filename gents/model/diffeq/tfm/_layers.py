# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Union, Iterable, Generator, Dict

from torchdyn.core.problems import ODEProblem
from torchdyn.numerics import odeint
from torchdyn.core.utils import standardize_vf_call_signature

from lightning import LightningModule
import torch
from torch import Tensor
import torch.nn as nn


from torchdyn.numerics.sensitivity import (
    _gather_odefunc_adjoint,
    _gather_odefunc_interp_adjoint,
)
from torchdyn.numerics.odeint import odeint
from torchdyn.numerics.solvers.ode import str_to_solver


class ODEProblem(nn.Module):
    def __init__(
        self,
        vector_field: Union[Callable, nn.Module],
        solver: Union[str, nn.Module],
        interpolator: Union[str, Callable, None] = None,
        order: int = 1,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        sensitivity: str = "autograd",
        solver_adjoint: Union[str, nn.Module, None] = None,
        atol_adjoint: float = 1e-6,
        rtol_adjoint: float = 1e-6,
        seminorm: bool = False,
        integral_loss: Union[Callable, None] = None,
        optimizable_params: Union[Iterable, Generator] = (),
    ):
        """An ODE Problem coupling a given vector field with solver and sensitivity algorithm to compute gradients w.r.t different quantities.

        Args:
            vector_field ([Callable]): the vector field, called with `vector_field(t, x)` for `vector_field(x)`.
                                       In the second case, the Callable is automatically wrapped for consistency
            solver (Union[str, nn.Module]):
            order (int, optional): Order of the ODE. Defaults to 1.
            atol (float, optional): Absolute tolerance of the solver. Defaults to 1e-4.
            rtol (float, optional): Relative tolerance of the solver. Defaults to 1e-4.
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): ODE solver for the adjoint. Defaults to None.
            atol_adjoint (float, optional): Defaults to 1e-6.
            rtol_adjoint (float, optional): Defaults to 1e-6.
            seminorm (bool, optional): Indicates whether the a seminorm should be used for error estimation during adjoint backsolves. Defaults to False.
            integral_loss (Union[Callable, None]): Integral loss to optimize for. Defaults to None.
            optimizable_parameters (Union[Iterable, Generator]): parameters to calculate sensitivies for. Defaults to ().
        Notes:
            Integral losses can be passed as generic function or `nn.Modules`.
        """
        super().__init__()
        # instantiate solver at initialization
        if type(solver) == str:
            solver = str_to_solver(solver)
        if solver_adjoint is None:
            solver_adjoint = solver
        else:
            solver_adjoint = str_to_solver(solver_adjoint)

        self.solver, self.interpolator, self.atol, self.rtol = (
            solver,
            interpolator,
            atol,
            rtol,
        )
        self.solver_adjoint, self.atol_adjoint, self.rtol_adjoint = (
            solver_adjoint,
            atol_adjoint,
            rtol_adjoint,
        )
        self.sensitivity, self.integral_loss = sensitivity, integral_loss

        # wrap vector field if `t, x` is not the call signature
        vector_field = standardize_vf_call_signature(vector_field)

        self.vf, self.order, self.sensalg = vector_field, order, sensitivity
        optimizable_params = tuple(optimizable_params)

        if len(tuple(self.vf.parameters())) > 0:
            self.vf_params = torch.cat(
                [p.contiguous().flatten() for p in self.vf.parameters()]
            )

        elif len(optimizable_params) > 0:
            # use `optimizable_parameters` if f itself does not have a .parameters() iterable
            # TODO: advanced logic to retain naming in case `state_dicts()` are passed
            for k, p in enumerate(optimizable_params):
                self.vf.register_parameter(f"optimizable_parameter_{k}", p)
            self.vf_params = torch.cat(
                [p.contiguous().flatten() for p in optimizable_params]
            )

        else:
            print("Your vector field does not have `nn.Parameters` to optimize.")
            dummy_parameter = nn.Parameter(torch.zeros(1))
            self.vf.register_parameter("dummy_parameter", dummy_parameter)
            self.vf_params = torch.cat(
                [p.contiguous().flatten() for p in self.vf.parameters()]
            )

    def _autograd_func(self):
        "create autograd functions for backward pass"
        self.vf_params = torch.cat(
            [p.contiguous().flatten() for p in self.vf.parameters()]
        )
        if (
            self.sensalg == "adjoint"
        ):  # alias .apply as direct call to preserve consistency of call signature
            return _gather_odefunc_adjoint(
                self.vf,
                self.vf_params,
                self.solver,
                self.atol,
                self.rtol,
                self.interpolator,
                self.solver_adjoint,
                self.atol_adjoint,
                self.rtol_adjoint,
                self.integral_loss,
                problem_type="standard",
            ).apply
        elif self.sensalg == "interpolated_adjoint":
            return _gather_odefunc_interp_adjoint(
                self.vf,
                self.vf_params,
                self.solver,
                self.atol,
                self.rtol,
                self.interpolator,
                self.solver_adjoint,
                self.atol_adjoint,
                self.rtol_adjoint,
                self.integral_loss,
                problem_type="standard",
            ).apply

    def odeint(self, x: Tensor, t_span: Tensor, save_at: Tensor = (), args={}):
        "Returns Tuple(`t_eval`, `solution`)"
        if self.sensalg == "autograd":
            return odeint(
                self.vf,
                x,
                t_span,
                self.solver,
                self.atol,
                self.rtol,
                interpolator=self.interpolator,
                save_at=save_at,
                args=args,
            )
        else:
            return self._autograd_func()(self.vf_params, x, t_span, save_at, args)

    def forward(self, x: Tensor, t_span: Tensor, save_at: Tensor = (), args={}):
        "For safety redirects to intended method `odeint`"
        return self.odeint(x, t_span, save_at, args)



class NeuralODE(ODEProblem, LightningModule):
    def __init__(
        self,
        vector_field: Union[Callable, nn.Module],
        solver: Union[str, nn.Module] = "tsit5",
        order: int = 1,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        sensitivity="autograd",
        solver_adjoint: Union[str, nn.Module, None] = None,
        atol_adjoint: float = 1e-4,
        rtol_adjoint: float = 1e-4,
        interpolator: Union[str, Callable, None] = None,
        integral_loss: Union[Callable, None] = None,
        seminorm: bool = False,
        return_t_eval: bool = True,
        optimizable_params: Union[Iterable, Generator] = (),
    ):
        """Generic Neural Ordinary Differential Equation.

        Args:
            vector_field ([Callable]): the vector field, called with `vector_field(t, x)` for `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped for consistency
            solver (Union[str, nn.Module]): 
            order (int, optional): Order of the ODE. Defaults to 1.
            atol (float, optional): Absolute tolerance of the solver. Defaults to 1e-4.
            rtol (float, optional): Relative tolerance of the solver. Defaults to 1e-4.
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): ODE solver for the adjoint. Defaults to None.
            atol_adjoint (float, optional): Defaults to 1e-6.
            rtol_adjoint (float, optional): Defaults to 1e-6.
            integral_loss (Union[Callable, None], optional): Defaults to None.
            seminorm (bool, optional): Whether to use seminorms for adaptive stepping in backsolve adjoints. Defaults to False.
            return_t_eval (bool): Whether to return (t_eval, sol) or only sol. Useful for chaining NeuralODEs in `nn.Sequential`.
            optimizable_parameters (Union[Iterable, Generator]): parameters to calculate sensitivies for. Defaults to ().
        Notes:
            In `torchdyn`-style, forward calls to a Neural ODE return both a tensor `t_eval` of time points at which the solution is evaluated
            as well as the solution itself. This behavior can be controlled by setting `return_t_eval` to False. Calling `trajectory` also returns
            the solution only. 

            The Neural ODE class automates certain delicate steps that must be done depending on the solver and model used. 
            The `prep_odeint` method carries out such steps. Neural ODEs wrap `ODEProblem`.
        """
        super().__init__(
            vector_field=standardize_vf_call_signature(
                vector_field, order, defunc_wrap=True
            ),
            order=order,
            sensitivity=sensitivity,
            solver=solver,
            atol=atol,
            rtol=rtol,
            solver_adjoint=solver_adjoint,
            atol_adjoint=atol_adjoint,
            rtol_adjoint=rtol_adjoint,
            seminorm=seminorm,
            interpolator=interpolator,
            integral_loss=integral_loss,
            optimizable_params=optimizable_params,
        )
        # data-control conditioning
        self._control, self.controlled, self.t_span = None, False, None
        self.return_t_eval = return_t_eval
        if integral_loss is not None:
            self.vf.integral_loss = integral_loss
        self.vf.sensitivity = sensitivity

    def _prep_integration(self, x: Tensor, t_span: Tensor) -> Tensor:
        "Performs generic checks before integration. Assigns data control inputs and augments state for CNFs"

        # assign a basic value to `t_span` for `forward` calls that do no explicitly pass an integration interval
        if t_span is None and self.t_span is None:
            t_span = torch.linspace(0, 1, 2)
        elif t_span is None:
            t_span = self.t_span

        # loss dimension detection routine; for CNF div propagation and integral losses w/ autograd
        excess_dims = 0
        if (self.integral_loss is not None) and self.sensitivity == "autograd":
            excess_dims += 1

        # handle aux. operations required for some jacobian trace CNF estimators e.g Hutchinson's
        # as well as datasets-control set to DataControl module
        for _, module in self.vf.named_modules():
            if hasattr(module, "trace_estimator"):
                if module.noise_dist is not None:
                    module.noise = module.noise_dist.sample((x.shape[0],))
                excess_dims += 1

            # data-control set routine. Is performed once at the beginning of odeint since the control is fixed to IC
            if hasattr(module, "_control"):
                self.controlled = True
                module._control = x[:, excess_dims:].detach()
        return x, t_span

    def forward(
        self,
        x: Union[Tensor, Dict],
        t_span: Tensor = None,
        save_at: Iterable = (),
        args={},
    ):
        x, t_span = self._prep_integration(x, t_span)
        t_eval, sol = super().forward(x, t_span, save_at, args)
        if self.return_t_eval:
            return t_eval, sol
        else:
            return sol

    def trajectory(self, x: torch.Tensor, t_span: Tensor):
        x, t_span = self._prep_integration(x, t_span)
        _, sol = odeint(
            self.vf, x, t_span, solver=self.solver, atol=self.atol, rtol=self.rtol
        )
        return sol

    def __repr__(self):
        npar = sum([p.numel() for p in self.vf.parameters()])
        return f"Neural ODE:\n\t- order: {self.order}\
        \n\t- solver: {self.solver}\n\t- adjoint solver: {self.solver_adjoint}\
        \n\t- tolerances: relative {self.rtol} absolute {self.atol}\
        \n\t- adjoint tolerances: relative {self.rtol_adjoint} absolute {self.atol_adjoint}\
        \n\t- num_parameters: {npar}\
        \n\t- NFE: {self.vf.nfe}"
