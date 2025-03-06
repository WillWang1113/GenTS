import torch
import torchdiffeq
import math
import numpy as np


def cheap_stack(tensors, dim):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)


def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    """Solves a tridiagonal system Ax = b.

    The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
    and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
    of size (k, k), with entries:

    D[0] U[0]
    L[0] D[1] U[1]
         L[1] D[2] U[2]                     0
              L[2] D[3] U[3]
                  .    .    .
                       .      .      .
                           .        .        .
                        L[k - 3] D[k - 2] U[k - 2]
           0                     L[k - 2] D[k - 1] U[k - 1]
                                          L[k - 1]   D[k]

    Arguments:
        b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
        A_upper: A tensor of shape (..., k - 1).
        A_diagonal: A tensor of shape (..., k).
        A_lower: A tensor of shape (..., k - 1).

    Returns:
        A tensor of shape (..., k), corresponding to the x solving Ax = b

    Warning:
        This implementation isn't super fast. You probably want to cache the result, if possible.
    """

    # This implementation is very much written for clarity rather than speed.

    A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)


class NaturalCubicSpline:
    """Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        times = torch.linspace(0, 1, 7)
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        X = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_spline_coeffs(times, X)
        # ...at this point you can save the coeffs, put them through PyTorch's Datasets and DataLoaders, etc...
        spline = NaturalCubicSpline(times, coeffs)
        t = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(t)
    """

    def __init__(self, times, coeffs, **kwargs):
        """
        Arguments:
            times: As was passed as an argument to natural_cubic_spline_coeffs.
            coeffs: As returned by natural_cubic_spline_coeffs.
        """
        super(NaturalCubicSpline, self).__init__(**kwargs)

        a, b, two_c, three_d = coeffs

        self._times = times
        self._a = a
        self._b = b
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        self._two_c = two_c
        self._three_d = three_d

    def _interpret_t(self, t):
        maxlen = self._b.size(-2) - 1
        index = (t > self._times).sum() - 1
        index = index.clamp(
            0, maxlen
        )  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._times; this is correct behaviour
        fractional_part = t - self._times[index]
        return fractional_part, index

    def evaluate(self, t):
        """Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = (
            0.5 * self._two_c[..., index, :]
            + self._three_d[..., index, :] * fractional_part / 3
        )
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        """Evaluates the derivative of the natural cubic spline at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = (
            self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        )
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv


class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(z)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out


def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """

    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError(
            "dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
            "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
            "dimensions)."
            "".format(
                tuple(control_gradient.shape),
                tuple(control_gradient.shape[:-1]),
                tuple(z0.shape),
                tuple(z0.shape[:-1]),
            )
        )
    vector_field = func(z0)
    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError(
            "func did not return a tensor with the same number of batch dimensions as z0. func returned "
            "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
            " dimensions)."
            "".format(
                tuple(vector_field.shape),
                tuple(vector_field.shape[:-2]),
                tuple(z0.shape),
                tuple(z0.shape[:-1]),
            )
        )
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError(
            "func did not return a tensor with the same number of hidden channels as z0. func returned "
            "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
            "".format(
                tuple(vector_field.shape),
                vector_field.size(-2),
                tuple(z0.shape),
                z0.shape.size(-1),
            )
        )
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError(
            "func did not return a tensor with the same number of input channels as dX_dt returned. "
            "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
            " channels)."
            "".format(
                tuple(vector_field.shape),
                vector_field.size(-1),
                tuple(control_gradient.shape),
                control_gradient.size(-1),
            )
        )
    if control_gradient.requires_grad and adjoint:
        raise ValueError(
            "Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
            "of the underlying torchdiffeq library.)"
        )

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    return out


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

    def forward(self, z):
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
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, (
                "coeff.shape[:-2] must be the same as final_index.shape. "
                "coeff.shape[:-2]={}, final_index.shape={}"
                "".format(batch_dims, final_index.shape)
            )

        cubic_spline = NaturalCubicSpline(times, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(
                    *batch_dims,
                    self.hidden_channels,
                    dtype=coeff.dtype,
                    device=coeff.device,
                )
            else:
                z0 = self.initial_network(cubic_spline.evaluate(times[0]))
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
        z_t = cdeint(
            dX_dt=cubic_spline.derivative, z0=z0, func=self.func, t=t, **kwargs
        )

        # Organise the output

        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
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


def _natural_cubic_spline_coeffs_without_missing_values(times, path):
    # path should be a tensor of shape (..., length)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    length = path.size(-1)

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = path[..., :1]
        b = (path[..., 1:] - path[..., :1]) / (times[..., 1:] - times[..., :1])
        two_c = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
        three_d = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
    else:
        # Set up some intermediate values
        time_diffs = times[1:] - times[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal**2
        three_path_diffs = 3 * (path[..., 1:] - path[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty(length, dtype=path.dtype, device=path.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(path)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(
            system_rhs, time_diffs_reciprocal, system_diagonal, time_diffs_reciprocal
        )

        # Do some algebra to find the coefficients of the spline
        a = path[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (
            six_path_diffs * time_diffs_reciprocal
            - 4 * knot_derivatives[..., :-1]
            - 2 * knot_derivatives[..., 1:]
        ) * time_diffs_reciprocal
        three_d = (
            -six_path_diffs * time_diffs_reciprocal
            + 3 * (knot_derivatives[..., :-1] + knot_derivatives[..., 1:])
        ) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d


def _natural_cubic_spline_coeffs_with_missing_values(t, path):
    if len(path.shape) == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _natural_cubic_spline_coeffs_with_missing_values_scalar(t, path)
    else:
        a_pieces = []
        b_pieces = []
        two_c_pieces = []
        three_d_pieces = []
        for p in path.unbind(dim=0):  # TODO: parallelise over this
            a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(
                t, p
            )
            a_pieces.append(a)
            b_pieces.append(b)
            two_c_pieces.append(two_c)
            three_d_pieces.append(three_d)
        return (
            cheap_stack(a_pieces, dim=0),
            cheap_stack(b_pieces, dim=0),
            cheap_stack(two_c_pieces, dim=0),
            cheap_stack(three_d_pieces, dim=0),
        )


def _natural_cubic_spline_coeffs_with_missing_values_scalar(times, path):
    # times and path both have shape (length,)

    # How to deal with missing values at the start or end of the time series? We're creating some splines, so one
    # option is just to extend the first piece backwards, and the final piece forwards. But polynomials tend to
    # behave badly when extended beyond the interval they were constructed on, so the results can easily end up
    # being awful.
    # Instead we impute an observation at the very start equal to the first actual observation made, and impute an
    # observation at the very end equal to the last actual observation made, and then procede with splines as
    # normal.

    not_nan = ~torch.isnan(path)
    path_no_nan = path.masked_select(not_nan)

    if path_no_nan.size(0) == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        # Note that we may assume that path.size(0) >= 2 by the checks in __init__ so "path.size(0) - 1" is a valid
        # thing to do.
        return (
            torch.zeros(path.size(0) - 1, dtype=path.dtype, device=path.device),
            torch.zeros(path.size(0) - 1, dtype=path.dtype, device=path.device),
            torch.zeros(path.size(0) - 1, dtype=path.dtype, device=path.device),
            torch.zeros(path.size(0) - 1, dtype=path.dtype, device=path.device),
        )
    # else we have at least one non-NaN entry, in which case we're going to impute at least one more entry (as
    # the path is of length at least 2 so the start and the end aren't the same), so we will then have at least two
    # non-Nan entries. In particular we can call _compute_coeffs safely later.

    need_new_not_nan = False
    if torch.isnan(path[0]):
        if not need_new_not_nan:
            path = path.clone()
            need_new_not_nan = True
        path[0] = path_no_nan[0]
    if torch.isnan(path[-1]):
        if not need_new_not_nan:
            path = path.clone()
            need_new_not_nan = True
        path[-1] = path_no_nan[-1]
    if need_new_not_nan:
        not_nan = ~torch.isnan(path)
        path_no_nan = path.masked_select(not_nan)
    times_no_nan = times.masked_select(not_nan)

    # Find the coefficients on the pieces we do understand
    # These all have shape (len - 1,)
    (a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan) = (
        _natural_cubic_spline_coeffs_without_missing_values(times_no_nan, path_no_nan)
    )

    # Now we're going to normalise them to give coefficients on every interval
    a_pieces = []
    b_pieces = []
    two_c_pieces = []
    three_d_pieces = []

    iter_times_no_nan = iter(times_no_nan)
    iter_coeffs_no_nan = iter(
        zip(
            a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan
        )
    )
    next_time_no_nan = next(iter_times_no_nan)
    for time in times[:-1]:
        # will always trigger on the first iteration because of how we've imputed missing values at the start and
        # end of the time series.
        if time >= next_time_no_nan:
            prev_time_no_nan = next_time_no_nan
            next_time_no_nan = next(iter_times_no_nan)
            next_a_no_nan, next_b_no_nan, next_two_c_no_nan, next_three_d_no_nan = next(
                iter_coeffs_no_nan
            )
        offset = prev_time_no_nan - time
        a_inner = (0.5 * next_two_c_no_nan - next_three_d_no_nan * offset / 3) * offset
        a_pieces.append(next_a_no_nan + (a_inner - next_b_no_nan) * offset)
        b_pieces.append(
            next_b_no_nan + (next_three_d_no_nan * offset - next_two_c_no_nan) * offset
        )
        two_c_pieces.append(next_two_c_no_nan - 2 * next_three_d_no_nan * offset)
        three_d_pieces.append(next_three_d_no_nan)

    return (
        cheap_stack(a_pieces, dim=0),
        cheap_stack(b_pieces, dim=0),
        cheap_stack(two_c_pieces, dim=0),
        cheap_stack(three_d_pieces, dim=0),
    )


# The mathematics of this are adapted from  http://mathworld.wolfram.com/CubicSpline.html, although they only treat the
# case of each piece being parameterised by [0, 1]. (We instead take the length of each piece to be the difference in
# time stamps.)
def natural_cubic_spline_coeffs(t, X):
    """Calculates the coefficients of the natural cubic spline approximation to the batch of controls given.

    Arguments:
        t: One dimensional tensor of times. Must be monotonically increasing.
        X: tensor of values, of shape (..., L, C), where ... is some number of batch dimensions, L is some length
            that must be the same as the length of t, and C is some number of channels. This is interpreted as a
            (batch of) paths taking values in a C-dimensional real vector space, with L observations. Missing values
            are supported, and should be represented as NaNs.

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        Calling this function can be pretty slow. Make sure to cache the result, and don't reinstantiate it on every
        forward pass, if at all possible.

    Returns:
        Four tensors, which should in turn be passed to `controldiffeq.NaturalCubicSpline`.

        Why do we do it like this? Because typically you want to use PyTorch tensors at various interfaces, for example
        when loading a batch from a DataLoader. If we wrapped all of this up into just the
        `controldiffeq.NaturalCubicSpline` class then that sort of thing wouldn't be possible.

        As such the suggested use is to:
        (a) Load your datasets.
        (b) Preprocess it with this function.
        (c) Save the result.
        (d) Treat the result as your datasets as far as PyTorch's `torch.utils.datasets.Dataset` and
            `torch.utils.datasets.DataLoader` classes are concerned.
        (e) Call NaturalCubicSpline as the first part of your model.

        See also the accompanying example.py.
    """

    if not t.is_floating_point():
        raise ValueError("t and X must both be floating point/")
    if not X.is_floating_point():
        raise ValueError("t and X must both be floating point/")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional.")
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")

    if len(X.shape) < 2:
        raise ValueError(
            "X must have at least two dimensions, corresponding to time and channels."
        )

    if X.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t.")

    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2.")

    if torch.isnan(X).any():
        # Transpose because channels are a batch dimension for the purpose of finding interpolating polynomials.
        # b, two_c, three_d have shape (..., channels, length - 1)
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(
            t, X.transpose(-1, -2)
        )
    else:
        # Can do things more quickly in this case.
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(
            t, X.transpose(-1, -2)
        )

    # These all have shape (..., length - 1, channels)
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    two_c = two_c.transpose(-1, -2)
    three_d = three_d.transpose(-1, -2)
    return a, b, two_c, three_d
