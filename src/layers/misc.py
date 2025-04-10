import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import six
# __all__ = ["sample_rademacher_like", "sample_gaussian_like", "exists", 'l2norm','default','noop','log','']

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sample_standard_gaussian(mu, sigma):
    device = mu.device
    d = torch.distributions.normal.Normal(
        torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device)
    )
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def softmax_sample(t, temperature, dim=-1):
    if isinstance(temperature, type(None)):
        return t.argmax(dim=dim)

    m = Categorical(logits=t / temperature)
    return m.sample()


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.0
    for i in range(y.shape[1]):
        sum_diag += (
            torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0]
            .contiguous()[:, i]
            .contiguous()
        )
    return sum_diag.contiguous()


def divergence_approx(f, y, e=None):
    samples = []
    sqnorms = []
    for e_ in e:
        e_dzdx = torch.autograd.grad(f, y, e_, create_graph=True)[0]
        n = e_dzdx.view(y.size(0), -1).pow(2).mean(dim=1, keepdim=True)
        sqnorms.append(n)
        e_dzdx_e = e_dzdx * e_
        samples.append(e_dzdx_e.view(y.shape[0], -1).sum(dim=1, keepdim=True))

    S = torch.cat(samples, dim=1)
    approx_tr_dzdx = S.mean(dim=1)

    N = torch.cat(sqnorms, dim=1).mean(dim=1)

    return approx_tr_dzdx, N


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    # "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


# FLOWs

def total_derivative(x, t, logp, dx, dlogp, unused_context):
    del logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

    try:
        u = torch.full_like(dx, 1 / x.numel(), requires_grad=True)
        tmp = torch.autograd.grad((u * dx).sum(), t, create_graph=True)[0]
        partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

        total_deriv = directional_dx + partial_dt
    except RuntimeError as e:
        if "One of the differentiated Tensors" in e.__str__():
            raise RuntimeError(
                'No partial derivative with respect to time. Use mathematically equivalent "directional_derivative" regularizer instead'
            )

    tdv2 = total_deriv.pow(2).view(x.size(0), -1)

    return 0.5 * tdv2.mean(dim=-1)


def directional_derivative(x, t, logp, dx, dlogp, unused_context):
    del t, logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    ddx2 = directional_dx.pow(2).view(x.size(0), -1)

    return 0.5 * ddx2.mean(dim=-1)


def quadratic_cost(x, t, logp, dx, dlogp, unused_context):
    del x, logp, dlogp, t, unused_context
    dx = dx.view(dx.shape[0], -1)
    return 0.5 * dx.pow(2).mean(dim=-1)


def jacobian_frobenius_regularization_fn(x, t, logp, dx, dlogp, context):
    sh = x.shape
    del logp, dlogp, t, dx, x
    sqjac = context.sqjacnorm

    return context.sqjacnorm


def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)



REGULARIZATION_FNS = {
    "kinetic_energy": quadratic_cost,
    "jacobian_norm2": jacobian_frobenius_regularization_fn,
    "total_deriv": total_derivative,
    "directional_penalty": directional_derivative
}

INV_REGULARIZATION_FNS = {v: k for k, v in six.iteritems(REGULARIZATION_FNS)}