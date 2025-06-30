"""
NOT A STRICT GENERATIVE MODEL
"""

import copy
import math
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

# from entmax import sparsemax, entmax15, entmax_bisect, EntmaxBisect
import torch.nn.functional as F

# import utils
# from torch.autograd import grad
# from IPython import embed
from torch.autograd import Function, Variable

from gents.common._modules import LayerNorm
from gents.model.base import BaseModel

"""
An implementation of entmax (Peters et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.

This builds on previous work with sparsemax (Martins & Astudillo, 2016).
See https://arxiv.org/pdf/1602.02068.
"""

# Author: Ben Peters


class EntmaxBisectFunction(Function):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X * (alpha - 1)
        max_val = max_val * (alpha - 1)

        # Note: when alpha < 1, tau_lo > tau_hi. This still works since dm < 0.
        tau_lo = max_val - cls._gp(1, alpha)
        tau_hi = max_val - cls._gp(1 / d, alpha)

        # Note: f_lo should always be non-negative.
        # f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):
            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1

            mask = (f_m >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)

        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:
            # alpha gradient computation
            # d_alpha = (partial_y / partial_alpha) * dY
            # NOTE: ensure alpha is not close to 1
            # since there is an indetermination
            # batch_size, _ = dY.shape

            # shannon terms
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            # shannon entropy
            ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)

            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)

        return dX, d_alpha, None, None, None


def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    """alpha-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.

    This function is differentiable with respect to both X and alpha.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        alpha.shape[j] == (X.shape[j] if j != dim else 1)
        A value of alpha=2 corresponds to sparsemax, and alpha=1 would in theory recover
        softmax. For numeric reasons, this algorithm does not work with `alpha=1`: if you
        want softmax, we recommend `torch.nn.softmax`.

    dim : int
        The dimension along which to apply alpha-entmax.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)


class EntmaxBisect(nn.Module):
    def __init__(self, alpha=1.5, dim=-1, n_iter=50):
        """alpha-entmax: normalizing sparse map (a la softmax) via bisection.

        Solves the optimization problem:

            max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

        where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
        using a bisection (root finding, binary search) algorithm.

        Parameters
        ----------
        alpha : float or torch.Tensor
            Tensor of alpha parameters (> 1) to use. If scalar
            or python float, the same value is used for all rows, otherwise,
            it must have shape (or be expandable to)
            alpha.shape[j] == (X.shape[j] if j != dim else 1)
            A value of alpha=2 corresponds to sparsemax; and alpha=1 would in theory recover
            softmax. For numeric reasons, this algorithm does not work with `alpha=1`; if you
            want softmax, we recommend `torch.nn.softmax`.

        dim : int
            The dimension along which to apply alpha-entmax.

        n_iter : int
            Number of bisection iterations. For float32, 24 iterations should
            suffice for machine precision.

        """
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        super().__init__()

    def forward(self, X):
        return entmax_bisect(X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter)


def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for sparsemax: optimal threshold and support size.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.

    dim : int
        The dimension along which to apply sparsemax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


def _entmax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for 1.5-entmax: optimal threshold and support size.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.

    dim : int
        The dimension along which to apply 1.5-entmax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output, supp_size

    @classmethod
    def backward(cls, ctx, grad_output, supp):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None, None


class Entmax15Function(Function):
    @classmethod
    def forward(cls, ctx, X, dim=0, k=None):
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, supp_size = _entmax_threshold_and_support(X, dim=dim, k=k)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y, supp_size

    @classmethod
    def backward(cls, ctx, dY, supp):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None, None


def sparsemax(X, dim=-1, k=None, return_support_size=False):
    """sparsemax: normalizing sparse transform (a la softmax).

    Solves the projection:

        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    dim : int
        The dimension along which to apply sparsemax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    return_support_size : bool
        Whether to return the support size of the result as well as the result
        itself.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    support : (optional) torch tensor, same shape as X except for dim,
              where it is 1.
    """
    P, support = SparsemaxFunction.apply(X, dim, k)
    if return_support_size:
        return P, support
    return P


def entmax15(X, dim=-1, k=None, return_support_size=False):
    """1.5-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    dim : int
        The dimension along which to apply 1.5-entmax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    return_support_size : bool
        Whether to return the support size of the result as well as the result
        itself.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    support : (optional) torch tensor, same shape as X except for dim,
              where it is 1.
    """

    P, support = Entmax15Function.apply(X, dim, k)
    if return_support_size:
        return P, support
    return P


class Sparsemax(nn.Module):
    def __init__(self, dim=-1, k=None, return_support_size=False):
        """sparsemax: normalizing sparse transform (a la softmax).

        Solves the projection:

            min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

        Parameters
        ----------
        dim : int
            The dimension along which to apply sparsemax.

        k : int or None
            number of largest elements to partial-sort over. For optimal
            performance, should be slightly bigger than the expected number of
            nonzeros in the solution. If the solution is more than k-sparse,
            this function is recursively called with a 2*k schedule.
            If `None`, full sorting is performed from the beginning.

        return_support_size : bool
            Whether to return the support size of the result as well as the
            result itself.
        """
        self.dim = dim
        self.k = k
        self.return_support_size = return_support_size
        super(Sparsemax, self).__init__()

    def forward(self, X):
        return sparsemax(
            X, dim=self.dim, k=self.k, return_support_size=self.return_support_size
        )


class Entmax15(nn.Module):
    def __init__(self, dim=-1, k=None, return_support_size=False):
        """1.5-entmax: normalizing sparse transform (a la softmax).

        Solves the optimization problem:

            max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

        where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

        Parameters
        ----------
        dim : int
            The dimension along which to apply 1.5-entmax.

        k : int or None
            number of largest elements to partial-sort over. For optimal
            performance, should be slightly bigger than the expected number of
            nonzeros in the solution. If the solution is more than k-sparse,
            this function is recursively called with a 2*k schedule.
            If `None`, full sorting is performed from the beginning.

        return_support_size : bool
            Whether to return the support size of the result as well as the
            result itself.
        """
        self.dim = dim
        self.k = k
        self.return_support_size = return_support_size
        super(Entmax15, self).__init__()

    def forward(self, X):
        return entmax15(
            X, dim=self.dim, k=self.k, return_support_size=self.return_support_size
        )


class EncoderDecoder(nn.Module):
    def __init__(self, params, emb, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.predict_steps = params.predict_steps

    def forward(self, x, idx):
        src_mask, encoder_out = self.encode(x[:, : -self.predict_steps, :], idx)
        # mu_en, sigma_en = self.generator(encoder_out)
        decoder_out = self.decode(
            encoder_out, x[:, -self.predict_steps :, :], idx, src_mask
        )
        q50, q90 = self.generator(decoder_out)

        # mu = torch.cat((mu_en, mu_de), 1)
        # sigma = torch.cat((sigma_en, sigma_de), 1)
        return q50, q90

    def encode(self, x, idx):
        src_mask = (x[:, :, 0] != 0).unsqueeze(-2)
        # src_mask1 = make_std_mask(x[:, :, 0], 0)
        embeded = self.emb(x, idx)
        encoder_out = self.encoder(embeded, None)

        return src_mask, encoder_out

    def decode(self, memory, x, idx, src_mask):
        tgt_mask = make_std_mask(x[:, :, 0], 0)
        embeded = self.emb(x, idx)
        decoder_out = self.decoder(embeded, memory, None, tgt_mask)
        return decoder_out


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embedding(nn.Module):
    def __init__(self, params, position):
        super(Embedding, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim)
        # self.embed1 = nn.Linear(params.embedding_dim + params.cov_dim+ 1, params.d_model) #!!!!!!!parts is 24, others 25..........
        """
        if(params.dataset == "wind"):
            self.embed1 = nn.Linear(6, params.d_model)
        else:
        """
        self.embedding
        self.embed1 = nn.Linear(
            params.embedding_dim + params.cov_dim + 1, params.d_model
        )
        self.embed2 = position

    def forward(self, x, idx):
        "Pass the input (and mask) through each layer in turn.  x : [bs, window_len, 5]"

        idx = idx.repeat(
            1, x.shape[1]
        )  # idx is the store id of this batch , [bs, window_len]
        """
        if(self.params.dataset=="wind"):
            idx = torch.unsqueeze(idx, -1)
            output = torch.cat((x, idx.float()), dim=2) # [bs, widow_len, 25]  [bs, window]  wind dataset!!!
        else:
        """
        onehot_embed = self.embedding(
            idx
        )  # [bs, windows_len, embedding_dim(default 20)]
        try:
            output = torch.cat((x, onehot_embed), dim=-1)
            output = self.embed2(self.embed1(output))
        except:
            raise Exception(
            "Masking not implemented for this attention type. Please use a mask with the same shape as scores."
        )
            # embed()
        return output


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.q50 = nn.Linear(params.d_model, 1)
        self.q90 = nn.Linear(params.d_model, 1)

    def forward(self, x):
        q50 = self.q50(x)
        q90 = self.q90(x)
        return torch.squeeze(q50), torch.squeeze(q90)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, params, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, params.N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, src_mask)
        encoder_output = self.norm(x)
        return encoder_output


class EncoderLayer(nn.Module):
    def __init__(self, params, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.params = params
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(params.d_model, dropout), 2)
        self.size = params.d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, params, layer):
        super(Decoder, self).__init__()
        self.layers = clones(layer, params.N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, params, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = params.d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class AlphaChooser(torch.nn.Module):
    def __init__(self, head_count):
        """head_count (int): number of attention heads"""
        super(AlphaChooser, self).__init__()
        self.pre_alpha = nn.Parameter(torch.randn(head_count))

    def forward(self):
        alpha = 1 + torch.sigmoid(self.pre_alpha)
        return torch.clamp(alpha, min=1.01, max=2)


# class EntmaxAlphaBencher(object):
#     def __init__(self, X, alpha, n_iter=25):
#         self.n_iter = n_iter
#         self.X_data = X
#         self.alpha = alpha

#     def __enter__(self):
#         self.X = self.X_data.clone().requires_grad_()
#         self.dY = torch.randn_like(self.X)
#         self.alpha = alpha
#         return self

#     def forward(self):
#         self.Y = entmax_bisect(self.X, self.alpha, dim=-1, n_iter=self.n_iter)

#     def backward(self):
#         grad(outputs=(self.Y,),
#              inputs=(self.X, self.alpha),
#              grad_outputs=(self.Y))

#     def __exit__(self, *args):
#         try:
#             del self.X
#             del self.alpha
#         except AttributeError:
#             pass

#         try:
#             del self.Y
#         except AttributeError:
#             pass


def attention(query, key, value, params, mask=None, dropout=None, alpha=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        try:
            scores = scores.masked_fill(mask == 0, -1e9)
        except: 
            raise Exception(
            "Masking not implemented for this attention type. Please use a mask with the same shape as scores."
        )
            # embed()

    if params.attn_type == "softmax":
        p_attn = F.softmax(scores, dim=-1)
    elif params.attn_type == "sparsemax":
        p_attn = sparsemax(scores, dim=-1)
    elif params.attn_type == "entmax15":
        p_attn = entmax15(scores, dim=-1)
    # ! WRONG SHAPE IN ORIGINAL CODE
    elif params.attn_type == "entmax":
        attn_fn = EntmaxBisect(alpha, n_iter=25)
        p_attn = attn_fn(scores)
        # p_attn = EntmaxBisect(scores, alpha, n_iter=25)
    else:
        raise Exception
    if dropout is not None:
        p_attn = dropout(p_attn)
    p_attn = p_attn.to(torch.float32)
    return torch.matmul(p_attn, value), scores, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, params, dropout=0.2):  # TODO : h , dropout
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert params.d_model % params.h == 0

        self.d_k = params.d_model // params.h
        self.h = params.h
        self.linears = clones(nn.Linear(params.d_model, params.d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.params = params
        self.scores = None
        self.alpha_choser = AlphaChooser(params.h)
        self.alpha = None
        self.attn_type = params.attn_type

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        if self.attn_type == "entmax":
            self.alpha = self.alpha_choser()
        x, self.scores, self.attn = attention(
            query,
            key,
            value,
            self.params,
            mask=mask,
            dropout=self.dropout,
            alpha=self.alpha,
        )
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=500):  # TODO:max_len
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(params.train_window, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


def loss_quantile(mu: Variable, labels: Variable, quantile: Variable):
    loss = 0
    for i in range(mu.shape[1]):
        mu_e = mu[:, i]
        labels_e = labels[:, i]

        I = (labels_e >= mu_e).float()
        each_loss = 2 * (
            torch.sum(
                quantile * ((labels_e - mu_e) * I)
                + (1 - quantile) * (mu_e - labels_e) * (1 - I)
            )
        )
        loss += each_loss

    return loss


class AST(BaseModel):
    ALLOW_CONDITION = ['predict']
    def __init__(
        self,
        seq_len,
        obs_len,
        seq_dim=1,
        covariate_dim=0,
        class_emb_dim=8,
        hidden_size=512,
        ff_hidden_size=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        attn_type="entmax15",
        lr=0.001,
        weight_decay=0.0001,
        condition="predict",
        n_classes=1,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition)
        # assert condition == "predict"
        assert seq_dim == 1
        assert attn_type in ["softmax", "sparsemax", "entmax15"]
        self.automatic_optimization = False
        self.save_hyperparameters()
        params = Namespace(
            train_window=seq_len + obs_len,
            predict_start=obs_len,
            predict_steps=seq_len,
            num_class=n_classes,
            d_model=hidden_size,
            d_ff=ff_hidden_size,
            dropout=dropout,
            N=num_layers,
            h=num_heads,
            attn_type=attn_type,
            cov_dim=covariate_dim,
            embedding_dim=class_emb_dim,
            # **self.hparams_initial,
        )
        self.params = params
        self.seq_dim = seq_dim
        self.num_covariates = covariate_dim

        attn = MultiHeadedAttention(params)
        ff = PositionwiseFeedForward(
            params.d_model, d_ff=params.d_ff, dropout=params.dropout
        )
        position = PositionalEncoding(params.d_model, dropout=params.dropout)
        # pt = transformer.TimeEncoding(params.d_model, dropout=0.1).cuda()
        ge = Generator(params)
        emb = Embedding(params, position)

        self.generator = EncoderDecoder(
            params=params,
            emb=emb,
            encoder=Encoder(
                params, EncoderLayer(params, attn, ff, dropout=params.dropout)
            ),
            decoder=Decoder(
                params,
                DecoderLayer(params, attn, attn, ff, dropout=params.dropout),
            ),
            generator=ge,
        )

        # self.generator = Generator(self.hparams)
        self.discriminator = Discriminator(self.params)
        self.adversarial_loss = torch.nn.BCELoss()

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        assert condition is not None
        assert n_sample == condition.shape[0]
        idx = kwargs.get("chnl_id", None)
        assert idx is not None

        x = kwargs["seq"].clone()
        # obs_x = condition

        # x = torch.cat((obs_x, x), dim=1)
        if self.num_covariates > 0:
            covariates = kwargs.get("covariates", None)
            assert covariates is not None
            x = torch.cat((x, covariates), dim=-1)
        x[:, 0, 0] = 0

        batch_size = n_sample
        sample_mu = torch.zeros(
            batch_size, self.params.predict_steps, device=self.device
        )
        sample_q90 = torch.zeros(
            batch_size, self.params.predict_steps, device=self.device
        )
        src_mask, memory = self.generator.encode(
            x[:, : self.params.predict_start, :], idx
        )
        for t in range(self.params.predict_steps):
            ys = x[:, self.params.predict_start : self.params.predict_start + t + 1, :]
            out = self.generator.decode(memory, ys, idx, src_mask)
            q50, q90 = self.generator.generator(out)
            if t != 0:
                q50 = q50[:, -1]
                q90 = q90[:, -1]
            sample_mu[:, t] = q50
            sample_q90[:, t] = q90
            if t < (self.params.predict_steps - 1):
                x[:, self.params.predict_steps + t + 1, 0] = q50

        return sample_mu

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        d_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return [g_optim, d_optim], []

    def training_step(self, batch, batch_idx, **kwargs):
        optimizer_G, optimizer_D = self.optimizers()

        x = batch["seq"]
        # obs_x = batch["c"]
        idx = batch.get("chnl_id", None)
        batch_size = x.shape[0]
        # assert x.shape[-1] == 1
        assert idx is not None

        # seq = torch.cat((obs_x, x), dim=1)
        labels_batch = x.squeeze(-1)
        train_batch = x
        if self.num_covariates > 0:
            assert batch.get("covariates", None) is not None
            covariates = batch["covariates"]  # [bs, obs+pred, num_covariates]
            train_batch = torch.cat((train_batch, covariates), dim=-1)
        train_batch[:, 0, 0] = 0
        # batch_size = train_batch.shape[0]

        # TODO: data transformation, covariates?
        # train_batch: # [bs, obs+pred len, 1(univariate) + num_covariates + 1(chnl_id)]
        # [bs, 0, 0] = 0 (first time step is always 0)
        # can only deal with univariate time series
        # index the channel id

        # labels_batch: # [bs, obs+pred]
        # train_batch = x
        # train_batch = train_batch.to(torch.float32).to(params.device)

        # labels_batch = torch.concat((obs_x, x), dim=1)
        # labels_batch = labels_batch.to(torch.float32).to(params.device)
        # idx = idx.unsqueeze(-1).to(params.device)
        # print(train_batch.shape)
        # print(labels_batch.shape)
        # print(idx.shape)

        # Adversarial ground truths
        self.toggle_optimizer(optimizer_G)
        valid = torch.ones(batch_size, 1).to(x)
        fake = torch.zeros(batch_size, 1).to(x)

        # valid = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        # fake = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        labels = labels_batch[:, self.params.predict_start :]
        q50, q90 = self.generator.forward(train_batch, idx)
        d_loss = 0

        fake_input = torch.cat((labels_batch[:, : self.params.predict_start], q50), 1)
        # -------------------------------------------------------------------
        # Train the generator
        # -------------------------------------------------------------------

        optimizer_G.zero_grad()
        loss = loss_quantile(
            q50, labels, torch.tensor(0.5)
        ) + 0.1 * self.adversarial_loss(self.discriminator(fake_input), valid)
        self.manual_backward(loss)
        # loss.backward()
        optimizer_G.step()
        g_loss = loss.item() / self.params.train_window
        self.untoggle_optimizer(optimizer_G)
        # loss_epoch[i] = g_loss

        # -------------------------------------------------------------------
        # Train the discriminator
        # -------------------------------------------------------------------
        self.toggle_optimizer(optimizer_D)
        optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(labels_batch), valid)
        fake_loss = self.adversarial_loss(self.discriminator(fake_input.detach()), fake)
        loss_d = 0.5 * (real_loss + fake_loss)
        self.manual_backward(loss_d)
        # loss_d.backward()
        optimizer_D.step()

        d_loss = loss_d.item()
        # d_loss_epoch[i] = d_loss
        self.untoggle_optimizer(optimizer_D)

        # if i % 1000 == 0:
        # logger.info("G_loss: {} ; D_loss: {}".format(g_loss, d_loss))
        self.log_dict({"g_loss": g_loss, "d_loss": d_loss})

        # return super().training_step(*args, **kwargs)
