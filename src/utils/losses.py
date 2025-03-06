import torch
import torch.nn.functional as F


def kl_loss(z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
    # COMPUTE KL DIV
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * (z_prior_logvar - z_post_logvar +
                   ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
    return kld_z.sum(dim=tuple(range(1, kld_z.ndim))).mean()
