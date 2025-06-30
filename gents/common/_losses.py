import torch


def kl_loss(z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
    # COMPUTE KL DIV
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * (
        z_prior_logvar
        - z_post_logvar
        + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var)
        - 1
    )
    return kld_z


# def log_normal(x, mu, var):
#     """Logarithm of normal distribution with mean=mu and variance=var
#        log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

#     Args:
#        x: (array) corresponding array containing the input
#        mu: (array) corresponding array containing the mean
#        var: (array) corresponding array containing the variance

#     Returns:
#        output: (array/float) depending on average parameters the result will be the mean
#                             of all the sample losses or an array with the losses per sample
#     """
#     eps = 1e-8
#     if eps > 0.0:
#         var = var + eps
#     # return -0.5 * torch.sum(
#     #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
#     return 0.5 * torch.mean(
#         torch.log(2.0 * torch.pi) + torch.log(var) + torch.pow(x - mu, 2) / var
#     )
