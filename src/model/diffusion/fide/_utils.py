import torch

from scipy.stats import genextreme
# from properscoring import crps_ensemble


def fitting_gev_and_sampling(ymax, num_samples):
    # Fit the data to a GEV distribution
    shape, loc, scale = genextreme.fit(ymax)
    # Generate a GEV distribution using the fitted parameters
    gev_distribution = genextreme(shape, loc=loc, scale=scale)
    # Perform a Kolmogorov-Smirnov test for goodness of fit
    # ks_statistic, p_value = kstest(ymax, cdf='genextreme', args=(shape, loc, scale))
    # print(f"Kolmogorov–Smirnov test: K-S Statistic: {ks_statistic}; p-value: {p_value}")

    # gev_samples = gev_distribution.rvs(size=num_samples)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))  # 1 row, 3 columns
    # # Plot data on each subplot
    # axs[0].hist(ymax, bins=30, density=True, alpha=0.5, color='b', label='Histogram')
    # sns.kdeplot(ymax, color='b', label='Smoothed (KDE)', ax=axs[0])
    # x = np.linspace(min(ymax), max(ymax), 100)
    # axs[0].plot(x, gev_distribution.pdf(x), 'r-', lw=2, label='GEV PDF')
    # axs[0].legend()
    # axs[0].set_title(f'Training/Fitted Data')

    # axs[1].hist(gev_samples, bins=30, density=True, alpha=0.5, color='b', label='Histogram')
    # sns.kdeplot(gev_samples, color='b', label='Smoothed (KDE)', ax=axs[1])
    # axs[1].plot(x, gev_distribution.pdf(x), 'r-', lw=2, label='GEV PDF')
    # axs[1].legend()
    # axs[1].set_title(f'Sampled Data')
    # # Adjust layout
    # plt.tight_layout()
    # # Show the plot
    # plt.show()
    return gev_distribution  # Return both sampled data and the fitted GEV model




def get_betas(steps):
    beta_start, beta_end = 1e-4, 0.2
    diffusion_ind = torch.linspace(0, 1, steps)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind

# diffusion_steps = 100
# betas = get_betas(diffusion_steps)
# alphas = torch.cumprod(1 - betas, dim=0)

gp_sigma = 0.05

def get_gp_covariance(t):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5 # for numerical stability
    return torch.exp(-torch.square(s / gp_sigma)) + diag

def add_noise(x, t, i, alphas):
    """
    x: Clean data sample, shape [B, S, D]
    t: Times of observations, shape [B, S, 1]
    i: Diffusion step, shape [B, S, 1]
    """
    noise_gaussian = torch.randn_like(x)

    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    noise = L @ noise_gaussian

    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

    return x_noisy, noise


# Example usage:
# train_history = np.random.rand(50) * 0.05  # Replace this with your actual training loss history
# plot_losses(train_history)

def linear_decaying_weight(input_tensor):
    # Assuming input_tensor is a torch tensor of shape [10000, 1]

    # Calculate the maximum and minimum values of the input tensor
    max_value = torch.max(input_tensor)
    min_value = torch.min(input_tensor)

    # Ensure the input tensor is in the range [0, 1]
    normalized_input = (input_tensor - min_value) / (max_value - min_value)

    # Calculate linear decay weights
    weights = 1 - normalized_input

    return weights

def linear_decay(input_tensor, diffusion_steps):
    # Set the decay starting point and ending point
    start_index = 0
    end_index = int(0.66*diffusion_steps)

    # Create an output tensor with the same size as the input tensor
    output_tensor = torch.zeros_like(input_tensor)

    # Apply linear decay to the values based on their indices
    for i in range(input_tensor.size(0)):
        index_value = input_tensor[i, 0].item()

        if index_value == start_index:
            output_tensor[i, 0] = 1.0
        elif start_index < index_value < end_index:
            # Linear decay function: f(x) = 1 - (x - start_index) / (end_index - start_index)
            output_tensor[i, 0] = 1.0 - (index_value - start_index) / (end_index - start_index)
        else:
            output_tensor[i, 0] = 0.0

    return output_tensor

# Example usage:
# input_values = i[:,0,:].reshape(-1,1)   # Assuming your input values range from 0 to 99
# weights = linear_decaying_weight(input_values)
