import torch

def reparameterize(mean, logvar, random_sampling=True):
    # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z
    else:
        return mean
    

def exists(x):
    """
    Check if the input is not None.

    Args:
        x: The input to check.

    Returns:
        bool: True if the input is not None, False otherwise.
    """
    return x is not None

def default(val, d):
    """
    Return the value if it exists, otherwise return the default value.

    Args:
        val: The value to check.
        d: The default value or a callable that returns the default value.

    Returns:
        The value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    """
    Return the input tensor unchanged.

    Args:
        t: The input tensor.
        *args: Additional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        The input tensor unchanged.
    """
    return t

def extract(a, t, x_shape):
    """
    Extracts values from tensor `a` at indices specified by tensor `t` and reshapes the result.
    Args:
        a (torch.Tensor): The input tensor from which values are extracted.
        t (torch.Tensor): The tensor containing indices to extract from `a`.
        x_shape (tuple): The shape of the tensor `x` which determines the final shape of the output.
    Returns:
        torch.Tensor: A tensor containing the extracted values, reshaped to match the shape of `x` except for the first dimension.
    """

    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

