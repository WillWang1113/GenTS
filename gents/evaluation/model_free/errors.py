from typing import Optional, Union

import numpy as np
from sklearn.metrics import mean_squared_error


def _check_shape(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: {y_true.shape} vs {y_pred.shape}. "
            "Both arrays must have the same shape."
        )


def mse(y_true, y_pred):
    _check_shape(y_true, y_pred)
    return mean_squared_error(y_true, y_pred)


def crps(y_true: np.ndarray, y_pred: np.ndarray, quantiles=np.arange(0.1, 1.0, 0.1)):
    if y_pred.shape[:-1] != y_true.shape:
        raise ValueError(
            f"Shape mismatch: {y_true.shape} vs {y_pred.shape}. "
            "The first several dimensions of y_true and y_pred must match."
        )

    if y_pred.shape[-1] != len(quantiles):
        y_pred_quantiles = np.quantile(y_pred, quantiles, axis=-1)
        y_pred_quantiles = np.moveaxis(y_pred_quantiles, 0, -1)
    else:
        y_pred_quantiles = y_pred

    return mqloss(y_true, y_pred_quantiles, quantiles)


# mqloss, mae, mse are from neuralforecast.losses.numpy
def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Multi-Quantile loss

    Calculates the Multi-Quantile loss (MQL) between `y` and `y_hat`.
    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute
    difference between predicted quantiles and observed values.

    $$ \mathrm{MQL}(\\mathbf{y}_{\\tau},[\\mathbf{\hat{y}}^{(q_{1})}_{\\tau}, ... ,\hat{y}^{(q_{n})}_{\\tau}]) = \\frac{1}{n} \\sum_{q_{i}} \mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q_{i})}_{\\tau}) $$

    The limit behavior of MQL allows to measure the accuracy
    of a full predictive distribution $\mathbf{\hat{F}}_{\\tau}$ with
    the continuous ranked probability score (CRPS). This can be achieved
    through a numerical integration technique, that discretizes the quantiles
    and treats the CRPS integral with a left Riemann approximation, averaging over
    uniformly distanced quantiles.

    $$ \mathrm{CRPS}(y_{\\tau}, \mathbf{\hat{F}}_{\\tau}) = \int^{1}_{0} \mathrm{QL}(y_{\\tau}, \hat{y}^{(q)}_{\\tau}) dq $$

    **Parameters:**<br>
    `y`: numpy array, Actual values.<br>
    `y_hat`: numpy array, Predicted values.<br>
    `quantiles`: numpy array,(n_quantiles). Quantiles to estimate from the distribution of y.<br>
    `mask`: numpy array, Specifies date stamps per serie to consider in loss.<br>

    **Returns:**<br>
    `mqloss`: numpy array, (single value).

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)<br>
    [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """
    if weights is None:
        weights = np.ones(y.shape)

    # _metric_protections(y, y_hat, weights)
    n_q = len(quantiles)

    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    mqloss = quantiles * sq + (1 - quantiles) * s1_q

    # Match y/weights dimensions and compute weighted average
    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    mqloss = np.average(mqloss, weights=weights, axis=axis)

    return mqloss
