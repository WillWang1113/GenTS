import numpy as np
from ._ts2vec import initialize_ts2vec
from scipy.linalg import sqrtm


def _calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def context_fid(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    device: str = "cpu",
    ts2vec_path: str = None,
    train_data: np.ndarray = None,
):
    """Calculate `context-FID <https://arxiv.org/abs/2108.00981>`__.

    Context-FID is a FID-like metric for evaluating how realistic the generated time series is (compared to the true time series).
    It requires to train a representative learning time series model (TS2Vec)
    on every time series dataset. Then calculate FID using the trained representative learning model.

    Args:
        train_data (np.ndarray): Time series training dataset. Used for training TS2Vec model.
        ori_data (np.ndarray): Time series test dataset.
        gen_data (np.ndarray): Generated time series.
        device (str, optional): Computing device. Defaults to "cpu".

    """
    if ts2vec_path is None and train_data is None:
        raise ValueError("Either ts2vec_path or train_data must be provided.")
    elif ts2vec_path is None and train_data is not None:
        fid_model = initialize_ts2vec(
            ori_data.shape[-1], train_data, device, ts2vec_path
        )
    elif ts2vec_path is not None and train_data is None:
        fid_model = initialize_ts2vec(ori_data.shape[-1], None, device, ts2vec_path)
    else:
        # print(
        #     "both ts2vec_path and train_data are provided, use train_data to train a ts2vec and save it to ts2vec_path."
        # )
        fid_model = initialize_ts2vec(
            ori_data.shape[-1], train_data, device, ts2vec_path
        )

    ori_repr = fid_model.encode(ori_data, encoding_window="full_series")
    gen_repr = fid_model.encode(gen_data, encoding_window="full_series")
    cfid = _calculate_fid(ori_repr, gen_repr)
    return cfid
