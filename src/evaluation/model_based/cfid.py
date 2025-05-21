import numpy as np
from ._ts2vec import initialize_ts2vec
from scipy.linalg import sqrtm


def calculate_fid(act1, act2):
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


def context_fid(train_data, ori_data, gen_data, device):
    fid_model = initialize_ts2vec(np.transpose(train_data, (0, 2, 1)), device)
    ori_repr = fid_model.encode(
        np.transpose(ori_data, (0, 2, 1)), encoding_window="full_series"
    )
    gen_repr = fid_model.encode(
        np.transpose(gen_data, (0, 2, 1)), encoding_window="full_series"
    )
    cfid = calculate_fid(ori_repr, gen_repr)
    return cfid


