import torch
from torch.utils.data import Dataset
from einops import repeat
import torchcde


class TSDataset(Dataset):
    """Time series dataset. A batch conatins:

    A batch
        "seq": (B, T, C) Target time series window
        "t": (B, T) Time index at each time step in the window.
        Could be either the last axis of input data, or default [0,1,...,T-1]
        "c": (B, T/OBS, C) Condition. Empty if unconditional.
        "coeffs": (B, T, C) Coefficients of cubic spline of NCDE-related models.
        Empty if add_coeffs is False.
        "chnl_id": (1,) channel id if channel_independent is True
    """

    def __init__(
        self,
        data: torch.Tensor,
        cond: torch.Tensor = None,
        cond_type: str = None,
        add_coeffs: str = None,
        time_idx_last: bool = False,
        channel_independent: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert data.dim() == 3
        assert add_coeffs in [None, "linear", "cubic_spline"]
        assert cond_type in [None, "predict", "impute"]
        self.data = data
        self.data_shape = data.shape
        self.sample_chnl = False
        # if input TS is multivariate and treat channel independent, we need to sample a channel
        if (self.data.shape[-1] > 1) and channel_independent:
            self.sample_chnl = True

        if time_idx_last:
            self.data = self.data[:, :, :-1]
            self.time_idx = self.data[:, :, -1]
        else:
            self.time_idx = torch.arange(data.shape[1]).float()
            self.time_idx = repeat(self.time_idx, "t -> b t", b=data.shape[0])

        self.cond = cond
        self.cond_shape = None
        if cond is not None:
            self.cond_shape = tuple(cond.shape[1:])
        if add_coeffs is not None:
            if add_coeffs == "linear":
                interp_fn = torchcde.linear_interpolation_coeffs
            else:
                interp_fn = torchcde.natural_cubic_spline_coeffs

            # from torchcde import natural_cubic_spline_coeffs, linear_interpolation_coeffs

            t = torch.arange(data.shape[1]).float()
            
            if (cond_type == 'impute') and (cond is not None):
                data_nan = data.masked_fill(cond.bool(), float("nan"))
            else:
                data_nan = data
                
            # if cond is None:
            #     data_nan = data
            # elif cond_type == "impute":
            #     # assert cond.type() == "torch.bool"
            #     data_nan = data.masked_fill(cond.bool(), float("nan"))
            self.coeffs = interp_fn(data_nan, t)
        else:
            self.coeffs = None

    def __getitem__(self, index):
        if self.sample_chnl:
            chnl = torch.randint(0, self.data_shape[-1], (1,))
            batch_dict = dict(
                seq=self.data[index, :, chnl], t=self.time_idx[index], chnl_id=chnl
            )
        else:
            chnl = ...
            batch_dict = dict(seq=self.data[index, :, :], t=self.time_idx[index])

        if self.cond_shape is not None:
            batch_dict["c"] = self.cond[index, ..., chnl]
        if self.coeffs is not None:
            batch_dict["coeffs"] = self.coeffs[index, :, chnl]
        return batch_dict

    def __len__(self):
        return len(self.data)
