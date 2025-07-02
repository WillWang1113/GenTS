from argparse import Namespace
from typing import Dict, List

import torch

from gents.model.base import BaseModel

from ._backbones import Denoiser, NSFormer
from ._utils import p_sample_loop, q_sample


class TMDM(BaseModel):
    """Transformer-Modulated Diffusion Models (`TMDM <https://openreview.net/pdf?id=qae04YACHs>`_) for Probabilistic Multivariate Time Series Forecasting

    Adapted from the `official codes <https://github.com/LiYuxin321/TMDM>`_
    
    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to "predict".
        emb_add_pos (bool, optional): Whether use position embedding of time series. Defaults to True.
        emb_add_temporal (bool, optional): Whether use time features of time series. Defaults to False.
        freq (str, optional): Time series frequency. Only effective when `emb_add_temporal=True`. Defaults to None.
        emb_temporal_type (str, optional): Time features embedding type. Choose from [None, 'timeF', 'fixed']. Only effective when `emb_add_temporal=True`.  Defaults to None.
        k_cond (float, optional): Weight coefficient of condition loss. Defaults to 1.0.
        k_z (float, optional): Weight coefficient of KL loss. Defaults to 1e-2.
        d_model (int, optional): Model size. Defaults to 512.
        n_heads (int, optional): Attention heads. Defaults to 8.
        e_layers (int, optional): Encoder layers. Defaults to 2.
        d_layers (int, optional): Decoder layers. Defaults to 1.
        d_ff (int, optional): Feedforward network size. Defaults to 2048.
        factor (float, optional): Factor in attention. Seems to be useless. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.05.
        activation (str, optional): Activation in transformer. Defaults to "gelu".
        p_hidden_dims (List[int], optional): Project hidden sizes in NSformer. Defaults to [64, 64].
        p_hidden_layers (int, optional): Project hidden layers in NSformer. Defaults to 2.
        CART_input_x_embed_dim (int, optional): Denoiser model size. Defaults to 32.
        n_diff_steps (int, optional): Total diffusion steps. Defaults to 100.
        beta_schedule (str, optional): Diffusion noise schedule. Choose from `['linear', 'const', 'quad', 'jsd', 'sigmoid', 'cosine', 'cosine_reverse', 'cosine_anneal']`. Defaults to "linear".
        beta_start (float, optional): First step noise schedule. Defaults to 1e-4.
        beta_end (float, optional): Last step noise schedule. Defaults to 2e-2.
        cat_x (bool, optional): Whether to concat look back window in the Denoisor when `cat_y_pred=False`. Defaults to True.
        cat_y_pred (bool, optional): Whether to concat NSformer prediction in the Denoisor. Defaults to True.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 0.0.
    """

    ALLOW_CONDITION = ["predict"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = "predict",
        emb_add_pos: bool = True,
        emb_add_temporal: bool = False,
        freq: str = None,
        emb_temporal_type: str = None,
        k_cond: float = 1.0,
        k_z: float = 1e-2,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        factor: float = 3,
        dropout: float = 0.05,
        activation: str = "gelu",
        p_hidden_dims: List[int] = [64, 64],
        p_hidden_layers: int = 2,
        CART_input_x_embed_dim: int = 32,
        n_diff_steps: int = 100,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cat_x: bool = True,
        cat_y_pred: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        self.condition = "predict"
        obs_len = self.obs_len
        args = Namespace(**self.hparams_initial)
        args.seq_len = obs_len
        args.pred_len = seq_len
        # ! label_len default half obs_len start
        args.label_len = obs_len // 2
        args.enc_in = seq_dim
        args.dec_in = seq_dim
        args.c_out = seq_dim
        args.output_attention = False
        args.timesteps = n_diff_steps
        args.embed = emb_temporal_type
        args.features = "S" if seq_dim == 1 else "M"

        self.args = args
        self.cond_pred_model = NSFormer(args)
        self.model = Denoiser(args)

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        self.args.n_z_samples = n_sample
        self.args.n_z_samples_depart = kwargs.get("n_times", n_sample)
        self.args.test_batch_size = condition.shape[0]

        def store_gen_y_at_step_t(config, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = self.model.num_timesteps - idx
            gen_y = (
                y_tile_seq[idx].reshape(
                    config.test_batch_size,
                    int(config.n_z_samples / config.n_z_samples_depart),
                    (config.label_len + config.pred_len),
                    config.c_out,
                )
                # .cpu()
                # .numpy()
            )
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = torch.concat(
                    [gen_y_by_batch_list[current_t], gen_y], dim=0
                )
            return gen_y

        gen_y_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]

        batch_x = condition
        batch_y = kwargs.get("seq")

        batch_x_mark = kwargs.get("obs_mark", torch.zeros_like(batch_x))
        batch_y_mark = kwargs.get("seq_mark", torch.zeros_like(batch_y))

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
        _, y_0_hat_batch, _, z_sample = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark
        )

        repeat_n = int(self.args.n_z_samples / self.args.n_z_samples_depart)
        y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
        y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
        y_T_mean_tile = y_0_hat_tile
        x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
        x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

        x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
        x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

        gen_y_box = []
        for _ in range(self.args.n_z_samples_depart):
            y_tile_seq = p_sample_loop(
                self.model,
                x_tile,
                x_mark_tile,
                y_0_hat_tile,
                y_T_mean_tile,
                self.model.num_timesteps,
                self.model.alphas,
                self.model.one_minus_alphas_bar_sqrt,
            )

            gen_y = store_gen_y_at_step_t(
                config=self.args,
                # config_diff=self.model.diffusion_config,
                idx=self.model.num_timesteps,
                y_tile_seq=y_tile_seq,
            )
            gen_y_box.append(gen_y)
        outputs = torch.concat(gen_y_box, dim=1)

        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, :, -self.args.pred_len :, f_dim:]
        # shape: [bs, n_samples, pred_len, dims]
        outputs = outputs.permute(0, 2, 3, 1)
        # shape: [bs, pred_len, dims, n_samples]

        return outputs

    def _get_loss(self, batch):
        batch_x = batch["c"]
        batch_y = batch["seq"][:, self.args.obs_len - self.args.label_len :]

        # ! add time features or not?
        batch_x_mark = batch.get("obs_mark", None)
        batch_y_mark = batch.get("seq_mark", None)

        if self.args.emb_add_temporal:
            assert batch_x_mark is not None
            assert batch_y_mark is not None

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark
        )

        # loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))
        loss_vae = torch.nn.functional.mse_loss(y_0_hat_batch, batch_y)
        # loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))

        loss_vae_all = loss_vae + self.args.k_z * KL_loss
        # y_0_hat_batch = z_sample

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)

        y_t_batch = q_sample(
            batch_y,
            y_T_mean,
            self.model.alphas_bar_sqrt,
            self.model.one_minus_alphas_bar_sqrt,
            t,
            noise=e,
        )
        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

        loss = (
            e[:, -self.args.pred_len :, :] - output[:, -self.args.pred_len :, :]
        ).square().mean() + self.args.k_cond * loss_vae_all

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        loss = self._get_loss(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.cond_pred_model.parameters()},
            ],
            lr=self.hparams.lr,
        )
