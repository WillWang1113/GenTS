from argparse import Namespace
from typing import List
from gents.model.base import BaseModel
from ._backbones import VAE
import torch


class LS4(BaseModel):
    """`Deep Latent State Space Models for Time-Series Generation <https://arxiv.org/abs/2212.12749>`__

    Adapted from the `official codes <https://github.com/alexzhou907/ls4>`__

    .. note::
        LS4 allows for irregular data. Imputation in LS4 is only interpolation, 
        i.e. only impute the missing steps that all channels are missing.
    
    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        latent_dim (int, optional): Latent dimension. Defaults to 10.
        bidirectional (bool, optional): Whether to be bidirectional. Defaults to False.
        sigma (float, optional): Std of generated data. Seems to be useless Defaults to 0.1.
        enc_hidden_size (int, optional): Encoder hidden size. Defaults to 64.
        enc_n_layers (int, optional): Encoder layers. Defaults to 4.
        enc_backbone (str, optional): Encoder backbone. Choose from `['autoreg', 'seq']`. Defaults to "autoreg".
        enc_use_unet (bool, optional): If `True`, encoder will use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling.
            All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
            We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
            for non-autoregressive models. Defaults to False.
        enc_pool (List[int], optional): Pooling factor at each level. Pooling shrinks the sequence length at lower levels.
            We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
            It's possible that a different combination of pooling factors and number of tiers may perform better. Defaults to [].
        enc_ff_layers (int, optional): Expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4). Defaults to 2.
        enc_expand (int, optional): Expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
            We generally found 2 to perform best (among 2, 4). Defaults to 2.
        enc_s4type (str, optional): S4 network type for encoder. Choose from `['s4', 's4d', 's4d_joint']`. Defaults to "s4".
        enc_dropout (float, optional): Encoder dropout. Defaults to 0.0.
        enc_use_latent (bool, optional): Whether to use latent variables. Defaults to True.
        enc_latent_type (str, optional): If `enc_use_latent=True`, choose from ['none', 'split', 'const_std', 'single', 'joint']. Defaults to "split".
        enc_lr (float, optional): Encoder learning rate. Defaults to 1e-3.
        dec_activation (str, optional): Decoder activation type. Choose from `[None, 'id', 'identity', 'linear', 'tanh', 'relu', 'gelu', 'swish', 'silu', 'glu', 'sigmoid', 'modrelu']` Defaults to "identity".
        dec_hidden_size (int, optional): Decoder hidden size. Defaults to 64.
        dec_n_layers (int, optional): Decoder layers. Defaults to 4.
        dec_backbone (str, optional): Decoder backbone. Choose from `['autoreg', 'seq']`. Defaults to "autoreg".
        dec_use_unet (bool, optional): If `True`, encoder will use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling.
            All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
            We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
            for non-autoregressive models. Defaults to False.
        dec_pool (List[int], optional): Pooling factor at each level. Pooling shrinks the sequence length at lower levels.
            We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
            It's possible that a different combination of pooling factors and number of tiers may perform better. Defaults to [].
        dec_ff_layers (int, optional): Expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4). Defaults to 2.
        dec_expand (int, optional): Expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
            We generally found 2 to perform best (among 2, 4). Defaults to 2.
        dec_s4type (str, optional): S4 network type for encoder. Choose from `['s4', 's4d', 's4d_joint']`. Defaults to "s4".
        dec_dropout (float, optional): Decoder dropout. Defaults to 0.0.
        dec_use_latent (bool, optional): Whether to use latent variable. Defaults to False.
        dec_latent_type (str, optional): If `enc_use_latent=True`, choose from ['none', 'split', 'const_std', 'single', 'joint']. Defaults to "none".
        dec_lr (float, optional): Decoder learning rate. Defaults to 1e-3.
        prior_hidden_size (int, optional): Prior net hidden size. Defaults to 64.
        prior_n_layers (int, optional): Prior net layers. Defaults to 4.
        prior_backbone (str, optional): Prior net backbone. Choose from `['autoreg', 'seq']`. Defaults to "autoreg".
        prior_use_unet (bool, optional): If `True`, encoder will use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling.
            All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
            We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
            for non-autoregressive models. Defaults to False.
        prior_pool (List[int], optional): Pooling factor at each level. Pooling shrinks the sequence length at lower levels.
            We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
            It's possible that a different combination of pooling factors and number of tiers may perform better. Defaults to [].
        prior_ff_layers (int, optional): Expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4). Defaults to 2.
        prior_expand (int, optional): Expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
            We generally found 2 to perform best (among 2, 4). Defaults to 2.
        prior_s4type (str, optional): S4 network type for encoder. Choose from `['s4', 's4d', 's4d_joint']`. Defaults to "s4".
        prior_dropout (float, optional): Prior net dropout rate. Defaults to 0.0.
        prior_use_latent (bool, optional): Whether to use latent variable.. Defaults to True.
        prior_latent_type (str, optional): If `enc_use_latent=True`, choose from ['none', 'split', 'const_std', 'single', 'joint']. Defaults to "split".
        prior_lr (float, optional): Prior net learning rate. Defaults to 1e-3.
        lr (float, optional): Learning rate for other parameters. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.

    """

    ALLOW_CONDITION = [None, "predict", "impute"]
    # Interpolation only, i.e. missing time steps

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int = 10,
        bidirectional: bool = False,
        condition: str = None,
        sigma: float = 0.1,
        # enc_use_spatial: bool = True,
        enc_hidden_size: int = 64,
        enc_n_layers: int = 4,
        enc_backbone: str = "autoreg",
        enc_use_unet: bool = False,
        enc_pool: List[int] = [],
        enc_ff_layers: int = 2,
        enc_expand: int = 2,
        enc_s4type: str = "s4",
        enc_dropout: float = 0.0,
        enc_use_latent: bool = True,
        enc_latent_type: str = "split",
        enc_lr: float = 1e-3,
        # dec_use_spatial: bool = True,
        dec_activation: str = "identity",
        dec_hidden_size: int = 64,
        dec_n_layers: int = 4,
        dec_backbone: str = "autoreg",
        dec_use_unet: bool = False,
        dec_pool: List[int] = [],
        dec_ff_layers: int = 2,
        dec_expand: int = 2,
        dec_s4type: str = "s4",
        dec_dropout: float = 0.0,
        dec_use_latent: bool = False,
        dec_latent_type: str = "none",
        dec_lr: float = 1e-3,
        prior_hidden_size: int = 64,
        prior_n_layers: int = 4,
        prior_backbone: str = "autoreg",
        prior_use_unet: bool = False,
        prior_pool: List[int] = [],
        prior_ff_layers: int = 2,
        prior_expand: int = 2,
        prior_s4type: str = "s4",
        prior_dropout: float = 0.0,
        prior_use_latent: bool = True,
        prior_latent_type: str = "split",
        prior_lr: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()

        encoder_config = Namespace(
            use_spatial=True,
            posterior=dict(
                d_input=seq_dim,
                d_output=latent_dim,
                aux_channels=0,
                d_state=enc_hidden_size,
                d_model=enc_hidden_size,
                n_layers=enc_n_layers,
                backbone=enc_backbone,
                use_unet=enc_use_unet,
                pool=enc_pool,
                ff=enc_ff_layers,
                expand=enc_expand,
                bidirectional=bidirectional,
                dropout=enc_dropout,
                s4_type=enc_s4type,
                use_latent=enc_use_latent,
                latent_type=enc_latent_type,
                lr=enc_lr,
            ),
        )
        decoder_config = Namespace(
            use_spatial=True,
            activation=dec_activation,
            decoder=dict(
                d_input=latent_dim,
                d_output=seq_dim,
                aux_channels=0,
                d_state=dec_hidden_size,
                d_model=dec_hidden_size,
                n_layers=dec_n_layers,
                backbone=dec_backbone,
                use_unet=dec_use_unet,
                pool=dec_pool,
                ff=dec_ff_layers,
                expand=dec_expand,
                bidirectional=bidirectional,
                dropout=dec_dropout,
                s4_type=dec_s4type,
                use_latent=dec_use_latent,
                latent_type=dec_latent_type,
                lr=dec_lr,
            ),
            prior=dict(
                d_input=latent_dim,
                d_output=latent_dim,
                aux_channels=0,
                d_state=prior_hidden_size,
                d_model=prior_hidden_size,
                n_layers=prior_n_layers,
                backbone=prior_backbone,
                use_unet=prior_use_unet,
                pool=prior_pool,
                ff=prior_ff_layers,
                expand=prior_expand,
                bidirectional=bidirectional,
                dropout=prior_dropout,
                s4_type=prior_s4type,
                use_latent=prior_use_latent,
                latent_type=prior_latent_type,
                lr=prior_lr,
            ),
        )
        config = Namespace(
            z_dim=latent_dim,
            bidirectional=bidirectional,
            sigma=sigma,
            in_channels=seq_dim,
            encoder=encoder_config,
            decoder=decoder_config,
        )
        self.model = VAE(config)
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        if condition == "predict":
            self.obs_len = kwargs.get("obs_len")
            assert self.obs_len > 0

    def training_step(self, batch, batch_idx):
        data, tp, mask, labels = self._reform_batch(batch)
        loss, loss_dict = self.model(data, tp, mask, labels)
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, tp, mask, labels = self._reform_batch(batch)
        loss, loss_dict = self.model(data, tp, mask, labels)
        loss_dict = {"val_" + key: value for key, value in loss_dict.items()}
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay
        # All parameters in the model
        all_parameters = list(self.model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s)
            for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": params, **hp})

        # Create a lr scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs
        )

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(
                " | ".join(
                    [
                        f"Optimizer group {i}",
                        f"{len(g['params'])} tensors",
                    ]
                    + [f"{k} {v}" for k, v in group_hps.items()]
                )
            )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ! TODO : device inaligment somewhere
    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        self.model.setup_rnn()

        if self.condition == "predict":
            total_seq_len = self.seq_len + self.obs_len
            t = kwargs.get("t", torch.arange(total_seq_len).float().to(self.device))

            tp_to_predict = t[self.obs_len :]
            observed_tp = t[: self.obs_len]
            observed_data = torch.nan_to_num(condition)
            mask_predicted_data = kwargs["data_mask"][:, self.obs_len :]
            # mask_predicted_data = torch.ones(
            #     [observed_data.shape[0], self.seq_len, self.seq_dim]
            # ).to(observed_data)

            all_samples = []
            for _ in range(n_sample):
                trajs = self.model.reconstruct(
                    observed_data,
                    observed_tp,
                    tp_to_predict,
                    masks=mask_predicted_data,
                    get_full_nll=False,
                )
                all_samples.append(trajs)
            all_samples = torch.stack(all_samples, dim=-1)

        elif self.condition == "impute":
            # data = kwargs.get("seq").clone()
            # condition = torch.isnan(condition)
            # data[condition] = 0.0
            # total_seq_len = self.seq_len
            t = kwargs["t"][0]

            mask = ~torch.isnan(condition)
            mask = mask.float()
            observed_data = torch.nan_to_num(condition)

            non_missing_tp = torch.sum(mask, (0, 2)) != 0.0
            observed_data = observed_data[:, non_missing_tp]
            observed_tp = t[non_missing_tp]

            tp_to_predict = t
            # observed_tp = t
            mask_predicted_data = kwargs.get("data_mask")

            all_samples = []
            for _ in range(n_sample):
                trajs = self.model.reconstruct(
                    observed_data,
                    observed_tp,
                    tp_to_predict,
                    masks=mask_predicted_data,
                    get_full_nll=False,
                )
                all_samples.append(trajs)
            all_samples = torch.stack(all_samples, dim=-1)
        else:
            all_samples = self.model.generate(
                n_sample, self.seq_len, device=self.device
            )
        return all_samples

    def _reform_batch(self, batch):
        data = batch["seq"].clone()
        tp = batch["t"][0].clone()
        # print(tp[:10])
        mask = batch["data_mask"].float()
        labels = None

        non_missing_tp = torch.sum(mask, (0, 2)) != 0.0
        data = data[:, non_missing_tp]
        tp = tp[non_missing_tp]
        mask = mask[:, non_missing_tp]
        # print(tp[:10])

        # if (self.condition is None) or (self.condition == 'impute'):
        #     data = batch["seq"].clone()
        #     tp = batch['t'][0].clone()
        #     mask = batch["data_mask"].float()
        #     labels = None

        #     non_missing_tp = torch.sum(mask, (0, 2)) != 0.
        #     data = data[:, non_missing_tp]
        #     tp = tp[non_missing_tp]

        # elif self.condition == "predict":
        #     data = batch["seq"]
        #     tp = batch['t'][0]
        #     # tp = torch.linspace(20.0 / data.shape[1], 20.0, data.shape[1]).to(data)
        #     mask = torch.ones_like(batch["seq"])
        #     mask[:, : self.obs_len] = 0.0

        #     labels = None

        #     non_missing_tp = torch.sum(mask, (0, 2)) != 0.
        #     data = data[:, non_missing_tp]
        #     tp = tp[non_missing_tp]

        # elif self.condition == "impute":
        #     data = batch['seq'].clone()
        #     mask = batch['data_mask'].float()
        #     # data = torch.nan_to_num(batch["c"])
        #     # mask = torch.isnan(batch["c"])
        #     # mask = 1 - mask.float()
        #     # data = batch["seq"].clone()
        #     # mask = batch["c"]
        #     # data[mask] = 0.0
        #     tp = batch['t'][0].clone()
        #     # tp = torch.linspace(20.0 / data.shape[1], 20.0, data.shape[1]).to(data)
        #     labels = None

        #     non_missing_tp = torch.sum(mask, (0, 2)) != 0.
        #     data = data[:, non_missing_tp]
        #     tp = tp[non_missing_tp]

        return data, tp, mask, labels
