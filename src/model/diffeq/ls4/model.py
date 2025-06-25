from argparse import Namespace
from src.model.base import BaseModel
from ._backbones import VAE
import torch

class LS4(BaseModel):
    ALLOW_CONDITION = [None, "predict", "impute"]
    # Interpolation only, i.e. missing time steps

    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim=10,
        bidirectional=False,
        condition=None,
        sigma=0.1,
        enc_use_spatial=True,
        enc_hidden_size=64,
        enc_n_layers=4,
        enc_backbone="autoreg",
        enc_use_unet=False,
        enc_pool=[],
        enc_ff_layers=2,
        enc_expand=2,
        enc_s4type="s4",
        enc_dropout=0.0,
        enc_use_latent=True,
        enc_latent_type="split",
        enc_lr=1e-3,
        dec_use_spatial=True,
        dec_activation="identity",
        dec_hidden_size=64,
        dec_n_layers=4,
        dec_backbone="autoreg",
        dec_use_unet=False,
        dec_pool=[],
        dec_ff_layers=2,
        dec_expand=2,
        dec_s4type="s4",
        dec_dropout=0.0,
        dec_use_latent=False,
        dec_latent_type="none",
        dec_lr=1e-3,
        prior_hidden_size=64,
        prior_n_layers=4,
        prior_backbone="autoreg",
        prior_use_unet=False,
        prior_pool=[],
        prior_ff_layers=2,
        prior_expand=2,
        prior_s4type="s4",
        prior_dropout=0.0,
        prior_use_latent=True,
        prior_latent_type="split",
        prior_lr=1e-3,
        lr=1e-3,
        weight_decay=1e-5,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()

        encoder_config = Namespace(
            use_spatial=enc_use_spatial,
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
            use_spatial=dec_use_spatial,
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
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        data, tp, mask, labels = self._reform_batch(batch)
        loss, loss_dict = self.model(data, tp, mask, labels)
        loss_dict = {"val" + key: value for key, value in loss_dict.items()}
        self.log_dict(loss_dict)

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
            t = kwargs['t']

            tp_to_predict = t[self.obs_len :]
            observed_tp = t[: self.obs_len]
            observed_data = torch.nan_to_num(condition)
            mask_predicted_data = kwargs['data_mask'][:,self.obs_len :]
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
            t = kwargs['t'][0]

            mask = ~torch.isnan(condition)
            mask = mask.float()
            observed_data = torch.nan_to_num(condition)
            
            non_missing_tp = torch.sum(mask, (0, 2)) != 0.
            observed_data = observed_data[:, non_missing_tp]
            observed_tp = t[non_missing_tp]
            
            tp_to_predict = t
            # observed_tp = t
            mask_predicted_data = kwargs.get('data_mask')

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
        tp = batch['t'][0].clone()
        # print(tp[:10])
        mask = batch["data_mask"].float()
        labels = None
        
        non_missing_tp = torch.sum(mask, (0, 2)) != 0.
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
