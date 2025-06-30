from gents.model.base import BaseModel
from ._backbones import LatentSDENet, LinearScheduler
import torch


class LatentSDE(BaseModel):
    ALLOW_CONDITION = [None]

    def __init__(
        self,
        seq_len,
        seq_dim,
        condition=None,
        latent_size=4,
        context_size=64,
        hidden_size=128,
        lr=1e-2,
        lr_gamma=0.997,
        noise_std=1e-2,
        solver="euler",
        adjoint=False,
        kl_anneal_iters=1000,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.total_seq_len = seq_len

        # # put checking_fn into BaseModel
        # if condition == "predict":
        #     # self.obs_len = kwargs.get("obs_len", None)
        #     # assert self.obs_len is not None
        #     # assert self.obs_len > 0
        #     self.total_seq_len += self.obs_len
        input_dim = seq_dim
        # input_dim = seq_dim * 2 if self.condition == "impute" else seq_dim

        self.net = LatentSDENet(
            input_dim, latent_size, context_size, hidden_size, output_size=seq_dim
        )
        self.kl_scheduler = LinearScheduler(iters=kl_anneal_iters)
        self.noise_std = noise_std
        self.adjoint = adjoint
        self.solver = solver

    def _get_loss(self, batch):
        t = torch.linspace(
            10.0 / self.total_seq_len,
            10.0,
            self.total_seq_len,
        ).to(batch["seq"].device)

        
        xs = batch["seq"]
        tp_to_predict = t
        xs_target = batch["seq"]
        flip = True
        
        # if self.condition == "predict":
        #     xs = batch["c"]
        #     tp_to_predict = t[self.obs_len :]
        #     xs_target = batch["seq"][:, -self.seq_len :]
        #     flip = False
        # elif self.condition == "impute":
        #     mask = torch.isnan(batch["c"])
        #     mask = (~mask).int()
        #     masked_data = mask * batch["seq"]
        #     xs = torch.concat([masked_data, mask], dim=-1)
        #     tp_to_predict = t
        #     xs_target = batch["seq"]
        #     flip = True
        # else:
        #     xs = batch["seq"]
        #     tp_to_predict = t
        #     xs_target = batch["seq"]
        #     flip = True

        preds, kld = self.net(
            xs, tp_to_predict, self.noise_std, self.adjoint, self.solver, flip=flip
        )
        preds = preds.permute(1, 0, 2)
        xs_dist = torch.distributions.Normal(loc=preds, scale=self.noise_std)

        log_pxs = xs_dist.log_prob(xs_target).sum(dim=(0, 2)).mean(dim=0)

        # loss = -log_pxs + kld * self.kl_scheduler.val

        return -log_pxs, kld * self.kl_scheduler.val

    def training_step(self, batch, batch_idx):
        rec_loss, kl_loss = self._get_loss(batch)
        loss = rec_loss + kl_loss

        self.log_dict({"loss": loss, "rec_loss": rec_loss, "kl_loss": kl_loss})
        return loss

    def validation_step(self, batch, batch_idx):
        rec_loss, kl_loss = self._get_loss(batch)
        loss = rec_loss + kl_loss
        self.log_dict({"val_loss": loss, "rec_loss": rec_loss, "kl_loss": kl_loss})
        return loss

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        t = torch.linspace(
            10.0 / self.total_seq_len,
            10.0,
            self.total_seq_len,
        ).to(self.device)

        trajs = self.net.sample(n_sample, t)
        trajs = trajs.permute(1, 0, 2)
        # if self.condition is None:
            # trajs = self.net.sample(n_sample, t)
            # trajs = trajs.permute(1, 0, 2)
        # else:
        #     if self.condition == "predict":
        #         xs = condition
        #         tp_to_predict = t[self.obs_len :]
        #         flip = False
        #     elif self.condition == "impute":
        #         condition = torch.isnan(condition)
        #         mask = (~condition).int()
        #         masked_data = mask * kwargs["seq"]
        #         xs = torch.concat([masked_data, mask], dim=-1)
        #         tp_to_predict = t
        #         flip = True

        #     if flip:
        #         ctx = self.net.encoder(torch.flip(xs, dims=(1,)))
        #     else:
        #         ctx = self.net.encoder(xs)

        #     ctx = torch.flip(ctx, dims=(1,))
        #     self.net.contextualize((tp_to_predict, ctx))
        #     qz0_mean, qz0_logstd = self.net.qz0_net(ctx[:, 0]).chunk(chunks=2, dim=1)

        #     trajs = []
        #     for _ in range(n_sample):
        #         z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        #         zs = torchsde.sdeint(
        #             self.net,
        #             z0,
        #             tp_to_predict,
        #             names={"drift": "h"},
        #             dt=1e-3,
        #             method=self.solver,
        #         )
        #         trajs.append(self.net.projector(zs))
        #     trajs = torch.stack(trajs, dim=-1)
        #     trajs = trajs.permute(1, 0, 2, 3)
        return trajs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.hparams.lr)
        scheduler = torch.torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=self.hparams.lr_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
