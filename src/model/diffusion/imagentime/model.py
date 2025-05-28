from argparse import Namespace
import torch
from contextlib import contextmanager

from src.model.base import BaseModel
from ._backbones import EDMPrecond
from ._layers import STFTEmbedder, DelayEmbedder, LitEma
from ._sampler import DiffusionProcess


class ImagenTime(BaseModel):
    ALLOW_CONDITION = [None, "predict", "impute"]

    def __init__(
        self,
        seq_len,
        seq_dim,
        missing_rate=0,
        condition=None,
        n_diff_steps=18,
        d_model=128,
        use_stft=False,
        delay=3,
        embedding=8,
        n_fft=101,
        hop_length=25,
        ch_mult=[1, 2, 2, 2],
        attn_resolution=[8, 4, 2],
        beta1=1e-5,
        betaT=1e-2,
        lr=1e-4,
        weight_decay=1e-5,
        ema=True,
        ema_warmup=100,
        deterministic_sampling=True,
        **kwargs,
    ):
        """
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        """

        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        
        if self.condition == "impute":
            assert missing_rate > 0, "mask_rate must be greater than 0 for imputation"
        
        if self.condition is not None:
            assert not use_stft, "STFT embedding is not supported for imputation or prediction"
            
        args = Namespace(**self.hparams_initial)
        args.obs_len = self.obs_len if self.condition == "predict" else 0
        args.diffusion_steps = n_diff_steps
        args.input_channels = seq_dim
        args.deterministic = deterministic_sampling
        args.mask_rate = missing_rate
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        # self.device = device

        # delay embedding is used
        if not args.use_stft:
            self.delay = args.delay
            self.embedding = args.embedding
            self.seq_len = args.seq_len

            # NOTE: added this
            self.ts_img = DelayEmbedder(args.seq_len, args.delay, args.embedding)
            args.img_resolution = args.embedding

        else:
            args.img_resolution = round((args.n_fft + 1) // 2)
            args.input_channels = 2 * args.input_channels
            # assert args.img_resolution == (args.seq_len // args.hop_length + 1)
            self.ts_img = STFTEmbedder(args.seq_len, args.n_fft, args.hop_length)

        self.net = EDMPrecond(
            args.img_resolution,
            args.input_channels,
            channel_mult=args.ch_mult,
            model_channels=args.d_model,
            attn_resolutions=args.attn_resolution,
        )

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(
                self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup
            )
        else:
            self.use_ema = False

        self.args = args

    def ts_to_img(self, signal, pad_val=None):
        """
        Args:
            signal: signal to convert to image
            pad_val: value to pad the image with, if delay embedding is used. Do not use for STFT embedding

        """
        # pad_val is used only for delay embedding, as the value to pad the image with
        # when creating the mask, we need to use 1 as padding value
        # if pad_val is given, it is used to overwrite the default value of 0
        return (
            self.ts_img.ts_to_img(signal, True, pad_val)
            if pad_val
            else self.ts_img.ts_to_img(signal)
        )

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    # init the min and max values for the STFTEmbedder, this function must be called before the training loop starts
    def init_stft_embedder(self, train_loader):
        """
        Args:
            train_loader: training data

        caches min and max values for the real and imaginary parts
        of the STFT transformation, which will be used for normalization.
        """
        assert type(self.ts_img) == STFTEmbedder, (
            "You must use the STFTEmbedder to initialize the min and max values"
        )
        data = []
        for i, data_batch in enumerate(train_loader):
            data.append(data_batch['seq'])
        self.ts_img.cache_min_max_params(torch.cat(data, dim=0))

    def loss_fn(self, x):
        """
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        """

        # to_log = {}

        output, weight = self.forward(x)

        # denoising matching term
        # loss = weight * ((output - x) ** 2)
        loss = (weight * (output - x).square()).mean()
        # to_log["karras loss"] = loss.detach().item()

        return loss

    def loss_fn_impute(self, x, mask):
        """
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        """

        # to_log = {}
        output, weight = self.forward_impute(x, mask)
        x = self.unpad(x * (1 - mask), x.shape)
        output = self.unpad(output * (1 - mask), x.shape)
        loss = (weight * (output - x).square()).mean()
        # to_log["karras loss"] = loss.detach().item()

        return loss

    def forward(self, x, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma
        D_yn = self.net(y + n, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight

    def forward_impute(self, x, mask, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # noisy impute part
        n = torch.randn_like(x) * sigma
        noise_impute = n * (1 - mask)
        x_to_impute = x * (1 - mask) + noise_impute

        # clear image
        x = x * mask
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)

        D_yn = self.net(y + x_to_impute, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight

    def forward_forecast(self, past, future, labels=None, augment_pipe=None):
        s, e = past.shape[-1], future.shape[-1]
        rnd_normal = torch.randn([past.shape[0], 1, 1, 1], device=past.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(past) if augment_pipe is not None else (past, None)
        )
        n = torch.randn_like(future) * sigma
        full_seq = self.pad_f(torch.cat([past, future + n], dim=-1))
        D_yn = self.net(full_seq, sigma, labels, augment_labels=augment_labels)[
            ..., s : (s + e)
        ]
        return D_yn, weight

    def pad_f(self, x):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(32, rows)
        padding = (
            0,
            max_side - rows,
            0,
            0,
        )  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode="constant", value=0)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to temporarily switch to EMA weights during inference.
        Args:
            context: some string to print when switching to EMA weights

        Returns:

        """
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_before_zero_grad(self, optimizer):
        if self.use_ema:
            self.model_ema(self.net)

    # def on_train_batch_end(self, *args):
    #     """
    #     this function updates the EMA model, if it is used
    #     Args:
    #         *args:

    #     Returns:

    #     """
    #     if self.use_ema:
    #         self.model_ema(self.net)

    def training_step(self, batch, batch_idx):
        if self.condition is None:
            x_ts = batch["seq"]
            x_img = self.ts_to_img(x_ts)
            loss = self.loss_fn(x_img)
        else:
            x_ts = batch["seq"]
            if self.condition == "impute":
                # --- generate random mask and mask x as it time series --- #
                B, T, N = x_ts.shape
                mask_ts = torch.rand((B, T, N)).to(x_ts.device)
                mask_ts[mask_ts <= self.args.mask_rate] = 0  # masked
                mask_ts[mask_ts > self.args.mask_rate] = 1  # remained
            else:
                mask_ts = torch.zeros_like(x_ts)
                mask_ts[:, : self.hparams.obs_len] = 1.0

            # transform to image
            x_ts_img = self.ts_to_img(x_ts)
            # pad mask with 1
            mask_ts_img = self.ts_to_img(mask_ts, pad_val=1)
            loss = self.loss_fn_impute(x_ts_img, mask_ts_img)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.args.device = self.device
        process = DiffusionProcess(
            self.args,
            self.net,
            (
                self.args.input_channels,
                self.args.img_resolution,
                self.args.img_resolution,
            ),
        )

        if self.condition is None:
            x_ts = batch["seq"]
            x_img = self.ts_to_img(x_ts)
            loss = self.loss_fn(x_img)
        else:
            x_ts = batch["seq"]
            if self.condition == "impute":
                # --- generate random mask and mask x as it time series --- #
                mask_ts = ~torch.isnan(batch['c'])
                mask_ts = mask_ts.float()
                # mask_ts = (~batch["c"]).float()
            else:
                mask_ts = torch.zeros_like(x_ts)
                mask_ts[:, : self.hparams.obs_len] = 1.0

            # transform to image
            x_ts_img = self.ts_to_img(x_ts)
            # pad mask with 1
            mask_ts_img = self.ts_to_img(mask_ts, pad_val=1)

            # sample from the model
            # and impute, both interpolation and extrapolation are similar just the mask is different
            x_img_sampled = process.interpolate(x_ts_img, mask_ts_img).to(
                x_ts_img.device
            )
            x_ts_sampled = self.img_to_ts(x_img_sampled)
            loss = torch.nn.functional.mse_loss(
                x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0]
            )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        process = DiffusionProcess(
            self.args,
            self.net,
            (
                self.args.input_channels,
                self.args.img_resolution,
                self.args.img_resolution,
            ),
        )
        if self.condition is None:
            x_img_sampled = process.sampling(sampling_number=n_sample)
            # --- convert to time series --
            all_ts_samples = self.img_to_ts(x_img_sampled)
        else:
            x_ts = kwargs["seq"]
            if self.condition == "impute":
                # --- generate random mask and mask x as it time series --- #
                mask_ts = ~torch.isnan(condition)
                mask_ts = mask_ts.float()
                # mask_ts = (~condition).float()
            else:
                mask_ts = torch.zeros_like(x_ts)
                mask_ts[:, : self.hparams.obs_len] = 1.0

            # transform to image
            x_ts_img = self.ts_to_img(x_ts)
            # pad mask with 1
            mask_ts_img = self.ts_to_img(mask_ts, pad_val=1)

            # sample from the model
            # and impute, both interpolation and extrapolation are similar just the mask is different
            all_ts_samples = []
            for i in range(n_sample):
                x_img_sampled = process.interpolate(x_ts_img, mask_ts_img).to(
                    x_ts_img.device
                )
                x_ts_sampled = self.img_to_ts(x_img_sampled)
                all_ts_samples.append(x_ts_sampled)
            all_ts_samples = torch.stack(all_ts_samples, dim=-1)

        return all_ts_samples[:, -self.args.seq_len :]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def on_fit_start(self):
        if self.args.use_stft:
            self.init_stft_embedder(self.trainer.datamodule.train_dataloader())
