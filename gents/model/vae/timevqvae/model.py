from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from gents.model.base import BaseModel

from ._backbones import BidirectionalTransformer, VQVAEDecoder, VQVAEEncoder
from ._utils import (
    compute_downsample_rate,
    linear_warmup_cosine_annealingLR,
    time_to_timefreq,
    timefreq_to_time,
    zero_pad_high_freq,
    zero_pad_low_freq,
)
from ._vq import VectorQuantize, quantize


class TimeVQVAE(BaseModel):
    """`TimeVQVAE <https://arxiv.org/abs/2111.08095>`__ for time series generation.
    
    Adapted from the `official codes <https://github.com/ML4ITS/TimeVQVAE>`__
    
    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given conditions, should be one of `ALLOW_CONDITION`. Defaults to None.
        resnet_init_dim (int, optional): Initial d_model of resnet. Defaults to 4.
        hidden_size (int, optional): Hidden size. Defaults to 128.
        n_fft (int, optional): Size of Fourier transform. Defaults to 4.
        n_resnet_blocks (int, optional): Blocks of Resnet. Defaults to 2.
        downsampled_width_l (int, optional): Low-frequency downsampled width. Defaults to 8.
        downsampled_width_h (int, optional): High-frequency downsampled width. Defaults to 32.
        codebook_size (int, optional): VQ codebook. Defaults to 1024.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        stage_split (float, optional): Training stage splite ratio, [0, 1]. Defaults to 0.5.
        cfg_scale (float, optional): Classifier free guidance rate for conditional generation, [0, 1]. Defaults to 0.5.
        prior_model_l_config (Dict[str, Any], optional): Prior model config for low-frequency. Defaults to dict( hidden_dim=128, n_layers=4, heads=2, ff_mult=1, use_rmsnorm=True, p_unconditional=0.2, model_dropout=0.3, emb_dropout=0.3, ).
        prior_model_h_config (Dict[str, Any], optional): Prior model config for high-frequency. Defaults to dict( hidden_dim=32, n_layers=1, heads=1, ff_mult=1, use_rmsnorm=True, p_unconditional=0.2, model_dropout=0.3, emb_dropout=0.3, ).
        choice_temperatures (Dict[str, Any], optional): Temperatures for randomness for low-freq and high-freq. Defaults to {"lf": 10, "hf": 0}.
        dec_iter_step (Dict[str, int], optional): Decoding iteration steps for low-freq and high-freq. Defaults to {"lf": 10, "hf": 10}.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """
    ALLOW_CONDITION = [None, "class"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        resnet_init_dim: int = 4,
        hidden_size: int = 128,
        n_fft: int = 4,
        n_resnet_blocks: int = 2,
        downsampled_width_l: int = 8,
        downsampled_width_h: int = 32,
        codebook_size: int = 1024,
        lr: float = 1e-3,
        stage_split: float = 0.5,
        cfg_scale: float = 0.5,
        prior_model_l_config: Dict[str, Any] = dict(
            hidden_dim=128,
            n_layers=4,
            heads=2,
            ff_mult=1,
            use_rmsnorm=True,
            p_unconditional=0.2,
            model_dropout=0.3,
            emb_dropout=0.3,
        ),
        prior_model_h_config: Dict[str, Any] = dict(
            hidden_dim=32,
            n_layers=1,
            heads=1,
            ff_mult=1,
            use_rmsnorm=True,
            p_unconditional=0.2,
            model_dropout=0.3,
            emb_dropout=0.3,
        ),
        choice_temperatures: Dict[str, Any] = {"lf": 10, "hf": 0},
        dec_iter_step: Dict[str, int] = {"lf": 10, "hf": 10},
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)

        self.save_hyperparameters()
        self.automatic_optimization = False
        n_classes = 0
        if self.condition == "class":
            n_classes = self.class_num
        self.seq_len = seq_len
        self.T = dec_iter_step
        self.choice_temperatures = choice_temperatures

        self.n_fft = n_fft
        init_dim = resnet_init_dim
        hid_dim = hidden_size
        # downsampled_width_l = config['encoder']['downsampled_width']['lf']
        # downsampled_width_h = config['encoder']['downsampled_width']['hf']
        downsample_rate_l = compute_downsample_rate(
            seq_len, self.n_fft, downsampled_width_l
        )
        downsample_rate_h = compute_downsample_rate(
            seq_len, self.n_fft, downsampled_width_h
        )

        # encoder
        self.encoder_l = VQVAEEncoder(
            init_dim,
            hid_dim,
            2 * seq_dim,
            downsample_rate_l,
            n_resnet_blocks,
            "lf",
            self.n_fft,
            frequency_indepence=True,
        )
        self.encoder_h = VQVAEEncoder(
            init_dim,
            hid_dim,
            2 * seq_dim,
            downsample_rate_h,
            n_resnet_blocks,
            "hf",
            self.n_fft,
            frequency_indepence=False,
        )

        # quantizer
        self.vq_model_l = VectorQuantize(hid_dim, codebook_size, **kwargs)
        self.vq_model_h = VectorQuantize(hid_dim, codebook_size, **kwargs)

        # decoder
        self.decoder_l = VQVAEDecoder(
            init_dim,
            hid_dim,
            2 * seq_dim,
            downsample_rate_l,
            n_resnet_blocks,
            seq_len,
            "lf",
            self.n_fft,
            seq_dim,
            frequency_indepence=True,
        )
        self.decoder_h = VQVAEDecoder(
            init_dim,
            hid_dim,
            2 * seq_dim,
            downsample_rate_h,
            n_resnet_blocks,
            seq_len,
            "hf",
            self.n_fft,
            seq_dim,
            frequency_indepence=False,
        )

        with torch.no_grad():
            inp = torch.randn((1, seq_dim, seq_len))
            _ = self.encoder_l(inp)
            _ = self.encoder_h(inp)

        # token lengths
        self.num_tokens_l = self.encoder_l.num_tokens.item()
        self.num_tokens_h = self.encoder_h.num_tokens.item()

        # latent space dim
        self.H_prime_l, self.H_prime_h = (
            self.encoder_l.H_prime.item(),
            self.encoder_h.H_prime.item(),
        )
        self.W_prime_l, self.W_prime_h = (
            self.encoder_l.W_prime.item(),
            self.encoder_h.W_prime.item(),
        )

        # transformers / prior models
        self.gamma = self._gamma_func("cosine")
        emb_dim = hidden_size
        self.transformer_l = BidirectionalTransformer(
            "lf",
            self.num_tokens_l,
            codebook_size,
            emb_dim,
            **prior_model_l_config,
            n_classes=n_classes,
        )

        self.transformer_h = BidirectionalTransformer(
            "hf",
            self.num_tokens_h,
            codebook_size,
            emb_dim,
            **prior_model_h_config,
            n_classes=n_classes,
            num_tokens_l=self.num_tokens_l,
        )
        self.mask_token_ids = {"lf": codebook_size, "hf": codebook_size}

    def _gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: torch.cos(r * torch.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        elif mode == "cubic":
            return lambda r: 1 - r**3
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _encode_to_z_q(self, x, encoder, vq_model, svq_temp: Union[float, None] = None):
        """
        encode x to zq

        x: (b c l)
        """
        z = encoder(x)
        zq, s, _, _ = quantize(
            z, vq_model, svq_temp=svq_temp
        )  # (b c h w), (b (h w) h), ...
        return zq, s

    def _randomly_mask_tokens(self, s, mask_token_id, device):
        """
        s: token set
        """
        b, n = s.shape

        # sample masking indices
        ratio = torch.rand(b)  # (b,)
        n_unmasks = torch.floor(self.gamma(ratio) * n)  # (b,)
        n_unmasks = torch.clip(
            n_unmasks, min=0, max=n - 1
        ).int()  # ensures that there's at least one masked token
        rand = torch.rand((b, n), device=device)  # (b n)
        mask = torch.zeros((b, n), dtype=torch.bool, device=device)  # (b n)

        for i in range(b):
            ind = rand[i].topk(n_unmasks[i], dim=-1).indices
            mask[i].scatter_(dim=-1, index=ind, value=True)

        # mask the token set
        masked_indices = mask_token_id * torch.ones((b, n), device=device)  # (b n)
        s_M = mask * s + (~mask) * masked_indices  # (b n); `~` reverses bool-typed data
        return s_M.long(), mask

    def _masked_prediction(self, transformer, class_condition, *s_in):
        """
        masked prediction with classifier-free guidance
        """
        if isinstance(class_condition, type(None)):
            # unconditional
            logits_null = transformer(*s_in, class_condition=None)  # (b n k)
            return logits_null
        else:
            # class-conditional
            if self.hparams_initial.cfg_scale == 1.0:
                logits = transformer(*s_in, class_condition=class_condition)  # (b n k)
            else:
                # with CFG
                logits_null = transformer(*s_in, class_condition=None)
                logits = transformer(*s_in, class_condition=class_condition)  # (b n k)
                logits = logits_null + self.hparams_initial.cfg_scale * (
                    logits - logits_null
                )
            return logits

    def _loss_stage2(self, batch, batch_idx):
        """
        x: (B, C, L)
        y: (B, 1)
        straight from [https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py]
        """
        x = batch["seq"].permute(0, 2, 1)
        y = batch.get("c", None)
        if y is not None:
            if y.ndim == 1:
                y = torch.unsqueeze(y, dim=-1)  # (b 1)
        # y = batch.get("label", None)
        self.encoder_l.eval()
        self.vq_model_l.eval()
        self.encoder_h.eval()
        self.vq_model_h.eval()

        _, s_l = self._encode_to_z_q(x, self.encoder_l, self.vq_model_l)  # (b n)
        _, s_h = self._encode_to_z_q(x, self.encoder_h, self.vq_model_h)  # (b m)

        # mask tokens
        s_l_M, mask_l = self._randomly_mask_tokens(
            s_l, self.mask_token_ids["lf"], self.device
        )  # (b n), (b n) where 0 for masking and 1 for un-masking
        s_h_M, mask_h = self._randomly_mask_tokens(
            s_h, self.mask_token_ids["hf"], self.device
        )  # (b n), (b n) where 0 for masking and 1 for un-masking

        # prediction
        logits_l = self._masked_prediction(
            self.transformer_l, y, s_l_M.detach()
        )  # (b n k)
        logits_h = self._masked_prediction(
            self.transformer_h, y, s_l.detach(), s_h_M.detach()
        )

        # maksed prediction loss
        logits_l_on_mask = logits_l[~mask_l]  # (bm k) where m < n
        s_l_on_mask = s_l[~mask_l]  # (bm) where m < n
        mask_pred_loss_l = F.cross_entropy(logits_l_on_mask.float(), s_l_on_mask.long())

        logits_h_on_mask = logits_h[~mask_h]  # (bm k) where m < n
        s_h_on_mask = s_h[~mask_h]  # (bm) where m < n
        mask_pred_loss_h = F.cross_entropy(logits_h_on_mask.float(), s_h_on_mask.long())

        mask_pred_loss = mask_pred_loss_l + mask_pred_loss_h
        return mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h)

    def _loss_stage1(self, batch, batch_idx, return_x_rec: bool = False):
        """
        :param x: input time series (b c l)
        """
        # x, y = batch
        x = batch["seq"].permute(0, 2, 1)
        # y = batch.get("label", None)

        recons_loss = {"LF.time": 0.0, "HF.time": 0.0}
        vq_losses = {"LF": None, "HF": None}
        perplexities = {"LF": 0.0, "HF": 0.0}

        # STFT
        in_channels = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, in_channels)  # (b c h w)
        u_l = zero_pad_high_freq(xf)  # (b c h w)
        x_l = F.interpolate(
            timefreq_to_time(u_l, self.n_fft, in_channels),
            self.seq_len,
            mode="linear",
        )  # (b c l)
        u_h = zero_pad_low_freq(xf)  # (b c h w)
        x_h = F.interpolate(
            timefreq_to_time(u_h, self.n_fft, in_channels),
            self.seq_len,
            mode="linear",
        )  # (b c l)

        # LF
        z_l = self.encoder_l(x)
        z_q_l, s_l, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
        xhat_l = self.decoder_l(z_q_l)  # (b c l)

        # HF
        z_h = self.encoder_h(x)
        z_q_h, s_h, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
        xhat_h = self.decoder_h(z_q_h)  # (b c l)

        if return_x_rec:
            x_rec = xhat_l + xhat_h  # (b c l)
            return x_rec  # (b c l)

        recons_loss["LF.time"] = F.mse_loss(x_l, xhat_l)
        perplexities["LF"] = perplexity_l
        vq_losses["LF"] = vq_loss_l

        recons_loss["HF.time"] = F.l1_loss(x_h, xhat_h)
        perplexities["HF"] = perplexity_h
        vq_losses["HF"] = vq_loss_h

        return recons_loss, vq_losses, perplexities

    def configure_optimizers(self):
        # assert self.trainer.max_steps > 0, "Trainer max_steps must be set in TimeVQVAE"
        self.total_steps = self.trainer.max_epochs
        stage1_param = (
            list(self.encoder_h.parameters())
            + list(self.encoder_l.parameters())
            + list(self.vq_model_h.parameters())
            + list(self.vq_model_l.parameters())
            + list(self.decoder_h.parameters())
            + list(self.decoder_l.parameters())
        )
        opt1 = torch.optim.AdamW(stage1_param, lr=self.hparams_initial.lr)
        scheduler1 = linear_warmup_cosine_annealingLR(
            opt1,
            int(self.total_steps * self.hparams_initial.stage_split),
            # self.config["exp_params"]["linear_warmup_rate"],
            # min_lr=self.config["exp_params"]["min_lr"],
        )

        stage2_param = list(self.transformer_h.parameters()) + list(
            self.transformer_l.parameters()
        )
        opt2 = torch.optim.AdamW(stage2_param, lr=self.hparams_initial.lr)
        scheduler2 = linear_warmup_cosine_annealingLR(
            opt2,
            self.total_steps - int(self.total_steps * self.hparams_initial.stage_split),
            # self.config["exp_params"]["linear_warmup_rate"],
            # min_lr=self.config["exp_params"]["min_lr"],
        )

        return [opt1, opt2], [scheduler1, scheduler2]

    def training_step(self, batch, batch_idx):
        """Two-stage training for TimeVQVAE. 
        
        Stage 1: Learning Vector Quantization
        
        Stage 2: Prior Learning
        
        .. note::
            Note that TimeVQVAE is limited by `max_steps` instead of `max_epochs`.
        """
        opt1, opt2 = self.optimizers()
        sch1, sch2 = self.lr_schedulers()
        if self.global_step < int(self.total_steps * self.hparams_initial.stage_split):
            self.toggle_optimizer(opt1)
            recons_loss, vq_losses, perplexities = self._loss_stage1(batch, batch_idx)
            loss = (
                (recons_loss["LF.time"] + recons_loss["HF.time"])
                + vq_losses["LF"]["loss"]
                + vq_losses["HF"]["loss"]
            )
            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()

            # lr scheduler
            sch1.step()

            # log
            loss_hist = {
                "loss": loss,
                "recons_loss.time": recons_loss["LF.time"] + recons_loss["HF.time"],
                "recons_loss.LF.time": recons_loss["LF.time"],
                "recons_loss.HF.time": recons_loss["HF.time"],
                "commit_loss.LF": vq_losses["LF"]["commit_loss"],
                "commit_loss.HF": vq_losses["HF"]["commit_loss"],
                "perplexity.LF": perplexities["LF"],
                "perplexity.HF": perplexities["HF"],
            }

            # log
            self.log("global_step", self.global_step)
            for k in loss_hist.keys():
                self.log(f"train_{k}", loss_hist[k])

            self.untoggle_optimizer(opt1)
        else:
            self.toggle_optimizer(opt2)

            mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self._loss_stage2(
                batch,
                batch_idx,
            )

            opt2.zero_grad()
            self.manual_backward(mask_pred_loss)
            opt2.step()

            # lr scheduler
            sch2.step()

            # log
            self.log("global_step", self.global_step)
            loss_hist = {
                "loss": mask_pred_loss,
                "mask_pred_loss": mask_pred_loss,
                "mask_pred_loss_l": mask_pred_loss_l,
                "mask_pred_loss_h": mask_pred_loss_h,
            }
            for k in loss_hist.keys():
                self.log(f"train_{k}", loss_hist[k])
            self.untoggle_optimizer(opt2)

    def validation_step(self, batch, batch_idx):
        if self.global_step < int(self.total_steps * self.hparams_initial.stage_split):
            recons_loss, vq_losses, perplexities = self._loss_stage1(batch, batch_idx)
            loss = (
                (recons_loss["LF.time"] + recons_loss["HF.time"])
                + vq_losses["LF"]["loss"]
                + vq_losses["HF"]["loss"]
            )

            # log
            loss_hist = {
                "loss": loss,
                "recons_loss.time": recons_loss["LF.time"] + recons_loss["HF.time"],
                "recons_loss.LF.time": recons_loss["LF.time"],
                "recons_loss.HF.time": recons_loss["HF.time"],
                "commit_loss.LF": vq_losses["LF"]["commit_loss"],
                "commit_loss.HF": vq_losses["HF"]["commit_loss"],
                "perplexity.LF": perplexities["LF"],
                "perplexity.HF": perplexities["HF"],
            }

            # log
            self.log("global_step", self.global_step)
            for k in loss_hist.keys():
                self.log(f"val_{k}", loss_hist[k])

            # return loss_hist
        else:
            mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self._loss_stage2(
                batch,
                batch_idx,
            )

            # log
            self.log("global_step", self.global_step)
            loss_hist = {
                "loss": mask_pred_loss,
                "mask_pred_loss": mask_pred_loss,
                "mask_pred_loss_l": mask_pred_loss_l,
                "mask_pred_loss_h": mask_pred_loss_h,
            }
            for k in loss_hist.keys():
                self.log(f"val_{k}", loss_hist[k])

    def _iterative_decoding(self, num=1, mode="cosine", class_index=None, device="cpu"):
        """
        It performs the iterative decoding and samples token indices for LF and HF.
        :param num: number of samples
        :return: sampled token indices for LF and HF
        """
        s_l = self._create_input_tokens_normal(
            num, self.num_tokens_l, self.mask_token_ids["lf"], device
        )  # (b n)
        s_h = self._create_input_tokens_normal(
            num, self.num_tokens_h, self.mask_token_ids["hf"], device
        )  # (b n)

        unknown_number_in_the_beginning_l = torch.sum(
            s_l == self.mask_token_ids["lf"], dim=-1
        )  # (b,)
        unknown_number_in_the_beginning_h = torch.sum(
            s_h == self.mask_token_ids["hf"], dim=-1
        )  # (b,)
        gamma = self._gamma_func(mode)
        # class_condition = (
        #     repeat(torch.Tensor([class_index]).int().to(device), "i -> b i", b=num)
        #     if class_index is not None
        #     else None
        # )  # (b 1)
        class_condition = class_index
        s_l = self._first_pass(
            s_l, unknown_number_in_the_beginning_l, class_condition, gamma, device
        )
        s_h = self._second_pass(
            s_l, s_h, unknown_number_in_the_beginning_h, class_condition, gamma, device
        )
        return s_l, s_h

    def _create_input_tokens_normal(self, num, num_tokens, mask_token_ids, device):
        """
        returns masked tokens
        """
        blank_tokens = torch.ones((num, num_tokens), device=device)
        masked_tokens = mask_token_ids * blank_tokens
        return masked_tokens.to(torch.int64)

    def _mask_by_random_topk(self, mask_len, probs, temperature=1.0, device="cpu"):
        """
        mask_len: (b 1)
        probs: (b n); also for the confidence scores

        This version keeps `mask_len` exactly.
        """

        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            """
            Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
            """
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(
            device
        )  # Gumbel max trick; 1e-5 for numerical stability; (b n)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(
            confidence, k=mask_len_unique, dim=-1, largest=False
        ).indices  # (b k)
        masking = torch.zeros_like(confidence).to(device)  # (b n)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.0
        masking = masking.bool()
        return masking

    def _first_pass(
        self,
        s_l: torch.Tensor,
        unknown_number_in_the_beginning_l,
        class_condition: Union[torch.Tensor, None],
        gamma,
        device,
    ):
        for t in range(self.T["lf"]):
            logits_l = self._masked_prediction(
                self.transformer_l, class_condition, s_l
            )  # (b n k)

            sampled_ids = torch.distributions.categorical.Categorical(
                logits=logits_l
            ).sample()  # (b n)
            unknown_map = (
                s_l == self.mask_token_ids["lf"]
            )  # which tokens need to be sampled; (b n)
            sampled_ids = torch.where(
                unknown_map, sampled_ids, s_l
            )  # keep the previously-sampled tokens; (b n)

            # create masking according to `t`
            ratio = 1.0 * (t + 1) / self.T["lf"]  # just a percentage e.g. 1 / 12
            ratio = torch.tensor(ratio).to(s_l)
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits_l, dim=-1)  # convert logits into probs; (b n K)
            selected_probs = torch.gather(
                probs, dim=-1, index=sampled_ids.unsqueeze(-1)
            ).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(
                unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS
            )  # assign inf probability to the previously-selected tokens; (b n)

            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning_l * mask_ratio), 1
            )  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(
                mask_len, min=0.0
            )  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self._mask_by_random_topk(
                mask_len,
                selected_probs,
                temperature=self.choice_temperatures["lf"] * (1.0 - ratio),
                device=device,
            )

            # Masks tokens with lower confidence.
            s_l = torch.where(masking, self.mask_token_ids["lf"], sampled_ids)  # (b n)

        # # use ESS (Enhanced Sampling Scheme)
        # if self.config['MaskGIT']['ESS']['use']:
        #     print(' ===== ESS: LF =====')
        #     t_star, s_star = self.critical_reverse_sampling(s_l, unknown_number_in_the_beginning_l, class_condition, 'lf')
        #     s_l = self.iterative_decoding_with_self_token_critic(t_star, s_star, 'lf', unknown_number_in_the_beginning_l, class_condition, device)

        return s_l

    def _second_pass(
        self,
        s_l: torch.Tensor,
        s_h: torch.Tensor,
        unknown_number_in_the_beginning_h,
        class_condition: Union[torch.Tensor, None],
        gamma,
        device,
    ):
        for t in range(self.T["hf"]):
            logits_h = self._masked_prediction(
                self.transformer_h, class_condition, s_l, s_h
            )  # (b m k)

            sampled_ids = torch.distributions.categorical.Categorical(
                logits=logits_h
            ).sample()  # (b m)
            unknown_map = (
                s_h == self.mask_token_ids["hf"]
            )  # which tokens need to be sampled; (b m)
            sampled_ids = torch.where(
                unknown_map, sampled_ids, s_h
            )  # keep the previously-sampled tokens; (b m)

            # create masking according to `t`
            ratio = 1.0 * (t + 1) / self.T["hf"]  # just a percentage e.g. 1 / 12
            ratio = torch.tensor(ratio).to(s_h)
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits_h, dim=-1)  # convert logits into probs; (b m K)
            selected_probs = torch.gather(
                probs, dim=-1, index=sampled_ids.unsqueeze(-1)
            ).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b m)
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(
                unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS
            )  # assign inf probability to the previously-selected tokens; (b m)

            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning_h * mask_ratio), 1
            )  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(
                mask_len, min=0.0
            )  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self._mask_by_random_topk(
                mask_len,
                selected_probs,
                temperature=self.choice_temperatures["hf"] * (1.0 - ratio),
                device=device,
            )

            # Masks tokens with lower confidence.
            s_h = torch.where(masking, self.mask_token_ids["hf"], sampled_ids)  # (b n)

        # # use ESS (Enhanced Sampling Scheme)
        # if self.config['MaskGIT']['ESS']['use']:
        #     print(' ===== ESS: HF =====')
        #     t_star, s_star = self.critical_reverse_sampling(s_l, unknown_number_in_the_beginning_h, class_condition, 'hf', s_h=s_h)
        #     s_h = self.iterative_decoding_with_self_token_critic(t_star, s_star, 'hf', unknown_number_in_the_beginning_h, class_condition, device, s_l=s_l)

        return s_h

    def _decode_token_ind_to_timeseries(
        self, s: torch.Tensor, frequency: str, return_representations: bool = False
    ):
        """
        It takes token embedding indices and decodes them to time series.
        :param s: token embedding index
        :param frequency:
        :param return_representations:
        :return:
        """
        self.eval()
        frequency = frequency.lower()
        assert frequency in ["lf", "hf"]

        vq_model = self.vq_model_l if frequency == "lf" else self.vq_model_h
        decoder = self.decoder_l if frequency == "lf" else self.decoder_h
        # zero_pad = zero_pad_high_freq if frequency == "lf" else zero_pad_low_freq

        zq = F.embedding(s, vq_model._codebook.embed)  # (b n d)
        zq = vq_model.project_out(zq)  # (b n c)
        zq = rearrange(zq, "b n c -> b c n")  # (b c n) == (b c (h w))
        H_prime = self.H_prime_l if frequency == "lf" else self.H_prime_h
        W_prime = self.W_prime_l if frequency == "lf" else self.W_prime_h
        zq = rearrange(zq, "b c (h w) -> b c h w", h=H_prime, w=W_prime)

        xhat = decoder(zq)

        if return_representations:
            return xhat, zq
        else:
            return xhat

    def _sample_impl(self, n_sample: int, condition: torch.Tensor = None, **kwargs):
        batch_size = kwargs.get("batch_size", n_sample)
        return_representations = kwargs.get("return_representations", False)
        if (condition is not None) and condition.ndim == 1:
            condition = torch.unsqueeze(condition, dim=-1)

        n_iters = n_sample // batch_size
        is_residual_batch = False
        if n_sample % batch_size > 0:
            n_iters += 1
            is_residual_batch = True

        x_new_l, x_new_h, x_new = [], [], []
        quantize_new_l, quantize_new_h = [], []
        sample_callback = self._iterative_decoding
        for i in range(n_iters):
            b = batch_size
            if (i + 1 == n_iters) and is_residual_batch:
                b = n_sample - ((n_iters - 1) * batch_size)
            embed_ind_l, embed_ind_h = sample_callback(
                num=b, device=self.device, class_index=condition
            )

            if return_representations:
                x_l, quantize_l = self._decode_token_ind_to_timeseries(
                    embed_ind_l, "lf", True
                )
                x_h, quantize_h = self._decode_token_ind_to_timeseries(
                    embed_ind_h, "hf", True
                )
                x_l, quantize_l, x_h, quantize_h = x_l, quantize_l, x_h, quantize_h
                quantize_new_l.append(quantize_l)
                quantize_new_h.append(quantize_h)
            else:
                x_l = self._decode_token_ind_to_timeseries(embed_ind_l, "lf")
                x_h = self._decode_token_ind_to_timeseries(embed_ind_h, "hf")

            x_new_l.append(x_l)
            x_new_h.append(x_h)
            x_new.append(x_l + x_h)  # (b c l); b=n_samples, c=1 (univariate)

        x_new_l = torch.cat(x_new_l)
        x_new_h = torch.cat(x_new_h)
        x_new = torch.cat(x_new).permute(0, 2, 1)

        return x_new
