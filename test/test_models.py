from math import sqrt
from lightning import Trainer, seed_everything
import src.model
from src.data.datamodules import Spiral2D
import matplotlib.pyplot as plt
import torch


seed_everything(3407, workers=True)
gpu = 0

model_names = src.model.__all__
# model_names = ['TimeVAE','TimeGAN', 'FourierDiffusion']
# model_names = ['COSCIGAN', 'VanillaVAE']
# model_names = ["PSAGAN", "VanillaVAE"]
# model_names = ['COSCIGAN', 'RCGAN']
# model_names = ['KoVAE', 'VanillaVAE', 'TimeVAE']
# model_names = ["KoVAE", "GTGAN"]
# model_names = ["ImagenTime", "VanillaVAE"]
# model_names = ['VanillaVAE','TMDM']
# model_names = ['MrDiff','VanillaVAE']
model_names = ["VanillaMAF","VanillaDDPM", ]
# model_names = model_names[:2]

# TODO: iter all, Model Capability
conditions = [
    "impute",
    None,
    # "predict",
]
# conditions = ["impute", "predict", None]
# conditions = ['class', None]
# conditions = [None, "class"]
# conditions = [None, "impute"]
# conditions = ["impute", None]
# conditions = ["impute", None]
batch_size = 128
seq_len = 64
add_coeffs = False
# imputation
missing_type = "random"
missing_rate = 0.2

# forecast
obs_len = 64
max_steps = 1000
max_epochs = 500
inference_batch_size = 7

# hparams
hparams = dict(
    seq_len=seq_len,
    seq_dim=2,
    covariate_dim=0,
    delay=8,
    embedding=16,
    use_stft=False,
    n_fft=15,
    hop_length=8,
    # n_classes=2,
    # latent_dim=20,
    hidden_size=128,
    poisson=True,
    # w_kl=1e-4,
    # n_diff_steps=200,
    # d_ff=512,
    depth_schedule=[20, 40, 60],
    # class_emb_dim=16,
    # lr={'G':1e-4, 'D':1e-4},
    # lr=1e-3,
    # beta=1e-1,
    # n_critic=1,
    # epoch_fade_in=2 * 25,
    ks_conv=7,
    ks_value=1,
    ks_key=1,
    ks_query=1,
    # key_features=1,
    # value_features=1,
    # gamma = 1.0,
)


n_row = len(model_names)
n_col = len(conditions)
fig, axs = plt.subplots(n_row, n_col, figsize=[3 * n_col, 3 * n_row], sharex="col")

for i in range(len(model_names)):
    if model_names[i] == "GTGAN":
        add_coeffs = "cubic_spline"
    elif model_names[i] == "SDEGAN":
        add_coeffs = "linear"

    else:
        add_coeffs = None
    if model_names[i] in ["AST", "FIDE"]:
        channel_independent = True
        hparams["seq_dim"] = 1
    else:
        channel_independent = False
        hparams["seq_dim"] = 2
    for j in range(len(conditions)):
        c = conditions[j]
        if c == "predict":
            dm = Spiral2D(
                seq_len,
                batch_size,
                condition=c,
                obs_len=obs_len,
                add_coeffs=add_coeffs,
                inference_batch_size=inference_batch_size,
                channel_independent=channel_independent,
            )
            cond_hparams = dict(**hparams, obs_len=obs_len, condition=c)
        elif c == "impute":
            if model_names[i] == "KoVAE":
                add_coeffs = "cubic_spline"
            dm = Spiral2D(
                seq_len,
                batch_size,
                condition=c,
                missing_rate=missing_rate,
                missing_type=missing_type,
                add_coeffs=add_coeffs,
                inference_batch_size=inference_batch_size,
                channel_independent=channel_independent,
            )
            cond_hparams = dict(**hparams, condition=c, missing_rate=missing_rate)
        elif c == "class":
            dm = Spiral2D(
                seq_len,
                batch_size,
                condition=c,
                add_coeffs=add_coeffs,
                inference_batch_size=inference_batch_size,
                channel_independent=channel_independent,
            )
            cond_hparams = dict(**hparams, n_classes=2, condition=c)
        else:
            dm = Spiral2D(
                seq_len,
                batch_size,
                condition=c,
                add_coeffs=add_coeffs,
                inference_batch_size=inference_batch_size,
                channel_independent=channel_independent,
            )
            cond_hparams = hparams

        test_model_cls = getattr(src.model, model_names[i])
        test_model = test_model_cls(**cond_hparams)
        trainer = Trainer(
            devices=[gpu],
            # max_steps=max_steps,
            max_epochs=max_epochs,
            log_every_n_steps=20,
            check_val_every_n_epoch=5,
        )

        trainer.fit(test_model, dm)
        dm.setup("test")

        batch = next(iter(dm.test_dataloader()))
        for k in batch:
            batch[k] = batch[k].to(f"cuda:{gpu}")
        test_model.to(f"cuda:{gpu}")

        test_cond = None
        # if c in ["predict", "impute"]:
        #     test_cond = batch["c"].unsqueeze(0)
        samples = (
            test_model.sample(10, condition=batch.get("c"), **batch)
            .squeeze(0)
            .cpu()
        )
        print(samples.shape)
        batch["seq"] = batch["seq"].cpu()
        try:
            batch["c"] = batch["c"].cpu()
        except:
            pass

        if c == "impute":
            mask = torch.isnan(batch["c"])
            real = batch["seq"].masked_fill(~mask, float("nan"))
            timeaxis = torch.arange(seq_len)

            axs[i, j].plot(timeaxis, batch["seq"].squeeze()[0])
            axs[i, j].scatter(timeaxis, real.squeeze()[0, ..., 0], c="C0")
            axs[i, j].scatter(timeaxis, real.squeeze()[0, ..., 1], c="C1")

            if samples.ndim == 4:
                for iii in range(samples.shape[-1]):
                    samples[..., iii] = samples[..., iii].masked_fill(
                        ~mask, float("nan")
                    )

                for ii in range(samples.shape[2]):
                    for jj in range(samples.shape[-1]):
                        axs[i, j].scatter(
                            timeaxis,
                            samples[0, :, ii, jj].squeeze(),
                            # samples[0, :, ii].squeeze().mean(dim=-1),
                            c=f"C{ii + 2}",
                            alpha=0.5,
                        )
            else:
                axs[i, j].plot(range(seq_len), samples[0])
        elif c == "predict":
            # axs[i, j].plot(range(obs_len), batch["c"].squeeze()[0])
            print(batch["seq"].shape)
            axs[i, j].plot(range(0, obs_len + seq_len), batch["seq"].squeeze()[0])
            for ii in range(samples.shape[2]):
                axs[i, j].plot(
                    range(obs_len, obs_len + seq_len),
                    samples[0, :, ii].squeeze(),
                    c=f"C{ii}",
                    alpha=0.5,
                )
        else:
            axs[i, j].plot(range(seq_len), batch["seq"].squeeze()[0])
            axs[i, j].plot(range(seq_len), samples[0])

        axs[i, j].set_title(model_names[i] + "_" + f"{c if c is not None else 'syn'}")
        break
fig.suptitle("Model Comparison")
fig.tight_layout()
fig.savefig("test_model.png", bbox_inches="tight")
