from math import sqrt
from lightning import Trainer, seed_everything
import src.model
from src.data.dataloader import SynDataModule
import matplotlib.pyplot as plt

seed_everything(3407, workers=True)

model_names = src.model.__all__
# model_names = ['TimeVAE','TimeGAN', 'FourierDiffusion']
# model_names = ['COSCIGAN', 'VanillaVAE']
# model_names = ["PSAGAN", "VanillaVAE"]
# model_names = ['COSCIGAN', 'RCGAN']
# model_names = ['KoVAE', 'VanillaVAE', 'TimeVAE']
# model_names = ["KoVAE", "GTGAN"]
model_names = ["DiffusionTS", "VanillaVAE"]
# model_names = ['VanillaVAE','TMDM']
# model_names = ['MrDiff','VanillaVAE']
# model_names = ['VanillaVAE', 'VanillaGAN', 'VanillaMAF', 'VanillaDDPM']
# model_names = model_names[:2]

# TODO: iter all, Model Capability
# conditions = [
#     "predict",
#     None,
# ]
conditions = ["predict", 'impute']
# conditions = ['class', None]
# conditions = [None, "class"]
# conditions = [None, "impute"]
# conditions = ["impute", None]
# conditions = [None, "predict", "impute"]
batch_size = 128
seq_len = 64
add_coeffs = False
# imputation
missing_type = "random"
missing_rate = 0.2

# forecast
obs_len = 64
max_steps = 1000
max_epochs = 10
inference_batch_size = 4

# hparams
hparams = dict(
    seq_len=seq_len,
    seq_dim=2,
    covariate_dim=0,
    # n_classes=2,
    # latent_dim=20,
    hidden_size=128,
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
        add_coeffs = True
    else:
        add_coeffs = False
    if model_names[i] == "AST":
        channel_independent = True
        hparams["seq_dim"] = 1
    else:
        channel_independent = False
    for j in range(len(conditions)):
        c = conditions[j]
        if c == "predict":
            dm = SynDataModule(
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
                add_coeffs = True
            dm = SynDataModule(
                seq_len,
                batch_size,
                condition=c,
                missing_rate=missing_rate,
                missing_type=missing_type,
                add_coeffs=add_coeffs,
                inference_batch_size=inference_batch_size,
                channel_independent=channel_independent,
            )
            cond_hparams = dict(**hparams, condixtion=c)
        elif c == "class":
            dm = SynDataModule(
                seq_len,
                batch_size,
                condition=c,
                add_coeffs=add_coeffs,
                inference_batch_size=inference_batch_size,
                channel_independent=channel_independent,
            )
            cond_hparams = dict(**hparams, n_classes=2, condition=c)
        else:
            dm = SynDataModule(
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
        trainer = Trainer(devices=[1], max_epochs=max_epochs, log_every_n_steps=20)
        trainer.fit(test_model, dm)
        dm.setup("test")

        batch = next(iter(dm.test_dataloader()))
        for k in batch:
            batch[k] = batch[k].to(test_model.device)

        test_cond = None
        # if c in ["predict", "impute"]:
        #     test_cond = batch["c"].unsqueeze(0)
        samples = (
            test_model.sample(
                inference_batch_size, condition=batch.get("c", None), **batch
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
        print(samples.shape)
        if c == "impute":
            batch["seq"] = batch["seq"].masked_fill(batch["c"].bool(), float("nan"))
            axs[i, j].plot(range(seq_len), batch["seq"].squeeze()[0])
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
fig.suptitle("Model Comparison")
fig.tight_layout()
fig.savefig("test_model.png", bbox_inches="tight")
