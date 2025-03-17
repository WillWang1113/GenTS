from math import sqrt
from lightning import Trainer, seed_everything
import src.model
from src.data.dataloader import SynDataModule
import matplotlib.pyplot as plt

seed_everything(3407, workers=True)

model_names = src.model.__all__
# model_names = ['TimeVAE','TimeGAN', 'FourierDiffusion']
# model_names = ['COSCIGAN', 'VanillaVAE']
model_names = ["PSAGAN", "VanillaVAE"]
# model_names = ['COSCIGAN', 'RCGAN']
# model_names = ['TimeVQVAE','KoVAE', 'VanillaVAE', 'TimeVAE']
# model_names = ['KoVAE','VanillaVAE']
# model_names = ['VanillaVAE','TimeVAE', 'KoVAE']
# model_names = ['VanillaVAE', 'VanillaGAN', 'VanillaMAF', 'VanillaDDPM']
# model_names = model_names[:2]

# TODO: iter all, Model Capability
conditions = [None, 'class', ]
# conditions = ['class', None]
# conditions = [None, "class"]
# conditions = [None, 'impute']
# conditions = [None, "predict", "impute"]
batch_size = 128
seq_len = 64

# imputation
missing_type = "random"
missing_rate = 0.2

# forecast
obs_len = 96
max_steps = 1000
max_epochs = 1000

# hparams
hparams = dict(
    seq_len=seq_len,
    seq_dim=2,
    # latent_dim=64,
    hidden_size=128,
    depth_schedule=[5*25, 10*25, 15*25],
    # class_emb_dim=16,
    # lr={'G':1e-4, 'D':1e-4},
    n_critic = 1,
    epoch_fade_in=2*25,
    ks_conv=7,
    ks_value = 1,
    ks_key = 1,
    ks_query = 1,
    key_features= 1,
    value_features= 1,
    # gamma = 1.0,
)


n_row = len(model_names)
n_col = len(conditions)
fig, axs = plt.subplots(n_row, n_col, figsize=[3 * n_col, 3 * n_row], sharex="col")

for i in range(len(model_names)):
    for j in range(len(conditions)):
        c = conditions[j]
        if c == "predict":
            dm = SynDataModule(seq_len, batch_size, condition=c, obs_len=obs_len)
            cond_hparams = dict(**hparams, obs_len=obs_len, condition=c)
        elif c == "impute":
            dm = SynDataModule(
                seq_len,
                batch_size,
                condition=c,
                missing_rate=missing_rate,
                missing_type=missing_type,
            )
            cond_hparams = dict(**hparams, condition=c)
        elif c == "class":
            dm = SynDataModule(seq_len, batch_size, condition=c)
            cond_hparams = dict(**hparams, n_classes=2, condition=c)
        else:
            dm = SynDataModule(seq_len, batch_size, condition=c)
            cond_hparams = hparams

        test_model_cls = getattr(src.model, model_names[i])
        test_model = test_model_cls(**cond_hparams)
        trainer = Trainer(devices=[1], max_epochs=max_epochs, log_every_n_steps=20)
        trainer.fit(test_model, dm)
        batch = next(iter(dm.val_ds))
        test_cond = None
        # if c in ["predict", "impute"]:
        #     test_cond = batch["c"].unsqueeze(0)
        samples = test_model.sample(1, condition=test_cond).squeeze(0).cpu().numpy()
        if c == "impute":
            axs[i, j].plot(range(seq_len), batch["c"])
            axs[i, j].plot(range(seq_len), samples)
        elif c == "predict":
            axs[i, j].plot(range(obs_len), batch["c"])
            axs[i, j].plot(range(obs_len, obs_len + seq_len), samples)
        else:
            axs[i, j].plot(range(seq_len), batch["seq"])
            axs[i, j].plot(range(seq_len), samples)

        axs[i, j].set_title(model_names[i] + "_" + f"{c if c is not None else 'syn'}")
        break
    break
fig.suptitle("Model Comparison")
fig.tight_layout()
fig.savefig("test_model.png", bbox_inches="tight")
