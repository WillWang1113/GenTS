
# from gents.dataset.energy import Energy

# from gents.dataset.stocks import Stocks
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from gents.dataset.modules.weather import Weather
from gents.model.diffeq.latentsde.model import LatentSDE
from gents.model.gan.gtgan.model import GTGAN

seed_everything(9)

# setup dataset and model
dm = Weather(
    # obs_len=96,
    seq_len=24,
    # num_samples=100,
    batch_size=64,
    select_seq_dim=list(range(16)),
    add_coeffs='cubic_spline'
    # irregular_dropout=0.2,
    # condition="predict",
)


model = GTGAN(
    seq_len=dm.seq_len,
    seq_dim=dm.seq_dim,
)

# training (on CPU for example)
trainer = Trainer(
    max_epochs=200,
    devices=[0],
    callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min")],
    enable_progress_bar=True,
    # gradient_clip_val=1.0,
    # gradient_clip_algorithm=
)
#
trainer.fit(
    model,
    dm,
    ckpt_path="/mnt/ExtraDisk/wcx/research/GenTS_multivar_syn/lightning_logs/version_218/checkpoints/epoch=149-step=86550.ckpt",
    # ckpt_path="/mnt/ExtraDisk/wcx/research/GenTS_multivar_syn/lightning_logs/version_219/checkpoints/epoch=8-step=5193.ckpt",
)

print('Done!')

# # testing
# dm.setup("test")
# real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
# real_data_mask = torch.cat(
#     [batch["data_mask"] for batch in dm.test_dataloader()]
# )  # [N, 64, 2]
# t = torch.cat([batch["t"] for batch in dm.test_dataloader()])  # [N, 64, 2]
# cond_data = torch.cat([batch["c"] for batch in dm.test_dataloader()])  # [N, 64, 2]
# # model.cuda()
# gen_data = model.sample(
#     n_sample=10,
#     condition=cond_data,
#     # condition=real_data.masked_fill(~real_data_mask, float("nan"))[[3]],
#     t=t,
#     data_mask=real_data_mask,
#     seq=real_data
#     # seq=real_data
# )  # [N, 64, 2]

# print(real_data.shape)
# print(gen_data.shape)
# # visualization with tsne
# # qualitative_visual(real_data, gen_data, analysis="tsne", save_root="tsne.png")

# # fig, axs = plt.subplots(2)
# # axs[0].plot(real_data.masked_fill(~real_data_mask, float("nan"))[3, :, :].cpu())
# # axs[1].plot(gen_data[0, ..., 0].cpu())
# # axs[1].plot(gen_data[0, ..., 1].cpu())
# # axs[1].plot(gen_data[0, ..., 2].cpu())
# # fig.savefig("test_pred_x0_false.png")

# predict_visual(real_data, gen_data, real_data_mask, save_root='test_ImagenTime.png')
