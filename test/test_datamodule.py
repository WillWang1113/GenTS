from matplotlib import pyplot as plt
import torch
from src.evaluation.visual import imputation_visual
from src.model import VanillaDDPM
from src.dataset import SineND, Stocks, Energy, MuJoCo, Spiral2D
from src.dataset.physionet import Physionet
from src.evaluation import qualitative_visual
from lightning import Trainer

from src.model.diffeq.latentode.model import LatentODE
from src.model.diffusion.csdi.model import CSDI
from src.model.diffusion.diffusionts.model import DiffusionTS

# setup
# dm = Physionet(select_seq_dim=[4,8, 10], batch_size=64)
# model = VanillaDDPM(seq_len=dm.seq_len, seq_dim=dm.seq_dim)

# # training (on CPU for example)
# trainer = Trainer(max_epochs=10, accelerator="cpu")
# trainer.fit(model, dm)

# # testing
# dm.setup("test")
# real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
# gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].plot(real_data[0, :, :])
# ax[0].set_title("Real Data")
# ax[1].plot(gen_data[0, :, :])
# ax[1].set_title("Generated Data")
# ax[0].legend()
# ax[1].legend()
# fig.savefig('generated_vs_real.png')

# visualization(real_data, gen_data, analysis="tsne")



# # import ...
# from src.evaluation import crps, mse

# Forecasting
# dm = SineND(obs_len=64, seq_len=64, seq_dim=2, batch_size=64, condition='predict')
# model = VanillaDDPM(seq_len=64, seq_dim=2, obs_len=64, condition='predict')

"""Imputation"""
dm = Spiral2D(seq_len=96, batch_size=64, condition='impute', missing_rate=0.2, random_drop_data=0.2)
# dm = Physionet(select_seq_dim=[4,8, 10], batch_size=64, condition='impute', missing_rate=0.2)
# dm = SineND(missing_rate=0.2, seq_len=64, seq_dim=2, batch_size=64)
model = LatentODE(seq_len=dm.seq_len, seq_dim=dm.seq_dim, condition='impute', z0_encoder='rnn')

dm.setup('test')


for batch in dm.test_dataloader():
    print(batch['seq'][0,:,0])
    print(batch['c'][0,:,0])
    print(batch['data_mask'][0,:,0])
    break

# # training (same before)
# trainer = Trainer(max_epochs=50, devices=[0])
# trainer.fit(model, dm)

# # testing
# dm.setup("test")
# real_data, gen_data = [], []
# data_mask, cond_mask = [], []
# for batch in dm.test_dataloader():
#     real_data.append(batch['seq'][:,-dm.seq_len:])  # [N, 64, 2]
#     # in conditional generation, n_sample means how many inference times
#     gen_data.append(model.sample(n_sample=10, condition=batch['c'], **batch))  # [N, 64, 2, 100]
#     data_mask.append(batch['data_mask'])
#     cond_mask.append(~torch.isnan(batch['c']))
# real_data = torch.concat(real_data)
# gen_data = torch.concat(gen_data)
# data_mask = torch.concat(data_mask)
# cond_mask = torch.concat(cond_mask)

# imputation_viz(real_data, gen_data, cond_mask, data_mask, save_root='./test_impute.png')

# # fig, ax = plt.subplots(1)
# # ax.plot(real_data[0, :, :], label="Real Data")
# # ax.plot(gen_data[0, :, :, 0], label="Generated Data")
# # ax.set_title("Real Data")
# # fig.savefig('generated_vs_real.png')

# # visualization with tsne
# # print(crps(real_data, gen_data))