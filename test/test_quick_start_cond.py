import torch
from src.model import VanillaDDPM, DiffusionTS
from src.dataset import SineND
from src.evaluation import predict_visual
from lightning import Trainer
import matplotlib.pyplot as plt


# # setup dataset and model
# dm = SineND(seq_len=64, seq_dim=2, batch_size=64)
# # model = DiffusionTS(seq_len=64, seq_dim=2)
# model = VanillaDDPM(seq_len=64, seq_dim=2, pred_x0=False)

# # training (on CPU for example)
# trainer = Trainer(max_epochs=30, accelerator="cpu")
# trainer.fit(model, dm)

# # testing
# dm.setup("test")
# real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
# data_mask = torch.cat([batch["data_mask"] for batch in dm.test_dataloader()])  # [N, 64, 2]
# # cond_data = torch.cat([batch["c"] for batch in dm.test_dataloader()])
# gen_data = model.sample(n_sample=10)  # [N, 64, 2]
# fig, axs = plt.subplots(2)
# axs[0].plot(real_data[0, :, :])
# axs[1].plot(gen_data[1, :, :])
# fig.savefig('test_pred_x0_false.png')

# visualization
# predict_visual(real_data, gen_data, data_mask, save_root='predict.png')


import torch
from src.model import VanillaDDPM
from src.dataset import SineND
from src.evaluation import predict_visual, imputation_visual
from lightning import Trainer

# setup dataset and model
# dm = SineND(seq_len=64, seq_dim=3, batch_size=64, condition='impute', missing_rate=0.2)
# model = VanillaDDPM(seq_len=64, seq_dim=3, condition='impute', missing_rate=0.2, pred_x0=False)

# predict
dm = SineND(seq_len=64, seq_dim=3, batch_size=64, condition="predict", obs_len=64)
model = VanillaDDPM(
    seq_len=64, seq_dim=3, condition="predict", obs_len=64, pred_x0=False
)

# training (on CPU for example)
trainer = Trainer(max_epochs=1000, devices=[0])
trainer.fit(model, dm)

# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
data_mask = torch.cat(
    [batch["data_mask"] for batch in dm.test_dataloader()]
)  # [N, 64, 2]
cond_data = torch.cat([batch["c"] for batch in dm.test_dataloader()])
gen_data = model.sample(n_sample=10, condition=cond_data)  # [N, 64, 2]
print(gen_data.shape)

# visualization
predict_visual(real_data, gen_data, data_mask, save_root="predict.png")
# imputation_visual(real_data, gen_data, cond_data, data_mask, save_root='impute.png')
