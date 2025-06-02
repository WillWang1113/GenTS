import torch
from src.model import VanillaDDPM
from src.data import SineND
from src.evaluation import visualization
from lightning import Trainer

# setup
dm = SineND(seq_len=64, seq_dim=2, batch_size=64)
model = VanillaDDPM(seq_len=64, seq_dim=2)

# training (on CPU for example)
trainer = Trainer(max_epochs=100, accelerator="cpu")
trainer.fit(model, dm)

# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

visualization(real_data, gen_data, analysis="tsne")
