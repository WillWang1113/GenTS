from src.model.vae.timevqvae import TimeVQVAE
import torch

model = TimeVQVAE(seq_dim=1, seq_len=10)
x = torch.randn(32, 10, 1)
print(model(dict(seq=x), None))