Quick Start
===========

A minimal example of (unconditional) time series generation:

.. code-block:: python
        
    import torch
    from gents.model import VanillaDDPM
    from gents.dataset import SineND
    from gents.evaluation import qualitative_visual
    from lightning import Trainer

    # setup dataset and model
    dm = SineND(seq_len=64, seq_dim=2, batch_size=64)
    model = VanillaDDPM(seq_len=dm.seq_len, seq_dim=dm.seq_dim)

    # training (on CPU for example)
    trainer = Trainer(max_epochs=100, accelerator="cpu")
    trainer.fit(model, dm)

    # testing
    dm.setup("test")
    real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
    gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

    # visualization with tsne
    tsne_visual(real_data, gen_data, save_root="tsne.png")

The resulting "fake" time series versus real time series:

.. image:: samples.png
  :width: 500

Throughout the test set, visualization with TSNE:

.. image:: tsne.png
  :width: 500