Available Models
================

``GenTS`` included 25+ state-of-the-art time series generation models, with different capabilities. Our model zoo is updated in a regular basis. Please refer to :ref:`customization <customization>`  for developing your own model under our framework!

.. note::
    Most of the models are implemented based on the original papers, execpt for ``VanillaVAE``, ``VanillaGAN``, ``VanillaMAF``, and ``VanillaDDPM``. They are naive implementations of VAE, GAN, MAF, and DDPM, respectively. Users can use them as a baseline for their own models.

.. list-table::
   :header-rows: 1

   * - Name
     - Model Type
     - Synthesis
     - Forecasting
     - Imputation
     - Class label
   * - `VanillaVAE <https://arxiv.org/abs/1312.6114>`_
     - VAE
     - ✅
     - ✅
     - ✅
     - ✅
   * - `TimeVAE <https://arxiv.org/abs/2111.08095>`_
     - VAE
     - ✅
     - 
     - 
     - 
   * - `TimeVQVAE <https://arxiv.org/abs/2303.04743>`_
     - VAE
     - ✅
     - 
     - 
     - ✅
   * - `KoVAE <https://arxiv.org/abs/2305.19480>`_
     - VAE
     - ✅
     - 
     - 
     - 
   * - `VanillaGAN <https://proceedings.neurips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html>`_
     - GAN
     - ✅
     - ✅
     - ✅
     - ✅
   * - `TimeGAN <https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html?ref=https://githubhelp.com>`_
     - GAN
     - ✅
     - 
     - 
     - 
   * - `COSCI-GAN <https://proceedings.neurips.cc/paper_files/paper/2022/hash/d3408794e41dd23e34634344d662f5e9-Abstract-Conference.html>`_
     - GAN
     - ✅
     - 
     - 
     - 
   * - `GT-GAN <https://proceedings.neurips.cc/paper_files/paper/2022/hash/f03ce573aa8bce26f77b76f1cb9ee979-Abstract-Conference.html>`_
     - GAN
     - ✅
     - 
     - 
     - 
   * - `PSA-GAN <https://arxiv.org/abs/2108.00981>`_
     - GAN
     - ✅
     - 
     - 
     - 
   * - `RCGAN <https://arxiv.org/abs/1706.02633>`_
     - GAN
     - ✅
     - ✅
     - 
     - ✅
   * - `VanillaMAF <https://proceedings.neurips.cc/paper_files/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html>`_
     - Flow
     - ✅
     - ✅
     - ✅
     - ✅
   * - `Fourier Flow <https://openreview.net/forum?id=PpshD0AXfA>`_
     - Flow
     - ✅
     - 
     - 
     - 
   * - `VanillaDDPM <https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html>`_
     - Diffusion
     - ✅
     - ✅
     - ✅
     - ✅
   * - `CSDI <https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html>`_
     - Diffusion
     - 
     - ✅
     - ✅
     - 
   * - `Diffusion-TS <https://arxiv.org/abs/2403.01742>`_
     - Diffusion
     - ✅
     - ✅
     - ✅
     - 
   * - `TMDM <https://openreview.net/forum?id=qae04YACHs>`_
     - Diffusion
     - 
     - ✅
     - 
     - 
   * - `FourierDiffusion <https://arxiv.org/abs/2402.05933>`_
     - Diffusion
     - ✅
     - 
     - 
     - 
   * - `ImagenTime <https://proceedings.neurips.cc/paper_files/paper/2024/hash/dc6748383752138af7f00b3185a0a404-Abstract-Conference.html>`_
     - Diffusion
     - ✅
     - ✅
     - ✅
     - 
   * - `FIDE <https://proceedings.neurips.cc/paper_files/paper/2024/hash/cfce727868dcaf5295c0125f9d6fbc0b-Abstract-Conference.html>`_
     - Diffusion
     - ✅
     - 
     - 
     - 
   * - `Latent ODE <https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html>`_
     - Diff. Eq.
     - ✅
     - ✅
     - interpolation
     - 
   * - `Latent SDE <https://proceedings.mlr.press/v108/li20i.html>`_
     - Diff. Eq.
     - ✅
     - 
     - 
     - 
   * - `SDEGAN <https://proceedings.mlr.press/v139/kidger21b.html>`_
     - Diff. Eq.
     - ✅
     - 
     - 
     - 
   * - `LS4 <https://proceedings.mlr.press/v202/zhou23i.html>`_
     - Diff. Eq.
     - ✅
     - ✅
     - interpolation
     - 

