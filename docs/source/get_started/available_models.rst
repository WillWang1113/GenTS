Available Models
================

``GenTS`` included 25+ state-of-the-art time series generation models, with different capabilities. Our model zoo is updated in a regular basis. Please refer to :ref:`customization` for developing your own model under our framework!

.. note::
    Most of the models are implemented based on the original papers, execpt for ``VanillaVAE``, ``VanillaGAN``, ``VanillaMAF``, and ``VanillaDDPM``. They are naive implementations of VAE, GAN, MAF, and DDPM, respectively. Users can use them as a baseline for their own models.

.. note::
    Some models are limited, for example, ``FourierFlow`` is originally implemented on CPU and does not support multivariate time series. Please refer to our API documentation for more details.

.. list-table::
   :header-rows: 1

   * - Name
     - Model Type
     - Synthesis
     - Forecasting
     - Imputation
     - Class label
   * - VanillaVAE
     - VAE
     - ✅
     - ✅
     - ✅
     - ✅
   * - `TimeVAE <https://arxiv.org/abs/2111.08095>`__
     - VAE
     - ✅
     - 
     - 
     - 
   * - `TimeVQVAE <https://arxiv.org/abs/2111.08095>`__
     - VAE
     - ✅
     - 
     - 
     - ✅
   * - `KoVAE <https://openreview.net/pdf?id=eY7sLb0dVF>`__
     - VAE
     - ✅
     - 
     - 
     - 
   * - VanillaGAN
     - GAN
     - ✅
     - ✅
     - ✅
     - ✅
   * - `TimeGAN <https://proceedings.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf>`__
     - GAN
     - ✅
     - 
     - 
     - 
   * - `COSCI-GAN <https://openreview.net/pdf?id=RP1CtZhEmR>`__
     - GAN
     - ✅
     - 
     - 
     - 
   * - `GT-GAN <https://proceedings.neurips.cc/paper_files/paper/2022/file/f03ce573aa8bce26f77b76f1cb9ee979-Paper-Conference.pdf>`__
     - GAN
     - ✅
     - 
     - 
     - 
   * - `PSA-GAN <https://openreview.net/pdf?id=Ix_mh42xq5w>`__
     - GAN
     - ✅
     - 
     - 
     - 
   * - `RCGAN <https://arxiv.org/pdf/1706.02633>`__ 
     - GAN
     - ✅
     - ✅
     - 
     - ✅
   * - VanillaMAF
     - Flow
     - ✅
     - ✅
     - ✅
     - ✅
   * - `Fourier Flow <https://openreview.net/pdf?id=PpshD0AXfA>`__
     - Flow
     - ✅
     - 
     - 
     - 
   * - VanillaDDPM
     - Diffusion
     - ✅
     - ✅
     - ✅
     - ✅
   * - `CSDI <https://arxiv.org/pdf/2107.03502>`__
     - Diffusion
     - 
     - ✅
     - ✅
     - 
   * - `Diffusion-TS <https://openreview.net/pdf?id=4h1apFjO99>`__
     - Diffusion
     - ✅
     - ✅
     - ✅
     - 
   * - `TMDM <https://openreview.net/pdf?id=qae04YACHs>`__
     - Diffusion
     - 
     - ✅
     - 
     - 
   * - `MrDiff <https://openreview.net/pdf?id=mmjnr0G8ZY>`__
     - Diffusion
     - 
     - ✅
     - 
     - 
   * - `FourierDiffusion <https://arxiv.org/pdf/2402.05933>`__
     - Diffusion
     - ✅
     - 
     - 
     - 
   * - `ImagenTime <https://openreview.net/forum?id=2NfBBpbN9x&noteId=uYWwrwEW6Y>`__
     - Diffusion
     - ✅
     - ✅
     - ✅
     - 
   * - `FIDE <https://openreview.net/pdf?id=5HQhYiGnYb>`__
     - Diffusion
     - ✅
     - 
     - 
     - 
   * - `Latent ODE <https://arxiv.org/abs/1907.03907>`__ with ODE-RNN
     - Diff. Eq.
     - ✅
     - ✅
     - interpolation
     - 
   * - `Latent ODE <https://arxiv.org/abs/1907.03907>`__ with RNN
     - Diff. Eq.
     - ✅
     - ✅
     - interpolation
     - 
   * - `Latent SDE <https://arxiv.org/pdf/2001.01328>`__
     - Diff. Eq.
     - ✅
     - 
     - 
     - 
   * - `SDEGAN <https://arxiv.org/pdf/2102.03657>`__
     - Diff. Eq.
     - ✅
     - 
     - 
     - 
   * - `LS4 <https://arxiv.org/abs/2212.12749>`__
     - Diff. Eq.
     - ✅
     - ✅
     - interpolation
     - 

