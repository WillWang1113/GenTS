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
   * - VanillaVAE
     - VAE
     - ✅
     - ✅
     - ✅
     - ✅
   * - TimeVAE
     - VAE
     - ✅
     - 
     - 
     - 
   * - TimeVQVAE
     - VAE
     - ✅
     - 
     - 
     - ✅
   * - KoVAE
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
   * - TimeGAN
     - GAN
     - ✅
     - 
     - 
     - 
   * - COSCI-GAN
     - GAN
     - ✅
     - 
     - 
     - 
   * - GT-GAN
     - GAN
     - ✅
     - 
     - 
     - 
   * - PSA-GAN
     - GAN
     - ✅
     - 
     - 
     - 
   * - RCGAN
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
   * - Fourier Flow
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
   * - CSDI
     - Diffusion
     - 
     - ✅
     - ✅
     - 
   * - Diffusion-TS
     - Diffusion
     - ✅
     - ✅
     - ✅
     - 
   * - TMDM
     - Diffusion
     - 
     - ✅
     - 
     - 
   * - mr-diff
     - Diffusion
     - 
     - ✅
     - 
     - 
   * - FourierDiffusion
     - Diffusion
     - ✅
     - 
     - 
     - 
   * - ImagenTime
     - Diffusion
     - ✅
     - ✅
     - ✅
     - 
   * - FIDE
     - Diffusion
     - ✅
     - 
     - 
     - 
   * - Latent ODE w. ODE-RNN
     - Diff. Eq.
     - ✅
     - ✅
     - interpolation
     - 
   * - Latent ODE w. RNN
     - Diff. Eq.
     - ✅
     - ✅
     - interpolation
     - 
   * - Latent SDE
     - Diff. Eq.
     - ✅
     - 
     - 
     - 
   * - SDEGAN
     - Diff. Eq.
     - ✅
     - 
     - 
     - 
   * - LS4
     - Diff. Eq.
     - ✅
     - ✅
     - interpolation
     - 

