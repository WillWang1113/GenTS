Available Datasets
==================

``GenTS`` presets 13 widely used time series generation datasets from multiple domains. Some of them come naturally with missing values and class labels, supporting to benchmark different kinds of models.

.. note::
    ``SineND``, ``Spiral2D``, and ``MoJoCo`` are simulated datasets, so they have arbitrary resolution (and time steps).

.. note::
    ``Physionet`` originally records 48-hour patients status at 1-minute resolution, but it can also be aggregated to other resolutions. We recommend the lowest resolution (1 hour) for benchmarking.

.. list-table:: Data Overview
   :header-rows: 1

   * - Name
     - Resolution
     - Dimension
     - Missing value
     - Class label
     - Domain
   * - SineND
     - continuous
     - N
     - 
     - 
     - Physics
   * - Spiral2D
     - continuous
     - 2
     - 
     - 2
     - Physics
   * - Stocks
     - 1 day
     - 6
     - 
     - 
     - Financial
   * - Energy
     - 10 min
     - 28
     - 
     - 
     - Energy
   * - ETT
     - 1 hour/15 min
     - 7
     - 
     - 
     - Energy
   * - Electricity
     - 1 hour
     - 321
     - 
     - 
     - Energy
   * - Traffic
     - 1 hour
     - 862
     - 
     - 
     - Traffic
   * - Exchange
     - 1 day
     - 8
     - 
     - 
     - Financial
   * - MoJoCo
     - continuous
     - 14
     - 
     - 
     - Physics
   * - Physionet
     - 1 min - 1 hour
     - 35
     - ✅
     - 2
     - Healthcare
   * - ECG
     - ~700 Hz
     - 1
     - 
     - 5
     - Healthcare
   * - Air quality
     - 1 hour
     - 6
     - ✅
     - 
     - Environment
   * - Weather
     - 10 min
     - 6
     - 
     - 
     - Environment
