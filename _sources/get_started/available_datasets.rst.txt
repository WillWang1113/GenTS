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
   * - `Stocks <https://finance.yahoo.com/quote/GOOG/history/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAItg_cTf-Qqq-o8JkhX8sFlI5jjnjbuwEtUM9yei8HImkUYslPS6NF8xw-f8TZ0hMpiRXNC-A-KhQmrhLZzeb-75add2NFj8GKZixCElhjzP0Pju6Y3n7nGrQ0bgDTWwWlJ0i0nQSxoOIRGuJOJzW6MTbeHhZnasz_FNCvgAEcY5>`_
     - 1 day
     - 6
     - 
     - 
     - Financial
   * - `Energy <https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction>`_
     - 10 min
     - 28
     - 
     - 
     - Energy
   * - `ETT dataset <https://github.com/zhouhaoyi/ETDataset>`_
     - 1 hour/15 min
     - 7
     - 
     - 
     - Energy
   * - `Electricity dataset <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>`_
     - 1 hour
     - 321
     - 
     - 
     - Energy
   * - `Traffic dataset <http://pems.dot.ca.gov>`_
     - 1 hour
     - 862
     - 
     - 
     - Traffic
   * - `Exchange <https://github.com/laiguokun/multivariate-time-series-data>`_
     - 1 day
     - 8
     - 
     - 
     - Financial
   * - `MoJoCo <https://github.com/google-deepmind/mujoco>`_
     - continuous
     - 14
     - 
     - 
     - Physics
   * - `Physionet <https://www.physionet.org/content/challenge-2012/1.0.0/>`_
     - 1 min - 1 hour
     - 35
     - ✅
     - 2
     - Healthcare
   * - `ECG <https://www.timeseriesclassification.com/description.php?Dataset=ECG5000>`_
     - ~700 Hz
     - 1
     - 
     - 5
     - Healthcare
   * - `AirQuality <https://zenodo.org/records/4656719>`_
     - 1 hour
     - 6
     - ✅
     - 
     - Environment
   * - `Weather <https://www.bgc-jena.mpg.de/wetter/>`_
     - 10 min
     - 6
     - 
     - 
     - Environment
