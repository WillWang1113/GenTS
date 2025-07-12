from .mujoco import MuJoCo
from .simple import SineND, Spiral2D
from .stocks import Stocks
from .energy import Energy
from .electricity import Electricity
from .weather import Weather
from .traffic import Traffic
from .exchange import Exchange
from .ett import ETTh1, ETTh2, ETTm1, ETTm2
from .air_quality import AirQuality
from .ecg import ECG
from .physionet import Physionet

__all__ = [
    "Spiral2D",
    "SineND",
    "Stocks",
    "Energy",
    "MuJoCo",
    "Electricity",
    "Weather",
    "Traffic",
    "Exchange",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "AirQuality",
    "ECG",
    "Physionet",
]
