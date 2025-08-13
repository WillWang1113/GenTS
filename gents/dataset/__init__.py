from .modules.mujoco import MuJoCo
from .modules.simple import SineND, Spiral2D
from .modules.stocks import Stocks
from .modules.energy import Energy
from .modules.electricity import Electricity
from .modules.weather import Weather
from .modules.traffic import Traffic
from .modules.exchange import Exchange
from .modules.ett import ETTh1, ETTh2, ETTm1, ETTm2
from .modules.air_quality import AirQuality
from .modules.ecg import ECG
from .modules.physionet import Physionet

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
