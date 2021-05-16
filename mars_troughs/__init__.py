from .accumulation_model import (
    ACCUMULATION_MODEL_MAP,
    AccumulationModel,
    LinearInsolationAccumulation,
    QuadraticInsolationAccumulation,
)
from .datapaths import DATAPATHS
from .lag_model import LAG_MODEL_MAP, ConstantLag, LagModel, LinearLag
from .model import Model
from .trough import Trough

__author__ = [
    "Tom McClintock <thmsmcclintock@gmail.com>",
    "Kristell Izquierdo <kristell.izquierdo@gmail.com>",
]
__version__ = "0.0.1"
__docs__ = "Simulating martian ice troughs."
