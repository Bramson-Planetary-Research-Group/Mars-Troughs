from .accumulation_model import (
    ACCUMULATION_MODEL_MAP,
    AccumulationModel,
    LinearInsolationAccumulation,
    QuadraticInsolationAccumulation,
)
from .datapaths import DATAPATHS
from .lag_model import ConstantLag, LAG_MODEL_MAP, LagModel, LinearLag
from .model import Model
from .custom_lag_model import CustomLagModel
from .custom_accu_model import CustomAccuModel
from .trough import Trough

__author__ = [
    "Tom McClintock <thmsmcclintock@gmail.com>",
    "Kristel Izquierdo <kristell.izquierdo@gmail.com>",
    "Kris Laferriere <klaferri@purdue.edu>",
]
__version__ = "0.0.1"
__docs__ = "Simulating martian ice troughs."
