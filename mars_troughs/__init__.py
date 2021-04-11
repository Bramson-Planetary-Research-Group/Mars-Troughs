from .accumulation_model import AccumulationModel, LinearInsolationAccumulation
from .datapaths import DEFAULT_DATAPATH_DICT
from .dataset import Dataset
from .lag_model import ConstantLag, LagModel, LinearLag
from .model import Model
from .trough import Trough

__author__ = [
    "Tom McClintock <thmsmcclintock@gmail.com>",
    "Kristell Izquierdo <kristell.izquierdo@gmail.com>",
]
__version__ = "0.0.1"
__docs__ = "Simulating martian ice troughs."
