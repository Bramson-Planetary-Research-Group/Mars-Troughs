from .custom_acc_models import (Linear_Obliquity, 
                         Quadratic_Obliquity,
                         Cubic_Obliquity,
                         PowerLaw_Obliquity,
                         Linear_Insolation, 
                         Quadratic_Insolation,
                         Cubic_Insolation,
                         PowerLaw_Insolation)
from .custom_lag_models import (ConstantLag,
                         QuadraticLag,
                         LinearLag,
                         CubicLag,
                         PowerLawLag)
from .datapaths import (DATAPATHS,
                       load_insolation_data, 
                       load_obliquity_data,
                       load_retreat_data,
		      load_angle,
	               load_TMP_data)
from .model import Model
from .trough import Trough
from .mcmc import MCMC, softAgePriorMCMC, hardAgePriorMCMC

__author__ = [
    "Tom McClintock <thmsmcclintock@gmail.com>",
    "Kristel Izquierdo <kristell.izquierdo@gmail.com>",
    "Kris Laferriere <klaferri@purdue.edu>",
]
__version__ = "0.0.1"
__docs__ = "Simulating martian ice troughs."
