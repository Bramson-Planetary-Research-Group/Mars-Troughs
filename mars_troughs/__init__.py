from .custom_acc_models import (Linear_Obliquity, 
                         Quadratic_Obliquity,
                         Cubic_Obliquity,
                         PowerLaw_Obliquity,
                         Linear_Insolation, 
                         Quadratic_Insolation,
                         Cubic_Insolation,
                         PowerLaw_Insolation)
from .custom_retr_models import (Constant_Retreat, 
                          Linear_RetreatO,
                         Quadratic_RetreatO,
                         Cubic_RetreatO,
                         PowerLaw_RetreatO,
                         Linear_RetreatI, 
                         Quadratic_RetreatI,
                         Cubic_RetreatI,
                         PowerLaw_RetreatI)
from .datapaths import (DATAPATHS,
                       load_insolation_data, 
                       load_obliquity_data)
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
