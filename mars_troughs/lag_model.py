"""
Models for the lag as a function of time.
"""
from abc import abstractmethod
from typing import Dict

import numpy as np

from mars_troughs.model import Model
from mars_troughs.linear_model import LinearModel


class LagModel(Model):
    """
    Abstract class for lag models, that have a method
    called :meth:`get_lag_at_t` that returns the lag
    as a function of time.
    """

    @abstractmethod
    def get_lag_at_t(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConstantLag(LagModel):
    """
    The lag thickness is constant and does not depend on time.

    Args:
        constant (float, optional): default is 1 millimeter. The lag
            thickness at all times.
    """

    def __init__(self, constant: float = 1.0):
        self.constant = constant

    @property
    def parameter_names(self) -> Dict[str, float]:
        return ["constant"]

    def get_lag_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Lag as a function of time: returns constant lag values.

        Args:
            time (np.ndarray): times at which we want to calculate the lag.
        Output:
            np.ndarray of the same size as time input containing values of lag.
            All elements in the array are the same since the lag is
            constant.
        """
        return self.constant * np.ones_like(time)


class LinearLag(LagModel, LinearModel):
    """
    The lag thickness is linear in time. Lag changes as
    lag(t) = intercept + slope*t.

    Args:
        intercept (float, optional): default is 1 millimeter. The lag
            thickness at time t=0 (present day).
        slope (float, optional): default is 1e-6 mm per year. The rate
            of change of the lag per time.
    """
    


    def __init__(self, intercept: float = 1.0, slope: float = 1e-6):
        self.intercept = intercept
        self.slope = slope

    @property
    def parameter_names(self) -> Dict[str, float]:
        return ["intercept", "slope"]
    
    def get_lag_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Compute lag thickness at each value of time using Linear Model

        Args:
            time (np.ndarray): times at which we want to calculate the lag.
        Output:
            np.ndarray of the same size as time input containing values of lag
            thickness.

        """
        return self.eval(time)
