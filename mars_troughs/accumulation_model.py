"""
Model for the accumulation rates.
"""
from abc import abstractmethod
from typing import Dict

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from mars_troughs.model import Model


class AccumulationModel(Model):
    """
    Abstract class for computing the amount of ice accumulation.
    """

    @abstractmethod
    def accumulation(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class InsolationAccumulationModel(AccumulationModel):
    """
    An accumulation model that depends only on solar insolation.
    Interpolated splines are created for the insolation as a function
    of time for faster integration.

    Args:
        times (np.ndarray): times at which the solar insolation is known
        insolations (np.ndarray): value of the solar insolations
    """

    def __init__(self, times: np.ndarray, insolations: np.ndarray):
        self._ins_times = times
        self._insolations = insolations
        self._ins_spline = IUS(self._ins_times, self._insolations)
        self._ins_spline_integ = self._ins_spline.antiderivative()
        self._ins2_spline = IUS(self._ins_times, self._insolations ** 2)
        self._ins2_spline_integ = self.ins2_spline.antiderivative()


class LinearInsolationAccumulation(InsolationAccumulationModel):
    """
    Accumulation is linear in solar insolation.

    Args:
        times (np.ndarray): times at which the solar insolation is known
        insolations (np.ndarray): value of the solar insolations
        intercept (float, optional): default is 0 millimeters
        slope (float, optional): default is 1e-6 mm per unit
            of solar insolation square.
    """

    def __init__(
        self,
        times: np.ndarray,
        insolations: np.ndarray,
        intercept: float = 1.0,
        slope: float = 1e-6,
    ):
        super().__init__(times, insolations)
        self.intercept = intercept
        self.slope = slope

    @property
    def parameters(self) -> Dict[str, float]:
        return {"intercept": self.intercept, "slope": self.slope}

    def accumulation(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError
