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
    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class InsolationAccumulationModel(AccumulationModel):
    """
    An accumulation rate model that depends on solar insolation, A(Ins(t)).
    A in in m/year. Interpolated splines are created for the insolation as 
    a function of time for faster integration.

    Args:
        times (np.ndarray): times at which the solar insolation is known
                            (in years)
        insolation (np.ndarray): values of the solar insolation (in W/m^2)
    """

    def __init__(self, times: np.ndarray, insolations: np.ndarray):
        self._ins_times = times
        self._insolations = insolations
        self._ins_data_spline = IUS(self._ins_times, self._insolations)
        self._int_ins_data_spline = self._ins_spline.antiderivative()
        self._ins2_spline = IUS(self._ins_times, self._insolations ** 2)
        self._ins2_spline_integ = self.ins2_spline.antiderivative()


class LinearInsolationAccumulation(InsolationAccumulationModel):
    """
    Accumulation is linear in solar insolation. 
    A(ins(t)) = intercept + slope*ins(t).

    Args:
        times (np.ndarray): times at which the solar insolation is known 
                            (in years)
        insolation values (np.ndarray): values of solar insolation (in W/m^2)
        intercept (float, optional): accumulation rate at present time.
                                     Default is 1 m
        slope (float, optional): default is 1e-6 m/year per unit
                                 of solar insolation.
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

    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Calculates the accumulation rate per time

        Args:
            time (np.ndarray): times at which we want to calculate A, in years.
        Output:
            np.ndarray of the same size as time input containing values of 
            accumulation rates A in m/year
        
        """
        return self.intercept +  (self.slope * self._ins_data_spline(time))
    
    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at times t traveled by a point
        in the center of the high side of the trough. This distance  is a 
        function of the accumulation rate A as y(t)=int(A(ins(t)), dt) or 
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of 
            the vertical distance y, in meters.
        
        """
        return self.intercept -1 *(self.slope * (self._int_ins_data_spline(time) 
                                               - self._int_ins_data_spline(0)))
