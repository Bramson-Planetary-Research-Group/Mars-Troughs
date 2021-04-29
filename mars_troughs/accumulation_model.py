"""
Model for the accumulation rates.
"""
from abc import abstractmethod

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


class AccumulationModel():
    """
    Abstract class for computing the amount of ice accumulation.
    """

    @abstractmethod
    def accumulation(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class AccumulationDepInsolation(AccumulationModel):
    """
    An accumulation model that depends only on solar insolation.
    Interpolated splines are created for the insolation as a function
    of time for faster integration.

    Args:
        times (np.ndarray): times at which the solar insolation is known
        insolations (np.ndarray): value of the solar insolations
    """

    def __init__(self, times: np.ndarray, insolation: np.ndarray):
        
        self.ins_data_spline = IUS(times, insolation)
        self.iins_data_spline = self.ins_data_spline.antiderivative()
        self.ins2_data_spline = IUS(times, insolation ** 2)
        self.iins2_data_spline = self.ins2_data_spline.antiderivative()


class LinearAccuIns(AccumulationDepInsolation):
    """
    Accumulation is linear in solar insolation.

    Args:
        times (np.ndarray): times at which the solar insolation is known
        insolations (np.ndarray): value of the solar insolations
        intercept (float, optional): default is 0 millimeters
        slope (float, optional): default is 1e-6 mm per unit
            of solar insolation square.
    """
    def __init__(self, intercept: float = 0.0, slope: float = 1e-6):
        
        self.intercept=intercept
        self.slope=slope
        
    def get_accumulation_at_t(self, time: np.ndarray):
        """
        Accumulation as a  linear function of insolation: 
        accu(t) = a + b*ins(t).

        Args:
            time (np.ndarray): times at which we want to calculate the lag.
        Output:
            np.ndarray of the same size as time input containing values of lag. 
        
        """
        return self.intercept + self.slope * time
        
        


