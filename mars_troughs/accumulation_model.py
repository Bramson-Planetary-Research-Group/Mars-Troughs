"""
Model for the accumulation rates.
"""
from abc import abstractmethod
from typing import Dict

import numpy as np

from mars_troughs.model import Model


class AccumulationModel(Model):
    """
    Abstract class for computing the amount of ice accumulation.
    """

    @abstractmethod
    def accumulation(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearAccumulation(Model):
    """
    Accumulation is linear in solar insolation.

    Args:
        intercept (float, optional): default is 0 millimeters
        slope (float, optional): default is 1e-6 mm per unit
            of solar insolation square.
    """

    def __init__(self, intercept: float = 1.0, slope: float = 1e-6):
        self.intercept = intercept
        self.slope = slope

    @property
    def parameters(self) -> Dict[str, float]:
        return {"intercept": self.intercept, "slope": self.slope}

    def accumulation(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError
