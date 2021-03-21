"""
Models for the lag as a function of time.
"""
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class LagModel(ABC):
    @property
    def parameters(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def lag(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConstantLag(LagModel):
    """
    Constant lag model. The lag thickness does not depend on time at all.

    Args:
        constant (float, optional): default is 1 millimeter. The lag
            thickness at all times.
    """

    def __init__(self, constant: float = 1.0):
        self.constant = 1.0

    @property
    def parameters(self) -> Dict[str, float]:
        return {"constant": self.constant}

    def lag(self, time: np.ndarray) -> np.ndarray:
        """
        Lag as a function of time. Always returns a constant.

        Args:
            time (np.ndarray): ignored
        """
        return self.constant


class LinearLag(LagModel):
    """
    The lag thickness is linear in time.

    Args:
        intercept (float, optional): default is 1 millimeter. The lag
            thickness at time t=0 (present day).
        slope (float, optional): default is 1e-6 mm per year. The rate
            of change of the lag each year.
    """

    def __init__(self, intercept: float = 1.0, slope: float = 1e-6):
        self.intercept = intercept
        self.slope = slope

    @property
    def parameters(self) -> Dict[str, float]:
        return {"intercept": self.intercept, "slope": self.slope}

    def lag(self, time: np.ndarray) -> np.ndarray:
        """
        Lag as a function of time.

        Args:
            time (np.ndarray): times to evaluate the lag
        """
        return self.intercept + self.slope * time
