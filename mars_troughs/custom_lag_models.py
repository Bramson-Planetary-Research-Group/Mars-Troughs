#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom lag models
"""
import numpy as np
from mars_troughs.model import Model
from mars_troughs.generic_model import (ConstantModel, 
                                        LinearModel,
                                        QuadModel, 
                                        CubicModel, 
                                        PowerLawModel)


class LagModel(Model):
    """
    Abstract class for lag models, that have a method
    called :meth:`get_lag_at_t` that returns the lag
    as a function of time.
    """

    prefix: str = "lag_"
    """All parameters of lag models start with 'lag'."""

    def get_lag_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Lag as a function of time

        Args:
            time (np.ndarray): times at which we want to calculate the lag.

        Output:
            np.ndarray of the same size as time input containing values of lag.
        """
        return self.eval(time)
    
class ConstantLag(LagModel, ConstantModel):
    """
    The lag thickness is constant and does not depend on time.

    Args:
        constant (float, optional): default is 1 millimeter. The lag
            thickness at all times.
    """

    def __init__(
        self,
        constant: float = 1e-6,
    ):
        super().__init__()  # note: `super` maps to the LagModel parent class
        ConstantModel.__init__(self, constant=constant)


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

    def __init__(
        self,
        constant: float = 1e-6,
        slope: float = 1e-8,
    ):
        super().__init__()
        LinearModel.__init__(self, constant=constant, slope=slope)
    
class QuadraticLag(LagModel,QuadModel):
    """
    Lag is quadratic with time
    """
    
    def __init__(
        self,
        constant: float = 1e-6,
        slope: float = 1e-8,
        quad: float = 1e-20,
    ):
        super().__init__()
        QuadModel.__init__(self, constant=constant, 
                                 slope=slope,
                                 quad=quad)
        
class CubicLag(LagModel,CubicModel):
    """
    Lag is cubic with time
    """
    
    def __init__(
        self,
        constant: float = 1e-6,
        slope: float = 1e-8,
        quad: float = 1e-20,
        cubic: float = 1e-30,
    ):
        super().__init__()
        CubicModel.__init__(self, constant=constant, 
                                  slope=slope,
                                  quad=quad,
                                  cubic=cubic)
    
class PowerLawLag(LagModel,PowerLawModel):
    """
    Lag follows a power law function with time
    """
    def __init__(
        self,
        coeff: float = 5,
        exponent: float = 1e-2,
        ):
        super().__init__()
        PowerLawModel.__init__(self, coeff, exponent)
        
        
    
        
    
    