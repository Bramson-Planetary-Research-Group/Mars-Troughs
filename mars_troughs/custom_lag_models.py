#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:00:25 2021

@author: kris
"""
import numpy as np
from mars_troughs.generic_model import QuadModel, CubicModel
from mars_troughs.model import Model

class CustomLagModel(Model):
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
    
class QuadraticLag(CustomLagModel,QuadModel):
    """
    Lag is quadratic with time
    """
    
    def __init__(
        self,
        intercept: float = 1e-6,
        linearCoeff: float = 1e-6,
        quadCoeff: float = 1e-6,
    ):
        super().__init__()
        QuadModel.__init__(self, intercept=intercept, 
                                linearCoeff=linearCoeff,
                                quadCoeff=quadCoeff)
        
class CubicLag(CustomLagModel,CubicModel):
    """
    Lag is cubic with time
    """
    
    def __init__(
        self,
        intercept: float = 1e-6,
        linearCoeff: float = 1e-6,
        quadCoeff: float = 1e-6,
        cubicCoeff: float = 1e-6,
    ):
        super().__init__()
        CubicModel.__init__(self, intercept=intercept, 
                                linearCoeff=linearCoeff,
                                quadCoeff=quadCoeff,
                                cubicCoeff=cubicCoeff)
        
    
    