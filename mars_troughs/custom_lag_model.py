#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:02:49 2021

@author: kris
"""

from typing import List
import numpy as np
from mars_troughs.model import Model


class CustomLagModel(Model):
    """
    Custom lag model
    """
    def __init__(
        self,
        intercept: float=1e-6,
        linearCoeff: float = 1e-6,
        quadCoeff: float = 1e-6,
        ):
        
        self.intercept=intercept
        self.linearCoeff=linearCoeff
        self.quadCoeff=quadCoeff
        

    def get_lag_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Lag as a function of time

        Args:
            time (np.ndarray): times at which we want to calculate the lag.

        Output:
            np.ndarray of the same size as time input containing values of lag.
        """
        return self.intercept + self.linearCoeff*time + self.quadCoeff*time

    @property
    def parameter_names(self) -> List[str]:
        return ["intercept", "linearCoeff","quadCoeff"]
